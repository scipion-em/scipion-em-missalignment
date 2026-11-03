# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Scipion Team
# *
# * National Center of Biotechnology, CSIC, Spain
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
import logging
from enum import Enum

from missalignment import Plugin
from missalignment.constants import IN_TS_SET, GLOBAL, ETOMO_TO_WARP
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from typing import List, Tuple
from pyworkflow.protocol import PointerParam, EnumParam, IntParam, LEVEL_ADVANCED, FloatParam, StringParam, GPU_LIST, \
    GE, BooleanParam
from pyworkflow.utils import Message, createAbsLink, makePath

from pyworkflow.object import Boolean

from tomo.utils import getTsIdsDicts
from os.path import join, abspath

import os
import csv
import math
import yaml

from tomo.tests import tiltStep

logger = logging.getLogger(__name__)


class Outputobjects(Enum):
    # model = MissAlignment_model
    pass


class ProtMissalignmentTraining(EMProtocol):
    """Iteratively train and realign a dataset over a series of coarse-to-fine macro-iterations.
    Each macro-iteration runs two phases: first a model is trained to score
    reconstruction quality, then that model (with frozen weights) optimizes the
    tilt-series alignment by gradient descent. The aligned output of one
    macro-iteration becomes the input to the next, and the `iteration_settings`
    schedule in the config moves from coarse (downsampled) to fine alignment."""

    _label = 'Training'
    _devStatus = BETA

    _possibleOutputs = Outputobjects

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tsDict = {}

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam(IN_TS_SET, PointerParam,
                      pointerClass='SetOfTiltSeries',
                      label='Tilt-series',
                      important=True,
                      help='Tilt-series with a coarse alignment.')
        form.addParam('preprocess', BooleanParam,
                      label='Run coarse alignment first?',
                      default=False,
                      help='Run a cross-correlation alignment at the start of the run, before running any iteration.'
                           'Use it if the tilt-series do not have a coarse alignment yet, otherwisw it would throw away'
                           'the estimated shifts.')
        form.addParam('prepare_stack', FloatParam,
                      label='Working pixel size (Å, opt.)',
                      allowsNull=True,
                      help='Pixel size (in Å) the tilt-series are rescaled to before training. Increasing it '
                           'reduces the number of pixels, so training and alignment are faster.')

        form.addSection(label='Alignment')
        group = form.addGroup('Iterations')
        group.addParam('downsample', StringParam,
                       label='Downsample per iteration',
                       default='3 2 1 1 1 1 1 1',
                       help='One value per iteration, separated by spaces. The number of values is the '
                            'number of iterations. 3 = images reduced by a factor 3, 1 = full resolution.')
        group.addParam('alignment', StringParam,
                       label='Alignment type',
                       default='A A G G 3x3 3x3 3x3 3x3',
                       help='One type per iteration, separated by spaces: '
                            'G = global, '
                            'A = anchoring, '
                            '3x3 = grid such as 3x3 (local warping). '
                            'If there are fewer values than iterations, the last one is repeated. '
                            'More values than iterations is an error.')  # TODO create methods to be used when creating the .yaml to convert 3x3 in [3,3], G and A and that guarantee that if downsample>alignment, the remaining downsample values are assigned to the last alignment
        form.addParam('batch_size', IntParam,
                      label='Batch size',
                      default=32,
                      help='Amount of patches simultaneously reconstructed in memory.'
                           'A value of 32 works well for 24 GB cards.')
        form.addParam('patch_size', IntParam,
                      label='Patch size (pixels)',
                      expertLevel=LEVEL_ADVANCED,
                      default=96,
                      help='Size of the cubic patches (small subtomograms) '
                           'reconstructed from the tilt-series. They are used both to '
                           'train the model and to refine the alignment, so the same '
                           'value is applied to both steps. Larger patches contain more '
                           'context but need more GPU memory.')
        form.addParam('patch_overlap', FloatParam,
                      label='Patch overlap',
                      expertLevel=LEVEL_ADVANCED,
                      default=0.1,
                      help='Tolerated overlap between patches used for optimizing the alignment.')

        form.addSection(label='Training Parameters')
        form.addParam('learning_rate', FloatParam,
                      label='Learning rate',
                      levelExpert=LEVEL_ADVANCED,
                      default=0.001)
        form.addParam('max_epochs_for_iterations', IntParam,
                      label='Max number of iterations',
                      default=30,
                      help='Absolute maximum of stopping.')

        # TODO missalignment wants gpus ids to be comma separated
        form.addSection(label='GPU IDs')
        form.addParam('training_devices', StringParam,
                      default='0',
                      label='Training GPUs',
                      help='GPUs used to train the model, as positions in the list of the available GPUs '
                           '(0 = the first one), separated by commas. Each entry runs one training process.')
        form.addParam('reconstruction_devices', StringParam,
                      default='0',
                      label='Reconstruction GPUs',
                      help='One entry per reconstruction worker. An index can be repeated to run '
                           'several workers on the same GPU (0,0,1,1 = two workers on each of the '
                           'first two GPUs). Can overlap with the training GPUs.')
        form.addParam('dataloaders_per_trainer', IntParam,
                      default=2, validators=[GE(1)],
                      expertLevel=LEVEL_ADVANCED,
                      label='Data loaders per training GPU',
                      help='CPU workers per training GPU.')
        # form.addParam('pool_size', IntParam,
        #               label='Pool size',
        #               default=1000,
        #               levelExpert=LEVEL_ADVANCED,
        #               help='Maximum number of subtomgram reconstructions to write to temporary storage during training '
        #                    'of the misalignment model.')

    # --------------------------- INSERT steps functions ----------------------
    def _initialize(self):
        self.tsDict = getTsIdsDicts(self.inTsSet.get())

    def _insertAllSteps(self):
        self._initialize()

        cInputId = []
        for tsId in self.tsDict.keys():
            stepId = self._insertFunctionStep(self.convertInputStep, tsId,
                                              prerequisites=[],
                                              needsGPU=False)
            cInputId.append(stepId)
        warpId = self._insertFunctionStep(self.toWarpStep,
                                          prerequisites=cInputId,
                                          needsGPU=False)
        wConfId = self._insertFunctionStep(self.writeConfigStep,
                                           prerequisites=warpId,
                                           needsGPU=False)
        trainId = self._insertFunctionStep(self.trainingStep,
                                           prerequisites=wConfId,
                                           needsGPU=True)
        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=trainId,
                                 needsGPU=False)

    # -------------------------- STEPS functions ------------------------------
    def convertInputStep(self, tsId):
        logger.info(cyanStr('Compiling imod format files...'))

        ts = self.tsDict[tsId]
        imodDir = self._makeImodDir(tsId)

        present = ts.getTsPresentAcqOrders()
        self.writeRawTltFile(ts, imodDir, present=present)
        self.writeXfFile(ts, join(imodDir, f'{tsId}.xf'), present=present)

        source = ts.getFirstItem().getFileName()
        destination = join(imodDir, f'{tsId}.st')
        if ts.hasExcludedViews():
            ts.reStack(source, destination, present)
        else:
            createAbsLink(abspath(source), abspath(destination))

    def toWarpStep(self):
        logger.info(cyanStr('Converting in .xml files...'))

        dims = []
        tiltAxis_angle = []
        outputDir = self._makeOutputDir()

        for tsId, ts in self.tsDict.items():
            dims.append(self._getTomoDims(ts))
            # Average the tilt axis angle over the whole dataset
            tiltAxis_angle.append(ts.getAcquisition().getTiltAxisAngle())
        tilt_angle = sum(tiltAxis_angle) / len(tiltAxis_angle)

        # The tomogram shape should tightly the sample in all three dimensions (X, Y, Z) to avoid training
        # on empty regions — when samples vary in thickness, use the thickest one as the reference
        ref_shape = shape = [max(d[i] for d in dims) for i in range(3)]

        # Call the script etomo_to_warp.py to convert to the .xml format
        args = [
            f'--input-dir {self.getImodDir()}',
            f'--output-dir {self.getOutoutDir}',
            f'--tilt-axis-angle {tilt_angle}',
            f'--volume-shape {shape[0]} {shape[1]} {shape[2]}'
        ]
        if self.prepare_stack.get():
            args.append(f'--output-pixel-size {self.prepare_stack.get()}')
        else:
            args.append(f'--output-pixel-size {10}')
        args = ' '.join(args)

        Plugin.runMissalignment(self, f'python "{ETOMO_TO_WARP}"', args, useGpu=False)

    def writeConfigStep(self):
        logger.info(cyanStr('Creating configuration file...'))
        with open(CONFIG_TEMPLATE) as f:
            config = yaml.safe_load(f)

        config['general']['training_directory'] = self.getOutputDir()
        # config['general']['iteration_settings'] = self. #TODO

        config['model_training']['learning_rate'] = self.learning_rate.get()
        config['model_training']['max_epochs_per_iteration'] = self.max_epochs_for_iterations.get()

        config['data_loading']['patch_size'] = self.patch_size.get()

        config['tilt_series_alignment']['batch_size'] = self.batch_size.get()
        config['tilt_series_alignment']['patch_size'] = self.patch_size.get()
        config['tilt_series_alignment']['patch_overlap'] = self.patch_overlap.get()

        with open(self._getExtraPath('config.yaml'), 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None)

    def trainingStep(self):
        pass

    def createOutputStep(self):
        pass

    # -------------------------- UTILS functions ------------------------------
    def _makeImodDir(self, tsId):
        tsDir = self._getExtraPath('imod', tsId)
        makePath(tsDir)
        return tsDir

    def getImodDir(self):
        return self._getExtraPath('imod')

    def _makeOutputDir(self):
        outputDir = self._getExtraPath('warp_tiltseries')
        makePath(outputDir)
        return outputDir

    def getOutputDir(self):
        return self._getExtraPath('warp_tiltseries')

    def _getTomoDims(self, ts) -> Tuple[int, int, int]:
        x = int(ts.getDimensions()[0])
        y = int(ts.getDimensions()[1])
        # The binary swap is exact only near φ∈{0,90}. Near 45°, the detector rectangle rotated into
        # the tomogram frame needs a box larger than either W×H or H×W. The exact axis-aligned bounding
        # box of the rotated rectangle covers every imaged pixel at any angle and self-pads near 45°I
        tiltAxisAngle = ts.getAcquisition().getTiltAxisAngle()
        c, s = abs(math.cos(tiltAxisAngle)), abs(math.sin(tiltAxisAngle))
        tomoX = math.ceil(x * c + y * s)
        tomoY = math.ceil(x * s + y * c)
        tomoZ = int(max(tomoX, tomoY) / 3)
        return tomoX, tomoY, tomoZ


    def writeRawTltFile(self, ts, ts_folder, present):
        """Writes a raw tlt file.
        :param ts_folder: path of the directory in which the tlt file will be generated.
        :param present: retrieve ts acquisition order.
        """
        xtiltPath = ts_folder + '/%s.rawtlt' % ts.getTsId()
        with open(xtiltPath, 'w') as f:
            for ti in ts.iterItems(orderBy=ts.INDEX):
                if ti.getAcquisitionOrder() not in present:
                    continue
                f.write(str(ti.getTiltAngle()) + '\n')
            # For parallel processing, ensure that the file is completely written and persists on disk
            f.flush()  # Empty python buffer
            os.fsync(f.fileno())  # Empty system buffer

    def writeXfFile(self, ts, transformFilePath, present, delimiter='\t', factor=1):
        """ This method takes a tilt series and the output transformation file
        path and creates an IMOD-based transform
        file in the location indicated. """

        tsMatrixTransformList = []

        for ti in ts.iterItems(orderBy=ts.INDEX):
            if ti.getAcquisitionOrder() not in present:
                continue
            if ti.getTransform() is not None:
                transform = ti.getTransform().getMatrix().flatten()
                transformIMOD = ['%.7f' % transform[0],
                                 '%.7f' % transform[1],
                                 '%.7f' % transform[3],
                                 '%.7f' % transform[4],
                                 '%.3f' % (float(transform[2]) / factor),
                                 '%.3f' % (float(transform[5]) / factor)]
            else:
                from pyworkflow.utils import yellowStr
                logging.info(
                    yellowStr('WARNING: The Tilt series lacks of alignment information (transformation matrices). '
                              'The identity transformation will be written in the .xf file'))

                #  This is the identity matrix
                transformIMOD = ['1.0000000',
                                 '0.0000000',
                                 '0.0000000',
                                 '1.0000000',
                                 '0.000',
                                 '0.000']
            tsMatrixTransformList.append(transformIMOD)

        with open(transformFilePath, 'w') as f:
            csvW = csv.writer(f, delimiter=delimiter)
            csvW.writerows(tsMatrixTransformList)
            # For parallel processing, ensure that the file is completely written and persists on disk
            f.flush()  # Empty python buffer
            os.fsync(f.fileno())  # Empty system buffer

    # -------------------------- INFO functions ------------------------------
    def _validate(self) -> List[str]:
        errors = []

        downsample = (self.downsample.get() or '').split()
        alignment = (self.alignment.get() or '').split()
        iterations = len(downsample)

        inTsSet = self.inTsSet.get()
        preprocess = self.preprocess.get()

        if inTsSet is not None and inTsSet.interpolated():
            errors.append('The tilt-series are interpolated (the alignment is already applied to the images).'
                          'Provide the series with the alignment stored as a transformation.')

        if iterations < 1:
            errors.append('At least one downsampling value is required.')
        if len(alignment) < 1:
            errors.append('At least one alignment type is required.')
        if len(alignment) > iterations:
            errors.append('The number of alignment types cannot exceed the number of iterations.')

        inTsSet = self.inTsSet.get()
        if inTsSet is not None:
            nAligned = sum(1 for ts in inTsSet if ts.hasAlignment())
            nTotal = inTsSet.getSize()
            if nAligned == 0 and not self.preprocess.get():
                errors.append("The tilt-series have no coarse alignment: set 'Preprocess' to Yes.")
            elif 0 < nAligned < nTotal:
                errors.append(f'Only {nAligned} of {nTotal} tilt-series have a coarse alignment. '
                              f'Provide all of them aligned, or none and use Preprocess.')

        pool_size = 1000  # TODO check if you want the user to set it
        n_partitions = len(self.training_devices.get()) * self.dataloaders_per_trainer.get()
        partition_size = pool_size // n_partitions
        if partition_size < (2 * self.batch_size.get()):
            errors.append('Use fewer training GPUs or data loaders per GPU.')

            return errors

    def _warnings(self) -> List[str]:
        warnings = []

        inTsSet = self.inTsSet.get()
        if inTsSet is not None and self.preprocess.get():
            if any(ts.hasAlignment() for ts in inTsSet):
                warnings.append('Some tilt-series already have a coarse alignment: Preprocess will recompute it.')
            return warnings
