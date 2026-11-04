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
from missalignment.constants import IN_TS_SET, GLOBAL, ETOMO_TO_WARP, CONFIG_TEMPLATE
from missalignment.objects import MissalignmentModel
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from typing import List, Tuple
from pyworkflow.protocol import PointerParam, EnumParam, IntParam, LEVEL_ADVANCED, FloatParam, StringParam, GPU_LIST, \
    GE, BooleanParam, ElementGroup, GT
from pyworkflow.utils import Message, createAbsLink, makePath, cyanStr, redStr

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
    model = MissalignmentModel


class ProtMissalignmentTraining(EMProtocol):
    """Iteratively train and realign a dataset over a series of coarse-to-fine macro-iterations.
    Each macro-iteration runs two phases: first a model is trained to score
    reconstruction quality, then that model (with frozen weights) optimizes the
    tilt-series alignment by gradient descent. The aligned output of one
    macro-iteration becomes the input to the next, and the `iteration_settings`
    schedule in the config moves from coarse (downsampled) to fine alignment."""

    _label = 'Training'
    _devStatus = BETA
    _program = 'miss-alignment train'
    _possibleOutputs = Outputobjects

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tsDict = {}
        self.failedTsIds = []

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
                           'Use it if the tilt-series do not have a coarse alignment yet, otherwise it would throw away'
                           'the estimated shifts.')
        form.addParam('prepare_stacks', FloatParam,
                      label='Working pixel size (Å, opt.)',
                      allowsNull=True,
                      help='Pixel size (in Å) the tilt-series are rescaled to before training. Increasing it '
                           'reduces the number of pixels, so training and alignment are faster.')

        form.addSection(label='Alignment',
                        help='Each iteration does two things: '
                             '1. Trains (or fine-tunes) the scoring model at the given downsampling. '
                             '2. Uses that model to realign every tilt-series. '
                             'The alignment from one iteration is the starting point for the next, so '
                             'order the rows from coarse to fine.'
                        )
        group = form.addGroup('Iterations')
        group.addParam('num_iterations', IntParam,
                       label='Number of iterations',
                       validators=[GT(0)],
                       default=8,
                       help='Number of training iterations.')
        group.addParam('downsample', StringParam,
                       label='Downsample per iteration',
                       default='3 2 1 1 1 1 1 1',
                       help='Integer binning factor. '
                            'n = images reduced by a factor n, 1 = full resolution. '
                            'One value per training iteration, separated by spaces.')
        group.addParam('alignment', StringParam,
                       label='Alignment type',
                       default='A A G G 3x3 3x3 3x3 3x3',
                       help='One type per iteration: '
                            'G = global (one X/Y shift per tilt, refined in a single pass), '
                            'A = anchoring (one X/Y shift per tilt, optimized iteratively: tilts that do not improve are '
                            'kept anchored), '
                            'MxN = local image warping (e.g. 3x3). '
                            'One value per training iteration, separated by spaces.')
        form.addParam('batch_size', IntParam,
                      label='Batch size',
                      default=32,
                      help='Amount of patches simultaneously reconstructed in memory.'
                           'A value of 32 works well for 24 GB cards.')
        form.addParam('start_at_iteration', IntParam,
                      label='Start at iteration',
                      default=0,
                      help='Continue from a specific iteration in the config.')
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
                      label='Max number of epochs per iteration',
                      default=30,
                      help='Absolute maximum of stopping.')

        form.addSection(label='GPU IDs')
        form.addParam('training_devices', StringParam,
                      default='0',
                      label='Training GPUs',
                      help='GPUs indices used to train the model, as positions in the list of the available GPUs '
                           '(0 = the first one). Each entry runs one training process.')
        form.addParam('reconstruction_devices', StringParam,
                      default='0',
                      label='Reconstruction devices',
                      help='Comma-separated list of GPU device indices for reconstruction workers.'
                           'Each entry spawns one worker on that device. Devices can be repeated '
                           'to run multiple workers per GPU (e.g., "0,0,1,1" runs 2 workers each '
                           'on GPU 0 and GPU 1). Can overlap with training devices if needed.')
        form.addParam('dataloaders_per_trainer', IntParam,
                      default=2, validators=[GE(1)],
                      expertLevel=LEVEL_ADVANCED,
                      label='Data loaders per training GPU',
                      help='Number of CPU DataLoader workers per training device.')
        # form.addParam('pool_size', IntParam,
        #               label='Pool size',
        #               default=1000,
        #               levelExpert=LEVEL_ADVANCED,
        #               help='Maximum number of subtomgram reconstructions to write to temporary storage during training '
        #                    'of the misalignment model.')

        form.addHidden(GPU_LIST, StringParam,
                       default='0',
                       label="Choose GPU IDs you want to use both for training and reconstruction.",
                       help="")

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

        try:
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

        except Exception as e:
            self.failedTsIds.append(tsId)
            logger.error(redStr(f'tsId = {tsId} -> input conversion failed with the exception -> {e}'))
            logger.error(traceback.format_exc())

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
            f'--output-dir {self.getOutputDir()}',
            f'--tilt-axis-angle {tilt_angle}',
            f'--volume-shape {shape[0]} {shape[1]} {shape[2]}'
        ]
        if self.prepare_stacks.get():
            args.append(f'--output-pixel-size {self.prepare_stacks.get()}')
        else:
            args.append(f'--output-pixel-size {10}')
        args = ' '.join(args)

        Plugin.runMissalignment(self, f'python "{ETOMO_TO_WARP}"', args, useGpu=False)

    def writeConfigStep(self):
        logger.info(cyanStr('Creating configuration file...'))
        with open(CONFIG_TEMPLATE) as f:
            config = yaml.safe_load(f)

        config['general']['training_directory'] = self.getOutputDir()
        config['general']['iteration_settings'] = self._getIteration()

        config['model_training']['learning_rate'] = self.learning_rate.get()
        config['model_training']['max_epochs_per_iteration'] = self.max_epochs_for_iterations.get()

        config['data_loading']['patch_size'] = self.patch_size.get()

        config['tilt_series_alignment']['batch_size'] = self.batch_size.get()
        config['tilt_series_alignment']['patch_size'] = self.patch_size.get()
        config['tilt_series_alignment']['patch_overlap'] = self.patch_overlap.get()

        with open(self._getExtraPath('config.yaml'), 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None)

    def trainingStep(self):
        logger.info(cyanStr(f"Training the model...'"))

        try:
            Plugin.runMissalignment(self, self._program, self._generateArguments(), useGpu=True)
        except Exception as e:
            logger.error(redStr('Training failed!'))
            raise

    def createOutputStep(self):
        logger.info(cyanStr(f"Creating output...'"))

        model = MissalignmentModel(runDirectory=self.getOutputDir(), nIterations=self.num_iterations.get())
        self._defineOutputs(**{Outputobjects.model.name: model})
        self._defineSourceRelation(self.inTsSet.get(), model)


    # -------------------------- UTILS functions ------------------------------
    def _makeImodDir(self, tsId):
        tsDir = self._getExtraPath('imod', tsId)
        makePath(tsDir)
        return tsDir

    def getImodDir(self):
        return abspath(self._getExtraPath('imod'))

    def _makeOutputDir(self):
        outputDir = self._getExtraPath('warp_tiltseries')
        makePath(outputDir)
        return outputDir

    def getOutputDir(self):
        return abspath(self._getExtraPath('warp_tiltseries'))

    def _parseAlignment(self, token):
        token = token.lower()  # 'G' and 'g', 'A' and 'a' are the same thing
        if token == 'g':
            return 'global'
        if token == 'a':
            return 'anchoring'

        if 'x' in token:  # converting the grid
            parts = token.split('x')  # 3x3 turns to [3, 3]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                m, n = int(parts[0]), int(parts[1])
                if m > 0 and n > 0:
                    return [m, n]
        raise ValueError(f'Alignment: "{token}" is not valid. '
                         f'Use G (global), A (anchoring) or a grid MxN.')

    def _getIteration(self):
        """One dict per iteration: {'downsample': int, 'alignment': 'global' | 'anchoring' | [M, N]}"""
        n = self.num_iterations.get()
        downsample = (self.downsample.get() or '').split()
        alignment = (self.alignment.get() or '').split()

        settings = []
        for i in range(n):
            entry = {}
            entry['downsample'] = int(downsample[i])
            entry['alignment'] = self._parseAlignment(alignment[i])
            settings.append(entry)
        return settings

    def _getTomoDims(self, ts) -> Tuple[int, int, int]:
        x = int(ts.getDimensions()[0])
        y = int(ts.getDimensions()[1])
        # The binary swap is exact only near φ∈{0,90}. Near 45°, the detector rectangle rotated into
        # the tomogram frame needs a box larger than either W×H or H×W. The exact axis-aligned bounding
        # box of the rotated rectangle covers every imaged pixel at any angle and self-pads near 45°I
        tiltAxisAngle = math.radians(ts.getAcquisition().getTiltAxisAngle())
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

    def _getConfigFile(self):
        return abspath(self._getExtraPath('config.yaml'))

    def _generateArguments(self) -> str:
        training_devices = ' '.join([str(el) for el in self.training_devices.get()])
        reconstruction_devices = ' '.join([str(el) for el in self.reconstruction_devices.get()])
        cmd = [
            f'--config-file {self._getConfigFile()}',
            f'--training-devices {self.training_devices.get()}',
            f'--reconstruction-devices {self.reconstruction_devices.get()}',
            f'--dataloaders-per-trainer {self.dataloaders_per_trainer.get()}'
        ]

        if self.preprocess:
            cmd.append('--preprocess')

        return ' '.join(cmd)

    # -------------------------- INFO functions ------------------------------
    def _validate(self) -> List[str]:
        errors = []

        downsample = (self.downsample.get() or '').split()
        alignment = (self.alignment.get() or '').split()
        n_downsample = len(downsample)
        n_alignment = len(alignment)
        n_iterations = self.num_iterations.get()

        inTsSet = self.inTsSet.get()
        preprocess = self.preprocess.get()

        if inTsSet is not None and inTsSet.interpolated():
            errors.append('The tilt-series are interpolated (the alignment is already applied to the images).'
                          'Provide the series with the alignment stored as a transformation.')

        if n_downsample != n_iterations:
            errors.append('The number of downsample values must match the number of iterations.')
        if n_alignment != n_iterations:
            errors.append('The number of alignment types must match the number of iterations.')

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
