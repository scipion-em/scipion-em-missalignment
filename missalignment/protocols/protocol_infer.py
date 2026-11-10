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
from os.path import join, abspath
import traceback


import yaml
from missalignment import Plugin
from missalignment.constants import IN_TS_SET, INFER_CONFIG_TEMPLATE
from missalignment.objects import MissalignmentModel
from missalignment.protocols.protocol_base import ProtMissalignmentBase
from pwem.protocols import EMProtocol
from pyworkflow import BETA

from typing import List
from pyworkflow.protocol import PointerParam, FloatParam, BooleanParam, IntParam, GT, StringParam, LEVEL_ADVANCED, \
    GPU_LIST
from pyworkflow.utils import Message, cyanStr, redStr
from tomo.objects import SetOfTiltSeries
from tomo.utils import getTsIdsDicts

logger = logging.getLogger(__name__)


class Outputobjects(Enum):
   tiltSeries = SetOfTiltSeries


class ProtMissalignmentInfer(ProtMissalignmentBase):
    """Align a dataset by applying models from a previous training run.

    For each iteration it loads the model that a previous `train` run saved at
    `<model_run_directory>/iter{N}/model.ckpt` and runs only the alignment
    phase, following the `iteration_settings` schedule from the config."""

    _label = 'Infer'
    _devStatus = BETA
    _program = 'miss-alignment infer'
    _possibleOutputs = Outputobjects

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tsDict = {}
        self.failedTsIds = []

    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam(IN_TS_SET, PointerParam,
                      pointerClass='SetOfTiltSeries',
                      label='Tilt-series',
                      important=True,
                      help='Tilt-series to align.')
        form.addParam('model', PointerParam,
                      pointerClass='MissalignmentModel',
                      important=True,
                      label='Missalignment Model',
                      allowsNull=False,
                      help='Select a trained Missalignment model.'
                      )
        form.addParam('preprocess', BooleanParam,
                      label='Run coarse alignment first?',
                      default=False,
                      help='Run a cross-correlation alignment at the start of the run, before running any iteration.'
                           'Use it if the tilt-series do not have a coarse alignment yet, otherwise it would throw away'
                           'the estimated shifts.')
        form.addParam('prepare_stacks', FloatParam,
                      label='Working pixel size (Å, opt.)',
                      allowsNull=True,
                      help='Pixel size (in Angstroms) for preprocessing tilt stacks. '
                           'If provided, loads raw tilt images, rescales to this pixel size, '
                           'and creates tilt stacks with thumbnails before training.')


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
        form.addParam('patch_overlap', FloatParam,
                      label='Patch overlap',
                      expertLevel=LEVEL_ADVANCED,
                      default=0.1,
                      help='Tolerated overlap between patches used for optimizing the alignment.')

        form.addHidden(GPU_LIST, StringParam,
                       default='0',
                       label="Choose GPU IDs",
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
        inferId = self._insertFunctionStep(self.inferStep,
                                           prerequisites=wConfId,
                                           needsGPU=True)
        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=inferId,
                                 needsGPU=False)


    # -------------------------- STEPS functions ------------------------------
    def writeConfigStep(self):
        logger.info(cyanStr('Creating configuration file...'))

        model = self.model.get()

        with open(INFER_CONFIG_TEMPLATE) as f:
            config = yaml.safe_load(f)

        config['general']['data_directory'] = self.getOutputDir()
        config['general']['model_run_directory'] = abspath(model.getPath())
        config['general']['iteration_settings'] = self._getIteration()

        config['tilt_series_alignment']['batch_size'] = self.batch_size.get()
        config['tilt_series_alignment']['patch_size'] = model.getPatchSize() #same used to train the model
        config['tilt_series_alignment']['patch_overlap'] = self.patch_overlap.get()

        with open(self._getExtraPath('inference_config.yaml'), 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None)

    def inferStep(self):
        logger.info(cyanStr(f"Retrieving the alignment..."))
        try:
            Plugin.runMissalignment(self, self._program, self._generateArguments(), useGpu=True)
        except Exception as e:
            logger.error(redStr('Alignment failed!'))
            raise

    def createOutputStep(self):
        logger.info(cyanStr('Exporting the alignment to Scipion tilt-series...'))
        inTsSet = self.inTsSet.get()
        outTsSet = SetOfTiltSeries.create(self._getPath(), template='tiltseries', suffix='aligned')
        outTsSet.copyInfo(inTsSet)
        outTsSet.setDim(inTsSet.getDim())

        exported, skipped = [], []
        for tsId in self._getReadyTsIds():
            ts = self.tsDict[tsId]
            try:
                matrixByOrder = self._getAlignedMatrices(ts)
            except Exception as e:
                skipped.append(tsId)
                logger.error(redStr(f'tsId = {tsId} -> alignment could not be exported: {e}'))
                logger.error(traceback.format_exc())
                continue

            newTs = TiltSeries(tsId=tsId)
            newTs.copyInfo(ts)
            outTsSet.append(newTs)
            for ti in ts.iterItems(orderBy=ts.INDEX):
                newTi = ti.clone()
                matrix = matrixByOrder.get(ti.getAcquisitionOrder())
                if matrix is not None:  # excluded views keep what they had
                    transform = Transform()
                    transform.setMatrix(matrix)
                    newTi.setTransform(transform)
                newTs.append(newTi)

            newTs.setDim(ts.getDim())
            newTs.setAlignment2D()
            newTs.write()
            outTsSet.update(newTs)
            exported.append(tsId)

        if not exported:
            raise Exception('No alignment could be exported: check the log of this step.')
        if skipped:
            logger.warning(redStr(f'Alignment not exported for {len(skipped)} tilt-series: {skipped}'))

        outTsSet.write()
        self._defineOutputs(**{self._possibleOutputs.tiltSeries.name: outTsSet})
        self._defineSourceRelation(self.inTsSet, outTsSet)
        self._defineSourceRelation(self.model, outTsSet)


    # -------------------------- UTILS functions ------------------------------
    def _getConfigFile(self):
        return abspath(self._getExtraPath('inference_config.yaml'))

    def _generateArguments(self):
        cmd =[
            f'--config-file {self._getConfigFile()}',
            f'--start-at-iteration {self.start_at_iteration.get()}'
        ]

        if self.preprocess:
            cmd.append('--preprocess')

        return ' '.join(cmd)

    # ------------------- alignment export: Warp XML -> Scipion matrices -------------------
    @staticmethod
    def _readGlobalAlignment(xmlPath):
        """(offsetX, offsetY, useTilt, imageDimsAngstrom) from a Warp tilt-series XML."""
        root = ET.parse(xmlPath).getroot()

        def values(tag):
            return np.array([float(v) for v in root.find(tag).text.split()])

        useTilt = np.array([v == 'True' for v in root.find('UseTilt').text.split()])
        imageDims = [float(v) for v in root.get('ImageDimensionsAngstrom').split(',')]
        return values('AxisOffsetX'), values('AxisOffsetY'), useTilt, imageDims

    @staticmethod
    def _offsetsToXf(offsetsYx, rotXf, pixelSize):
        """Inverse of parse_transform_file (etomo_to_warp.py): (offset_y, offset_x) in A -> .xf rows in pixels."""
        corrected = offsetsYx / pixelSize  # Angstrom -> pixels
        vXy = -1.0 * np.flip(corrected, axis=1)  # undo sign change and Y/X swap
        rot = rotXf.reshape(-1, 2, 2)
        shifts = np.einsum('nij,nj->ni', rot, vXy)  # put back the rotation of each view
        return np.hstack([rotXf, shifts])  # a11 a12 a21 a22 dx dy

    @staticmethod
    def _xfRowToMatrix(row):
        """Same layout as TiltSeries.writeXfFile: columns [m00, m01, m10, m11, m02, m12]."""
        a11, a12, a21, a22, dx, dy = [float(v) for v in row]
        return np.array([[a11, a12, dx], [a21, a22, dy], [0.0, 0.0, 1.0]])

    def _getAlignedMatrices(self, ts):
        """{acquisition order: 3x3 matrix} for the active views of a series, from the aligned XML."""
        tsId = ts.getTsId()
        xmlPath = join(self.getOutputDir(), f'{tsId}.xml')
        xfPath = join(self.getImodDir(), tsId, f'{tsId}.xf')

        # active views, in the order of the .xf rows and of the XML values (INDEX order)
        present = ts.getTsPresentAcqOrders()
        views = [ti for ti in ts.iterItems(orderBy=ts.INDEX) if ti.getAcquisitionOrder() in present]

        xf = np.atleast_2d(np.loadtxt(xfPath))
        offsetX, offsetY, useTilt, imageDims = self._readGlobalAlignment(xmlPath)
        if not (len(views) == xf.shape[0] == len(offsetX)):
            raise ValueError(f'views: {len(views)} active, {xf.shape[0]} in the .xf, {len(offsetX)} in the XML')
        if not useTilt.all():
            raise ValueError('the XML has views with UseTilt=False: only active views are expected')

        # pixel size used when the XML was written: ImageDimensionsAngstrom / width in pixels
        width = ts.getFirstItem().getDim()[0]
        pixelSize = imageDims[0] / float(width)

        rows = self._offsetsToXf(np.stack([offsetY, offsetX], axis=1), xf[:, :4], pixelSize)
        return {ti.getAcquisitionOrder(): self._xfRowToMatrix(row) for ti, row in zip(views, rows)}

    # -------------------------- INFO functions ------------------------------
    def _validate(self) -> List[str]:
        errors = []

        downsample = (self.downsample.get() or '').split()
        alignment = (self.alignment.get() or '').split()
        n_downsample = len(downsample)
        n_alignment = len(alignment)
        n_iterations = self.num_iterations.get()

        inTsSet = self.inTsSet.get()
        model = self.model.get()

        if model is not None and self.num_iterations.get() > model.getNIterations():
            errors.append(f'The model has {model.getNIterations()} iterations: '
                          f'the inference cannot use more iterations than the run has models for.')

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

        return errors

    def _warnings(self) -> List[str]:
        warnings = []

        inTsSet = self.inTsSet.get()
        if inTsSet is not None and self.preprocess.get():
            if any(ts.hasAlignment() for ts in inTsSet):
                warnings.append('Some tilt-series already have coarse alignment: Preprocess will recompute it.')
            return warnings