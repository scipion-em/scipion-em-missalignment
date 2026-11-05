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
from os.path import abspath

import yaml

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
    setOfTiltSeries = SetOfTiltSeries


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
        form.addParam('start_at_iteration', IntParam,
                      label='Start at iteration',
                      default=0,
                      help='Continue from a specific iteration in the config.')

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
        config['tilt_series_alignment']['patch_size'] = model.getPatchSize()
        config['tilt_series_alignment']['patch_overlap'] = self.patch_overlap.get()

        with open(self._getExtraPath('inference_config.yaml'), 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None)

    def inferStep(self):
        logger.info(cyanStr(f"Tilt-series alignment...'"))

        try:
            Plugin.runMissalignment(self, self._program, self._generateArguments(), useGpu=True)
        except Exception as e:
            logger.error(redStr('Alignment failed!'))
            raise

    def createOutputStep(self):
        pass

    # -------------------------- UTILS functions ------------------------------
    def _getConfigFile(self):
        return abspath(self._getExtraPath('inference_config.yaml'))

    def _generateArguments(self):

        cmd =[
            f'--config_file {self._getConfigFile()}',
            f'--start_at_iteration {self.start_at_iteration.get()}'
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
                warnings.append('Some tilt-series already have a coarse alignment: Preprocess will recompute it.')
            return warnings