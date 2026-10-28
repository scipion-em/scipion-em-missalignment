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

from missalignment.constants import IN_TS_SET, GLOBAL
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from typing import List
from pyworkflow.protocol import PointerParam, EnumParam, IntParam, LEVEL_ADVANCED, FloatParam, StringParam, GPU_LIST, \
    GE, BooleanParam
from pyworkflow.utils import Message, createAbsLink, makePath

from pyworkflow.object import Boolean

from tomo.utils import getTsIdsDicts
from os.path import join

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
        wConfId = self._insertFunctionStep(self.writeConfigStep,
                                           prerequisites=cInputId,
                                           needsGPU=False)
        trainId = self._insertFunctionStep(self.trainingStep,
                                           prerequisites=wConfId,
                                           needsGPU=True)
        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=trainId,
                                 needsGPU=False)

    # -------------------------- STEPS functions ------------------------------
    def convertInputStep(self, tsId):
        ts = self.tsDict[tsId]
        imodDir = self._makeTsDir(tsId)


        ts.writeTltFile(imodDir)
        ts.writeXfFile(join(imodDir, f'{tsId}.xf')) #TODO excluded views handling

        source = ts.getFirstItem().getFileName()
        destination = join(imodDir, f'{tsId}.st')
        createAbsLink(source, destination)

    def writeConfigStep(self):
        pass
    def trainingStep(self):
        pass
    def createOutputStep(self):
        pass



    # -------------------------- UTILS functions ------------------------------
    def _makeTsDir(self, tsId):
        tsDir = self._getExtraPath('imod', tsId)
        makePath(tsDir)
        return tsDir



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
