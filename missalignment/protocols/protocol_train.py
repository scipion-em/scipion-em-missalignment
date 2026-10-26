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

from missalignment.constants import IN_TS_SET
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.gui import form
from pyworkflow.object import String

from pyworkflow.protocol import StringParam, Form, PointerParam, EnumParam, IntParam, LEVEL_ADVANCED, FloatParam
from pyworkflow.utils import Message

logger = logging.getLogger(__name__)

class Outputobjects(Enum):
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
    def _defineParams(self,fomr):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam(IN_TS_SET,PointerParam,
                      label='Tilt-series',
                      important=True,
                      help='Tilt-series with a coarse alignment')

        group = form.addGroup('Alignment')
        group.addParam('iteration',EnumParam,
                       label='Alignment modes',
                       choices = ['global', 'anchoring', 'spline'],
                       default=GLOBAL) #TODO
        group.addParam('batch_size', IntParam,
                       label='Batch size',
                       default=32,
                       help='Amount of patches simultaneously reconstructed in memory')
        group.addParam('patch_size', IntParam,
                       label='Patch size',
                       expertLevel= LEVEL_ADVANCED,
                       default= 96,
                       help='Size of the cubic patches (small subtomograms) '
                           'reconstructed from the tilt-series. They are used both to '
                           'train the model and to refine the alignment, so the same '
                           'value is applied to both steps. Larger patches contain more '
                           'context but need more GPU memory.')
        group.addParam('patch_overlap', FloatParam,
                       label='Patch overlap',
                       expertLevel= LEVEL_ADVANCED,
                       default = 0.1,
                       help='Tolerated overlap between patches used for optimizing the alignment')





