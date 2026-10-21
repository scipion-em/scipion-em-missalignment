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
MISSALIGNMENT = 'missalignment'
MISSALIGNMENT_HOME = 'MISSALIGNMENT_HOME'

# Supported versions
V0_2_0 = '0.2.0'
V3_12 = '3.12'
V12_9 = '12.9'
V2_8_0 = '2.8.0'
V0_11 = '0.11.0'
MISSALIGNMENT_DEFAULT_VERSION = V0_2_0
PYTHON_VERSION = V3_12
CUDA_TOOLKIT_VERSION = V12_9
TORCH_VERSION = V2_8_0
TORCH_PROJECTORS_VERSION = V0_11

MISSALIGNMENT_ENV_NAME = f'{MISSALIGNMENT}--{MISSALIGNMENT_DEFAULT_VERSION}'
MISSALIGNMENT_ENV_ACTIVATION = 'MISSALIGNMENT_ENV_ACTIVATION'
MISSALIGNMENT_DEFAULT_ACTIVATION_CMD = f'conda activate {MISSALIGNMENT_ENV_NAME}'


