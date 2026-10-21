# -*- coding: utf-8 -*-
import pwem
from pyworkflow import TOMO

from missalignment.constants import V0_2_0, MISSALIGNMENT_ENV_ACTIVATION, MISSALIGNMENT_DEFAULT_ACTIVATION_CMD, \
    MISSALIGNMENT, MISSALIGNMENT_DEFAULT_VERSION, MISSALIGNMENT_ENV_NAME

_logo = "img.png"
__version__ = '3.0.0'
_references = ['Chaillet2026.04.29.721716']

class Plugin(pwem.Plugin):
    _supportedVersion = [V0_2_0]
    _url = 'https://github.com/warpem/miss-alignment.git'
    _processingField = [TOMO]



