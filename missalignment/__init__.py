# -*- coding: utf-8 -*-
import pwem
from pyworkflow import TOMO

from missalignment.constants import V0_2_0, MISSALIGNMENT_ENV_ACTIVATION, MISSALIGNMENT_DEFAULT_ACTIVATION_CMD, \
    MISSALIGNMENT, MISSALIGNMENT_DEFAULT_VERSION, MISSALIGNMENT_ENV_NAME, PYTHON_VERSION, CUDA_TOOLKIT_VERSION, \
    V2_8_0, PYTHON_VERSION, V0_11, TORCH_VERSION, TORCH_PROJECTORS_VERSION

_logo = "icon.png"
__version__ = '3.0.0'
_references = ['Chaillet2026.04.29.721716']


class Plugin(pwem.Plugin):
    _supportedVersions = [V0_2_0]
    _url = 'https://github.com/warpem/miss-alignment.git'
    _processingField = [TOMO]

    @classmethod
    def _defineVariables(cls):
        cls._defineVar(MISSALIGNMENT_ENV_ACTIVATION, MISSALIGNMENT_DEFAULT_ACTIVATION_CMD)  # association key-value

    @classmethod
    def getMissAlignmentEnvActivation(cls):
        return cls.getVar(MISSALIGNMENT_ENV_ACTIVATION)

    @classmethod
    def defineBinaries(cls, env):
        MISSALIGNMENT_INSTALLED = f'{MISSALIGNMENT}_{MISSALIGNMENT_DEFAULT_VERSION}_installed'
        installationCmd = cls.getCondaActivationCmd()

        # Create the environment
        installationCmd += (f'conda create -n {MISSALIGNMENT_ENV_NAME} -c conda-forge python={PYTHON_VERSION} '
                            f'cuda-toolkit={CUDA_TOOLKIT_VERSION} -y &&')

        # Activate new environment
        installationCmd += f'conda activate {MISSALIGNMENT_ENV_NAME} &&'

        # Fix some GPU dependencies
        installationCmd += f'python -m pip install torch=={TORCH_VERSION} &&'
        installationCmd += f'python -m pip install torch-projectors>={TORCH_PROJECTORS_VERSION} --index-url https://warpem.github.io/torch-projectors/cu129/simple/ &&'

        # Install missalignment
        installationCmd += f'python -m pip install miss-alignment=={MISSALIGNMENT_DEFAULT_VERSION} &&'

        # Flag installation finished
        installationCmd += f'touch {MISSALIGNMENT_INSTALLED}'

        # Checkpoint
        MISSALIGNMENT_commands = [(installationCmd, MISSALIGNMENT_INSTALLED)]
        # envPath = os.environ.get('PATH', "")  # keep path since conda likely in there
        # installEnvVars = {'PATH': envPath} if envPath else None

        # Scipion package
        env.addPackage(MISSALIGNMENT,
                       version=MISSALIGNMENT_DEFAULT_VERSION,
                       tar='void.tgz',
                       commands=MISSALIGNMENT_commands,
                       neededProgs=cls.getDependencies(),
                       # check if dependencies are available before starting the installation
                       # vars=installEnvVars,
                       default=True)

    @classmethod
    def getDependencies(cls):
        # try to get CONDA activation command
        condaActivationCmd = cls.getCondaActivationCmd()
        neededProgs = []
        if not condaActivationCmd:
            neededProgs.append('conda')
        return neededProgs

    @classmethod
    def getEnviron(cls):
        """ Set up the environment variables needed to launch miss-alignment. """
        environ = Environ(os.environ)


    @classmethod
    def runMissalignment(cls, protocol, program, args, cwd=None, numberOfMpi=1, useGpu:bool=True):
        """ Run miss-alignment command from a given protocol. """
        cmd = cls.getCondaActivationCmd() + " "
        cmd += cls.getMissAlignmentEnvActivation()
        cmd += f" && CUDA_VISIBLE_DEVICES=%(GPU)s {program} " if useGpu else f" && {program} "

        protocol.runJob(cmd, args, env=cls.getEnviron(), cwd=cwd, numberOfMpi=numberOfMpi)
