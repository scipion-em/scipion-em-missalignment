from pwem import EMObject
import pyworkflow.object as pwobj

class MissalignmentModel(EMObject):
    def __init__(self, runDirectory=None, nIterations=0, **kwargs):
        EMObject.__init__(self, **kwargs)
        self._runDirectory = pwobj.String(runDirectory)
        self._nIterations = pwobj.Integer(nIterations)