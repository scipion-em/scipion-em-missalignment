from pwem import EMObject
import pyworkflow.object as pwobj

class MissalignmentModel(EMObject):
    def __init__(self, runDirectory=None, nIterations=0,patch_size=96, **kwargs):
        EMObject.__init__(self, **kwargs)
        self._runDirectory = pwobj.String(runDirectory)
        self._nIterations = pwobj.Integer(nIterations)
        self._patch_size = pwobj.Integer(patch_size)

    def getPath(self):
        return self._runDirectory.get() #.warp_tiltseries

    def getNIterations(self):
        return self._nIterations.get()

    def getPatchSize(self):
        return self._patch_size.get()
