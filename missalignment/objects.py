from pwem import EMObject
import pyworkflow.object as pwobj
from tomo.objects import TiltSeries


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


class MissalignmentTiltSeries(TiltSeries):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._xmlFile = String()

    def setXmlFile(self, xmlFile):
        self._xmlFile.set(xmlFile)

class SetOfMissalignmentTiltSeries(TiltSeries):
    ITEM_TYPE = MissalignmentTiltSeries
    EXPOSE_ITEMS = True