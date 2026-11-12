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
import os
from typing import Tuple, Union, List, Optional

from imod.constants import OUTPUT_TILTSERIES_NAME
from imod.protocols import ProtImodExcludeViews, ProtImodImportTransformationMatrix
from missalignment.objects import MissalignmentModel
from missalignment.protocols import ProtMissalignmentInfer, ProtMissalignmentTraining
from tomo.objects import SetOfTiltSeries, SetOfCTFTomoSeries, CTFTomoSeries, TiltSeries

from tomo.tests import EMPIAR_10491_ODD_EVEN, DataSetEmd10491_Odd_Even

from pyworkflow.tests import setupTestProject, DataSet

from pyworkflow.utils import cyanStr, magentaStr, magenta

from tomo.protocols import ProtImportTs

from pwem import ALIGN_2D

from tomo.tests.test_base_centralized_layer import TestBaseCentralizedLayer

TS_1 = 'TS_1'
TS_32 = 'TS_32'


class TestMissalignment(TestBaseCentralizedLayer):
    nTs = 2
    expectedTsDims = [5760, 4092, 41]
    anglesCountDict = {
        TS_1: 41,
        TS_32: 41,
    }
    anglesCountRestackDict = {
        TS_1: 37,
        TS_32: 36,
    }
    excludedViewsDict = {
        TS_1: [0, 1, 2, 40],
        TS_32: [0, 1, 38, 39, 40]
    }

    UNMODIFIED = 'unmodified'
    EXC_VIEWS = 'exc. views'
    RE_STACKED = 're-stacked'

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet(EMPIAR_10491_ODD_EVEN)
        cls.sRate = 3.16

    @classmethod
    def _runPreviousProtocols(cls, eVTs: bool = False, noCoarseAl:bool=False) \
            -> Tuple[SetOfTiltSeries]:
        print(cyanStr('--------------------------------- RUNNING PREVIOUS PROTOCOLS ---------------------------------'))
        importedTs = cls._runImportTs()
        if not noCoarseAl:
            importedTs = cls._runImportTrMatrix(importedTs)
        if eVTs:
            cls._excludeSetViews(importedTs)
        print(
            cyanStr(
                '\n-------------------------------- PREVIOUS PROTOCOLS FINISHED ---------------------------------'))
        return importedTs

    @classmethod
    def _excludeSetViews(cls,
                         inSet: Union[SetOfTiltSeries],
                         excludedViewsDict: Union[dict, None] = None) -> None:
        if not excludedViewsDict:
            excludedViewsDict = cls.excludedViewsDict
        objList = [obj.clone(ignoreAttrs=[]) for obj in inSet]
        for obj in objList:
            cls._excIntermediateSetViews(inSet, obj, excludedViewsDict[obj.getTsId()])

    @staticmethod
    def _excIntermediateSetViews(inSet: Union[SetOfTiltSeries],
                                 obj: Union[TiltSeries],
                                 excludedViewsList: List[int]) -> None:
        tiList = [ti.clone() for ti in obj]
        for i, ti in enumerate(tiList):
            if i in excludedViewsList:
                ti._objEnabled = False
                obj.update(ti)
        obj.write()
        inSet.update(obj)
        inSet.write()
        inSet.close()

    @classmethod
    def _runImportTs(cls) -> SetOfTiltSeries:
        print(magentaStr("\n==> Importing the tilt series:"))
        acq = DataSetEmd10491_Odd_Even.acq.value
        protImportTs = cls.newProtocol(ProtImportTs,
                                       filesPath=cls.ds.getFile(DataSetEmd10491_Odd_Even.tsDir.value),
                                       filesPattern=DataSetEmd10491_Odd_Even.tsPattern.value,
                                       exclusionWords='_even.mrcs _odd.mrcs',
                                       anglesFrom=0,  # From range
                                       minAngle=-34,
                                       maxAngle=46,
                                       stepAngle=acq.getStep(),
                                       voltage=acq.getVoltage(),
                                       magnification=acq.getMagnification(),
                                       sphericalAberration=acq.getSphericalAberration(),
                                       amplitudeContrast=acq.getAmplitudeContrast(),
                                       samplingRate=cls.sRate,
                                       doseInitial=acq.getDoseInitial(),
                                       dosePerFrame=acq.getDosePerFrame(),
                                       tiltAxisAngle=acq.getTiltAxisAngle())


        cls.launchProtocol(protImportTs)
        tsImported = getattr(protImportTs, 'outputTiltSeries', None)
        return tsImported

    @classmethod
    def _runImportTrMatrix(cls, inTsSet):
        print(magentaStr("\n==> Importing the TS' transformation matrices with IMOD:"))
        protImportTrMatrix = cls.newProtocol(ProtImodImportTransformationMatrix,
                                             filesPath=cls.ds.getFile(DataSetEmd10491_Odd_Even.tsAliDir.value),
                                             filesPattern=DataSetEmd10491_Odd_Even.aliFilesPattern.value,
                                             inputSetOfTiltSeries=inTsSet)
        cls.launchProtocol(protImportTrMatrix)
        outTsSet = getattr(protImportTrMatrix, OUTPUT_TILTSERIES_NAME, None)
        return outTsSet

    @classmethod
    def _runExcludeViewsProt(cls,
                             inTsSet: SetOfTiltSeries,
                             objLabel: str = None) -> SetOfTiltSeries:
        print(magentaStr("\n==> Running the TS exclusion of views:"))
        protExcViews = cls.newProtocol(ProtImodExcludeViews, inputSetOfTiltSeries=inTsSet)
        if objLabel:
            protExcViews.setObjLabel(objLabel)
        cls.launchProtocol(protExcViews)
        outTsSet = getattr(protExcViews, OUTPUT_TILTSERIES_NAME, None)
        return outTsSet

    def _runMissalignmentTraining(self,
                                  inTsSet: SetOfTiltSeries,
                                  tsSetMsg: str,
                                  preprocess=False) -> Optional[MissalignmentModel]:
        print(magenta(f'\n==> Running Missalignment - Training:'
                      f'\n\t- Tilt-series = {tsSetMsg}'))
        protTraining = self.newProtocol(ProtMissalignmentTraining,
                                        inTsSet=inTsSet,
                                        preprocess=preprocess,
                                        num_iterations=1,
                                        downsample='3',
                                        alignment='A',
                                        #training_devices='0,1',  # TODO can the gpuids be set
                                        reconstruction_devices='0,0,1,1'
                                        )
        objLabel = (f'\nMissalignment - Training: '
                    f'\n\t- ts = {tsSetMsg}')
        protTraining.setObjLabel(objLabel)
        self.launchProtocol(protTraining)
        modelTraining = getattr(protTraining, protTraining._possibleOutputs.model.name, None)
        return modelTraining

    def _checkModel(self, model: MissalignmentModel) -> None:
        # Check the Model
        self.assertIsNotNone(model, 'The training did not produce the final model')  # check if the output exists
        modelPath = model.getPath()
        self.assertTrue(os.path.exists(modelPath), f'{modelPath} does not exist')  # check if the file exists

    def _runMissalignmentInfer(self, inTsSet: SetOfTiltSeries,
                               model: MissalignmentModel) \
            -> SetOfTiltSeries:
        protInfer = self.newProtocol(ProtMissalignmentInfer,
                                     inTsSet=inTsSet,
                                     model=model,
                                     num_iterations=1,
                                     downsample='3',
                                     alignment='A'
                                     )
        objLabel = f'\nMissalignment - Inference: '
        protInfer.setObjLabel(objLabel)
        self.launchProtocol(protInfer)
        setOfTs = getattr(protInfer, protInfer._possibleOutputs.tiltSeries.name, None)
        return setOfTs

    def _checkTiltSeries(self, inTsSet: SetOfTiltSeries, eVTs=False, restacked=False) -> None:
        # Check the Tilt-series sets
        acq = DataSetEmd10491_Odd_Even.acq.value
        counts = self.anglesCountRestackDict if restacked else self.anglesCountDict
        self.checkTiltSeries(inTsSet,
                             expectedSetSize=self.nTs,
                             expectedSRate=self.sRate,
                             expectedDimensions=self.expectedTsDims, #TODO CAMBIA NUMERO DI IMGS
                             hasAlignment=True,
                             alignment=ALIGN_2D,
                             testSetAcqObj=acq,
                             testAcqObj=acq,
                             anglesCount=counts,
                             excludedViewsDict=self.excludedViewsDict if eVTs else None,
                             checkHeaderApix=False
                             )

    # TEST
    def _runTestMissalignment(self,
                              inTsSet: SetOfTiltSeries,
                              tsSetMsg: str,
                              eVTs=False,
                              restack=False,
                              preprocess=False
                              ) -> None:
        # Run Training
        model = self._runMissalignmentTraining(inTsSet=inTsSet, tsSetMsg=tsSetMsg)
        # Check the model
        self._checkModel(model)
        # Run inference
        setOfTs = self._runMissalignmentInfer(
            inTsSet=inTsSet,
            model=model)
        # Check tilt-series with alignment infos
        self._checkTiltSeries(setOfTs, eVTs=eVTs, restacked=restacked)

    def testMissalignment(self):
        importedTs = self._runPreviousProtocols()
        self._runTestMissalignment(inTsSet=importedTs,
                                   tsSetMsg=self.UNMODIFIED)

    def testMissalignment_noCoarseAl(self):
        importedTs = self._runPreviousProtocols(noCoarseAl=True)
        self._runTestMissalignment(inTsSet=importedTs,
                                   tsSetMsg=self.UNMODIFIED,
                                   preprocess=True)

    def testMissalignment_EV_Ts(self):
        importedTs = self._runPreviousProtocols(eVTs=True)
        self._runTestMissalignment(inTsSet=importedTs,
                                   tsSetMsg=self.EXC_VIEWS,
                                   eVTs=True
                                   )

    def testMissalignment_EV_Restacked_Ts(self):
        importedTs = self._runPreviousProtocols(eVTs=True)
        tsSetRestacked = self._runExcludeViewsProt(importedTs)  # Re-stack the TS
        self._runTestMissalignment(inTsSet=tsSetRestacked,
                                   tsSetMsg=self.RE_STACKED,
                                   restacked=True)
