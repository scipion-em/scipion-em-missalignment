# **************************************************************************
# *
# * Authors:     Grigory Sharov (gsharov@mrc-lmb.cam.ac.uk)
# *              Yunior C. Fonseca Reyna (cfonseca@cnb.csic.es)
# *
# * MRC Laboratory of Molecular Biology (MRC-LMB)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
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
from pwem.protocols import EMProtocol
from typing import List, Tuple

logger = logging.getLogger(__name__)

class ProtMissalignmentBase(EMProtocol):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # -------------------------- STEPS functions ------------------------------
    def convertInputStep(self, tsId):
        logger.info(cyanStr('Compiling imod format files...'))

        try:
            ts = self.tsDict[tsId]
            imodDir = self._makeImodDir(tsId)

            present = ts.getTsPresentAcqOrders()
            self.writeRawTltFile(ts, imodDir, present=present)
            self.writeXfFile(ts, join(imodDir, f'{tsId}.xf'), present=present)

            source = ts.getFirstItem().getFileName()
            destination = join(imodDir, f'{tsId}.st')
            if ts.hasExcludedViews():
                ts.reStack(source, destination, present)
            else:
                createAbsLink(abspath(source), abspath(destination))

        except Exception as e:
            self.failedTsIds.append(tsId)
            logger.error(redStr(f'tsId = {tsId} -> input conversion failed with the exception -> {e}'))
            logger.error(traceback.format_exc())

    def toWarpStep(self):
        logger.info(cyanStr('Converting in .xml files...'))

        dims = []
        tiltAxis_angle = []

        for tsId, ts in self.tsDict.items():
            dims.append(self._getTomoDims(ts))
            # Average the tilt axis angle over the whole dataset
            tiltAxis_angle.append(ts.getAcquisition().getTiltAxisAngle())
        tilt_angle = sum(tiltAxis_angle) / len(tiltAxis_angle)

        # The tomogram shape should tightly the sample in all three dimensions (X, Y, Z) to avoid training
        # on empty regions — when samples vary in thickness, use the thickest one as the reference
        shape = [max(d[i] for d in dims) for i in range(3)]

        # Call the script etomo_to_warp.py to convert to the .xml format
        args = [
            f'--input-dir {self.getImodDir()}',
            f'--output-dir {self.getOutputDir()}',
            f'--tilt-axis-angle {tilt_angle}',
            f'--volume-shape {shape[0]} {shape[1]} {shape[2]}'
        ]
        if self.prepare_stacks.get():
            args.append(f'--output-pixel-size {self.prepare_stacks.get()}')
        else:
            args.append(f'--output-pixel-size {10}')
        args = ' '.join(args)

        Plugin.runMissalignment(self, f'python "{ETOMO_TO_WARP}"', args, useGpu=False)

    # ---------------------Utils functions-----------------------------

    def _makeImodDir(self, tsId):
        tsDir = self._getExtraPath('imod', tsId)
        makePath(tsDir)
        return tsDir

    def getImodDir(self):
        return abspath(self._getExtraPath('imod'))

    def _makeOutputDir(self):
        outputDir = self._getExtraPath('warp_tiltseries')
        makePath(outputDir)
        return outputDir

    def getOutputDir(self):
        return abspath(self._getExtraPath('warp_tiltseries'))

    def _parseAlignment(self, token):
        token = token.lower()  # 'G' and 'g', 'A' and 'a' are the same thing
        if token == 'g':
            return 'global'
        if token == 'a':
            return 'anchoring'

        if 'x' in token:  # converting the grid
            parts = token.split('x')  # 3x3 turns to [3, 3]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                m, n = int(parts[0]), int(parts[1])
                if m > 0 and n > 0:
                    return [m, n]
        raise ValueError(f'Alignment: "{token}" is not valid. '
                         f'Use G (global), A (anchoring) or a grid MxN.')

    def _getIteration(self):
        """One dict per iteration: {'downsample': int, 'alignment': 'global' | 'anchoring' | [M, N]}"""
        n = self.num_iterations.get()
        downsample = (self.downsample.get() or '').split()
        alignment = (self.alignment.get() or '').split()

        settings = []
        for i in range(n):
            entry = {}
            entry['downsample'] = int(downsample[i])
            entry['alignment'] = self._parseAlignment(alignment[i])
            settings.append(entry)
        return settings

    def _getTomoDims(self, ts) -> Tuple[int, int, int]:
        x = int(ts.getDimensions()[0])
        y = int(ts.getDimensions()[1])
        # The binary swap is exact only near φ∈{0,90}. Near 45°, the detector rectangle rotated into
        # the tomogram frame needs a box larger than either W×H or H×W. The exact axis-aligned bounding
        # box of the rotated rectangle covers every imaged pixel at any angle and self-pads near 45°I
        tiltAxisAngle = math.radians(ts.getAcquisition().getTiltAxisAngle())
        c, s = abs(math.cos(tiltAxisAngle)), abs(math.sin(tiltAxisAngle))
        tomoX = math.ceil(x * c + y * s)
        tomoY = math.ceil(x * s + y * c)
        tomoZ = int(max(tomoX, tomoY) / 3)
        return tomoX, tomoY, tomoZ

    def writeRawTltFile(self, ts, ts_folder, present):
        """Writes a raw tlt file.
        :param ts_folder: path of the directory in which the tlt file will be generated.
        :param present: retrieve ts acquisition order.
        """
        xtiltPath = ts_folder + '/%s.rawtlt' % ts.getTsId()
        with open(xtiltPath, 'w') as f:
            for ti in ts.iterItems(orderBy=ts.INDEX):
                if ti.getAcquisitionOrder() not in present:
                    continue
                f.write(str(ti.getTiltAngle()) + '\n')
            # For parallel processing, ensure that the file is completely written and persists on disk
            f.flush()  # Empty python buffer
            os.fsync(f.fileno())  # Empty system buffer

    def writeXfFile(self, ts, transformFilePath, present, delimiter='\t', factor=1):
        """ This method takes a tilt series and the output transformation file
        path and creates an IMOD-based transform
        file in the location indicated. """

        tsMatrixTransformList = []

        for ti in ts.iterItems(orderBy=ts.INDEX):
            if ti.getAcquisitionOrder() not in present:
                continue
            if ti.getTransform() is not None:
                transform = ti.getTransform().getMatrix().flatten()
                transformIMOD = ['%.7f' % transform[0],
                                 '%.7f' % transform[1],
                                 '%.7f' % transform[3],
                                 '%.7f' % transform[4],
                                 '%.3f' % (float(transform[2]) / factor),
                                 '%.3f' % (float(transform[5]) / factor)]
            else:
                from pyworkflow.utils import yellowStr
                logging.info(
                    yellowStr('WARNING: The Tilt series lacks of alignment information (transformation matrices). '
                              'The identity transformation will be written in the .xf file'))

                #  This is the identity matrix
                transformIMOD = ['1.0000000',
                                 '0.0000000',
                                 '0.0000000',
                                 '1.0000000',
                                 '0.000',
                                 '0.000']
            tsMatrixTransformList.append(transformIMOD)

        with open(transformFilePath, 'w') as f:
            csvW = csv.writer(f, delimiter=delimiter)
            csvW.writerows(tsMatrixTransformList)
            # For parallel processing, ensure that the file is completely written and persists on disk
            f.flush()  # Empty python buffer
            os.fsync(f.fileno())  # Empty system buffer
