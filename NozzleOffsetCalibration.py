#!/usr/bin/python3
# -*- coding: utf-8 -*-

# MIT License
# ===========
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import os
import math
import cv2
import requests
import json
from time import time  # hack-ish
import yaml
import numpy as np
from enum import Enum

from Job import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import uic


class PrinterSettings(object):
    def __init__(self):
        self.host = 'octopi.local'
        self.apiKey = ''
        self.octoprintPort = 80
        self.mjpgstreamerPort = 8080
        self.tools = 2
        self.parameters = '{\n}'
        self.loadCmds = 'M400'
        self.loadEndPos = '({X} {Y} {Z})'
        self.unloadCmds = 'M400'
        self.unloadEndPos = '({X} {Y} {Z})'

class PrinterThreadState(Enum):
    IDLE = 1
    POLLING = 2
    STOPPED = 3


class Printer(QThread):

    def __init__(self, settings):
        super(Printer, self).__init__()
        self.pos = None
        self.tool = None
        self.settings = settings
        self.camURL = 'http://{0}:{1}/?action=stream'.format(settings.host, settings.mjpgstreamerPort)
        self.baseURL = 'http://{0}:{1}/api'.format(settings.host, settings.octoprintPort)
        self.state = PrinterThreadState.IDLE
        print('Version: ', self._get('version').content)

    def run(self):
        cam = cv2.VideoCapture(self.camURL)
        polltime = time()
        while not self.state == PrinterThreadState.STOPPED:
            if not cam.isOpened():
                cam = cv2.VideoCapture(self.camURL)
            if cam.isOpened():
                ok, self.frame = cam.read()
                if ok: self.emit(SIGNAL('updateImage()'))

            if self.state == PrinterThreadState.POLLING and time() > polltime:
                params = {'exclude': 'sd,temperature', 'history': 'false'}
                r = self._get('printer', params)
                if r.status_code == 200 and r.json()['state']['text'] == 'Operational':
                    self.state = PrinterThreadState.IDLE
                else:
                    polltime = time() + 0.5
            self.yieldCurrentThread()

    def _get(self, command, params={}):
        url = '{0}/{1}'.format(self.baseURL, command);
        headers = {'X-Api-Key': self.settings.apiKey}
        r = requests.get(url, headers=headers, params=params)
        return r

    def _post(self, command, payload={}):
        print('start post: {0} {1}'.format(command, payload))
        url = '{0}/{1}'.format(self.baseURL, command);
        headers = {
            'content-type': 'application/json',
            'X-Api-Key':  self.settings.apiKey
        }
        r = requests.post(url, headers=headers, data=json.dumps(payload))

        print('end post: {0} {1}'.format(command, r))
        return r

    def _execute(self, commands):
        url = '{0}/files/local'.format(self.baseURL);
        headers = {
            'X-Api-Key': self.settings.apiKey
        }
        multipart_form_data = {
            'file': ('NozzleOffsetCalibration.gcode', commands),
            'select': ('', 'true'),
            'print': ('', 'true')
        }
        r = requests.post(url, headers=headers, files=multipart_form_data)

        self.state = PrinterThreadState.POLLING
        while self.state == PrinterThreadState.POLLING:
            QCoreApplication.processEvents()

    def home(self):
        self._execute('G28')
        self.pos = (0.0, 0.0, 0.0)

    def moveAbsWait(self, X=float('nan'), Y=float('nan'), Z=float('nan'), feed=float('nan')):
        p = [
            self.pos[0] if math.isnan(X) else X,
            self.pos[1] if math.isnan(Y) else Y,
            self.pos[2] if math.isnan(Z) else Z
        ]
        self._execute('G0 X{0[0]:.3f} Y{0[1]:.3f} Z{0[2]:.3f} F{1:.0f}'.format(p, feed));
        self.pos = p

    def moveAbs(self, X=float('nan'), Y=float('nan'), Z=float('nan'), feed=float('nan')):
        p = [
            self.pos[0] if math.isnan(X) else X,
            self.pos[1] if math.isnan(Y) else Y,
            self.pos[2] if math.isnan(Z) else Z
        ]

        payload = {
            'command': 'jog',
            'absolute': 'true',
            'x': p[0], 'y': p[1], 'z': p[2],
            'speed': 'false' if math.isnan(feed) else int(feed)
        }
        self._post('printer/printhead', payload)
        self.pos = p

    def moveRel(self, X=float('nan'), Y=float('nan'), Z=float('nan'), feed=float('nan'), wait = False):
        if wait:
            self.moveAbsWait(self.pos[0] + X, self.pos[1] + Y, self.pos[2] + Z, feed)
        else:
            self.moveAbs(self.pos[0] + X, self.pos[1] + Y, self.pos[2] + Z, feed)

    def loadTool(self, tool, setting):
        params = eval(self.settings.parameters, None, {'tool': tool})
        params['X'] = self.pos[0]
        params['Y'] = self.pos[1]
        params['Z'] = self.pos[2]
        self._execute(self.settings.loadCmds.format(**params))
        self.pos = eval(self.settings.loadEndPos, None, params)
        self.tool = tool

    def unloadTool(self, setting):
        params = eval(self.settings.parameters, None, {'tool': self.tool})
        params['X'] = self.pos[0]
        params['Y'] = self.pos[1]
        params['Z'] = self.pos[2]
        self._execute(self.settings.unloadCmds.format(**params))
        self.pos = eval(self.settings.unloadEndPos, None, params)
        self.tool = None

    def inspectTool(self, tool, position, setting):
        commands = []
        if self.tool is not None:
            params = eval(self.settings.parameters, None, {'tool': self.tool})
            commands.append(self.settings.unloadCmds.format(**params))
        params = eval(self.settings.parameters, None, {'tool': tool})
        commands.append(self.settings.loadCmds.format(**params))

        params['X'] = self.pos[0]
        params['Y'] = self.pos[1]
        params['Z'] = self.pos[2]
        diff = np.array(eval(self.settings.loadEndPos, None, params)) - position
        length = np.linalg.norm(diff)
        if length > 2:
            # run fast if distance is more than 2mm, but ensure that Z axis will go first
            fast = position + diff * 2/length
            commands.append('G0 Z{2}\nG0 X{0} Y{1}'.format(fast[0], fast[1], position[2]))
        commands.append('G0 Z{2}\nG0 X{0} Y{1} F1200'.format(*position))

        self._execute('\n'.join(commands))
        self.tool = tool
        self.pos = position


class Tool(yaml.YAMLObject):
    yaml_tag = u'!Tool'

    def __init__(self, nr, image=None, position=None, radii=(50, 100), ZOffset=0):
        self.nr = nr
        self.image = image
        self.position = position
        self.radii = radii
        self.ZOffset = ZOffset

    @classmethod
    def to_yaml(cls, dumper, data):
        node = Tool(data.nr)
        node.position = tuple(float(e) for e in data.position)
        node.radii = data.radii
        node.ZOffset = data.ZOffset
        return dumper.represent_yaml_object(
            cls.yaml_tag, node, cls, flow_style=cls.yaml_flow_style
        )

class Image(object):

    def __init__(self, im, crop=-1):
        rows, cols = im.shape[0:2]
        if crop == -1:
            self.size = min(rows, cols)
        else:
            self.size = crop
        self.image = im[(rows - self.size) // 2:(rows + self.size) // 2,
                     (cols - self.size) // 2:(cols + self.size) // 2]
        self.gray = None
        self.laplacian = None

    def getPixmap(self, radii):
        im = cv2.flip(self.image, 0)

        for radius in radii:
            cv2.circle(im, (self.size // 2, self.size // 2), radius, (0, 255, 0), 1)

        return QPixmap(
            QImage(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), self.size, self.size, 3 * self.size, QImage.Format_RGB888)
        )

    def _masked_pixels(self, im, radii):
        c = self.size // 2
        (ri, ro) = radii
        pixels = np.append(im[c - ro - 1:c - ri, c], im[c + ri - 1:c + ro, c])

        for y in range(1, ri):
            xo = int(math.sqrt(ro ** 2 - y ** 2))
            xi = int(math.sqrt(ri ** 2 - y ** 2))
            pixels = np.append(pixels, im[c - xo - 1:c - xi, c - y])
            pixels = np.append(pixels, im[c - xo - 1:c - xi, c + y])
            pixels = np.append(pixels, im[c + xi - 1:c + xo, c - y])
            pixels = np.append(pixels, im[c + xi - 1:c + xo, c + y])

        for y in range(ri, ro + 1):
            xo = int(math.sqrt(ro ** 2 - y ** 2))
            pixels = np.append(pixels, im[c - xo - 1:c + xo, c - y])
            pixels = np.append(pixels, im[c - xo - 1:c + xo, c + y])

        return pixels

    def _getLaplacian(self):
        if self.gray is None:
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if self.laplacian is None:
            self.laplacian = cv2.Laplacian(np.float32(self.gray), -1, ksize=5)

        return self.laplacian

    def variance(self, radii):
        return self._masked_pixels(self._getLaplacian(), tuple(sorted(radii))).var()

    def getOffset(self, ref):
        if self._getLaplacian().shape != ref._getLaplacian().shape:
            return np.array((float('nan'), float('nan')))
        return cv2.phaseCorrelate(self._getLaplacian(), ref._getLaplacian())[0]


class PrinterSettingDialog(QDialog):
    def __init__(self, parent=None):
        super(PrinterSettingDialog, self).__init__(parent)
        uic.loadUi('QTPrinterSetting.ui', self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.show()

    def setData(self, data):
        self.leHost.setText(data.host)
        self.leApiKey.setText(data.apiKey)
        self.sbOctoprintPort.setValue(data.octoprintPort)
        self.sbMjpgstreamerPort.setValue(data.mjpgstreamerPort)
        self.sbTools.setValue(data.tools)
        self.teParameters.setText(data.parameters)
        self.teLoadCmds.setText(data.loadCmds)
        self.leLoadEndPos.setText(data.loadEndPos)
        self.teUnloadCmds.setText(data.unloadCmds)
        self.leUnloadEndPos.setText(data.unloadEndPos)

    def getData(self):
        data = PrinterSettings()
        data.host = str(self.leHost.text())
        data.apiKey = str(self.leApiKey.text())
        data.octoprintPort = self.sbOctoprintPort.value()
        data.mjpgstreamerPort = self.sbMjpgstreamerPort.value()
        data.parameters = str(self.teParameters.toPlainText())
        data.loadCmds = str(self.teLoadCmds.toPlainText())
        data.loadEndPos = str(self.leLoadEndPos.text())
        data.unloadCmds = str(self.teUnloadCmds.toPlainText())
        data.unloadEndPos = str(self.leUnloadEndPos.text())
        return data


class PrinterCalibration(QMainWindow):
    def __init__(self):
        super(PrinterCalibration, self).__init__()
        uic.loadUi('NozzleOffsetCalibration.ui', self)
        self.cbStepSize.addItems(['0.01', '0.05', '0.1', '0.5', '1', '5', '10', '50'])

        self.progress = QProgressBar()
        self.stop = QPushButton('Stop')
        self.statusbar.addPermanentWidget(self.progress)
        self.statusbar.addPermanentWidget(self.stop)
        self.progress.hide()
        self.stop.hide()

        self.printerSettings = None
        self.printer = None
        self.job = False
        self.camera = None
        self._acNew()

        self.btHome.clicked.connect(self._moveHome)
        self.btXPYP.clicked.connect(self._moveXPYP)
        self.btYP.clicked.connect(self._moveYP)
        self.btXNYP.clicked.connect(self._moveXNYP)
        self.btXP.clicked.connect(self._moveXP)
        self.btXN.clicked.connect(self._moveXN)
        self.btXPYN.clicked.connect(self._moveXPYN)
        self.btYN.clicked.connect(self._moveYN)
        self.btXNYN.clicked.connect(self._moveXNYN)
        self.btAutoZ.clicked.connect(self._autoZ)
        self.btZP.clicked.connect(self._moveZP)
        self.btZN.clicked.connect(self._moveZN)
        self.btInspectPos.clicked.connect(self._setInspectPos)
        self.btRegisterTool.clicked.connect(self._registerTool)

        self.btInspect.clicked.connect(self._inspectTool)
        self.btLoad.clicked.connect(self._loadTool)
        self.btUnload.clicked.connect(self._unloadTool)

        self.acNew.triggered.connect(self._acNew)
        self.acOpen.triggered.connect(self._acOpen)
        self.acSave.triggered.connect(self._acSave)
        self.acSaveAs.triggered.connect(self._acSaveAs)
        self.acQuit.triggered.connect(self.close)
        self.acCheck.triggered.connect(self._acCheck)
        self.acRepeatability.triggered.connect(self._acRepeatability)
        self.acPrinter.triggered.connect(self._acPrinterTriggered)

        self.show()

    def _changeSettings(self, new):
        if self.printer:
            self.printer.state = PrinterThreadState.STOPPED
            self.printer.wait();

        self.printer = Printer(new)
        self.lbCaptureImage.connect(self.printer, SIGNAL('updateImage()'), self.updateImage)
        self.printer.start()

        self.sbTool.setMaximum(new.tools - 1)
        if len(self.tools) > new.tools:
            self.tools = self.tools[0:new.tools]
        else:
            while len(self.tools) < new.tools:
                self.tools.append(Tool(len(self.tools)))

        self.focus = float('nan')
        self.offset = (float('nan'), float('nan'))
        self.resolution = float('nan')

        self._updateUI()
        self.printerSettings = new

    def _acNew(self):
        self.filename = ''
        self.inspectPos = None
        self.tools = []
        self._changeSettings(PrinterSettings())
        self._updateUI()
        self._updateOffsets()

    def _save(self, filename):
        if filename == '':
            filename = QFileDialog.getSaveFileName(
                self, 'Save printer settings as ', self.filename,
                'Printer Settings (*.yaml);;All Files (*)',
                options=QFileDialog.DontUseNativeDialog
            )

        if filename != '':
            with open(filename, 'w') as f:
                data = {
                    'PrinterSettings': self.printerSettings,
                    'InspectPosition': self.inspectPos,
                    'Tools': self.tools
                }
                yaml.dump(data, f, )
                self.filename = filename

    def _acOpen(self):
        filename = QFileDialog.getOpenFileName(
            self, 'Open printer settings', '',
            'Printer settings (*.yaml);;All Files (*)',
            options=QFileDialog.DontUseNativeDialog
        )

        if filename != '':
            with open(filename, 'r') as f:
                data = yaml.load(f)
                self.tools = data['Tools']
                self.inspectPos = data['InspectPosition']
                self._changeSettings(data['PrinterSettings'])
                self._updateOffsets()
                self.filename = filename

    def _acSave(self):
        self._save(self.filename)

    def _acSaveAs(self):
        self._save('')

    def _acPrinterTriggered(self):
        dl = PrinterSettingDialog(self)
        dl.setData(self.printerSettings)
        if dl.exec_():
            self._changeSettings(dl.getData())

    def _runJob(self, job):
        self.job = True
        self.jobobj = job

        job.run(self, self.printer)

        self.job = False
        self._updateUI()

    def _acRepeatability(self):
        self._runJob(Repeatability())

    def _acCheck(self):
        self._runJob(Check())

    def _moveHome(self):
        self.printer.home()

    def _moveStep(self, X=float('nan'), Y=float('nan'), Z=float('nan')):
        step = float(self.cbStepSize.currentText())
        self.printer.moveRel(X=step * X, Y=step * Y, Z=step * Z)

    def _moveXNYP(self):
        self._moveStep(X=-1, Y=1)

    def _moveYP(self):
        self._moveStep(Y=1)

    def _moveXPYP(self):
        self._moveStep(X=1, Y=1)

    def _moveXN(self):
        self._moveStep(X=-1)

    def _moveXP(self):
        self._moveStep(X=1)

    def _moveXNYN(self):
        self._moveStep(X=-1, Y=-1)

    def _moveYN(self):
        self._moveStep(Y=-1)

    def _moveXPYN(self):
        self._moveStep(X=1, Y=-1)

    def _autoZ(self):
        self._runJob(AutoZ())

    def _moveZP(self):
        self._moveStep(Z=1)

    def _moveZN(self):
        self._moveStep(Z=-1)

    def _setInspectPos(self):
        self.inspectPos = self.printer.pos
        self._updateUI()

    def _registerTool(self):
        if self.printer.tool is not None:
            self.tools[self.printer.tool].position = self.printer.pos
            self.tools[self.printer.tool].image = self.image
            self.tools[self.printer.tool].radii = tuple(
                sorted((self.spInnerRadius.value(), self.spOuterRadius.value())))
            self.tools[self.printer.tool].ZOffset = self.spZOffset.value()
            self._updateOffsets()

        self._updateUI()

    def _inspectTool(self):
        T = self.tools[self.sbTool.value()]
        pos = T.position
        if pos is None:
            pos = (self.inspectPos[0], self.inspectPos[1], self.inspectPos[2] + self.spZOffset.value())
        else:
            self.spZOffset.setValue(T.ZOffset)

        self.spInnerRadius.setValue(T.radii[0])
        self.spOuterRadius.setValue(T.radii[1])
        self.printer.inspectTool(self.sbTool.value(), pos, self.printerSettings)
        self._updateUI()

    def _loadTool(self):
        T = self.tools[self.sbTool.value()]
        self.spInnerRadius.setValue(T.radii[0])
        self.spOuterRadius.setValue(T.radii[1])
        self.spZOffset.setValue(T.ZOffset)
        self.printer.loadTool(self.sbTool.value(), self.printerSettings)
        self._updateUI()

    def _unloadTool(self):
        self.printer.unloadTool(self.printerSettings)
        self._updateUI()

    def _updateOffsets(self):
        positions = []
        for tool in self.tools:
            if tool.position is not None:
                positions.append(tool.position)

        out = ''
        if len(positions) > 0:
            x, y = np.average(np.array(positions), axis=0)[0:2]
            z = np.array(positions).min(axis=0)[2]
            center = np.array((x, y, z))

            for tool in self.tools:
                if tool.position is not None:
                    out += 'Tool:{0} X:{1[0]:>6.2f} Y:{1[1]:>6.2f} Z:{1[2]:>6.2f}\n'.format(tool.nr,
                                                                                            tool.position - center)

        self.lbOffset.setText(out)

    def _updateUI(self):
        default = self.printer is not None and not self.job and \
                  self.printer.pos is not None and self.printer.state == PrinterThreadState.IDLE
        for bt in [self.btXPYP, self.btYP, self.btXNYP, self.btXP, self.btXN,
                   self.btXPYN, self.btYN, self.btXNYN, self.btZP, self.btZN, self.btAutoZ]:
            bt.setEnabled(default)

        for bt in [self.btUnload, self.btInspectPos, self.btRegisterTool]:
            bt.setEnabled(default and self.printer.tool is not None)

        self.btHome.setEnabled(self.printer is not None and not self.job)
        self.btInspect.setEnabled(default and self.inspectPos is not None)
        self.btLoad.setEnabled(default and self.printer.tool is None)

        if self.inspectPos is not None:
            self.lbInspectPos.setText('Inspect position X:{0[0]:.3f} Y:{0[1]:.3f} Z:{0[2]:.3f}'.format(self.inspectPos))
        else:
            self.lbInspectPos.setText('Inspect position not set')

        self.spOuterRadius.setMaximum(self.sbCrop.value() / 2)
        self.spInnerRadius.setMaximum(self.sbCrop.value() / 2)

        self.setWindowTitle('Printer Calibration - {0}'.format(self.filename))

    def updateImageData(self, stats=False):
        if stats and self.image is not None:
            radii = (self.spInnerRadius.value(), self.spOuterRadius.value())
            self.focus = self.image.variance(radii)

            if self.printer is not None and self.printer.tool is not None and \
                    self.tools[self.printer.tool].image is not None:
                tool = self.tools[self.printer.tool]
                self.offset = np.array(self.image.getOffset(tool.image))
                dpx = np.linalg.norm(self.offset)
                dm = np.linalg.norm(np.array(tool.position) - self.printer.pos)
                if dpx > 3:
                    self.resolution = dm * 1000 / dpx
                else:
                    self.resolution = float('nan')
        else:
            self.offset = [float('nan'), float('nan')]
            self.resolution = float('nan')
            self.focus = float('nan')

        self.statusbar.showMessage('focus:{0:.0f} offset:{1[0]:.1f}:{1[1]:.1f} resolution:{2:.1f} um/px' \
                                   .format(self.focus, self.offset, self.resolution))

    def updateImage(self):
        self.image = Image(self.printer.frame, self.sbCrop.value())
        radii = (self.spInnerRadius.value(), self.spOuterRadius.value())
        pixmap = self.image.getPixmap(radii)
        self.lbCaptureImage.hide()  # prevents flicker, can't explain that
        self.lbCaptureImage.setPixmap(pixmap)
        self.lbCaptureImage.resize(self.image.size, self.image.size)
        self.lbCaptureImage.show()
        self.updateImageData(stats = self.job)
        self._updateUI()

        if self.job and self.printer.state == PrinterThreadState.IDLE:
            self.jobobj.notify()

        if self.printer and self.printer.pos:
            self.lbPrinterData.setText(
                'X:{0[0]:.3f} Y:{0[1]:.3f} Z:{0[2]:.3f} Tool:{1}'.format(self.printer.pos, self.printer.tool)
            )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("cleanlooks"))
    window = PrinterCalibration()
    sys.exit(app.exec_())
