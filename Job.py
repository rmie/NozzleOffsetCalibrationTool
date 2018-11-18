# -*- coding: utf-8 -*-
#
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

import numpy as np
import cv2

from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4 import uic
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Job(object):

    def __init__(self, cycles=1):
        self.cycles = cycles
        self.exit = False

    def _exit(self):
        self.exit = True

    def run(self, window, printer):
        self.window = window
        self.printer = printer

        if self.init():
            cycle = 0
            self.exit = False

            self.window.progress.setMaximum(self.cycles)
            self.window.stop.clicked.connect(self._exit)
            self.window.progress.show()
            self.window.stop.show()

            while not self.exit and cycle < self.cycles and self.loop(cycle):
                cycle += 1
                self.window.progress.setValue(cycle)

            self.window.stop.clicked.disconnect()
            self.window.progress.hide()
            self.window.stop.hide()

            if cycle == self.cycles:
                self.summary()

    def notify(self):
        self.blocked = False

    def block(self, count=1):
        for n in range(0, count):
            self.blocked = True
            while self.blocked:
                QCoreApplication.instance().processEvents()

    def init(self):
        return True

    def loop(self, count):
        return True

    def summary(self):
        pass

    def dialog(self, filename):
        D = QDialog(self.window)
        uic.loadUi(filename, D)
        return D


class AutoZ(Job):

    def _aquire(self, steps, res, repeat):
        Z = self.printer.pos[2]
        variances = []
        for n in range(0, 2 * steps + 1):
            self.printer.moveAbsWait(Z=Z + (n - steps) * res, feed=1000)
            self.window.progress.setValue(self.window.progress.value() + 1)
            for r in range(0, repeat):
                self.block()
                v = self.window.image.variance(self.radii)
                if r == 0:
                    variances.append([])
                variances[n].append(v)

        best = np.array(variances).sum(axis=1).argmax()
        self.printer.moveAbsWait(Z=Z + (best - steps) * res, feed=1000)
        return best > 1 and best < 2 * steps - 1

    def loop(self, count):
        self.radii = (self.window.spInnerRadius.value(), self.window.spOuterRadius.value())
        self.window.progress.setMaximum(2 * 5 + 1 + 2 * 8 + 1)
        self.window.progress.setValue(0)
        if self._aquire(5, 0.1, 2):
            self._aquire(8, 0.01, 5)
        else:
            msg = QMessageBox()
            msg.setText("Focus seems to be off. Fine aquisition is skipped")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()


class Repeatability(Job):

    def _exit(self, result):
        super(Repeatability, self)._exit()
        self.D.finished.disconnect()

    def init(self):
        D = self.dialog('QTJobInit.ui')
        model = QStandardItemModel(D.lvTools)
        for tool in self.window.tools:
            item = QStandardItem('Tool {0}'.format(tool.nr))
            item.setCheckable(True)
            item.setCheckState(True)
            model.appendRow(item)
        D.lvTools.setModel(model)

        if D.exec_() == QDialog.Accepted:
            size = self.window.sbCrop.value()
            fourcc = cv2.VideoWriter.fourcc(*'XVID')
            self.video = cv2.VideoWriter('output.avi', fourcc, 25, (size, size))
            self.last = None

            self.cycles = D.sbCycles.value()

            self.tools = []
            for n in range(0, len(self.window.tools)):
                if model.item(n, 0).checkState():
                    self.tools.append(self.window.tools[n])
            self.offsets = np.zeros((2, len(self.window.tools), self.cycles), dtype=float)

            self.window.sbTool.setValue(self.tools[0].nr)
            self.window._inspectTool()
            self.printer.moveRel(X=-0.2, Y=-0.2, wait=True)
            self.block(count=5)
            O1 = self.window.offset
            self.printer.moveRel(X=0.4, Y=0.4, wait=True)
            self.block(count=5)
            distpx = np.linalg.norm(self.window.offset - O1)
            dist = np.linalg.norm((400, 400))
            self.resolution = dist / distpx
            print(dist, distpx, self.resolution)

            self.D = self.dialog('QTJobOutput.ui')
            self.D.button.hide()
            self.D.setWindowModality(Qt.NonModal)
            self.figures = []
            self.figures.append(FigureCanvas(Figure()))
            self.xyplot = self.figures[-1].figure.add_subplot(111, aspect='equal')
            self.D.graph.addWidget(self.figures[-1], 0, 0, 2, 1)

            self.figures.append(FigureCanvas(Figure()))
            self.xplot = self.figures[-1].figure.add_subplot(111)
            self.D.graph.addWidget(self.figures[-1], 0, 1)

            self.figures.append(FigureCanvas(Figure()))
            self.yplot = self.figures[-1].figure.add_subplot(111)
            self.D.graph.addWidget(self.figures[-1], 1, 1)
            self.D.finished.connect(self._exit)

            self.D.resize(self.window.size())
            self.D.show()
            self.D.move(self.window.pos())

            return True

        return False

    def loop(self, count):
        for tool in self.tools:
            self.window.sbTool.setValue(tool.nr)
            self.window._inspectTool()
            self.block(count=5)
            self.offsets[0, tool.nr, count] = self.window.offset[0] * self.resolution
            self.offsets[1, tool.nr, count] = self.window.offset[1] * self.resolution

            self.video.write(self.window.image.image)

        self.window._unloadTool()
        if count > 0:
            self._plot(count)
        return True

    def _plotSingleAxis(self, ax, axis, data, limit):
        ax.cla()

        cc = ax._get_lines.color_cycle
        for tool in self.tools:
            t = np.arange(0, data.shape[2])
            fit = np.polyfit(t, data[axis, tool.nr], 1)
            poly = np.poly1d(fit)
            color = next(cc)
            ax.plot(t, data[axis, tool.nr], '{0}.'.format(color))
            ax.plot(t, poly(t), '{0}-'.format(color))
        ax.set_ylim([-limit, limit])
        ax.set_ylabel('offset [ux]')
        ax.set_xlabel('cycle')

    def _plot(self, count):
        data = self.offsets[:, :, 0:count + 1]
        limit = max(data.max(), -data.min())
        self._plotSingleAxis(self.xplot, 0, data, limit)
        self.xplot.set_title('X offset per tool')
        self._plotSingleAxis(self.yplot, 1, data, limit)
        self.yplot.set_title('Y offset per tool')

        self.xyplot.cla()

        legend = []
        for tool in self.tools:
            nr = tool.nr
            self.xyplot.plot(data[0][nr], data[1][nr], 'o', label='T{0}'.format(nr))
            legend.append('Tool {0}'.format(nr))

        handles, labels = self.xyplot.get_legend_handles_labels()
        self.xyplot.legend(handles, legend)
        self.xyplot.set_xlim([-limit, limit])
        self.xyplot.set_ylim([-limit, limit])
        self.xyplot.set_title('X/Y offset per tool')
        self.xyplot.set_xlabel('X offset [ux]')
        self.xyplot.set_ylabel('Y offset [ux]')

        for figure in self.figures:
            figure.draw()

    def summary(self):
        self.video.release()
        self.D.finished.disconnect()

        text = ''
        for tool in self.tools:
            axises = (('X', 0), ('Y', 1))
            text += u'Tool {0}:\t'.format(tool.nr)
            for axis in axises:
                data = self.offsets[axis[1], tool.nr]
                fit = np.polyfit(np.arange(0, self.cycles), data, 1)
                text += u'{0}: mean:{1:>5.1f} µm  std:{2:>5.1f} µm  trend:{3:>6.3f} µm/cycle  max:{4:>5.1f} µm\t' \
                    .format(axis[0], data.mean(), data.std(), fit[0], max(data.max(), -data.min()))
            text += '\n'
        self.D.text.setText(text)
