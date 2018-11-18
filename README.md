# Nozzle Offset Calibration Tool
The purpose of this tool is to reduce the time needed to calibrate Nozzle offset for 3D printers that have
multiple extruders and/or tool changing capabilities, by using a modified webcam as inspection microscope.

It utilizes [OctoPrint](https://octoprint.org/) to communicate with the printer and camera.

# Current State
I consider the tool be in a serious **alpha** state, not ready for the masses, contributions (documentation,
bug reports, patches) are welcome. I released the code mostly as a “technology demonstration”, and I’m happy
to support anyone who wants to take this idea and integrate it into eg. OctoPrint as a plugin (I tried but
failed, too steep of a learning curve for me :)) MIT License was specifically chosen to have as little as
possible legal hurdles with taking code/ideas out of this project.

# How it works
An upwards facing camera/microscope, mounted to the bed of the printer is used to get an high magnification
(~5µm/pixel) image of a nozzle and moved to the center of the image using the XY carriage of the printer, thus
the XY offset between all nozzles/tools can be determined from the carriage position. Z offset is determined
by maximizing the contrast between two rings.

This [video](https://www.youtube.com/watch?v=g1wAQ0f_Whs&t=80s) shows the whole process in detail.

Further details about how it evolved and how the camera is built can be found here:

https://hackaday.io/project/26053-tool-switching-multi-extrusion/log/67725-update-cameras-and-calibration-tool

https://hackaday.io/project/26053-tool-switching-multi-extrusion/log/66799-wip-offset-calibration-tool

https://hackaday.io/project/26053-tool-switching-multi-extrusion/log/66200-wip-offset-calibration-camera

https://hackaday.io/project/26053-tool-switching-multi-extrusion/log/64076-nozzle-offset-calibration

# Setup
The very basic setup step (OctoPrint host/port/api-key) should be self explaining.

Setting up the tool changer is a bit more tricky, and might involve some knowledge of python. In short,
*parameters* is evaluated using *eval* and must return a dictionary, which is than used to format *Tool load*
and *Tool unload* gCode sequences. As OctoPrint provides no way to determine the printer position.
*Load end position* and *Unload end position* need to be given instead.

If this sounds complicated, it is, but on the other hand imposes gives a lot of freedom to calculate/specify
the commands needed to load/unload tools.

Example:

**parameters:**
```
{
  'x_enter': 157 + tool * 60,
  'x': 150 + tool * 60,
  'x_leave': 143 + tool * 60,
  'y_safe': 250,
  'y_fast': 288,
  'y': 290,
  'fast': 15000,
  'slow': 1000
}
```
Note: *tool* is a variable and indicates the number of the tool to be loaded/unloaded

**unloadCmds**:
```
G0 X{x} Y{y_safe} F{fast}
G0 Y{y}
G0 X{x_leave} F{slow}
G0 Y{y_safe} F{fast}
```

**unloadEndPos**:
```
(x, y_safe, Z)
```
Note: this is a python tuple, the values need to match the position after the previous gCode is executed.
X, Y and Z are injected variables, refer to the printer position from before. In the given example *unload*
doesn't move the printer in Z.

**loadCmds**:
```
G0 X{x} Y{y_safe} F{fast}
G0 Y{y}
G0 X{x_leave} F{slow}
G0 Y{y_safe} F{fast}
```

**loadEndPos**:
```
(x_leave, y_safe, Z)
```

# Dependencies
The tool is written in Python and requires the following dependencies
* sudo pip3 install opencv-python-headless
* sudo apt-get install python3-pyqt4
* sudo apt-get install python3-yaml
* sudo apt-get install python3-matplotlib

# License
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
