# William Rodriguez
"""
**********Training and sample programs are heavily based on tensorflow-char-rnn by crazydonkey200**********
Link: https://github.com/crazydonkey200/tensorflow-char-rnn

Description:
    A program intended to help organize the construction of projects that require numerous
    neural networks. This is done by assigning each neural network to a 'Project' which saves all of the variables and file paths
    necessary to recreate it. This means that starting, stopping, saving, loading, and cloning your neural networks is much easier.

    HOWEVER: this program was only intended to be ran on my setup for a very long time, and therefore may require a lot of work for it
    to be ran on different setups.

    My Setup:

        A NVidia graphics card (GTX 1080 ti)
        An Intel CPU (8700k)
        tensorflow-gpu installed as the default version of tensorflow
        A Python3 environment named cpu_env placed in this project's directory
                This environment has the tensorflow version for CPU
                I use the GPU version to train the AI and the CPU version to generate samples while it's training

        **Although the particular hardware should not effect whether or not this program runs, the brands and software setup
        will definitely effect how everything works**
"""

import sys
import functools
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import glob
import json
import os
import shutil
import numpy as np
import threading
import ntpath
import train
import io
from subprocess import Popen

import char_rnn_model

class NNViewer(QWidget):

    def __init__(self):
        super().__init__()

        self.run_thread = None
        self.sample_process = None
        self.initialized = False
        self.loadDictionary()
        self.loadSettings()
        self.loadErrorMessages()
        self.initUI()

    def check(self):
        if not os.path.isdir(self.settings['Output Directory']):
            self.noOutputDirectory.show()
            return False

        fullPath = os.path.join(self.settings['Output Directory'], self.settings['Project Name'])

        if not os.path.exists(self.settings['Data File']):
            self.dataFileNotValid.show()
            return False

        return True

    def checkSampleDependencies(self):
        fullPath = os.path.join(self.settings['Output Directory'], self.settings['Project Name'])
        resultPath = os.path.join(fullPath, 'result.json')

        if os.path.exists(os.path.join(fullPath, resultPath)):
            with open(resultPath, 'r') as f:
                json_data = json.loads(f.read())

                if 'latest_model' not in json_data.keys() or 'best_model' not in json_data.keys():
                    self.missingModel.show()

                latest_path = os.path.join(fullPath, 'saved_models', json_data['latest_model'] + '.index')
                best_path = os.path.join(fullPath, 'best_model', json_data['best_model'] + '.index')
                if not os.path.exists(latest_path):
                    self.missingModel.show()
                    return False
                if not os.path.exists(best_path):
                    self.missingModel.show()
                    return False
        else:
            self.missingModel.show()
            return False

        if not os.path.exists(self.settings['Output File']):
            self.missingOutputFile.show()
            return False

        return True

    def save(self, copy=True):
        fullPath = os.path.join(self.settings['Output Directory'], self.settings['Project Name'])

        if not os.path.isdir(fullPath):
            os.mkdir(fullPath)

        self.originalInitDir = self.settings['Initialize Directory']
        if copy and self.settings['Initialize Directory'] != '' and fullPath != self.settings['Initialize Directory']:
            items = glob.glob(os.path.join(self.settings['Initialize Directory'], '*'))

            targetDirectoryItems = glob.glob(os.path.join(fullPath, '*'))
            if len(targetDirectoryItems) > 0:
                reply = QMessageBox.question(self, 'The project directory is not empty', 'Would you like to override this folder?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

                if reply == QMessageBox.No:
                    return False

                for t in targetDirectoryItems:
                    if os.path.isfile(t):
                        os.remove(t)
                    else:
                        shutil.rmtree(t)

            for i in items:
                head, tail = ntpath.split(i)
                itemPath = os.path.join(fullPath, tail)

                if os.path.isfile(i):
                    shutil.copy(i, itemPath)
                else:
                    shutil.copytree(i, itemPath)
            self.setSetting('Initialize Directory', fullPath)

        with open(os.path.join(fullPath, 'save.json'), 'w+') as outfile:
            r = json.dump(self.settings, outfile)

        self.edited = False
        self.updateTitle()

        return True

    def loadErrorMessages(self):
        self.invalidDirectory = QMessageBox()
        self.invalidDirectory.setIcon(QMessageBox.Critical)
        self.invalidDirectory.setText("No project at directory")
        self.invalidDirectory.setWindowTitle("Invalid Directory")

        self.noOutputDirectory = QMessageBox()
        self.noOutputDirectory.setIcon(QMessageBox.Critical)
        self.noOutputDirectory.setText("Output directory does not exist")
        self.noOutputDirectory.setWindowTitle("Please choose an output directory")

        self.dataFileNotValid = QMessageBox()
        self.dataFileNotValid.setIcon(QMessageBox.Critical)
        self.dataFileNotValid.setText("Data file does not exist")
        self.dataFileNotValid.setWindowTitle("Please choose a data file")

        self.invalidDataType = QMessageBox()
        self.invalidDataType.setIcon(QMessageBox.Critical)
        self.invalidDataType.setText("One or more of the fields contain invalid values")
        self.invalidDataType.setWindowTitle("Please check your values")

        self.missingModel = QMessageBox()
        self.missingModel.setIcon(QMessageBox.Critical)
        self.missingModel.setText("The current directory does not have a pretrained model")
        self.missingModel.setWindowTitle("Maybe change project directories or train a model")

        self.missingOutputFile = QMessageBox()
        self.missingOutputFile.setIcon(QMessageBox.Critical)
        self.missingOutputFile.setText("The Output File does not exist")
        self.missingOutputFile.setWindowTitle("Please choose an output file")

    def newProject(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select New Project Directory"))
        if not folder:
            return

        self.textDictionary['Output Directory'].setText(folder)
        self.updateTitle()

    def openProject(self):
        if self.edited:
            reply = QMessageBox.question(self, 'Continue from Project?', 'All unsaved changes will be lost', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.No:
                return

        folder = str(QFileDialog.getExistingDirectory(self, "Initialize Project Directory"))
        if not folder: # Prompt was cancelled
            return

        files = glob.glob(folder + '/save.json')
        if len(files) == 1:
            f = open(files[0])

            json_data = json.loads(f.read())
            for name, value in json_data.items():
                self.setSetting(name, value)
        else:
            self.invalidDirectory.show()
            return

        self.edited = False
        self.updateTitle()
        self.initialized = self.settings['Initialize Directory'] != ''

        if self.initialized:
            for name in self.uneditableIfInitialized:
                self.textDictionary[name].setDisabled(True)

    def initialize(self):

        folder = str(QFileDialog.getExistingDirectory(self, "Initialize Project Directory"))
        if not folder: # Prompt was cancelled
            return

        fullPath = os.path.join(self.settings['Output Directory'], self.settings['Project Name'])
        if fullPath == folder:
            for name in self.uneditableIfInitialized:
                self.textDictionary[name].setDisabled(True)

            self.setSetting('Initialize Directory', folder)
            self.initialized = True
            return

        files = glob.glob(folder + '/save.json')
        if len(files) == 1:
            f = open(files[0])

            reply = QMessageBox.question(self, 'Loading settings', 'Load all settings from the initialized project?', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

            if reply == QMessageBox.No:
                json_data = json.loads(f.read())
                for name, value in json_data.items():
                    if name in self.uneditableIfInitialized:
                        if name != 'Project Name' and name != 'Output Directory':
                            self.setSetting(name, value)
            else:
                json_data = json.loads(f.read())
                for name, value in json_data.items():
                    if name != 'Project Name' and name != 'Output Directory':
                        self.setSetting(name, value)
        else:
            self.invalidDirectory.show()
            return

        for name in self.uneditableIfInitialized:
            self.textDictionary[name].setDisabled(True)

        self.setSetting('Initialize Directory', folder)
        self.initialized = True

    def deinitialize(self):
        for name in self.uneditableIfInitialized:
            self.textDictionary[name].setDisabled(False)

        self.setSetting('Initialize Directory', '')
        self.initialized = False

    def cast_gr(self, value):
        value = '0' + value
        try:
            value = int(value)
            if value > 0:
                return value
            return None
        except:
            return None

    def cast_geq(self, value):
        value = '0' + value
        try:
            value = int(value)
            if value >= 0:
                return value
            return None
        except:
            return None

    def cast_gr_float(self, value):
        value = '0' + value
        try:
            value = float(value)
            if value > 0:
                return value
            return None
        except:
            return None

    def cast_geq_float(self, value):
        value = '0' + value
        try:
            value = float(value)
            if value >= 0:
                return value
            return None
        except:
            return None

    def cast_bool(self, value):
        return value.lower() == 'true'

    def loadSettings(self):
        self.settings = {
            'Project Name' : 'Untitled',
            'Initialize Directory' : '',
            'Data File' : '',
            'Output Directory' : '',
            'Save Frequency' : 1,
            'Hidden Size' : 128,
            'Number of Layers' : 3,
            'Number of Unrollings' : 10,
            'Model' : 'lstm',
            'Number of Epochs' : 100,
            'Batch Size' : 256,
            'Train Fraction' : 0.9,
            'Valid Fraction' : 0.05,
            'Dropout' : 0.01,
            'Input Dropout' : 0.01,
            'Current Learning Rate' : 0.002,
            'Final Learning Rate' : 2e-10,
            'Length' : -1,

            'Output File' : '',
            'Sample Length' : 1000,
            'Temperature' : 1,
            'Start Text' : '',
            'Seed' : -1,
            'Evaluate' : False,
            'Latest' : False
        }

        self.uneditable = [
            'Output Directory',
            'Initialize Directory',
            'Output File'
        ]

        self.uneditableIfInitialized = [
            'Hidden Size',
            'Number of Layers',
            'Model'
        ]

        self.trainSettings = {
            'Project Name',
            'Initialize Directory',
            'Data File',
            'Output Directory',
            'Save Frequency',
            'Hidden Size',
            'Number of Layers',
            'Number of Unrollings',
            'Model',
            'Number of Epochs',
            'Batch Size',
            'Train Fraction',
            'Valid Fraction',
            'Dropout',
            'Input Dropout',
            'Current Learning Rate',
            'Final Learning Rate',
            'Length'
        }

        self.sampleSettings = {
            'Temperature',
            'Start Text',
            'Sample Length',
            'Seed',
            'Evaluate',
            'Output File',
            'Latest'
        }

        self.data_types = {
            str : {
                'Project Name', 'Initialize Directory', 'Data File', 'Output Directory',
                'Model', 'Start Text', 'Output File'
            },

            self.cast_gr : {
                'Save Frequency', 'Hidden Size', 'Number of Layers',
                'Number of Epochs', 'Batch Size', 'Sample Length'
            },

            self.cast_geq : {
                'Number of Unrollings'
            },

            int : {
                'Length', 'Seed'
            },

            self.cast_geq_float : {
                'Train Fraction', 'Valid Fraction'
            },

            self.cast_gr_float : {
                'Dropout', 'Input Dropout', 'Current Learning Rate',
                'Final Learning Rate', 'Temperature'
            },

            self.cast_bool : {
                'Evaluate', 'Latest'
            }
        }

    def loadDefaults(self):
        reply = QMessageBox.question(self, 'Set Defaults', 'Set all settings to default?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.No:
            return

        self.loadSettings()
        for name, default in self.settings.items():
            self.textDictionary[name].setText(default)

        self.initialized = False
        self.enable()

    def openData(self):
        f = QFileDialog.getOpenFileName(self, "Choose Data File", "", "Text Files (*.txt)")
        if f:
            self.setSetting('Data File', str(f[0]))

    def disable(self):
        for name, txt in self.textDictionary.items():
            if name in self.trainSettings:
                txt.setDisabled(True)

        for name, btn in self.btnDictionary.items():
            btn.setDisabled(True)

        for name, btn in self.logbtn.items():
            btn.setDisabled(False)

    def enable(self):
        for name, txt in self.textDictionary.items():
            if name in self.trainSettings and name not in self.uneditable:
                if self.initialized:
                    if name not in self.uneditableIfInitialized:
                        txt.setDisabled(False)
                else:
                    txt.setDisabled(False)

        for name, btn in self.btnDictionary.items():
            btn.setDisabled(False)

        for name, btn in self.logbtn.items():
            btn.setDisabled(True)

    def sampleDisable(self):
        for name, txt in self.textDictionary.items():
            if name in self.sampleSettings:
                txt.setDisabled(True)

        for name, btn in self.samplebtn.items():
            if name != 'Stop Sample':
                btn.setDisabled(True)

        self.samplebtn['Stop Sample'].setDisabled(False)

    def sampleEnable(self):
        for name, txt in self.textDictionary.items():
            if name in self.sampleSettings and name not in self.uneditable:
                txt.setDisabled(False)

        for name, btn in self.samplebtn.items():
            btn.setDisabled(False)

        self.samplebtn['Stop Sample'].setDisabled(True)

    def run(self):
        returnEarly = False
        for name, txt in self.textDictionary.items():
            if name not in self.trainSettings:
                continue

            if self.validateType(txt, name) is None:
                returnEarly = True
                txt.setPalette(self.red)

        if returnEarly:
            self.invalidDataType.show()
            return

        if not self.check():
            return

        if not self.save():
            return

        self.disable()

        self.run_thread = threading.Thread(target=train.train, args=(self.settings,))
        self.run_thread.daemon = True
        self.run_thread.start()

        QTimer.singleShot(1000, self.update)

    def callSample(self):
        command = []
        python_version = 'python3'
        if self.run_thread is not None and self.run_thread.isAlive():
            dir_path = os.path.dirname(os.path.realpath(__file__))
            python_version = os.path.join(dir_path, 'cpu_env/bin/python3')
            if not os.path.exists(python_version):
                """
                This addition was pretty last minute since I was only expecting to use this program myself.
                The entire process of generating and training AI on the GPU and CPU will have to be changed
                if this program is intended to run on varied setups.
                """
                print("Cannot generate a sample on the CPU while a NN is training on the GPU")
                return

        fullPath = os.path.join(self.settings['Output Directory'], self.settings['Project Name'])

        command.append(python_version)
        command.append('sample.py')
        command.append('--init_dir')
        command.append(str(fullPath))
        command.append('--length')
        command.append(str(self.settings['Sample Length']))
        command.append('--output_file')
        command.append(str(self.settings['Output File']))
        if str(self.settings['Start Text']) != '':
            command.append('--start_text')
            command.append(str(self.settings['Start Text']))
        command.append('--temperature')
        command.append(str(self.settings['Temperature']))
        command.append('--seed')
        command.append(str(self.settings['Seed']))
        if self.settings['Evaluate']:
            command.append('--evaluate')

        print(' '.join(command))
        self.sample_process = Popen(command)

    def generateSample(self):
        returnEarly = False
        for name, txt in self.textDictionary.items():
            if name not in self.sampleSettings:
                continue

            if self.validateType(txt, name) is None:
                returnEarly = True
                txt.setPalette(self.red)

        if returnEarly:
            self.invalidDataType.show()
            return

        if not self.checkSampleDependencies():
            return

        if not self.save(copy=False):
            return

        self.sampleDisable()
        self.callSample()

        QTimer.singleShot(1000, self.sampleUpdate)

    def stopSample(self):
        if self.sample_process is not None:
            self.sample_process.kill()
        self.sampleEnable()

    def update(self):
        if self.run_thread.isAlive():
            if train.file_read:
                self.setSetting('Initialize Directory', self.originalInitDir)
                self.save(copy = False)
                train.file_read = False

            if train.learning_rate > 0:
                if self.settings['Current Learning Rate'] != train.learning_rate:
                    self.setSetting('Current Learning Rate', float(train.learning_rate))
                    self.save(copy = False)

            QTimer.singleShot(1000, self.update)
        else:
            if char_rnn_model.error != '':
                reply = QMessageBox.question(self, 'ERROR', char_rnn_model.error, QMessageBox.Ok, QMessageBox.Ok)
                char_rnn_model.error = ''

            char_rnn_model.stop = False
            self.enable()

    def sampleUpdate(self):
        if self.sample_process.poll() is None: # is running
            QTimer.singleShot(1000, self.sampleUpdate)
        else:
            self.sampleEnable()

    def outputFile(self):
        f = QFileDialog.getSaveFileName(self,"Output Sample As","","Text Files (*.txt)")
        if f:
            self.setSetting('Output File', str(f[0]))

    def loadDictionary(self):
        self.trainCommands = {
            'Choose Output Directory' : self.newProject,
            'Choose Data File' : self.openData,
            'Load Defaults' : self.loadDefaults,
            'Open Project' : self.openProject,
            'Initialize Project' : self.initialize,
            'Deinitialize Project' : self.deinitialize,
            'Run' : self.run,
        }

        self.logCommands = {
            'Stop' : self.stopThread,
        }

        self.sampleCommands = {
            'Choose Output File' : self.outputFile,
            'Generate' : self.generateSample,
            'Stop Sample' : self.stopSample
        }

    def validateType(self, txt, name):
        dtype = None
        for d, names in self.data_types.items():
            if name in names:
                dtype = d
                break

        value = dtype(txt.text())

        if value is not None:
            self.settings[name] = dtype(txt.text())
        return value

    def updateSettings(self, txt, name):
        self.edited = True
        txt.setPalette(self.normal)

        t = txt.text()
        if t[:7] == 'file://':
            t = t[7:]
            txt.setText(t)

        self.setSetting(name, t)

    def setSetting(self, name, value):
        if self.settings[name] != value:
            self.textDictionary[name].setText(str(value))
            self.settings[name] = value
            self.edited = True
            self.updateTitle()

    def updateTitle(self):
        title = self.settings['Project Name']
        if self.edited:
            self.setWindowTitle(title + '*')
        else:
            self.setWindowTitle(title)

    def closeEvent(self, event):
        accept = True
        if self.edited:
            reply = QMessageBox.question(self, 'You have unsaved changes', 'Are you sure you want to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.No:
                accept = False

        if self.run_thread is not None and self.run_thread.isAlive():
            reply = QMessageBox.question(self, 'A network is currently training', 'Are you sure you want to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.No:
                accept = False
            else:
                if train.file_read == False:
                    self.setSetting('Initialize Directory', self.originalInitDir)
                    self.save(copy = False)

        if self.sample_process is not None and self.sample_process.poll() is None:
            reply = QMessageBox.question(self, 'A sample is currently being generated', 'Are you sure you want to quit?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.No:
                accept = False
            else:
                self.stopSample()

        if accept:
            event.accept()
        else:
            event.ignore()

    def stopThread(self):
        if self.run_thread.isAlive():
            char_rnn_model.stop = True

    def initUI(self):
        self.setWindowTitle(self.settings['Project Name'])
        self.setGeometry(300, 300, 1000, 150)

        QToolTip.setFont(QFont('SansSerif', 10))

        self.red = QPalette()
        self.red.setColor(QPalette.Text, Qt.red)
        self.normal = QPalette()

        self.layout = QVBoxLayout(self)

        tabs = QTabWidget()
        self.layout.addWidget(tabs)

        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()
        tabs.addTab(tab1, 'Settings')
        tabs.addTab(tab2, 'Progress Viewer')
        tabs.addTab(tab3, 'Generate Sample')

        # ===============================TAB 3===================================
        vertLayout = QVBoxLayout()
        tab3.setLayout(vertLayout)

        horLayout = QHBoxLayout()
        vertLayout.addLayout(horLayout)

        self.samplebtn = {}
        for name, command in self.sampleCommands.items():
            btn = QPushButton(name)
            self.samplebtn[name] = btn
            horLayout.addWidget(btn)

            btn.clicked.connect(command)

        grid = QGridLayout()
        vertLayout.addLayout(grid)

        names = [n for n in self.settings.keys() if n in self.sampleSettings]
        positions = [(i,j) for i in range(1, 7) for j in range(0, 6, 2)]

        self.textDictionary = {}
        for position, name in zip(positions, names):
            lbl = QLabel(name + ':')
            txt = QLineEdit()
            txt.setText(str(self.settings[name]))
            self.textDictionary[name] = txt

            txt.textEdited.connect(functools.partial(self.updateSettings, txt, name))

            grid.addWidget(lbl, *position)
            grid.addWidget(txt, position[0], position[1] + 1)

        self.sampleEnable()

        # ===============================TAB 2===================================
        vertLayout = QVBoxLayout()
        tab2.setLayout(vertLayout)

        horLayout = QHBoxLayout()
        vertLayout.addLayout(horLayout)

        self.logbtn = {}
        for name, command in self.logCommands.items():
            btn = QPushButton(name)
            btn.setDisabled(True)
            self.logbtn[name] = btn
            horLayout.addWidget(btn)

            btn.clicked.connect(command)

        # ===============================TAB 1===================================
        vertLayout = QVBoxLayout()
        tab1.setLayout(vertLayout)

        horLayout = QHBoxLayout()
        vertLayout.addLayout(horLayout)

        self.btnDictionary = {}
        for name, command in self.trainCommands.items():
            b = QPushButton(name)
            self.btnDictionary[name] = b

            b.clicked.connect(command)
            horLayout.addWidget(b)

        grid = QGridLayout()
        vertLayout.addLayout(grid)

        names = [n for n in self.settings.keys() if n in self.trainSettings]
        positions = [(i,j) for i in range(1, 7) for j in range(0, 6, 2)]

        for position, name in zip(positions, names):
            lbl = QLabel(name + ':')
            txt = QLineEdit()
            txt.setText(str(self.settings[name]))
            self.textDictionary[name] = txt

            txt.textEdited.connect(functools.partial(self.updateSettings, txt, name))
            grid.addWidget(lbl, *position)
            grid.addWidget(txt, position[0], position[1] + 1)

        # Make certain boxes uneditable
        for u in self.uneditable:
            self.textDictionary[u].setDisabled(True)

        self.edited = False
        self.updateTitle()
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = NNViewer()
    sys.exit(app.exec_())
