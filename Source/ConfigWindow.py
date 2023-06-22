
import os
import shutil, glob
from PyQt5 import QtCore, QtGui, QtWidgets
from pathlib import Path

from Source.MainConfig import MainConfig
from Source.ConfigFile import ConfigFile
from Source.CalibrationFileReader import CalibrationFileReader
from Source.AnomalyDetection import AnomAnalWindow
from Source.SeaBASSHeader import SeaBASSHeader
from Source.SeaBASSHeaderWindow import SeaBASSHeaderWindow
from Source.GetAnc import GetAnc
from Source.OCproductsWindow import OCproductsWindow


class ConfigWindow(QtWidgets.QDialog):
    ''' Configuration window object '''
    def __init__(self, name, inputDir, parent=None):
        super().__init__(parent)
        # self.setStyleSheet("background-color: #e3e6e1;")
        self.setModal(True)
        self.name = name
        self.inputDirectory = inputDir
        self.initUI()


    def initUI(self):
        ''' Initialize the GUIs '''

        intValidator = QtGui.QIntValidator()
        doubleValidator = QtGui.QDoubleValidator()
        # oddValidator = QtGui.QRegExpValidator(rx,self)

        # sensor type
        sensorTypeLabel = QtWidgets.QLabel("Sensor Type:", self)
        self.sensorTypeComboBox = QtWidgets.QComboBox(self)
        self.sensorTypeComboBox.addItems(["Choose a sensor ...", "SeaBird", "TriOS"])
        CurrentSensor = ConfigFile.settings["SensorType"]
        index = self.sensorTypeComboBox.findText(CurrentSensor,QtCore.Qt.MatchFixedString)
        self.sensorTypeComboBox.setCurrentIndex(index)
        self.sensorTypeComboBox.setEnabled(True)
        self.sensorTypeComboBox.currentIndexChanged.connect(self.sensorTypeChanged)
        # self.setSensorSettings()
        # Calibration Config Settings
        self.addCalibrationFileButton = QtWidgets.QPushButton("Add Cals")
        self.addCalibrationFileButton.clicked.connect(self.addCalibrationFileButtonPressed)
        self.deleteCalibrationFileButton = QtWidgets.QPushButton("Remove Cals")
        self.deleteCalibrationFileButton.clicked.connect(self.deleteCalibrationFileButtonPressed)

        calFiles = ConfigFile.settings["CalibrationFiles"]
        print("Calibration Files:")
        self.calibrationFileComboBox = QtWidgets.QComboBox(self)
        for file in calFiles:
            print(file)
        self.calibrationFileComboBox.addItems(sorted(calFiles.keys()))
        fsm = QtWidgets.QFileSystemModel()
        fsm.setNameFilters(["*.cal", "*.tdf", "*.ini", ".dat"])
        fsm.setNameFilterDisables(False)
        fsm.setFilter(QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Files)
        calibrationDir = os.path.splitext(self.name)[0] + "_Calibration"
        configPath = os.path.join("Config", calibrationDir)
        index = fsm.setRootPath(configPath)
        self.calibrationFileComboBox.setModel(fsm)
        self.calibrationFileComboBox.setRootModelIndex(index)
        self.calibrationFileComboBox.currentIndexChanged.connect(self.calibrationFileChanged)

        # Config File Settings
        self.calibrationEnabledCheckBox = QtWidgets.QCheckBox("Enabled", self)
        self.calibrationEnabledCheckBox.stateChanged.connect(self.calibrationEnabledStateChanged)
        self.calibrationEnabledCheckBox.setEnabled(False)

        # CurrentSensor = ConfigFile.settings["SensorType"]
        if CurrentSensor.lower() == "seabird":
            calibrationFrameTypeLabel = QtWidgets.QLabel("Frame Type:", self)
            self.calibrationFrameTypeComboBox = QtWidgets.QComboBox(self)
            self.calibrationFrameTypeComboBox.addItem("ShutterLight")
            self.calibrationFrameTypeComboBox.addItem("ShutterDark")
            self.calibrationFrameTypeComboBox.addItem("Not Required")
            # self.calibrationFrameTypeComboBox.addItem("LightAncCombined")
            self.calibrationFrameTypeComboBox.currentIndexChanged.connect(self.calibrationFrameTypeChanged)
            self.calibrationFrameTypeComboBox.setEnabled(False)

        elif CurrentSensor.lower() == "trios":
            calibrationFrameTypeLabel = QtWidgets.QLabel("Frame Type:", self)
            self.calibrationFrameTypeComboBox = QtWidgets.QComboBox(self)
            self.calibrationFrameTypeComboBox.addItem("LI")
            self.calibrationFrameTypeComboBox.addItem("LT")
            self.calibrationFrameTypeComboBox.addItem("ES")
            self.calibrationFrameTypeComboBox.addItem("Not Required")
            self.calibrationFrameTypeComboBox.currentIndexChanged.connect(self.calibrationFrameTypeChanged)
            self.calibrationFrameTypeComboBox.setEnabled(False)

        # L1A
        l1aLabel = QtWidgets.QLabel("Level 1A Processing", self)
        l1aLabel_font = l1aLabel.font()
        l1aLabel_font.setPointSize(12)
        l1aLabel_font.setBold(True)
        l1aLabel.setFont(l1aLabel_font)
        l1aSublabel = QtWidgets.QLabel(" Raw binary to HDF5", self)

        self.l1aUTCOffsetLabel = QtWidgets.QLabel("     Raw UTC Offset [+/-]", self)
        self.l1aUTCOffsetLineEdit = QtWidgets.QLineEdit(self)
        self.l1aUTCOffsetLineEdit.setText(str(ConfigFile.settings["fL1aUTCOffset"]))
        self.l1aUTCOffsetLineEdit.setValidator(doubleValidator)

        l1aCleanSZALabel = QtWidgets.QLabel("     Solar Zenith Angle Filter", self)
        self.l1aCleanSZACheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL1aCleanSZA"]) == 1:
            self.l1aCleanSZACheckBox.setChecked(True)
        self.l1aCleanSZAMaxLabel = QtWidgets.QLabel("     SZA Max", self)
        self.l1aCleanSZAMaxLineEdit = QtWidgets.QLineEdit(self)
        self.l1aCleanSZAMaxLineEdit.setText(str(ConfigFile.settings["fL1aCleanSZAMax"]))
        self.l1aCleanSZAMaxLineEdit.setValidator(doubleValidator)

        self.l1aCleanSZACheckBoxUpdate()
        self.l1aCleanSZACheckBox.clicked.connect(self.l1aCleanSZACheckBoxUpdate)

        # L1AQC
        l1aqcLabel = QtWidgets.QLabel("Level 1AQC Processing", self)
        l1aqcLabel.setFont(l1aLabel_font)
        l1aqcSublabel = QtWidgets.QLabel(" Filter on pitch, roll, yaw, and azimuth", self)

        #   SolarTracker
        self.l1aqcSolarTrackerLabel = QtWidgets.QLabel(" SolarTracker or pySAS", self)
        self.l1aqcSolarTrackerCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL1aqcSolarTracker"]) == 1:
            self.l1aqcSolarTrackerCheckBox.setChecked(True)

        #   Rotator
        self.l1aqcRotatorHomeAngleLabel = QtWidgets.QLabel(" Rotator Home Angle Offset", self)
        self.l1aqcRotatorHomeAngleLineEdit = QtWidgets.QLineEdit(self)
        self.l1aqcRotatorHomeAngleLineEdit.setText(str(ConfigFile.settings["fL1aqcRotatorHomeAngle"]))
        self.l1aqcRotatorHomeAngleLineEdit.setValidator(doubleValidator)

        self.l1aqcRotatorDelayLabel = QtWidgets.QLabel(" Rotator Delay (Seconds)", self)
        self.l1aqcRotatorDelayCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL1aqcRotatorDelay"]) == 1:
            self.l1aqcRotatorDelayCheckBox.setChecked(True)
        self.l1aqcRotatorDelayLineEdit = QtWidgets.QLineEdit(self)
        self.l1aqcRotatorDelayLineEdit.setText(str(ConfigFile.settings["fL1aqcRotatorDelay"]))
        self.l1aqcRotatorDelayLineEdit.setValidator(doubleValidator)
        self.l1aqcRotatorDelayCheckBoxUpdate()

        #   Pitch and Roll
        self.l1aqcCleanPitchRollLabel = QtWidgets.QLabel(" Pitch/Roll Filter (where present)", self)
        self.l1aqcCleanPitchRollCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL1aqcCleanPitchRoll"]) == 1:
            self.l1aqcCleanPitchRollCheckBox.setChecked(True)
        self.l1aqcPitchRollPitchLabel = QtWidgets.QLabel("       Max Pitch/Roll Angle", self)
        self.l1aqcPitchRollPitchLineEdit = QtWidgets.QLineEdit(self)
        self.l1aqcPitchRollPitchLineEdit.setText(str(ConfigFile.settings["fL1aqcPitchRollPitch"]))
        self.l1aqcPitchRollPitchLineEdit.setValidator(doubleValidator)

         #  Rotator
        self.l1aqcRotatorAngleLabel = QtWidgets.QLabel(" Absolute Rotator Angle Filter", self)
        self.l1aqcRotatorAngleCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL1aqcRotatorAngle"]) == 1:
            self.l1aqcRotatorAngleCheckBox.setChecked(True)
        self.l1aqcRotatorAngleMinLabel = QtWidgets.QLabel("       Rotator Angle Min", self)
        self.l1aqcRotatorAngleMinLineEdit = QtWidgets.QLineEdit(self)
        self.l1aqcRotatorAngleMinLineEdit.setText(str(ConfigFile.settings["fL1aqcRotatorAngleMin"]))
        self.l1aqcRotatorAngleMinLineEdit.setValidator(doubleValidator)
        self.l1aqcRotatorAngleMaxLabel = QtWidgets.QLabel("       Rotator Angle Max", self)
        self.l1aqcRotatorAngleMaxLineEdit = QtWidgets.QLineEdit(self)
        self.l1aqcRotatorAngleMaxLineEdit.setText(str(ConfigFile.settings["fL1aqcRotatorAngleMax"]))
        self.l1aqcRotatorAngleMaxLineEdit.setValidator(doubleValidator)
        self.l1aqcSolarTrackerCheckBoxUpdate()
        self.l1aqcRotatorAngleCheckBoxUpdate()

        #   Relative SZA
        l1aqcCleanSunAngleLabel = QtWidgets.QLabel(" Relative Solar Azimuth Filter", self)
        self.l1aqcCleanSunAngleCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL1aqcCleanSunAngle"]) == 1:
            self.l1aqcCleanSunAngleCheckBox.setChecked(True)
        self.l1aqcSunAngleMinLabel = QtWidgets.QLabel("       Rel Angle Min", self)
        self.l1aqcSunAngleMinLineEdit = QtWidgets.QLineEdit(self)
        self.l1aqcSunAngleMinLineEdit.setText(str(ConfigFile.settings["fL1aqcSunAngleMin"]))
        self.l1aqcSunAngleMinLineEdit.setValidator(doubleValidator)
        self.l1aqcSunAngleMaxLabel = QtWidgets.QLabel("       Rel Angle Max", self)
        self.l1aqcSunAngleMaxLineEdit = QtWidgets.QLineEdit(self)
        self.l1aqcSunAngleMaxLineEdit.setText(str(ConfigFile.settings["fL1aqcSunAngleMax"]))
        self.l1aqcSunAngleMaxLineEdit.setValidator(doubleValidator)
        self.l1aqcCleanSunAngleCheckBoxUpdate()

        self.l1aqcSolarTrackerCheckBox.clicked.connect(self.l1aqcSolarTrackerCheckBoxUpdate)
        self.l1aqcRotatorDelayCheckBox.clicked.connect(self.l1aqcRotatorDelayCheckBoxUpdate)
        self.l1aqcCleanPitchRollCheckBox.clicked.connect(self.l1aqcCleanPitchRollCheckBoxUpdate)
        self.l1aqcRotatorAngleCheckBox.clicked.connect(self.l1aqcRotatorAngleCheckBoxUpdate)
        self.l1aqcCleanSunAngleCheckBox.clicked.connect(self.l1aqcCleanSunAngleCheckBoxUpdate)

        #   Deglitcher
        self.l1aqcDeglitchLabel = QtWidgets.QLabel("  Deglitch Data", self)
        self.l1aqcDeglitchCheckBox = QtWidgets.QCheckBox("", self)
        if ConfigFile.settings["bL1aqcDeglitch"]:
            self.l1aqcDeglitchCheckBox.setChecked(True)

        #   Launch Deglitcher Analysis
        self.l1aqcAnomalyButton = QtWidgets.QPushButton("Launch Anomaly Analysis")
        self.l1aqcAnomalyButton.clicked.connect(self.l1aqcAnomalyButtonPressed)
        self.l1aqcDeglitchCheckBoxUpdate()
        self.l1aqcDeglitchCheckBox.clicked.connect(self.l1aqcDeglitchCheckBoxUpdate)

        # L1B
        l1bLabel = QtWidgets.QLabel("Level 1B Processing", self)
        l1bLabel.setFont(l1aLabel_font)
        l1bSublabel1 = QtWidgets.QLabel(" Dark offsets, calibrations and corrections. Interpolate", self)
        l1bSublabel2 = QtWidgets.QLabel("  to common timestamps and wavebands.", self)

        l1bSublabel3 = QtWidgets.QLabel("   Ancillary data are required for Zhang glint correction and", self)
        l1bSublabel4 = QtWidgets.QLabel("   can fill in wind for M99 and QC. Select database download:", self)
        l1bSublabel5 = QtWidgets.QLabel("    (GMAO PROMPTS FOR EARTHDATA LOGIN: <a href=\"https://oceancolor.gsfc.nasa.gov/registration/\">register</a>)", self)
        l1bSublabel6 = QtWidgets.QLabel("    Fallback values when no model available:", self)
        l1bSublabel5.setOpenExternalLinks(True)
        self.l1bGetAncCheckBox1 = QtWidgets.QCheckBox("GMAO MERRA2", self)
        self.l1bGetAncCheckBox2 = QtWidgets.QCheckBox("ECMWF", self)
        if int(ConfigFile.settings["bL1bGetAnc"]) == 1:
            self.l1bGetAncCheckBox1.setChecked(True)
            self.l1bGetAncCheckBox2.setChecked(False)
        self.l1bGetAncCheckBox1.clicked.connect(self.l1bGetAncCheckBoxUpdate1)
        if int(ConfigFile.settings["bL1bGetAnc"]) == 2:
            self.l1bGetAncCheckBox1.setChecked(False)
            self.l1bGetAncCheckBox2.setChecked(True)
        self.l1bGetAncCheckBox2.clicked.connect(self.l1bGetAncCheckBoxUpdate2)


        self.l1bDefaultWindSpeedLabel = QtWidgets.QLabel("          Default Wind Speed (m/s)", self)
        self.l1bDefaultWindSpeedLineEdit = QtWidgets.QLineEdit(self)
        self.l1bDefaultWindSpeedLineEdit.setText(str(ConfigFile.settings["fL1bDefaultWindSpeed"]))
        self.l1bDefaultWindSpeedLineEdit.setValidator(doubleValidator)
        self.l1bDefaultAODLabel = QtWidgets.QLabel("          Default AOD(550)", self)
        self.l1bDefaultAODLineEdit = QtWidgets.QLineEdit(self)
        self.l1bDefaultAODLineEdit.setText(str(ConfigFile.settings["fL1bDefaultAOD"]))
        self.l1bDefaultAODLineEdit.setValidator(doubleValidator)
        self.l1bDefaultSaltLabel = QtWidgets.QLabel("          Default Salinity (psu)", self)
        self.l1bDefaultSaltLineEdit = QtWidgets.QLineEdit(self)
        self.l1bDefaultSaltLineEdit.setText(str(ConfigFile.settings["fL1bDefaultSalt"]))
        self.l1bDefaultSaltLineEdit.setValidator(doubleValidator)
        self.l1bDefaultSSTLabel = QtWidgets.QLabel("          Default SST (C)", self)
        self.l1bDefaultSSTLineEdit = QtWidgets.QLineEdit(self)
        self.l1bDefaultSSTLineEdit.setText(str(ConfigFile.settings["fL1bDefaultSST"]))
        self.l1bDefaultSSTLineEdit.setValidator(doubleValidator)

        l1bCalLabel = QtWidgets.QLabel(" Select calibration/correction regime:", self)
        self.DefaultCalRadioButton = QtWidgets.QRadioButton("     Factory")
        self.DefaultCalRadioButton.setAutoExclusive(False)
        if ConfigFile.settings["bL1bCal"]==1:
            self.DefaultCalRadioButton.setChecked(True)
        self.DefaultCalRadioButton.clicked.connect(self.l1bDefaultCalRadioButtonClicked)

        self.ClassCalRadioButton = QtWidgets.QRadioButton("Class-based")
        self.ClassCalRadioButton.setAutoExclusive(False)
        if ConfigFile.settings["bL1bCal"]==2:
            self.ClassCalRadioButton.setChecked(True)
        self.ClassCalRadioButton.clicked.connect(self.l1bClassCalRadioButtonClicked)
        # '''
        # BUG: While class-based is being debugged
        # '''
        # self.ClassCalRadioButton.setChecked(False)
        # self.ClassCalRadioButton.setEnabled(False)

        self.FullCalRadioButton = QtWidgets.QRadioButton("Full Characterization:")
        self.FullCalRadioButton.setAutoExclusive(False)
        # self.FullCalRadioButton.setDisabled(1) # <---- Needs to have a test for full cal file availability
        if ConfigFile.settings["bL1bCal"]==3:
            self.FullCalRadioButton.setChecked(True)
        self.FullCalRadioButton.clicked.connect(self.l1bFullCalRadioButtonClicked)

        self.l1bGetFRMCheck1 = QtWidgets.QCheckBox("local files", self)

        ''' NOTE: disabled until fidraddb_api can be resolved.'''
        self.l1bGetFRMCheck2 = QtWidgets.QCheckBox("FidRadDB", self)
        self.l1bGetFRMCheck2.setDisabled(1)

        if int(ConfigFile.settings["FidRadDB"]) == 0:
            self.l1bGetFRMCheck1.setChecked(True)
            self.l1bGetFRMCheck2.setChecked(False)
        if int(ConfigFile.settings["FidRadDB"]) == 1:
            self.l1bGetFRMCheck1.setChecked(False)
            self.l1bGetFRMCheck2.setChecked(True)
        self.FullCalDir = ConfigFile.settings['FullCalDir']
        self.l1bGetFRMCheck1.clicked.connect(self.l1bGetFRMCheckUpdate1)
        self.l1bGetFRMCheck2.clicked.connect(self.l1bGetFRMCheckUpdate2)

        # if ConfigFile.settings['FullCalDir'] == ' ':
        #     self.FullCalDirButton = QtWidgets.QPushButton('Select Characterization Folder', self)
        # else:
        #     self.FullCalDirButton = QtWidgets.QPushButton(os.path.basename(ConfigFile.settings['FullCalDir']), self)
        # self.FullCalDirButton.clicked.connect(self.FullCalDirButtonPressed)

        l1bInterpIntervalLabel = QtWidgets.QLabel("    Interpolation Interval (nm)", self)
        self.l1bInterpIntervalLineEdit = QtWidgets.QLineEdit(self)
        self.l1bInterpIntervalLineEdit.setText(str(ConfigFile.settings["fL1bInterpInterval"]))
        self.l1bInterpIntervalLineEdit.setValidator(doubleValidator)

        l1bPlotTimeInterpLabel = QtWidgets.QLabel(f"    Generate Plots ({os.path.split(MainConfig.settings['outDir'])[-1]}/Plots/L1B_Interp/)", self)
        self.l1bPlotTimeInterpCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL1bPlotTimeInterp"]) == 1:
            self.l1bPlotTimeInterpCheckBox.setChecked(True)
        self.l1bPlotTimeInterpCheckBox.clicked.connect(self.l1bPlotTimeInterpCheckBoxUpdate)

        l1bPlotIntervalLabel = QtWidgets.QLabel("      Plot Interval (nm)", self)
        self.l1bPlotIntervalLineEdit = QtWidgets.QLineEdit(self)
        self.l1bPlotIntervalLineEdit.setText(str(ConfigFile.settings["fL1bPlotInterval"]))
        self.l1bPlotIntervalLineEdit.setValidator(doubleValidator)

        # L1BQC
        l1bqcLabel = QtWidgets.QLabel("Level 1BQC Processing", self)
        l1bqcLabel_font = l1bqcLabel.font()
        l1bqcLabel_font.setPointSize(12)
        l1bqcLabel_font.setBold(True)
        l1bqcLabel.setFont(l1bqcLabel_font)
        l1bqcSublabel = QtWidgets.QLabel(" Data quality control filters.", self)

        #   Lt UV<NIR
        l1bqcLtUVNIRLabel= QtWidgets.QLabel("   Eliminate where Lt(NIR)>Lt(UV)", self)
        self.l1bqcLtUVNIRCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL1bqcLtUVNIR"]) == 1:
            self.l1bqcLtUVNIRCheckBox.setChecked(True)
        self.l1bqcLtUVNIRCheckBox.clicked.connect(self.l1bqcLtUVNIRCheckBoxUpdate)

        #   L1BQC Max Wind
        l1bqcMaxWindLabel = QtWidgets.QLabel("   Max. Wind Speed (m/s)", self)
        self.l1bqcMaxWindLineEdit = QtWidgets.QLineEdit(self)
        self.l1bqcMaxWindLineEdit.setText(str(ConfigFile.settings["fL1bqcMaxWind"]))
        self.l1bqcMaxWindLineEdit.setValidator(doubleValidator)

        #   L1BQC Min/Max SZA
        l1bqcSZAMinLabel = QtWidgets.QLabel("   SZA Minimum (deg)", self)
        self.l1bqcSZAMinLineEdit = QtWidgets.QLineEdit(self)
        self.l1bqcSZAMinLineEdit.setText(str(ConfigFile.settings["fL1bqcSZAMin"]))
        self.l1bqcSZAMinLineEdit.setValidator(doubleValidator)
        l1bqcSZAMaxLabel = QtWidgets.QLabel("   SZA Maximum (deg)", self)
        self.l1bqcSZAMaxLineEdit = QtWidgets.QLineEdit(self)
        self.l1bqcSZAMaxLineEdit.setText(str(ConfigFile.settings["fL1bqcSZAMax"]))
        self.l1bqcSZAMaxLineEdit.setValidator(doubleValidator)

        # L1BQC Spectral Outlier Filter
        l1bqcSpecQualityCheckLabel = QtWidgets.QLabel("  Enable Spectral Outlier Filter & Plots", self)
        self.l1bqcSpecQualityCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL1bqcEnableSpecQualityCheck"]) == 1:
            self.l1bqcSpecQualityCheckBox.setChecked(True)

        self.l1bqcSpecFilterEsLabel = QtWidgets.QLabel("       Filter Sigma Es", self)
        self.l1bqcSpecFilterEsLineEdit = QtWidgets.QLineEdit(self)
        self.l1bqcSpecFilterEsLineEdit.setText(str(ConfigFile.settings["fL1bqcSpecFilterEs"]))
        self.l1bqcSpecFilterEsLineEdit.setValidator(doubleValidator)
        self.l1bqcSpecFilterLiLabel = QtWidgets.QLabel("       Filter Sigma Li", self)
        self.l1bqcSpecFilterLiLineEdit = QtWidgets.QLineEdit(self)
        self.l1bqcSpecFilterLiLineEdit.setText(str(ConfigFile.settings["fL1bqcSpecFilterLi"]))
        self.l1bqcSpecFilterLiLineEdit.setValidator(doubleValidator)
        self.l1bqcSpecFilterLtLabel = QtWidgets.QLabel("       Filter Sigma Lt", self)
        self.l1bqcSpecFilterLtLineEdit = QtWidgets.QLineEdit(self)
        self.l1bqcSpecFilterLtLineEdit.setText(str(ConfigFile.settings["fL1bqcSpecFilterLt"]))
        self.l1bqcSpecFilterLtLineEdit.setValidator(doubleValidator)

        self.l1bqcSpecQualityCheckBoxUpdate()

        # L1BQC Meteorology Flags
        l1bqcQualityFlagLabel = QtWidgets.QLabel("   Enable Meteorological Filters", self)
        self.l1bqcQualityFlagCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL1bqcEnableQualityFlags"]) == 1:
            self.l1bqcQualityFlagCheckBox.setChecked(True)

        self.l1bqcCloudFlagLabel = QtWidgets.QLabel("       Cloud Li(750)/Es(750)>", self)
        self.l1bqcCloudFlagLineEdit = QtWidgets.QLineEdit("", self)
        self.l1bqcCloudFlagLineEdit.setText(str(ConfigFile.settings["fL1bqcCloudFlag"]))
        self.l1bqcCloudFlagLineEdit.setValidator(doubleValidator)

        self.l1bqcEsFlagLabel = QtWidgets.QLabel("       Significant Es(480) (uW cm^-2 nm^-1)", self)
        self.l1bqcEsFlagLineEdit = QtWidgets.QLineEdit(self)
        self.l1bqcEsFlagLineEdit.setText(str(ConfigFile.settings["fL1bqcSignificantEsFlag"]))
        self.l1bqcEsFlagLineEdit.setValidator(doubleValidator)

        self.l1bqcDawnDuskFlagLabel = QtWidgets.QLabel("       Dawn/Dusk Es(470/680)<", self)
        self.l1bqcDawnDuskFlagLineEdit = QtWidgets.QLineEdit("", self)
        self.l1bqcDawnDuskFlagLineEdit.setText(str(ConfigFile.settings["fL1bqcDawnDuskFlag"]))
        self.l1bqcDawnDuskFlagLineEdit.setValidator(doubleValidator)

        self.l1bqcRainfallHumidityFlagLabel = QtWidgets.QLabel("       Rain/Humid. Es(720/370)<", self)
        self.l1bqcRainfallHumidityFlagLineEdit = QtWidgets.QLineEdit("", self)
        self.l1bqcRainfallHumidityFlagLineEdit.setText(str(ConfigFile.settings["fL1bqcRainfallHumidityFlag"]))
        self.l1bqcRainfallHumidityFlagLineEdit.setValidator(doubleValidator)

        self.l1bqcQualityFlagCheckBoxUpdate()
        self.l1bqcSpecQualityCheckBox.clicked.connect(self.l1bqcSpecQualityCheckBoxUpdate)
        self.l1bqcQualityFlagCheckBox.clicked.connect(self.l1bqcQualityFlagCheckBoxUpdate)

        # L2
        l2Label = QtWidgets.QLabel("Level 2 Processing", self)
        l2Label.setFont(l1aLabel_font)
        l2Sublabel = QtWidgets.QLabel(" Temporal binning, glitter reduction, glint", self)
        l2Sublabel2 = QtWidgets.QLabel("  correction, residual correction, QC,", self)
        l2Sublabel3 = QtWidgets.QLabel("  satellite convolution, OC product generation,", self)
        l2Sublabel4 = QtWidgets.QLabel("  SeaBASS file output.", self)

        #   L2 Ensembles
        l2ensLabel = QtWidgets.QLabel("L2 Ensembles", self)
        l2ensLabel.setFont(l1aLabel_font)

        #   L2 Station breakout
        l2StationsLabel = QtWidgets.QLabel("Extract Cruise Stations", self)
        self.l2StationsCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2Stations"]) == 1:
            self.l2StationsCheckBox.setChecked(True)

        #   L2 Time Average Rrs
        l2TimeIntervalLabel = QtWidgets.QLabel("  Ensemble Interval (secs; 0=None)", self)
        self.l2TimeIntervalLineEdit = QtWidgets.QLineEdit(self)
        self.l2TimeIntervalLineEdit.setText(str(ConfigFile.settings["fL2TimeInterval"]))
        self.l2TimeIntervalLineEdit.setValidator(intValidator)

        #   L2 Set percentage Lt filter
        self.l2EnablePercentLtLabel = QtWidgets.QLabel("    Enable Percent Lt Calculation", self)
        self.l2EnablePercentLtCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2EnablePercentLt"]) == 1:
            self.l2EnablePercentLtCheckBox.setChecked(True)
        self.l2PercentLtLabel = QtWidgets.QLabel("     Percent Lt (%)", self)
        self.l2PercentLtLineEdit = QtWidgets.QLineEdit(self)
        self.l2PercentLtLineEdit.setText(str(ConfigFile.settings["fL2PercentLt"]))
        self.l2PercentLtLineEdit.setValidator(doubleValidator)

        self.l2EnablePercentLtCheckBoxUpdate()

        #   L2 Rho Sky Correction
        l2RhoSkyLabel = QtWidgets.QLabel("L2 Sky/Sunglint Correction (ρ)", self)
        l2RhoSkyLabel.setFont(l1aLabel_font)

        self.RhoRadioButtonDefault = QtWidgets.QRadioButton("Mobley (1999) ρ")
        self.RhoRadioButtonDefault.setAutoExclusive(False)
        if ConfigFile.settings["bL2DefaultRho"]==1:
            self.RhoRadioButtonDefault.setChecked(True)
        self.RhoRadioButtonDefault.clicked.connect(self.l2RhoRadioButtonDefaultClicked)

        self.RhoRadioButtonZhang = QtWidgets.QRadioButton("Zhang et al. (2017) ρ")
        self.RhoRadioButtonZhang.setAutoExclusive(False)
        if ConfigFile.settings["bL2ZhangRho"]==1:
            self.RhoRadioButtonZhang.setChecked(True)
        self.RhoRadioButtonZhang.clicked.connect(self.l2RhoRadioButtonZhangClicked)

        self.RhoRadoButton3C = QtWidgets.QRadioButton("Groetsch et al. (2017)")
        self.RhoRadoButton3C.setAutoExclusive(False)
        self.RhoRadoButton3C.setDisabled(True)

        self.RhoRadioButtonYour = QtWidgets.QRadioButton("Your Glint (2023) ρ")
        self.RhoRadioButtonYour.setAutoExclusive(False)
        self.RhoRadioButtonYour.setDisabled(True)
        # if ConfigFile.settings["bL2YourRho"]==1:
        #     self.RhoRadioButtonYour.setChecked(True)
        # self.RhoRadioButtonYour.clicked.connect(self.l2RhoRadioButtonYourClicked)


        #   L2 NIR AtmoCorr
        l2NIRCorrectionLabel = QtWidgets.QLabel("NIR Residual Correction", self)
        self.l2NIRCorrectionCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2PerformNIRCorrection"]) == 1:
            self.l2NIRCorrectionCheckBox.setChecked(True)

        self.SimpleNIRRadioButton = QtWidgets.QRadioButton("   Mueller and Austin (1995) (blue water)")
        self.SimpleNIRRadioButton.setAutoExclusive(False)
        if ConfigFile.settings["bL2SimpleNIRCorrection"] == 1:
            self.SimpleNIRRadioButton.setChecked(True)
        self.SimpleNIRRadioButton.clicked.connect(self.l2SimpleNIRRadioButtonClicked)
        self.SimSpecNIRRadioButton = QtWidgets.QRadioButton("   SimSpec. Ruddick et al. (2006) (turbid)")
        self.SimSpecNIRRadioButton.setAutoExclusive(False)
        if ConfigFile.settings["bL2SimSpecNIRCorrection"] == 1:
            self.SimSpecNIRRadioButton.setChecked(True)
        self.SimSpecNIRRadioButton.clicked.connect(self.l2SimSpecNIRRadioButtonClicked)
        self.YourNIRRadioButton = QtWidgets.QRadioButton("   Your NIR Residual (2023) (universal)")
        self.YourNIRRadioButton.setAutoExclusive(False)
        # if ConfigFile.settings["bL2YourNIRCorrection"] == 1:
        #     self.YourNIRRadioButton.setChecked(True)
        # self.YourNIRRadioButton.clicked.connect(self.l2YourNIRRadioButtonClicked)
        self.YourNIRRadioButton.setDisabled(True)

        self.l2NIRCorrectionCheckBoxUpdate()

        #   L2 Remove negative spectra
        #   Could add spectral range here
        self.l2NegativeSpecLabel = QtWidgets.QLabel("Remove Negative Spectra", self)
        self.l2NegativeSpecCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2NegativeSpec"]) == 1:
            self.l2NegativeSpecCheckBox.setChecked(True)

        self.l2NegativeSpecCheckBoxUpdate()

        #   BRDF Correction
        self.l2BRDFLabel = QtWidgets.QLabel("BRDF Correction", self)
        self.l2BRDFCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2BRDF"]) == 1:
            self.l2BRDFCheckBox.setChecked(True)
        self.l2BRDF_fQLabel = QtWidgets.QLabel("Morel fQ", self)
        self.l2BRDF_fQCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2BRDF_fQ"]) == 1:
            self.l2BRDF_fQCheckBox.setChecked(True)
        self.l2BRDF_IOPLabel = QtWidgets.QLabel("Lee IOP", self)
        self.l2BRDF_IOPCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2BRDF_IOP"]) == 1:
            self.l2BRDF_IOPCheckBox.setChecked(True)
        self.l2BRDFCheckBoxUpdate()


        l2ProductLabel = QtWidgets.QLabel("L2 Products", self)
        l2ProductLabel.setFont(l1aLabel_font)

        #   Spectral Weighting/Convolution
        l2WeightsLabel = QtWidgets.QLabel("Convolve to Satellite Bands:", self)

        l2WeightMODISALabel = QtWidgets.QLabel("AQUA *", self)
        self.l2WeightMODISACheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2WeightMODISA"]) == 1:
            self.l2WeightMODISACheckBox.setChecked(True)
        l2WeightMODISALabel2 = QtWidgets.QLabel("* Automatic for Derived Products", self)

        l2WeightMODISTLabel = QtWidgets.QLabel("TERRA", self)
        self.l2WeightMODISTCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2WeightMODIST"]) == 1:
            self.l2WeightMODISTCheckBox.setChecked(True)

        l2WeightVIIRSNLabel = QtWidgets.QLabel("V-NPP", self)
        self.l2WeightVIIRSNCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2WeightVIIRSN"]) == 1:
            self.l2WeightVIIRSNCheckBox.setChecked(True)

        l2WeightVIIRSJLabel = QtWidgets.QLabel("V-JPSS", self)
        self.l2WeightVIIRSJCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2WeightVIIRSJ"]) == 1:
            self.l2WeightVIIRSJCheckBox.setChecked(True)

        l2WeightSentinel3ALabel = QtWidgets.QLabel("Sen-3A", self)
        self.l2WeightSentinel3ACheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2WeightSentinel3A"]) == 1:
            self.l2WeightSentinel3ACheckBox.setChecked(True)

        l2WeightSentinel3BLabel = QtWidgets.QLabel("Sen-3B", self)
        self.l2WeightSentinel3BCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2WeightSentinel3B"]) == 1:
            self.l2WeightSentinel3BCheckBox.setChecked(True)

        #   Plots
        l2PlotsLabel = QtWidgets.QLabel("Generate Spectral Plots", self)
        l2PlotRrsLabel = QtWidgets.QLabel("Rrs", self)
        self.l2PlotRrsCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2PlotRrs"]) == 1:
            self.l2PlotRrsCheckBox.setChecked(True)

        l2PlotnLwLabel = QtWidgets.QLabel("nLw", self)
        self.l2PlotnLwCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2PlotnLw"]) == 1:
            self.l2PlotnLwCheckBox.setChecked(True)

        l2PlotEsLabel = QtWidgets.QLabel("Es", self)
        self.l2PlotEsCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2PlotEs"]) == 1:
            self.l2PlotEsCheckBox.setChecked(True)

        l2PlotLiLabel = QtWidgets.QLabel("Li", self)
        self.l2PlotLiCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2PlotLi"]) == 1:
            self.l2PlotLiCheckBox.setChecked(True)

        l2PlotLtLabel = QtWidgets.QLabel("Lt", self)
        self.l2PlotLtCheckBox = QtWidgets.QCheckBox("", self)
        if int(ConfigFile.settings["bL2PlotLt"]) == 1:
            self.l2PlotLtCheckBox.setChecked(True)

        self.l2StationsCheckBox.clicked.connect(self.l2StationsCheckBoxUpdate)
        self.l2EnablePercentLtCheckBox.clicked.connect(self.l2EnablePercentLtCheckBoxUpdate)
        self.l2NIRCorrectionCheckBox.clicked.connect(self.l2NIRCorrectionCheckBoxUpdate)
        self.l2NegativeSpecCheckBox.clicked.connect(self.l2NegativeSpecCheckBoxUpdate)
        self.l2BRDFCheckBox.clicked.connect(self.l2BRDFCheckBoxUpdate)
        self.l2BRDF_fQCheckBox.clicked.connect(self.l2BRDF_fQCheckBoxUpdate)
        self.l2BRDF_IOPCheckBox.clicked.connect(self.l2BRDF_IOPCheckBoxUpdate)

        self.l2OCproducts = QtWidgets.QPushButton("Derived L2 Ocean Color Products", self)
        self.l2OCproducts.clicked.connect(self.l2OCproductsButtonPressed)

        l2SaveSeaBASSLabel = QtWidgets.QLabel("Save SeaBASS Files", self)
        self.l2SeaBASSHeaderEditButton = QtWidgets.QPushButton("Edit SeaBASS Header", self)
        self.l2SeaBASSHeaderEditButton.clicked.connect(self.l2SeaBASSHeaderEditButtonPressed)
        self.l2SaveSeaBASSCheckBox = QtWidgets.QCheckBox("", self)
        self.l2SaveSeaBASSCheckBox.clicked.connect(self.l2SaveSeaBASSCheckBoxUpdate)
        if int(ConfigFile.settings["bL2SaveSeaBASS"]) == 1:
            self.l2SaveSeaBASSCheckBox.setChecked(True)
        # self.l2SaveSeaBASSCheckBox.clicked.connect(self.l2SaveSeaBASSCheckBoxUpdate)
        # self.l1bSaveSeaBASSCheckBoxUpdate()
        self.l2SaveSeaBASSCheckBoxUpdate()

        self.l2SeaBASSHeaderLabel = QtWidgets.QLabel(f'  {ConfigFile.settings["seaBASSHeaderFileName"]}', self)

        l2WriteReportLabel = QtWidgets.QLabel("Write PDF Report", self)
        self.l2WriteReportCheckBox = QtWidgets.QCheckBox("", self)
        self.l2WriteReportCheckBox.clicked.connect(self.l2WriteReportCheckBoxUpdate)
        if int(ConfigFile.settings["bL2WriteReport"]) == 1:
            self.l2WriteReportCheckBox.setChecked(True)
        # self.l2WriteReportCheckBox.clicked.connect(self.l2SWriteReportCheckBoxUpdate)
        self.l2WriteReportCheckBoxUpdate()

        self.saveButton = QtWidgets.QPushButton("Save/Close")
        self.saveAsButton = QtWidgets.QPushButton("Save As")
        self.cancelButton = QtWidgets.QPushButton("Cancel")

        self.saveButton.clicked.connect(self.saveButtonPressed)
        self.saveAsButton.clicked.connect(self.saveAsButtonPressed)
        self.cancelButton.clicked.connect(self.cancelButtonPressed)

        #################################################################################

        # Whole Window Box
        VBox = QtWidgets.QVBoxLayout()

        # Vertical Box (left)
        VBox1 = QtWidgets.QVBoxLayout()

        # sensor type box
        VBox1.addWidget(sensorTypeLabel)
        VBox1.addWidget(self.sensorTypeComboBox)
        # Instrument Files Setup
        # Horizontal Box
        calHBox1 = QtWidgets.QHBoxLayout()
        calHBox1.addWidget(self.addCalibrationFileButton)
        calHBox1.addWidget(self.deleteCalibrationFileButton)
        VBox1.addLayout(calHBox1)
        # Horizontal Box
        calHBox = QtWidgets.QHBoxLayout()
        calHBox.addWidget(self.calibrationFileComboBox)
        calHBox.addWidget(self.calibrationEnabledCheckBox)
        VBox1.addLayout(calHBox)

        VBox1.addWidget(calibrationFrameTypeLabel)
        VBox1.addWidget(self.calibrationFrameTypeComboBox)

        # L1A
        VBox1.addWidget(l1aLabel)
        VBox1.addWidget(l1aSublabel)

        UTCOffsetHBox = QtWidgets.QHBoxLayout()
        UTCOffsetHBox.addWidget(self.l1aUTCOffsetLabel)
        UTCOffsetHBox.addWidget(self.l1aUTCOffsetLineEdit)
        VBox1.addLayout(UTCOffsetHBox)

        VBox1.addWidget(l1aCleanSZALabel)
        # Horizontal Box; SZA Filter
        szaHBox = QtWidgets.QHBoxLayout()
        szaHBox.addWidget(self.l1aCleanSZAMaxLabel)
        szaHBox.addWidget(self.l1aCleanSZACheckBox)
        szaHBox.addWidget(self.l1aCleanSZAMaxLineEdit)
        VBox1.addLayout(szaHBox)

        # L1AQC
        VBox1.addWidget(l1aqcLabel)
        VBox1.addWidget(l1aqcSublabel)

        #   L1AQC Pitch & Roll
        PitchRollHBox = QtWidgets.QHBoxLayout()
        PitchRollHBox.addWidget(self.l1aqcCleanPitchRollLabel)
        PitchRollHBox.addWidget(self.l1aqcCleanPitchRollCheckBox)
        VBox1.addLayout(PitchRollHBox)
        PitchRollHBox2 = QtWidgets.QHBoxLayout()
        PitchRollHBox2.addWidget(self.l1aqcPitchRollPitchLabel)
        PitchRollHBox2.addWidget(self.l1aqcPitchRollPitchLineEdit)
        VBox1.addLayout(PitchRollHBox2)

        #   SolarTracker
        SolarTrackerHBox = QtWidgets.QHBoxLayout()
        SolarTrackerHBox.addWidget(self.l1aqcSolarTrackerLabel)
        SolarTrackerHBox.addWidget(self.l1aqcSolarTrackerCheckBox)
        VBox1.addLayout(SolarTrackerHBox)

        #   L1AQC Rotator Home
        RotHomeAngleHBox = QtWidgets.QHBoxLayout()
        RotHomeAngleHBox.addWidget(self.l1aqcRotatorHomeAngleLabel)
        RotHomeAngleHBox.addWidget(self.l1aqcRotatorHomeAngleLineEdit)
        VBox1.addLayout(RotHomeAngleHBox)
        RotatorDelayHBox = QtWidgets.QHBoxLayout()
        RotatorDelayHBox.addWidget(self.l1aqcRotatorDelayLabel)
        RotatorDelayHBox.addWidget(self.l1aqcRotatorDelayCheckBox)
        RotatorDelayHBox.addWidget(self.l1aqcRotatorDelayLineEdit)
        VBox1.addLayout(RotatorDelayHBox)

        #   L1AQC Rotator Absolute
        rotateHBox = QtWidgets.QHBoxLayout()
        rotateHBox.addWidget(self.l1aqcRotatorAngleLabel)
        rotateHBox.addWidget(self.l1aqcRotatorAngleCheckBox)
        VBox1.addLayout(rotateHBox)
        RotMinHBox = QtWidgets.QHBoxLayout()
        RotMinHBox.addWidget(self.l1aqcRotatorAngleMinLabel)
        RotMinHBox.addWidget(self.l1aqcRotatorAngleMinLineEdit)
        VBox1.addLayout(RotMinHBox)
        RotMaxHBox = QtWidgets.QHBoxLayout()
        RotMaxHBox.addWidget(self.l1aqcRotatorAngleMaxLabel)
        RotMaxHBox.addWidget(self.l1aqcRotatorAngleMaxLineEdit)
        VBox1.addLayout(RotMaxHBox)

        #   L1AQC Relative SZA
        CleanSunAngleHBox = QtWidgets.QHBoxLayout()
        CleanSunAngleHBox.addWidget(l1aqcCleanSunAngleLabel)
        CleanSunAngleHBox.addWidget(self.l1aqcCleanSunAngleCheckBox)
        VBox1.addLayout(CleanSunAngleHBox)
        SunAngleMinHBox = QtWidgets.QHBoxLayout()
        SunAngleMinHBox.addWidget(self.l1aqcSunAngleMinLabel)
        SunAngleMinHBox.addWidget(self.l1aqcSunAngleMinLineEdit)
        VBox1.addLayout(SunAngleMinHBox)
        SunAngleMaxHBox = QtWidgets.QHBoxLayout()
        SunAngleMaxHBox.addWidget(self.l1aqcSunAngleMaxLabel)
        SunAngleMaxHBox.addWidget(self.l1aqcSunAngleMaxLineEdit)
        VBox1.addLayout(SunAngleMaxHBox)

        #   L1AQC Deglitcher
        deglitchHBox = QtWidgets.QHBoxLayout()
        deglitchHBox.addWidget(self.l1aqcDeglitchLabel)
        deglitchHBox.addWidget(self.l1aqcDeglitchCheckBox)
        VBox1.addLayout(deglitchHBox)
        #       L1AQC Anomaly Launcher
        # VBox1.addWidget(l1aqcAnomalySublabel1)
        # VBox1.addWidget(l1aqcAnomalySublabel2)
        VBox1.addWidget(self.l1aqcAnomalyButton)


        VBox1.addStretch()

        # Second Vertical Box
        VBox2 = QtWidgets.QVBoxLayout()
        # VBox2.setAlignment(QtCore.Qt.AlignBottom)

        # L1B
        VBox2.addWidget(l1bLabel)
        VBox2.addWidget(l1bSublabel1)
        VBox2.addWidget(l1bSublabel2)

        #  Ancillary Models
        VBox2.addWidget(l1bSublabel3)
        VBox2.addWidget(l1bSublabel4)
        l1bGetAncHBox1 = QtWidgets.QHBoxLayout()
        l1bGetAncHBox1.addWidget(self.l1bGetAncCheckBox1)
        # l1bGetAncHBox1.addWidget(l1bSublabel4)
        l1bGetAncHBox1.addWidget(self.l1bGetAncCheckBox2)
        VBox2.addLayout(l1bGetAncHBox1)
        VBox2.addWidget(l1bSublabel5)
        VBox2.addWidget(l1bSublabel6)

        #   Default Wind
        WindSpeedHBox2 = QtWidgets.QHBoxLayout()
        WindSpeedHBox2.addWidget(self.l1bDefaultWindSpeedLabel)
        WindSpeedHBox2.addWidget(self.l1bDefaultWindSpeedLineEdit)
        VBox2.addLayout(WindSpeedHBox2)
        #   Default AOD
        AODHBox2 = QtWidgets.QHBoxLayout()
        AODHBox2.addWidget(self.l1bDefaultAODLabel)
        AODHBox2.addWidget(self.l1bDefaultAODLineEdit)
        VBox2.addLayout(AODHBox2)
        #   Default Salt
        SaltHBox2 = QtWidgets.QHBoxLayout()
        SaltHBox2.addWidget(self.l1bDefaultSaltLabel)
        SaltHBox2.addWidget(self.l1bDefaultSaltLineEdit)
        VBox2.addLayout(SaltHBox2)
        #   Default SST
        SSTHBox2 = QtWidgets.QHBoxLayout()
        SSTHBox2.addWidget(self.l1bDefaultSSTLabel)
        SSTHBox2.addWidget(self.l1bDefaultSSTLineEdit)
        VBox2.addLayout(SSTHBox2)

        #   Instrument/Cal Files
        VBox2.addWidget(l1bCalLabel)
        CalHBox2 = QtWidgets.QHBoxLayout()
        CalHBox2.addWidget(self.DefaultCalRadioButton)
        CalHBox2.addWidget(self.ClassCalRadioButton)

        ''' NOTE: Disable in master while under development in dev '''
        # self.ClassCalRadioButton.setDisabled(True)

        VBox2.addLayout(CalHBox2)

        # CalHBox3 = QtWidgets.QHBoxLayout()
        # CalHBox3.addWidget(self.FullCalRadioButton)
        # CalHBox3.addWidget(self.FullCalDirButton)
        # CalHBox3.addStretch(1)
        # VBox2.addLayout(CalHBox3)

        CalHBox3 = QtWidgets.QHBoxLayout()
        CalHBox3.addWidget(self.FullCalRadioButton)
        CalHBox3.addWidget(self.l1bGetFRMCheck1)
        CalHBox3.addWidget(self.l1bGetFRMCheck2)

        CalHBox3.addStretch(1)
        VBox2.addLayout(CalHBox3)


        #   Interpolation interval (wavelength)
        interpHBox = QtWidgets.QHBoxLayout()
        interpHBox.addWidget(l1bInterpIntervalLabel)
        interpHBox.addWidget(self.l1bInterpIntervalLineEdit)
        VBox2.addLayout(interpHBox)

        l1bPlotTimeInterpHBox = QtWidgets.QHBoxLayout()
        l1bPlotTimeInterpHBox.addWidget(l1bPlotTimeInterpLabel)
        l1bPlotTimeInterpHBox.addWidget(self.l1bPlotTimeInterpCheckBox)
        VBox2.addLayout(l1bPlotTimeInterpHBox)

        #   Plot interval (wavelength)
        plotInterpHBox = QtWidgets.QHBoxLayout()
        plotInterpHBox.addWidget(l1bPlotIntervalLabel)
        plotInterpHBox.addWidget(self.l1bPlotIntervalLineEdit)
        VBox2.addLayout(plotInterpHBox)

        VBox2.addSpacing(10)
        VBox2.addStretch()

        # L1BQC
        VBox2.addWidget(l1bqcLabel)
        VBox2.addWidget(l1bqcSublabel)

        # Lt UV<NIR
        LtUVNIRHBox = QtWidgets.QHBoxLayout()
        LtUVNIRHBox.addWidget(l1bqcLtUVNIRLabel)
        LtUVNIRHBox.addWidget(self.l1bqcLtUVNIRCheckBox)
        VBox2.addLayout(LtUVNIRHBox)

        #   Max wind
        maxWindBox = QtWidgets.QHBoxLayout()
        maxWindBox.addWidget(l1bqcMaxWindLabel)
        maxWindBox.addWidget(self.l1bqcMaxWindLineEdit)
        VBox2.addLayout(maxWindBox)

        #   SZA Min/Max
        SZAHBox1 = QtWidgets.QHBoxLayout()
        SZAHBox1.addWidget(l1bqcSZAMinLabel)
        SZAHBox1.addWidget(self.l1bqcSZAMinLineEdit)
        VBox2.addLayout(SZAHBox1)

        SZAHBox2 = QtWidgets.QHBoxLayout()
        SZAHBox2.addWidget(l1bqcSZAMaxLabel)
        SZAHBox2.addWidget(self.l1bqcSZAMaxLineEdit)
        VBox2.addLayout(SZAHBox2)

        VBox2.addStretch()

        # Third Vertical box
        VBox3 = QtWidgets.QVBoxLayout()
        # VBox3.setAlignment(QtCore.Qt.AlignBottom)

         #  Spectral Outlier Filter
        SpecFilterHBox = QtWidgets.QHBoxLayout()
        SpecFilterHBox.addWidget(l1bqcSpecQualityCheckLabel)
        SpecFilterHBox.addWidget(self.l1bqcSpecQualityCheckBox)
        VBox3.addLayout(SpecFilterHBox)
        SpecFilterEsHBox = QtWidgets.QHBoxLayout()
        SpecFilterEsHBox.addWidget(self.l1bqcSpecFilterEsLabel)
        SpecFilterEsHBox.addWidget(self.l1bqcSpecFilterEsLineEdit)
        VBox3.addLayout(SpecFilterEsHBox)

        #   Spectral Outlier Filter
        SpecFilterLiHBox = QtWidgets.QHBoxLayout()
        SpecFilterLiHBox.addWidget(self.l1bqcSpecFilterLiLabel)
        SpecFilterLiHBox.addWidget(self.l1bqcSpecFilterLiLineEdit)
        VBox3.addLayout(SpecFilterLiHBox)
        SpecFilterLtHBox = QtWidgets.QHBoxLayout()
        SpecFilterLtHBox.addWidget(self.l1bqcSpecFilterLtLabel)
        SpecFilterLtHBox.addWidget(self.l1bqcSpecFilterLtLineEdit)
        VBox3.addLayout(SpecFilterLtHBox)

        #   Meteorology Flags
        QualityFlagHBox = QtWidgets.QHBoxLayout()
        QualityFlagHBox.addWidget(l1bqcQualityFlagLabel)
        QualityFlagHBox.addWidget(self.l1bqcQualityFlagCheckBox)
        VBox3.addLayout(QualityFlagHBox)
        CloudFlagHBox = QtWidgets.QHBoxLayout()
        CloudFlagHBox.addWidget(self.l1bqcCloudFlagLabel)
        CloudFlagHBox.addWidget(self.l1bqcCloudFlagLineEdit)
        VBox3.addLayout(CloudFlagHBox)
        EsFlagHBox = QtWidgets.QHBoxLayout()
        EsFlagHBox.addWidget(self.l1bqcEsFlagLabel)
        EsFlagHBox.addWidget(self.l1bqcEsFlagLineEdit)
        VBox3.addLayout(EsFlagHBox)
        DawnFlagHBox =QtWidgets.QHBoxLayout()
        DawnFlagHBox.addWidget(self.l1bqcDawnDuskFlagLabel)
        DawnFlagHBox.addWidget(self.l1bqcDawnDuskFlagLineEdit)
        VBox3.addLayout(DawnFlagHBox)
        RainFlagHBox = QtWidgets.QHBoxLayout()
        RainFlagHBox.addWidget(self.l1bqcRainfallHumidityFlagLabel)
        RainFlagHBox.addWidget(self.l1bqcRainfallHumidityFlagLineEdit)
        VBox3.addLayout(RainFlagHBox)

        VBox3.addSpacing(30)
        VBox3.addStretch()

        # L2
        VBox3.addWidget(l2Label)
        VBox3.addWidget(l2Sublabel)
        VBox3.addWidget(l2Sublabel2)
        VBox3.addWidget(l2Sublabel3)
        VBox3.addWidget(l2Sublabel4)

        #   L2 Ensembles
        VBox3.addWidget(l2ensLabel)

        #   L2 Stations
        StationsHBox = QtWidgets.QHBoxLayout()
        StationsHBox.addWidget(l2StationsLabel)
        StationsHBox.addWidget(self.l2StationsCheckBox)
        VBox3.addLayout(StationsHBox)

        #   L2 Time Average Rrs
        TimeAveHBox = QtWidgets.QHBoxLayout()
        TimeAveHBox.addWidget(l2TimeIntervalLabel)
        TimeAveHBox.addWidget(self.l2TimeIntervalLineEdit)
        VBox3.addLayout(TimeAveHBox)

        #   L2 Percent Light; Hooker & Morel 2003
        PercentLtHBox = QtWidgets.QHBoxLayout()
        PercentLtHBox.addWidget(self.l2EnablePercentLtLabel)
        PercentLtHBox.addWidget(self.l2EnablePercentLtCheckBox)
        VBox3.addLayout(PercentLtHBox)
        PercentLtHBox2 = QtWidgets.QHBoxLayout()
        PercentLtHBox2.addWidget(self.l2PercentLtLabel)
        PercentLtHBox2.addWidget(self.l2PercentLtLineEdit)
        VBox3.addLayout(PercentLtHBox2)

        #   L2 Rho Skyglint/Sunglint Correction
        VBox3.addWidget(l2RhoSkyLabel)

        #   Rho model
        RhoHBox2 = QtWidgets.QHBoxLayout()
        RhoHBox2.addWidget(self.RhoRadioButtonDefault)
        RhoHBox2.addWidget(self.RhoRadioButtonZhang)
        VBox3.addLayout(RhoHBox2)
        RhoHBox3 = QtWidgets.QHBoxLayout()
        RhoHBox3.addWidget(self.RhoRadoButton3C)
        RhoHBox3.addWidget(self.RhoRadioButtonYour)
        VBox3.addLayout(RhoHBox3)

        VBox3.addStretch()

        # Right Vertical box
        VBox4 = QtWidgets.QVBoxLayout()

        #   L2 NIR AtmoCorr
        NIRCorrectionHBox = QtWidgets.QHBoxLayout()
        NIRCorrectionHBox.addWidget(l2NIRCorrectionLabel)
        NIRCorrectionHBox.addWidget(self.l2NIRCorrectionCheckBox)
        VBox4.addLayout(NIRCorrectionHBox)
        VBox4.addWidget(self.SimpleNIRRadioButton)
        VBox4.addWidget(self.SimSpecNIRRadioButton)
        VBox4.addWidget(self.YourNIRRadioButton)

        VBox4.addSpacing(5)

        #   L2 Remove negative spectra
        NegativeSpecHBox = QtWidgets.QHBoxLayout()
        NegativeSpecHBox.addWidget(self.l2NegativeSpecLabel)
        NegativeSpecHBox.addWidget(self.l2NegativeSpecCheckBox)
        VBox4.addLayout(NegativeSpecHBox)

        #   L2 BRDF
        BRDFVBox = QtWidgets.QVBoxLayout()
        BRDFHBox1 = QtWidgets.QHBoxLayout()
        BRDFHBox1.addWidget(self.l2BRDFLabel)
        BRDFHBox1.addWidget(self.l2BRDFCheckBox)
        BRDFVBox.addLayout(BRDFHBox1)
        BRDFHBox2 = QtWidgets.QHBoxLayout()
        BRDFHBox2.addWidget(self.l2BRDF_fQLabel)
        BRDFHBox2.addWidget(self.l2BRDF_fQCheckBox)
        BRDFHBox2.addWidget(self.l2BRDF_IOPLabel)
        BRDFHBox2.addWidget(self.l2BRDF_IOPCheckBox)
        BRDFVBox.addLayout(BRDFHBox2)
        VBox4.addLayout(BRDFVBox)

        #   L2 Products
        VBox4.addWidget(l2ProductLabel)

        #   L2 Spectral weighting to satellites
        VBox4.addWidget(l2WeightsLabel)
        l2WeightHBox = QtWidgets.QHBoxLayout()
        l2WeightHBox.addSpacing(45)
        l2WeightHBox.addWidget(l2WeightMODISALabel)
        l2WeightHBox.addWidget(self.l2WeightMODISACheckBox)
        l2WeightHBox.addWidget(l2WeightSentinel3ALabel)
        l2WeightHBox.addWidget(self.l2WeightSentinel3ACheckBox)
        l2WeightHBox.addWidget(l2WeightVIIRSNLabel)
        l2WeightHBox.addWidget(self.l2WeightVIIRSNCheckBox)
        VBox4.addLayout(l2WeightHBox)
        l2WeightHBox2 = QtWidgets.QHBoxLayout()
        l2WeightHBox2.addSpacing(45)
        l2WeightHBox2.addWidget(l2WeightMODISTLabel)
        l2WeightHBox2.addWidget(self.l2WeightMODISTCheckBox)
        l2WeightHBox2.addWidget(l2WeightSentinel3BLabel)
        l2WeightHBox2.addWidget(self.l2WeightSentinel3BCheckBox)
        l2WeightHBox2.addWidget(l2WeightVIIRSJLabel)
        l2WeightHBox2.addWidget(self.l2WeightVIIRSJCheckBox)
        VBox4.addLayout(l2WeightHBox2)
        VBox4.addWidget(l2WeightMODISALabel2)

        VBox4.addSpacing(5)

        #   L2 Plotting
        VBox4.addWidget(l2PlotsLabel)
        l2PlotHBox = QtWidgets.QHBoxLayout()
        l2PlotHBox.addSpacing(45)
        l2PlotHBox.addWidget(l2PlotRrsLabel)
        l2PlotHBox.addWidget(self.l2PlotRrsCheckBox)
        l2PlotHBox.addWidget(l2PlotnLwLabel)
        l2PlotHBox.addWidget(self.l2PlotnLwCheckBox)
        l2PlotHBox.addWidget(l2PlotEsLabel)
        l2PlotHBox.addWidget(self.l2PlotEsCheckBox)
        l2PlotHBox.addWidget(l2PlotLiLabel)
        l2PlotHBox.addWidget(self.l2PlotLiCheckBox)
        l2PlotHBox.addWidget(l2PlotLtLabel)
        l2PlotHBox.addWidget(self.l2PlotLtCheckBox)
        VBox4.addLayout(l2PlotHBox)

        VBox4.addSpacing(5)

        l2OCproductsHBox = QtWidgets.QHBoxLayout()
        l2OCproductsHBox.addWidget(self.l2OCproducts)
        VBox4.addLayout(l2OCproductsHBox)

        #   Horizontal Box; Save SeaBASS
        l2SeaBASSHBox = QtWidgets.QHBoxLayout()
        l2SeaBASSHBox.addWidget(l2SaveSeaBASSLabel)
        l2SeaBASSHBox.addWidget(self.l2SaveSeaBASSCheckBox)
        VBox4.addLayout(l2SeaBASSHBox)
        l2SeaBASSHeaderHBox2 = QtWidgets.QHBoxLayout()
        l2SeaBASSHeaderHBox2.addWidget(self.l2SeaBASSHeaderEditButton)
        VBox4.addLayout(l2SeaBASSHeaderHBox2)
        VBox4.addWidget(self.l2SeaBASSHeaderLabel)

        #   Horizontal Box; Write Report
        l2ReportHBox = QtWidgets.QHBoxLayout()
        l2ReportHBox.addWidget(l2WriteReportLabel)
        l2ReportHBox.addWidget(self.l2WriteReportCheckBox)
        VBox4.addLayout(l2ReportHBox)

        VBox4.addSpacing(20)
        VBox4.addStretch()

        # Save/Cancel
        saveHBox = QtWidgets.QHBoxLayout()
        saveHBox.addStretch(1)
        saveHBox.addWidget(self.saveButton)
        saveHBox.addWidget(self.saveAsButton)
        saveHBox.addWidget(self.cancelButton)
        VBox4.addLayout(saveHBox)

        # Add 3 Vertical Boxes to Horizontal Box hBox
        hBox = QtWidgets.QHBoxLayout()
        hBox.addLayout(VBox1)
        hBox.addLayout(VBox2)
        hBox.addLayout(VBox3)
        hBox.addLayout(VBox4)

        VBox.addLayout(hBox)

        self.setLayout(VBox)
        # self.setGeometry(300, 100, 0, 0)
        self.setWindowTitle(f'Configuration: {self.name}')

    ###############################################################
    def addCalibrationFileButtonPressed(self):
        print("CalibrationEditWindow - Add Calibration File Pressed")
        fnames = QtWidgets.QFileDialog.getOpenFileNames(self, "Add Calibration Files")
        print(fnames)

        configName = self.name
        calibrationDir = os.path.splitext(configName)[0] + "_Calibration"
        configPath = os.path.join("Config", calibrationDir)

        if fnames:
            if ".sip" in fnames[0][0]:
                src = fnames[0][0]
                (_, filename) = os.path.split(src)
                dest = os.path.join(configPath, filename)
                print(src)
                print(dest)
                shutil.copy(src, dest)
                CalibrationFileReader.readSip(dest)
                [folder,_] = filename.split('.')
                os.rmdir(os.path.join(configPath,folder))

            else:
                for src in fnames[0]:
                    (_, filename) = os.path.split(src)
                    dest = os.path.join(configPath, filename)
                    print(src)
                    print(dest)
                    shutil.copy(src, dest)

    def deleteCalibrationFileButtonPressed(self):
        print("CalibrationEditWindow - Remove Calibration File Pressed")
        configName = self.name
        calibrationDir = os.path.splitext(configName)[0] + "_Calibration"
        configPath = os.path.join("Config", calibrationDir)
        os.remove(os.path.join(configPath,self.calibrationFileComboBox.currentText()))


    def getCalibrationSettings(self):
        print("CalibrationEditWindow - getCalibrationSettings")
        ConfigFile.refreshCalibrationFiles()
        calFileName = self.calibrationFileComboBox.currentText()
        calConfig = ConfigFile.getCalibrationConfig(calFileName)
        self.calibrationEnabledCheckBox.blockSignals(True)
        self.calibrationFrameTypeComboBox.blockSignals(True)

        self.calibrationEnabledCheckBox.setChecked(bool(calConfig["enabled"]))
        index = self.calibrationFrameTypeComboBox.findText(str(calConfig["frameType"]))
        self.calibrationFrameTypeComboBox.setCurrentIndex(index)

        self.calibrationEnabledCheckBox.blockSignals(False)
        self.calibrationFrameTypeComboBox.blockSignals(False)


    def sensorTypeChanged(self):
        print("CalibrationEditWindow - Sensor Type Changed")
        sensor = self.sensorTypeComboBox.currentText()
        ConfigFile.settings["SensorType"] = sensor

        self.l1aqcDeglitchCheckBoxUpdate()


    def setCalibrationSettings(self):
        print("CalibrationEditWindow - setCalibrationSettings")
        calFileName = self.calibrationFileComboBox.currentText()
        enabled = self.calibrationEnabledCheckBox.isChecked()
        frameType = self.calibrationFrameTypeComboBox.currentText()
        ConfigFile.setCalibrationConfig(calFileName, enabled, frameType)


    def calibrationFileChanged(self, i):
        print("CalibrationEditWindow - Calibration File Changed")
        print("Current index",i,"selection changed ", self.calibrationFileComboBox.currentText())
        calFileName = self.calibrationFileComboBox.currentText()
        calDir = ConfigFile.getCalibrationDirectory()
        calPath = os.path.join(calDir, calFileName)
        #print("calPath: " + calPath)
        if os.path.isfile(calPath):
            self.getCalibrationSettings()
            self.calibrationEnabledCheckBox.setEnabled(True)
            self.calibrationFrameTypeComboBox.setEnabled(True)
        else:
            self.calibrationEnabledCheckBox.setEnabled(False)
            self.calibrationFrameTypeComboBox.setEnabled(False)


    def calibrationEnabledStateChanged(self):
        print("CalibrationEditWindow - Calibration Enabled State Changed")
        print(self.calibrationEnabledCheckBox.isChecked())
        self.setCalibrationSettings()

    def calibrationFrameTypeChanged(self, i):
        print("CalibrationEditWindow - Calibration Frame Type Changed")
        print("Current index",i,"selection changed ", self.calibrationFrameTypeComboBox.currentText())
        self.setCalibrationSettings()

    def l1aCleanSZACheckBoxUpdate(self):
        print("ConfigWindow - l1aCleanSZAAngleCheckBoxUpdate")

        disabled = (not self.l1aCleanSZACheckBox.isChecked())
        self.l1aCleanSZAMaxLineEdit.setDisabled(disabled)
        if disabled:
            ConfigFile.settings["bL1aCleanSZA"] = 0
        else:
            ConfigFile.settings["bL1aCleanSZA"] = 1

    def l1aqcSolarTrackerCheckBoxUpdate(self):
        print("ConfigWindow - l1aqcSolarTrackerCheckBoxUpdate")

        disabled = (not self.l1aqcSolarTrackerCheckBox.isChecked())
        self.l1aCleanSZAMaxLabel.setDisabled(disabled)
        self.l1aCleanSZACheckBox.setDisabled(disabled)
        self.l1aCleanSZAMaxLineEdit.setDisabled(disabled)
        self.l1aqcRotatorDelayLabel.setDisabled(disabled)
        self.l1aqcRotatorDelayLineEdit.setDisabled(disabled)
        self.l1aqcRotatorDelayCheckBox.setDisabled(disabled)
        self.l1aqcRotatorAngleLabel.setDisabled(disabled)
        self.l1aqcRotatorAngleCheckBox.setDisabled(disabled)
        self.l1aqcRotatorAngleMinLabel.setDisabled(disabled)
        self.l1aqcRotatorAngleMinLineEdit.setDisabled(disabled)
        self.l1aqcRotatorAngleMaxLabel.setDisabled(disabled)
        self.l1aqcRotatorAngleMaxLineEdit.setDisabled(disabled)
        if disabled:
            ConfigFile.settings["fL1aCleanSZAMax"] = 90
            ConfigFile.settings["bL1aqcSolarTracker"] = 0
            ConfigFile.settings["bL1aqcRotatorDelay"] = 0
            self.l1aqcRotatorDelayCheckBox.setChecked(False)
            self.l1aqcRotatorAngleCheckBox.setChecked(False)
        else:
            ConfigFile.settings["bL1aqcSolarTracker"] = 1


    def l1aqcRotatorDelayCheckBoxUpdate(self):
        print("ConfigWindow - l1aqcRotatorDelayCheckBoxUpdate")

        disabled = (not self.l1aqcRotatorDelayCheckBox.isChecked())
        self.l1aqcRotatorDelayLineEdit.setDisabled(disabled)
        if disabled:
            ConfigFile.settings["bL1aqcRotatorDelay"] = 0
        else:
            ConfigFile.settings["bL1aqcRotatorDelay"] = 1

    def l1aqcCleanPitchRollCheckBoxUpdate(self):
        print("ConfigWindow - l1aqcCleanPitchRollCheckBoxUpdate")

        disabled = (not self.l1aqcCleanPitchRollCheckBox.isChecked())
        self.l1aqcPitchRollPitchLabel.setDisabled(disabled)
        self.l1aqcPitchRollPitchLineEdit.setDisabled(disabled)
        if disabled:
            ConfigFile.settings["bL1aqcCleanPitchRoll"] = 0
        else:
            ConfigFile.settings["bL1aqcCleanPitchRoll"] = 1

    def l1aqcRotatorAngleCheckBoxUpdate(self):
        print("ConfigWindow - l1aqcRotatorAngleCheckBoxUpdate")

        disabled = (not self.l1aqcRotatorAngleCheckBox.isChecked())
        self.l1aqcRotatorAngleMinLabel.setDisabled(disabled)
        self.l1aqcRotatorAngleMinLineEdit.setDisabled(disabled)
        self.l1aqcRotatorAngleMaxLabel.setDisabled(disabled)
        self.l1aqcRotatorAngleMaxLineEdit.setDisabled(disabled)
        if disabled:
            ConfigFile.settings["bL1aqcRotatorAngle"] = 0
        else:
            ConfigFile.settings["bL1aqcRotatorAngle"] = 1

    def l1aqcCleanSunAngleCheckBoxUpdate(self):
        print("ConfigWindow - l1aqcCleanSunAngleCheckBoxUpdate")

        disabled = (not self.l1aqcCleanSunAngleCheckBox.isChecked())
        self.l1aqcSunAngleMinLabel.setDisabled(disabled)
        self.l1aqcSunAngleMinLineEdit.setDisabled(disabled)
        self.l1aqcSunAngleMaxLabel.setDisabled(disabled)
        self.l1aqcSunAngleMaxLineEdit.setDisabled(disabled)
        if disabled:
            ConfigFile.settings["bL1aqcCleanSunAngle"] = 0
        else:
            ConfigFile.settings["bL1aqcCleanSunAngle"] = 1

    def l1aqcDeglitchCheckBoxUpdate(self):
        print("ConfigWindow - l1aqcDeglitchCheckBoxUpdate")

        # Confirm SeaBird
        sensor = self.sensorTypeComboBox.currentText()
        ConfigFile.settings["SensorType"] = sensor

        if sensor.lower() == 'trios':
            self.l1aqcDeglitchCheckBox.setChecked(False)
            self.l1aqcDeglitchCheckBox.setEnabled(False)
            self.l1aqcDeglitchLabel.setEnabled(False)
            self.l1aqcAnomalyButton.setEnabled(False)
        elif sensor.lower() == 'seabird':
            self.l1aqcDeglitchCheckBox.setEnabled(True)
            self.l1aqcDeglitchLabel.setEnabled(True)
            self.l1aqcAnomalyButton.setEnabled(True)

        disabled = (not self.l1aqcDeglitchCheckBox.isChecked())
        if disabled:
            ConfigFile.settings["bL1dDeglitch"] = 0
        else:
            ConfigFile.settings["bL1dDeglitch"] = 1

    def l1aqcAnomalyButtonPressed(self):
        print("CalibrationEditWindow - Launching anomaly analysis module")
        ConfigWindow.refreshConfig(self)
        anomAnalDialog = AnomAnalWindow(self.inputDirectory, self)
        anomAnalDialog.show()

    def l1bDefaultCalRadioButtonClicked(self):
        print("ConfigWindow - L1b Calibration set to Factory")
        self.DefaultCalRadioButton.setChecked(True)
        self.ClassCalRadioButton.setChecked(False)
        self.FullCalRadioButton.setChecked(False)
        ConfigFile.settings["bL1bCal"] = 1

    def l1bClassCalRadioButtonClicked(self):
        print("ConfigWindow - L1b Calibration set to Class-based")
        self.DefaultCalRadioButton.setChecked(False)
        self.ClassCalRadioButton.setChecked(True)
        self.FullCalRadioButton.setChecked(False)
        ConfigFile.settings["bL1bCal"] = 2

    def l1bFullCalRadioButtonClicked(self):
        print("ConfigWindow - L1b Calibration set to Instrument-specific FRM")
        self.DefaultCalRadioButton.setChecked(False)
        self.ClassCalRadioButton.setChecked(False)
        self.FullCalRadioButton.setChecked(True)
        ConfigFile.settings["bL1bCal"] = 3

    def l1bGetFRMCheckUpdate1(self):
        print("ConfigWindow - l1bGetFRMCheckUpdate local files")
        if self.l1bGetFRMCheck1.isChecked():
            self.l1bGetFRMCheck2.setChecked(False)
            ConfigFile.settings['FidRadDB'] = 0
            self.FullCalDir = ConfigFile.settings['FullCalDir']
            self.FullCalDir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Directory.', self.FullCalDir)
            print('Full characterization directory changed: ', self.FullCalDir)
            ConfigFile.settings['FullCalDir'] = self.FullCalDir
            return self.FullCalDir

    def l1bGetFRMCheckUpdate2(self):
        print("ConfigWindow - l1bGetFRMCheckUpdate FidRadDB")
        if self.l1bGetFRMCheck2.isChecked():
            self.l1bGetFRMCheck1.setChecked(False)
            ConfigFile.settings['FidRadDB'] = 1


    def FullCalDirButtonPressed(self):
        if not ConfigFile.settings['FullCalDir'].startswith('Choose'):
            srcDir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Directory', ConfigFile.settings['FullCalDir'])
        else:
            srcDir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose Directory')
        print('Full characterization folders selected for copy: ', srcDir)

        configName = self.name
        calibrationDir = os.path.splitext(configName)[0] + "_Calibration"
        configPath = os.path.join("Config", calibrationDir)

        calDir = Path(srcDir)
        # files = Path(calDir).glob("CP*.TXT")
        files = glob.iglob(os.path.join(Path(calDir), '*.[tT][xX][tT]'))
        for file in files:
            # dest = Path(configPath) / file.name
            dest = Path(configPath) / os.path.basename(file)
            if not dest.exists():
                shutil.copy(file,dest)

        self.FullCalDirButton.setText(os.path.basename(calibrationDir))
        ConfigFile.settings['FullCalDir'] = configPath


    def l1bPlotTimeInterpCheckBoxUpdate(self):
        print("ConfigWindow - l1bPlotTimeInterpCheckBoxUpdate")
        if self.l1bPlotTimeInterpCheckBox.isChecked():
            ConfigFile.settings["bL1bPlotTimeInterp"] = 1
        else:
            ConfigFile.settings["bL1bPlotTimeInterp"] = 0

    def l1bqcLtUVNIRCheckBoxUpdate(self):
        print("ConfigWindow - l2UVNIRCheckBoxUpdate")

        if self.l1bqcLtUVNIRCheckBox.isChecked():
            ConfigFile.settings["bL1bqcLtUVNIR"] = 1
        else:
            ConfigFile.settings["bL1bqcLtUVNIR"] = 0

    def l1bqcSpecQualityCheckBoxUpdate(self):
        print("ConfigWindow - l1bqcSpecQualityCheckBoxUpdate")

        disabled = (not self.l1bqcSpecQualityCheckBox.isChecked())
        self.l1bqcSpecFilterLiLabel.setDisabled(disabled)
        self.l1bqcSpecFilterLiLineEdit.setDisabled(disabled)
        self.l1bqcSpecFilterLtLabel.setDisabled(disabled)
        self.l1bqcSpecFilterLtLineEdit.setDisabled(disabled)
        self.l1bqcSpecFilterEsLabel.setDisabled(disabled)
        self.l1bqcSpecFilterEsLineEdit.setDisabled(disabled)
        if disabled:
            ConfigFile.settings["bL2EnableSpecQualityCheck"] = 0
        else:
            ConfigFile.settings["bL2EnableSpecQualityCheck"] = 1

    def l1bqcQualityFlagCheckBoxUpdate(self):
        print("ConfigWindow - l1bqcQualityFlagCheckBoxUpdate")

        disabled = (not self.l1bqcQualityFlagCheckBox.isChecked())
        self.l1bqcCloudFlagLabel.setDisabled(disabled)
        self.l1bqcCloudFlagLineEdit.setDisabled(disabled)
        self.l1bqcEsFlagLabel.setDisabled(disabled)
        self.l1bqcEsFlagLineEdit.setDisabled(disabled)
        self.l1bqcDawnDuskFlagLabel.setDisabled(disabled)
        self.l1bqcDawnDuskFlagLineEdit.setDisabled(disabled)
        self.l1bqcRainfallHumidityFlagLabel.setDisabled(disabled)
        self.l1bqcRainfallHumidityFlagLineEdit.setDisabled(disabled)
        if disabled:
            ConfigFile.settings["bL2EnableQualityFlags"] = 0
        else:
            ConfigFile.settings["bL2EnableQualityFlags"] = 1

    def l1bGetAncCheckBoxUpdate1(self):
        print("ConfigWindow - l1bGetAncCheckBoxUpdate GMAO MERRA2")
        if self.l1bGetAncCheckBox1.isChecked():
            self.l1bGetAncCheckBox2.setChecked(False)
            if not ConfigFile.settings["bL1bObpgCreds"]:
                usr = QtWidgets.QInputDialog.getText(None,
                                                "Earthdata Username",
                                                "Username (Cancel to use current credentials):",
                                                QtWidgets.QLineEdit.Normal,
                                                "")
                if usr[1]:
                    pwd = QtWidgets.QInputDialog.getText(None,
                                                    "Earthdata Password",
                                                    "Password:",
                                                    QtWidgets.QLineEdit.Normal,
                                                    "")
                    GetAnc.userCreds(usr[0],pwd[0])
                else:
                    # If the user cancels out of these, presume their account is
                    # already set up properly and skip netrc file creation.
                    print('Credentials skipped. Will try to use current credentials.')
                    ConfigFile.settings["bL1bpObpgCreds"] = 1

            ConfigFile.settings["bL1bpGetAnc"] = 1
            # self.RhoRadoButton3C.setDisabled(0)
            self.RhoRadioButtonZhang.setDisabled(0)
        else:
            ConfigFile.settings["bL1bpGetAnc"] = 0
            ConfigFile.settings["bL1bpObpgCreds"] = 0
            self.RhoRadioButtonZhang.setChecked(0)
            self.RhoRadioButtonZhang.setDisabled(1)
            # self.RhoRadoButton3C.setChecked(1)
            self.RhoRadioButtonDefault.setChecked(1)

            print("ConfigWindow - l2RhoCorrection set to M99")
            ConfigFile.settings["bL23CRho"] = 0
            ConfigFile.settings["bL2ZhangRho"] = 0
            ConfigFile.settings["bL2DefaultRho"] = 1

    def l1bGetAncCheckBoxUpdate2(self):
        print("ConfigWindow - l1bGetAncCheckBoxUpdate ECMWF")
        if self.l1bGetAncCheckBox2.isChecked():
            self.l1bGetAncCheckBox1.setChecked(False)
            ConfigFile.settings["bL1bpGetAnc"] = 2
            # self.RhoRadoButton3C.setDisabled(0)
            self.RhoRadioButtonZhang.setDisabled(0)
        else:
            ConfigFile.settings["bL1bpGetAnc"] = 0
            ConfigFile.settings["bL1bpObpgCreds"] = 0
            self.RhoRadioButtonZhang.setChecked(0)
            self.RhoRadioButtonZhang.setDisabled(1)
            # self.RhoRadoButton3C.setChecked(1)
            self.RhoRadioButtonDefault.setChecked(1)
            print("ConfigWindow - l2RhoCorrection set to M99")
            ConfigFile.settings["bL23CRho"] = 0
            ConfigFile.settings["bL2ZhangRho"] = 0
            ConfigFile.settings["bL2DefaultRho"] = 1

    def l2StationsCheckBoxUpdate(self):
        print("ConfigWindow - l2StationsCheckBoxUpdate")

        disabled = (not self.l2StationsCheckBox.isChecked())
        if disabled:
            ConfigFile.settings["bL2Stations"] = 0
        else:
            ConfigFile.settings["bL2Stations"] = 1

    def l2EnablePercentLtCheckBoxUpdate(self):
        print("ConfigWindow - l2EnablePercentLtCheckBoxUpdate")

        disabled = (not self.l2EnablePercentLtCheckBox.isChecked())
        self.l2PercentLtLabel.setDisabled(disabled)
        self.l2PercentLtLineEdit.setDisabled(disabled)
        if disabled:
            ConfigFile.settings["bL2EnablePercentLt"] = 0
        else:
            ConfigFile.settings["bL2EnablePercentLt"] = 1

    def l2RhoRadoButton3CClicked(self):
        print("ConfigWindow - l2RhoCorrection set to Ruddick")
        self.RhoRadoButton3C.setChecked(True)
        self.RhoRadioButtonZhang.setChecked(False)
        self.RhoRadioButtonDefault.setChecked(False)
        ConfigFile.settings["bL23CRho"] = 1
        ConfigFile.settings["bL2ZhangRho"] = 0
        ConfigFile.settings["bL2DefaultRho"] = 0
    def l2RhoRadioButtonZhangClicked(self):
        print("ConfigWindow - l2RhoCorrection set to Zhang")
        self.RhoRadoButton3C.setChecked(False)
        self.RhoRadioButtonZhang.setChecked(True)
        self.RhoRadioButtonDefault.setChecked(False)
        ConfigFile.settings["bL23CRho"] = 0
        ConfigFile.settings["bL2ZhangRho"] = 1
        ConfigFile.settings["bL2DefaultRho"] = 0
        if ConfigFile.settings["fL1bqcSZAMax"] > 60:
            print("SZA outside model limits; adjusting to 60")
            ConfigFile.settings["fL1bqcSZAMax"] = 60
            self.l1bqcSZAMaxLineEdit.setText(str(60.0))
    def l2RhoRadioButtonDefaultClicked(self):
        print("ConfigWindow - l2RhoCorrection set to Default")
        self.RhoRadoButton3C.setChecked(False)
        self.RhoRadioButtonZhang.setChecked(False)
        self.RhoRadioButtonDefault.setChecked(True)
        ConfigFile.settings["bL23CRho"] = 0
        ConfigFile.settings["bL2ZhangRho"] = 0
        ConfigFile.settings["bL2DefaultRho"] = 1
    def l2RhoRadioButtonYourClicked(self):
        print("ConfigWindow - l2RhoCorrection set to Default. You have not submitted your method.")
        self.RhoRadoButton3C.setChecked(False)
        self.RhoRadioButtonZhang.setChecked(False)
        self.RhoRadioButtonYour.setChecked(True)
        ConfigFile.settings["bL23CRho"] = 0
        ConfigFile.settings["bL2ZhangRho"] = 0
        ConfigFile.settings["bL2DefaultRho"] = 1 # This is a mock up. Use Default

    def l2SimpleNIRRadioButtonClicked(self):
        print("ConfigWindow - l2NIRCorrection set to Simple")
        self.SimpleNIRRadioButton.setChecked(True)
        self.SimSpecNIRRadioButton.setChecked(False)
        self.YourNIRRadioButton.setChecked(False)
        ConfigFile.settings["bL2SimpleNIRCorrection"] = 1
        ConfigFile.settings["bL2SimSpecNIRCorrection"] = 0
    def l2SimSpecNIRRadioButtonClicked(self):
        print("ConfigWindow - l2NIRCorrection set to SimSpec")
        self.SimpleNIRRadioButton.setChecked(False)
        self.SimSpecNIRRadioButton.setChecked(True)
        self.YourNIRRadioButton.setChecked(False)
        ConfigFile.settings["bL2SimpleNIRCorrection"] = 0
        ConfigFile.settings["bL2SimSpecNIRCorrection"] = 1
    def l2YourNIRRadioButtonClicked(self):
        print("ConfigWindow - l2NIRCorrection set to Simple. You have not submitted Your method.")
        self.SimpleNIRRadioButton.setChecked(True)
        self.SimSpecNIRRadioButton.setChecked(False)
        self.YourNIRRadioButton.setChecked(True)
        ConfigFile.settings["bL2SimpleNIRCorrection"] = 1 # Mock up. Use Simple
        ConfigFile.settings["bL2SimSpecNIRCorrection"] = 0
    def l2NIRCorrectionCheckBoxUpdate(self):
        print("ConfigWindow - l2NIRCorrectionCheckBoxUpdate")
        disabled = (not self.l2NIRCorrectionCheckBox.isChecked())
        self.SimpleNIRRadioButton.setDisabled(disabled)
        self.SimSpecNIRRadioButton.setDisabled(disabled)
        self.YourNIRRadioButton.setDisabled(disabled)
        if disabled:
            ConfigFile.settings["bL2PerformNIRCorrection"] = 0
        else:
            ConfigFile.settings["bL2PerformNIRCorrection"] = 1

    def l2NegativeSpecCheckBoxUpdate(self):
        print("ConfigWindow - l2NegativeSpecCheckBoxUpdate")

        disabled = (not self.l2NegativeSpecCheckBox.isChecked())
        if disabled:
            ConfigFile.settings["bL2NegativeSpec"] = 0
        else:
            ConfigFile.settings["bL2NegativeSpec"] = 1

    def l2BRDFCheckBoxUpdate(self):
        print("ConfigWindow - l2BRDFCheckBoxUpdate")

        disabled = (not self.l2BRDFCheckBox.isChecked())
        self.l2BRDF_fQCheckBox.setDisabled(disabled)
        self.l2BRDF_fQLabel.setDisabled(disabled)
        # self.l2BRDF_IOPCheckBox.setDisabled(disabled)
        self.l2BRDF_IOPCheckBox.setDisabled(True)
        # self.l2BRDF_IOPLabel.setDisabled(disabled)
        self.l2BRDF_IOPLabel.setDisabled(True)

        if disabled:
            ConfigFile.settings["bL2BRDF"] = 0
            ConfigFile.settings["bL2BRDF_fQ"] = 0
            ConfigFile.settings["bL2BRDF_IOP"] = 0
            self.l2BRDF_fQCheckBox.setChecked(False)
            self.l2BRDF_IOPCheckBox.setChecked(False)
        else:
            ConfigFile.settings["bL2BRDF"] = 1

    def l2BRDF_fQCheckBoxUpdate(self):
        print("ConfigWindow - l2BRDF_fQCheckBoxUpdate")

        disabled = (not self.l2BRDF_fQCheckBox.isChecked())
        if disabled:
            ConfigFile.settings["bL2BRDF_fQ"] = 0
        else:
            ConfigFile.settings["bL2BRDF_fQ"] = 1
            # This will require chlor_a in derived products to be turned on
            # which in turn requires MODIS bands...
            self.l2WeightMODISACheckBox.setChecked(True)
            ConfigFile.products["bL2Prodoc3m"] = 1


    def l2BRDF_IOPCheckBoxUpdate(self):
        print("ConfigWindow - l2BRDF_IOPCheckBoxUpdate")

        disabled = (not self.l2BRDF_IOPCheckBox.isChecked())
        if disabled:
            ConfigFile.settings["bL2BRDF_IOP"] = 0
        else:
            ConfigFile.settings["bL2BRDF_IOP"] = 1


    def l2OCproductsButtonPressed(self):
        print("OC Products Dialogue")

        ConfigWindow.refreshConfig(self)
        OCproductsDialog = OCproductsWindow(self)
        OCproductsDialog.exec()

        if int(ConfigFile.settings["bL2WeightMODISA"]) == 1:
            self.l2WeightMODISACheckBox.setChecked(True)


    def l2SaveSeaBASSCheckBoxUpdate(self):
        print("ConfigWindow - l2SaveSeaBASSCheckBoxUpdate")
        disabled = (not self.l2SaveSeaBASSCheckBox.isChecked())

        self.l2SeaBASSHeaderEditButton.setDisabled(disabled)

    def l2SeaBASSHeaderEditButtonPressed(self):
        print("Edit seaBASSHeader Dialogue")

        ConfigWindow.refreshConfig(self)
        seaBASSHeaderFileName = ConfigFile.settings["seaBASSHeaderFileName"]
        inputDir = self.inputDirectory
        seaBASSHeaderPath = os.path.join("Config", seaBASSHeaderFileName)
        if os.path.isfile(seaBASSHeaderPath):
            SeaBASSHeader.loadSeaBASSHeader(seaBASSHeaderFileName)
            # Update comments to reflect any changes in ConfigWindow
            SeaBASSHeaderWindow.configUpdateButtonPressed(self, 'config')
            seaBASSHeaderDialog = SeaBASSHeaderWindow(seaBASSHeaderFileName, inputDir, self)
            seaBASSHeaderDialog.show()
        else:
            print("Creating New SeaBASSHeader File: ", seaBASSHeaderFileName)
            SeaBASSHeader.createDefaultSeaBASSHeader(seaBASSHeaderFileName)
            SeaBASSHeader.loadSeaBASSHeader(seaBASSHeaderFileName)
            seaBASSHeaderDialog = SeaBASSHeaderWindow(seaBASSHeaderFileName, inputDir, self)
            seaBASSHeaderDialog.show()
            # print("SeaBASS Header file lost. Please restore to Config directory or recreate.")
        self.l2SeaBASSHeaderLabel.setText(f'  {ConfigFile.settings["seaBASSHeaderFileName"]}')

    def l2WriteReportCheckBoxUpdate(self):
        print("ConfigWindow - l2WriteReportCheckBoxUpdate")
        # disabled = not self.l2WriteReportCheckBox.isChecked()

    def saveButtonPressed(self):
        print("ConfigWindow - Save Pressed")

        ConfigWindow.refreshConfig(self)

        ConfigFile.saveConfig(self.name)

        # Confirm that SeaBASS Headers need to be/are updated
        SeaBASSHeader.loadSeaBASSHeader(ConfigFile.settings["seaBASSHeaderFileName"])
        # This now updates the SeaBASS Header comments to reflect the ConfigWindow parameters automatically.
        SeaBASSHeaderWindow.configUpdateButtonPressed(self, 'config')
        SeaBASSHeader.saveSeaBASSHeader(ConfigFile.settings["seaBASSHeaderFileName"])

        self.close()

    def refreshConfig(self):
        print("ConfigWindow - refreshConfig")

        ConfigFile.settings["fL1aUTCOffset"] = float(self.l1aUTCOffsetLineEdit.text())
        ConfigFile.settings["bL1aCleanSZA"] = int(self.l1aCleanSZACheckBox.isChecked())
        ConfigFile.settings["fL1aCleanSZAMax"] = float(self.l1aCleanSZAMaxLineEdit.text())

        ConfigFile.settings["bL1aqcSolarTracker"] = int(self.l1aqcSolarTrackerCheckBox.isChecked())
        ConfigFile.settings["fL1aqcRotatorHomeAngle"] = float(self.l1aqcRotatorHomeAngleLineEdit.text())
        ConfigFile.settings["bL1aqcRotatorDelay"] = int(self.l1aqcRotatorDelayCheckBox.isChecked())
        ConfigFile.settings["fL1aqcRotatorDelay"] = float(self.l1aqcRotatorDelayLineEdit.text())
        ConfigFile.settings["bL1aqcCleanPitchRoll"] = int(self.l1aqcCleanPitchRollCheckBox.isChecked())
        ConfigFile.settings["fL1aqcPitchRollPitch"] = float(self.l1aqcPitchRollPitchLineEdit.text())
        ConfigFile.settings["fL1aqcPitchRollRoll"] = float(self.l1aqcPitchRollPitchLineEdit.text())
        ConfigFile.settings["bL1aqcRotatorAngle"] = int(self.l1aqcRotatorAngleCheckBox.isChecked())
        ConfigFile.settings["fL1aqcRotatorAngleMin"] = float(self.l1aqcRotatorAngleMinLineEdit.text())
        ConfigFile.settings["fL1aqcRotatorAngleMax"] = float(self.l1aqcRotatorAngleMaxLineEdit.text())
        ConfigFile.settings["bL1aqcCleanSunAngle"] = int(self.l1aqcCleanSunAngleCheckBox.isChecked())
        ConfigFile.settings["fL1aqcSunAngleMin"] = float(self.l1aqcSunAngleMinLineEdit.text())
        ConfigFile.settings["fL1aqcSunAngleMax"] = float(self.l1aqcSunAngleMaxLineEdit.text())

        ConfigFile.settings["bL1aqcDeglitch"] = int(self.l1aqcDeglitchCheckBox.isChecked())

        if self.l1bGetAncCheckBox1.isChecked():
            ConfigFile.settings["bL1bGetAnc"] = 1
        elif self.l1bGetAncCheckBox2.isChecked():
            ConfigFile.settings["bL1bGetAnc"] = 2
        else:
            ConfigFile.settings["bL1bGetAnc"] = 0
        ConfigFile.settings["fL1bDefaultWindSpeed"] = float(self.l1bDefaultWindSpeedLineEdit.text())
        ConfigFile.settings["fL1bDefaultAOD"] = float(self.l1bDefaultAODLineEdit.text())
        ConfigFile.settings["fL1bDefaultSalt"] = float(self.l1bDefaultSaltLineEdit.text())
        ConfigFile.settings["fL1bDefaultSST"] = float(self.l1bDefaultSSTLineEdit.text())
        ConfigFile.settings["fL1bInterpInterval"] = float(self.l1bInterpIntervalLineEdit.text())
        ConfigFile.settings["bL1bPlotTimeInterp"] = int(self.l1bPlotTimeInterpCheckBox.isChecked())
        ConfigFile.settings["fL1bPlotInterval"] = float(self.l1bPlotIntervalLineEdit.text())

        ConfigFile.settings["bL1bqcLtUVNIR"] = int(self.l1bqcLtUVNIRCheckBox.isChecked())
        ConfigFile.settings["fL1bqcMaxWind"] = float(self.l1bqcMaxWindLineEdit.text())
        ConfigFile.settings["fL1bqcSZAMin"] = float(self.l1bqcSZAMinLineEdit.text())
        if int(self.RhoRadioButtonZhang.isChecked()) and float(self.l1bqcSZAMaxLineEdit.text()) > 60:
            print("SZA outside Zhang model limits; adjusting.")
            self.l1bqcSZAMaxLineEdit.setText(str(60.0))
        ConfigFile.settings["fL1bqcSZAMax"] = float(self.l1bqcSZAMaxLineEdit.text())
        ConfigFile.settings["bL1bqcEnableSpecQualityCheck"] = int(self.l1bqcSpecQualityCheckBox.isChecked())
        ConfigFile.settings["fL1bqcSpecFilterEs"] = float(self.l1bqcSpecFilterEsLineEdit.text())
        ConfigFile.settings["fL1bqcSpecFilterLi"] = float(self.l1bqcSpecFilterLiLineEdit.text())
        ConfigFile.settings["fL1bqcSpecFilterLt"] = float(self.l1bqcSpecFilterLtLineEdit.text())

        ConfigFile.settings["bL1bqcEnableQualityFlags"] = int(self.l1bqcQualityFlagCheckBox.isChecked())
        ConfigFile.settings["fL1bqcCloudFlag"] = float(self.l1bqcCloudFlagLineEdit.text())
        ConfigFile.settings["fL1bqcSignificantEsFlag"] = float(self.l1bqcEsFlagLineEdit.text())
        ConfigFile.settings["fL1bqcDawnDuskFlag"] = float(self.l1bqcDawnDuskFlagLineEdit.text())
        ConfigFile.settings["fL1bqcRainfallHumidityFlag"] = float(self.l1bqcRainfallHumidityFlagLineEdit.text())

        ConfigFile.settings["bL2Stations"] = int(self.l2StationsCheckBox.isChecked())
        ConfigFile.settings["fL2TimeInterval"] = int(self.l2TimeIntervalLineEdit.text())
        ConfigFile.settings["bL2EnablePercentLt"] = int(self.l2EnablePercentLtCheckBox.isChecked())
        ConfigFile.settings["fL2PercentLt"] = float(self.l2PercentLtLineEdit.text())
        # ConfigFile.settings["fL2RhoSky"] = float(self.l2RhoSkyLineEdit.text())
        ConfigFile.settings["bL23CRho"] = int(self.RhoRadoButton3C.isChecked())
        ConfigFile.settings["bL2ZhangRho"] = int(self.RhoRadioButtonZhang.isChecked())
        ConfigFile.settings["bL2DefaultRho"] = int(self.RhoRadioButtonDefault.isChecked())

        ConfigFile.settings["bL2PerformNIRCorrection"] = int(self.l2NIRCorrectionCheckBox.isChecked())
        ConfigFile.settings["bL2SimpleNIRCorrection"] = int(self.SimpleNIRRadioButton.isChecked())
        ConfigFile.settings["bL2SimSpecNIRCorrection"] = int(self.SimSpecNIRRadioButton.isChecked())

        ConfigFile.settings["bL2NegativeSpec"] = int(self.l2NegativeSpecCheckBox.isChecked())

        ConfigFile.settings["bL2BRDF"] = int(self.l2BRDFCheckBox.isChecked())
        ConfigFile.settings["bL2BRDF_fQ"] = int(self.l2BRDF_fQCheckBox.isChecked())
        ConfigFile.settings["bL2BRDF_IOP"] = int(self.l2BRDF_IOPCheckBox.isChecked())

        ConfigFile.settings["bL2WeightMODISA"] = int(self.l2WeightMODISACheckBox.isChecked())
        ConfigFile.settings["bL2WeightSentinel3A"] = int(self.l2WeightSentinel3ACheckBox.isChecked())
        ConfigFile.settings["bL2WeightVIIRSN"] = int(self.l2WeightVIIRSNCheckBox.isChecked())
        ConfigFile.settings["bL2WeightMODIST"] = int(self.l2WeightMODISTCheckBox.isChecked())
        ConfigFile.settings["bL2WeightSentinel3B"] = int(self.l2WeightSentinel3BCheckBox.isChecked())
        ConfigFile.settings["bL2WeightVIIRSJ"] = int(self.l2WeightVIIRSJCheckBox.isChecked())
        ConfigFile.settings["bL2PlotRrs"] = int(self.l2PlotRrsCheckBox.isChecked())
        ConfigFile.settings["bL2PlotnLw"] = int(self.l2PlotnLwCheckBox.isChecked())
        ConfigFile.settings["bL2PlotEs"] = int(self.l2PlotEsCheckBox.isChecked())
        ConfigFile.settings["bL2PlotLi"] = int(self.l2PlotLiCheckBox.isChecked())
        ConfigFile.settings["bL2PlotLt"] = int(self.l2PlotLtCheckBox.isChecked())
        ConfigFile.settings["bL2SaveSeaBASS"] = int(self.l2SaveSeaBASSCheckBox.isChecked())
        ConfigFile.settings["bL2WriteReport"] = int(self.l2WriteReportCheckBox.isChecked())

        # Confirm necessary satellite bands are processed
        if ConfigFile.products["bL2Prodoc3m"] or ConfigFile.products["bL2Prodkd490"] or \
            ConfigFile.products["bL2Prodpic"] or ConfigFile.products["bL2Prodpoc"] or \
                ConfigFile.products["bL2Prodgocad"] or ConfigFile.products["bL2Prodgiop"] or \
                ConfigFile.products["bL2Prodqaa"]:

            ConfigFile.settings["bL2WeightMODISA"] = 1
            self.l2WeightMODISACheckBox.setChecked(True)


    def saveAsButtonPressed(self):
        print("ConfigWindow - Save As Pressed")
        self.newName, ok = QtWidgets.QInputDialog.getText(self, 'Save As Config File', 'Enter File Name')
        if ok:
            print("Create Config File: ", self.newName)

            if not self.newName.endswith(".cfg"):
                self.newName = self.newName + ".cfg"
            ConfigFile.filename = self.newName

            ConfigWindow.refreshConfig(self)
            ConfigFile.saveConfig(ConfigFile.filename)

            # Copy Calibration files into new Config folder
            fnames = ConfigFile.settings['CalibrationFiles']
            oldConfigName = self.name
            newConfigName = ConfigFile.filename
            oldCalibrationDir = os.path.splitext(oldConfigName)[0] + "_Calibration"
            newCalibrationDir = os.path.splitext(newConfigName)[0] + "_Calibration"
            oldConfigPath = os.path.join("Config", oldCalibrationDir)
            newConfigPath = os.path.join("Config", newCalibrationDir)
            for src in fnames:
                srcPath = os.path.join(oldConfigPath, src)
                destPath = os.path.join(newConfigPath, src)
                print(srcPath)
                print(destPath)
                shutil.copy(srcPath, destPath)

            # Confirm that SeaBASS Headers need to be/are updated
            if ConfigFile.settings["bL2SaveSeaBASS"]:
                SeaBASSHeaderWindow.configUpdateButtonPressed(self, 'config')
            else:
                self.close()

            self.setWindowTitle(ConfigFile.filename)


    def cancelButtonPressed(self):
        print("ConfigWindow - Cancel Pressed")
        self.close()

