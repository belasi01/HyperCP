
import numpy as np
import pandas as pd
import calendar
# import matplotlib.pyplot as plt

import HDFRoot
from ConfigFile import ConfigFile
from Utilities import Utilities

class ProcessL1d:
    '''
    # The Deglitching process departs signicantly from ProSoft and PySciDON
    # Reference: ProSoft 7.7 Rev. K May 8, 2017, SAT-DN-00228
    # More information can be found in AnomalyDetection.py
    '''    
    @staticmethod
    def darkDataDeglitching(darkData, sensorType):        
        ''' Dark deglitching is now based on double-pass discrete linear convolution of the residual 
        with a stationary std over a rolling average'''        
        # print(str(sensorType))
        windowSize = int(ConfigFile.settings["fL1dDeglitch0"])
        sigma = float(ConfigFile.settings["fL1dDeglitch2"])

        darkData.datasetToColumns()
        columns = darkData.columns

        waveindex = 0
        badIndex = []
        for k in columns.items():   # Loop over all wavebands     
            timeSeries = k[1]   # Ignores waveband label (e.g. '1142.754')
            # Note: the moving average is not tolerant to 2 or fewer records
            avg = Utilities.movingAverage(timeSeries, windowSize).tolist()        
            # avg = Utilities.windowAverage(timeSeries, windowSize).mean().values.tolist()  
            residual = np.array(timeSeries) - np.array(avg)
            stdData = np.std(residual)                                  

            # First pass
            badIndex1 = Utilities.darkConvolution(timeSeries,avg,stdData,sigma)  

            # Second pass
            timeSeries2 = np.array(timeSeries[:])
            timeSeries2[badIndex1] = np.nan # BEWARE: NaNs introduced
            timeSeries2 = timeSeries2.tolist()
            avg2 = Utilities.movingAverage(timeSeries2, windowSize).tolist()        
            # avg2 = Utilities.windowAverage(timeSeries2, windowSize).mean().values.tolist()        
            residual = np.array(timeSeries2) - np.array(avg2)
            stdData = np.nanstd(residual)        

            badIndex2 = Utilities.darkConvolution(timeSeries2,avg2,stdData,sigma)  
            
            # This will eliminate data from all wavebands for glitches found in any one waveband        
            if waveindex==0:
                # badIndex = badIndex1[:]
                for i in range(len(badIndex1)):
                    if badIndex1[i] is True or badIndex2[i] is True:
                        badIndex.append(True)
                    else:
                        badIndex.append(False)
            else:
                for i in range(len(badIndex)):
                    if badIndex1[i] is True or badIndex2[i] is True or badIndex[i] is True:
                        badIndex[i] = True
                    else:
                        badIndex[i] = False # this is redundant
            # print(badIndex[i])                
            waveindex += 1
        return badIndex
           
    @staticmethod
    def lightDataDeglitching(lightData, sensorType):        
        ''' Light deglitching is now based on double-pass discrete linear convolution of the residual
        with a ROLLING std over a rolling average'''
        
        # print(str(sensorType))
        windowSize = int(ConfigFile.settings["fL1dDeglitch1"])
        sigma = float(ConfigFile.settings["fL1dDeglitch3"])

        lightData.datasetToColumns()
        columns = lightData.columns

        waveindex = 0
        badIndex = []
        for k in columns.items():        
            timeSeries = k[1]       
            # Note: the moving average is not tolerant to 2 or fewer records     
            avg = Utilities.movingAverage(timeSeries, windowSize).tolist()        
            residual = np.array(timeSeries) - np.array(avg)
                           
             # Calculate the variation in the distribution of the residual
            residualDf = pd.DataFrame(residual)
            testing_std_as_df = residualDf.rolling(windowSize).std()
            rolling_std = testing_std_as_df.replace(np.nan,
                testing_std_as_df.iloc[windowSize - 1]).round(3).iloc[:,0].tolist() 
            # This rolling std on the residual has a tendancy to blow up for extreme outliers,
            # replace it with the median residual std when that happens
            y = np.array(rolling_std)
            y[y > np.median(y)+3*np.std(y)] = np.median(y)
            rolling_std = y.tolist()

            
            # First pass
            badIndex1 = Utilities.lightConvolution(timeSeries,avg,rolling_std,sigma)

            # Second pass
            timeSeries2 = np.array(timeSeries[:])
            timeSeries2[badIndex1] = np.nan
            timeSeries2 = timeSeries2.tolist()
            avg2 = Utilities.movingAverage(timeSeries2, windowSize).tolist()        
            # avg2 = Utilities.windowAverage(timeSeries2, windowSize).mean.values.tolist()        
            residual2 = np.array(timeSeries2) - np.array(avg2)        
            # Calculate the variation in the distribution of the residual
            residualDf2 = pd.DataFrame(residual2)
            testing_std_as_df2 = residualDf2.rolling(windowSize).std()
            rolling_std2 = testing_std_as_df2.replace(np.nan,
                testing_std_as_df2.iloc[windowSize - 1]).round(3).iloc[:,0].tolist()
            # This rolling std on the residual has a tendancy to blow up for extreme outliers,
            # replace it with the median residual std when that happens
            y = np.array(rolling_std2)
            y[np.isnan(y)] = np.nanmedian(y)
            y[y > np.nanmedian(y)+3*np.nanstd(y)] = np.nanmedian(y)
            rolling_std2 = y.tolist()

            badIndex2 = Utilities.lightConvolution(timeSeries2,avg2,rolling_std2,sigma)
            
            # This will eliminate data from all wavebands for glitches found in any one waveband        
            if waveindex==0:
                # badIndex = badIndex1[:]
                for i in range(len(badIndex1)):
                    if badIndex1[i] is True or badIndex2[i] is True:
                        badIndex.append(True)
                    else:
                        badIndex.append(False)
            else:
                for i in range(len(badIndex)):
                    if badIndex1[i] is True or badIndex2[i] is True or badIndex[i] is True:
                        badIndex[i] = True
                    else:
                        badIndex[i] = False # this is redundant
            # print(badIndex[i])                
            waveindex += 1
        return badIndex

    @staticmethod
    def processDataDeglitching(node, sensorType):   
        msg = sensorType
        print(msg)    
        Utilities.writeLogFile(msg)

        darkData = None
        lightData = None
        windowSizeDark = int(ConfigFile.settings["fL1dDeglitch0"])
        windowSizeLight = int(ConfigFile.settings["fL1dDeglitch1"])
        for gp in node.groups:
            if gp.attributes["FrameType"] == "ShutterDark" and sensorType in gp.datasets:
                darkData = gp.getDataset(sensorType)
            if gp.attributes["FrameType"] == "ShutterLight" and sensorType in gp.datasets:
                lightData = gp.getDataset(sensorType)
            
            # Rolling averages required for deglitching of data are intolerant to 2 or fewer data points
            # Furthermore, 5 or fewer datapoints is a suspiciously short sampling time. Finally,
            # Having fewer data points than the size of the rolling window won't work. Exit processing if 
            # these conditions are met.
            
            # Problems with the sizes of the datasets:
            if darkData is not None and lightData is not None:
                if len(darkData.data) <= 2 or \
                    len(lightData.data) <= 5 or \
                    len(darkData.data) < windowSizeDark or \
                    len(lightData.data) < windowSizeLight:

                        return True # Sets the flag to true


        if darkData is None:
            msg = "Error: No dark data to deglitch"
            print(msg)
            Utilities.writeLogFile(msg)
        else:
            msg = "Deglitching dark"
            print(msg)
            Utilities.writeLogFile(msg)

            badIndexDark = ProcessL1d.darkDataDeglitching(darkData, sensorType)
            msg = f'Data reduced by {sum(badIndexDark)} ({round(100*sum(badIndexDark)/len(darkData.data))}%)'
            print(msg)
            Utilities.writeLogFile(msg)
            

        if lightData is None:
            msg = "Error: No light data to deglitch"
            print(msg)
            Utilities.writeLogFile(msg)
        else:                
            msg = "Deglitching light"
            print(msg)
            Utilities.writeLogFile(msg)

            badIndexLight = ProcessL1d.lightDataDeglitching(lightData, sensorType)      
            msg = f'Data reduced by {sum(badIndexLight)} ({round(100*sum(badIndexLight)/len(lightData.data))}%)'
            print(msg)
            Utilities.writeLogFile(msg)

        # Delete the glitchy rows of the datasets
        for gp in node.groups:
            if gp.attributes["FrameType"] == "ShutterDark" and sensorType in gp.datasets:
               gp.datasetDeleteRow(np.where(badIndexDark))

            if gp.attributes["FrameType"] == "ShutterLight" and sensorType in gp.datasets:
                lightData = gp.getDataset(sensorType)
                gp.datasetDeleteRow(np.where(badIndexLight))
                
        return False
            

    @staticmethod
    def darkCorrection(darkData, darkTimer, lightData, lightTimer):
        if (darkData == None) or (lightData == None):            
            msg  = f'Dark Correction, dataset not found: {darkData} , {lightData}'
            print(msg)
            Utilities.writeLogFile(msg)
            return False

        '''                
        # HyperInSPACE - Interpolate Dark values to match light measurements (e.g. Brewin 2016, Prosoft
        # 7.7 User Manual SAT-DN-00228-K)
        '''

        if Utilities.hasNan(lightData):
            msg = "**************Found NAN 0"
            print(msg)
            Utilities.writeLogFile(msg)
            exit

        if Utilities.hasNan(darkData):
            msg = "**************Found NAN 1"
            print(msg)
            Utilities.writeLogFile(msg)
            exit

        # Interpolate Dark Dataset to match number of elements as Light Dataset
        newDarkData = np.copy(lightData.data)        
        for k in darkData.data.dtype.fields.keys(): # For each wavelength
            # x = np.copy(darkTimer.data["NONE"]).tolist() # darktimer
            x = np.copy(darkTimer.data).tolist() # darktimer
            y = np.copy(darkData.data[k]).tolist()  # data at that band over time
            # new_x = lightTimer.data["NONE"].tolist()  # lighttimer
            new_x = lightTimer.data  # lighttimer

            if len(x) < 3 or len(y) < 3 or len(new_x) < 3:
                msg = "**************Cannot do cubic spline interpolation, length of datasets < 3"
                print(msg)
                Utilities.writeLogFile(msg)
                return False

            if not Utilities.isIncreasing(x):
                msg = "**************darkTimer does not contain strictly increasing values"
                print(msg)
                Utilities.writeLogFile(msg)
                return False
            if not Utilities.isIncreasing(new_x):
                msg = "**************lightTimer does not contain strictly increasing values"
                print(msg)
                Utilities.writeLogFile(msg)
                return False

            # print(x[0], new_x[0])
            #newDarkData[k] = Utilities.interp(x,y,new_x,'cubic')
            if len(x) >= 3:
                # newDarkData[k] = Utilities.interpSpline(x,y,new_x)
                
                # Because x is now a list of datetime tuples, they'll need to be
                # converted to Unix timestamp values
                xTS = [calendar.timegm(xDT.utctimetuple()) + xDT.microsecond / 1E6 for xDT in x]
                newXTS = [calendar.timegm(xDT.utctimetuple()) + xDT.microsecond / 1E6 for xDT in new_x]

                # newDarkData[k] = Utilities.interp(x,y,new_x, fill_value=np.nan)
                newDarkData[k] = Utilities.interp(xTS,y,newXTS, fill_value=np.nan)


                for val in newDarkData[k]:
                    if np.isnan(val):
                        print('nan')
                        exit
                # newDarkData[k] = Utilities.interp(x,y,new_x)
            else:
                msg = '**************Record too small for splining. Exiting.'
                print(msg)
                Utilities.writeLogFile(msg)
                return False            

        darkData.data = newDarkData

        if Utilities.hasNan(darkData):
            msg = "**************Found NAN 2"
            print(msg)
            Utilities.writeLogFile(msg)
            exit

        # Correct light data by subtracting interpolated dark data from light data
        for k in lightData.data.dtype.fields.keys():
            for x in range(lightData.data.shape[0]):
                lightData.data[k][x] -= newDarkData[k][x]

        if Utilities.hasNan(lightData):
            msg = "**************Found NAN 3"
            print(msg)
            Utilities.writeLogFile(msg)
            exit

        return True


    # Copies TIMETAG2 values to Timer and converts to seconds
    @staticmethod
    def copyTimetag2(timerDS, tt2DS):
        if (timerDS.data is None) or (tt2DS.data is None):
            msg = "copyTimetag2: Timer/TT2 is None"
            print(msg)
            Utilities.writeLogFile(msg)
            return

        #print("Time:", time)
        #print(ds.data)
        for i in range(0, len(timerDS.data)):
            tt2 = float(tt2DS.data["NONE"][i])
            t = Utilities.timeTag2ToSec(tt2)
            timerDS.data["NONE"][i] = t        

    @staticmethod
    def processDarkCorrection(node, sensorType):
        msg = f'Dark Correction: {sensorType}'
        print(msg)
        Utilities.writeLogFile(msg)
        darkGroup = None
        darkData = None
        darkDateTime = None
        lightGroup = None
        lightData = None
        lightDateTime = None

        for gp in node.groups:
            if gp.attributes["FrameType"] == "ShutterDark" and gp.getDataset(sensorType):
                darkGroup = gp
                darkData = gp.getDataset(sensorType)
                darkDateTime = gp.getDataset("DATETIME")

            if gp.attributes["FrameType"] == "ShutterLight" and gp.getDataset(sensorType):
                lightGroup = gp
                lightData = gp.getDataset(sensorType) 
                lightDateTime = gp.getDataset("DATETIME")

        if darkGroup is None or lightGroup is None:
            msg = f'No radiometry found for {sensorType}'
            print(msg)
            Utilities.writeLogFile(msg)
            return False        

        # Instead of using TT2 or seconds, use python datetimes to avoid problems crossing 
        # UTC midnight.
        if not ProcessL1d.darkCorrection(darkData, darkDateTime, lightData, lightDateTime):
            msg = f'ProcessL1d.darkCorrection failed  for {sensorType}'
            print(msg)
            Utilities.writeLogFile(msg)
            return False
            
        # Now that the dark correction is done, we can strip the dark shutter data from the        
        # HDF object.            
        for gp in node.groups:
            if gp.attributes["FrameType"] == "ShutterDark" and gp.getDataset(sensorType):                
                node.removeGroup(gp)
        # And rename the corrected light frame
        for gp in node.groups:
            if gp.attributes["FrameType"] == "ShutterLight" and gp.getDataset(sensorType):
                gp.id = gp.id[0:2] # Strip off "_LIGHT" from the name
        return True
    
    @staticmethod
    def processL1d(node):
        '''
        Apply data deglitching to light and shutter-dark data, then apply dark shutter correction to light data.
        '''
        root = HDFRoot.HDFRoot()
        root.copy(node) 
        root.attributes["PROCESSING_LEVEL"] = "1d"
        
        # Add a dataset to each group for DATETIME, as defined by TIMETAG2 and DATETAG        
        root  = Utilities.rootAddDateTime(root)

        # Fix in case time doesn't increase from one sample to the next
        # or there are fewer than 2 two stamps remaining.
        for gp in root.groups:
            if gp.id != "SOLARTRACKER_STATUS":
                msg = f'Screening {gp.id} for clean timestamps.'
                print(msg)
                Utilities.writeLogFile(msg)
                if not Utilities.fixDateTime(gp):
                    msg = '***********Too few records in the file to continue. Exiting.'
                    print(msg)
                    Utilities.writeLogFile(msg)
                    return None

        if int(ConfigFile.settings["bL1dDeglitch"]) == 1:
            root.attributes["DEGLITCH_PRODAT"] = "ON"
            root.attributes["DEGLITCH_REFDAT"] = "ON"
            flagES = ProcessL1d.processDataDeglitching(root, "ES")
            flagLI = ProcessL1d.processDataDeglitching(root, "LI")
            flagLT = ProcessL1d.processDataDeglitching(root, "LT")

            if flagES or flagLI or flagLT:
                msg = '***********Too few records in the file to continue. Exiting.'
                print(msg)
                Utilities.writeLogFile(msg)
                return None

        else:
            root.attributes["DEGLITCH_PRODAT"] = "OFF"
            root.attributes["DEGLITCH_REFDAT"] = "OFF"
            #root.attributes["STRAY_LIGHT_CORRECT"] = "OFF"
            #root.attributes["THERMAL_RESPONSIVITY_CORRECT"] = "OFF"        
        
        if not ProcessL1d.processDarkCorrection(root, "ES"):
            msg = 'Error dark correcting ES'
            print(msg)
            Utilities.writeLogFile(msg)
            return None
        if not ProcessL1d.processDarkCorrection(root, "LI"):
            msg = 'Error dark correcting LI'
            print(msg)
            Utilities.writeLogFile(msg)
            return None
        if not ProcessL1d.processDarkCorrection(root, "LT"):
            msg = 'Error dark correcting LT'
            print(msg)
            Utilities.writeLogFile(msg)
            return None

        # Datetime format is not supported in HDF5; remove
        # DATETIME is not supported in HDF5; remove
        for gp in root.groups:
            if (gp.id == "SOLARTRACKER_STATUS") is False:
                del gp.datasets["DATETIME"]    

        return root