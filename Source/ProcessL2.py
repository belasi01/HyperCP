''' Process L1BQC to L2 '''
import os
import collections
import warnings
import time

import datetime
import copy
import numpy as np
import scipy as sp

from PyQt5 import QtWidgets
from tqdm import tqdm

from Source.HDFRoot import HDFRoot
from Source.Utilities import Utilities
from Source.ConfigFile import ConfigFile
from Source.RhoCorrections import RhoCorrections
from Source.Uncertainty_Analysis import Propagate
from Source.Weight_RSR import Weight_RSR
from Source.ProcessL2OCproducts import ProcessL2OCproducts
from Source.ProcessL2BRDF import ProcessL2BRDF
from Source.ProcessInstrumentUncertainties import Trios, HyperOCR, Dalec


class ProcessL2:
    ''' Process L2 '''

    @staticmethod
    def nirCorrectionSatellite(root, sensor, rrsNIRCorr, nLwNIRCorr):
        newReflectanceGroup = root.getGroup("REFLECTANCE")
        newRrsData = newReflectanceGroup.getDataset(f'Rrs_{sensor}')
        newnLwData = newReflectanceGroup.getDataset(f'nLw_{sensor}')

        # These will include all slices in root so far
        # Below the most recent/current slice [-1] will be selected for processing
        rrsSlice = newRrsData.columns
        nLwSlice = newnLwData.columns

        for k in rrsSlice:
            if (k != 'Datetime') and (k != 'Datetag') and (k != 'Timetag2'):
                rrsSlice[k][-1] -= rrsNIRCorr
        for k in nLwSlice:
            if (k != 'Datetime') and (k != 'Datetag') and (k != 'Timetag2'):
                nLwSlice[k][-1] -= nLwNIRCorr

        newRrsData.columnsToDataset()
        newnLwData.columnsToDataset()


    @staticmethod
    def nirCorrection(node, sensor, F0):
        # F0 is sensor specific, but ultimately, SimSpec can only be applied to hyperspectral data anyway,
        # so output the correction and apply it to satellite bands later.
        simpleNIRCorrection = int(ConfigFile.settings["bL2SimpleNIRCorrection"])
        simSpecNIRCorrection = int(ConfigFile.settings["bL2SimSpecNIRCorrection"])

        newReflectanceGroup = node.getGroup("REFLECTANCE")
        newRrsData = newReflectanceGroup.getDataset(f'Rrs_{sensor}')
        newnLwData = newReflectanceGroup.getDataset(f'nLw_{sensor}')
        newRrsUNCData = newReflectanceGroup.getDataset(f'Rrs_{sensor}_unc')
        newnLwUNCData = newReflectanceGroup.getDataset(f'nLw_{sensor}_unc')

        newNIRData = newReflectanceGroup.getDataset(f'nir_{sensor}')
        newNIRnLwData = newReflectanceGroup.getDataset(f'nir_nLw_{sensor}')

        # These will include all slices in node so far
        # Below the most recent/current slice [-1] will be selected for processing
        rrsSlice = newRrsData.columns
        nLwSlice = newnLwData.columns
        nirSlice = newNIRData.columns
        nirnLwSlice = newNIRnLwData.columns

        # # Perform near-infrared residual correction to remove additional atmospheric and glint contamination
        # if ConfigFile.settings["bL2PerformNIRCorrection"]:
        if simpleNIRCorrection:
            # Data show a minimum near 725; using an average from above 750 leads to negative reflectances
            # Find the minimum between 700 and 800, and subtract it from spectrum (spectrally flat)
            Utilities.writeLogFileAndPrint("Perform simple residual NIR subtraction.")

            # rrs correction
            NIRRRs = []
            for k in rrsSlice:
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue
                if float(k) >= 700 and float(k) <= 800:
                    NIRRRs.append(rrsSlice[k][-1])
            rrsNIRCorr = min(NIRRRs)
            if rrsNIRCorr < 0:
                #   NOTE: SeaWiFS protocols for residual NIR were never intended to ADD reflectance
                #   This is most likely in blue, non-turbid waters not intended for NIR offset correction.
                #   Revert to NIR correction of 0 when this happens. No good way to update the L2 attribute
                #   metadata because it may only be on some ensembles within a file.
                Utilities.writeLogFileAndPrint('Bad NIR Correction. Revert to No NIR correction.')
                rrsNIRCorr = 0
            # Subtract average from each waveband
            for k in rrsSlice:
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue

                rrsSlice[k][-1] -= rrsNIRCorr

            nirSlice['NIR_offset'].append(rrsNIRCorr)

            # nLw correction
            NIRRRs = []
            for k in nLwSlice:
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue
                if float(k) >= 700 and float(k) <= 800:
                    NIRRRs.append(nLwSlice[k][-1])
            nLwNIRCorr = min(NIRRRs)
            # Subtract average from each waveband
            for k in nLwSlice:
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue
                nLwSlice[k][-1] -= nLwNIRCorr

            nirnLwSlice['NIR_offset'].append(nLwNIRCorr)

        elif simSpecNIRCorrection:
            # From Ruddick 2005, Ruddick 2006 use NIR normalized similarity spectrum
            # (spectrally flat)
            Utilities.writeLogFileAndPrint("Perform similarity spectrum residual NIR subtraction.")

            # For simplicity, follow calculation in rho (surface reflectance), then covert to rrs
            ρSlice = copy.deepcopy(rrsSlice)
            for k,value in ρSlice.items():
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue
                ρSlice[k][-1] = value[-1] * np.pi

            # These ratios are for rho = pi*Rrs
            α1 = 2.35 # 720/780 only good for rho(720)<0.03
            α2 = 1.91 # 780/870 try to avoid, data is noisy here
            threshold = 0.03

            # Retrieve TSIS-1s
            wavelength = [float(key) for key in F0.keys()]
            F0 = [value for value in F0.values()]

            # Rrs
            ρ720 = []
            x = []
            for k in ρSlice:
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue
                if float(k) >= 700 and float(k) <= 750:
                    x.append(float(k))

                    # convert to surface reflectance ρ = π * Rrs
                    ρ720.append(ρSlice[k][-1]) # Using current element/slice [-1]

            # if not ρ720:
            #     print("Error: NIR wavebands unavailable")
            #     if os.environ["HYPERINSPACE_CMD"].lower() == 'false':
            #         QtWidgets.QMessageBox.critical("Error", "NIR wavebands unavailable")
            ρ1 = sp.interpolate.interp1d(x,ρ720)(720)
            F01 = sp.interpolate.interp1d(wavelength,F0)(720)
            ρ780 = []
            x = []
            for k in ρSlice:
                if k in ('Datetime', 'Datetag', 'Timetag2'):
                    continue
                if float(k) >= 760 and float(k) <= 800:
                    x.append(float(k))
                    ρ780.append(ρSlice[k][-1])
            if not ρ780:
                print("Error: NIR wavebands unavailable")
                if os.environ["HYPERINSPACE_CMD"].lower() == 'false':
                    QtWidgets.QMessageBox.critical("Error", "NIR wavebands unavailable")
            ρ2 = sp.interpolate.interp1d(x,ρ780)(780)
            F02 = sp.interpolate.interp1d(wavelength,F0)(780)
            ρ870 = []
            x = []
            for k in ρSlice:
                if k in ('Datetime', 'Datetag', 'Timetag2'):
                    continue
                if float(k) >= 850 and float(k) <= 890:
                    x.append(float(k))
                    ρ870.append(ρSlice[k][-1])
            if not ρ870:
                Utilities.writeLogFileAndPrint('No data found at 870 nm')
                ρ3 = None
                F03 = None
            else:
                ρ3 = sp.interpolate.interp1d(x,ρ870)(870)
                F03 = sp.interpolate.interp1d(wavelength,F0)(870)

            # Reverts to primary mode even on threshold trip in cases where no 870nm available
            if ρ1 < threshold or not ρ870:
                ε = (α1*ρ2 - ρ1)/(α1-1)
                εnLw = (α1*ρ2*F02 - ρ1*F01)/(α1-1)
                Utilities.writeLogFileAndPrint(f'offset(rrs) = {ε}; offset(nLw) = {εnLw}')
            else:
                Utilities.writeLogFileAndPrint("SimSpec threshold tripped. Using 780/870 instead.")
                ε = (α2*ρ3 - ρ2)/(α2-1)
                εnLw = (α2*ρ3*F03 - ρ2*F02)/(α2-1)
                Utilities.writeLogFileAndPrint(f'offset(rrs) = {ε}; offset(nLw) = {εnLw}')

            rrsNIRCorr = ε/np.pi
            nLwNIRCorr = εnLw/np.pi

            # Now apply to rrs and nLw
            # NOTE: This correction is also susceptible to a correction that ADDS to reflectance
            #   spectrally, depending on spectral shape (see test_SimSpec.m).
            #   This is most likely in blue, non-turbid waters not intended for SimSpec.
            #   Revert to NIR correction of 0 when this happens. No good way to update the L2 attribute
            #   metadata because it may only be on some ensembles within a file.
            if rrsNIRCorr < 0:
                Utilities.writeLogFileAndPrint('Bad NIR Correction. Revert to No NIR correction.')
                rrsNIRCorr = 0
                nLwNIRCorr = 0
                # L2 metadata will be updated

            for k in rrsSlice:
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue

                rrsSlice[k][-1] -= float(rrsNIRCorr) # Only working on the last (most recent' [-1]) element of the slice
                nLwSlice[k][-1] -= float(nLwNIRCorr)


            nirSlice['NIR_offset'].append(rrsNIRCorr)
            nirnLwSlice['NIR_offset'].append(nLwNIRCorr)

        newRrsData.columnsToDataset()
        newnLwData.columnsToDataset()
        newRrsUNCData.columnsToDataset()
        newnLwUNCData.columnsToDataset()
        newNIRData.columnsToDataset()
        newNIRnLwData.columnsToDataset()

        return rrsNIRCorr, nLwNIRCorr


    @staticmethod
    def spectralReflectance(node, sensor, timeObj, xSlice, F0, F0_unc, rhoScalar, rhoVec, waveSubset, xUNC):
        ''' The slices, stds, F0, rhoVec here are sensor-waveband specific '''
        esXSlice = xSlice['es'] # mean
        esXmedian = xSlice['esMedian']
        esXRemaining = xSlice['esRemaining']
        esXstd = xSlice['esSTD']
        liXSlice = xSlice['li']
        liXmedian = xSlice['liMedian']
        liXRemaining = xSlice['liRemaining']
        liXstd = xSlice['liSTD']
        ltXSlice = xSlice['lt']
        ltXmedian = xSlice['ltMedian']
        ltXRemaining = xSlice['ltRemaining']
        ltXstd = xSlice['ltSTD']
        dateTime = timeObj['dateTime']
        dateTag = timeObj['dateTag']
        timeTag = timeObj['timeTag']

        threeCRho = int(ConfigFile.settings["bL23CRho"])
        ZhangRho = int(ConfigFile.settings["bL2ZhangRho"])

        # Root (new/output) groups:
        newReflectanceGroup = node.getGroup("REFLECTANCE")
        newRadianceGroup = node.getGroup("RADIANCE")
        newIrradianceGroup = node.getGroup("IRRADIANCE")

        newRhoHyper, newRhoUNCHyper, newNIRnLwData, newNIRData, nLw = None, None, None, None, None

        # If this is the first ensemble spectrum, set up the new datasets
        if not f'Rrs_{sensor}' in newReflectanceGroup.datasets:
            newESData = newIrradianceGroup.addDataset(f"ES_{sensor}")
            newLIData = newRadianceGroup.addDataset(f"LI_{sensor}")
            newLTData = newRadianceGroup.addDataset(f"LT_{sensor}")
            newLWData = newRadianceGroup.addDataset(f"LW_{sensor}")

            newESDataMedian = newIrradianceGroup.addDataset(f"ES_{sensor}_median")
            newLIDataMedian = newRadianceGroup.addDataset(f"LI_{sensor}_median")
            newLTDataMedian = newRadianceGroup.addDataset(f"LT_{sensor}_median")

            newRrsData = newReflectanceGroup.addDataset(f"Rrs_{sensor}")
            newRrsUncorrData = newReflectanceGroup.addDataset(f"Rrs_{sensor}_uncorr") # Preserve uncorrected Rrs (= lt/es)
            newnLwData = newReflectanceGroup.addDataset(f"nLw_{sensor}")

            # September 2023. For clarity, drop the "Delta" nominclature in favor of
            # either STD (standard deviation of the sample) or UNC (uncertainty)
            newESSTDData = newIrradianceGroup.addDataset(f"ES_{sensor}_sd")
            newLISTDData = newRadianceGroup.addDataset(f"LI_{sensor}_sd")
            newLTSTDData = newRadianceGroup.addDataset(f"LT_{sensor}_sd")

            # No average (mean or median) or standard deviation values associated with Lw or reflectances,
            #   because these are calculated from the means of Lt, Li, Es
      
            newESUNCData = newIrradianceGroup.addDataset(f"ES_{sensor}_unc")
            newLIUNCData = newRadianceGroup.addDataset(f"LI_{sensor}_unc")
            newLTUNCData = newRadianceGroup.addDataset(f"LT_{sensor}_unc")
            newLWUNCData = newRadianceGroup.addDataset(f"LW_{sensor}_unc")
            newRrsUNCData = newReflectanceGroup.addDataset(f"Rrs_{sensor}_unc")
            newnLwUNCData = newReflectanceGroup.addDataset(f"nLw_{sensor}_unc")

            # Add standard deviation datasets for comparison
            newLWSTDData = newRadianceGroup.addDataset(f"LW_{sensor}_sd")
            newRrsSTDData = newReflectanceGroup.addDataset(f"Rrs_{sensor}_sd")
            newnLwSTDData = newReflectanceGroup.addDataset(f"nLw_{sensor}_sd")

            # For CV, use CV = STD/n

            if sensor == 'HYPER':
                newRhoHyper = newReflectanceGroup.addDataset(f"rho_{sensor}")
                newRhoUNCHyper = newReflectanceGroup.addDataset(f"rho_{sensor}_unc")
                if ConfigFile.settings["bL2PerformNIRCorrection"]:
                    newNIRData = newReflectanceGroup.addDataset(f'nir_{sensor}')
                    newNIRnLwData = newReflectanceGroup.addDataset(f'nir_nLw_{sensor}')
        else:
            newESData = newIrradianceGroup.getDataset(f"ES_{sensor}")
            newLIData = newRadianceGroup.getDataset(f"LI_{sensor}")
            newLTData = newRadianceGroup.getDataset(f"LT_{sensor}")
            newLWData = newRadianceGroup.getDataset(f"LW_{sensor}")

            newESDataMedian = newIrradianceGroup.getDataset(f"ES_{sensor}_median")
            newLIDataMedian = newRadianceGroup.getDataset(f"LI_{sensor}_median")
            newLTDataMedian = newRadianceGroup.getDataset(f"LT_{sensor}_median")

            newRrsData = newReflectanceGroup.getDataset(f"Rrs_{sensor}")
            newRrsUncorrData = newReflectanceGroup.getDataset(f"Rrs_{sensor}_uncorr")
            newnLwData = newReflectanceGroup.getDataset(f"nLw_{sensor}")

            newESSTDData = newIrradianceGroup.getDataset(f"ES_{sensor}_sd")
            newLISTDData = newRadianceGroup.getDataset(f"LI_{sensor}_sd")
            newLTSTDData = newRadianceGroup.getDataset(f"LT_{sensor}_sd")

            # No average (mean or median) or standard deviation values associated with Lw or reflectances,
            #   because these are calculated from the means of Lt, Li, Es
      
            newESUNCData = newIrradianceGroup.getDataset(f"ES_{sensor}_unc")
            newLIUNCData = newRadianceGroup.getDataset(f"LI_{sensor}_unc")
            newLTUNCData = newRadianceGroup.getDataset(f"LT_{sensor}_unc")
            newLWUNCData = newRadianceGroup.getDataset(f"LW_{sensor}_unc")
            newRrsUNCData = newReflectanceGroup.getDataset(f"Rrs_{sensor}_unc")
            newnLwUNCData = newReflectanceGroup.getDataset(f"nLw_{sensor}_unc")

            newLWSTDData = newRadianceGroup.getDataset(f"LW_{sensor}_sd")
            newRrsSTDData = newReflectanceGroup.getDataset(f"Rrs_{sensor}_sd")
            newnLwSTDData = newReflectanceGroup.getDataset(f"nLw_{sensor}_sd")

            if sensor == 'HYPER':
                newRhoHyper = newReflectanceGroup.getDataset(f"rho_{sensor}")
                newRhoUNCHyper = newReflectanceGroup.getDataset(f"rho_{sensor}_unc")
                if ConfigFile.settings["bL2PerformNIRCorrection"]:
                    newNIRData = newReflectanceGroup.getDataset(f'nir_{sensor}')
                    newNIRnLwData = newReflectanceGroup.addDataset(f'nir_nLw_{sensor}')

        # Add datetime stamps back onto ALL datasets associated with the current sensor
        # If this is the first spectrum, add date/time, otherwise append
        # Groups REFLECTANCE, IRRADIANCE, and RADIANCE are intiallized with empty datasets, but
        # ANCILLARY is not.
        if "Datetag" not in newRrsData.columns:
            for gp in node.groups:
                if gp.id == "ANCILLARY": # Ancillary is already populated. The other groups only have empty (named) datasets
                    continue
                else:
                    for ds in gp.datasets:
                        if sensor in ds: # Only add datetime stamps to the current sensor datasets
                            gp.datasets[ds].columns["Datetime"] = [dateTime] # mean of the ensemble datetime stamp
                            gp.datasets[ds].columns["Datetag"] = [dateTag]
                            gp.datasets[ds].columns["Timetag2"] = [timeTag]
        else:
            for gp in node.groups:
                if gp.id == "ANCILLARY":
                    continue
                else:
                    for ds in gp.datasets:
                        if sensor in ds:
                            gp.datasets[ds].columns["Datetime"].append(dateTime)
                            gp.datasets[ds].columns["Datetag"].append(dateTag)
                            gp.datasets[ds].columns["Timetag2"].append(timeTag)

        # Organise Uncertainty into wavebands
        lwUNC = {}
        rrsUNC = {}
        rhoUNC = {}
        esUNC = {}
        liUNC = {}
        ltUNC = {}

        # Only Factory - Trios has no uncertainty here
        if ConfigFile.settings['fL1bCal'] >= 2 or ConfigFile.settings['SensorType'].lower() == 'seabird':
            #or ConfigFile.settings['SensorType'].lower() == 'dalec':
            esUNC = xUNC[f'esUNC_{sensor}']  # should already be convolved to hyperspec
            liUNC = xUNC[f'liUNC_{sensor}']  # added reference to HYPER as band convolved uncertainties will no longer
            ltUNC = xUNC[f'ltUNC_{sensor}']  # overwite normal instrument uncertainties during processing
            rhoUNC = xUNC[f'rhoUNC_{sensor}']
            for i, wvl in enumerate(waveSubset):
                k = str(wvl)
                if (any([wvl == float(x) for x in esXSlice]) and
                        any([wvl == float(x) for x in liXSlice]) and
                        any([wvl == float(x) for x in ltXSlice])):  # More robust (able to handle sensor and hyper bands
                    if sensor == 'HYPER':
                        lwUNC[k] = xUNC['lwUNC'][i]
                        rrsUNC[k] = xUNC['rrsUNC'][i]
                    else:  # apply the sensor specific Lw and Rrs uncertainties
                        lwUNC[k] = xUNC[f'lwUNC_{sensor}'][i]
                        rrsUNC[k] = xUNC[f'rrsUNC_{sensor}'][i]
            #print(sensor+" rrsUNC")
            #print(rrsUNC)
        else:
            # factory case
            for wvl in waveSubset:
                k = str(wvl)
                if (any([wvl == float(x) for x in esXSlice]) and
                        any([wvl == float(x) for x in liXSlice]) and
                        any([wvl == float(x) for x in ltXSlice])):  # old version had issues with '.0'
                    esUNC[k] = 0
                    liUNC[k] = 0
                    ltUNC[k] = 0
                    rhoUNC[k] = 0
                    lwUNC[k] = 0
                    rrsUNC[k] = 0

        deleteKey = []
        for i, wvl in enumerate(waveSubset):  # loop through wavebands
            k = str(wvl)
            if (any(wvl == float(x) for x in esXSlice) and
                    any(wvl == float(x) for x in liXSlice) and
                    any(wvl == float(x) for x in ltXSlice)):
                # Initialize the new dataset if this is the first slice
                if k not in newESData.columns:
                    newESData.columns[k] = []
                    newLIData.columns[k] = []
                    newLTData.columns[k] = []
                    newLWData.columns[k] = []
                    newRrsData.columns[k] = []
                    newRrsUncorrData.columns[k] = []
                    newnLwData.columns[k] = []

                    # No average (mean or median) or standard deviation values associated with Lw or reflectances,
                    #   because these are calculated from the means of Lt, Li, Es
                    newESDataMedian.columns[k] = []
                    newLIDataMedian.columns[k] = []
                    newLTDataMedian.columns[k] = []

                    newESSTDData.columns[k] = []
                    newLISTDData.columns[k] = []
                    newLTSTDData.columns[k] = []
                    newESUNCData.columns[k] = []
                    newLIUNCData.columns[k] = []
                    newLTUNCData.columns[k] = []
                    newLWUNCData.columns[k] = []
                    newRrsUNCData.columns[k] = []
                    newnLwUNCData.columns[k] = []

                    newLWSTDData.columns[k] = []
                    newRrsSTDData.columns[k] = []
                    newnLwSTDData.columns[k] = []

                    if sensor == 'HYPER':
                        newRhoHyper.columns[k] = []
                        newRhoUNCHyper.columns[k] = []
                        if ConfigFile.settings["bL2PerformNIRCorrection"]:
                            newNIRData.columns['NIR_offset'] = [] # not used until later; highly unpythonic
                            newNIRnLwData.columns['NIR_offset'] = []

                # At this waveband (k); still using complete wavelength set
                es = esXSlice[k][0] # Always the zeroth element; i.e. XSlice data are independent of past slices and node
                li = liXSlice[k][0]
                lt = ltXSlice[k][0]
                esRemaining = np.asarray(esXRemaining[k]) # array of remaining ensemble values in this band
                liRemaining = np.asarray(liXRemaining[k])
                ltRemaining = np.asarray(ltXRemaining[k])
                f0 = F0[k]
                f0UNC = F0_unc[k]

                esMedian = esXmedian[k][0]
                liMedian = liXmedian[k][0]
                ltMedian = ltXmedian[k][0]

                esSTD = esXstd[k][0]
                liSTD = liXstd[k][0]
                ltSTD = ltXstd[k][0]

                # Calculate the remote sensing reflectance
                nLwUNC = {}
                lwRemainingSD = 0
                rrsRemainingSD = 0
                nLwRemainingSD = 0
                if threeCRho:
                    lw = lt - (rhoScalar * li)
                    rrs = lw / es
                    nLw = rrs*f0

                    # Now calculate the std for lw, rrs
                    lwRemaining = ltRemaining - (rhoScalar * liRemaining)
                    rrsRemaining = lwRemaining / esRemaining
                    lwRemainingSD = np.std(lwRemaining)
                    rrsRemainingSD = np.std(rrsRemaining)
                    nLwRemainingSD = np.std(rrsRemaining*f0)

                elif ZhangRho:
                    # Only populate the valid wavelengths
                    if float(k) in waveSubset:
                        lw = lt - (rhoVec[k] * li)
                        rrs = lw / es
                        nLw = rrs*f0

                        # Now calculate the std for lw, rrs
                        lwRemaining = ltRemaining - (rhoVec[k] * liRemaining)
                        rrsRemaining = lwRemaining / esRemaining
                        lwRemainingSD = np.std(lwRemaining)
                        rrsRemainingSD = np.std(rrsRemaining)
                        nLwRemainingSD = np.std(rrsRemaining*f0)

                else:
                    lw = lt - (rhoScalar * li)
                    rrs = lw / es
                    nLw = rrs*f0

                    # Now calculate the std for lw, rrs
                    lwRemaining = ltRemaining - (rhoScalar * liRemaining)
                    rrsRemaining = lwRemaining / esRemaining
                    lwRemainingSD = np.std(lwRemaining)
                    rrsRemainingSD = np.std(rrsRemaining)
                    nLwRemainingSD = np.std(rrsRemaining*f0)

                # nLw uncertainty;
                nLwUNC[k] = np.power((rrsUNC[k]**2)*(f0**2) + (rrs**2)*(f0UNC**2), 0.5)

                newESData.columns[k].append(es)
                newLIData.columns[k].append(li)
                newLTData.columns[k].append(lt)

                rrs_uncorr = lt / es

                newESSTDData.columns[k].append(esSTD)
                newLISTDData.columns[k].append(liSTD)
                newLTSTDData.columns[k].append(ltSTD)

                newLWSTDData.columns[k].append(lwRemainingSD)
                newRrsSTDData.columns[k].append(rrsRemainingSD)
                newnLwSTDData.columns[k].append(nLwRemainingSD)

                newESDataMedian.columns[k].append(esMedian)
                newLIDataMedian.columns[k].append(liMedian)
                newLTDataMedian.columns[k].append(ltMedian)

                # Only populate valid wavelengths. Mark others for deletion
                if float(k) in waveSubset:  # should be redundant!
                    newRrsUncorrData.columns[k].append(rrs_uncorr)
                    newLWData.columns[k].append(lw)
                    newRrsData.columns[k].append(rrs)
                    newnLwData.columns[k].append(nLw)

                    newLWUNCData.columns[k].append(lwUNC[k])
                    newRrsUNCData.columns[k].append(rrsUNC[k])
                    # newnLwUNCData.columns[k].append(nLwUNC)
                    newnLwUNCData.columns[k].append(nLwUNC[k])
                    if ConfigFile.settings['fL1bCal']==1 and ((ConfigFile.settings['SensorType'].lower() == 'trios' or\
                                                                ConfigFile.settings['SensorType'].lower() == 'sorad') or \
                                                              ConfigFile.settings['SensorType'].lower() == 'dalec'):
                    # Specifique case for Factory-Trios and Dalec
                        newESUNCData.columns[k].append(esUNC[k])
                        newLIUNCData.columns[k].append(liUNC[k])
                        newLTUNCData.columns[k].append(ltUNC[k])
                    else:
                        newESUNCData.columns[k].append(esUNC[k][0])
                        newLIUNCData.columns[k].append(liUNC[k][0])
                        newLTUNCData.columns[k].append(ltUNC[k][0])

                    if sensor == 'HYPER':
                        if ZhangRho:
                            newRhoHyper.columns[k].append(rhoVec[k])
                            if xUNC is not None:  # TriOS factory does not require uncertainties
                                newRhoUNCHyper.columns[k].append(xUNC[f'rhoUNC_{sensor}'][k])
                            else:
                                newRhoUNCHyper.columns[k].append(np.nan)
                        else:
                            newRhoHyper.columns[k].append(rhoScalar)
                            if xUNC is not None:  # perhaps there is a better check for TriOS Factory branch?
                                try:
                                    # TODO: explore why rho UNC is 1 index smaller than everything else
                                    # last wvl is missing
                                    newRhoUNCHyper.columns[k].append(xUNC[f'rhoUNC_{sensor}'][k])
                                except KeyError:
                                    newRhoUNCHyper.columns[k].append(0)
                            else:
                                newRhoUNCHyper.columns[k].append(np.nan)
                else:
                    deleteKey.append(k)

        # Eliminate reflectance keys/values in wavebands outside of valid set for the sake of Zhang model
        deleteKey = list(set(deleteKey))
        for key in deleteKey:
            # Only need to do this for the first ensemble in file
            if key in newRrsData.columns:
                del newLWData.columns[key]
                del newRrsUncorrData.columns[key]
                del newRrsData.columns[key]
                del newnLwData.columns[key]

                del newLWUNCData.columns[key]
                del newRrsUNCData.columns[key]
                del newnLwUNCData.columns[key]
                if sensor == 'HYPER':
                    del newRhoHyper.columns[key]

        newESData.columnsToDataset()
        newLIData.columnsToDataset()
        newLTData.columnsToDataset()
        newLWData.columnsToDataset()
        newRrsUncorrData.columnsToDataset()
        newRrsData.columnsToDataset()
        newnLwData.columnsToDataset()

        newESDataMedian.columnsToDataset()
        newLIDataMedian.columnsToDataset()
        newLTDataMedian.columnsToDataset()

        newESSTDData.columnsToDataset()
        newLISTDData.columnsToDataset()
        newLTSTDData.columnsToDataset()
        newLWSTDData.columnsToDataset()
        newRrsSTDData.columnsToDataset()
        newnLwSTDData.columnsToDataset()
        newESUNCData.columnsToDataset()
        newLIUNCData.columnsToDataset()
        newLTUNCData.columnsToDataset()
        newLWUNCData.columnsToDataset()
        newRrsUNCData.columnsToDataset()
        newnLwUNCData.columnsToDataset()

        if sensor == 'HYPER':
            newRhoHyper.columnsToDataset()
            newRhoUNCHyper.columnsToDataset()
            newRrsUncorrData.columnsToDataset()


    @staticmethod
    def filterData(group, badTimes, sensor = None):
        ''' Delete flagged records. Sensor is only specified to get the timestamp.
            All data in the group (including satellite sensors) will be deleted. '''

        Utilities.writeLogFileAndPrint(f'Remove {group.id} Data')
        timeStamp = None
        if sensor is None:
            if group.id == "ANCILLARY":
                timeStamp = group.getDataset("LATITUDE").data["Datetime"]
            if group.id == "IRRADIANCE":
                timeStamp = group.getDataset("ES").data["Datetime"]
            if group.id == "RADIANCE":
                timeStamp = group.getDataset("LI").data["Datetime"]
            if group.id == "SIXS_MODEL":
                timeStamp = group.getDataset("direct_ratio").data["Datetime"]
        else:
            if group.id == "IRRADIANCE":
                timeStamp = group.getDataset(f"ES_{sensor}").data["Datetime"]
            if group.id == "RADIANCE":
                timeStamp = group.getDataset(f"LI_{sensor}").data["Datetime"]
            if group.id == "REFLECTANCE":
                timeStamp = group.getDataset(f"Rrs_{sensor}").data["Datetime"]

        startLength = len(timeStamp)
        Utilities.writeLogFileAndPrint(f'   Length of dataset prior to removal {startLength} long')

        # Delete the records in badTime ranges from each dataset in the group
        finalCount = 0
        originalLength = len(timeStamp)
        for dateTime in badTimes:
            # Need to reinitialize for each loop
            startLength = len(timeStamp)
            newTimeStamp = []

            # Utilities.writeLogFileAndPrint(f'Eliminate data between: {dateTime}'
            # 
            # 

            start = dateTime[0]
            stop = dateTime[1]

            if startLength > 0:
                rowsToDelete = []
                for i in range(startLength):
                    if start <= timeStamp[i] and stop >= timeStamp[i]:
                        try:
                            rowsToDelete.append(i)
                            finalCount += 1
                        except Exception as err:
                            print(err)
                    else:
                        newTimeStamp.append(timeStamp[i])
                group.datasetDeleteRow(rowsToDelete)
            else:
                Utilities.writeLogFileAndPrint('Data group is empty. Continuing.')
            timeStamp = newTimeStamp.copy()

        if len(badTimes) == 0:
            startLength = 1 # avoids div by zero below when finalCount is 0

        for ds in group.datasets:
            # if ds != "STATION":
            try:
                group.datasets[ds].datasetToColumns()
            except Exception as err:
                print(err)

        Utilities.writeLogFileAndPrint(f'   Length of dataset after removal {originalLength-finalCount} long: {round(100*finalCount/originalLength)}% removed')
        return finalCount/originalLength


    @staticmethod
    def interpolateColumn(columns, wl):
        ''' Interpolate wavebands to estimate a single, unsampled waveband '''
        #print("interpolateColumn")
        # Values to return
        return_y = []

        # Column to interpolate to
        new_x = [wl]

        # Get wavelength values
        wavelength = []
        for k in columns:
            #print(k)
            wavelength.append(float(k))
        x = np.asarray(wavelength)

        # get the length of a column
        num = len(list(columns.values())[0])

        # Perform interpolation for each row
        for i in range(num):
            values = []
            for k in columns:
                #print("b")
                values.append(columns[k][i])
            y = np.asarray(values)

            new_y = sp.interpolate.interp1d(x, y)(new_x)
            return_y.append(new_y[0])

        return return_y


    @staticmethod
    def negReflectance(reflGroup, field, VIS = None):
        ''' Perform negative reflectance spectra checking '''
        # Run for entire file, not just one ensemble
        if VIS is None:
            VIS = [400,700]

        reflData = reflGroup.getDataset(field)
        # reflData.datasetToColumns()
        reflColumns = reflData.columns
        reflDate = reflColumns.pop('Datetag')
        reflTime = reflColumns.pop('Timetag2')
        # reflColumns.pop('Datetag')
        # reflColumns.pop('Timetag2')
        timeStamp = reflColumns.pop('Datetime')

        badTimes = []
        for indx, timeTag in enumerate(timeStamp):
            # If any spectra in the vis are negative, delete the whole spectrum
            reflVIS = []
            wavelengths = []
            for wave in reflColumns:
                wavelengths.append(float(wave))
                if float(wave) > VIS[0] and float(wave) < VIS[1]:
                    reflVIS.append(reflColumns[wave][indx])
                # elif float(wave) > NIR[0] and float(wave) < NIR[1]:
                #     ltNIR.append(ltColumns[wave][indx])

            # Flag entire record for removal
            if any(item < 0 for item in reflVIS):
                badTimes.append(timeTag)

            # Set negatives to 0
            NIR = [VIS[-1]+1,max(wavelengths)]
            UV = [min(wavelengths),VIS[0]-1]
            for wave in reflColumns:
                if ((float(wave) >= UV[0] and float(wave) < UV[1]) or \
                            (float(wave) >= NIR[0] and float(wave) <= NIR[1])) and \
                            reflColumns[wave][indx] < 0:
                    reflColumns[wave][indx] = 0

        badTimes = np.unique(badTimes)
        badTimes = np.rot90(np.matlib.repmat(badTimes,2,1), 3) # Duplicates each element to a list of two elements (start, stop)
        Utilities.writeLogFileAndPrint(f'{len(np.unique(badTimes))/len(timeStamp)*100:.1f}% of {field} spectra flagged')

        # # Need to add these at the beginning of the ODict
        reflColumns['Timetag2'] = reflTime
        reflColumns['Datetag'] = reflDate
        reflColumns['Datetime'] = timeStamp
        reflColumns.move_to_end('Timetag2', last=False)
        reflColumns.move_to_end('Datetag', last=False)
        reflColumns.move_to_end('Datetime', last=False)

        reflData.columnsToDataset()

        if len(badTimes) == 0:
            badTimes = None
        return badTimes


    @staticmethod
    def columnToSlice(columns, start, end):
        ''' Take a slice of a dataset stored in columns '''

        # Each column is a time series either at a waveband for radiometer columns, or various grouped datasets for ancillary
        # Start and end are defined by the interval established in the Config (they are indexes)
        newSlice = collections.OrderedDict()
        for col in columns:
            if start == end:
                newSlice[col] = columns[col][start:end+1] # otherwise you get nada []
            else:
                newSlice[col] = columns[col][start:end] # up to not including end...next slice will pick it up
        return newSlice


    @staticmethod
    def sliceAveHyper(y, hyperSlice):
        ''' Take the slice mean of the lowest X% of hyperspectral slices '''
        xSlice = collections.OrderedDict()
        xSliceRemaining = collections.OrderedDict()
        xMedian = collections.OrderedDict()
        hasNan = False
        # Ignore runtime warnings when array is all NaNs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for k in hyperSlice: # each k is a time series at a waveband.
                v = [hyperSlice[k][i] for i in y] # selects the lowest 5% within the interval window...
                mean = np.nanmean(v) # ... and averages them
                median = np.nanmedian(v) # ... and the median spectrum
                xSlice[k] = [mean]
                xMedian[k] = [median]
                if np.isnan(mean):
                    hasNan = True

                # Retain remaining spectra for use in calculating Rrs_sd
                xSliceRemaining[k] = v

        return hasNan, xSlice, xMedian, xSliceRemaining


    @staticmethod
    def sliceAveOther(node, start, end, y, ancGroup, sixSGroup):
        ''' Take the slice AND the mean averages of ancillary and 6S data with X% '''        

        def _sliceAveOther(node, start, end, y, group):
            if node.getGroup(group.id):
                newGroup = node.getGroup(group.id)
            else:
                newGroup = node.addGroup(group.id)

            for dsID in group.datasets:
                if newGroup.getDataset(dsID):
                    newDS = newGroup.getDataset(dsID)
                else:
                    newDS = newGroup.addDataset(dsID)
                ds = group.getDataset(dsID)

                # Set relAz to abs(relAz) prior to averaging
                if dsID == 'REL_AZ':
                    ds.columns['REL_AZ'] = np.abs(ds.columns['REL_AZ']).tolist()

                ds.datasetToColumns()
                dsSlice = ProcessL2.columnToSlice(ds.columns,start, end)
                dsXSlice, date, sliceTime, subDScol = None, None, None, None

                for subDScol in dsSlice: # each dataset contains columns (including date, time, data, and possibly flags)
                    if subDScol == 'Datetime':
                        timeStamp = dsSlice[subDScol]
                        # Stores the mean datetime by converting to (and back from) epoch second
                        if len(timeStamp) > 0:
                            epoch = datetime.datetime(1970, 1, 1,tzinfo=datetime.timezone.utc) #Unix zero hour
                            tsSeconds = []
                            for dt in timeStamp:
                                tsSeconds.append((dt-epoch).total_seconds())
                            meanSec = np.mean(tsSeconds)
                            dateTime = datetime.datetime.utcfromtimestamp(meanSec).replace(tzinfo=datetime.timezone.utc)
                            date = Utilities.datetime2DateTag(dateTime)
                            sliceTime = Utilities.datetime2TimeTag2(dateTime)
                    if subDScol not in ('Datetime', 'Datetag', 'Timetag2'):
                        v = [dsSlice[subDScol][i] for i in y] # y is an array of indexes for the lowest X%

                        if dsXSlice is None:
                            dsXSlice = collections.OrderedDict()
                            dsXSlice['Datetag'] = [date]
                            dsXSlice['Timetag2'] = [sliceTime]
                            dsXSlice['Datetime'] = [dateTime]

                        if subDScol not in dsXSlice:
                            dsXSlice[subDScol] = []
                        if (subDScol.endswith('FLAG')) or (subDScol.endswith('STATION')):
                            # Find the most frequest element
                            dsXSlice[subDScol].append(Utilities.mostFrequent(v))
                        else:
                            # Otherwise take a nanmean of the slice
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                dsXSlice[subDScol].append(np.nanmean(v)) # Warns of empty when empty...

                # Just test a sample column to see if it needs adding or appending
                if subDScol not in newDS.columns:
                    newDS.columns = dsXSlice
                else:
                    for item in newDS.columns:
                        newDS.columns[item] = np.append(newDS.columns[item], dsXSlice[item])

                newDS.columns.move_to_end('Timetag2', last=False)
                newDS.columns.move_to_end('Datetag', last=False)
                newDS.columns.move_to_end('Datetime', last=False)
                newDS.columnsToDataset()

        _sliceAveOther(node, start, end, y, ancGroup)        
        _sliceAveOther(node, start, end, y, sixSGroup)

    @staticmethod
    def ensemblesReflectance(node, sasGroup, refGroup, ancGroup, uncGroup,
                             esRawGroup, liRawGroup, ltRawGroup,
                             sixSGroup, start, end):
        '''Calculate the lowest X% Lt(780). Check for Nans in Li, Lt, Es, or wind. Send out for
        meteorological quality flags. Perform glint corrections. Calculate the Rrs. Correct for NIR
        residuals.'''

        esData = refGroup.getDataset("ES")
        liData = sasGroup.getDataset("LI")
        ltData = sasGroup.getDataset("LT")

        # Copy datasets to dictionary
        esData.datasetToColumns()
        esColumns = esData.columns
        liData.datasetToColumns()
        liColumns = liData.columns
        ltData.datasetToColumns()
        ltColumns = ltData.columns

        esSlice = ProcessL2.columnToSlice(esColumns,start, end)
        liSlice = ProcessL2.columnToSlice(liColumns,start, end)
        ltSlice = ProcessL2.columnToSlice(ltColumns,start, end)
        n = len(list(ltSlice.values())[0])

        # Test for ensemble shorter than 1 minute, which will have too few darks in Lt at L1AQC for uncertainties
        esStartTime = esSlice['Datetime'][0]
        esStopTime = esSlice['Datetime'][-1]
        if (esStopTime - esStartTime) < datetime.timedelta(seconds=60):
            Utilities.writeLogFileAndPrint('ProcessL2.ensemblesReflectance ensemble is less than 1 minute. Skipping.')
            return False
        #SIXS
        if sixSGroup is not None:
            diffuseData = sixSGroup.getDataset("diffuse_ratio")
            directData = sixSGroup.getDataset("direct_ratio")
            sixSesData = sixSGroup.getDataset("sixS_irradiance")
            # no need to retain SZA
            # Copy datasets to dictionary
            diffuseData.datasetToColumns()
            # diffuseColumns = diffuseData.columns
            directData.datasetToColumns()
            # directColumns = directData.columns
            sixSesData.datasetToColumns()
            # sixSesColumns = sixSesData.columns

            # diffuseSlice = ProcessL2.columnToSlice(diffuseColumns,start, end)
            # directSlice = ProcessL2.columnToSlice(directColumns,start, end)
            # sixSesSlice = ProcessL2.columnToSlice(sixSesColumns,start, end)

        # process raw groups for generating standard deviations
        def _sliceRawData(ES_raw, LI_raw, LT_raw):
            es_slce = {}
            li_slce = {}
            lt_slce = {}
            for (ds, ds1, ds2) in zip(ES_raw, LI_raw, LT_raw):
                if ConfigFile.settings['SensorType'].lower() == "seabird":
                    mask = [ds.id == "DATETIME", ds1.id == "DATETIME", ds2.id == "DATETIME"]
                    if any(mask):
                        # slice DATETIME
                        DS = [(i, d) for i, d in enumerate([ds, ds1, ds2]) if mask[i]]
                        for i, d in DS:
                            if i == 0:
                                es_slce["datetime"] = d.data[start:end]
                            if i == 1:
                                li_slce["datetime"] = d.data[start:end]
                            if i == 2:
                                lt_slce["datetime"] = d.data[start:end]
                if ds.id == "ES":
                    es_slce["data"] = ProcessL2.columnToSlice(ds.columns, start, end)
                if ds1.id == "LI":
                    li_slce["data"] = ProcessL2.columnToSlice(ds1.columns, start, end)
                if ds2.id == "LT":
                    lt_slce["data"] = ProcessL2.columnToSlice(ds2.columns, start, end)
                # Get Dalec Dark Counts
                if ds.id == "DARK_CNT":
                    es_slce["dc"] = ProcessL2.columnToSlice(ds.columns, start, end)
                if ds1.id == "DARK_CNT":
                    li_slce["dc"] = ProcessL2.columnToSlice(ds1.columns, start, end)
                if ds2.id == "DARK_CNT":
                    lt_slce["dc"] = ProcessL2.columnToSlice(ds2.columns, start, end)
            if all([es_slce, li_slce, lt_slce]):
                return es_slce, li_slce, lt_slce
            else:
                return False

        if not any([esRawGroup, liRawGroup, ltRawGroup]):
            Utilities.writeLogFileAndPrint("No L1AQC groups found")
        else:
            # slice L1AQC (aka "Raw" here) Data depending on SensorType
            if ConfigFile.settings['SensorType'].lower() == "trios" \
                or ConfigFile.settings['SensorType'].lower() == "dalec" or\
                      ConfigFile.settings['SensorType'].lower() == "sorad":
                esRawSlice, liRawSlice, ltRawSlice = _sliceRawData(
                                    esRawGroup.datasets.values(),
                                    liRawGroup.datasets.values(),
                                    ltRawGroup.datasets.values(),
                                    )
            elif ConfigFile.settings['SensorType'].lower() == "seabird":
                esRawSlice = dict()
                liRawSlice = dict()
                ltRawSlice = dict()

                esRawSlice['LIGHT'], liRawSlice['LIGHT'], ltRawSlice['LIGHT'] = \
                    _sliceRawData(
                                  esRawGroup['LIGHT'].datasets.values(),
                                  liRawGroup['LIGHT'].datasets.values(),
                                  ltRawGroup['LIGHT'].datasets.values(),
                                  )
                esRawSlice['DARK'], liRawSlice['DARK'], ltRawSlice['DARK'] = \
                    _sliceRawData(
                                  esRawGroup['DARK'].datasets.values(),
                                  liRawGroup['DARK'].datasets.values(),
                                  ltRawGroup['DARK'].datasets.values(),
                                  )
            else:
                Utilities.writeLogFileAndPrint("unrecognisable sensor type")
                return False

        rhoDefault = float(ConfigFile.settings["fL2RhoSky"])
        threeCRho = int(ConfigFile.settings["bL23CRho"])
        ZhangRho = int(ConfigFile.settings["bL2ZhangRho"])
        enablePercentLt = float(ConfigFile.settings["bL2EnablePercentLt"])
        percentLt = float(ConfigFile.settings["fL2PercentLt"])

        timeStamp = esSlice.pop("Datetime")
        esSlice.pop("Datetag")
        esSlice.pop("Timetag2")

        liSlice.pop("Datetag")
        liSlice.pop("Timetag2")
        liSlice.pop("Datetime")

        ltSlice.pop("Datetag")
        ltSlice.pop("Timetag2")
        ltSlice.pop("Datetime")

        # Process StdSlices for Band Convolution
        # Get common wavebands from esSlice to interp stats
        instrument_wb = np.asarray(list(esSlice.keys()), dtype=float)

        # rawGroup required only for some group attributes, Group data not used as is not ensemble.
        if ConfigFile.settings['SensorType'].lower() == "trios" or  ConfigFile.settings['SensorType'].lower() == "sorad":
            instrument = Trios()  # overwrites all Instrument class functions with TriOS specific ones
            stats = instrument.generateSensorStats("TriOS",
                        dict(ES=esRawGroup, LI=liRawGroup, LT=ltRawGroup),
                        dict(ES=esRawSlice, LI=liRawSlice, LT=ltRawSlice),
                        instrument_wb)
        elif ConfigFile.settings['SensorType'].lower() == "seabird":
            instrument = HyperOCR()  # overwrites all Instrument class functions with HyperOCR specific ones
            stats = instrument.generateSensorStats("SeaBird",
                        dict(ES=esRawGroup, LI=liRawGroup, LT=ltRawGroup),
                        dict(ES=esRawSlice, LI=liRawSlice, LT=ltRawSlice),
                        instrument_wb)
            # after dark substitution is done, condense to only dark corrected data (LIGHT key)
            esRawGroup = esRawGroup['LIGHT']
            liRawGroup = liRawGroup['LIGHT']
            ltRawGroup = ltRawGroup['LIGHT']
            esRawGroup.id = "ES_L1AQC"
            liRawGroup.id = "LI_L1AQC"
            ltRawGroup.id = "LT_L1AQC"
        elif ConfigFile.settings['SensorType'].lower() == "dalec":
            instrument = Dalec()  # overwrites all Instrument class functions with Dalec specific ones
            stats = instrument.generateSensorStats("Dalec",
                        dict(ES=esRawGroup, LI=liRawGroup, LT=ltRawGroup),
                        dict(ES=esRawSlice, LI=liRawSlice, LT=ltRawSlice),
                        instrument_wb)
        else:
            Utilities.writeLogFileAndPrint("class type not recognised")
            return False

        if not stats:
            Utilities.writeLogFileAndPrint("statistics not generated")
            return False

        # make std into dictionaries (data are ODs, but should not matter)
        esStdSlice = {k: [stats['ES']['std_Signal_Interpolated'][k][0]] for k in esSlice}
        liStdSlice = {k: [stats['LI']['std_Signal_Interpolated'][k][0]] for k in liSlice}
        ltStdSlice = {k: [stats['LT']['std_Signal_Interpolated'][k][0]] for k in ltSlice}

        # Convolve es/li/lt slices to satellite bands using RSRs
        if ConfigFile.settings['bL2WeightMODISA']:
            print("Convolving MODIS Aqua (ir)radiances in the slice")
            esSliceMODISA = Weight_RSR.processMODISBands(esSlice, sensor='A')
            liSliceMODISA = Weight_RSR.processMODISBands(liSlice, sensor='A')
            ltSliceMODISA = Weight_RSR.processMODISBands(ltSlice, sensor='A')
            esXstdMODISA = Weight_RSR.processMODISBands(esStdSlice, sensor='A')
            liXstdMODISA = Weight_RSR.processMODISBands(liStdSlice, sensor='A')
            ltXstdMODISA = Weight_RSR.processMODISBands(ltStdSlice, sensor='A')
        if ConfigFile.settings['bL2WeightMODIST']:
            print("Convolving MODIS Terra (ir)radiances in the slice")
            esSliceMODIST = Weight_RSR.processMODISBands(esSlice, sensor='T')
            liSliceMODIST = Weight_RSR.processMODISBands(liSlice, sensor='T')
            ltSliceMODIST = Weight_RSR.processMODISBands(ltSlice, sensor='T')
            esXstdMODIST = Weight_RSR.processMODISBands(esStdSlice, sensor='T')
            liXstdMODIST = Weight_RSR.processMODISBands(liStdSlice, sensor='T')
            ltXstdMODIST = Weight_RSR.processMODISBands(ltStdSlice, sensor='T')
        if ConfigFile.settings['bL2WeightVIIRSN']:
            print("Convolving VIIRS NPP (ir)radiances in the slice")
            esSliceVIIRSN = Weight_RSR.processVIIRSBands(esSlice, sensor='N')
            liSliceVIIRSN = Weight_RSR.processVIIRSBands(liSlice, sensor='N')
            ltSliceVIIRSN = Weight_RSR.processVIIRSBands(ltSlice, sensor='N')
            esXstdVIIRSN = Weight_RSR.processVIIRSBands(esStdSlice, sensor='N')
            liXstdVIIRSN = Weight_RSR.processVIIRSBands(liStdSlice, sensor='N')
            ltXstdVIIRSN = Weight_RSR.processVIIRSBands(ltStdSlice, sensor='N')
        if ConfigFile.settings['bL2WeightVIIRSJ']:
            print("Convolving VIIRS JPSS (ir)radiances in the slice")
            esSliceVIIRSJ = Weight_RSR.processVIIRSBands(esSlice, sensor='N')
            liSliceVIIRSJ = Weight_RSR.processVIIRSBands(liSlice, sensor='N')
            ltSliceVIIRSJ = Weight_RSR.processVIIRSBands(ltSlice, sensor='N')
            esXstdVIIRSJ = Weight_RSR.processVIIRSBands(esStdSlice, sensor='N')
            liXstdVIIRSJ = Weight_RSR.processVIIRSBands(liStdSlice, sensor='N')
            ltXstdVIIRSJ = Weight_RSR.processVIIRSBands(ltStdSlice, sensor='N')
        if ConfigFile.settings['bL2WeightSentinel3A']:
            print("Convolving Sentinel 3A (ir)radiances in the slice")
            esSliceSentinel3A = Weight_RSR.processSentinel3Bands(esSlice, sensor='A')
            liSliceSentinel3A = Weight_RSR.processSentinel3Bands(liSlice, sensor='A')
            ltSliceSentinel3A = Weight_RSR.processSentinel3Bands(ltSlice, sensor='A')
            esXstdSentinel3A = Weight_RSR.processSentinel3Bands(esStdSlice, sensor='A')
            liXstdSentinel3A = Weight_RSR.processSentinel3Bands(liStdSlice, sensor='A')
            ltXstdSentinel3A = Weight_RSR.processSentinel3Bands(ltStdSlice, sensor='A')
        if ConfigFile.settings['bL2WeightSentinel3B']:
            print("Convolving Sentinel 3B (ir)radiances in the slice")
            esSliceSentinel3B = Weight_RSR.processSentinel3Bands(esSlice, sensor='B')
            liSliceSentinel3B = Weight_RSR.processSentinel3Bands(liSlice, sensor='B')
            ltSliceSentinel3B = Weight_RSR.processSentinel3Bands(ltSlice, sensor='B')
            esXstdSentinel3B = Weight_RSR.processSentinel3Bands(esStdSlice, sensor='B')
            liXstdSentinel3B = Weight_RSR.processSentinel3Bands(liStdSlice, sensor='B')
            ltXstdSentinel3B = Weight_RSR.processSentinel3Bands(ltStdSlice, sensor='B')

        # Store the mean datetime of the slice
        if len(timeStamp) > 0:
            epoch = datetime.datetime(1970, 1, 1,tzinfo=datetime.timezone.utc) #Unix zero hour
            tsSeconds = []
            for dt in timeStamp:
                tsSeconds.append((dt-epoch).total_seconds())
            meanSec = np.mean(tsSeconds)
            dateTime = datetime.datetime.utcfromtimestamp(meanSec).replace(tzinfo=datetime.timezone.utc)
            dateTag = Utilities.datetime2DateTag(dateTime)
            timeTag = Utilities.datetime2TimeTag2(dateTime)

            timeObj = {}
            timeObj['dateTime'] = dateTime
            timeObj['dateTag'] = dateTag
            timeObj['timeTag'] = timeTag

        # Calculates the lowest X% (based on Hooker & Morel 2003; Hooker et al. 2002; Zibordi et al. 2002, IOCCG Protocols)
        # X will depend on FOV and integration time of instrument. Hooker cites a rate of 2 Hz.
        # It remains unclear to me from Hooker 2002 whether the recommendation is to take the average of the ir/radiances
        # within the threshold and calculate Rrs, or to calculate the Rrs within the threshold, and then average, however IOCCG
        # Protocols pretty clearly state to average the ir/radiances first, then calculate the Rrs...as done here.
        x = round(n*percentLt/100) # number of retained values
        Utilities.writeLogFileAndPrint(f'{n} spectra in slice (ensemble).')

        # There are sometimes only a small number of spectra in the slice,
        #  so the percent Lt estimation becomes highly questionable and is overridden here.
        if n <= 5 or x == 0:
            x = n  # if only 5 or fewer records retained, use them all...

        if enablePercentLt and x > 1:
            # Find the indexes for the lowest X%
            lt780 = ProcessL2.interpolateColumn(ltSlice, 780.0)
            index = np.argsort(lt780)  # gives indexes if values were to be sorted
            # returns indexes of the first x values (if values were sorted); i.e. the indexes of the lowest X% of unsorted lt780
            y = index[0:x]
            Utilities.writeLogFileAndPrint(f'{len(y)} spectra remaining in slice to average after filtering to lowest {percentLt}%.')
        else:
            # If Percent Lt is turned off, this will average the whole slice, and if
            # ensemble is off (set to 0), just the one spectrum will be used.
            first_band = next(iter(ltSlice))
            first_band_values = ltSlice[first_band]
            y=list(range(0,len(first_band_values)))

        EnsembleN = len(y) # After taking lowest X%
        if 'Ensemble_N' not in node.getGroup('REFLECTANCE').datasets:
            node.getGroup('REFLECTANCE').addDataset('Ensemble_N')
            node.getGroup('IRRADIANCE').addDataset('Ensemble_N')
            node.getGroup('RADIANCE').addDataset('Ensemble_N')
            node.getGroup('REFLECTANCE').datasets['Ensemble_N'].columns['N'] = []
            node.getGroup('IRRADIANCE').datasets['Ensemble_N'].columns['N'] = []
            node.getGroup('RADIANCE').datasets['Ensemble_N'].columns['N'] = []
        node.getGroup('REFLECTANCE').datasets['Ensemble_N'].columns['N'].append(EnsembleN)
        node.getGroup('IRRADIANCE').datasets['Ensemble_N'].columns['N'].append(EnsembleN)
        node.getGroup('RADIANCE').datasets['Ensemble_N'].columns['N'].append(EnsembleN)
        node.getGroup('REFLECTANCE').datasets['Ensemble_N'].columnsToDataset()
        node.getGroup('IRRADIANCE').datasets['Ensemble_N'].columnsToDataset()
        node.getGroup('RADIANCE').datasets['Ensemble_N'].columnsToDataset()

        # Take the mean of the lowest X% in the slice
        sliceAveFlag = []
        flag,esXSlice,esXmedian,esXRemaining = ProcessL2.sliceAveHyper(y, esSlice)
        sliceAveFlag.append(flag)
        flag, liXSlice, liXmedian, liXRemaining = ProcessL2.sliceAveHyper(y, liSlice)
        sliceAveFlag.append(flag)
        flag, ltXSlice, ltXmedian, ltXRemaining = ProcessL2.sliceAveHyper(y, ltSlice)
        sliceAveFlag.append(flag)

        # Take the mean of the lowest X% for satellite weighted (ir)radiances in the slice
        # y indexes are from the hyperspectral data
        if ConfigFile.settings['bL2WeightMODISA']:
            flag, esXSliceMODISA, esXmedianMODISA, esXRemainingMODISA = ProcessL2.sliceAveHyper(y, esSliceMODISA)
            sliceAveFlag.append(flag)
            flag, liXSliceMODISA, liXmedianMODISA, liXRemainingMODISA = ProcessL2.sliceAveHyper(y, liSliceMODISA)
            sliceAveFlag.append(flag)
            # flag, liXSliceMODISA, liXmedianMODISA = ProcessL2.sliceAveHyper(y, liSliceMODISA)
            # sliceAveFlag.append(flag)
            flag, ltXSliceMODISA, ltXmedianMODISA, ltXRemainingMODISA = ProcessL2.sliceAveHyper(y, ltSliceMODISA)
            sliceAveFlag.append(flag)
        if ConfigFile.settings['bL2WeightMODIST']:
            flag, esXSliceMODIST, esXmedianMODIST, esXRemainingMODIST = ProcessL2.sliceAveHyper(y, esSliceMODIST)
            sliceAveFlag.append(flag)
            flag, liXSliceMODIST, liXmedianMODIST, liXRemainingMODIST = ProcessL2.sliceAveHyper(y, liSliceMODIST)
            sliceAveFlag.append(flag)
            flag, ltXSliceMODIST, ltXmedianMODIST, ltXRemainingMODIST = ProcessL2.sliceAveHyper(y, ltSliceMODIST)
            sliceAveFlag.append(flag)
        if ConfigFile.settings['bL2WeightVIIRSN']:
            flag, esXSliceVIIRSN, esXmedianVIIRSN, esXRemainingVIIRSN = ProcessL2.sliceAveHyper(y, esSliceVIIRSN)
            sliceAveFlag.append(flag)
            flag, liXSliceVIIRSN, liXmedianVIIRSN, liXRemainingVIIRSN = ProcessL2.sliceAveHyper(y, liSliceVIIRSN)
            sliceAveFlag.append(flag)
            flag, ltXSliceVIIRSN, ltXmedianVIIRSN, ltXRemainingVIIRSN = ProcessL2.sliceAveHyper(y, ltSliceVIIRSN)
            sliceAveFlag.append(flag)
        if ConfigFile.settings['bL2WeightVIIRSJ']:
            flag, esXSliceVIIRSJ, esXmedianVIIRSJ, esXRemainingVIIRSJ = ProcessL2.sliceAveHyper(y, esSliceVIIRSJ)
            sliceAveFlag.append(flag)
            flag, liXSliceVIIRSJ, liXmedianVIIRSJ, liXRemainingVIIRSJ = ProcessL2.sliceAveHyper(y, liSliceVIIRSJ)
            sliceAveFlag.append(flag)
            flag, ltXSliceVIIRSJ, ltXmedianVIIRSJ, ltXRemainingVIIRSJ = ProcessL2.sliceAveHyper(y, ltSliceVIIRSJ)
            sliceAveFlag.append(flag)
        if ConfigFile.settings['bL2WeightSentinel3A']:
            flag, esXSliceSentinel3A, esXmedianSentinel3A, esXRemainingSentinel3A = ProcessL2.sliceAveHyper(y, esSliceSentinel3A)
            sliceAveFlag.append(flag)
            flag, liXSliceSentinel3A, liXmedianSentinel3A, liXRemainingSentinel3A = ProcessL2.sliceAveHyper(y, liSliceSentinel3A)
            sliceAveFlag.append(flag)
            flag, ltXSliceSentinel3A, ltXmedianSentinel3A, ltXRemainingSentinel3A = ProcessL2.sliceAveHyper(y, ltSliceSentinel3A)
            sliceAveFlag.append(flag)
        if ConfigFile.settings['bL2WeightSentinel3B']:
            flag, esXSliceSentinel3B, esXmedianSentinel3B, esXRemainingSentinel3B = ProcessL2.sliceAveHyper(y, esSliceSentinel3B)
            sliceAveFlag.append(flag)
            flag, liXSliceSentinel3B, liXmedianSentinel3B, liXRemainingSentinel3B = ProcessL2.sliceAveHyper(y, liSliceSentinel3B)
            sliceAveFlag.append(flag)
            flag, ltXSliceSentinel3B, ltXmedianSentinel3B, ltXRemainingSentinel3B= ProcessL2.sliceAveHyper(y, ltSliceSentinel3B)
            sliceAveFlag.append(flag)

        # Make sure the XSlice averaging didn't bomb
        if np.isnan(sliceAveFlag).any():
            Utilities.writeLogFileAndPrint('ProcessL2.ensemblesReflectance: Slice X"%" average error: Dataset all NaNs.')
            return False

        # Take the mean of the lowest X% for the ancillary group in the slice
        # (Combines Slice and XSlice -- as above -- into one method)
        ProcessL2.sliceAveOther(node, start, end, y, ancGroup, sixSGroup)
        newAncGroup = node.getGroup("ANCILLARY") # Just populated above
        newAncGroup.attributes['ANC_SOURCE_FLAGS'] = ['0: Undetermined, 1: Field, 2: Model, 3: Fallback']

        # Extract the last/current element/slice for each dataset and hold for use in calculating reflectances
        # Ancillary group, unlike most groups, will have named data columns in datasets (i.e. not NONE)
        # This allows for multiple data arrays in one dataset (e.g. FLAGS)

        # These are required and will have been filled in with field data, models, and or defaults
        WINDSPEEDXSlice = newAncGroup.getDataset('WINDSPEED').data['WINDSPEED'][-1].copy()
        if isinstance(WINDSPEEDXSlice, list):
            WINDSPEEDXSlice = WINDSPEEDXSlice[0]
        SZAXSlice = newAncGroup.getDataset('SZA').data['SZA'][-1].copy()
        if isinstance(SZAXSlice, list):
            SZAXSlice = SZAXSlice[0]
        SSTXSlice = newAncGroup.getDataset('SST').data['SST'][-1].copy()
        if isinstance(SSTXSlice, list):
            SSTXSlice = SSTXSlice[0]
        # if 'SAL' in newAncGroup.datasets:
        #     SalXSlice = newAncGroup.getDataset('SAL').data['SAL'][-1].copy()
        if 'SALINITY' in newAncGroup.datasets:
            SalXSlice = newAncGroup.getDataset('SALINITY').data['SALINITY'][-1].copy()
        if isinstance(SalXSlice, list):
            SalXSlice = SalXSlice[0]
        RelAzXSlice = newAncGroup.getDataset('REL_AZ').data['REL_AZ'][-1].copy()
        if isinstance(RelAzXSlice, list):
            RelAzXSlice = RelAzXSlice[0]

        RelAzXSlice = abs(RelAzXSlice)

        # Only required in Zhang17 currently
        try:
            AODXSlice = newAncGroup.getDataset('AOD').data['AOD'][-1].copy()
            if isinstance(AODXSlice, list):
                AODXSlice = AODXSlice[0]
        except Exception:
            if ZhangRho:
                Utilities.writeLogFileAndPrint('ProcessL2.ensemblesReflectance: No AOD data present in Ancillary. Activate model acquisition in L1B.')
                return False

        # These are optional; in fact, there is no implementation of incorporating CLOUD or WAVEs into
        # any of the current Rho corrections yet (even though cloud IS passed to Zhang_Rho)
        if "CLOUD" in newAncGroup.datasets:
            CloudXSlice = newAncGroup.getDataset('CLOUD').data['CLOUD'].copy()
            if isinstance(CloudXSlice, list):
                CloudXSlice = CloudXSlice[0]
        else:
            CloudXSlice = None
        # if "WAVE_HT" in newAncGroup.datasets:
        #     WaveXSlice = newAncGroup.getDataset('WAVE_HT').data['WAVE_HT'].copy()
        #     if isinstance(WaveXSlice, list):
        #         WaveXSlice = WaveXSlice[0]
        # else:
        #     WaveXSlice = None
        # if "STATION" in newAncGroup.datasets:
        #     StationSlice = newAncGroup.getDataset('STATION').data['STATION'].copy()
        #     if isinstance(StationSlice, list):
        #         StationSlice = StationSlice[0]
        # else:
        #     StationSlice = None

        ########################################################################
        # Calculate Rho_sky
        wavebands = [*esColumns] # just grabs the keys
        wavelength = []
        wavelengthStr = []
        for k in wavebands:
            if k != "Datetag" and k != "Datetime" and k != "Timetag2":
                wavelengthStr.append(k)
                wavelength.append(float(k))
        waveSubset = wavelength  # Only used for Zhang; No subsetting for threeC or Mobley corrections
        rhoVec = {}

        Rho_Uncertainty_Obj = Propagate(M=10, cores=1)

        # Rho will be the same across the entire ensemble slice based on input averages
        if threeCRho:
            # NOTE: Placeholder for Groetsch et al. 2017

            li750 = ProcessL2.interpolateColumn(liXSlice, 750.0)
            es750 = ProcessL2.interpolateColumn(esXSlice, 750.0)
            sky750 = li750[0]/es750[0]

            rhoScalar, rhoUNC = RhoCorrections.threeCCorr(sky750, rhoDefault, WINDSPEEDXSlice)
            # The above is not wavelength dependent. No need for seperate values/vectors for satellites
            rhoVec = None

        elif ZhangRho:
            # Zhang rho is based on Zhang et al. 2017 and calculates the wavelength-dependent rho vector
            # separated for sun and sky to include polarization factors.

            # Model limitations: AOD 0 - 0.5, Solar zenith 0-60 deg, Wavelength 350-1000 nm, SVA 30 or 40 degrees.

            # reduce number of draws because of how computationally intensive the Zhang method is
            Rho_Uncertainty_Obj = Propagate(M=10, cores=1)

            # Need to limit the input for the model limitations. This will also mean cutting out Li, Lt, and Es
            # from non-valid wavebands.
            if AODXSlice >0.5:
                Utilities.writeLogFileAndPrint(f'AOD = {AODXSlice:.3f}. Maximum Aerosol Optical Depth Reached. Setting to 0.5. Expect larger, uncaptured errors.')
                AODXSlice = 0.5
            if WINDSPEEDXSlice > 15:
                Utilities.writeLogFileAndPrint(f'WIND = {WINDSPEEDXSlice:.1f}. Maximum Wind Speed Reached. Setting to 15.0. Expect larger, uncaptured errors.')
                WINDSPEEDXSlice = 15
            if SZAXSlice > 60:
                # Zhang is stricter and limited to SZA <= 60
                # Utilities.writeLogFileAndPrint(f'SZA = {SZAXSlice:.2f}. Maximum Solar Zenith Exceeded. Aborting slice.'
                Utilities.writeLogFileAndPrint(f'SZA = {SZAXSlice:.1f}. Maximum Solar Zenith Exceeded. Setting to 60. Expect larger, uncaptured errors.')
                SZAXSlice = 60
            if min(wavelength) < 350 or max(wavelength) > 1000:
                Utilities.writeLogFileAndPrint('Wavelengths extend beyond model limits. Truncating to 350 - 1000 nm.')
                wave_old = wavelength.copy()
                wave_list = [(i, band) for i, band in enumerate(wave_old) if (band >=350) and (band <= 1000)]
                wave_array = np.array(wave_list)
                # wavelength is now truncated to only valid wavebands for use in Zhang models
                waveSubset = wave_array[:,1].tolist()

            SVA = ConfigFile.settings['fL2SVA']

            rhoVector, rhoUNC = RhoCorrections.ZhangCorr(WINDSPEEDXSlice,AODXSlice, CloudXSlice, SZAXSlice, SSTXSlice, SalXSlice, RelAzXSlice,
                                                          SVA, waveSubset, Rho_Uncertainty_Obj)

            for i, k in enumerate(waveSubset):
                rhoVec[str(k)] = rhoVector[i]

            rhoScalar = None

        else:
            # Full Mobley 1999 model from LUT
            try:
                AODXSlice = newAncGroup.getDataset('AOD').data['AOD'][-1].copy()
                if isinstance(AODXSlice, list):
                    AODXSlice = AODXSlice[0]
                rhoScalar, rhoUNC = RhoCorrections.M99Corr(WINDSPEEDXSlice, SZAXSlice, RelAzXSlice,
                                                             Rho_Uncertainty_Obj,
                                                             AOD=AODXSlice, cloud=CloudXSlice, wTemp=SSTXSlice,
                                                             sal=SalXSlice, waveBands=waveSubset)
            except NameError:
                rhoScalar, rhoUNC = RhoCorrections.M99Corr(WINDSPEEDXSlice, SZAXSlice, RelAzXSlice,
                                                             Rho_Uncertainty_Obj)

        # Calculate hyperspectral Coddingtion TSIS_1 hybrid F0 function
        # NOTE: TSIS uncertainties reported as 1-sigma
        F0_hyper, F0_unc, F0_raw, F0_unc_raw, wv_raw = Utilities.TSIS_1(dateTag, wavelength)
        # Recycling _raw in TSIS_1 calls below prevents the dataset having to be reread

        if F0_hyper is None:
            Utilities.writeLogFileAndPrint("No hyperspectral TSIS-1 F0. Aborting.")
            return False

        # Calculate TSIS-1 for each of the satellite bandsets
        if ConfigFile.settings['bL2WeightMODISA'] or ConfigFile.settings['bL2WeightMODIST']:
            MODISwavelength = Weight_RSR.MODISBands()
            wave_old = MODISwavelength.copy()
            wave_list = [(i, band) for i, band in enumerate(wave_old) if (band >=350) and (band <= 1000)]
            wave_array = np.array(wave_list)
            # wavelength is now truncated to only valid wavebands for use in Zhang models
            waveSubsetMODIS = wave_array[:,1].tolist()
            F0_MODIS,F0_MODIS_unc = Utilities.TSIS_1(dateTag, MODISwavelength, F0_raw, F0_unc_raw, wv_raw)[0:2]
        if ConfigFile.settings['bL2WeightVIIRSN'] or ConfigFile.settings['bL2WeightVIIRSJ']:
            VIIRSwavelength = Weight_RSR.VIIRSBands()
            wave_old = VIIRSwavelength.copy()
            wave_list = [(i, band) for i, band in enumerate(wave_old) if (band >=350) and (band <= 1000)]
            wave_array = np.array(wave_list)
            # wavelength is now truncated to only valid wavebands for use in Zhang models
            waveSubsetVIIRS = wave_array[:,1].tolist()
            F0_VIIRS, F0_VIIRS_unc = Utilities.TSIS_1(dateTag, VIIRSwavelength, F0_raw, F0_unc_raw, wv_raw)[0:2]
        if ConfigFile.settings['bL2WeightSentinel3A'] or ConfigFile.settings['bL2WeightSentinel3B']:
            Sentinel3wavelength = Weight_RSR.Sentinel3Bands()
            wave_old = Sentinel3wavelength.copy()
            wave_list = [(i, band) for i, band in enumerate(wave_old) if (band >=350) and (band <= 1000)]
            wave_array = np.array(wave_list)
            # wavelength is now truncated to only valid wavebands for use in Zhang models
            waveSubsetSentinel3 = wave_array[:,1].tolist()
            F0_Sentinel3, F0_Sentinel3_unc = Utilities.TSIS_1(dateTag, Sentinel3wavelength, F0_raw, F0_unc_raw, wv_raw)[0:2]

        # Build a slice object for (ir)radiances to be passed to spectralReflectance method
        # These slices are unique and independant of node data or earlier slices in the same node object
        xSlice = {}
        # Full hyperspectral
        sensor = 'HYPER'
        # Means:
        xSlice['es'] = esXSlice  # this sometimes has negative values because of instrument noise, we should take the absolute
        xSlice['li'] = liXSlice
        xSlice['lt'] = ltXSlice
        # Medians:
        xSlice['esMedian'] = esXmedian
        xSlice['liMedian'] = liXmedian
        xSlice['ltMedian'] = ltXmedian

        xSlice['esSTD'] = esStdSlice  # standard deviation at common wavebands
        xSlice['liSTD'] = liStdSlice
        xSlice['ltSTD'] = ltStdSlice

        xSlice['esSTD_RAW'] = stats['ES']['std_Signal']  # non-interpolated std for uncertainty calculation
        xSlice['liSTD_RAW'] = stats['LI']['std_Signal']
        xSlice['ltSTD_RAW'] = stats['LT']['std_Signal']

        xSlice['esRemaining'] = esXRemaining
        xSlice['liRemaining'] = liXRemaining
        xSlice['ltRemaining'] = ltXRemaining

        # insert Uncertainties into analysis
        xUNC = {}

        tic = time.process_time()
        with warnings.catch_warnings(action="ignore"):  # added to suppress comet-maths warnings which clog up terminal
            if ConfigFile.settings["fL1bCal"] <= 2:  # and
                L1B_UNC = instrument.ClassBased(node, uncGroup, stats)
                if L1B_UNC:
                    xSlice.update(L1B_UNC)  # update the xSlice dict with uncertianties and samples
                    del L1B_UNC  # delete to save memory as no longer required
                    # convert uncertainties back into absolute form using the signals recorded from ProcessL2
                    xSlice['esUnc'] = {u[0]: [u[1][0]*np.abs(s[0])] for u, s in zip(xSlice['esUnc'].items(), esXSlice.values())}
                    xSlice['liUnc'] = {u[0]: [u[1][0]*np.abs(s[0])] for u, s in zip(xSlice['liUnc'].items(), liXSlice.values())}
                    xSlice['ltUnc'] = {u[0]: [u[1][0]*np.abs(s[0])] for u, s in zip(xSlice['ltUnc'].items(), ltXSlice.values())}

                    xUNC.update(instrument.ClassBasedL2(node, uncGroup, rhoScalar, rhoVec, rhoUNC, waveSubset, xSlice))
                elif ((ConfigFile.settings['SensorType'].lower() == "trios") or \
                    (ConfigFile.settings['SensorType'].lower() == "dalec")) and (ConfigFile.settings["fL1bCal"] == 1):
                    xUNC = None
                else:
                    Utilities.writeLogFileAndPrint("Instrument uncertainty processing failed: ProcessL2")
                    return False

            elif ConfigFile.settings["fL1bCal"] == 3:
                xSlice.update(
                    instrument.FRM(node, uncGroup,
                                dict(ES=esRawGroup, LI=liRawGroup, LT=ltRawGroup),
                                dict(ES=esRawSlice, LI=liRawSlice, LT=ltRawSlice),
                                stats, np.array(waveSubset, float)))  # instrument_WB
                xUNC.update(instrument.FRM_L2(rhoScalar, rhoVec, rhoUNC, waveSubset, xSlice))

                if ConfigFile.settings['bL2UncertaintyBreakdownPlot']:
                    from Source.Uncertainty_Visualiser import UncertaintyGUI
                    gui = UncertaintyGUI()
                    gui.plot_FRM(
                        node,
                        uncGroup,
                        dict(ES=esRawGroup, LI=liRawGroup, LT=ltRawGroup),
                        dict(ES=esRawSlice, LI=liRawSlice, LT=ltRawSlice),
                        stats,
                        rhoScalar,
                        rhoVec,
                        rhoUNC,
                        waveSubset
                    )

        Utilities.writeLogFileAndPrint(f'Uncertainty Update Elapsed Time: {time.process_time() - tic:.1f} s')

        # move uncertainties from xSlice to xUNC
        if xUNC is not None:
            for sliceKey in list(xSlice.keys()):
                if "sample" in sliceKey.lower():
                    xSlice.pop(sliceKey)  # samples are no longer needed
                elif "unc" in sliceKey.lower():
                    xUNC[f"{sliceKey[0:2]}UNC_HYPER"] = xSlice.pop(sliceKey)  # transfer instrument uncs to xUNC

            # for convolving to satellite bands
            esUNCSlice = xUNC["esUNC_HYPER"]  # ODicts... whereas lwUNC and rrsUNC are simple arrays
            liUNCSlice = xUNC["liUNC_HYPER"]
            ltUNCSlice = xUNC["ltUNC_HYPER"]

        # Populate the relevant fields in node
        ProcessL2.spectralReflectance(node, sensor, timeObj, xSlice, F0_hyper, F0_unc, rhoScalar, rhoVec, waveSubset, xUNC)

        # Apply residual NIR corrections
        # Perfrom near-infrared residual correction to remove additional atmospheric and glint contamination
        if ConfigFile.settings["bL2PerformNIRCorrection"]:
            rrsNIRCorr, nLwNIRCorr = ProcessL2.nirCorrection(node, sensor, F0_hyper)

        # Satellites
        if ConfigFile.settings['bL2WeightMODISA'] or ConfigFile.settings['bL2WeightMODIST']:
            # F0 = F0_MODIS
            rhoVecMODIS = None

            if ConfigFile.settings['bL2WeightMODISA']:
                print('Processing MODISA')

                if ZhangRho:
                    rhoVecMODIS = Weight_RSR.processMODISBands(rhoVec,sensor='A')
                    # Weight_RSR process is designed to return list of lists in the ODict; convert to list
                    rhoVecMODIS = {key:value[0] for (key,value) in rhoVecMODIS.items()}

                xSlice['es'] = esXSliceMODISA
                xSlice['li'] = liXSliceMODISA
                xSlice['lt'] = ltXSliceMODISA

                xSlice['esRemaining'] = esXRemainingMODISA
                xSlice['liRemaining'] = liXRemainingMODISA
                xSlice['ltRemaining'] = ltXRemainingMODISA

                xSlice['esMedian'] = esXmedianMODISA
                xSlice['liMedian'] = liXmedianMODISA
                xSlice['ltMedian'] = ltXmedianMODISA

                xSlice['esSTD'] = esXstdMODISA
                xSlice['liSTD'] = liXstdMODISA
                xSlice['ltSTD'] = ltXstdMODISA

                # NOTE: According to AR, this may not be a robust way of estimating convolved uncertainties.
                # He has implemented another way, but it is very slow due to multiple MC runs. Comment this out
                # for now, but a sensitivity analysis may show it to be okay.
                # NOTE: 1/2024 Why is this not commented out if the slow, more accurate way is now implemented?
                if xUNC is not None:
                    xUNC['esUNC'] = Weight_RSR.processMODISBands(esUNCSlice, sensor='A')
                    xUNC['liUNC'] = Weight_RSR.processMODISBands(liUNCSlice, sensor='A')
                    xUNC['ltUNC'] = Weight_RSR.processMODISBands(ltUNCSlice, sensor='A')

                sensor = 'MODISA'
                ProcessL2.spectralReflectance(node, sensor, timeObj, xSlice, F0_MODIS, F0_MODIS_unc, rhoScalar, rhoVecMODIS, waveSubsetMODIS, xUNC)
                if ConfigFile.settings["bL2PerformNIRCorrection"]:
                    # Can't apply good NIR corrs at satellite bands, so use the correction factors from the hyperspectral instead.
                    ProcessL2.nirCorrectionSatellite(node, sensor, rrsNIRCorr, nLwNIRCorr)

            if ConfigFile.settings['bL2WeightMODIST']:
                print('Processing MODIST')

                if ZhangRho:
                    rhoVecMODIS = Weight_RSR.processMODISBands(rhoVec,sensor='T')
                    rhoVecMODIS = {key:value[0] for (key,value) in rhoVecMODIS.items()}

                xSlice['es'] = esXSliceMODIST
                xSlice['li'] = liXSliceMODIST
                xSlice['lt'] = ltXSliceMODIST

                xSlice['esRemaining'] = esXRemainingMODIST
                xSlice['liRemaining'] = liXRemainingMODIST
                xSlice['ltRemaining'] = ltXRemainingMODIST

                xSlice['esMedian'] = esXmedianMODIST
                xSlice['liMedian'] = liXmedianMODIST
                xSlice['ltMedian'] = ltXmedianMODIST

                xSlice['esSTD'] = esXstdMODIST
                xSlice['liSTD'] = liXstdMODIST
                xSlice['ltSTD'] = ltXstdMODIST

                if xUNC is not None:
                    xUNC['esUNC'] = Weight_RSR.processMODISBands(esUNCSlice, sensor='T')
                    xUNC['liUNC'] = Weight_RSR.processMODISBands(liUNCSlice, sensor='T')
                    xUNC['ltUNC'] = Weight_RSR.processMODISBands(ltUNCSlice, sensor='T')

                sensor = 'MODIST'
                ProcessL2.spectralReflectance(node, sensor, timeObj, xSlice, F0_MODIS, F0_MODIS_unc, rhoScalar, rhoVecMODIS, waveSubsetMODIS,  xUNC)
                if ConfigFile.settings["bL2PerformNIRCorrection"]:
                    # Can't apply good NIR corrs at satellite bands, so use the correction factors from the hyperspectral instead.
                    ProcessL2.nirCorrectionSatellite(node, sensor, rrsNIRCorr, nLwNIRCorr)

        if ConfigFile.settings['bL2WeightVIIRSN'] or ConfigFile.settings['bL2WeightVIIRSJ']:
            # F0 = F0_VIIRS
            rhoVecVIIRS = None

            if ConfigFile.settings['bL2WeightVIIRSN']:
                print('Processing VIIRSN')

                if ZhangRho:
                    rhoVecVIIRS = Weight_RSR.processVIIRSBands(rhoVec,sensor='A')
                    rhoVecVIIRS = {key:value[0] for (key,value) in rhoVecVIIRS.items()}

                xSlice['es'] = esXSliceVIIRSN
                xSlice['li'] = liXSliceVIIRSN
                xSlice['lt'] = ltXSliceVIIRSN

                xSlice['esRemaining'] = esXRemainingVIIRSN
                xSlice['liRemaining'] = liXRemainingVIIRSN
                xSlice['ltRemaining'] = ltXRemainingVIIRSN

                xSlice['esMedian'] = esXmedianVIIRSN
                xSlice['liMedian'] = liXmedianVIIRSN
                xSlice['ltMedian'] = ltXmedianVIIRSN

                xSlice['esSTD'] = esXstdVIIRSN
                xSlice['liSTD'] = liXstdVIIRSN
                xSlice['ltSTD'] = ltXstdVIIRSN

                if xUNC is not None:
                    xUNC['esUNC'] = Weight_RSR.processVIIRSBands(esUNCSlice, sensor='N')
                    xUNC['liUNC'] = Weight_RSR.processVIIRSBands(liUNCSlice, sensor='N')
                    xUNC['ltUNC'] = Weight_RSR.processVIIRSBands(ltUNCSlice, sensor='N')

                sensor = 'VIIRSN'
                ProcessL2.spectralReflectance(node, sensor, timeObj, xSlice, F0_VIIRS, F0_VIIRS_unc, rhoScalar, rhoVecVIIRS,  waveSubsetVIIRS,  xUNC)
                if ConfigFile.settings["bL2PerformNIRCorrection"]:
                    # Can't apply good NIR corrs at satellite bands, so use the correction factors from the hyperspectral instead.
                    ProcessL2.nirCorrectionSatellite(node, sensor, rrsNIRCorr, nLwNIRCorr)

            if ConfigFile.settings['bL2WeightVIIRSJ']:
                print('Processing VIIRSJ')

                if ZhangRho:
                    rhoVecVIIRS = Weight_RSR.processVIIRSBands(rhoVec,sensor='T')
                    rhoVecVIIRS = {key:value[0] for (key,value) in rhoVecVIIRS.items()}

                xSlice['es'] = esXSliceVIIRSJ
                xSlice['li'] = liXSliceVIIRSJ
                xSlice['lt'] = ltXSliceVIIRSJ

                xSlice['esRemaining'] = esXRemainingVIIRSJ
                xSlice['liRemaining'] = liXRemainingVIIRSJ
                xSlice['ltRemaining'] = ltXRemainingVIIRSJ

                xSlice['esMedian'] = esXmedianVIIRSJ
                xSlice['liMedian'] = liXmedianVIIRSJ
                xSlice['ltMedian'] = ltXmedianVIIRSJ

                xSlice['esSTD'] = esXstdVIIRSJ
                xSlice['liSTD'] = liXstdVIIRSJ
                xSlice['ltSTD'] = ltXstdVIIRSJ

                if xUNC is not None:
                    xUNC['esUNC'] = Weight_RSR.processVIIRSBands(esUNCSlice, sensor='N')
                    xUNC['liUNC'] = Weight_RSR.processVIIRSBands(liUNCSlice, sensor='N')
                    xUNC['ltUNC'] = Weight_RSR.processVIIRSBands(ltUNCSlice, sensor='N')

                sensor = 'VIIRSJ'
                ProcessL2.spectralReflectance(node, sensor, timeObj, xSlice, F0_VIIRS, F0_VIIRS_unc, rhoScalar, rhoVecVIIRS, waveSubsetVIIRS,  xUNC)
                if ConfigFile.settings["bL2PerformNIRCorrection"]:
                    # Can't apply good NIR corrs at satellite bands, so use the correction factors from the hyperspectral instead.
                    ProcessL2.nirCorrectionSatellite(node, sensor, rrsNIRCorr, nLwNIRCorr)

        if ConfigFile.settings['bL2WeightSentinel3A']:
            # F0 = F0_Sentinel3
            rhoVecSentinel3 = None

            if ConfigFile.settings['bL2WeightSentinel3A']:
                print('Processing Sentinel3A')

                if ZhangRho:
                    rhoVecSentinel3 = Weight_RSR.processSentinel3Bands(rhoVec,sensor='A')
                    rhoVecSentinel3 = {key:value[0] for (key,value) in rhoVecSentinel3.items()}

                xSlice['es'] = esXSliceSentinel3A
                xSlice['li'] = liXSliceSentinel3A
                xSlice['lt'] = ltXSliceSentinel3A

                xSlice['esRemaining'] = esXRemainingSentinel3A
                xSlice['liRemaining'] = liXRemainingSentinel3A
                xSlice['ltRemaining'] = ltXRemainingSentinel3A

                xSlice['esMedian'] = esXmedianSentinel3A
                xSlice['liMedian'] = liXmedianSentinel3A
                xSlice['ltMedian'] = ltXmedianSentinel3A

                xSlice['esSTD'] = esXstdSentinel3A
                xSlice['liSTD'] = liXstdSentinel3A
                xSlice['ltSTD'] = ltXstdSentinel3A

                # if xUNC is not None:
                #     xUNC['esUNC'] = Weight_RSR.processSentinel3Bands(esUNCSlice, sensor='A')
                #     xUNC['liUNC'] = Weight_RSR.processSentinel3Bands(liUNCSlice, sensor='A')
                #     xUNC['ltUNC'] = Weight_RSR.processSentinel3Bands(ltUNCSlice, sensor='A')

                sensor = 'Sentinel3A'
                ProcessL2.spectralReflectance(node, sensor, timeObj, xSlice, F0_Sentinel3, F0_Sentinel3_unc, rhoScalar, rhoVecSentinel3, waveSubsetSentinel3,  xUNC)
                if ConfigFile.settings["bL2PerformNIRCorrection"]:
                    # Can't apply good NIR corrs at satellite bands, so use the correction factors from the hyperspectral instead.
                    ProcessL2.nirCorrectionSatellite(node, sensor, rrsNIRCorr, nLwNIRCorr)

            if ConfigFile.settings['bL2WeightSentinel3B']:
                print('Processing Sentinel3B')

                if ZhangRho:
                    rhoVecSentinel3 = Weight_RSR.processSentinel3Bands(rhoVec,sensor='B')
                    rhoVecSentinel3 = {key:value[0] for (key,value) in rhoVecSentinel3.items()}

                xSlice['es'] = esXSliceSentinel3B
                xSlice['li'] = liXSliceSentinel3B
                xSlice['lt'] = ltXSliceSentinel3B

                xSlice['esRemaining'] = esXRemainingSentinel3B
                xSlice['liRemaining'] = liXRemainingSentinel3B
                xSlice['ltRemaining'] = ltXRemainingSentinel3B

                xSlice['esMedian'] = esXmedianSentinel3B
                xSlice['liMedian'] = liXmedianSentinel3B
                xSlice['ltMedian'] = ltXmedianSentinel3B

                xSlice['esSTD'] = esXstdSentinel3B
                xSlice['liSTD'] = liXstdSentinel3B
                xSlice['ltSTD'] = ltXstdSentinel3B

                # if xUNC is not None:
                #     xUNC['esUNC'] = Weight_RSR.processSentinel3Bands(esUNCSlice, sensor='B')
                #     xUNC['liUNC'] = Weight_RSR.processSentinel3Bands(liUNCSlice, sensor='B')
                #     xUNC['ltUNC'] = Weight_RSR.processSentinel3Bands(ltUNCSlice, sensor='B')

                sensor = 'Sentinel3B'
                ProcessL2.spectralReflectance(node, sensor, timeObj, xSlice, F0_Sentinel3, F0_Sentinel3_unc, rhoScalar, rhoVecSentinel3, waveSubsetSentinel3,  xUNC)
                if ConfigFile.settings["bL2PerformNIRCorrection"]:
                    # Can't apply good NIR corrs at satellite bands, so use the correction factors from the hyperspectral instead.
                    ProcessL2.nirCorrectionSatellite(node, sensor, rrsNIRCorr, nLwNIRCorr)

        # Transfer attributes from L1BQC to L2 for sensors (place in [sensor]_HYPER datasets)
        for gp in node.groups:
            if gp.id == 'IRRADIANCE':
                gp.datasets['ES_HYPER'].attributes = refGroup.datasets['ES'].attributes.copy()
            if gp.id == 'RADIANCE':
                gp.datasets['LT_HYPER'].attributes = sasGroup.datasets['LT'].attributes.copy()
                gp.datasets['LI_HYPER'].attributes = sasGroup.datasets['LI'].attributes.copy()

        return True


    @staticmethod
    def stationsEnsemblesReflectance(node, root, station=None):
        ''' Extract stations if requested, then pass to ensemblesReflectance for ensemble
            averages, rho calcs, Rrs, Lwn, NIR correction, satellite convolution, OC Products.'''

        print("stationsEnsemblesReflectance")

        # Create a third HDF for copying root without altering it
        rootCopy = HDFRoot()
        rootCopy.addGroup("ANCILLARY")
        rootCopy.addGroup("IRRADIANCE")
        rootCopy.addGroup("RADIANCE")
        rootCopy.addGroup('SIXS_MODEL')

        rootCopy.getGroup('ANCILLARY').copy(root.getGroup('ANCILLARY'))
        rootCopy.getGroup('IRRADIANCE').copy(root.getGroup('IRRADIANCE'))
        rootCopy.getGroup('RADIANCE').copy(root.getGroup('RADIANCE'))

        sixS_available = False
        for gp in root.groups:
            if gp.id == 'SIXS_MODEL':
                sixS_available = True
                rootCopy.getGroup('SIXS_MODEL').copy(root.getGroup('SIXS_MODEL'))
                break

        if ConfigFile.settings['SensorType'].lower() == 'seabird':
            rootCopy.addGroup("ES_DARK_L1AQC")
            rootCopy.addGroup("ES_LIGHT_L1AQC")
            rootCopy.addGroup("LI_DARK_L1AQC")
            rootCopy.addGroup("LI_LIGHT_L1AQC")
            rootCopy.addGroup("LT_DARK_L1AQC")
            rootCopy.addGroup("LT_LIGHT_L1AQC")
            rootCopy.getGroup('ES_LIGHT_L1AQC').copy(root.getGroup('ES_LIGHT_L1AQC'))
            rootCopy.getGroup('ES_DARK_L1AQC').copy(root.getGroup('ES_DARK_L1AQC'))
            rootCopy.getGroup('LI_LIGHT_L1AQC').copy(root.getGroup('LI_LIGHT_L1AQC'))
            rootCopy.getGroup('LI_DARK_L1AQC').copy(root.getGroup('LI_DARK_L1AQC'))
            rootCopy.getGroup('LT_LIGHT_L1AQC').copy(root.getGroup('LT_LIGHT_L1AQC'))
            rootCopy.getGroup('LT_DARK_L1AQC').copy(root.getGroup('LT_DARK_L1AQC'))

            esRawGroup = {"LIGHT": rootCopy.getGroup('ES_LIGHT_L1AQC'), "DARK": rootCopy.getGroup('ES_DARK_L1AQC')}
            liRawGroup = {"LIGHT": rootCopy.getGroup('LI_LIGHT_L1AQC'), "DARK": rootCopy.getGroup('LI_DARK_L1AQC')}
            ltRawGroup = {"LIGHT": rootCopy.getGroup('LT_LIGHT_L1AQC'), "DARK": rootCopy.getGroup('LT_DARK_L1AQC')}

        elif ConfigFile.settings['SensorType'].lower() == 'trios' or \
            ConfigFile.settings['SensorType'].lower() == 'dalec' or  \
                ConfigFile.settings['SensorType'].lower() == 'sorad':
            rootCopy.addGroup("ES_L1AQC")
            rootCopy.addGroup("LI_L1AQC")
            rootCopy.addGroup("LT_L1AQC")
            rootCopy.getGroup('ES_L1AQC').copy(root.getGroup('ES_L1AQC'))
            rootCopy.getGroup('LI_L1AQC').copy(root.getGroup('LI_L1AQC'))
            rootCopy.getGroup('LT_L1AQC').copy(root.getGroup('LT_L1AQC'))

            esRawGroup = rootCopy.getGroup('ES_L1AQC')
            liRawGroup = rootCopy.getGroup('LI_L1AQC')
            ltRawGroup = rootCopy.getGroup('LT_L1AQC')

        # rootCopy will be manipulated in the making of node, but root will not
        referenceGroup = rootCopy.getGroup("IRRADIANCE")
        sasGroup = rootCopy.getGroup("RADIANCE")
        ancGroup = rootCopy.getGroup("ANCILLARY")
        if sixS_available:
            sixSGroup = rootCopy.getGroup("SIXS_MODEL")
        else:
            sixSGroup = None

        if ConfigFile.settings["fL1bCal"] >= 2 or ConfigFile.settings['SensorType'].lower() == 'seabird':
            #or ConfigFile.settings['SensorType'].lower() == 'dalec':
            rootCopy.addGroup("RAW_UNCERTAINTIES")
            rootCopy.getGroup('RAW_UNCERTAINTIES').copy(root.getGroup('RAW_UNCERTAINTIES'))
            uncGroup = rootCopy.getGroup("RAW_UNCERTAINTIES")
        # Only Factory-Trios has no unc
        else:
            uncGroup = None

        Utilities.rawDataAddDateTime(rootCopy) # For L1AQC data carried forward
        Utilities.rootAddDateTimeCol(rootCopy)

        ###############################################################################
        #
        # Stations
        #   Simplest approach is to run station extraction seperately from (i.e. in addition to)
        #   underway data. This means if station extraction is selected in the GUI, all non-station
        #   data will be discarded here prior to any further filtering or processing.

        if ConfigFile.settings["bL2Stations"]:
            Utilities.writeLogFileAndPrint("Extracting station data only. All other records will be discarded.")

            # If we are here, the station was already chosen in Controller
            try:
                stations = ancGroup.getDataset("STATION").columns["STATION"]
                dateTime = ancGroup.getDataset("STATION").columns["Datetime"]
            except Exception:
                Utilities.writeLogFileAndPrint("No station data found in ancGroup. Aborting.")
                return False

            badTimes = []
            start = False
            stop = False
            for index, stn in enumerate(stations):
                # print(f'index: {index}, station: {station}, datetime: {dateTime[index]}')
                # if np.isnan(station) and start == False:
                if (stn != station) and (start is False):
                    start = dateTime[index]
                # if not np.isnan(station) and not (start == False) and (stop == False):
                if not (stn!=station) and (start is not False) and (stop is False):
                    stop = dateTime[index-1]
                    badTimes.append([start, stop])
                    start = False
                    stop = False
                # End of file, no active station
                # if np.isnan(station) and not (start == False) and (index == len(stations)-1):
                if (stn != station) and not (start is False) and (index == len(stations)-1):
                    stop = dateTime[index]
                    badTimes.append([start, stop])

            if badTimes is not None and len(badTimes) != 0:
                print('Removing records...')
                check = ProcessL2.filterData(referenceGroup, badTimes)
                if check == 1.0:
                    Utilities.writeLogFileAndPrint("100% of irradiance data removed. Abort.")
                    return False
                ProcessL2.filterData(sasGroup, badTimes)
                ProcessL2.filterData(ancGroup, badTimes)
                if sixS_available:
                    ProcessL2.filterData(sixSGroup, badTimes)

        #####################################################################
        #
        # Ensembles. Break up data into time intervals, and calculate averages and reflectances
        #
        esData = referenceGroup.getDataset("ES")
        esColumns = esData.columns
        timeStamp = esColumns["Datetime"]
        esLength = len(list(esColumns.values())[0])
        interval = float(ConfigFile.settings["fL2TimeInterval"])

        # interpolate Light/Dark data for Raw groups if HyperOCR data is being processed
        if ConfigFile.settings['SensorType'].lower() == "seabird":
            # in seabird case interpolate dark data to light timer before breaking into stations
            instrument = HyperOCR()
            # if not any([instrument.darkToLightTimer(esRawGroup, 'ES'),
            if not all([instrument.darkToLightTimer(esRawGroup, 'ES'),
                        instrument.darkToLightTimer(liRawGroup, 'LI'),
                        instrument.darkToLightTimer(ltRawGroup, 'LT')]):
                Utilities.writeLogFileAndPrint("failed to interpolate dark data to light data timer")
        if interval == 0:
            # Here, take the complete time series
            print("No time binning. This can take a moment.")
            progressBar = tqdm(total=esLength, unit_scale=True, unit_divisor=1)
            for i in range(0, esLength-1):
                progressBar.update(1)
                start = i
                end = i+1

                if not ProcessL2.ensemblesReflectance(node, sasGroup, referenceGroup, ancGroup,
                                                    uncGroup, esRawGroup,liRawGroup, ltRawGroup,
                                                    sixSGroup, start, end):
                    Utilities.writeLogFileAndPrint('ProcessL2.ensemblesReflectance unsliced failed. Abort.')
                    continue
        else:
            Utilities.writeLogFileAndPrint('Binning datasets to ensemble time interval.')

            # Iterate over the time ensembles
            start = 0
            endTime = timeStamp[0] + datetime.timedelta(0,interval)
            endFileTime = timeStamp[-1]
            EndOfFileFlag = False
            # endTime is theoretical based on interval
            if endTime > endFileTime:
                endTime = endFileTime
                EndOfFileFlag = True # In case the whole file is shorter than the selected interval

            for i in range(0, esLength):
                timei = timeStamp[i]
                if (timei > endTime) or EndOfFileFlag: # end of increment reached
                    if EndOfFileFlag:
                        end = len(timeStamp)-1 # File shorter than interval; include all spectra
                        if not ProcessL2.ensemblesReflectance(node, sasGroup, referenceGroup, ancGroup, 
                                                            uncGroup, esRawGroup,liRawGroup, ltRawGroup,
                                                            sixSGroup, start, end):
                            Utilities.writeLogFileAndPrint('ProcessL2.ensemblesReflectance with slices failed. Continue.')
                            break # End of file reached. Safe to break

                        break # End of file reached. Safe to break
                    else:
                        endTime = timei + datetime.timedelta(0,interval) # increment for the next bin loop
                        end = i # end of the slice is up to and not including...so -1 is not needed

                    if endTime > endFileTime:
                        endTime = endFileTime
                        EndOfFileFlag = True
             
                    if not ProcessL2.ensemblesReflectance(node, sasGroup, referenceGroup, ancGroup, 
                                                            uncGroup, esRawGroup,liRawGroup, ltRawGroup,
                                                            sixSGroup, start, end):
                        Utilities.writeLogFileAndPrint('ProcessL2.ensemblesReflectance with slices failed. Continue.')

                        start = i
                        continue # End of ensemble reached. Continue.
                    start = i

                    if EndOfFileFlag:
                        # No need to continue incrementing; all records captured in one ensemble
                        break

            # For the rare case where end of record is reached at, but not exceeding, endTime...
            if not EndOfFileFlag:
                end = i+1 # i is the index of end of record; plus one to include i due to -1 list slicing
                if not ProcessL2.ensemblesReflectance(node, sasGroup, referenceGroup, ancGroup, 
                                                            uncGroup, esRawGroup,liRawGroup, ltRawGroup,
                                                            sixSGroup, start, end):
                    Utilities.writeLogFileAndPrint('ProcessL2.ensemblesReflectance ender clause failed.')

        #####################################
        #
        # Reflectance calculations complete
        #

        # Filter reflectances for negative ensemble spectra
        # NOTE: Any spectrum that has any negative values between
        #  400 - 700ish (hard-coded below), remove the entire spectrum. Otherwise,
        # set negative bands to 0.

        if ConfigFile.settings["bL2NegativeSpec"]:
            fRange = [400, 680]
            Utilities.writeLogFileAndPrint("Filtering reflectance spectra for negative values.")
            # newReflectanceGroup = node.groups[0]
            newReflectanceGroup = node.getGroup("REFLECTANCE")
            if not newReflectanceGroup.datasets:
                Utilities.writeLogFileAndPrint("Ensemble is empty. Aborting.")
                return False

            badTimes1 = ProcessL2.negReflectance(newReflectanceGroup, 'Rrs_HYPER', VIS = fRange)
            badTimes2 = ProcessL2.negReflectance(newReflectanceGroup, 'nLw_HYPER', VIS = fRange)

            badTimes = None
            if badTimes1 is not None and badTimes2 is not None:
                badTimes = np.append(badTimes1,badTimes2, axis=0)
            elif badTimes1 is not None:
                badTimes = badTimes1
            elif badTimes2 is not None:
                badTimes = badTimes2

            if badTimes is not None:
                print('Removing records...')

                # Even though HYPER is specified here, ALL data at badTimes in the group,
                # including satellite data, will be removed.
                check = ProcessL2.filterData(newReflectanceGroup, badTimes, sensor = "HYPER")
                if check > 0.99:
                    Utilities.writeLogFileAndPrint("Too few spectra remaining. Abort.")
                    return False
                ProcessL2.filterData(node.getGroup("IRRADIANCE"), badTimes, sensor = "HYPER")
                ProcessL2.filterData(node.getGroup("RADIANCE"), badTimes, sensor = "HYPER")
                ProcessL2.filterData(node.getGroup("ANCILLARY"), badTimes)
                if sixS_available:
                    ProcessL2.filterData(node.getGroup("SIXS_MODEL"), badTimes)

        return True

    @staticmethod
    def processL2(root,station=None):
        '''Calculates Rrs and nLw after quality checks and filtering, glint removal, residual
            subtraction. Weights for satellite bands, and outputs plots and SeaBASS datasets'''

        # Root is the input from L1BQC, node is the output
        # Root should not be impacted by data reduction in node...
        node = HDFRoot()
        node.addGroup("ANCILLARY")
        node.addGroup("REFLECTANCE")
        node.addGroup("IRRADIANCE")
        node.addGroup("RADIANCE")
        node.addGroup("SIXS_MODEL")
        node.copyAttributes(root)
        node.attributes["PROCESSING_LEVEL"] = "2"
        # Remaining attributes managed below...

        # Copy attributes from root and for completeness, flip datasets into columns in all groups
        for grp in root.groups:
            for gp in node.groups:
                if gp.id == grp.id:
                    gp.copyAttributes(grp)
            for ds in grp.datasets:
                grp.datasets[ds].datasetToColumns()

            # Carry over L1AQC data for use in uncertainty budgets
            if grp.id.endswith('_L1AQC'): #or grp.id.startswith('SIXS_MODEL'):
                newGrp = node.addGroup(grp.id)
                newGrp.copy(grp)
                for ds in newGrp.datasets:
                    newGrp.datasets[ds].datasetToColumns()

        # Process stations, ensembles to reflectances, OC prods, etc.
        if not ProcessL2.stationsEnsemblesReflectance(node, root,station):
            return None

        # Reflectance
        gp = node.getGroup("REFLECTANCE")
        gp.attributes["Rrs_UNITS"] = "1/sr"
        gp.attributes["nLw_UNITS"] = "uW/cm^2/nm/sr"
        if ConfigFile.settings['bL23CRho']:
            gp.attributes['GLINT_CORR'] = 'Groetsch et al. 2017'
        if ConfigFile.settings['bL2ZhangRho']:
            gp.attributes['GLINT_CORR'] = 'Zhang et al. 2017'
        if ConfigFile.settings['bL2DefaultRho']:
            gp.attributes['GLINT_CORR'] = 'Mobley 1999'
        if ConfigFile.settings['bL2PerformNIRCorrection']:
            if ConfigFile.settings['bL2SimpleNIRCorrection']:
                
                gp.attributes['NIR_RESID_CORR'] = 'Mueller and Austin 1995'
            if ConfigFile.settings['bL2SimSpecNIRCorrection']:
                gp.attributes['NIR_RESID_CORR'] = 'Ruddick et al. 2005/2006'
        if ConfigFile.settings['bL2NegativeSpec']:
            gp.attributes['NEGATIVE_VALUE_FILTER'] = 'ON'

        # Stations and Ensembles
        if ConfigFile.settings['bL2Stations']:
            node.attributes['STATION_EXTRACTION'] = 'ON'
        node.attributes['ENSEMBLE_DURATION'] = str(ConfigFile.settings['fL2TimeInterval']) + ' sec'

        # Check to insure at least some data survived quality checks
        if node.getGroup("REFLECTANCE").getDataset("Rrs_HYPER").data is None:
            msg = "All data appear to have been eliminated from the file. Aborting."
            print(msg)
            Utilities.writeLogFile(msg)
            return None

        # If requested, proceed to calculation of derived geophysical and
        # inherent optical properties
        totalProds = sum(list(ConfigFile.products.values()))
        if totalProds > 0:
            ProcessL2OCproducts.procProds(node)

        # If requested, process BRDF corrections to Rrs and nLw
        if ConfigFile.settings["bL2BRDF"]:

            if ConfigFile.settings['bL2BRDF_fQ']:
                Utilities.writeLogFileAndPrint("Applying iterative Morel et al. 2002 BRDF correction to Rrs and nLw")
                ProcessL2BRDF.procBRDF(node, BRDF_option='M02')

            if ConfigFile.settings['bL2BRDF_IOP']:
                Utilities.writeLogFileAndPrint("Applying Lee et al. 2011 BRDF correction to Rrs and nLw")
                ProcessL2BRDF.procBRDF(node, BRDF_option='L11')

            if ConfigFile.settings['bL2BRDF_O23']:
                Utilities.writeLogFileAndPrint("Applying Pitarch et al. 2025 BRDF correction to Rrs and nLw")
                ProcessL2BRDF.procBRDF(node, BRDF_option='O23')


        # Strip out L1AQC data
        for gp in reversed(node.groups):
            if gp.id.endswith('_L1AQC'):
                node.removeGroup(gp)

        # In the case of TriOS Factory, strip out uncertainty datasets
        if  ((ConfigFile.settings['SensorType'].lower() == 'trios' or \
             ConfigFile.settings['SensorType'].lower() == 'dalec') or  \
                ConfigFile.settings['SensorType'].lower() == 'sorad') and ConfigFile.settings['fL1bCal'] == 1:
            for gp in node.groups:
                if gp.id in ('IRRADIANCE', 'RADIANCE', 'REFLECTANCE'):
                    removeList = []
                    for dsName in reversed(gp.datasets):
                        if dsName.endswith('_unc'):
                            removeList.append(dsName)
                    for dsName in removeList:
                        gp.removeDataset(dsName)

        # Change _median nomiclature to _uncorr
        for gp in node.groups:
            if gp.id in ('IRRADIANCE', 'RADIANCE', 'REFLECTANCE'):
                changeList = []
                for dsName in gp.datasets:
                    if dsName.endswith('_median'):
                        changeList.append(dsName)
                for dsName in changeList:
                    gp.datasets[dsName].changeDatasetName(gp,dsName,dsName.replace('_median','_uncorr'))


        # Now strip datetimes from all datasets
        for gp in node.groups:
            for dsName in gp.datasets:
                ds = gp.datasets[dsName]
                if "Datetime" in ds.columns:
                    ds.columns.pop("Datetime")
                ds.columnsToDataset()

        now = datetime.datetime.now()
        timestr = now.strftime("%d-%b-%Y %H:%M:%S")
        node.attributes["FILE_CREATION_TIME"] = timestr

        return node
