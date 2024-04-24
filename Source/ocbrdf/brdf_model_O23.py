import netCDF4
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
import os
from matplotlib import pyplot as plt
import sys
import xarray as xr

from .brdf_utils import ADF_OCP, solve_2nd_order_poly
from .Raman import Raman


''' O23 BRDF correction from EUMETSAT BRDF4OLCI study
    Ref. ATBD_EUM-CO-21-4600002626-JIG, 29/03/2024
'''

# Init Raman class
Raman = Raman()

""" Class for O23 coefficients """
class Coeffs():
    def __init__(self,Gw0,Gw1,Gp0,Gp1):
        self.Gw0 = Gw0
        self.Gw1 = Gw1
        self.Gp0 = Gp0
        self.Gp1 = Gp1

""" Class for O23 BRDF model """
class O23:

    """ Initialise O23 model: BRDF LUT, coeffs, QAA parameters, water IOPs LUT
        Note: bands are fixed and defined at class initilization, but could be initialized in init_pixels if needed
    """
    def __init__(self, bands, adf=None):
        if adf is None:
            adf = ADF_OCP

        # Check required bands are existing, within a 10 nm threshold
        self.bands = bands
        threshold = 10.
        bands_required = [442, 490, 560, 665]
        bands_ref = bands.sel(bands=bands_required, method='nearest')
        for band_ref, band_required in zip(bands_ref, bands_required):
            assert abs(band_ref - band_required) < threshold, 'Band %d nm missing or too far'%br
        self.b442, self.b490, self.b560, self.b665 = bands_ref

        # Read BRDF LUT and compute default coeffs
        LUT_OCP = xr.open_dataset(adf,group='BRDF/O23')
        self.LUT = xr.Dataset()
        self.LUT['Gw0'] = LUT_OCP.Gw0
        self.LUT['Gw1'] = LUT_OCP.Gw1 
        self.LUT['Gp0'] = LUT_OCP.Gp0 
        self.LUT['Gp1'] = LUT_OCP.Gp1 

        self.coeffs0 = self.interp(0.,0.,0.)
        self.coeffs = Coeffs(np.nan,np.nan,np.nan,np.nan)

        # Read IOPs of pure water (store in LUT for further spectral interpolation)
        self.awLUT = LUT_OCP.aw.rename({'IOP_wl':'bands'})
        self.bbwLUT = LUT_OCP.bbw.rename({'IOP_wl':'bands'})

        # Read QAA parameters
        self.a0 = LUT_OCP.a0.values
        self.gamma = LUT_OCP.gamma.values
        self.niter = LUT_OCP.niter.values
              
    """ Initialize pixel: coefficient at current geometry and water IOP at current bands """
    def init_pixels(self, theta_s, theta_v, delta_phi):
        self.coeffs = self.interp(theta_s, theta_v, delta_phi)

        # Compute IOPs at current bands
        self.aw = self.awLUT.interp(bands = self.bands, kwargs={'fill_value':'extrapolate'})
        self.bbw = self.bbwLUT.interp(bands = self.bands, kwargs={'fill_value':'extrapolate'})

    """ Interpolate coefficients """
    def interp(self, theta_s, theta_v, delta_phi):
        Gw0 = self.LUT.Gw0.interp(theta_s=theta_s,theta_v=theta_v,delta_phi=delta_phi)
        Gw1 = self.LUT.Gw1.interp(theta_s=theta_s,theta_v=theta_v,delta_phi=delta_phi)
        Gp0 = self.LUT.Gp0.interp(theta_s=theta_s,theta_v=theta_v,delta_phi=delta_phi)
        Gp1 = self.LUT.Gp1.interp(theta_s=theta_s,theta_v=theta_v,delta_phi=delta_phi)
        return Coeffs(Gw0,Gw1,Gp0,Gp1)

    """ Compute remote-sensing reflectance, without Raman effect (vanish in the normalization factor) """
    def forward(self, omegab, etab, normalized=False):
        if normalized:
            coeffs = self.coeffs0
        else:
            coeffs = self.coeffs
        Rrs = (coeffs.Gw0+coeffs.Gw1*omegab*etab)*omegab*etab + (coeffs.Gp0+coeffs.Gp1*omegab*(1-etab))*omegab*(1-etab)
        return Rrs

    """ Apply QAA to retrieve IOP (omega_b, eta_b) from rrs """
    def backward(self, Rrs, iter_brdf):

        # Select G coeff according to iteration
        if iter_brdf == 0:
            coeffs = self.coeffs
        else:
            coeffs = self.coeffs0

        # Apply Raman correction
        Rrs = Raman.correct(Rrs)
       
        # Local renaming of bands 
        b442, b490, b560, b665 = self.b442, self.b490, self.b560, self.b665

        # Apply upper and lower limits to Rrs(665) #TODO currently not applied
        #"""
        Rrs442 = Rrs.sel(bands=b442)
        Rrs490 = Rrs.sel(bands=b490)
        Rrs560 = Rrs.sel(bands=b560)
        Rrs665 = Rrs.sel(bands=b665)
        mask= ((Rrs665 > 20*np.power(Rrs560,1.5)) | (Rrs665 < 0.9*np.power(Rrs560, 1.7)))
        if np.any(mask):
            Rrs665_ = 1.27*np.power(Rrs560, 1.47) + 0.00018*np.power(Rrs490/Rrs560,-3.19)
            # Redefine Rrs665 and Rrs[bands=b665] (both important for computations below)
            Rrs665 = xr.where(mask, Rrs665_, Rrs665)
            Rrs.loc[dict(bands=665)] = Rrs665
        #"""

        # Calculate rrs below water for absorption computation
        rrs = Rrs / (0.52 + 1.7*Rrs)

        # Define reference band band0 at 560 nm
        # and compute total absorption
        Rrs0 = Rrs.sel(bands=b560)
        band0 = xr.zeros_like(Rrs0) + b560
        aw0 = xr.zeros_like(Rrs0) + self.aw.sel(bands=b560)
        bbw0 = xr.zeros_like(Rrs0) + self.bbw.sel(bands=b560)
        # Compute a0 when band0 = b560
        rrs442 = rrs.sel(bands=b442)
        rrs490 = rrs.sel(bands=b490)
        rrs560 = rrs.sel(bands=b560)
        rrs665 = rrs.sel(bands=b665)
        chi = np.log10((rrs442 + rrs490) / (rrs560 + 5.0 * rrs665*rrs665 / rrs490))
        poly = np.polynomial.polynomial.polyval(chi, self.a0)
        a0 = aw0 + np.power(10., poly)

        # Compute bbp at band0 by 2nd order polynomial inversion
        k0 = a0 + bbw0
        cA = coeffs.Gp0 + coeffs.Gp1 - Rrs0
        cB = coeffs.Gw0 * bbw0 + (coeffs.Gp0 -2*Rrs0) *k0
        cC = (coeffs.Gw0 * bbw0 - Rrs0 * k0) * k0 + coeffs.Gw1 * bbw0 * bbw0
        bbp0 = solve_2nd_order_poly(cA, cB, cC)

        # Compute bbp slope and extrapolate at all bands
        gamma = self.gamma[0] * (1.0 - self.gamma[1] * np.power(rrs442 / rrs560, -self.gamma[2]))
        bbp = bbp0 * np.power(band0 / self.bands, gamma)

        # Compute total bb
        bb = self.bbw + bbp

        # Compute quasi-diffuse attenuation coefficient k at each band
        # by 2nd order polynomial inversion
        cA = Rrs
        cB = - (coeffs.Gw0 * self.bbw + coeffs.Gp0 * bbp)
        cC = - (coeffs.Gw1 * self.bbw *self.bbw + coeffs.Gp1 * bbp * bbp)
        k = solve_2nd_order_poly(cA, cB, cC)
        # Set 0 to nan to avoid division by zero
        k = xr.where(k > 0, k, np.nan)

        # Compute final IOPs
        omega_b = bb / k
        eta_b = self.bbw / bb

        return omega_b, eta_b

