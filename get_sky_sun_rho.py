# import argparse
from functools import partial
import pathlib
import sys
import time

import numpy as np
from scipy.interpolate import interpn
import xarray as xr

π = np.pi

db_path = './Data/db.mat'
#Groups in db.mat
db, quads, sdb, vdb = None, None, None, None
#Vars. in db.mat
skyrad0, sunrad0, rad_boa_sca, rad_boa_vec = None, None, None, None

def load_db(db_path=db_path):
    global db, quads, sdb, vdb
    global skyrad0, sunrad0, rad_boa_sca, rad_boa_vec
    
    db = xr.open_dataset(db_path, group='db')
    quads = xr.open_dataset(db_path, group='quads')
    sdb = xr.open_dataset(db_path, group='sdb')
    vdb = xr.open_dataset(db_path, group='vdb')
    skyrad0 = xr.open_dataset(db_path)['skyrad0']
    sunrad0 = xr.open_dataset(db_path)['sunrad0']
    rad_boa_sca = xr.open_dataset(db_path)['Radiance_BOA_sca']
    rad_boa_vec = xr.open_dataset(db_path)['Radiance_BOA_vec']

def my_sph2cart(azm, zen, r=1):
    """
    Converts spherical coordinates to cartesian

    Inputs
    -------
    azm [Numpy array] : -azimuth angle 
    zen [Numpy array] : zenith angle
    r  [float] : radius = 1

    Outputs
    -------
    x :
    y :
    z :
    """
    def sph2cart(azm, elev, r):
        cos_elev = np.cos(elev)
        x = r * cos_elev * np.cos(azm)
        y = r * cos_elev * np.sin(azm)
        z = r * np.sin(elev)
        return x, y, z

    x, y, z = sph2cart(azm, π/2 - zen, r)
    return np.c_[x.ravel(), y.ravel(), z.ravel()].squeeze()

def find_quads(zen, azm):
    """
    Finds location in quads (why is it called find_quads?)

    Inputs
    ------
    zen :
    azm :

    Outputs
    -------
    locs :
    """
    loc = None
    try:
        # with xr.open_dataset(db_path, group='quads') as quads:
        tmp = np.sqrt((quads.zen[:] - zen)**2 + (quads.azm[:] - azm)**2)
        loc = np.argmin(tmp.data)
    except:
        print('Unable to read quads data')
    finally:
        return loc


def get_prob(wind, vec):
    """
    Computes probability of sky light being reflected into the sensor

    Inputs
    ------
    wind :
    vec : sensor vector

    Outputs
    -------
    prob [np.array] :  probability of sky light reflected into the sensor
    angr_sky[np.array] : reflection angle
    """
    
    # with xr.open_dataset(db_path, group='quads') as quads:
    zen = quads.zen.data[0]
    if len(vec.shape) == 1:
        vec = vec.reshape(1,vec.size)
    prob = np.nan * np.ones((len(zen), len(wind)*vec.shape[0]))
    angr_sky = prob.copy()
    
    k = -1
    for w in wind:
        for v in vec:
            k = k + 1
            prob[:,k], angr_sky[:,k] = skylight_reflection2(w, v)    

    return prob, angr_sky
    
def skylight_reflection2(wind, sensor):
    """
    Computes probability of light reflection at angle.

    Inputs
    ------
    wind : Wind speed (m/s)
    sensor : Numpy array
             The vector of reflected light measured by the sensor
    quads :
            Sky light quads
    """
    def gen_vec(zens,azms):
        # generate vectors from permutation of zenith and azimuth angles
        zens,azms = np.meshgrid(zens,azms)
        zens = zens[:]
        azms = azms[:]
        # vector expression
        vec = my_sph2cart(azms,zens,1)
        return vec

    def gen_vec_quad(zen,du,azm,dphi,num):
        half_azm = np.linspace(-dphi/2,dphi/2,num)
        half_zen = np.linspace(-du/2/np.sin(zen), du/2/np.sin(zen),num)
        vec = gen_vec(zen+half_zen, azm+half_azm)
        return vec

    # initialize
    prob = quads.zen.data[0].copy()
    ang = prob.copy()

    # polar quad, 1st in quads
    zen0 = quads.zen.data[0][0]
    # generate sky vector
    num = 100
    p_vec = gen_vec_polar(zen0, quads.sun05.data, num)
    # -p_vec represent vectors coming from the sky
    prob[0], ang[0] = prob_reflection(-p_vec, sensor, wind)
    
    # non-polar quads
    num = 10 # the number of individual vectors

    du = quads.du.data
    dphi = quads.dphi.data
    t0 = time.time()
    for i in np.arange(1, prob.size):        
        # t01 = time.time()
        sky = gen_vec_quad(quads.zen.data[0][i],du,quads.azm.data[0][i],dphi,num)
        # t02 = time.time()
        # print(f'Inside time elapsed: {t02-t01}')
        prob[i],ang[i] = prob_reflection(-sky,sensor,wind)

    # # This should be the equivalent of the loop above but as a comprehension
    ## Blows up, doesn't work
    # x, y = [(prob_reflection(
    #                 -gen_vec_quad(quads.zen.data[0][i],
    #                 du,quads.azm.data[0][i],
    #                 dphi,num),sensor,wind)) for i in np.arange(1, prob.size)]
    t1 = time.time()
    print(f'Time elapsed: {t1-t0}')
                    
    # prob = x
    # ang = y
    return prob, ang
    
def my_cart2sph(n):
    def cart2sph(x,y,z):
        azimuth = np.arctan2(y,x)
        elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
        r = np.sqrt(x**2 + y**2 + z**2)
        return azimuth, elevation, r

    azm,zen,r = cart2sph(n[:,0],n[:,1],n[:,2])
    zen = π/2 - zen

    return azm, zen, r

def gen_vec_polar(zen, sun05, num):
    """
    Generates vectros for the polar cap, quad, and sun disk.
    By convention, the sun disk is at XZ plane, i.e., azimuth = 0.

    Inputs
    ------
    zen :
         Sun zenith angle

    sun05 :

    num :
          Number of angles to consider
    Outputs
    -------
    vec :
          Polar cap vector
    """
    
    ϕ = np.linspace(0, 2*π, 100)
    sin_sun05 = np.sin(sun05)
    x = (sin_sun05*np.cos(ϕ)).tolist()
    x1 = np.insert(x,0,0)    
    y = (sin_sun05*np.sin(ϕ)).tolist()
    y1 = np.insert(y,0,0)
    z = (np.cos(sun05)*np.ones_like(ϕ)).tolist()
    z1 = np.insert(z,0,1)

    tmp = np.array([x1,y1,z1])
    vec = np.fliplr(np.rot90(tmp,k=-1))
    return vec    

def prob_reflection(inc, refl, wind):
    """
    Estimates probability of facets reflecting incident ligth into given direction and wind.

    Inputs
    ------
    inc : incident light vector
    refl : reflected light vector
    wind : wind speeds

    Outputs
    -------
    prob : Probability
    ang : Reflection angle
    """

    def vec_length(a):
        # the length of vector a
        al = np.sum(abs(a)**2, 1)**0.5
        return al

    def cox_munk(wind):
        # Cox and Munk slope distribution of capillary wave facets
        sigma = np.sqrt(0.003+0.00512*wind)
        return sigma

    def rayleighcdf(x,s):
        # Cumulative distribution function for Rayleigh distribution
        t = (x/s)**2
        y = 1-np.exp(-t/2)
        return y


    # Elementwise broadcasting 1x3(refl) onto 101x3(inc)
    n = refl - inc
    vLen = vec_length(n).reshape(vec_length(n)[:].shape[0],1)
    n = n/vLen

    # the zenith and azimuth angles of the facets
    azm_n,zen_n,_ = my_cart2sph(n)

    # convert facet zenith angle to slopes
    slope = np.tan(zen_n)

    # estimate wind-roughned probability of facets
    # sigma2 = 0.003 + 0.00512*wind;
    # sigma = sigma2^0.5;
    sigma = cox_munk(wind)
    # p1 = normcdf(max(slope),0,sigma) - normcdf(min(slope),0,sigma);
    # !!! see document On the Cox and Munk
    sigma = sigma/np.sqrt(2)
    p1 = rayleighcdf(max(slope),sigma)-rayleighcdf(min(slope),sigma)
    #} !!! 
    # azimuth angle ranges from -180 to 180. Need to treat the cases when the
    # azimuth angles cover both positive ang negative ranges.
    # case 1: -2 -1 1 2
    # case 2: -179, -178, 178, 179
    # case 3: -179 -120 2 5 130 178
    # cases 1 and 2: the range should be 4
    # case 3: the range should be 357
    azm_nx = max(azm_n)
    azm_nn = min(azm_n)

    if azm_nx*azm_nn >0: # not an issue
        p2 = (azm_nx-azm_nn)/2/π
    elif any(abs(azm_n)<π/2): # cases 1 and 3 
        p2 = (azm_nx-azm_nn)/2/π
    else: # case 2
        ind = azm_n<0
        azm_n[ind] = azm_n[ind]+2*π
        azm_nx = max(azm_n)
        azm_nn = min(azm_n)
        p2 = (azm_nx-azm_nn)/2/π

    prob = 2*p1*p2 # factor 2 accounts for 180 degree ambiguity
    
    # incident angle
    # cosw = sum(bsxfun(@times,n,refl),2)
    cosw = np.sum(n*refl,1)
    ang = np.arccos(cosw)
    ind = ang>π/2
    ang[ind] = π - ang[ind]
    ang = np.mean(ang)
    return prob, ang



def sw_fresnel(wv,ang,T,S):
    """
    Calcualtes Fresnel reflectance for seawater.

    Inputs
    ------
    wv : Wavelength (nm)
    ang : Reflectance angle
    T : Temperature (̊ C)
    S : Salinity (PSU)

    Outputs
    -------
    m : Refractive index
    ref : Fresnel reflectance of seawater
    """
    m = index_w(wv,T,S)
    ref = fresnel(m,ang)
    return ref

def index_w(wv, T, S):
    """
    Calculates water refractive index
     mw(wv,T,S)=n0+(n1+n2T+n3T^2)S+n4T2+(n5+n6S+n7T)/wv+n8/wv^2+n9/wv^3;

    Inputs
    -------
    wv : Wavelength (nm)
    T : Temperature (̊ C)
    S : Salinity (PPT)
    """
    n0=1.31405
    n1=1.779e-4
    n2=-1.05e-6
    n3=1.6e-8
    n4=-2.02e-6
    n5=15.868
    n6=0.01155
    n7=-0.00423
    n8=-4382
    n9=1.1455e6

    n0_4=n0+(n1+n2*T+n3*T**2)*S+n4*T**2
    n5_7=n5+n6*S+n7*T
    wv = np.array(wv, dtype=np.float)
    mw=n0_4+n5_7*(wv**-1)+n8*(wv**-2)+n9*(wv**-3)
    return mw

def fresnel(m ,ang):
    """
    This function calculates the Fresnel reflectances for electric vector
    parallel (Rp), perpendicular (Rr) and unpolarized incident light.
     The reflection matrix = 
     [R11, R12, 0; R12, R11, 0; 0, 0, R33]
     Only accounts for I, Q, U and ignore the V component.
     Revision History
     2016-07-10:   1st version, just compute R11, i.e, R
     2016-12-14:   add other reflection matrix elements R12 and R33
                   Also found an error in the previous equaiton for Rp1

    Inputs
    ------
    m : Relative refractive index
    ang : Reflectance (incident) angle

    Outputs
    -------
    R : Fresnel reflectance matrix element (1, 1)
    R12 : Fresnel reflectance matrix element (1, 2)
    R33 : Fresnel reflectance matrix element (3, 3)
    """
    ang = ang[:] # column vector
    # m = m[:] #'  # row vector
    # m = m.reshape(m.size,1)
    # m = np.rot90(m)

    cosang = abs(np.cos(ang))  # cosine of incident angle
    sinangr = np.sin(ang)*(1/m)  # sine of refraction angle
    cosangr = (1-sinangr**2)**0.5  # cosine of refraction angle

    # # reflection coefficient for perpendicular incident light
    tmp = cosangr*m
    Rr1 = (cosang - tmp)/(cosang + tmp)
    # # Rr1=(cosang-m*cosangr)./(cosang+m*cosangr) 

    # # reflection coefficient for parallel incident light
    tmp = cosang*m
    # this was previous one
    # Rp1 = bsxfun(@minus,cosangr,tmp)./bsxfun(@plus,cosangr,tmp)     
    Rp1 =  (tmp - cosangr)/(cosangr + tmp)
    # Rp1=(cosangr-m*cosang)./(cosangr+m*cosang);
    
    Rr = np.abs(Rr1)**2 # reflectance for perpendicular incident light

    Rp = np.abs(Rp1)**2  # reflectance for parallel incident light

    R = (Rr+Rp)/2 
    R12 = (Rp-Rr)/2 
    R33 = np.real(Rr1*np.conj(Rp1))
    
    return [R, R12, R33]


def my_interpn(dbPoints, dbArray, interpArray):
    '''
    Interpolates the n-D array from the database defined by axes/values dbPoints 
    to the points defined in interArray
    
    Inputs
    ---
    dbPoints : tuple of n arrays, each defining the values that result in values in dbArray
    dbArray : n-D array of model outputs in the database
    interpPoints : tuple of n arrays, each defining the new values at which to interpolate dbArray
    
    Outputs
    ---
    interpArray : dbArray values interpolated to interPoints
    '''            
    # mg = np.meshgrid(interpArray)
    # interp_points = np.moveaxis(mg, 0, -1)
    # result_presqueeze = interpn(dbPoints,
    #                             skyrad0, interp_points)
    # result = np.squeeze(result_presqueeze, axis=(0,1))
    
    # da = xr.DataArray(name='dbArray',
    #                 data=dbArray,
    #                 dims=[])
    # return result


def Main(env, sensor):
    """
    Computes sea surface reflectance of skylight.
    Based on: Zhang, X., S. He, A. Shabani, P.-W. Zhai, and K. Du. 2017. Spectral sea
    surface reflectance of skylight. Opt. Express 25: A1-A13, doi:10.1364/OE.25.0000A1.
    Translated from Matlab by D. Aurin 1/2020

    Inputs
    ------
    env : Environmental variables C(cloud), od(aerosol optical depth), sal(salinity), wind, wtem(water temp), zen_sun(solar zenith angle)
    sensor: Sensor configurations ang([zenith angle, 180-relative solar azimuth angle]), wv(list of waveband centers)

    Outputs
    -------
    ρ : Spectral sea surface reflectance of sun/sky glint including sun(solar rho), sky(sky rho), sca2vec(), rho(total rho)
    """
    # calls:
    #   deg2rad()✅
    #   my_sph2cart()✅
    #   find_quads()
    #   get_prob()
    #   sum()
    #   sw_Fresnel()
    #   squeeze(interpn())
    #   bsxfun()
    #   sum(bsxfun())
    #   squeeze(interpn())
    #   get_vec_polar(deg2rad())
    #   prob_reflection()

    load_db()
    tmp = np.arange(1,skyrad0.data.shape[1]+1)

    sensor['pol'] = np.deg2rad(sensor['ang']) # the sensor polar coordinate
    sensor['vec'] = my_sph2cart(sensor['pol'][1], sensor['pol'][0]) # sensor vector
    sensor['ang2'] = sensor['ang'] + np.array([0, 180])
    sensor['pol2'] = np.deg2rad(sensor['ang2']) # the skylight polar coordinate
    sensor['loc2'] = find_quads(*sensor['pol2'])
    
    # Probability and reflection angle of reflecting skylight into the sensor
    prob, angr_sky = get_prob(env['wind'], sensor['vec'])

    tprob = np.sum(prob,1)
    ref = sw_fresnel(sensor['wv'],angr_sky,env['wtem'],env['sal'])

    # skyrad = np.squeeze(interpn(db.zen_sun.data,db.od.data,tmp,db.wv.data,skyrad0.data,
    #     env['zen_sun'],env['od'],tmp,sensor['wv']))
    # skyrad = my_interpn(db.zen_sun.data,db.od.data,tmp,db.wv.data,skyrad0.data,
    #     env['zen_sun'],env['od'],tmp,sensor['wv'])
    solzen = db.zen_sun.data.flatten()
    aod = db.od.data.flatten()
    index = np.arange(1,skyrad0.data.shape[1]+1)
    wave = db.wv.data.flatten()
    da = xr.DataArray(name='skyrad0',
                data=skyrad0.data,
                dims=['wave','index','aod','solzen'],
                coords=[wave, index, aod, solzen])
    skyradXR = da.loc[sensor['wv'], index, env['od'], env['zen_sun']].squeeze()
    skyrad = np.swapaxes(skyradXR.data, 0, 1)
    # skyrad = skyrad0.loc[env['zen_sun'],env['od'],tmp,sensor['wv']].squeeze()


    N0 = skyrad[sensor.loc2.data]
    N = skyrad/N0
    # rho.sky = sum(bsxfun(@times,ref.*N,prob/tprob),1)
    rho.sky = np.sum((ref * N) * (prob / tprob),1)


    rho= None

    return rho
