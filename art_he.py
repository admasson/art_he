#######################################################################################################################################

## Module holding the python version of Manuel Lopez-Puertas' Radiative transfert code for He: granada_RT -> "Granada Radiative Transfert code for He" G
## Coding : utf-8
## Author : Adrien Masson (adrien.masson@obspm.fr)
## Date   : February 2024

# This model is the Python adaptation of Manuel Lopez-Puertas' radiative transfert code for the He metastable triplet study.
# The files containing the original IDL code are: ctd_he_12.pro, ctd_he_12_transits_ip.idl, tra_helium_vel2.pro
# This file contains the model as a class. It needs to be instantiated with the planetary & model parameters and can then be used
# for direct He line computation, for retrieval (of mass loss rate, Temperature, wind velocity & H/He) given some data, and for Chi² maps with some data.

# It is heavely recommended to run the example notebook "manuel_model.ipynb" for a full example on how to use this model !

#######################################################################################################################################

#-------------------------------------------------------------------------------
# Import usefull libraries and modules
#-------------------------------------------------------------------------------
#<f Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import astropy.units as u
import datetime
import time
from astropy import constants as const
from astropy.convolution import convolve
from astropy.io import fits
from astropy.modeling.functional_models import Voigt1D
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.special import voigt_profile
from PyAstronomy import pyasl
from scipy import optimize
import os
import copy
import math
# some global variables
cpu_count = os.cpu_count()-1 # will use all available cpu except 1 by default, change here if you want to use less/more
#f>

# Safety check when module has been imported for debugging purpose
print(f'{datetime.datetime.now().time()} : granada_RT.py has been loaded')

def find_nearest(array, value):
    ''' find index of the closest element to the given value in an array'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

def lsf_carmenes(lbda,x):
    '''
    Compute CARMENES' LSF depending on the wavelength (CARMENES has a wavelength dependent LSF)
    Supplied to Manuel Lopez-Puertas by Evangelos

    - lbda: the central wavelength (in micron) around which CARMENES' LSF is computed and sampled
    - x: the wavelengths (in microns) used for the sampling of CARMENES' LSF around lbda
    - output : ils_vgt
    '''

    # NIR FWHM (DeltaL/L) supplied by Evangelos
    res_gauss=1.18e-5 # constant for Doppler part in Voigt
    res_lorentz=1.7e-6 # constant for Lorentz part in Voigt

    # ad checked with Evangelos
    ad=res_gauss*lbda/ (2.*np.sqrt(np.log(2)))
    al=res_lorentz*lbda/2.
    a= (al/ad)
    u = (x/ad)
    v1 = Voigt1D(a)
    ils_vgt=(1./(ad*np.sqrt(np.pi)))*v1(u)
    ils_vgt=ils_vgt/np.max(ils_vgt)

    return(ils_vgt)

class He_model:
    #<f Documentation
    '''
    This class holds the radiative transfert calculation of the metastable triplet lines, taking into account the transit geometry (keplerian orbit, transit phase and position of the planet and its atmosphere in front of the stellar disk),
    and the different absorbing layers of escaping atmosphere crossed by the stellar light along the line of sight (LOS) at a given altitude.

    Usage:
    - first initialize the class:       $ MyModel = He_model() # here you can pass a verbose=True or False to enable/disable the log
    - then set the model's parameters:  $ MyModel.set_params(params) # see the set_params() documentation to see how to define the parameters in the "params" dictionnary
    - finally compute the model:        $ MyModel.compute_model(zlev,abund_profile,vel_profile,plot=True) # where zlev is the array of vertical layers at which computing model, and abunbd_profile & vel_profile are the hydrodynamical profiles of the escape (from your custome calculation or from pwinds)
    '''
    #f>

    def __init__(self,verbose=True):
        '''
        Simply initialize the class with all its attributes.
        Set verbose = False if you don't want the log
        '''
        # planet & star properties
        self.r_planet           = None # km
        self.r_star             = None # km
        self.limb_dark_coeff    = None # limb darkening coefficients: must have 0, 2, or 4 elements (0: no LD, 2: quadratic LD law, 4: non-linear LD law)
        # orbital properties
        self.b                  = None # impact parameter
        self.a                  = None # semi-major axis (m)
        self.per                = None # orbital period (BJDTBD)
        self.e                  = None # eccentricity (btw 0 & 1)
        self.Omega              = None # spin-orbit orbital angle (°)
        self.w                  = None # argument at periastron (°)  In case of circular orbit, w is -90 if define in the planetary orbit, and +90 if define in the stellar orbit (see pyasl.KeplerEllipse documentation)
        self.i                  = None # orbital inclination (°)
        self.time_samples       = None # array of floats -> the transit phases, in time from mid transit (days in BJD-TDB) at which computing the transit geometry
        self.transit_grid_size  = None # size of the grid used for computing the transit geometry. The grid must have enough resolution such that the smallest altitude layers covers at least 2 pixels of the grid. Otherwise, the model will throw an error and suggest a minimum nb of pixels for the transit grid size. If computation gets too long, another solution is to reduce the number of altitude layers or to space them linearly.
        # atmospheric grid parameters
        self.dnu                = None # wavelength step, microns
        self.nrp                = None # Vertical extension of the grid (in number of Rp)

        # Broadening parameters
        self.rad_vel            = None # if True, applies the lat/longitudinal depedent velocities profile
        self.fac_d              = None # Doppler broadening (doppler thermal + turbulent, see Lampon et al 2020) -> corresponds to "alpha_D" has defined in Lampon et al 2020
        self.vshift             = None # km/s Overall shift of the whole signature (-> global escape velocity)
        # Long/latitude dependent doppler shift. /!\ vb & vr must be >= 0 ! The appropriate sign will be given later in the program to get an appropriate blue/red shift
        self.vb                 = None # km/s >= 0, blueshift of the blueshifted part of the planet surface
        self.vr                 = None # km/s >= 0, redshift of the redshifted part of the planet surface
        self.frac_red           = None # fraction (from 0 to 1) of the planet surface corresponding to a redshift
        self.frac_blue          = None # fraction (from 0 to 1) of the planet surface corresponding to a blueshift
        # whether wavelengths are defined in air or vacuum
        self.air                = None # if True, wavelengths are defined in air. False = vacuum

        # converting in meters for convenience in stellar grid calculation
        self.Rs                 = None # stellar radius in meters
        self.Rp                 = None # planetary radius in meters

        # verbose
        self.verbose            = verbose # If True then it will print the log

        if self.verbose: print('Class initialize. You may now use the class .set_params() to set the model parameters.')

    def set_params(self,params):
        '''
        Set the model parameters using the "params" dictionnary provided has argument. The dictionnary must have the following format:
        # planet & star properties
        params['r_planet']        : float # km
        params['r_star']          : float # km
        params['limb_dark_coeff'] : array of float  # limb darkening coefficients: must have 0, 2, or 4 elements (0: no LD, 2: quadratic LD law, 4: non-linear LD law)

        # orbital properties
        params['b']              : float # impact parameter
        params['a']              : float # semi-major axis (m)
        params['Porb']           : float # orbital period (BJDTBD)
        params['e']              : float # eccentricity (btw 0 & 1)
        params['lbda']           : float # spin-orbit orbital angle (°)
        params['w']              : float # argument at periastron (°)  In case of circular orbit, w is -90 if define in the planetary orbit, and +90 if define in the stellar orbit (see pyasl.KeplerEllipse documentation)
        params['i']              : float # orbital inclination (°)

        # atmospheric grid parameters
        params['dnu']           : float # microns, wavelength step, microns
        params['nrp']           : int   # Vertical extension of the grid (in number of Rp)
        params['z1']            : int   # altitude, in Rp, of the lowest level
        params['dz']            : float # step, in Rp, between each altitude layer
        params['time_samples']  : array of floats # the transit phase, in time from mid transit (days in BJD-TDB) at which computing the transit geometry
        params['transit_grid_size'] : int # size of the grid used for computing the transit geometry. The grid must have enough resolution such that the smallest altitude layers covers at least 2 pixels of the grid. Otherwise, the model will throw an error and suggest a minimum nb of pixels for the transit grid size. If computation gets too long, another solution is to reduce the number of altitude layers or to space them linearly.

        # Broadening parameters
        params['rad_vel']       : bool  # if True, applies the lat/longitudinal depedent velocities profile
        params['fac_d']         : float # Doppler broadening (doppler thermal + turbulent, see Lampon et al 2020) -> corresponds to "alpha_D" has defined in Lampon et al 2020
        params['vshift']        : float # km/s Overall shift of the whole signature (-> global escape velocity)

        # Long/latitude dependent doppler shift. /!\ vb & vr must be >= 0 ! The appropriate sign will be given later in the program to get an appropriate blue/red shift
        params['vb']            : float # km/s >= 0, blueshift of the blueshifted part of the planet surface
        params['vr']            : float # km/s >= 0, redshift of the redshifted part of the planet surface
        params['frac_red']      : float # fraction (from 0 to 1) of the planet surface corresponding to a redshift
        params['frac_blue']     : float # fraction (from 0 to 1) of the planet surface corresponding to a blueshift

        # whether wavelengths are defined in air or vacuum
        params['air']           : bool  # if True, wavelengths are defined in air. False = vacuum

        # Files name
        params['profile_name']  : str   # name of the file containing the He I triplet abundance profile
        params['profile_path']  : str   # path to the hydrodynamical profiles/grid
        params['vel_name']      : str   # name of the velocity profile name
        params['vel_path']      : str   # path to the velocity profiles folder
        '''

        # Set model parameters
        self.r_planet           = params['r_planet']
        self.r_star             = params['r_star']
        self.b                  = params['b']
        self.dnu                = params['dnu']
        self.rad_vel            = params['rad_vel']
        self.fac_d              = params['fac_d']
        self.vshift             = params['vshift']
        self.vb                 = params['vb']
        self.vr                 = params['vr']
        self.frac_red           = params['frac_red']
        self.frac_blue          = params['frac_blue']
        self.air                = params['air']
        self.nrp                = params['nrp']
        self.limb_dark_coeff    = params['limb_dark_coeff']
        self.a                  = params['a']     # m
        self.per                = params['Porb']  # BJDTBD
        self.e                  = params['e']
        self.Omega              = params['lbda']  # °
        self.w                  = params['w']     # °  In case of circular orbit, w is -90 if define in the planetary orbit, and +90 if define in the stellar orbit (see pyasl.KeplerEllipse documentation)
        self.i                  = params['i']     # °
        self.time_samples       = params['time_samples'] # the transit phase, in time from mid transit (days) at which computing the transit geometry
        self.transit_grid_size  = params['transit_grid_size'] # size of the grid used for computing the transit geometry. The grid must have enough resolution such that the smallest altitude layers covers at least 2 pixels of the grid. Otherwise, the model will throw an error and suggest a minimum nb of pixels for the transit grid size. If computation gets too long, another solution is to reduce the number of altitude layers or to space them linearly.

        # converting in meters for convenience in stellar grid calculation
        self.Rs                 = self.r_star*1e3   # stellar radius in meters
        self.Rp                 = self.r_planet*1e3 # planetary radius in meters

        # Fixed parameters for the model
        self.params = copy.deepcopy(params)

        # compute orbit
        self.compute_kepler_orbit()

        # define the area in which everything the model is computed -> IN MICRONS !!
        # Air
        if self.air:
            self.xmin=1.08270
            self.xmax=1.08330
            self.xminn=1.0828  # xmin for plotting
            self.xmaxx=1.08320 # xmax for plotting
        # Vacuum
        else:
            self.xmin= 1.0828
            self.xmax= 1.0838
            self.xminn=1.0830  # xmin for plotting
            self.xmaxx=1.0836  # xmax for plotting

        # nb of points on which the He line transmission model is computed
        self.tr_np = int((self.xmax-self.xmin)/self.dnu)      # nb of wavelength points
        self.wn    = self.xmin+np.arange(self.tr_np)*self.dnu # wavelength vector corresponding to the transmission model

        if self.verbose: print('Model succesfully parametrised')

    def compute_kepler_orbit(self):
        '''
        Uses pyasl.KeplerEllipse to compute the planetary elliptical orbit and store it as an attribute
        transit (dic) : a dictionnary containing the informations of the transit (see transit_info.py for an example)
        more information on pyasl.KeplerEllipse definition : https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/keplerOrbitAPI.html

        position and velocity can then be access with respect to time using self.ke.xyzPos(time) & self.xyzVel(time) for a given time.
        The unity corresponds to those provided in the transit dictionnary, e.g if transit['a'] is in meters and transit['Porb'] in seconds, coordinates will be in meter & velocity in m/s

        In order to align the origin of time with the KeplerEllipse orbit, we compute the time of mid transit corresponding to the model and store it in self.model_midpoint
        Thus, time from mid transit can be convert from data_set time to keplerEllipse time using self.time_from_mid+self.model_midpoint
        '''

        # Set the orbital model
        a       = self.a         # m
        per     = self.per       # BJDTBD
        e       = self.e
        Omega   = self.Omega     # °
        w       = self.w         # °  In case of circular orbit, w is -90 if define in the planetary orbit, and +90 if define in the stellar orbit (see pyasl.KeplerEllipse documentation)
        i       = self.i         # °

        # the KeplerEllipse object is stored as an attribute
        self.ke = pyasl.KeplerEllipse(a=a, per=per, e=e, Omega=Omega, w=w, i=i, ks=pyasl.MarkleyKESolver)

        # find the mid-transit time of the KeplerEllipse models, defined as the time at which the modeled orbital position is the closest to the stellar center.
        def f(time):
            pos = self.ke.xyzPos(time)
            r = np.sqrt(pos[:,0]**2 + pos[:,1]**2 )/self.Rs
            return(r)

        # time in BJDTBD
        time = np.linspace(0,per,100)
        pos = self.ke.xyzPos(time)

        # find orbital model's midpoint i.e corresponding to the minimal distance to the stellar surface in the xy plan
        # only select point during primary transit, otherwise planet could be behind the star
        r = np.sqrt(pos[:,0]**2 + pos[:,1]**2 )/self.Rs  # distance from center of star during primary transit in stellar radii
        m = np.logical_and(pos[:,2]<=0, r<=1.0)                         # z negative = planet in front of star, sqrt(x**2+y**2)<Rs = planet crossing stellar surface
        result = optimize.minimize(f, x0=time[m][m.sum()//2])           # we take the mean position of the planet in front of the stellar disk as our first guess
        self.model_midpoint = result.x                                  # transit's midpoint, in BJDTBD, corresponding to the KeplerEllipse model

    def custom_draw_transit(self,grid_size,time_from_mid,plot=False,transit=True):
        '''
        Simulate the stellar surface in a grid with pixels set as 0 behind the transiting planetary disk for a given time from mid transit.
        The pixels are normalized such that the sum of all pixels equals 1 when off-transit
        The planet's trajectory is computed by calling self.compute_kepler_orbit, taking into account eccentricity & spin-orbit alignement.
        The stellar grid also takes into account non-linear limb darkening if it has been provided to the transit_info dictionnary during class initialization

        Takes:
        - grid_size     : int, the size of the stellar radius, in pixels.
        - time_from_mid : array of int, contains the times from mid-transit, in days (BJD-TBD), at which sampling the planet's position
        - plot          : bool, set to True to show the grid
        - transit       : bool, whether or not add the transiting planet disk. Default is True, set to False to get an out of transit flux map

        Returns:
        - flux_map      : a 2D grid of size (grid_size,grid_size), containing the normalized flux in each pixel. The sum of all pixels equals 1 when off transit, and pixels are set to 0 outside stellar disk & behind planetary disk if in-transit
        - transit_depth : float, the transit_depth due to the opaque planetary disk (at 1Rp) at given time
        - r_map         : a 2D grid containing the distance, in meters, of each pixel from the planetary center
        '''

        def limb_dark(mu):
            # we check what limb darkening (LB) law to use depending on the number of coefficients given in the transit_dic
            # mu is the "limb darkening angle", see http://lkreidberg.github.io/batman/ for examples
            # 4 coeff -> non-linear LD law
            if len(self.limb_dark_coeff)==4:
                c1,c2,c3,c4 = self.limb_dark_coeff                                                     # get the LD coeff
                limb_avg = 1-c1-c2-c3-c4+0.8*c1+(2./3.)*c2+(4./7.)*c3+0.5*c4                            # compute the disk-averaged LD intensity for normalisation
                limb = limb_avg * (1-c1*(1-np.sqrt(mu))-c2*(1-mu)-c3*(1-np.power(mu,3/2))-c4*(1-mu**2)) # normalise the LD law to be unity when disk-averaged
                # 2 coeff -> quadratic LD law
            elif len(self.limb_dark_coeff)==2:
                    c1, c2 = self.limb_dark_coeff  # get the coeffs
                    limb = 1-c1*(1-mu)-c2*(1-mu)**2 # disk-average normalisation
                # 0 coeff -> no LD (so we set it as unity everywhere)
            elif len(self.limb_dark_coeff)==0:
                    limb = np.zeros(mu.shape)
                    limb[np.isfinite(mu)] = 1.
            # if the number of coeffs doesn't match any of the above: raise an error.
            else:
                raise NameError(f'Limb dark function with coeff {self.limb_dark_coeff} have not been implemented in this function : need to update using formula in http://lkreidberg.github.io/batman/docs/html/tutorial.html#limb-darkening-options')
            return(limb)

        # compute the kepler elliptical orbit model
        position = self.ke.xyzPos(time_from_mid+self.model_midpoint)    # position array from keplerEll computation
        x_planet = position[:,0]/self.Rs                 # x axis position, in stellar radii, of the planet in the grid
        y_planet = position[:,1]/self.Rs                 # y axis position, in stellar raddi, of the planet in the grid

        # set the grid: we define everything in stellar radii, so the grid extend to -1 to +1 stellar radii in both x & y axis
        X = np.linspace(-1.0, 1.0, grid_size)
        Y = np.linspace(-1.0, 1.0, grid_size)
        X, Y = np.meshgrid(X, Y)

        # 2D grid containing the distance of each pixel from the planet's center at a given time from mid-transit
        r_map = np.sqrt((X-x_planet)**2 + (Y-y_planet)**2)*self.Rs # distance of each cell from planet center, in meters

        r_s = np.sqrt(X**2+Y**2) # grid containing the distance of each pixel to the stellar center (at (0,0)). Used for building the stellar disk
        # alpha and mu angles are spherical coordinates for the stellar surface
        alpha = np.arcsin(r_s)
        mu = np.cos(alpha)      # limb dark angle
        limb = limb_dark(mu)    # the limb-darkening intensity map of the star

        flux_map = np.copy(limb)              # fill stellar disk with limb darkening
        flux_map[~np.isfinite(flux_map)] = 0. # fill the outside of stellar surface with 0.

        flux_map /= flux_map.sum()     # Normalize the in-transit stellar flux grid by the disk-averaged off-transit intensity grid of the star
        intensity_off = flux_map.sum() # total off-transit sum of the cell's grid. should be very close to 1 (if not then normalisation failed somehow)

        dx = self.Rs/grid_size # size of a pixel, in meters
        if transit: flux_map[r_map<=(self.Rp+dx)] = 0. # we add a "dx", which is the size of a pixel, to also incorporate pixels which are not fully covered by the planetary disk

        intensity_on = flux_map.sum() # compute the total intensity during the transit

        transit_depth = intensity_off-intensity_on # compute transit depth due to planet surface (at 1Rp) at this given time of the transit

        # show transit map
        if plot:
            plt.figure()
            plt.imshow(flux_map,origin='lower')
            plt.colorbar()
            plt.plot(x_planet,y_planet,'r+',label=f'planet center at T = {time_from_mid}')

        return flux_map, transit_depth, r_map

    def area_intersec(self,zmin,zmax,time_from_mid,flux_out,flux_in,r_planet,plot=False):
        '''
        Compute the intersection area between a layer of the atmosphere (extending from zmin to zmax) and the stellar surface disk,
        at a given time from mid transit and using the limb darkening law if any in self.params.

        takes:
            zmin, zmax: lower and upper limit of the atmospheric layer, in stellar radii and with origine at 1Rp (ie at planetary surface)
            time_from_mid: time from mid transit at which computing the area. In days (BJD-TBD), < 0 if before mid transit
            grid_size: the size of the grid used to compute the stellar surface. We recommend a sampling of ~1000 points to get an accurate estimation of the intersec area
            plots: whether or not to plot the transit grid

        returns:
            area_intersec: the intersection area between the layer and the stellar surface disk (+ limb darkening), in the units of self.params['r_star']**2
        '''
        position = self.ke.xyzPos(time_from_mid+self.model_midpoint) # position array from keplerEllipse computation at given time from mid transit
        x_planet = position[:,0]/self.Rs # x axis position, in stellar radii, of the planet in the grid
        y_planet = position[:,1]/self.Rs # y axis position, in stellar raddi, of the planet in the grid

        r_planet = (r_planet-self.Rp) / self.Rs # distance, in Rs, of any point from the planet surface. Remove 1Rp because altitude is defined with origin at 1Rp

        intersec = (r_planet>=zmin) * (r_planet<=zmax) * flux_in # 2D grid of the intersection. Contains the flux map where the amtospheric layer intersec the stellar disk

        area_intersec = (intersec.sum() / flux_out.sum())*self.r_star**2 # intersection area: cross product btw nb of pixel in area, tot nb of pixel in stellar disk, & real stellar disk

        if plot: # plot the 2d flux map with the intersection area at given atmospheric layer
            plt.figure()
            plt.imshow(flux_in,origin='lower',extent=[-1,1,-1,1])
            plt.imshow(intersec,origin='lower',extent=[-1,1,-1,1],alpha=0.5)
            plt.xlabel('Stellar Radius')
            plt.ylabel('Stellar Radius')
            plt.title('Transit geometry: intensity map')
            c1 = plt.Circle((x_planet,y_planet),zmin+(self.Rp/self.Rs),color='r',fill=False)
            plt.gca().add_patch(c1)
            c2 = plt.Circle((x_planet,y_planet),zmax+(self.Rp/self.Rs),color='r',fill=False)
            plt.gca().add_patch(c2)

        return area_intersec

    def slantp(self,zmin,z1,z2):
        '''
        Compute the slant incidence from Wang et al. JAS, 1984
        -> calculates the distance (km) btw two points z1 (km) & z2 (km)
        along a line with origin zmin (km) forming a solar zentigh angle sza=90°
        -> This function is used to integrate all points along the LOS for a given altitude layer
        '''
        sza = 90 # °
        dzs = sza*2.*np.pi/360
        h = self.r_planet+zmin
        ht1 = self.r_planet+z1
        ht2 = self.r_planet+z2

        slantp = np.sqrt(ht2**2 - (h*np.sin(dzs))**2) - np.sqrt(ht1**2 - (h*np.sin(dzs))**2)

        return(slantp)

    def area_sector(self,R,h):
        return(R**2 *np.arccos((R-h)/R) - (R-h)*np.sqrt(2.*R*h-h**2))

    def helium_lineshape(self,th_i):
        '''
        This function return the He transmission line shape at a given altitude level "th_i", taking into account the altitude layers
        crossed along the Line of Sight (LOS) and the radial velocity of the escape if rad_vel = True
        '''

        # some constants
        M_he        = 4.002602          # Helium mass number
        vlight      = const.c.value     # m/s
        kb          = const.k_B.value   # Blotzmann constant
        amu         = 1.660539040e-27   # atomic mass unit in kg
        M           = M_he*amu          # He atomic mass in kg

        lambda0     = np.array([10832.057 ,10833.217, 10833.307])               # He triplet lines wavelength in Vacuum, Angstroms
        if self.air == 1: lambda0=np.array([10829.0911,10830.2501,10830.3398])  # He triplet lines in Air, Angstroms

        lambda0*=1e-4 # convert in microns

        xsec = np.array([6.2246205e-28,1.8663579e-27, 3.1106996e-27]) # He lines cross sections (m2 m)
        xsec = xsec*1.e4*1.e6 # convert in(cm^2 micron)

        # prepare transit geometry
        tra  = np.ones(self.wn.shape)    # this array will hold the He line's shape
        z    = self.zlev                 # array containing the altitude level for the extension's sampling
        nz   = len(z)                    # nb of altitude levels
        zmin = th_i                      # min altitude level
        nzi,val = find_nearest(z,th_i)   # equivalent to IDL function "value_locate"
        nzf  = nz-2

        c = const.c.value/1e3 # speed of light, km/s

        # integrating over the points along the line of sight (LOS) at the given altitude. The integration is done from the upper layer to the bottom one
        for i in np.arange(nzf,nzi-1,-1):
            # get the upper and lower boundary of the current altitude's layer
            z1 = z[i]
            z2 = z[i+1]
            dzpath = self.slantp(zmin,z1,z2) # the absorption crossed by the light along the LOS at the given altitude z -> column density
            sumhe = self.hez[i]*dzpath*1e5   # absorption coefficient: product of the He abundance (self.hez) with the column density (dzpath). The 1e5 factor is for converting from km to cm, because the cross sections are in cm²

            # include the component of the radial vel along LOS
            if self.rad_vel:
                y = (self.r_planet+zmin)/(self.r_planet+(z1+z2)/2.)
                ang = np.arccos(y) # angular coordinate over the planet
                v_wind = (self.vel[i]+self.vel[i+1])/2. * np.sin(ang) # wind's speed, in km/s
            else:
                v_wind = 0

            # Limb path side closer to the observer. Wind approaching to the observer
            bshift=c/(c+v_wind)
            Lam1=lambda0[0]*bshift
            Lam2=lambda0[1]*bshift
            Lam3=lambda0[2]*bshift
            ad1=Lam1/vlight*np.sqrt(self.fac_d*kb*self.t_d/M)
            ad2=Lam2/vlight*np.sqrt(self.fac_d*kb*self.t_d/M)
            ad3=Lam3/vlight*np.sqrt(self.fac_d*kb*self.t_d/M)
            ils_dop1b=1./(ad1*np.sqrt(np.pi)) * np.exp(-((self.wn-Lam1)/ad1)**2.)
            ils_dop2b=1./(ad2*np.sqrt(np.pi)) * np.exp(-((self.wn-Lam2)/ad2)**2.)
            ils_dop3b=1./(ad3*np.sqrt(np.pi)) * np.exp(-((self.wn-Lam3)/ad3)**2.)

            # Limb path side farther away from the observer.
            # Assume homogeneous atmos. in both sides, same sumhe
            rshift=c/(c-v_wind)
            Lam1=lambda0[0]*rshift
            Lam2=lambda0[1]*rshift
            Lam3=lambda0[2]*rshift
            ad1=Lam1/vlight*np.sqrt(self.fac_d*kb*self.t_d/M)
            ad2=Lam2/vlight*np.sqrt(self.fac_d*kb*self.t_d/M)
            ad3=Lam3/vlight*np.sqrt(self.fac_d*kb*self.t_d/M)
            ils_dop1r=1./(ad1*np.sqrt(np.pi)) * np.exp(-((self.wn-Lam1)/ad1)**2.)
            ils_dop2r=1./(ad2*np.sqrt(np.pi)) * np.exp(-((self.wn-Lam2)/ad2)**2.)
            ils_dop3r=1./(ad3*np.sqrt(np.pi)) * np.exp(-((self.wn-Lam3)/ad3)**2.)

            #Combining both paths
            ils_dop1=ils_dop1b+ils_dop1r
            ils_dop2=ils_dop2b+ils_dop2r
            ils_dop3=ils_dop3b+ils_dop3r

            # computing the optical depth and the final line's shape
            tau=(ils_dop1*xsec[0]+ils_dop2*xsec[1]+ils_dop3*xsec[2])*sumhe
            tra*=np.exp(-tau)

        return(tra)

    def compute_model(self,zlev,t_d,abund_profile,vel_profile=[],plot=False):
        '''
        Compute the model for the given parameters:
        - zlev : the layers level (in km) used to sample the altitude grid. The grid must start at the planet surface (at 1Rp) and with the origine at planet "surface" (i.e the lowest layer in the grid is "0km" and correspond to an altitude of 1Rp)
        - t_d : the temperature of the escape, must correspond to the one used to generate the profiles
        - abund_profile : array with as many elements as self.zlev : contains the He3 abundance profile (atom/cm³)
        - vel_profile : array, only required if self.rad_vel = True : contains the radial velocity profile (km/s)

        # compute model by averaging model computed at different transit phase in a grid (like pwinds). the time_samples are provided in an array in the params, defined in days as time from mid transit.
        # it is also possible to provide a single value in the time_samples array, for example to use an effective time that give the same output as if several transit phase were averaged together.
        '''

        # store the altitude grid & temperature
        self.zlev = zlev
        self.t_d  = t_d
        if self.verbose:print('\n----------- Computing Model -----------')
        # for computation time recording
        t0 = time.time()

        if not self.rad_vel and len(vel_profile)==0: raise NameError('rad_vel is True: you must provide a velocity profile path')

        # store the lower and upper levels of the altitude grid
        self.th0 = 0.*self.r_planet
        self.th = self.zlev
        self.nth = len(self.th) # nb of altitude layers

        # Define the final transmission array size, given we have excluded atmospheric layers before
        self.trm_phase  = np.zeros((self.time_samples.size,self.tr_np))   # contains the final tranmsission spectrum at each phase
        self.trm        = np.zeros(self.tr_np)                                      # array containing the final transmission model
        self.tr         = np.zeros((self.nth,self.tr_np))                           # array containing the transmission model for each layer
        self.weights    = np.zeros((self.time_samples.size,self.nth))     # array containing the contribution weight of each atmo layer to the total transmission

        # store the He3 abundance & velocity profiles
        self.hez = abund_profile
        self.vel = vel_profile

        # Compute the line shape with doppler broadening (radial velocity profile + turbulence) and transit geometry
        for i in range(self.nth): # loop over altitude's layers
            self.tr[i] = self.helium_lineshape(self.th[i])
        if plot: # plot the abundance & velocity profiles
            plt.figure()
            for i in range(self.nth):
                if i%10==0:
                    plt.plot(self.wn,self.tr[i],label=f'{self.th[i]/self.r_planet:.1f}Rp')
            plt.legend()
            plt.title('Individual line shape from the altitude layers')
            plt.xlabel('microns')
            plt.ylabel('Flux')

        # Apply the global shift to the whole signature
        c = const.c.value
        bshift = c / (c-self.vshift*1e3)
        f = interp1d(self.wn*bshift,self.tr,fill_value='extrapolate')
        if plot: # plot to check the shift was one in the correct direction
            plt.figure()
            plt.plot(self.wn,self.tr[0],label='before overall shifting')
        self.tr = f(self.wn) # applying the overall shift to the line profiles (same shift for all altitude layers)
        if plot:
            plt.plot(self.wn,self.tr[0],label='after overall shifting')
            plt.legend()

        # Calculate the annular (or segment) area of the altitude layer intersecting with the stellar disk, and the mean absorption, for each transit phase
        for kk,time_from_mid in enumerate(self.time_samples): # loop over transit phases
            sum0 = np.zeros(self.tr_np)
            # compute transit geometry
            flux_map,transit_depth,r_planet = self.custom_draw_transit(self.transit_grid_size,time_from_mid,plot=plot) # in-transit flux map, transit depth and distance from planet center in 2D grids at the given transit phase
            f_out,__,__ = self.custom_draw_transit(self.transit_grid_size,time_from_mid,transit=False) # off-transit flux map
            # for each altitude layer, compute its absorption considering all other layers crossed along the LOS & the intersection with the stellar disk
            for ith in range(self.nth-1): # loop over atmospheric layers
                # compute area of intersection between atmopsheric layer and the stellar disk at given phase
                zmin = self.th[ith]/self.params['r_star'] # bottom of current atmospheric layer
                zmax = self.th[ith+1]/self.params['r_star'] # up of current atmospheric layer

                # raise error if the transit grid resolution is lower than half the altitude sampling resolution: we want at least 2 pixels of the grid to lies within the smallest altitude layer to ensure proper sampling
                if (4./(zmax-zmin))>self.transit_grid_size:
                    raise NameError(f'Current transit grid resolution is too low to accurately sample the altitude layers. Please provide a transit_grid_size parameter higher than {math.ceil((4./(zmax-zmin)))}, or lower the number of altitude layers (zlev).')

                dz1 = self.area_intersec(zmin,zmax,time_from_mid,f_out,flux_map,r_planet,plot=False) # area of intersection btw the layer and stellar disk

                # stop if the intersection area is full of 0., meaning the layers are now outside stellar disk
                if dz1 == 0.:
                    if self.verbose: print(f'At time from mid transit = {time_from_mid:2e} bjd-tdb, integration stopped at {self.th[ith]/self.r_planet:.2e} Rp -> exciting stellar disk after this altitude')
                    break # break loop

                sum0 += (1.0-(self.tr[ith]+self.tr[ith+1])/2.0) * dz1

                # calculate the weight of this layer at current phase
                trw1 = np.sum(1-self.tr[ith])
                trw2 = np.sum(1-self.tr[ith+1])
                self.weights[kk,ith] = ((trw1+trw2)/2)*dz1

            self.trm_phase[kk] = -sum0/(self.r_star**2 )
            self.trm_phase[kk]=1+self.trm_phase[kk]

        self.trm = np.mean(self.trm_phase,axis=0)

        t1 = time.time()
        if self.verbose:
            print(f'Done in {(t1-t0)//60:.0f}min {(t1-t0)%60:.0f}sec')
            print(f'---------------------------------------')

    def convolve_CARMENES(self, plot=True):
        '''
        convolve computed model with CARMENES instrumental function
        '''

        # Convolve with CARMENES spectra
        grid=self.dnu
        xmaxx0=8e-5
        xminn0=-xmaxx0
        nk=np.round(2*xmaxx0/grid)+1
        x = xminn0+np.arange(nk)*grid

        # output grid
        wmin=self.wn[0]+1e-4 #Reduced range to avoid problems in the convolution at the edges
        wmax=self.wn[-1]-1e-4

        grido=1.0e-6      # microns: Output grip
        nwo=int(np.round((wmax-wmin)/grido)+1.)
        self.wno=wmin+np.arange(nwo)*grido
        self.tro=np.zeros(nwo)

        # Loop over all output wns
        for i in range(nwo):
            i0,a=find_nearest(self.wn,self.wno[i])
            if np.abs(self.wn[i0] - self.wno[i]) > np.abs(self.wn[i0+1] - self.wno[i]) : i0=i0+1
            i1=np.max([0,i0-nk/2])
            i2=np.min([i0+nk/2,len(self.wn)-1])
            trr=self.trm[int(i1):int(i2)] # int() ??? see L376 & 377 of manuel's IDL code
            # Calculate the ILS at the given wavelength
            kernel=lsf_carmenes(self.wno[i],x)

            trx=convolve(trr,kernel,normalize_kernel=True)
            self.tro[i]=trx[int(nk/2)] # int() ???

        if plot:
            plt.figure()
            plt.plot(self.wn,self.trm,label='Transmission')
            plt.plot(self.wno,self.tro,label='After CARMENES convolution')
            plt.legend()

        if self.verbose: print(f'Convolution with CARMENES instrumental function done !')

    def convolve_SPIRou(self, plot=True):
        '''
        convolve computed model with SPIRou instrumental function
        '''
        pixel_size = 2*2.28e3 # SPIRou element resolution in m/s
        nb_points = 11 # size, in pixel, of the door function used for the convolution

        half_size = pixel_size / 2
        pixel = np.linspace(-half_size,half_size,nb_points)

        convolved_spec = np.zeros(self.trm.size)

        f = interp1d(self.wn,self.trm,fill_value=np.nan)

        for v in pixel:
            # mask wavelength shifted outside the interpolation domain
            mask_down = (self.wn / (1 + v/const.c.value)) < self.wn.min()
            mask_up   = (self.wn / (1 + v/const.c.value)) > self.wn.max()
            mask = np.logical_or(mask_down,mask_up) # contains True where shifted wavelength are outside the valid interpolation domain
            convolved_spec[~mask] += f(self.wn[~mask] / (1 + v/const.c.value))
            # replace values outside range by nan
            convolved_spec[mask] = np.nan

        # normalise
        convolved_spec /= len(pixel)

        # cut invalid values and store
        mask = np.isfinite(convolved_spec)
        self.wno = self.wn[mask]
        self.tro = convolved_spec[mask]

        if plot:
            plt.figure()
            plt.plot(self.wn,self.trm,label='Transmission')
            plt.plot(self.wno,self.tro,label='After SPIRou convolution')
            plt.legend()

    def chi2(self,data,error,use_convolved=True,plot=True):
        '''
        return the chi2 value computed between this model and some data
        if use_convolved=True, chi2 use computed using the convolved model (self.tro)
        - data must have:
            - wavelength in first column in same units as model
            - normalised transmission in second column
        - error must have same shape as data and contains the error on each data point

        if plot=True: plot data Vs model
        '''

        data_wave = data[:,0] # wavelength of the data

        # mask for chi2 computation
        if self.air: chi2_mask = (data_wave > 1.0828)*(data_wave < 1.08315)
        else: chi2_mask = (data_wave > 1.0831)*(data_wave < 1.08345)
        self.chi2_mask = chi2_mask

        x_data = data[chi2_mask,1]
        x_error = error[chi2_mask]

        # interpolate model on data wavelength
        if use_convolved: self.x_model = interp1d(self.wno,self.tro)(data_wave)
        else: self.x_model = interp1d(self.wn,self.trm)(data_wave)
        x_model = self.x_model[chi2_mask]

        # compute chi2
        Chi2 = np.sum((x_data-x_model)**2/x_error**2)

        if plot:
            plt.figure()
            plt.errorbar(data[:,0],data[:,1],yerr=error,label='data',fmt='.-k')
            if use_convolved:plt.plot(self.wno,self.tro,label='model convolved',color='r')
            else: plt.plot(self.wn,self.trm,label='model (not convolved)',color='r')
            if self.air:
                for w in [10829.0911,10830.2501,10830.3398]:
                    plt.vlines(w*1e-4,0.9854,1.005,color='k',ls='dotted')
            else:
                for w in [10832.057 ,10833.217, 10833.307]:
                    plt.vlines(w*1e-4,0.9854,1.005,color='k',ls='dotted')
            plt.hlines(1,self.xminn,self.xmaxx,ls='dotted',color='k')
            # plt.legend()
            plt.xlim(self.xminn,self.xmaxx)
            plt.xlabel("Wavelength [microns]")
            plt.ylabel('Normalized flux')
            # plots weights of the model
            plt.figure()
            plt.plot(self.th/self.r_planet,100*np.mean(self.weights/np.sum(self.weights),axis=0),'k-')#'k+-') # if more than 1 phases have been average, weight is a 2D matrix with first axis being the nb of phases & second axis the nb of altitude layers
            plt.xlabel('Radius [Rp]')
            plt.ylabel('Contribution to total absorption [%]')
            ax2 = plt.twinx(plt.gca())
            ax2.plot(self.th/self.r_planet,np.cumsum(np.mean(self.weights/np.sum(self.weights),axis=0)),color='r')
            ax2.set_ylabel('Cumulative contribution')
            ax2.tick_params(axis='y',colors='red')
            plt.xlim(0,self.nrp)

        return Chi2















































#
