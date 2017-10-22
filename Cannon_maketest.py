
# coding: utf-8

# In[1]:

# Compatibility with Python 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    get_ipython().magic(u'matplotlib inline')
    get_ipython().magic(u"config InlineBackend.figure_format='retina'")
except:
    pass

# Basic packages
import numpy as np
import os
import sys
import glob
import pickle
import astropy.io.fits as pyfits
import scipy
from scipy.stats import norm
from scipy.interpolate import interp1d

# Matplotlib adjustments (you might not need all of these)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.mlab as mlab
import matplotlib.cm as cmx
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
matplotlib.rc('text', usetex = True)
params = {'text.latex.preamble': [r'\usepackage{upgreek}', r'\usepackage{amsmath}'],'font.family' : 'lmodern','font.size' : 11}   
plt.rcParams.update(params)

# Change Work directory (if Gemini2)
try:
    localFilePath = '/shared-storage/buder/svn-repos/trunk/GALAH/'
    os.chdir(localFilePath)
except:
    print('Could not change Path to '+localFilePath)

########################################
# THIS IS THE IMPORTANT PART TO ADJUST #
########################################

# IRAF REDUCTION VERSION

DR                  = 'dr5.2'  # default: 'dr5.2', this code is also compatible with 'dr5.1'
backup_DR_date      = '170523' # insert here only the last known date! By default, the code will try to use the latest
complete_DR         = True     # default: True, otherwise provide files in 'SPECTRA/FIELD/*.fits'
field               = ''       # default: not set, if complete_DR == False, set field name here

# ADDITIONAL CORRECTIONS BY WG4

telluric_correction = True #True
skyline_correction  = True #True
renormalise         = False

# CANNON WAVELENGTH GRID

include_ccd4        = True

# Define the 4 CCD grids for the Cannon leaving at least 20 km/s to ab lines
x1=np.arange(4715.94,4896.00,0.046) # ab lines 4716.3 - 4892.3
x2=np.arange(5650.06,5868.25,0.055) # ab lines 5646.0 - 5867.8 # increased red start
x3=np.arange(6480.52,6733.92,0.064) # ab lines 6481.6 - 6733.4
x4=np.arange(7693.50,7875.55,0.074) # ab lines 7691.2 - 7838.5 # increased red start

########################################
#        END OF PART TO ADJUST         #
########################################

# some initial values:
clight = 299792.458 # speed of light in km/s
large  = 100.


# In[ ]:

########################################
#       IMPORT OF SOBJECT IRAF         #
########################################

# This file was intended for 'dr52' but is also compatible with irafdr51

if DR == 'dr5.2':
    versions = glob.glob('DATA/sobject_iraf_52_*.fits')
else:
    versions = ['DATA/iraf_dr51_09232016_corrected.fits']

# Read in information from the IRAF FITS
door = pyfits.open(versions[-1])
iraf = door[1].data
door.close()

print(versions[-1]+' will be used.')
print('Available entries in IRAF FITS:  '+str(len(iraf['sobject_id'])))


# In[ ]:

########################################
#       FIND OBSERVATION DATES         #
########################################

# First we change the working directory to the DR spectra one
try:
    localFilePath = '/shared-storage/buder/svn-repos/trunk/GALAH/SPECTRA/'+DR
    os.chdir(localFilePath)
except:
    print('Could not change Path to '+localFilePath)

# Find all directories with observations (OBS_DATE: YYMMDD), starting with 1 until 2020 and 2 for 2020-2029
obs_dates            = np.concatenate((glob.glob('1*'),glob.glob('2*')))
obs_dates.sort()
obs_date_file_number_01 = []
obs_date_file_number_02 = []
for each_obs_date in obs_dates:
    # Find number of files in each obs_date directory ending with '1.fits'
    obs_date_file_number_01.append(len(glob.glob(each_obs_date+'/standard/com/*1.fits')))
for each_obs_date in obs_dates:
    # Find number of files in each obs_date directory ending with '1.fits'
    obs_date_file_number_02.append(len(glob.glob(each_obs_date+'/standard/com2/*1.fits')))


try:
    localFilePath = '/shared-storage/buder/svn-repos/trunk/GALAH/'
    os.chdir(localFilePath)
except:
    print('Could not change Path to '+localFilePath)
    
# Print available directories and number of spectra in them

print('Available OBS_DATES:')
print(zip(obs_dates,obs_date_file_number_01,obs_date_file_number_02))
print('Total:')
print(str(len(obs_dates))+' OBS_DATES')
print(str(sum(obs_date_file_number_01))+' 01 FITS and '+str(sum(obs_date_file_number_02))+' 01 FITS')


# In[ ]:

########################################
#  CREATE TEST_DATA FOR EACH OBS_DATE  #
########################################

for each_obs_date in obs_dates:
    
    # Find all FITS (only take those from CCD1 to avoid duplicates)
    
    # Name of the PICKLE file
    if include_ccd4 == True:
        file_in_name = '/shared-storage/buder/svn-repos/trunk/GALAH/CANNON/'+DR+'/pickle_'+DR+'_4ccds/'+DR+'_4ccds_'+each_obs_date+'.pickle'
    else:
        file_in_name = '/shared-storage/buder/svn-repos/trunk/GALAH/CANNON/'+DR+'/pickle_'+DR+'_3ccds/'+DR+'_3ccds_'+each_obs_date+'.pickle'

    if not glob.glob(file_in_name):

        print(each_obs_date+' - creating PICKLE')

        try:

            file_in = open(file_in_name,'w')


            localFilePath = '/shared-storage/buder/svn-repos/trunk/GALAH/SPECTRA/'+DR+'/'+each_obs_date+'/standard/com/'
            os.chdir(localFilePath)
            obs_date_files_01 = glob.glob('*1.fits')
            try:
                localFilePath = '/shared-storage/buder/svn-repos/trunk/GALAH/SPECTRA/'+DR+'/'+each_obs_date+'/standard/com2/'
                os.chdir(localFilePath)
                obs_date_files_02 = glob.glob('*1.fits')
            except:
                obs_date_files_02 = []
            localFilePath = '/shared-storage/buder/svn-repos/trunk/GALAH/'
            os.chdir(localFilePath)

            # Get rid of '1.fits' end to be able to match with SOBJECT_ID
            obs_date_fits_01  = np.array(map(lambda each: obs_date_files_01[each][0:15],range(len(obs_date_files_01))))
            obs_date_fits_01.sort()
            try:
                obs_date_fits_02  = np.array(map(lambda each: obs_date_files_02[each][0:15],range(len(obs_date_files_02))))
                obs_date_fits_02.sort()
            except:
                obs_date_fits_02 = []

            # Now we have to pull the information for each FITS form the IRAF table

            '''
            Start the main part of the code here: creating PICKLE files for each OBS_DATE
            '''

            # For EACH_FITS of OBS_DATE_FITS, collect GALAH_ID and flux

            galah_fits    = []
            galah_id      = []

            cannon_wave   = []
            cannon_flux   = []
            cannon_eflux  = []

            fits_in_iraf    = []

            '''
            Start the main iteration (EACH_FITS) for each OBS_DATE, i.e. the input for PICKLE
            '''

            for each_fits in obs_date_fits_01:

                # We can of course only use the spectrum, if the 4th extension is available
                try:

                    # Cross-match FITS name with IRAF SOBJECT_ID
                    fits_in_iraf = np.where(int(each_fits) == iraf['sobject_id'])[0]

                    # Forced crash if FITS not found
                    if not fits_in_iraf:
                        sys.exit('The FITS '+each_fits+' is not in IRAF '+DR)
                    else:
                        fits_in_iraf = fits_in_iraf[0]

                    # Pull VRAD and V_BARY from IRAF or other source

                    if DR=='dr5.2':
                        vrad = iraf['rv_guess_shift'][fits_in_iraf]
                        v_bary = iraf['v_bary'][fits_in_iraf]
                        #print(iraf['red_flag'][fits_in_iraf])
                    if DR == 'dr51':
                        vrad = iraf['vrad'][fits_in_iraf]
                        combs=len(fits1[0].header['COMB*'])
                        bary_fits=pyfits.open('DATA/GALAH_vbary_09232016.fits')
                        v_bary=[]
                        for baries in range(0,combs):
                            bary_pos=np.where(bary_fits[1].data['out_name']==fits1[0].header['COMB'+str(baries)])[0]
                            if len(bary_pos)==1:
                                    v_bary.append(bary_fits[1].data['v_bary'][bary_pos[0]])
                            else:
                                    sys.exit('NO V_BARY ENTRY FOUND')
                        v_bary=np.mean(v_bary)

                    ''' IMPORT CCD1 '''

                    if complete_DR == True:
                        if DR=='dr5.2':
                                fits1 = pyfits.open("SPECTRA/dr5.2/"+each_fits[0:6]+"/standard/com/"+each_fits+"1.fits")
                        if DR=='dr5.1':
                                fits1 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"1.fits")
                    else:
                        fits1 = pyfits.open("SPECTRA/"+field+"/"+each_fits+"1.fits")

                    if telluric_correction == True:
                        telluric_fits = pyfits.open('DATA/telluric_noao_21k.fits')
                        wave_tel      = telluric_fits[1].data['wave']/(1.0+(vrad-v_bary)/clight)

                    if skyline_correction  == True:
                        sky_mask=pyfits.open('DATA/Skyspectrum_161105.fits')
                        wave_sky=sky_mask[1].data['wave']/(1.0+(vrad-v_bary)/clight)

                    ws=fits1[4].header["CRVAL1"]
                    inc=fits1[4].header["CDELT1"]
                    nax=fits1[4].header["NAXIS1"]
                    ref=fits1[4].header["CRPIX1"]
                    if ref == 0:
                        ref=1
                    x1raw=map(lambda x:((x-ref+1)*inc+ws),range(0,nax))

                    # save normalized flux to y1 and uncertainties to z1
                    if renormalise!=True:
                        # EITHER TAKE FITS-EXTENSION 4: NORMALIZED FLUX
                        y1raw=fits1[4].data[0:nax]
                        z1raw=fits1[4].data[0:nax]*fits1[1].data[0:nax]
                        y1=np.interp(x1,x1raw,y1raw)
                        z1=np.interp(x1,x1raw,z1raw)
                                    #z1=y1/np.sqrt(np.median(fits1[0].data)*fits[0].header["RO_GAIN"])
                    else:
                        # OR USE FITS-EXTENSION 0: REDUCED FLUX AND RENORMALIZE
                        y1raw=fits1[4].data[0:nax]
                        z1raw=fits1[4].data[0:nax]*fits1[1].data[0:nax]
                        y1=np.interp(x1,x1raw,y1raw)
                        z1=np.interp(x1,x1raw,z1raw)
                        #fit chebychev 2nd order polynomial to fits-extension 0 with continuum pixels estimated during prior training step
                        fit1 = np.polynomial.chebyshev.Chebyshev.fit(x=x1[cont1], y=y1[cont1], w=z1[cont1] , deg=3)
                        y1=y1/fit1(x1)
                        z1=y1/np.sqrt(np.median(fits1[0].data)*fits1[0].header["RO_GAIN"])

                    if telluric_correction == True:
                        telluric_interp=np.interp(x1,wave_tel,telluric_fits[1].data['flux'])
                        telluric_interp[np.logical_or(np.isnan(telluric_interp),telluric_interp<0.81)]=0.81
                        telluric_interp[telluric_interp>0.995]=1.0
                        z1 += (1./(telluric_interp*5.-4) - 1.)

                    if skyline_correction == True:
                        sky_interp=np.interp(x1,wave_sky,sky_mask[1].data['sky'])
                        z1 += large*sky_interp

                    y1[np.logical_or(x1<=x1raw[0],x1>=x1raw[-1])]=1.
                    z1[np.logical_or(x1<=x1raw[0],x1>=x1raw[-1])]=large
                    fits1.close()

                    ''' IMPORT CCD2 '''
                    if complete_DR == True:
                        if DR=='dr5.2':
                                fits2 = pyfits.open("SPECTRA/dr5.2/"+each_fits[0:6]+"/standard/com/"+each_fits+"2.fits")
                        if DR=='dr5.1':
                                fits2 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"2.fits")
                    else:
                        fits1 = pyfits.open("SPECTRA/"+field+"/"+each_fits+"2.fits")

                    ws=fits2[4].header["CRVAL1"]
                    inc=fits2[4].header["CDELT1"]
                    nax=fits2[4].header["NAXIS1"]
                    ref=fits2[4].header["CRPIX1"]
                    if ref == 0:
                        ref=1
                    x2raw=map(lambda x:((x-ref+1)*inc+ws),range(0,nax))
                            # save normalized flux to y2 and uncertainties to z2
                    if renormalise!=True:
                        # EITHER TAKE FITS-EXTENSION 4: NORMALIZED FLUX
                        y2raw=fits2[4].data[0:nax]
                        z2raw=fits2[4].data[0:nax]*fits2[1].data[0:nax]
                        y2=np.interp(x2,x2raw,y2raw)
                        z2=np.interp(x2,x2raw,z2raw)
                        #z2=y2/np.sqrt(np.median(fits2[0].data)*fits2[0].header["RO_GAIN"])
                    else:
                        # OR USE FITS-EXTENSION 4: REDUCED FLUX AND RENORMALIZE
                        y2raw=fits2[4].data[0:nax]
                        z2raw=fits2[4].data[0:nax]*fits2[1].data[0:nax]
                        y2=np.interp(x2,x2raw,y2raw)
                        z2=np.interp(x2,x2raw,z2raw)
                        #fit chebychev 2nd order polynomial to fits-extension 0 with continuum pixels estimated during prior training step
                        fit2 = np.polynomial.chebyshev.Chebyshev.fit(x=x2[cont2], y=y2[cont2], w=z2[cont2] , deg=2) # there could be weights included, but since we assume same S/N for GALAH, this would not be helpful
                        y2=y2/fit2(x2)
                        z2=y2/np.sqrt(np.median(fits2[0].data)*fits2[0].header["RO_GAIN"])

                    if telluric_correction == True:
                        telluric_interp=np.interp(x2,wave_tel,telluric_fits[1].data['flux'])
                        telluric_interp[np.logical_or(np.isnan(telluric_interp),telluric_interp<0.81)]=0.81
                        telluric_interp[telluric_interp>0.995]=1.0
                        z2 += (1./(telluric_interp*5.-4)-1.)

                    if skyline_correction == True:
                        sky_interp=np.interp(x2,wave_sky,sky_mask[1].data['sky'])
                        z2 += large*sky_interp

                    y2[np.logical_or(x2<=x2raw[0],x2>=x2raw[-1])]=1.
                    z2[np.logical_or(x2<=x2raw[0],x2>=x2raw[-1])]=large
                    fits2.close()

                    ''' IMPORT CCD3 '''
                    if complete_DR == True:
                        if DR=='dr5.2':
                                fits3 = pyfits.open("SPECTRA/dr5.2/"+each_fits[0:6]+"/standard/com/"+each_fits+"3.fits")
                        if DR=='dr5.1':
                                fits3 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"3.fits")
                    else:
                        fits1 = pyfits.open("SPECTRA/"+field+"/"+each_fits+"3.fits")

                    ws=fits3[4].header["CRVAL1"]
                    inc=fits3[4].header["CDELT1"]
                    nax=fits3[4].header["NAXIS1"] # taken fixed 4096 because of varying nax +-2
                    #nax=4096
                    ref=fits3[4].header["CRPIX1"]
                    if ref == 0:
                        ref=1
                    x3raw=map(lambda x:((x-ref+1)*inc+ws),range(0,nax))
                    # save normalized flux to y3 and uncertainties to z3
                    if renormalise!=True:
                        # EITHER TAKE FITS-EXTENSION 4: NORMALIZED FLUX
                        y3raw=fits3[4].data[0:nax]
                        z3raw=fits3[4].data[0:nax]*fits3[1].data[0:nax]
                        y3=np.interp(x3,x3raw,y3raw)
                        z3=np.interp(x3,x3raw,z3raw)
                        #z3=y3/np.sqrt(np.median(fits3[0].data)*fits3[0].header["RO_GAIN"])
                    else:
                        # OR USE FITS-EXTENSION 4: REDUCED FLUX AND RENORMALIZE
                        y3raw=fits3[4].data[0:nax]
                        y3=np.interp(x3,x3raw,y3raw)
                        #fit chebychev 2nd order polynomial to fits-extension 0 with continuum pixels estimated during prior training step
                        fit3 = np.polynomial.chebyshev.Chebyshev.fit(x=x3[pixlist3], y=y3[pixlist3] , deg=2) # there could be weights included, but since we assume same S/N for GALAH, this would not be helpful
                        y3=y3/fit3(x3)
                        z3=y3/np.sqrt(np.median(fits3[0].data)*fits3[0].header["RO_GAIN"])

                    if telluric_correction == True:
                        telluric_interp=np.interp(x3,wave_tel,telluric_fits[1].data['flux'])
                        telluric_interp[np.logical_or(np.isnan(telluric_interp),telluric_interp<0.81)]=0.81
                        telluric_interp[telluric_interp>0.995]=1.0
                        z3 += (1./(telluric_interp*5.-4) - 1.)

                    if skyline_correction == True:
                        sky_interp=np.interp(x3,wave_sky,sky_mask[1].data['sky'])
                        z3 += large*sky_interp

                    y3[np.logical_or(x3<=x3raw[0],x3>=x3raw[-1])]=1.
                    z3[np.logical_or(x3<=x3raw[0],x3>=x3raw[-1])]=large
                    fits3.close()

                    ''' IMPORT CCD4 '''
                    if include_ccd4 == True:
                        if complete_DR == True:
                            if DR=='dr5.2':
                                    fits4 = pyfits.open("SPECTRA/dr5.2/"+each_fits[0:6]+"/standard/com/"+each_fits+"4.fits")
                            if DR=='dr5.1':
                                    fits4 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"4.fits")
                        else:
                            fits1 = pyfits.open("SPECTRA/"+field+"/"+each_fits+"4.fits")

                        ws=fits4[4].header["CRVAL1"]
                        inc=fits4[4].header["CDELT1"]
                        nax=fits4[4].header["NAXIS1"] # taken fixed 4096 because of varying nax +-2
                        naxir=fits4[1].header["NAXIS1"]
                        #nax=4096
                        ref=fits4[4].header["CRPIX1"]
                        if ref == 0:
                            ref=1
                        # ir_cut is included, because of the low wavelength cut in fits-extension 4 (to get rid of H20 band < 7700)
                        #ir_cut=len(fits4[4].data)
                        x4raw=map(lambda x:((x-ref+1)*inc+ws),range(0,nax))

                        # save normalized flux to y4 and uncertainties to z4
                        if renormalise!=True:
                            # EITHER TAKE FITS-EXTENSION 4: NORMALIZED FLUX
                            y4raw=fits4[4].data[0:nax]
                            z4raw=fits4[4].data[0:nax]*fits4[1].data[naxir-nax:naxir]
                            y4=np.interp(x4,x4raw,y4raw)
                            z4=np.interp(x4,x4raw,z4raw)
                            #z4=y4/np.sqrt(np.median(fits4[0].data)*fits4[0].header["RO_GAIN"])
                        else:
                            # OR USE FITS-EXTENSION 4: REDUCED FLUX AND RENORMALIZE
                            y4raw=fits4[4].data[nax-ir_cut:nax]
                            y4s=np.interp(x4,x4raw,y4raw)
                            #fit chebychev 2nd order polynomial to fits-extension 0 with continuum pixels estimated during prior training step
                            fit4 = np.polynomial.chebyshev.Chebyshev.fit(x=x4[pixlist4], y=y4s[pixlist4] , deg=4) # there could be weights included, but since we assume same S/N for GALAH, this would not be helpful
                            y4=y4s/fit4(x4)
                            z4=y4/np.sqrt(np.median(fits4[0].data)*fits4[0].header["RO_GAIN"])

                        if telluric_correction == True:
                            telluric_interp=np.interp(x4,wave_tel,telluric_fits[1].data['flux'])
                            telluric_interp[np.logical_or(np.isnan(telluric_interp),telluric_interp<0.81)]=0.81
                            telluric_interp[telluric_interp>0.995]=1.0
                            z4 += (1./(telluric_interp*5.-4) -1.)

                        if skyline_correction == True:
                            sky_interp=np.interp(x4,wave_sky,sky_mask[1].data['sky'])
                            z4 += large*sky_interp

                        y4[np.logical_or(x4<=x4raw[0],x4>=x4raw[-1])]=1.
                        z4[np.logical_or(x4<=x4raw[0],x4>=x4raw[-1])]=large
                        fits4.close()

                    ''' COMBINE CCDs '''
                    if include_ccd4==True:
                        x = np.concatenate((x1,x2,x3,x4))
                        y = np.concatenate((y1,y2,y3,y4))
                        z = np.concatenate((z1,z2,z3,z4))

                        #print(x2raw[0],x4raw[0])

        #                 f,(ax1,ax2,ax3,ax4) = plt.subplots(4)    
        #                 ax1.fill_between(x1,y1-z1,y1+z1,alpha=0.5,facecolor='k',lw=0)
        #                 ax1.plot(x1,y1,'k',lw=1)
        #                 ax2.fill_between(x2,y2-z2,y2+z2,alpha=0.5,facecolor='k',lw=0)
        #                 ax2.plot(x2,y2,'k',lw=1)
        #                 ax3.fill_between(x3,y3-z3,y3+z3,alpha=0.5,facecolor='k',lw=0)
        #                 ax3.plot(x3,y3,'k',lw=1)
        #                 ax4.fill_between(x4,y4-z4,y4+z4,alpha=0.5,facecolor='k',lw=0)
        #                 ax4.plot(x4,y4,'k',lw=1)
        #                 ax1.set_ylim(0.05,1.55)
        #                 ax2.set_ylim(0.05,1.55)
        #                 ax3.set_ylim(0.05,1.55)
        #                 ax4.set_ylim(0.05,1.55)
        #                 #ax4.set_xlim(7820,7830)
        #                 plt.tight_layout()

                    else:
                        x = np.concatenate((x1,x2,x3))
                        y = np.concatenate((y1,y2,y3))
                        z = np.concatenate((z1,z2,z3))

                    bady = np.isnan(y)
                    badz = np.isnan(z)
                    y[bady] = 1.
                    z[badz] = large
                    bady = np.logical_or(y > 1.5,y <0.0)
                    y[bady] = 1.
                    z[bady] = large

                    galah_fits.append(each_fits)
                    galah_id.append(iraf['galah_id'][fits_in_iraf])
                    cannon_wave.append(x)
                    cannon_flux.append(y)
                    cannon_eflux.append(z)

                except:
                    print('   '+each_fits+' does not have a 4th extension, flag_guess = '+str(iraf['flag_guess'][fits_in_iraf]))

            print('   reading in - done 01')

            if len(obs_date_fits_02) > 0:
                for each_fits in obs_date_fits_02:

                    # We can of course only use the spectrum, if the 4th extension is available
                    try:

                        # Cross-match FITS name with IRAF SOBJECT_ID
                        fits_in_iraf = np.where(int(each_fits) == iraf['sobject_id'])[0]

                        # Forced crash if FITS not found
                        if not fits_in_iraf:
                            sys.exit('The FITS '+each_fits+' is not in IRAF '+DR)
                        else:
                            fits_in_iraf = fits_in_iraf[0]

                        # Pull VRAD and V_BARY from IRAF or other source

                        if DR=='dr5.2':
                            vrad = iraf['rv_guess_shift'][fits_in_iraf]
                            v_bary = iraf['v_bary'][fits_in_iraf]
                            #print(iraf['red_flag'][fits_in_iraf])
                        if DR == 'dr51':
                            vrad = iraf['vrad'][fits_in_iraf]
                            combs=len(fits1[0].header['COMB*'])
                            bary_fits=pyfits.open('DATA/GALAH_vbary_09232016.fits')
                            v_bary=[]
                            for baries in range(0,combs):
                                bary_pos=np.where(bary_fits[1].data['out_name']==fits1[0].header['COMB'+str(baries)])[0]
                                if len(bary_pos)==1:
                                        v_bary.append(bary_fits[1].data['v_bary'][bary_pos[0]])
                                else:
                                        sys.exit('NO V_BARY ENTRY FOUND')
                            v_bary=np.mean(v_bary)

                        ''' IMPORT CCD1 '''

                        if complete_DR == True:
                            if DR=='dr5.2':
                                    fits1 = pyfits.open("SPECTRA/dr5.2/"+each_fits[0:6]+"/standard/com2/"+each_fits+"1.fits")
                            if DR=='dr5.1':
                                    fits1 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"1.fits")
                        else:
                            fits1 = pyfits.open("SPECTRA/"+field+"/"+each_fits+"1.fits")

                        if telluric_correction == True:
                            telluric_fits = pyfits.open('DATA/telluric_noao_21k.fits')
                            wave_tel      = telluric_fits[1].data['wave']/(1.0+(vrad-v_bary)/clight)

                        if skyline_correction  == True:
                            sky_mask=pyfits.open('DATA/Skyspectrum_161105.fits')
                            wave_sky=sky_mask[1].data['wave']/(1.0+(vrad-v_bary)/clight)

                        ws=fits1[4].header["CRVAL1"]
                        inc=fits1[4].header["CDELT1"]
                        nax=fits1[4].header["NAXIS1"]
                        ref=fits1[4].header["CRPIX1"]
                        if ref == 0:
                            ref=1
                        x1raw=map(lambda x:((x-ref+1)*inc+ws),range(0,nax))

                        # save normalized flux to y1 and uncertainties to z1
                        if renormalise!=True:
                            # EITHER TAKE FITS-EXTENSION 4: NORMALIZED FLUX
                            y1raw=fits1[4].data[0:nax]
                            z1raw=fits1[4].data[0:nax]*fits1[1].data[0:nax]
                            y1=np.interp(x1,x1raw,y1raw)
                            z1=np.interp(x1,x1raw,z1raw)
                                        #z1=y1/np.sqrt(np.median(fits1[0].data)*fits[0].header["RO_GAIN"])
                        else:
                            # OR USE FITS-EXTENSION 0: REDUCED FLUX AND RENORMALIZE
                            y1raw=fits1[4].data[0:nax]
                            z1raw=fits1[4].data[0:nax]*fits1[1].data[0:nax]
                            y1=np.interp(x1,x1raw,y1raw)
                            z1=np.interp(x1,x1raw,z1raw)
                            #fit chebychev 2nd order polynomial to fits-extension 0 with continuum pixels estimated during prior training step
                            fit1 = np.polynomial.chebyshev.Chebyshev.fit(x=x1[cont1], y=y1[cont1], w=z1[cont1] , deg=3)
                            y1=y1/fit1(x1)
                            z1=y1/np.sqrt(np.median(fits1[0].data)*fits1[0].header["RO_GAIN"])

                        if telluric_correction == True:
                            telluric_interp=np.interp(x1,wave_tel,telluric_fits[1].data['flux'])
                            telluric_interp[np.logical_or(np.isnan(telluric_interp),telluric_interp<0.81)]=0.81
                            telluric_interp[telluric_interp>0.995]=1.0
                            z1 += (1./(telluric_interp*5.-4) - 1.)

                        if skyline_correction == True:
                            sky_interp=np.interp(x1,wave_sky,sky_mask[1].data['sky'])
                            z1 += large*sky_interp

                        y1[np.logical_or(x1<=x1raw[0],x1>=x1raw[-1])]=1.
                        z1[np.logical_or(x1<=x1raw[0],x1>=x1raw[-1])]=large
                        fits1.close()

                        ''' IMPORT CCD2 '''
                        if complete_DR == True:
                            if DR=='dr5.2':
                                    fits2 = pyfits.open("SPECTRA/dr5.2/"+each_fits[0:6]+"/standard/com2/"+each_fits+"2.fits")
                            if DR=='dr5.1':
                                    fits2 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"2.fits")
                        else:
                            fits1 = pyfits.open("SPECTRA/"+field+"/"+each_fits+"2.fits")

                        ws=fits2[4].header["CRVAL1"]
                        inc=fits2[4].header["CDELT1"]
                        nax=fits2[4].header["NAXIS1"]
                        ref=fits2[4].header["CRPIX1"]
                        if ref == 0:
                            ref=1
                        x2raw=map(lambda x:((x-ref+1)*inc+ws),range(0,nax))
                                # save normalized flux to y2 and uncertainties to z2
                        if renormalise!=True:
                            # EITHER TAKE FITS-EXTENSION 4: NORMALIZED FLUX
                            y2raw=fits2[4].data[0:nax]
                            z2raw=fits2[4].data[0:nax]*fits2[1].data[0:nax]
                            y2=np.interp(x2,x2raw,y2raw)
                            z2=np.interp(x2,x2raw,z2raw)
                            #z2=y2/np.sqrt(np.median(fits2[0].data)*fits2[0].header["RO_GAIN"])
                        else:
                            # OR USE FITS-EXTENSION 4: REDUCED FLUX AND RENORMALIZE
                            y2raw=fits2[4].data[0:nax]
                            z2raw=fits2[4].data[0:nax]*fits2[1].data[0:nax]
                            y2=np.interp(x2,x2raw,y2raw)
                            z2=np.interp(x2,x2raw,z2raw)
                            #fit chebychev 2nd order polynomial to fits-extension 0 with continuum pixels estimated during prior training step
                            fit2 = np.polynomial.chebyshev.Chebyshev.fit(x=x2[cont2], y=y2[cont2], w=z2[cont2] , deg=2) # there could be weights included, but since we assume same S/N for GALAH, this would not be helpful
                            y2=y2/fit2(x2)
                            z2=y2/np.sqrt(np.median(fits2[0].data)*fits2[0].header["RO_GAIN"])

                        if telluric_correction == True:
                            telluric_interp=np.interp(x2,wave_tel,telluric_fits[1].data['flux'])
                            telluric_interp[np.logical_or(np.isnan(telluric_interp),telluric_interp<0.81)]=0.81
                            telluric_interp[telluric_interp>0.995]=1.0
                            z2 += (1./(telluric_interp*5.-4)-1.)

                        if skyline_correction == True:
                            sky_interp=np.interp(x2,wave_sky,sky_mask[1].data['sky'])
                            z2 += large*sky_interp

                        y2[np.logical_or(x2<=x2raw[0],x2>=x2raw[-1])]=1.
                        z2[np.logical_or(x2<=x2raw[0],x2>=x2raw[-1])]=large
                        fits2.close()

                        ''' IMPORT CCD3 '''
                        if complete_DR == True:
                            if DR=='dr5.2':
                                    fits3 = pyfits.open("SPECTRA/dr5.2/"+each_fits[0:6]+"/standard/com2/"+each_fits+"3.fits")
                            if DR=='dr5.1':
                                    fits3 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"3.fits")
                        else:
                            fits1 = pyfits.open("SPECTRA/"+field+"/"+each_fits+"3.fits")

                        ws=fits3[4].header["CRVAL1"]
                        inc=fits3[4].header["CDELT1"]
                        nax=fits3[4].header["NAXIS1"] # taken fixed 4096 because of varying nax +-2
                        #nax=4096
                        ref=fits3[4].header["CRPIX1"]
                        if ref == 0:
                            ref=1
                        x3raw=map(lambda x:((x-ref+1)*inc+ws),range(0,nax))
                        # save normalized flux to y3 and uncertainties to z3
                        if renormalise!=True:
                            # EITHER TAKE FITS-EXTENSION 4: NORMALIZED FLUX
                            y3raw=fits3[4].data[0:nax]
                            z3raw=fits3[4].data[0:nax]*fits3[1].data[0:nax]
                            y3=np.interp(x3,x3raw,y3raw)
                            z3=np.interp(x3,x3raw,z3raw)
                            #z3=y3/np.sqrt(np.median(fits3[0].data)*fits3[0].header["RO_GAIN"])
                        else:
                            # OR USE FITS-EXTENSION 4: REDUCED FLUX AND RENORMALIZE
                            y3raw=fits3[4].data[0:nax]
                            y3=np.interp(x3,x3raw,y3raw)
                            #fit chebychev 2nd order polynomial to fits-extension 0 with continuum pixels estimated during prior training step
                            fit3 = np.polynomial.chebyshev.Chebyshev.fit(x=x3[pixlist3], y=y3[pixlist3] , deg=2) # there could be weights included, but since we assume same S/N for GALAH, this would not be helpful
                            y3=y3/fit3(x3)
                            z3=y3/np.sqrt(np.median(fits3[0].data)*fits3[0].header["RO_GAIN"])

                        if telluric_correction == True:
                            telluric_interp=np.interp(x3,wave_tel,telluric_fits[1].data['flux'])
                            telluric_interp[np.logical_or(np.isnan(telluric_interp),telluric_interp<0.81)]=0.81
                            telluric_interp[telluric_interp>0.995]=1.0
                            z3 += (1./(telluric_interp*5.-4) - 1.)

                        if skyline_correction == True:
                            sky_interp=np.interp(x3,wave_sky,sky_mask[1].data['sky'])
                            z3 += large*sky_interp

                        y3[np.logical_or(x3<=x3raw[0],x3>=x3raw[-1])]=1.
                        z3[np.logical_or(x3<=x3raw[0],x3>=x3raw[-1])]=large
                        fits3.close()

                        ''' IMPORT CCD4 '''
                        if include_ccd4 == True:
                            if complete_DR == True:
                                if DR=='dr5.2':
                                        fits4 = pyfits.open("SPECTRA/dr5.2/"+each_fits[0:6]+"/standard/com2/"+each_fits+"4.fits")
                                if DR=='dr5.1':
                                        fits4 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"4.fits")
                            else:
                                fits1 = pyfits.open("SPECTRA/"+field+"/"+each_fits+"4.fits")

                            ws=fits4[4].header["CRVAL1"]
                            inc=fits4[4].header["CDELT1"]
                            nax=fits4[4].header["NAXIS1"] # taken fixed 4096 because of varying nax +-2
                            naxir=fits4[1].header["NAXIS1"]
                            #nax=4096
                            ref=fits4[4].header["CRPIX1"]
                            if ref == 0:
                                ref=1
                            # ir_cut is included, because of the low wavelength cut in fits-extension 4 (to get rid of H20 band < 7700)
                            #ir_cut=len(fits4[4].data)
                            x4raw=map(lambda x:((x-ref+1)*inc+ws),range(0,nax))

                            # save normalized flux to y4 and uncertainties to z4
                            if renormalise!=True:
                                # EITHER TAKE FITS-EXTENSION 4: NORMALIZED FLUX
                                y4raw=fits4[4].data[0:nax]
                                z4raw=fits4[4].data[0:nax]*fits4[1].data[naxir-nax:naxir]
                                y4=np.interp(x4,x4raw,y4raw)
                                z4=np.interp(x4,x4raw,z4raw)
                                #z4=y4/np.sqrt(np.median(fits4[0].data)*fits4[0].header["RO_GAIN"])
                            else:
                                # OR USE FITS-EXTENSION 4: REDUCED FLUX AND RENORMALIZE
                                y4raw=fits4[4].data[nax-ir_cut:nax]
                                y4s=np.interp(x4,x4raw,y4raw)
                                #fit chebychev 2nd order polynomial to fits-extension 0 with continuum pixels estimated during prior training step
                                fit4 = np.polynomial.chebyshev.Chebyshev.fit(x=x4[pixlist4], y=y4s[pixlist4] , deg=4) # there could be weights included, but since we assume same S/N for GALAH, this would not be helpful
                                y4=y4s/fit4(x4)
                                z4=y4/np.sqrt(np.median(fits4[0].data)*fits4[0].header["RO_GAIN"])

                            if telluric_correction == True:
                                telluric_interp=np.interp(x4,wave_tel,telluric_fits[1].data['flux'])
                                telluric_interp[np.logical_or(np.isnan(telluric_interp),telluric_interp<0.81)]=0.81
                                telluric_interp[telluric_interp>0.995]=1.0
                                z4 += (1./(telluric_interp*5.-4) -1.)

                            if skyline_correction == True:
                                sky_interp=np.interp(x4,wave_sky,sky_mask[1].data['sky'])
                                z4 += large*sky_interp

                            y4[np.logical_or(x4<=x4raw[0],x4>=x4raw[-1])]=1.
                            z4[np.logical_or(x4<=x4raw[0],x4>=x4raw[-1])]=large
                            fits4.close()

                        ''' COMBINE CCDs '''
                        if include_ccd4==True:
                            x = np.concatenate((x1,x2,x3,x4))
                            y = np.concatenate((y1,y2,y3,y4))
                            z = np.concatenate((z1,z2,z3,z4))

                            #print(x2raw[0],x4raw[0])

            #                 f,(ax1,ax2,ax3,ax4) = plt.subplots(4)    
            #                 ax1.fill_between(x1,y1-z1,y1+z1,alpha=0.5,facecolor='k',lw=0)
            #                 ax1.plot(x1,y1,'k',lw=1)
            #                 ax2.fill_between(x2,y2-z2,y2+z2,alpha=0.5,facecolor='k',lw=0)
            #                 ax2.plot(x2,y2,'k',lw=1)
            #                 ax3.fill_between(x3,y3-z3,y3+z3,alpha=0.5,facecolor='k',lw=0)
            #                 ax3.plot(x3,y3,'k',lw=1)
            #                 ax4.fill_between(x4,y4-z4,y4+z4,alpha=0.5,facecolor='k',lw=0)
            #                 ax4.plot(x4,y4,'k',lw=1)
            #                 ax1.set_ylim(0.05,1.55)
            #                 ax2.set_ylim(0.05,1.55)
            #                 ax3.set_ylim(0.05,1.55)
            #                 ax4.set_ylim(0.05,1.55)
            #                 #ax4.set_xlim(7820,7830)
            #                 plt.tight_layout()

                        else:
                            x = np.concatenate((x1,x2,x3))
                            y = np.concatenate((y1,y2,y3))
                            z = np.concatenate((z1,z2,z3))

                        bady = np.isnan(y)
                        badz = np.isnan(z)
                        y[bady] = 1.
                        z[badz] = large
                        bady = np.logical_or(y > 1.5,y <0.0)
                        y[bady] = 1.
                        z[bady] = large

                        galah_fits.append(each_fits)
                        galah_id.append(iraf['galah_id'][fits_in_iraf])
                        cannon_wave.append(x)
                        cannon_flux.append(y)
                        cannon_eflux.append(z)

                    except:
                        print('   '+each_fits+' does not have a 4th extension, flag_guess = '+str(iraf['flag_guess'][fits_in_iraf]))

                print('   reading in - done 02')


            # this combines the data into a single array of these vectors
            galah_fits    = np.array(galah_fits)
            galah_id      = np.array(galah_id)
            cannon_wave   = np.array(cannon_wave)
            cannon_flux   = np.array(cannon_flux)
            cannon_eflux  = np.array(cannon_eflux)

            npix          = np.shape(cannon_wave[0])[0]

            dataall       = np.zeros((npix, len(galah_fits), 3))
            countit       = np.arange(0,len(cannon_flux),1)

            # populate the dataall array with the wavelength, flux and error
            for a,b,c,jj in zip(cannon_wave,cannon_flux,cannon_eflux, countit):
                dataall[:,jj,0] = a
                dataall[:,jj,1] = b
                dataall[:,jj,2] = c

            nstars = np.shape(dataall)[1]

            #  check for bad pixels and set these to 1 and their uncertainties to the value "large"
            for jj in range(nstars):
                bad = np.logical_or(dataall[:,jj,1] < 0.1, dataall[:,jj,1] > 1.5)
                dataall[bad,jj,1] = 1.
                dataall[bad,jj,2]  = large

            meds = np.median(dataall[:,:,1], axis =0)
            meds = np.array(meds)
            take1 = np.logical_and(meds > 0.8, meds < 1.1) # Upper limit was 1.1 before!

            print('   Pickle-shapes: ', np.shape(dataall[:,take1,:]),np.shape(galah_fits),np.shape(galah_id))
            pickle.dump((dataall[:,take1,:],galah_fits,galah_id),file_in)
            file_in.close()

            print('   Saving - done')

        except:
            print('Could not change Path to '+localFilePath)
    else:
        print('Does already exist - Skip')
print('Done')


# In[ ]:

import smtplib

sender = 'buder@mpia.de'
receivers = ['buder@mpia.de']

message = """From: Gemini2
To: buder@mpia.de
Subject: """+'Creating TEST data '+DR+""" run finished
"""

try:
    smtpObj = smtplib.SMTP('localhost')
    smtpObj.sendmail(sender, receivers, message)         
    print("Successfully sent email")
except SMTPException:
    print("Error: unable to send email")


# In[ ]:

# Convert IPYNB to PY

os.chdir('/shared-storage/buder/svn-repos/trunk/GALAH/TheGALAHCannon/')

convert_command = 'ipython nbconvert --to script Cannon_maketest.ipynb'
os.system(convert_command)

os.chdir('/shared-storage/buder/svn-repos/trunk/GALAH/')

