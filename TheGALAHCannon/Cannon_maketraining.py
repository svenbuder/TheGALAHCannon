
# coding: utf-8

# # Initial

# In[ ]:

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
import astropy.table as table
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

willi_blau = [0.0722666667, 0.4886666667, 0.8467]

# PARULA MAP
_parula_data = [[0.2081, 0.1663, 0.5292], 
                [0.2116238095, 0.1897809524, 0.5776761905], 
                [0.212252381, 0.2137714286, 0.6269714286], 
                [0.2081, 0.2386, 0.6770857143], 
                [0.1959047619, 0.2644571429, 0.7279], 
                [0.1707285714, 0.2919380952, 0.779247619], 
                [0.1252714286, 0.3242428571, 0.8302714286], 
                [0.0591333333, 0.3598333333, 0.8683333333], 
                [0.0116952381, 0.3875095238, 0.8819571429], 
                [0.0059571429, 0.4086142857, 0.8828428571], 
                [0.0165142857, 0.4266, 0.8786333333], 
                [0.032852381, 0.4430428571, 0.8719571429], 
                [0.0498142857, 0.4585714286, 0.8640571429], 
                [0.0629333333, 0.4736904762, 0.8554380952], 
                [0.0722666667, 0.4886666667, 0.8467], 
                [0.0779428571, 0.5039857143, 0.8383714286], 
                [0.079347619, 0.5200238095, 0.8311809524], 
                [0.0749428571, 0.5375428571, 0.8262714286], 
                [0.0640571429, 0.5569857143, 0.8239571429], 
                [0.0487714286, 0.5772238095, 0.8228285714], 
                [0.0343428571, 0.5965809524, 0.819852381], 
                [0.0265, 0.6137, 0.8135], 
                [0.0238904762, 0.6286619048, 0.8037619048], 
                [0.0230904762, 0.6417857143, 0.7912666667], 
                [0.0227714286, 0.6534857143, 0.7767571429], 
                [0.0266619048, 0.6641952381, 0.7607190476], 
                [0.0383714286, 0.6742714286, 0.743552381], 
                [0.0589714286, 0.6837571429, 0.7253857143], 
                [0.0843, 0.6928333333, 0.7061666667], 
                [0.1132952381, 0.7015, 0.6858571429], 
                [0.1452714286, 0.7097571429, 0.6646285714], 
                [0.1801333333, 0.7176571429, 0.6424333333], 
                [0.2178285714, 0.7250428571, 0.6192619048], 
                [0.2586428571, 0.7317142857, 0.5954285714], 
                [0.3021714286, 0.7376047619, 0.5711857143], 
                [0.3481666667, 0.7424333333, 0.5472666667], 
                [0.3952571429, 0.7459, 0.5244428571], 
                [0.4420095238, 0.7480809524, 0.5033142857], 
                [0.4871238095, 0.7490619048, 0.4839761905], 
                [0.5300285714, 0.7491142857, 0.4661142857], 
                [0.5708571429, 0.7485190476, 0.4493904762],
                [0.609852381, 0.7473142857, 0.4336857143], 
                [0.6473, 0.7456, 0.4188], 
                [0.6834190476, 0.7434761905, 0.4044333333], 
                [0.7184095238, 0.7411333333, 0.3904761905], 
                [0.7524857143, 0.7384, 0.3768142857], 
                [0.7858428571, 0.7355666667, 0.3632714286], 
                [0.8185047619, 0.7327333333, 0.3497904762], 
                [0.8506571429, 0.7299, 0.3360285714], 
                [0.8824333333, 0.7274333333, 0.3217], 
                [0.9139333333, 0.7257857143, 0.3062761905], 
                [0.9449571429, 0.7261142857, 0.2886428571], 
                [0.9738952381, 0.7313952381, 0.266647619], 
                [0.9937714286, 0.7454571429, 0.240347619], 
                [0.9990428571, 0.7653142857, 0.2164142857], 
                [0.9955333333, 0.7860571429, 0.196652381], 
                [0.988, 0.8066, 0.1793666667], 
                [0.9788571429, 0.8271428571, 0.1633142857], 
                [0.9697, 0.8481380952, 0.147452381], 
                [0.9625857143, 0.8705142857, 0.1309], 
                [0.9588714286, 0.8949, 0.1132428571], 
                [0.9598238095, 0.9218333333, 0.0948380952], 
                [0.9661, 0.9514428571, 0.0755333333], 
                [0.9763, 0.9831, 0.0538]]

from matplotlib.colors import ListedColormap

parula = ListedColormap(_parula_data, name='parula')
parula_zero = _parula_data[0]


# In[ ]:


# Change Work directory (if Sven's computer)
try:
    localFilePath = '/shared-storage/buder/svn-repos/trunk/GALAH/'
    os.chdir(localFilePath)
except:
    print('Could not change Path to '+localFilePath)

if sys.argv[1] == '-f':
    print('You are creating a trainingset for the Cannon in IPYNB not PY mode, using default values for output/DR/mode/')
    output   = 'Cannon3.0.1'
    DR       = 'dr5.2'
    mode     = 'Li'
    print(output,DR,mode)
else:
    print('You are running the Cannon in PY mode')
    output   = sys.argv[1]
    DR       = sys.argv[2]
    try:
        mode = sys.argv[3]
        print(output,DR,mode)
    except:
        # No obs_date chosen, i.e. do the actual training step
        mode = 'Sp'
        print(output,DR,mode)
    
# IRAF REDUCTION VERSION

#DR                  = 'dr5.2'  # default: 'dr5.2', this code is also compatible with 'dr5.1'
backup_DR_date      = '170523' # insert here only the last known date! By default, the code will try to use the latest
complete_DR         = True     # default: True, otherwise provide files in 'SPECTRA/FIELD/*.fits'
field               = ''       # default: not set, if complete_DR == False, set field name here

# ADDITIONAL CORRECTIONS BY WG4

telluric_correction = False #True
skyline_correction  = True  #True
renormalise         = False

# CANNON SPECIFICATIONS

include_ccd4        = True
subset              = '_SMEmasks'
mode_in             = '_'+mode
filteroff           = 0



# In[ ]:

iteration_outlier = []

iterations = glob.glob('CANNON/'+DR+'/'+output+'/'+output+mode_in+subset+'_model.pickle')

print(iterations)

if len(iterations) > 0:
    print('STARTING NEW ITERATION!')
    door=open('CANNON/'+DR+'/'+output+'/'+output+mode_in+subset+'_training_data.pickle')
    training_spectra,filters,training_label,labels,training_fits,training_galah_ids=pickle.load(door)
    door.close()
    try:
        door=open('CANNON/'+DR+'/'+output+'/'+output+mode_in+subset+'_selftest_tags.pickle')
        label_self, error_label_self,  covs_self, chi2_self, ids_self, chi2_good_self, chi2_each_self=pickle.load(door)
    except:
        door=open('CANNON/'+DR+'/'+output+'/'+output+mode_in+subset+'_selftest_tags.pickle')
        label_self, error_label_self,  covs_self, chi2_self, ids_self, chi2_good_self=pickle.load(door)
    door.close()
    
if len(iterations) > 0:
    
    if len(labels) == 6:
        f = plt.figure(figsize = (2*6.4, 2*4.8))
    elif len(labels) == 7:
        f = plt.figure(figsize = (8./3.*6.4, 3*4.8))
    for each in range(len(label_self[0])):
        bias = np.mean(training_label[:,each]-label_self[:,each])
        std  = np.std(training_label[:,each]-label_self[:,each])
        rms  = (np.sum([(xx-yy)**2 for xx,yy in zip(training_label[:,each],label_self[:,each])])/len(training_label[:,each]))**0.5
        sigma_outlier   = np.where(abs(training_label[:,each]-label_self[:,each]) > 2. * rms)[0]

        iteration_outlier.append(sigma_outlier)
        
        def ab_scatter(X, Y, ax=plt.gca, **kwargs):
            """
            This function gives back a scatter plot

            """

            c = kwargs.get('c',parula_zero)
            s = kwargs.get('s',2)
            s1 = ax.scatter(X,Y,c=c,s=s,alpha=0.5,rasterized=True,label=r'$\Delta$ iteration '+str(len(iterations)))

            return ax

        def ab_dens2d(X, Y, ax=plt.gca, min_per_bin=5, zeroslines=True, interimlines=True, colorbar=True, **kwargs):
            """
            This function gives back a 2D density plot 
            of the data put in as X and Y with 
            all points as scatter below certain density

            """

            #first make sure to only use finite X and Y values
            XY_finite = (np.isfinite(X) & np.isfinite(Y))
            X = X[XY_finite]
            Y = Y[XY_finite]

            # General kwargs:
            xlabel = kwargs.get('xlabel',labels[each]+' input-output')
            ylabel = kwargs.get('ylabel', r'$\Delta$~'+labels[each]+' input-output')
            xlim   = kwargs.get('xlim', (-3.0,0.65))
            ylim   = kwargs.get('ylim', (-0.5,1.00))
            cmap = kwargs.get('cmap', parula)
            bins = kwargs.get('bins', (0.05,0.025))
            if np.shape(bins) != ():
                # assuming input in dex
                bins = bins

            # plot all points as scatter before density structure is overlaid
            scatter = ab_scatter(X,Y,ax=ax)

            H, xedges, yedges = np.histogram2d(X,Y,bins=bins)
            H=np.rot90(H)
            H=np.flipud(H)
            Hmasked = np.ma.masked_where(H<min_per_bin,H)

            dens2d=ax.pcolormesh(xedges,yedges,Hmasked,cmap=cmap)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            xticks = kwargs.get('xticks',ax.get_xticks())
            ax.set_xticks(xticks)

            ax.axhline(0,c='k',lw=0.5)
            
            if colorbar == True:
                c = plt.colorbar(dens2d,ax=ax)
                c.set_label('Counts')

        if len(labels) == 6:
            ax = plt.subplot(3,2,each+1)
        elif len(labels) == 7:
            ax = plt.subplot(4,2,each+1)
            
        label_min = np.min([np.nanmin(training_label[:,each]),np.nanmin(label_self[:,each])])
        label_max = np.max([np.nanmax(training_label[:,each]),np.nanmax(label_self[:,each])])
        label_mima = label_max - label_min
        
        dlabel_min = np.nanmin(training_label[:,each]-label_self[:,each])
        dlabel_max = np.nanmax(training_label[:,each]-label_self[:,each])
        dlabel_dd  = np.max([abs(dlabel_min),abs(dlabel_max)])
        
        ab_dens2d(
            ax=ax,
            X = training_label[:,each],
            Y = training_label[:,each]-label_self[:,each],
            xlim = (label_min-0.05*label_mima,label_max+0.05*label_mima),
            ylim = (-1.05*dlabel_dd,1.05*dlabel_dd),
            bins = [
                np.arange(label_min-0.05*label_mima,label_max+0.05*label_mima+0.001,1.1*label_mima/40.),
                np.arange(-1.05*dlabel_dd,1.05*dlabel_dd+0.001,2.1*dlabel_dd/40.)
                ]
            )
        ax.text(0.025,0.95,'Bias: ',ha='left',va='center',transform=ax.transAxes)
        ax.text(0.025,0.90,'Std:  ',ha='left',va='center',transform=ax.transAxes)
        ax.text(0.025,0.85,'RMS:  ',ha='left',va='center',transform=ax.transAxes)
        ax.text(0.175,0.95,str(0.01*round(100*bias)),ha='left',va='center',transform=ax.transAxes)
        ax.text(0.175,0.90,str(0.01*round(100*std)),ha='left',va='center',transform=ax.transAxes)
        ax.text(0.175,0.85,str(0.01*round(100*rms)),ha='left',va='center',transform=ax.transAxes)
        si1 = ax.axhline(2.*rms,c='k',ls='dashed',label=r'$2\,\sigma~\mathrm{outlier}$')
        si2 = ax.axhline(-2.*rms,c='k',ls='dashed')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('CANNON/'+DR+'/'+output+'/diagnostic_plots/'+output+mode_in+subset+'_it1_outlier.pdf',dpi=600)

    iteration_outlier = np.unique(np.concatenate((iteration_outlier)))

    t = table.Table()
    ts_old = t.read('CANNON/'+DR+'/'+output+'/trainingset/'+output+mode_in+subset+'_trainingset.fits')
    print('ORIGINAL FITS HAS '+str(len(ts_old['sobject_id']))+' ENTRIES')

    subset = subset+'_it'+str(len(iterations))
    print('THIS IS ITERATION NR. '+str(len(iterations)))
    print('EXCLUDING '+str(len(iteration_outlier))+' SPECTRA ('+str("{:.1f}".format((100*len(iteration_outlier))/len(training_label)))+'\% OF '+str(len(training_label))+' SPECTRA)')

    np.savetxt('CANNON/'+DR+'/'+output+'/trainingset/'+output+mode_in+subset+'_leftout',iteration_outlier,fmt='%s')

    print(len(ts_old))
    ts_old.remove_rows(iteration_outlier)

    print(len(ts_old))

    print(str(len(ts_old['sobject_id']))+' STARS LEFT')
    ts_old.write('CANNON/'+DR+'/'+output+'/trainingset/'+output+mode_in+subset+'_trainingset.fits',overwrite=True)


# In[ ]:


include_parameters = np.array(['Teff','Logg','Feh','Vmic','Vsini']) # 'Alpha_fe',
include_auxiliary  = np.array(['Ak']) # ['Ebv']
include_abundances = np.array([])
if mode != 'Sp':
    include_abundances = np.array([mode])
#include_abundances = np.array(['O','Na','Mg','Si','Ca','Ti','Cr'])

labels = np.concatenate((include_parameters,include_auxiliary,include_abundances))

# Define the 4 CCD grids for the Cannon leaving at least 20 km/s to ab lines
x1=np.arange(4715.94,4896.00,0.046) # ab lines 4716.3 - 4892.3
x2=np.arange(5650.06,5868.25,0.055) # ab lines 5646.0 - 5867.8
x3=np.arange(6480.52,6733.92,0.064) # ab lines 6481.6 - 6733.4
x4=np.arange(7693.50,7875.55,0.074) # ab lines 7691.2 - 7838.5

include_ccd4=True#False
plot_spectra=False#True


# In[ ]:

########################################
#       IMPORT OF SOBJECT IRAF         #
########################################

# This file was intended for 'dr52' but is also compatible with irafdr51

if DR == 'dr5.2':
    versions = glob.glob('DATA/sobject_iraf_52_170926.fits')
else:
    versions = ['DATA/iraf_dr51_09232016_corrected.fits']

# Read in information from the IRAF FITS
door = pyfits.open(versions[-1])
iraf = door[1].data
door.close()

print(versions[-1]+' will be used.')
print('Available entries in IRAF FITS:  '+str(len(iraf['sobject_id'])))


# In[ ]:

print('Cannon version --- '+output+subset+' --- will be created')
print('GUESS-normalized spectra FITS[4] will be used')

# Initialize variables that will be filled later
flux_take = []
wavelx_take = []
npix = len(x1)+len(x2)+len(x3)
if include_ccd4 == True:
        npix += len(x4)

error_take = []
name_take = [] 
galah_id_take = []
usable = []

# speed of light and definition of large errors
clight = 299792.458 # speed of light in km/s
large = 100.

possible_parameters = np.array(['Teff','Logg','Feh','Alpha_fe','Vmic','Vsini'])
possible_auxiliary  = np.array(['Ak','Ebv'])
possible_abundances = np.array(['Li','C','O','Na','Mg','Al','Al6696','Al7835','Si','K','K5802','K7699','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Cu5700','Cu5782','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','Ba5854','Ba6497','La','Ce','Nd','Sm','Eu'])
possible_labels     = np.concatenate((possible_parameters,possible_auxiliary,possible_abundances))


# In[ ]:

# read in FITS-data and labels from each chosen field(s)
t = table.Table()

sme = t.read('CANNON/'+DR+'/'+output+'/trainingset/'+output+mode_in+subset+'_trainingset.fits')

spectratotake=np.arange(len(sme['sobject_id']))

name_fits=sme['sobject_id']
galah_id=sme['galah_id']
field_id=sme['field']
vrad=sme['rv_sme']

possible_metaall = np.ones((len(sme['Teff_sme']),),dtype={'names':possible_labels,'formats':[float for it in range(len(possible_labels))]})
possible_filters = np.ones((npix,),dtype={'names':possible_labels,'formats':[int for it in range(len(possible_labels))]})

for i in possible_parameters:
        possible_metaall[i] = sme[i+'_sme']
possible_metaall['Ebv'] = sme['ebv']
possible_metaall['Ak'] = sme['Ak']

for i in possible_abundances:
    possible_metaall[i] = sme[i+'_abund_sme']

if mode != 'Sp':    
    if filteroff == 0:
        if include_ccd4 == True:
            possible_filters[mode] = np.loadtxt('CANNON/'+DR+'/masks_4ccds/DR1_'+mode+'.txt',usecols=(5,),unpack=1)
        else:
            possible_filters[mode] = np.loadtxt('CANNON/'+DR+'/masks_3ccds/DR1_'+mode+'.txt',usecols=(5,),unpack=1)                        
    else:
        sys.exit(str(i)+' not in sme / not used')


# In[ ]:

snr = np.array(sme['snr2_c2_iraf'])

print('DONE reading in labels: '+str(len(name_fits))+' labels available')

snr_cut1=np.min(snr)
snr_cut2=np.max(snr)

low=[]
high=[]
high_fits = []
notfound=[]

for each in spectratotake:
        try:
            #print(each,name_fits[each])
            if DR == 'dr5.2':
                try:
                    fits1 = pyfits.open("SPECTRA/dr5.2/"+str(name_fits[each])[0:6]+"/standard/com/"+str(name_fits[each])+"1.fits")
                except:
                    fits1 = pyfits.open("SPECTRA/irafdr51/"+str(name_fits[each])[0:6]+"/combined/"+str(name_fits[each])+"1.fits")
            else:
                fits1 = pyfits.open("SPECTRA/irafdr51/"+str(name_fits[each])[0:6]+"/combined/"+str(name_fits[each])+"1.fits")
            if fits1[0].header['SLITMASK']=='OUT':
                low.append(each)
            else:
                high.append(each)
                high_fits.append(name_fits[each])
        except:
                notfound.append(name_fits[each])
        fits1.close()

        
spectratotake=np.array(low)

keep=[]
for i in range(0,len(spectratotake)):
    if name_fits[spectratotake[i]] not in iteration_outlier:
        keep.append(spectratotake[i])
spectratotake=np.array(keep)
print(str(len(spectratotake))+' will be used for trainingset')
print('Taken out:')
print('HighRes spectra:   '+str(len(high)))
print(list(high_fits))
print('Not found spectra: '+str(len(notfound)))
print('Iteration outlier: '+str(len(iteration_outlier)))

np.savetxt('CANNON/'+DR+'/'+output+'/element_runs/'+mode+subset+'_used.txt',np.array(spectratotake),fmt='%s')


# In[ ]:

for each in spectratotake:
    try:
        
        error = 0
        
        each_fits = str(name_fits[each])

        if np.where(spectratotake==each)[0]%100==0: 
            print(str(0.01*round(np.where(spectratotake==each)[0]*10000.00/len(spectratotake)))+' %')

        # Only necessary, if VRAD/VBARY needed

        if (telluric_correction == True) | (skyline_correction  == True):

            # Cross-match FITS name with IRAF SOBJECT_ID
            try:
                fits_in_iraf = np.where(int(each_fits) == iraf['sobject_id'])[0]
                fits_in_iraf = fits_in_iraf[0]
            except:
                sys.exit('The FITS '+each_fits+' is not in IRAF '+DR)

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

        error += 1 # if error == 1: no fits 1
        
        if complete_DR == True:
            if DR=='dr5.2':
                try:
                    fits1 = pyfits.open("SPECTRA/dr5.2/"+each_fits[0:6]+"/standard/com/"+each_fits+"1.fits")
                except:
                    fits1 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"1.fits")
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

        error += 1 # if error == 2: no fits extension
        ws=fits1[4].header["CRVAL1"]
        inc=fits1[4].header["CDELT1"]
        nax=fits1[4].header["NAXIS1"]
        ref=fits1[4].header["CRPIX1"]
        if ref == 0:
            ref=1
        x1raw=map(lambda x:((x-ref+1)*inc+ws),range(0,nax))

        error += 1 # if error == 3: something with extension 4 of ccd1

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

        error += 1 # if error == 4: something with extension 4 of ccd2

        ''' IMPORT CCD2 '''
        if complete_DR == True:
            if DR=='dr5.2':
                try:
                    fits2 = pyfits.open("SPECTRA/dr5.2/"+each_fits[0:6]+"/standard/com/"+each_fits+"2.fits")
                except:
                    fits2 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"2.fits")
            if DR=='dr5.1':
                    fits2 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"2.fits")
        else:
            fits2 = pyfits.open("SPECTRA/"+field+"/"+each_fits+"2.fits")

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

        error += 1 # if error == 5: something with extension 4 of ccd3

        ''' IMPORT CCD3 '''
        if complete_DR == True:
            if DR=='dr5.2':
                try:
                    fits3 = pyfits.open("SPECTRA/dr5.2/"+each_fits[0:6]+"/standard/com/"+each_fits+"3.fits")
                except:
                    fits3 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"3.fits")
            if DR=='dr5.1':
                    fits3 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"3.fits")
        else:
            fits3 = pyfits.open("SPECTRA/"+field+"/"+each_fits+"3.fits")

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

        error += 1 # if error == 6: something with extension 4 of ccd4

        ''' IMPORT CCD4 '''
        if include_ccd4 == True:
            if complete_DR == True:
                if DR=='dr5.2':
                    try:
                        fits4 = pyfits.open("SPECTRA/dr5.2/"+each_fits[0:6]+"/standard/com/"+each_fits+"4.fits")
                    except:
                        fits4 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"4.fits")
                if DR=='dr5.1':
                        fits4 = pyfits.open("SPECTRA/irafdr51/"+each_fits[0:6]+"/combined/"+each_fits+"4.fits")
            else:
                fits4 = pyfits.open("SPECTRA/"+field+"/"+each_fits+"4.fits")

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
            
        error += 1 # if error == 7: something with combining the data

        ''' COMBINE CCDs '''
        if include_ccd4==True:
            x = np.concatenate((x1,x2,x3,x4))
            y = np.concatenate((y1,y2,y3,y4))
            z = np.concatenate((z1,z2,z3,z4))
        else:
            x = np.concatenate((x1,x2,x3))
            y = np.concatenate((y1,y2,y3))
            z = np.concatenate((z1,z2,z3))

        bady = np.isnan(y)
        badz = np.isnan(z)
        y[bady] = 1.
        z[badz] = large
        bady = np.logical_or(y > 1.2,y <0.0)
        y[bady] = 1.
        z[bady] = large

        wavelx_take.append(x)
        flux_take.append(y)
        error_take.append(z)

        name_take.append(name_fits[each])
        galah_id_take.append(galah_id[each])

        usable.append(each)
        
    except:
        if error == 1:
            print(each,name_fits[each],' from field '+str(field_id[each])+' error: 1st extension')
        if error == 2:
            print(each,name_fits[each],' from field '+str(field_id[each])+' error: 4th extenstion (GUESS flag?)')
        if error == 3:
            print(each,name_fits[each],' from field '+str(field_id[each])+' error: CCD1')
        if error == 4:
            print(each,name_fits[each],' from field '+str(field_id[each])+' error: CCD2')
        if error == 5:
            print(each,name_fits[each],' from field '+str(field_id[each])+' error: CCD3')
        if error == 6:
            print(each,name_fits[each],' from field '+str(field_id[each])+' error: CCD4')
        if error == 7:
            print(each,name_fits[each],' from field '+str(field_id[each])+' error: can not combine')

print('DONE reading in spectra')

# this combines the data into a single array of these vectors
data = zip(name_take, wavelx_take, flux_take, error_take) 
error_take = np.array(error_take)
flux_take = np.array(flux_take)
name_take = np.array(name_take)

# we have three labels that we will train and solve for 
npix = np.shape(wavelx_take[0])[0]
nmeta = len(labels) 

# the dataall is the actual spectral data 
# the metaall is the labels for the stasr of teff, logg and [fe/H]
dataall = np.zeros((npix, len(name_take), 3))
filterall = np.ones((npix, nmeta))
metaall = np.ones((len(name_take), nmeta))
countit = np.arange(0,len(flux_take),1)
newwl = np.arange(0,len(wavelx_take),1) 

# populate the dataall array with the wavelength, flux and error
for a,b,c,jj in zip(wavelx_take, flux_take, error_take, countit):
    dataall[:,jj,0] = a
    dataall[:,jj,1] = b
    dataall[:,jj,2] = c
    
nstars = np.shape(dataall)[1]

#  check for bad pixels and set these to 1 and their uncertainties to the value "large"

for jj in range(nstars):
    bad = np.logical_or(dataall[:,jj,1] < 0.0, dataall[:,jj,1] > 1.2)
    dataall[bad,jj,1] = 1.
    dataall[bad,jj,2]  = large
    
# now we want to populate the metaall array with the labels 
for k in range(0,nmeta):
        metaall[:,k]   = possible_metaall[labels[k]][np.array(usable)]
        filterall[:,k] = possible_filters[labels[k]]

# This is the name of the array that we are going to save into 
file_in = open('CANNON/'+DR+'/'+output+'/'+output+mode_in+subset+'_training_data.pickle', 'w')
galah_id = np.array(galah_id) 
nstars = np.shape(dataall)[1]
meds = np.median(dataall[:,:,1], axis =0) 
meds = np.array(meds)
take1 = np.logical_and(meds > 0.8, meds < 1.1) # Upper limit was 1.1 before!

print('shapes of pickle-output: ', np.shape(dataall[:,take1,:]),np.shape(filterall),np.shape(metaall[take1,:]),np.shape(labels),np.shape(name_take),np.shape(galah_id_take))
pickle.dump((dataall[:,take1,:], filterall, metaall[take1,:], labels, name_take, galah_id_take),  file_in)
file_in.close() 

print('DONE saving trainingset in *pickle*')
print('Used labels :', end=' ')
print(labels)


# In[ ]:

# Convert IPYNB to PY

os.chdir('/shared-storage/buder/svn-repos/trunk/GALAH/CANNON/'+DR+'/'+output+'/')

convert_command = 'ipython nbconvert --to script Cannon_maketraining.ipynb'
os.system(convert_command)

os.chdir('/shared-storage/buder/svn-repos/trunk/GALAH/')

