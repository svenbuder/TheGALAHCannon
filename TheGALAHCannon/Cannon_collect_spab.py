
# coding: utf-8

# # Cannon collecting routine for 1 SP Cannon and several AB Cannons

# In[20]:

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

# PARULA MAP
_parula_data = [
    [0.2081, 0.1663, 0.5292],
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
parula_r = ListedColormap(_parula_data[::-1], name='parula_r')
parula_0 = ListedColormap(_parula_data, name='parula_0')
parula_0.set_bad((1,1,1))


# In[79]:

if sys.argv[1] == '-f':
    print('You are running the Cannon in IPYNB not PY mode, using default values for output/DR/mode')
    output   = 'Cannon3.0.1'
    DR       = 'dr5.2'
    mode     = 'Li_O_Na_Mg_Al_Si_Ca_Ti'
    mode = mode.split('_')
    print(output,DR,mode)
else:
    print('You are running the Cannon in PY mode')
    output   = sys.argv[1]
    DR       = sys.argv[2]
    mode     = sys.argv[3]
    mode = mode.split('_')
    print(output,DR,mode)
    
subset      = '_SMEmasks' # '_SMEmasks_it1'
filteroff   = 0

os.chdir('/shared-storage/buder/svn-repos/trunk/GALAH/')


# In[80]:

possible_parameters = np.array(['Teff','Logg','Feh','Alpha_fe','Vmic','Vsini'])
possible_auxiliary  = np.array(['Ak','Ebv'])
possible_abundances = np.array(['Li','C','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu'])
possible_labels     = np.concatenate((possible_parameters,possible_auxiliary,possible_abundances))

# Define the 4 CCD grids for the Cannon leaving at least 20 km/s to ab lines
x1=np.arange(4715.94,4896.00,0.046) # ab lines 4716.3 - 4892.3
x2=np.arange(5650.06,5868.25,0.055) # ab lines 5646.0 - 5867.8
x3=np.arange(6480.52,6733.92,0.064) # ab lines 6481.6 - 6733.4
x4=np.arange(7693.50,7875.55,0.074) # ab lines 7691.2 - 7838.5


'''
Here we create the complete 'GALAH' dictionary structure which is later saved to WG4 FITS file ordered by 'save_order'
'''

galah = {}


# In[81]:

model_name_ab = {}
model_name_sp = 'CANNON/'+DR+'/'+output+'/'+output+'_Sp'+subset+'_model.pickle'
for each_mode in mode:
    model_name_ab[each_mode] = 'CANNON/'+DR+'/'+output+'/'+output+'_'+each_mode+subset+'_model.pickle'
    
selftest_ab = {}
selftest_sp = 'CANNON/'+DR+'/'+output+'/'+output+'_Sp'+subset+'_selftest_tags.pickle'
for each_mode in mode:
    selftest_ab[each_mode] = 'CANNON/'+DR+'/'+output+'/'+output+'_'+each_mode+subset+'_selftest_tags.pickle'
    
print(model_name_ab.keys())
print(selftest_ab.keys())


# In[82]:

spectra  = {}
labels   = {}
coeffs   = {}
params   = {}
e_params = {}
ids      = {}
used_obs = {}

door = open(model_name_sp,'r')
dataall_sp, metaall_sp, labels_sp, offsets_sp, coeffs_sp, covs_sp, scatters_sp,chis_sp,chisqs_sp = pickle.load(door)
door.close

door = open(selftest_sp,'r')
params_sp, e_params_sp, covs_sp,chi2_def_sp,ids_sp,chi2_good_sp,chi2_each_sp = pickle.load(door)
door.close

spectra['Sp']  = dataall_sp
labels['Sp']   = labels_sp
coeffs['Sp']   = coeffs_sp
params['Sp']   = params_sp
e_params['Sp'] = e_params_sp
ids['Sp']      = ids_sp

for each_mode in model_name_ab.keys():
    
    used_obs[each_mode] = np.loadtxt('CANNON/'+DR+'/'+output+'/element_runs/'+each_mode+'_used.txt',dtype=int)

    door = open(model_name_ab[each_mode],'r')
    dataall_ab, metaall_ab, labels_ab, offsets_ab, coeffs_ab, covs_ab, scatters_ab,chis_sp,chisqs_ab = pickle.load(door)
    door.close
    
    door = open(selftest_ab[each_mode],'r')
    params_ab, e_params_ab, covs_ab,chi2_def_ab,ids_ab,chi2_good_ab,chi2_each_ab = pickle.load(door)
    door.close

    labels[each_mode]   = labels_ab
    coeffs[each_mode]   = coeffs_ab
    params[each_mode]   = params_ab
    e_params[each_mode] = e_params_ab
    ids[each_mode]      = ids_ab

print('Got all model/selftest data')


# In[83]:

kwargs = dict(cmap=parula,lw=0.5,alpha=0.5,vmin=-2.,vmax=0.5)

for each_mode in model_name_ab.keys():
    f,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
    s1=ax1.scatter(params['Sp'][used_obs[each_mode],0],params['Sp'][used_obs[each_mode],1],c=params['Sp'][used_obs[each_mode],2],label='Sp output == '+each_mode+' input',**kwargs)
    s2=ax2.scatter(params[each_mode][:,0],params[each_mode][:,1],c=params[each_mode][:,2],label=each_mode+' output',**kwargs)
    #ax3.scatter(params[each_mode][:,2],params[each_mode][:,-1],c=params[each_mode][:,0],label=each_mode)
    ax1.set_xlim(8000,3000)
    ax1.set_ylim(6,0)
    ax2.set_xlim(8000,3000)
    ax2.set_ylim(6,0)
    ax1.legend(loc=3)
    ax2.legend(loc=3)
    ax1.set_xlabel(r'$T_\mathrm{eff}$')
    ax2.set_xlabel(r'$T_\mathrm{eff}$')
    ax1.set_ylabel(r'$\log g$')
    ax1.set_ylabel(r'$\log g$')
    c1 = plt.colorbar(s1,ax=ax1)
    c2 = plt.colorbar(s2,ax=ax2)
    c1.set_label('Fe/H')
    c2.set_label('Fe/H')
    plt.tight_layout()


# In[ ]:

outlier = {}
outlier['Ebv'] = []
outlier['Ak']  = []

outlier_sigma = 3.

for each_mode in model_name_ab.keys():
    
    outliers_unique = []

    plt.figure(figsize = (10,10))
    # Check 3sig outliers and plot them in red on top of the other stars contained in sp and ab run
    for label_i,each_label in enumerate(labels_sp):
        if each_label:# in ['Teff','Logg','Feh']:
            ax = plt.subplot(len(labels_sp),1,label_i+1)
            sp_out,ab_out = (params['Sp'][used_obs[each_mode],label_i],params[each_mode][:,label_i])
            outlier[each_label] = np.where(abs(sp_out-ab_out) > outlier_sigma*np.nanstd(sp_out-ab_out))[0]
            #print(each_mode,each_label)
            #print(outlier[each_label])
            #print(ids['Sp'][used_obs[each_mode]][outlier[each_label]])
            outliers_unique.append(outlier[each_label])
            ax.plot(sp_out,sp_out-ab_out,'ko')
            ax.plot(sp_out[outlier[each_label]],(sp_out-ab_out)[outlier[each_label]],'ro')
            for each_outlier in outlier[each_label]:
                ax.text(sp_out[each_outlier],(sp_out-ab_out)[each_outlier],str(ids['Sp'][used_obs[each_mode]][each_outlier]),fontsize=10)
            ax.set_xlabel('Sp result for '+each_label)
            ax.set_ylabel('Sp-'+each_mode+' '+each_label)
    plt.tight_layout()
    #plt.close()
    
    # For each label, plot the spectra of 3sig outliers
    for label_i,each_label in enumerate(labels_sp):
        if each_label:# in ['Teff','Logg','Feh']:
            outlier_in_sp = used_obs[each_mode][outlier[each_label]]

            f,(ax1,ax2,ax3,ax4) = plt.subplots(4,figsize=(15,10))

            kwargs = dict(lw=0.5,alpha=0.5)

            for each_outlier in outlier_in_sp:
                wave_outlier  = spectra['Sp'][:,each_outlier,0]
                sob_outlier   = spectra['Sp'][:,each_outlier,1]
                uob_outlier   = spectra['Sp'][:,each_outlier,2]

                ax1.plot(wave_outlier,sob_outlier,label=str(each_outlier)+': '+str(ids['Sp'][each_outlier]),**kwargs)
                ax2.plot(wave_outlier,sob_outlier,**kwargs)
                ax3.plot(wave_outlier,sob_outlier,**kwargs)
                ax4.plot(wave_outlier,sob_outlier,**kwargs)
            ax1.set_xlim(x1[0],x1[-1])
            ax2.set_xlim(x2[0],x2[-1])
            ax3.set_xlim(x3[0],x3[-1])
            ax4.set_xlim(x4[0],x4[-1])
            ax1.legend(loc=0)
            ax1.set_title('Outlier spectra for label '+each_label,fontsize=30)
            plt.tight_layout()

    outliers_unique = np.unique(np.concatenate((outliers_unique)))
    outlier[each_mode] = outliers_unique
    #print(outliers_unique)
    #print(ids['Sp'][used_obs[each_mode]][outliers_unique])


# In[68]:

sp_coeff      = np.array([0, 1,2,3,4,5,6, 7,8, 9,10,11,12, 13,14,15,16,17, 18,19,20,21, 22,23,24, 25,26, 27   ])
ab_coeff      = np.array([0, 1,2,3,4,5,6, 8,9,10,11,12,13, 15,16,17,18,19, 21,22,23,24, 26,27,28, 30,31, 33   ])
ab_coeff_only = np.array([             7,              14,             20,          25,       29,    32, 34,35])


# In[69]:

kwargs = dict(lw=0.5,alpha=0.5)

for each_order in range(len(sp_coeff)):
    plt.figure()
    try:
        plt.plot(coeffs['Sp'][:,sp_coeff[each_order]],label='Sp',**kwargs)
    except:
        pass
    for each_mode in model_name_ab.keys():
        plt.plot(coeffs[each_mode][:,ab_coeff[each_order]],label=each_mode,**kwargs)

for each_order in range(len(ab_coeff_only)):
    plt.figure()
    for each_mode in model_name_ab.keys():
        plt.plot(coeffs[each_mode][:,ab_coeff_only[each_order]],label=each_mode,**kwargs)


# In[ ]:



