
# coding: utf-8

# # The Cannon TRAIN

# ## preamble

# In[ ]:

import numpy as np
import glob
from scipy import optimize as opt
import pickle 
import os
import sys


# ## Adjust OUTPUT/DR/OBS_DATE

# In[ ]:

if sys.argv[1] == '-f':
    print('You are running the Cannon in IPYNB not PY mode, using default values for output/DR/obs_date')
    output   = 'Cannon3.0.1'
    DR       = 'dr5.2'
    mode     = 'O'
    obs_date = ''
    print(output,DR,mode,obs_date)
else:
    print('You are running the Cannon in PY mode')
    output   = sys.argv[1]
    DR       = sys.argv[2]
    mode     = sys.argv[3]
    try:
        obs_date = sys.argv[4]
        print(output,DR,mode,obs_date)
    except:
        # No obs_date chosen, i.e. do the actual training step
        obs_date = ''
        print(output,DR,mode,obs_date)


# ## Checking for previous runs

# In[ ]:

ccds     = '4'
subset   = '_SMEmasks'

mode_in = '_'+mode

LARGE  = 2000.

os.chdir('/shared-storage/buder/svn-repos/trunk/GALAH/')

check_iterations = glob.glob('CANNON/'+DR+'/'+output+'/'+output+mode_in+subset+'_it*_training_data.pickle')
print(check_iterations)
if len(check_iterations) > 0:
    subset = '_SMEmasks_it'+str(len(check_iterations))

print(subset)

LARGE  = 2000.

trainingset  = 'CANNON/'+DR+'/'+output+'/'+output+mode_in+subset+'_training_data.pickle' 
use_model    = 'CANNON/'+DR+'/'+output+'/'+output+mode_in+subset+'_model.pickle'



# ## Cannon definitions

# In[ ]:

def nonlinear_invert(f, sigmas, coeffs, scatters,labels): 
    xdata = np.vstack([coeffs])
    sigmavals = np.sqrt(sigmas ** 2 + scatters ** 2) 
    guessit = [0]*len(labels)
    try: 
        model, cov = opt.curve_fit(_func, xdata, f, sigma = sigmavals, maxfev=18000, p0 = guessit)
    except RuntimeError:
        model = [999]*len(labels)
        cov = np.ones((len(labels),len(labels) ))
    return model, cov

def _func(coeffs, *labels):
    """ Takes the dot product of coefficients vec & labels vector 
    
    Parameters
    ----------
    coeffs: numpy ndarray
        the coefficients on each element of the label vector
    *labels: numpy ndarray
        label vector
    Returns
    -------
    dot product of coeffs vec and labels vec
    """
    nlabels = len(labels)
    linear_terms = labels
    quadratic_terms = np.outer(linear_terms, linear_terms)[np.triu_indices(nlabels)]
    lvec = np.hstack((linear_terms, quadratic_terms))
    return np.dot(coeffs[:,1:], lvec)


def infer_labels_nonlinear(fn_pickle,testdata, ids, fout_pickle, weak_lower,weak_upper):
    file_in = open(fn_pickle, 'r') 
    dataall, metaall, labels, offsets, coeffs, covs, scatters,chis,chisq = pickle.load(file_in)
    file_in.close()
    nstars = (testdata.shape)[1]
    nlabels = len(labels)
    print '......Infering labels for '+str(np.shape(testdata))+' shaped data of '+str(nstars)+' stars with '+str(nlabels)+' labels'
    Params_all = np.zeros((nstars, nlabels))
    MCM_rotate_all = np.zeros((nstars, np.shape(coeffs)[1]-1, np.shape(coeffs)[1]-1))
    covs_all = np.zeros((nstars,nlabels, nlabels))
    errs_all = np.zeros((nstars,nlabels))
    for jj in range(0,nstars):
        if np.any(abs(testdata[:,jj,0] - dataall[:, 0, 0]) > 0.0001): 
            print testdata[range(5),jj,0], dataall[range(5),0,0]
            assert False
        xdata = testdata[:,jj,0]
        ydata = testdata[:,jj,1]
        ysigma = testdata[:,jj,2]
        ydata_norm = ydata  - coeffs[:,0] # subtract the mean 
        f = ydata_norm 
        Cinv = 1. / (ysigma ** 2 + scatters ** 2)
        Params,covs = nonlinear_invert(f, 1/Cinv**0.5 ,coeffs, scatters,labels) 
        Params = Params+offsets
        num_cut = -1*(np.shape(coeffs)[-1] -1) 
        coeffs_slice = coeffs[:,num_cut:]
        MCM_rotate = np.dot(coeffs_slice.T, Cinv[:,None] * coeffs_slice)
        Params_all[jj,:] = Params 
        MCM_rotate_all[jj,:,:] = MCM_rotate 
        covs_all[jj,:,:] = covs
        errs_all[jj,:] = covs.diagonal()
    filein = fout_pickle.split('_tags') [0] 
    if np.logical_or(filein == 'CANNON/'+DR+'/'+output+'/'+output+mode_in+subset+'_selftest', filein[0:8] == 'selftest'):
        print '......Confirmed: Selftest'
        file_in = open(fout_pickle, 'w')  
        file_normed = trainingset.split('.pickle')[0]
        chi2,chi2_good,chi2_each = get_goodness_fit(fn_pickle, file_normed, Params_all )
        chi2_def = chi2 # this will be the not reduced chi2 #  /(len(xdata)*1. - 3.) #7209. #8575-363-3 # len(xdata)*1.
        #good_pixels = map( lambda goodi: len(np.where(testdata[:,goodi,2]<=1.)[0]),range(0,len(testdata[0,:,2])))      
        #chi2_def2 = chi2_def/good_pixels
        pickle.dump((Params_all, errs_all, covs_all,chi2_def,ids,chi2_good,chi2_each),  file_in)
        file_in.close()
    else:
        file_in = open(fout_pickle, 'w')
        file_normed = testdataname.split('.pickle')[0]
        chi2,chi2_good,chi2_each = get_goodness_fit(fn_pickle, file_normed, Params_all )
        chi2_def = chi2 # this will be the not reduced chi2 #  /(len(xdata)*1. - 3.) #7209. #8575-363-3 # len(xdata)*1.
        #good_pixels = map( lambda goodi: len(np.where(testdata[:,goodi,2]<=1.)[0]),range(0,len(testdata[0,:,2])))
        #print good_pixels
        #print chi2_def
        #chi2_def2 = chi2_def/good_pixels
        #print chi2_def2
        pickle.dump((Params_all, errs_all, covs_all,chi2_def,ids,chi2_good,chi2_each),  file_in)
        file_in.close()
    return Params_all , MCM_rotate_all

def get_goodness_fit(fn_pickle, filein, Params_all ):
    print '......Get goodness of git for '+str(fn_pickle)+' fielin: '+str(filein)+' with parameters '+str(np.shape(Params_all))
    fd = open(fn_pickle,'r')
    dataall, metaall, labels, offsets, coeffs, covs, scatters, chis, chisq = pickle.load(fd) 
    fd.close() 
    file_with_star_data = str(filein)+".pickle"
    #file_with_star_data = "testdata.pickle"
    file_normed = trainingset.split('.pickle')[0]
    if filein != file_normed: 
        #print 'filein != file_normed'
        f_flux= open(file_with_star_data, 'r') 
        flux,num,num2 = pickle.load(f_flux) 
    if filein == file_normed: 
        #print 'filein == file_normed'
        f_flux = open(trainingset, 'r') 
        flux, filterall, metaall, labelthings,  cluster_name, ids = pickle.load(f_flux)
    f_flux.close() 
    labels = Params_all 
    nlabels = np.shape(labels)[1]
    nstars = np.shape(labels)[0]
    features_data = np.ones((nstars, 1))
    offsets = np.mean(labels, axis = 0) 
    features_data = np.hstack((features_data, labels - offsets)) 
    newfeatures_data = np.array([np.outer(m, m)[np.triu_indices(nlabels)] for m in (labels - offsets)])
    features_data = np.hstack((features_data, newfeatures_data)) 
    chi2_all = np.zeros(nstars) 
    chi2_all_good = np.zeros(nstars)
    chi2_all_each = np.zeros((nstars,len(flux[:,0,2])))
    for jj in range(nstars):
        model_gen = np.dot(coeffs,features_data.T[:,jj]) 
        data_star = flux[:,jj,1] 
        Cinv = 1. / (flux[:,jj, 2] ** 2 + scatters ** 2)  # invvar slice of data
        chi2_each = (Cinv) * (data_star - np.dot(coeffs, features_data.T[:,jj]))**2
        chi2 = sum(chi2_each)
        chi2_good = chi2/len(np.where(np.array(flux[:,jj,2])<=1.)[0])
        chi2_all[jj] = chi2
        chi2_all_good[jj] = chi2_good
        chi2_all_each[jj,:] = chi2_each
    return(chi2_all,chi2_all_good,chi2_all_each)

def do_regressions(dataall, filterall, features):
    """
    """
    nlam, nobj, ndata = dataall.shape
    nobj, npred = features.shape
    featuresall = np.zeros((nlam,nobj,npred))
    featuresall[:, :, :] = features[None, :, :]
    return map(do_one_regression, dataall, filterall, featuresall)

def do_one_regression_at_fixed_scatter(data, filter1, features, scatter):
    """
    Parameters
    ----------
    data: ndarray, [nobjs, 3]
        wavelengths, fluxes, invvars
    meta: ndarray, [nobjs, nmeta]
        Teff, Feh, etc, etc
    scatter:
    Returns
    -------
    coeff: ndarray
        coefficients of the fit
    MTCinvM: ndarray
        inverse covariance matrix for fit coefficients
    chi: float
        chi-squared at best fit
    logdet_Cinv: float
        inverse of the log determinant of the cov matrice
        :math:`\sum(\log(Cinv))`
    use the same terminology as in the paper 
    """
    #scatter = kwargs.get('scatter', 0)
    nmeta = filter1.shape[0]
    nobjs, npars = features.shape
    assert npars == nmeta * (nmeta + 3) / 2 + 1
    #print filter1.shape,nmeta, features.shape
    filter_features = [np.hstack((1, filter1)) ]
    filter1 = np.array([filter1])
    # the way the filters is done is repeating code that exists in train()
    filter_newfeatures = np.array([np.outer(m, m)[np.triu_indices(nmeta)] for m in filter1])
    filter_features = np.hstack((filter_features, filter_newfeatures))[0]
    filter_features = np.array(filter_features)
    filter_features_bool = filter_features.astype(bool)
    assert np.shape(filter_features_bool)[0] == npars
    D = np.sum(filter1)
    assert np.sum(filter_features_bool) == D * (D + 3) / 2 + 1
    ### make the filter above
    # least square fit
    Cinv = 1. / (data[:, 2] ** 2 + scatter ** 2)  # invvar slice of data
    M = features[:,filter_features_bool]
    MTCinvM = np.dot(M.T, Cinv[:, None] * M) # craziness b/c Cinv isnt a matrix
    x = data[:, 1] # intensity slice of data
    MTCinvx = np.dot(M.T, Cinv * x)
    coeff_full = np.zeros(len(filter_features))
    coeff_ind = np.arange(len(filter_features))[filter_features_bool]
    try:
        coeff = np.linalg.solve(MTCinvM, MTCinvx)
    except np.linalg.linalg.LinAlgError:
        print MTCinvM, MTCinvx, data[:,0], data[:,1], data[:,2]
        print features
    if not np.all(np.isfinite(coeff)):
        print "coefficients not finite"
        print coeff, median(data[:,2]), data.shape , scatter
        assert False
    for a,b in zip(coeff_ind ,coeff):
        coeff_full[a] = b
    chi = np.sqrt(Cinv) * (x - np.dot(M, coeff))
    logdet_Cinv = np.sum(np.log(Cinv))
    return (coeff_full, MTCinvM, chi, logdet_Cinv )

def do_one_regression(data, filter1, metadata):
    """
    does a regression at a single wavelength to fit calling the fixed scatter routine
    # inputs:
    """
    ln_s_values = np.arange(np.log(0.01), 0., 10.4)
    chis_eval = np.zeros_like(ln_s_values)
    for ii, ln_s in enumerate(ln_s_values):
        foo, bar, chi, logdet_Cinv = do_one_regression_at_fixed_scatter(data, filter1, metadata, scatter = np.exp(ln_s))
        chis_eval[ii] = np.sum(chi * chi) - logdet_Cinv
    if np.any(np.isnan(chis_eval)):
        s_best = np.exp(ln_s_values[-1])
        return do_one_regression_at_fixed_scatter(data, filter1, metadata, scatter = s_best) + (s_best, )
    lowest = np.argmin(chis_eval)
    if lowest == 0 or lowest == len(ln_s_values)-1:
        s_best = np.exp(ln_s_values[lowest])
        return do_one_regression_at_fixed_scatter(data, filter1, metadata, scatter = s_best) + (s_best, )

def get_normalized_training_data_firstcall(testfile_in):
    if glob.glob(trainingset):
        file_in2 = open(trainingset, 'r')
        dataall, filterall, metaall, labels, name_take, name_galah_id= pickle.load(file_in2)
        file_in2.close()
        return dataall, filterall, metaall, labels, name_take, name_galah_id

def train(dataall, filterall, metaall, order, fn, cluster_name, logg_cut=100., teff_cut=0., leave_out=None):
    """
    - `leave out` must be in the correct form to be an input to `np.delete`
    """
    if leave_out is not None: #
        dataall = np.delete(dataall, [leave_out], axis = 1)
        metaall = np.delete(metaall, [leave_out], axis = 0)
    nstars, nmeta = metaall.shape
    offsets = np.mean(metaall, axis=0)
    features = np.ones((nstars, 1))
    if order >= 1:
        features = np.hstack((features, metaall - offsets))
    if order >= 2:
        newfeatures = np.array([np.outer(m, m)[np.triu_indices(nmeta)] for m in (metaall - offsets)])
        features = np.hstack((features, newfeatures))
    blob = do_regressions(dataall, filterall, features)
    coeffs = np.array([b[0] for b in blob])
    covs = np.array([np.linalg.inv(b[1]) for b in blob])
    chis = np.array([b[2] for b in blob])
    chisqs = np.array([np.dot(b[2],b[2]) - b[3] for b in blob]) # holy crap be careful
    scatters = np.array([b[4] for b in blob])
    fd = open(fn, "w")
    pickle.dump((dataall, metaall, labels, offsets, coeffs, covs, scatters,chis,chisqs), fd)
    fd.close()
    return


# ## Perform TRAIN if necessary

# In[ ]:

if obs_date == '':
    if not glob.glob(use_model):
        print('WILL PERFORM TRAINING')

        print('IMPORT TRAININGSET')
        dataall, filterall, metaall, labels, galah_name, galah_id = get_normalized_training_data_firstcall(output)

        print('START TRAINING')
        train(dataall, filterall, metaall, 2, use_model,  galah_name, logg_cut= 40.,teff_cut = 0.)

        print('START SELFVALIDATION')
        a = open(trainingset, 'r') 
        testdataall, filterall, metadata, labels, ids, galah_name = pickle.load(a) 
        filterall[:] = 1.
        a.close()
        fieldself = 'CANNON/'+DR+'/'+output+'/'+output+mode_in+subset+'_selftest'
        testmetaall, inv_covars = infer_labels_nonlinear(use_model, testdataall, ids, fieldself+"_tags.pickle",0.00,1.40) 
        dataall, filterall, metaall, labels, galah_name, galah_id = get_normalized_training_data_firstcall(output)

        print('START LEAVE 20% OUT')
        take_out=np.arange(len(galah_name))
        np.random.shuffle(take_out)
        steps=5
        sz=len(galah_name)/steps # for 20% take out
        parts=[]
        for steping in range(0,steps):
            if steping==0:
                parts.append(take_out[0:sz])
            else:
                if steping==steps-1:
                    parts.append(take_out[(steps-1)*sz:len(galah_name)])
                else:
                    parts.append(take_out[steping*sz:(steping+1)*sz])

        for each in range(0,steps):
            print('STARTING LEAVE_OUT TEST '+str(each+1)+'/'+str(steps))
            print('START TRAINING PART '+str(each+1)+'/'+str(steps))
            np.savetxt('CANNON/'+DR+'/'+output+'/leaveouttest/'+output+mode_in+subset+'_shuffled_order_'+str(each+1)+'.txt',parts[each],fmt='%s')
            train(dataall,filterall,metaall,2,'CANNON/'+DR+'/'+output+'/leaveouttest/'+output+mode_in+subset+'_out'+str(each+1)+'_model.pickle',galah_name[parts[each]],logg_cut=40.,teff_cut=0.,leave_out=parts[each])
            print('START TEST '+str(each+1)+'/'+str(steps))
            fieldself = 'CANNON/'+DR+'/'+output+'/leaveouttest/'+output+mode_in+subset+'_out'
            #dataall_part= np.delete(dataall, parts[each], axis = 1)
            testdataname=trainingset
            testmetaall, inv_covars = infer_labels_nonlinear('CANNON/'+DR+'/'+output+'/leaveouttest/'+output+mode_in+subset+'_out'+str(each+1)+'_model.pickle', dataall, galah_name, fieldself+str(each+1)+'.pickle',0.00,1.40)

        import email
        import email.mime.application
        from email.MIMEMultipart import MIMEMultipart
        from email.MIMEText import MIMEText
        from email.MIMEImage import MIMEImage
        msg = MIMEMultipart()

        msg['From'] = 'gemini2'
        msg['To'] = 'buder@mpia.de'
        msg['Subject'] = 'TRAIN finished for '+output+mode_in+' '+DR

    #     filename='Cannon1.3_SMEmasks_selftest_HRDs.pdf'
    #     fp=open('CANNON/'+DR+'/'+output+'/diagnostic_plots/'+filename,'rb')
    #     att = email.mime.application.MIMEApplication(fp.read(),_subtype="pdf")
    #     fp.close()
    #     att.add_header('Content-Disposition','attachment',filename=filename)
    #     msg.attach(att)

        import smtplib
        mailer = smtplib.SMTP('localhost')
        mailer.sendmail('gemini2', 'buder@mpia.de', msg.as_string())
        mailer.close()
        print('Email sent')
    else:
        print('TRAINING SET EXISTS')
else:
    print('USE EXISTING TRAINING SET')


# ## Do some quick diagnostics

# In[ ]:

# door = open(use_model, 'r')
# used_model = pickle.load(door)
# door.close()


# ## Perform TEST step

# In[ ]:

if obs_date != '':

    print('WORKING ON TESTSET PART '+str(obs_date))
    testdataname='CANNON/'+DR+'/pickle_'+DR+'_'+ccds+'ccds/'+obs_date
    saveas = 'CANNON/'+DR+'/'+output+'/pickle_test/'+output+mode_in+subset+'_'+obs_date[:-7]+'_tags.pickle'

    if not glob.glob(saveas):

        a=open(testdataname,'r')
        testdataall, ids, galah_name = pickle.load(a)
        a.close()
        print('START TEST PART '+output+' '+subset+' '+DR+' '+str(obs_date))
        testmetaall, inv_covars = infer_labels_nonlinear(use_model, testdataall, ids, saveas,0.00,1.40)
    else:
        print('TESTSET ALREADY DONE')


# In[7]:

# Convert IPYNB to PY

os.chdir('/shared-storage/buder/svn-repos/trunk/GALAH/CANNON/'+DR+'/'+output+'/')

convert_command = 'ipython nbconvert --to script Cannon_train.ipynb'
os.system(convert_command)

os.chdir('/shared-storage/buder/svn-repos/trunk/GALAH/')


# In[ ]:



