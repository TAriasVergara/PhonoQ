# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 17:40:00 2020

@author: TOMAS
"""
import os
path_base = os.path.dirname(__file__)
import json
import numpy as np
import utils.eval_model as model
import utils.phoneme_features as phone
#==========================================================================
def compute(sig,fs,lang='Multi',stype='Manner'):
    #Choose the model's language
    langFolder = ''
    if lang == 'German (Germany)':#German (Germany)
        langFolder = 'DE_model'
    elif lang == 'English (US)':#English (USA)
        langFolder = 'EN_model'
    elif lang == 'Spanish (Mexico)':#Spanish (Mexico)
        langFolder = 'ES_model'
    elif lang == 'Multilanguage':#(BETA) Model trained with the above languages. 
        langFolder = 'Multi_model'
    path_model = path_base+'/utils/'+langFolder
    
    #Normalization parameters
    norm_param = []
    with open(path_model+'/norm_param.json', 'r') as fp:
            norm_param.append(json.load(fp))

    #Compute phoneme posteriors
    out = model.output_model(sig,fs,path_model,norm_param)
        
    #Get prediction's time stamps.  
    # select output
    if stype == 'Manner':
        posteriors = out['Posteriors'][:,0:8].copy()
        targets = out['Targets'][0:8]
    elif stype == 'Place':
        posteriors = np.hstack([out['Posteriors'][:,0].reshape(-1,1),out['Posteriors'][:,8:16]])
        targets = ['silence']
        targets.extend(out['Targets'][8:16])
    elif stype == 'Voicing':
        posteriors = np.hstack([out['Posteriors'][:,0].reshape(-1,1),out['Posteriors'][:,16:]])
        targets = ['silence']
        targets.extend(out['Targets'][16:])
    elif stype == 'All':
        posteriors = out['Posteriors']
        targets = out['Targets']
    #Compute time stamps
    times = get_predictions(posteriors,targets)
    
    #Compute features
    X = {}
    X.update(phone.get_phoneme_feats(posteriors[:,0:8],out['Predictions_manner'],targets[0:8]))
    X.update(phone.get_consonants(posteriors[:,0:8],out['Predictions_manner']))
    X.update(phone.get_phoneme_feats(posteriors[:,8:16],out['Predictions_place'],targets[8:16]))
    X.update(phone.get_phoneme_feats(posteriors[:,16:],out['Predictions_voicing'],targets[16:]))

    return X,times,posteriors

#---------------------------------------------------------------------------------------------------

def get_predictions(posteriors,targets):
    """
    Return a dictionary with the time stamps of the phoneme class

    Parameters
    ----------
    posteriors : Posterior probabilities
    targets : Class names

    Returns
    -------
    times : Dictionary with time stamps. 
    Each "key" has a list with the time stamps. The elements of the 
    list are the time stamps of the phoneme segments. The first element is 
    the staring time and the second is the ending time of the segment.
    """
    #Predictions
    y = np.argmax(posteriors,axis=1)
    #Time stamps
    times = {}
    t = 0#Target class flag
    for thr in range(posteriors.shape[1]):
        lims = get_times(y,thr)
        if len(lims)>0:
            #Save corresponding class time stamps
            for l in range(len(lims)):
                times[targets[t]+'_'+str(l)] = np.hstack([lims[l][0],lims[l][1]]).reshape(1,-1)
        t+=1
    return times

#---------------------------------------------------------------------------------------------------

def get_times(prediction,thr,stepTime=0.01):
    """
    Get the time stamps from an array of predictions

    Parameters
    ----------
    prediction : Array of predictions from the network
    thr : Phoneme class label
    stepTime : Time shift of the analysis window. The default is 0.01.

    Returns
    -------
    ph_tm : TYPE
        DESCRIPTION.

    """
  
    yp = np.zeros(len(prediction))
    yp[prediction!=thr] = -1
    yp[prediction==thr] = 1
    yp[yp<0] = 0
    #For the last segment
    if yp[-1:] == 1:
        yp = np.insert(yp, len(yp)-1,0 )
    #----------------------
    ydf = np.diff(yp)
    lim_end = np.where(ydf==-1)[0]+1
    lim_ini = np.where(ydf==1)[0]+1
    ph_tm = [] #Phoneme time stamps
    try:
        if lim_end[0]<lim_ini[0]:
            lim_ini = np.insert(lim_ini,0,0)

        for idx in range(len(lim_end)):
            #------------------------------------
    #        try:
            tini = lim_ini[idx]*stepTime
            tend = lim_end[idx]*stepTime

            ph_tm.append(np.hstack([tini,tend]))

    except:
        ph_tm = []
        # print('Length of arrays is invalid')
    return ph_tm