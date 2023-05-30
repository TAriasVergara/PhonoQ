# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 17:50:01 2021

@author: TOMAS
"""
import numpy as np

def get_phoneme_feats(posteriors,predictions,targets):

    X = {}
    for i in range(0,len(targets)):
        tag = targets[i]
        if len(predictions[predictions==i])!=0:
            out_dur = get_durations(predictions,posteriors[:,i],i)#Using the predicted labels
            
            #Proportion of phonemes (phoneme per second) and standard deviation of 
            #phoneme durations
            durSig = len(predictions)*0.01
            pGPI,dGPI = get_gpi(out_dur['Phone_Dur'],durSig)
            
            X[tag+'_Max_Post'] = out_dur['Max_Post']
            X[tag+'_LLR_Post'] = out_dur['LLR_Post']
            X[tag+'_Phone_rate'] = pGPI
            X[tag+'_Avg_dur'] = np.mean(out_dur['Phone_Dur'])
            X[tag+'_Std_dur'] = dGPI
        else:
            print('No '+tag+' detected')
            X[tag+'_Max_Post'] = 0
            X[tag+'_LLR_Post'] = 0
            X[tag+'_Phone_rate'] = 0
            X[tag+'_Avg_dur'] = 0
            X[tag+'_Std_dur'] = 0
        
        if tag.upper() == 'Vowel'.upper():
             rPVI,nPVI = get_pvi(out_dur['Phone_Dur'])
             X[tag+'_rPVI'] = rPVI
             X[tag+'_nPVI'] = nPVI
    return X


def get_consonants(posteriors,predictions):
    """
    Assuming that silence is label 0 and vowels is label 7
    """
    #Consonants
    predictions[(predictions!=0)&(predictions!=7)] = 1#0 Silence; 7 Vowel
    out_dur = get_durations(predictions,posteriors[:,0],1)#Using the predicted labels
    #Proportion of phonemes (phoneme per second) and standard deviation of 
    #phoneme durations
    durSig = len(predictions)*0.01
    pGPI,dGPI = get_gpi(out_dur['Phone_Dur'],durSig)
    rPVI,nPVI = get_pvi(out_dur['Phone_Dur'])
    tag = 'Consonants'
    X = {}
    X[tag+'_Phone_rate'] = pGPI
    X[tag+'_Avg_dur'] = np.mean(out_dur['Phone_Dur'])
    X[tag+'_Std_dur'] = dGPI
    X[tag+'_rPVI'] = rPVI
    X[tag+'_nPVI'] = nPVI
    return X

def get_durations(prediction,posteriors,thr,stepTime=0.01):
    """
    Get the duration (in seconds) of the phonemes and their time stamps
    """
    yp = np.zeros(len(prediction))
    yp[prediction!=thr] = -1
    yp[prediction==thr] = 1
    yp[yp<0] = 0
    #For the last segment
    if yp[-1:] == 1:
        yp = np.insert(yp, len(yp)-1,0 )
    ydf = np.diff(yp)
    lim_end = np.where(ydf==-1)[0]+1
    lim_ini = np.where(ydf==1)[0]+1
    try:
        if lim_end[0]<lim_ini[0]:
            lim_ini = np.insert(lim_ini,0,0)
        
        ph_dur = [] #Phoneme durations
        ph_max = []
        ph_avg = []
        ph_tm = [] #Phoneme time stamps
        for idx in range(len(lim_end)):
            #------------------------------------
    #        try:
            tini = lim_ini[idx]*stepTime
            tend = lim_end[idx]*stepTime
            ph_dur.append(np.hstack([tend-tini]))
            ph_tm.append(np.hstack([tini,tend]))
            
            seg = posteriors[lim_ini[idx]:lim_end[idx]]
            ph_max.append(np.max(seg))
            ph_avg.append(np.mean(seg))
    
        #Average posterior log-likelihood ratio 
        ph_avg = np.asarray(ph_avg)
        LLR_ph =  np.log((ph_avg+np.finfo(float).eps)/((1-ph_avg)+np.finfo(float).eps))
        LLR_ph = np.mean(LLR_ph)
        #Average maximum posterior
        ph_max = np.mean(np.asarray(ph_max))
    except:
        # print('Length of arrays is invalid')
        ph_dur = [0]
        ph_max = 0
        LLR_ph= 0
    #Output
    out = {'Phone_Dur':np.asarray(ph_dur),
           'Max_Post':ph_max,
           'LLR_Post':LLR_ph}
    return out

def get_pvi(d):
    """
    Rythm-based feature
    
    Raw and normalize Pairwise Variability Index (rPVI, nPVI) from:
    Grabe, E., & Low, E. L. (2002). Durational variability in 
    speech and the rhythm class hypothesis. Papers in laboratory 
    phonology, 7(515-546).
    
    (1) rPVI = SUM{k=1,m-1}|d_k - d_{k+1}|/(m -1)
    (2) nPVI = 100*SUM{k=1,m-1}|(d_k - d_{k+1})/(0.5*(d_k + d_{k+1}))|/(m -1)
    
    m   = number of intervals i.e., vocalic-, consonant-, voiced-,... segments
    d_k = duration of k-th interval
    
    input:
        d = list with duration of speech segments (vocalic, voiced, consonants,...)
    output:
        rPVI: Raw Pairwise Variability Index
        nPVI: Normalize Pairwise Variability Index
    """
    rPVI = 0
    nPVI = 0
    m = len(d)
    if m>1:
        for k in range(m-1):
            rPVI += np.abs(d[k]-d[k+1])
            nPVI += np.abs((d[k]-d[k+1])/(0.5*(d[k]+d[k+1])))
        rPVI = rPVI[0]/(m-1)
        nPVI = 100*nPVI[0]/(m-1)
    return rPVI,nPVI

def get_gpi(d,n):
    """
    Rythm-based feature
    
    Global proportions of intervals from:
    Ramus, F., Nespor, M., & Mehler, J. (1999). 
    Correlates of linguistic rhythm in the speech 
    signal. Cognition, 73(3), 265-292.
    
    pGPI = SUM d_k/n
    
    input:
        d = list with duration of speech segments (vocalic, voiced, consonants,...)
        n = Length of the recording [in seconds]
    output:
        pGPI: Global proportion of interval-Speech segments per second
        dGPI: variation of durations
    """
    pGPI = np.sum(d)/n
    dGPI = np.std(d)
    return pGPI,dGPI