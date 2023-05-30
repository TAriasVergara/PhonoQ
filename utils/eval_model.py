# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 22:06:14 2021

@author: TOMAS
"""
import os,gc
import json
import torch
import numpy as np
import scipy.signal as signal
import utils.models as model
# from scipy.io.wavfile import read #Leer y guardar audios
import utils.feature_extract as feats
#------------------For reproducibility-------------------------
seed = 0
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
#--------------------------------------------------------------

def output_model(sig,fs,path_model='./data_model',norm_param=[],win_time=[0.025],step_time=0.01):
    """
    wavfile: audio file
    path_model: Path with the model (.ckp) and the paramaters (.json)
    norm_param = File(s) with nomralization paramaters (min-max)
    """
    #Load model
    RNN_model,RNN_param = load_model(path_model)
    
    #Get features
    X = get_features(sig,fs,RNN_param,win_time,step_time)
    
    
    #Divide speech signal into tensors of size 64 frames x 64 features (mel filters)
    L = X[0].shape[0]#Duration of the signal
    shift = int(RNN_param['seq_dim']/2)
    step_seq = RNN_param['seq_dim']
    agr = int(RNN_param['seq_dim']-shift)
    p_ini = 0
    p_end = step_seq
    posteriors = []
    with torch.no_grad():
        while p_end <= L:

            #Create tensors
            data = get_tensor(X,p_ini,p_end,RNN_param,norm_param)
            
            #Forward pass data into the pre-trained model
            RNN_out = RNN_model(data)
            
            #Activation function
#            activation = torch.nn.Softmax(dim=2)
            activation = torch.nn.Sigmoid()
            scores = activation(RNN_out)
            #===================================================
            a = scores[0,:,:].cpu().numpy()#Activation (posteriors)       
            if p_ini==0:
                posteriors.append(a)
            else:
                posteriors.append(a[agr:,:])
            #Updated sequence to be evaluated
            p_ini = p_ini+shift
            p_end = p_ini+step_seq
            
        posteriors = np.vstack(posteriors)
        
        #Remove the posterior values for the added silence
#        posteriors = posteriors[isil:esil,:]
        nframes = L-((1-win_time[0])/step_time)#Remove padding
        posteriors = posteriors[0:int(nframes),:]
        #Get phoneme predictions
        predictions_manner = np.argmax(posteriors[:,0:8],axis=1) 
        #-
        psilence = posteriors[:,0].reshape(-1,1)
        pplace = np.hstack([psilence,posteriors[:,8:16]])
        predictions_place = np.argmax(pplace,axis=1)-1#Silence will count as -1
        #-
        pvoice = np.hstack([psilence,posteriors[:,16:]])
        predictions_voicing = np.argmax(pvoice,axis=1)-1#Silence will count as -1
        
    #Clear ram
    gc.collect()
    Targets = ['silence','stop','nasal','trill',
               'fricative','approximant','lateral',
               'vowel','labial','alveolar','velar',
               'palatal','postalveolar',
               'central','front','back',
               'voiceless','voiced']
    output = {'Posteriors':posteriors,
              'Predictions_manner':predictions_manner,
              'Predictions_place':predictions_place,
              'Predictions_voicing':predictions_voicing,
              'Targets':Targets}
    return output #posteriors# 

def get_tensor(X,p_ini,p_end,RNN_param,norm_param):
    #Create tensors
    data = np.zeros((1,RNN_param['channel_dim'],RNN_param['seq_dim'],RNN_param['input_dim']),dtype = np.float32)
    for ich in range(len(X)):
        data[0,ich,:,:] = X[ich][p_ini:p_end,:]
        data[0,ich,:,:] = tensor_min_max(data[0,ich,:,:],norm_param[ich])#Apply min-max normalization
    data = torch.from_numpy(data)
    return data
    

def load_model(path_model):
    cell_type = {'rnn':torch.nn.RNN,
                  'lstm':torch.nn.LSTM,
                  'gru':torch.nn.GRU}
    #Select device
    device = torch.device('cpu')
    
    model_files = os.listdir(path_model)
    for f in model_files:
        #Model paramaters
        if f.find('.json')!=-1:
            with open(path_model+'/'+f, 'r') as fp:
                RNN_param = json.load(fp)
            
        #Learned model
        elif f.find('.ckp')!=-1:
                path_RNN_model = path_model+'/'+f
    
    input_shape = (1,RNN_param['channel_dim'],RNN_param['seq_dim'],RNN_param['input_dim'])
    RNN_model = model.Conv_RNN(1,RNN_param,cell_type[RNN_param['cell_type']],input_shape)
    RNN_model.load_state_dict(torch.load(path_RNN_model, map_location=device))
    RNN_model.to(device)
    RNN_model.eval()
    return RNN_model,RNN_param

def get_features(sig,fs,RNN_param,win_time=[0.025],step_time=0.01):
    """
    Get features from wavfile and convert into torch tensor.
    Is the tensor has multiple channel, they can be added here.
    """
    #Remove DC. Re-scaling
    sig = sig-np.mean(sig)
    sig = sig/np.max(np.abs(sig))
    #Resample to 16kHz
    if fs != 16000:
        sig = resample_data(sig,fs,16000)
        fs = 16000
    
#    Add silence at the begining and end to ensure a signal of at least 1 second
    sig = add_silence(sig,fs,win_time[0],step_time)
    
    #Get input channels (features)
    X = []
    for i in win_time:#Win time should be a list, in case of multi_resolution spectrograms
        Xt = feats.get_mel_spec(sig,fs,i,step_time,RNN_param['input_dim'],nfft=1024,fmax=8000)
        X.append(Xt)
        
    return X

def tensor_min_max(input_tensor,norm_param):
    input_tensor = (input_tensor-np.float(norm_param['min']))/(np.float(norm_param['max'])-np.float(norm_param['min']))
    return input_tensor 

def add_silence(sig,fs,win,step):
    #The silence is base on the frame with less energy in the signal
    new_sig = np.zeros(len(sig)+int(fs))
    new_sig[0:len(sig)] = sig
    return new_sig

def resample_data(sig,fs,rs):
    """
    Resample signal

    Parameters
    ----------
    sig : signal to be re-sampled
    fs : Current sampling frequency of the signal
    rs :New sampling frequency

    Returns
    -------
    sig : resampled signal

    """
    num = int((len(sig)/fs)*rs)
    sig = signal.resample(sig,num)
    
    #Re-scaling
    sig = sig - np.mean(sig)
    sig = sig/np.max(np.abs(sig))
    
    return sig