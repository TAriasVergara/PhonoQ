#import sigproc
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt

def get_mel_spec(sig,fs,win=0.025,step=0.01,nfilt=64,nfft=1024,fmax=8000):
    melfb = get_filterbanks(nfilt,nfft = nfft,samplerate = fmax)
    frames = extract_windows(sig, int(win*fs), int(step*fs))
    pow_spec = powerspec(frames, fs, win,nfft)#Power spectrum with FFT. 
    pow_mel = mel_spectrum(pow_spec,melfb)
    return pow_mel

def plot_spec(spec):
    plt.imshow(np.flipud(spec.T),
           interpolation='bilinear',
           aspect='auto',)
    plt.title('Mel spectrum')
##########################################################################
def mel_spectrum(sig_spec,melfb):
    """
    sig_spec: STFT of the speech signal
    melfb: Mel filterbank - get_filterbanks()
    """
    spec_mel = np.dot(sig_spec,melfb.T)
    spec_mel = np.where(spec_mel == 0.0, np.finfo(float).eps, spec_mel)
    return np.log(spec_mel)
##########################################################################
def mfcc_opt(mel_spectrum,numcep=13): 
    feat = dct(mel_spectrum, type=2, axis=1, norm='ortho')[:,:numcep]
    feat -= (np.mean(feat, axis=0) + 1e-8)#Cepstral mean subtraction
    return feat
##########################################################################
def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.0)
    
def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A np array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,int(nfft/2+1)])
    for j in range(0,nfilt):
        for i in range(int(bin[j]),int(bin[j+1])):
            fbank[j,i] = (i - bin[j])/(bin[j+1]-bin[j])
        for i in range(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i)/(bin[j+2]-bin[j+1])
            
#    fbank -= (np.mean(fbank, axis=0) + 1e-8)
    return fbank
#------------------------------------------------------------------------------------------------
def lifter(cepstra,L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes,ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1+ (L/2)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra
    
#*****************************************************************************
def extract_windows(signal, size, step):
    # make sure we have a mono signal
    assert(signal.ndim == 1)
    
#    # subtract DC (also converting to floating point)
#    signal = signal - signal.mean()
    
    n_frames = int((len(signal) - size) / step)
    
    # extract frames
    windows = [signal[i * step : i * step + size] 
               for i in range(n_frames)]
    
    # stack (each row is a window)
    return np.vstack(windows)

def powerspec(X, rate, win_duration, n_padded_min=0):
    win_size = int(rate * win_duration)
    
    # apply hanning window
    X *= np.hanning(win_size)
    
    # zero padding to next power of 2
    if n_padded_min==0:
        n_padded = max(n_padded_min, int(2 ** np.ceil(np.log(win_size) / np.log(2))))
    else:
        n_padded = n_padded_min
    # Fourier transform
#    Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=n_padded)
    Y = np.absolute(Y)
    
    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]
    
    return np.abs(Y) ** 2
