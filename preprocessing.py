"""Module to preprocess the raw eeg data. Roughly speaking, it does the non-machine learning 
feature engineering"""
import numpy as np
import scipy
import pandas as pd
import math
import logging
from scipy.stats import skew, kurtosis
from logging import log

import datareader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def corr(data,type_corr):
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    w,v = np.linalg.eig(C)
    x = np.sort(w)
    x = np.real(x)
    return x

def preprocess_patients(patientlist, train=True, preproc='base'):
    """Preprocesses all files from the listed patients"""
    pathlist = []
    for patient in patientlist:
        if type(patient) != type(int()):
            raise Exception('Expected an integer for patient number')

        tmp, targets = datareader.get_data_paths(patient, train=train, preproc=False)
        pathlist += tmp

    logger.log(logging.INFO, 'Preprocessing files...')
    k = 0
    data_iterator = datareader.iterate_datafiles_maxmem(pathlist, datareader.mat_to_data, maxmem=200)
    for data_dict, path in zip(data_iterator, pathlist):
        k += 1
        logger.log(logging.INFO, 'Preprocessing file ' + str(k) + ' of ' + str(len(pathlist)))
        preproc_fct, kwargs = get_preproc(preproc)
        feat = preproc_fct(data_dict, **kwargs)
        newpath = datareader.get_preproc_path(path, train, preproc)[:-4] # get preproc path, without .mat
        print(newpath)
        np.save(newpath, feat)

def get_preproc(preproc):
    """returns the appropriate preprocessing fct"""
    if preproc=='base': return calculate_features, {}
    if preproc=='stft': return calculate_stft, {'max_freq':40}
    if preproc=='stft_polar': return calculate_stft, {'max_freq':40, 'polar':True}

def calculate_features(data_dict, **kwargs):
    # translation of the Matlab feature extractor 
    # Credit: https://www.kaggle.com/deepcnn
    f = data_dict
    fs = f['iEEGsamplingRate'][0,0]
    eegData = f['data']
    [nt, nc] = eegData.shape
    subsampLen = math.floor(fs * 60)
    numSamps = int(math.floor(nt / subsampLen));      # Num of 1-min samples
    sampIdx = range(0,(numSamps+1)*subsampLen,subsampLen)
    #print(sampIdx)
    feat = [] # Feature Vector
    for i in range(1, numSamps+1):
        epoch = eegData[sampIdx[i-1]:sampIdx[i], :]

        # compute Shannon's entropy, spectral edge and correlation matrix
        # segments corresponding to frequency bands
        lvl = np.array([0.1, 4, 8, 12, 30, 70, 180])  # Frequency levels in Hz
        lseg = np.round(nt/fs*lvl).astype('int')
        D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))
        D[0,:]=0                                # set the DC component to zero
        D /= D.sum()                      # Normalize each channel               
        
        dspect = np.zeros((len(lvl)-1,nc))
        for j in range(len(dspect)):
            dspect[j,:] = 2*np.sum(D[lseg[j]:lseg[j+1],:], axis=0)

        # Find the shannon's entropy
        spentropy = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)

        # Find the spectral edge frequency
        sfreq = fs
        tfreq = 40
        ppow = 0.5

        topfreq = int(round(nt/sfreq*tfreq))+1
        A = np.cumsum(D[:topfreq,:])
        B = A - (A.max()*ppow)
        spedge = np.min(np.abs(B))
        spedge = (spedge - 1)/(topfreq-1)*tfreq

        # Calculate correlation matrix and its eigenvalues (b/w channels)
        data = pd.DataFrame(data=epoch)
        type_corr = 'pearson'
        lxchannels = corr(data, type_corr)
        
        # Calculate correlation matrix and its eigenvalues (b/w freq)
        data = pd.DataFrame(data=dspect)
        lxfreqbands = corr(data, type_corr)
        
        # Spectral entropy for dyadic bands
        # Find number of dyadic levels
        ldat = int(math.floor(nt/2.0))
        no_levels = int(math.floor(math.log(ldat,2.0)))
        seg = math.floor(ldat/pow(2.0, no_levels-1))

        # Find the power spectrum at each dyadic level
        dspect = np.zeros((no_levels,nc))
        for j in range(no_levels-1,-1,-1):
            dspect[j,:] = 2*np.sum(D[int(math.floor(ldat/2.0))+1:ldat,:], axis=0)
            ldat = int(math.floor(ldat/2.0))

        # Find the Shannon's entropy
        spentropyDyd = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)

        # Find correlation between channels
        data = pd.DataFrame(data=dspect)
        lxchannelsDyd = corr(data, type_corr)
        
        # Fractal dimensions
        no_channels = nc
        #fd = np.zeros((2,no_channels))
        #for j in range(no_channels):
        #    fd[0,j] = pyeeg.pfd(epoch[:,j])
        #    fd[1,j] = pyeeg.hfd(epoch[:,j],3)
        #    fd[2,j] = pyeeg.hurst(epoch[:,j])

        #[mobility[j], complexity[j]] = pyeeg.hjorth(epoch[:,j)
        # Hjorth parameters
        # Activity
        activity = np.var(epoch)

        # Mobility
        mobility = np.divide(np.std(np.diff(epoch)), np.std(epoch))

        # Complexity
        complexity = np.divide(np.divide(np.diff(np.diff(epoch)),
                                         np.std(np.diff(epoch))), mobility)
        # Statistical properties
        # Skewness
        sk = skew(epoch)

        # Kurtosis
        kurt = kurtosis(epoch)

        # compile all the features
        feat = np.concatenate((feat,
                               spentropy.ravel(),

                     spedge.ravel(),
                               lxchannels.ravel(),
                               lxfreqbands.ravel(),
                               spentropyDyd.ravel(),
                               lxchannels.ravel(),
                               #fd.ravel(),
                               activity.ravel(),
                               mobility.ravel(),
                               complexity.ravel(),
                               sk.ravel(),
                               kurt.ravel()
                                ))

    return feat.astype(np.float32)

def stft(x, M, hop_ratio, fft_fct=scipy.fft):
    """Stft calculation. 
    x :         input signal
    M :         windowing length
    hop_ratio : window spacing. 1 means no overlap, 0.5 means all windows overlap halfway, etc."""
    if not (0<hop_ratio<=1):
        raise ValueError('The hop must be in the set (0,1]')
    w = np.hanning(M)
    hop = int(len(w)*hop_ratio)
    return np.array([fft_fct(w*x[i:i+M]) for i in range(0, len(x)-M, hop)])

def calculate_stft(data_dict, disp=False, polar=False, max_freq=40, **kwargs):
    """Processes the raw data into a stft"""
    vals = data_dict['data']
    decim_factor = 1
    fs = data_dict['iEEGsamplingRate'][0,0]/decim_factor
    M = 512

    # lowpass and decimate the data
    #vals = scipy.signal.decimate(vals, decim_factor, axis=0, zero_phase=True).astype(np.float32)

    # Apply stft
    max_freq_sample = int(round(M/fs*max_freq))
    newvals = np.array([stft(x, M, 0.85)[:,:max_freq_sample] for x in vals.T]).astype(np.complex64)

    #Dsiplay some stuff
    if disp:
        print(vals.nbytes/1024**2)
        print(newvals.nbytes/1024**2)
        time_res = vals.shape[0]/fs/newvals.shape[1]
        freq_res = fs/M
        print("%2.3f"%time_res + ' seconds per sample')
        print("%2.3f"%freq_res + ' Hz per sample. Max frequency: ' + "%2.3f"%(newvals.shape[2]*freq_res))

    # Save in polar coordinates if needed
    if polar:
        mag = np.absolute(newvals)
        angle = np.angle(newvals)
        out = mix_cat(mag, angle)
    # else, save as floats instead of complex by catting real and imaginary parts
    else:
        out = mix_cat(newvals.real, newvals.imag)

    return out


def mix_cat(x,y):
    """Concatenates two arrays of same dimension by mixing it along the last axis.
    output: x[...,0], y[...,0], x[...,1], etc."""
    if x.shape != y.shape:
        raise Exception('Both arrays must have the same shape')

    out = np.empty( list(x.shape[:-1]) + [x.shape[-1]*2], dtype=x.dtype)
    out[..., 0::2] = x
    out[..., 1::2] = y
    return out




def main():
    """Test function"""
    preprocess_patients([1], preproc='stft')
    preprocess_patients([1], preproc='stft_polar')


if __name__ == '__main__':
    main()




