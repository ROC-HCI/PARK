from pathlib import Path 
import pandas as pd 
from tqdm import tqdm
import subprocess
import shutil
import numpy as np
import speech_utils as su
from soundfile import read
import os
import json

def covert_mp4_to_wav16(input_file, output_file):
    if os.path.isfile(output_file):
        return 0
    command = "ffmpeg -i \"%s\" -ar 16000 \"%s\" "%(input_file,output_file)
    status = subprocess.call(command)
    return status

def convert_mp4_to_wav(input_directory,output_directory):
    files = os.listdir(input_directory)
    if not os.path.exists(output_directory):
        create_directory(output_directory)
    failed_videos = []
    for file in tqdm(files):
        if ".mp4" not in file:
            continue
        input_file = os.path.join(input_directory, file)
        output_file = os.path.join(output_directory, file[0:-4]+".wav")
        status = covert_mp4_to_wav16(input_file, output_file)
        if status != 0:
            #print(input_file)
            failed_videos.append(file)
    return failed_videos

def getFeatures(parent_directory, fileID):
    features = {}
    features['fileID'] = fileID
    
    try:
        features['f0m'] = None
        features['f0j'] = None
        features['f0jr'] = None
        features['ash'] = None
        features['ashr'] = None

        for c in range(0, 13):
            this_cepm = 'cepm' + str(c)
            features[this_cepm] = None
        for k in range(0, 13):
            this_cepj = 'cepj' + str(k)
            features[this_cepj] = None
        features['Hnorm'] = None
        features['alpha'] = None
        features['ppe'] = None
        features['relbandpower0'] = None
        features['relbandpower1'] = None
        features['relbandpower2'] = None
        features['relbandpower3'] = None
        features['f0std'] = None

        file_path = parent_directory / fileID
        
        data, rate = read(file_path.absolute().as_posix(), dtype='float32')
        
        N = data.shape[0]
        T = N/rate
        
        # Pitch Extraction
        f0dt = 0.02
        f0lo = 50
        f0hi = 500
        (f0, f0t, f0p) = su.swipep(data, rate, f0lo, f0hi, f0dt)
        
        # Plot Pitch
        # plt.plot(1000*f0t,f0,'b--')
        # plt.xlabel('Time (ms)')
        # plt.ylabel('Pitch (Hz)')
        # plt.ylim(f0lo,f0hi)
        # plt.show()

        # Pitch Features
        features['f0m'] = np.median(f0)
        features['f0j'] = np.mean(np.abs(np.diff(f0))/f0dt)
        features['f0jr'] = np.median(np.abs(np.diff(f0))/f0dt)
        features['f0std'] = np.std(f0)

        # MFCC Features
        # mfcc_features = mfcc(data,rate)
        mfcc_features = su.mfcc(data, rate).transpose()
        cepm = np.median(mfcc_features, axis=0)
        cept = np.linspace(0, T, mfcc_features.shape[0])

        # plt.plot(cept,mfcc_features[:,3])
        # plt.show()

        cepdt = cept[1] - cept[0]
        cepj = np.mean(np.abs(np.diff(mfcc_features, axis=0))/cepdt, axis=0)
        for j in range(0, 13):
            this_cepm = 'cepm' + str(j)
            this_cepj = 'cepj' + str(j)
            features[this_cepm] = cepm[j]
            features[this_cepj] = cepj[j]

        # Amplitude Features
        ampdt = 0.05
        ampwin = np.round(ampdt*rate).astype(int)
        abuf = su.buffer(data, ampwin)
        l1amp = np.mean(np.abs(abuf), axis=0)
        l2amp = np.sqrt(np.mean(np.power(abuf, 2), axis=0))
        linfamp = np.max(np.abs(abuf), axis=0)
        ampt = np.linspace(0, T, l1amp.shape[0])
        ash = np.mean(np.abs(np.diff(l2amp))/ampdt)
        ashr = np.median(np.abs(np.diff(l2amp))/ampdt)
        features['ash'] = ash
        features['ashr'] = ashr

        # Spectral Features
        specdt = 0.05
        bandedges = [0, 500, 1000, 2000, 4000]
        specwin = np.power(2, np.ceil(np.log2(np.round(specdt*rate)))).astype(int)
        specovl = np.floor(specwin/2).astype(int)
        sbuf = su.buffer(data, specwin, specovl)
        temp = np.log10(np.abs(np.power(np.fft.fft(sbuf, axis=0), 2)))
        psd = np.mean(temp, axis=1)
        specwin2 = np.floor(specwin/2).astype(int)

        psd = psd[0:specwin2]
        f = (np.arange(0, specwin2, 1).transpose()/specwin2)*(rate/2)
        Nbands = len(bandedges)-1
        relbandpower = np.zeros((Nbands,))
        for i in range(0, Nbands):
            iflo = np.argwhere(f >= bandedges[i])[0][0].astype(int)
            ifhi = np.argwhere(f < bandedges[i+1])[-1][0].astype(int)

            relbandpower[i] = np.median(psd[iflo:(ifhi+1)])
        relbandpower = relbandpower - np.max(relbandpower)

        features['relbandpower0'] = relbandpower[0]
        features['relbandpower1'] = relbandpower[1]
        features['relbandpower2'] = relbandpower[2]
        features['relbandpower3'] = relbandpower[3]

        # DFA
        lag = np.logspace(0.5, 1.5, 20)
        scales, fluct, alpha = su.dfa(data.flatten(), lag)
        features['alpha'] = alpha

        # PPE
        ppe = su.ppe(f0, f0lo, f0hi)
        features['ppe'] = ppe

        # RPDE
        Hnorm, rpd = su.rpde(
            data, tau=20, epsilon=np.std(data.flatten()), tmax=400)
        features['Hnorm'] = Hnorm
        
    except:
        print('Error with file: ', fileID)
        
    return features