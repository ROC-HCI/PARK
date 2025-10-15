import soundfile as sf
from transformers import AutoFeatureExtractor, WavLMModel
import torch.nn as nn
import torch
import os 
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import tqdm
from helpers import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

def covert_wav_file_to_ndarray(input_file):
    audio_data, sampling_rate = sf.read(input_file)
    audio_array = np.array(audio_data, dtype=np.float32)
    audio_array /= np.max(np.abs(audio_array))
    return audio_array.flatten()

def get_audio_representaions(input_directory):
    files = os.listdir(input_directory)
    rep_dict = {}
    for file in tqdm(files):
        if ".wav" not in file:
            continue
        input_file = os.path.join(input_directory, file)
        audio_array = covert_wav_file_to_ndarray(input_file)
        rep_dict[file] = audio_array
    
    return rep_dict

def extract_features(rept_dict, sampling_rate):
   
    features_dict = {}
    for file, rep in tqdm(rept_dict.items()):
        model.eval()
        torch.cuda.empty_cache()
        gc.collect()
        inputs = feature_extractor(rep, sampling_rate=sampling_rate, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state.cpu()
        last_hidden_states_average = torch.mean(last_hidden_states, 1).cpu().numpy().flatten().tolist()
        features_dict[file] = last_hidden_states_average
    return features_dict 

if __name__ == "__main__":
    input_directory = Path("./sample_data")
    output_directory = Path("./sample_data")
    fox_test_rep_dict = get_audio_representaions(input_directory)
    print(f"Number of audio files found: {len(fox_test_rep_dict)}")
    keys_to_remove=[]
    for k,v in fox_test_rep_dict.items():
        if np.isnan(v).any():
            keys_to_remove.append(k)
    for key in keys_to_remove:
        removed_value = fox_test_rep_dict.pop(key, None)

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    model = model.to(device)

    wavlm_test_features_dict = extract_features(fox_test_rep_dict,16000)
    wavlm_test_data_list = []
    for key, value in wavlm_test_features_dict.items():
        wavlm_test_data_list.append({"fileID": key, **{f"wavlm_feature{x}": value[x] for x in range(len(value))}})
    wavlm_test_features = pd.DataFrame(wavlm_test_data_list)
    wavlm_test_features = wavlm_test_features.sort_values(by='fileID').reset_index(drop=True)

    csv_name = 'wavlm_features'
    wavlm_test_features.to_csv(f'{input_directory.as_posix()}/{csv_name}.csv', index=False)
    del model