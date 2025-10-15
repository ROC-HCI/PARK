from pathlib import Path 
import pandas as pd 
import subprocess
import shutil
import numpy as np
import os
import json
import gc
from datetime import timedelta
import click 
import soundfile as sf
from tqdm import tqdm
from soundfile import read
import torch
import whisper

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("large")
whisper_model.to(device)

ffmpeg_location = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True).stdout.strip()

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def standardize_video(input_file, output_file):
    if output_file.exists():
        return output_file
    
    command = f"{ffmpeg_location} -i \"%s\" -vf \"fps=15\" -y \"%s\""% (input_file, output_file)
    status = subprocess.call(command, shell=True)
    
    if status != 0:
        raise Exception("Error converting file %s to mp4" % input_file)
    
    return output_file

def pre_process_quick_brown_fox_video(input_file, output_file, whisper_model):
    words_list = ['quick','brown','fox','dog','forest']
    start = float('inf')
    end = 0.
    result = whisper_model.transcribe(input_file.resolve().as_posix(), word_timestamps=True)
    
    for i in result['segments'] :
            if any(ele in i['text'] for ele in words_list):
                start = min(start,i['start'])
                end = max(end,i['end'])

    dur = get_length(input_file.resolve().as_posix())
    start = start - 0.5 if start - 0.5 > 0 else 0
    end = end + 0.5 if end + 0.5 < dur else dur

    formatted_start =  ''.join(['0',str(timedelta(seconds=start))])
    formatted_end = ''.join(['0',str(timedelta(seconds=end))])

    command = f"{ffmpeg_location} -i %s -ss %s -to %s -c:v libx264 %s" % (input_file, formatted_start, formatted_end, output_file)
    status = subprocess.call(command, shell=True)
    if status != 0:
        raise Exception("Error converting file %s to mp4" % input_file)

    return output_file

def covert_mp4_to_wav16(input_file, output_file):
    if output_file.exists():
        return output_file
    command = f"{ffmpeg_location} -i \"%s\" -ar 16000 \"%s\" "%(input_file,output_file)
    status = os.system(command)
    if status != 0:
        raise Exception("Error converting file %s to wav" % input_file)
    
    return output_file

def covert_wav_file_to_ndarray(input_file):
    audio_data, sampling_rate = sf.read(input_file)
    audio_array = np.array(audio_data, dtype=np.float32)
    audio_array /= np.max(np.abs(audio_array))
    return audio_array.flatten()

def process_audio_file(input_dir, FILE_NAME):
    input_file = input_dir / FILE_NAME
    mp4_sntadardized_video = standardize_video(input_file, input_file.with_name(f'{input_file.stem}_standardized.mp4'))
    preprocssed_video = pre_process_quick_brown_fox_video(mp4_sntadardized_video, mp4_sntadardized_video.with_name(f'{mp4_sntadardized_video.stem}_preprocessed.mp4'), whisper_model)
    wav16_file = covert_mp4_to_wav16(preprocssed_video, preprocssed_video.with_suffix('.wav'))
    
    # remove intermediate files
    mp4_sntadardized_video.unlink()
    preprocssed_video.unlink()

    # Get numpy array for further processing
    # audio_array = covert_wav_file_to_ndarray(wav16_file)
    # wav16_file.unlink()
    return

def process_video_file(input_dir, FILE_NAME):
    input_file = input_dir / FILE_NAME
    mp4_sntadardized_video = standardize_video(input_file, input_file.with_name(f'{input_file.stem}_standardized.mp4'))
    
    # remove intermediate files
    # mp4_sntadardized_video.unlink()
    return

@click.command()
@click.option('--file_path', default=None, help='Path to the input file (relative or absolute)')
def main(file_path):
    if file_path is None:
        file_path = "sample_data/QUICK_BROWN_FOX.mp4"

    file = Path(file_path)
    input_dir = file.parent
    FILE_NAME = file.name

    process_audio_file(input_dir, FILE_NAME)
    print("Audio preprocessing completed successfully.")
    
    process_video_file(input_dir, FILE_NAME)
    print("Video preprocessing completed successfully.")


if __name__ == '__main__':
    main()
