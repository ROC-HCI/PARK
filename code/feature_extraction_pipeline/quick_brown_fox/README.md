
# Quick Brown Fox — Video & Audio Preprocessing + WavLM Features

  

This repo contains a small pipeline to:

  

1)  **Trim and standardize MP4s** using Whisper word timestamps (looks for any of: `quick`, `brown`, `fox`, `dog`, `forest`)
2)  **Export 16 kHz WAV** audio from the trimmed clip
3)  **Extract WavLM embeddings** (microsoft/wavlm-large) and save them as a CSV

All steps can run **CPU-only** or on **GPU** (if available).
  

## Contents

-  `video_preprocess.py`
		- standardizes input video to 15 FPS
		- uses **OpenAI Whisper** to find a sub-segment containing the target words
		- trims the video to `[start-0.5s, end+0.5s]`
		- converts the result to **16 kHz WAV**

-  `extract_wavlm_features.py`
		- loads **microsoft/wavlm-large**
		- computes the mean pooled last-hidden-state for each `.wav` in a folder
		- writes `wavlm_features.csv` (one row per file)

  

## Requirements

 
-  **Python** 3.9+ (tested with 3.9)
-  **ffmpeg** & **ffprobe** available on `PATH`
-  **PyTorch** (CPU or CUDA)
- Python packages: `openai-whisper`, `transformers`, `soundfile`, `librosa`, `tqdm`, `pandas`, `numpy`, `scipy`

The first time you run, models will download automatically:
- Whisper model: `large`
- Hugging Face model: `microsoft/wavlm-large`
	  




## Data layout

  

```
repo/
├── video_preprocess.py
├── extract_wavlm_features.py
└── sample_data/
└── QUICK_BROWN_FOX.mp4
```

The preprocess script writes:

-  `sample_data/QUICK_BROWN_FOX_standardized.mp4`
-  `sample_data/QUICK_BROWN_FOX_standardized_preprocessed.mp4`
-  `sample_data/QUICK_BROWN_FOX_standardized_preprocessed.wav`

The feature script writes:

-  `sample_data/wavlm_features.csv`

  


  

## Environment setup

  

### Option A — Use the provided `environment.yml`

  

```yaml
name: park
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch=2.2
  - torchvision=0.17
  - torchaudio=2.2
  - pytorch-cuda=12.1
  - numpy=1.24
  - pandas=2.1
  - scikit-learn=1.4
  - seaborn=0.12
  - ipykernel
  - pip
  - pip:
      - wandb==0.16.2
      - baal==1.9.1
      - imbalanced-learn==0.12
      - plotly==5.21.0
      - kaleido==0.2.1
      - torchmetrics==0.9.3
      - timm==1.0.9
      - pytorch-lightning==1.9.5
      - mlxtend==0.23.1
      - nbformat>=4.2
      - great_tables==0.13.0
      - selenium==4.25.0
      - matplotlib-venn==0.11.10
      - shap==0.46.0
      - opencv-python-headless==4.10.0.84
      - mediapipe==0.10.11
      - smogn==0.1.2
      - soundfile==0.12.1
      - transformers==4.41.2
      - openai-whisper
```

  

Then:

  

```bash
conda  env  create  -f  environment.yml
conda  activate  park
python  -m  ipykernel  install  --user  --name  park  --display-name  "Python (park)"
```

  

### Option B — Install packages into an existing env

```bash
conda  activate  park
conda  install  -c  conda-forge  ffmpeg
pip  install  -U  openai-whisper  transformers==4.41.2  soundfile==0.12.1  librosa==0.10.1  numpy==1.26.4  scipy==1.12.0
pip  install  torch==2.2.2  torchaudio==2.2.2
```

  


## Usage

  

### 1) Preprocess a single video

  

```bash
python  video_preprocess.py  --file_path  path/to/your.mp4
```

  

### 2) Extract WavLM features

  

```bash
python  extract_wavlm_features.py
```

## Acknowledgments
 
-  **WavLM**: Microsoft (microsoft/wavlm-large)
-  **Whisper**: OpenAI (openai-whisper)
-  **FFmpeg**: FFmpeg project
-  **Transformers**: Hugging Face

 
## Citation

If this helps your work, please cite:

```bibtex
@article{adnan2025novel,
  title={A novel fusion architecture for detecting Parkinson’s Disease using semi-supervised speech embeddings},
  author={Adnan, Tariq and Abdelkader, Abdelrahman and Liu, Zipei and Hossain, Ekram and Park, Sooyong and Islam, Md Saiful and Hoque, Ehsan},
  journal={npj Parkinson's Disease},
  volume={11},
  number={1},
  pages={176},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
  

## License

  

MIT License or your preferred license.