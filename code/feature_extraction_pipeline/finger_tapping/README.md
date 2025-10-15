# Finger Tapping Feature Extraction

MediaPipe-based pipeline to extract quantitative features from finger-tapping videos (left/right hand).  
Outputs a single CSV with per-video metadata + features, and saves per-run intermediates to disk for fast re-runs.

---

## Features

- Detects left/right hand landmarks with **MediaPipe Hands**
- Computes per-frame wrist–thumb and wrist–index angles
- Denoises and trims signals; detects peaks/minima
- Derives:
  - Rhythm metrics (periods, frequency, aperiodicity/entropy)
  - Wrist movement stats (x/y/distance velocity & dispersion)
  - Freeze / interruption counts & max durations
  - Amplitude statistics and decrement trends
- Writes all features to a CSV (plus annotated video and intermediates)

---

## Repo Layout

```
.
├── feature_extraction.py        # CLI entrypoint (this file)
├── sample_videos/               # your input videos (you provide)
└── README.md                    # this file
```

When running, the script creates per-video subfolders under your `--output` directory:

```
<output>/<video_stem>/<HAND>/
  ├── output.mp4                 # annotated video with overlays
  └── intermediate_features.pkl  # cached arrays for fast re-runs
```

---

## Requirements

- **Python 3.9** (tested)
- **ffmpeg** (for `ffprobe` used by `get_length`)
- Python packages: OpenCV (headless or GUI), MediaPipe, NumPy, Pandas, SciPy, scikit-learn, statsmodels, seaborn, smogn, matplotlib

### Quick setup (Conda + pip)

```bash
conda create -n myenv39 python=3.9 ffmpeg -c conda-forge -y
conda activate myenv39

# Headless servers (recommended)
pip install   opencv-python-headless==4.10.0.84 mediapipe==0.10.11   numpy==1.24 pandas==2.1 scikit-learn==1.4 statsmodels==0.14 seaborn==0.12   smogn==0.1.2 matplotlib plotly kaleido torchmetrics timm pytorch-lightning mlxtend shap
```

> On desktops where you want to see OpenCV windows, replace `opencv-python-headless` with `opencv-python`. On servers, keep the **headless** build and don’t use GUI flags.

If `ffmpeg` isn’t installed:
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`  
- macOS (Homebrew): `brew install ffmpeg`

---

## Input Expectations

- Supported video formats: `.mp4`, `.webm`, `.mov`, `.avi`, `.mkv`, `.m4v`
- Optional filename pattern to auto-parse metadata:
  ```
  <prefix>_<participant_id>_<...>_(left|right)_YYYY-MM-DD.<ext>
  ```
  If not matched, **participant_id** defaults to the filename and **date** defaults to **today**.
- Optional labels CSV (flexible column names accepted):
  - `filename` (or `file` / `name` / `id`)
  - `rating` (optional)
  - `diagnosis` (optional)

The base name **without extension** is used to map labels.

---

## Usage

Activate the environment:

```bash
conda activate myenv39
```

### Single file

```bash
python feature_extraction.py   --file sample_videos/FINGER_TAPPING_RIGHT.mp4   --output outputs/   --csv features.csv   --hand auto   --labels labels.csv
```

### Folder (recursive)

```bash
python feature_extraction.py   --folder sample_videos/   --output outputs/   --csv features.csv   --hand auto   --labels labels.csv
```

**Arguments**

- `--file` / `--folder` (mutually exclusive): input source  
- `--output`: directory for intermediates and CSV  
- `--csv`: output CSV filename (created in `--output`, default: `features.csv`)  
- `--labels`: optional labels CSV  
- `--hand`: `left | right | auto` (default: `auto`)  
  - `auto` infers from filename (contains “left”/“right”), otherwise defaults to **left**

**CSV Columns (examples)**

- Metadata: `filename, participant_id, hand, date, rating, diagnosis`
- Features: e.g., `period_median_denoised`, `frequency_mean_trimmed`, `numFreeze_denoised`, `wrist_mvmnt_x_mean`, `amplitude_entropy_trimmed`, `aperiodicity_denoised`, etc.

---

## How It Works (High Level)

1. **Hand tracking** via MediaPipe; resolves multiple hands by label & size.  
2. **Signal construction**: angle at wrist between (wrist→thumb_tip) and (wrist→index_tip).  
3. **Denoise & trim**:
   - Polynomial interpolation (guarded) for missing frames
   - Keep the longest valid segment
   - Peak/minima detection with distance & height constraints
4. **Feature extraction**:
   - Period & frequency stats + entropy
   - FFT power-spectrum entropy (aperiodicity)
   - Freeze/interruption counts & max durations using speed thresholds
   - Wrist movement stats (x/y/distance)
   - Amplitude stats + decrement trends (linear fit & polynomial fit degree)

---

## Tips & Troubleshooting

- **Headless servers**: use `opencv-python-headless`; don’t open windows.  
- **“moov atom not found”**: input file is corrupted/incomplete. Check with:
  ```bash
  ffprobe -hide_banner path/to/video.mp4
  ```
  Re-export the video if ffprobe fails.
- **Re-runs are fast**: if a video was processed, `intermediate_features.pkl` will be reused.

---

## Reproducibility

- Tested with Python **3.9** and packages pinned above.
- Deterministic given fixed inputs (SciPy peak detection, NumPy ops).

---

## Citation

If this helps your work, please cite:

```bibtex
@article{islam2023using,
  title={Using AI to measure Parkinson’s disease severity at home},
  author={Islam, Md Saiful and Rahman, Wasifur and Abdelkader, Abdelrahman and Lee, Sangwu and Yang, Phillip T and Purks, Jennifer Lynn and Adams, Jamie Lynn and Schneider, Ruth B and Dorsey, Earl Ray and Hoque, Ehsan},
  journal={NPJ digital medicine},
  volume={6},
  number={1},
  pages={156},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

---

## License

Add a `LICENSE` file (e.g., MIT). Example:

```
MIT License

Copyright (c) 2025 …

Permission is hereby granted, free of charge, to any person obtaining a copy…
```

---

## Maintainers / Acknowledgments

- Maintainer: _Ehsan Hoque_ (your.email@domain)  
- Built with: MediaPipe Hands, OpenCV, SciPy, scikit-learn, statsmodels.
