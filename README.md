# PARK: Multimodal Feature Extraction, Unimodal and Fusion Modeling, and Evaluation Pipeline

End-to-end repo for extracting features from **smile videos**, **finger-tapping videos**, and **speech (quick brown fox)**; training **unimodal** and **uncertainty‑aware fusion** models; and running paper-ready **analyses** and figures.

---

## Table of Contents

- [Project Layout](#project-layout)
- [Quick Start](#quick-start)
- [Data Layout](#data-layout)
- [Feature Extraction](#feature-extraction)
  - [Smile (Facial Expression)](#smile-facial-expression)
  - [Finger Tapping](#finger-tapping)
  - [Speech — “Quick Brown Fox”](#speech--quick-brown-fox)
- [Unimodal Models](#unimodal-models)
- [Fusion Models](#fusion-models)
- [Analysis & Figures](#analysis--figures)
- [Pretrained Models](#pretrained-models)
- [Reproducibility Tips](#reproducibility-tips)
- [Citations](#citations)
- [License](#license)

---

## Project Layout

```
.
├── code
│   ├── analyses
│   ├── feature_extraction_pipeline
│   │   ├── facial_expression_smile
│   │   ├── finger_tapping
│   │   └── quick_brown_fox
│   ├── fusion_model
│   └── unimodal_models
├── data
│   ├── facial_expression_smile
│   ├── finger_tapping
│   └── quick_brown_fox
├── environment.yml 
└── models
```
Key highlights:
- `code/feature_extraction_pipeline/*` — task‑specific feature extraction.
- `code/unimodal_models/*` — per‑modality model training/inference.
- `code/fusion_model/` — uncertainty‑aware fusion (all three tasks/modalities).
- `code/analyses/` — metrics, stats, demographics, figures (paper‑ready).
- `data/` — canonical CSVs for each modality + fusion splits.
- `models/` — saved scalers and model checkpoints per modality + fusion.

---
## Quick Start

Use the YAML file **environment.yml** to create a conda env called park.

```bash
# From the repository root where environment.yml lives
# Create a new environment (override the name if you want)
conda env create -f environment.yml -n park

# Activate it
conda activate park

# If the environment already exists and you want to update it
# (this prunes removed deps and adds new ones)
conda env update -f environment.yml --prune
```

---

## Data Layout

Canonical CSVs and metadata live in `data/`:

- **Fusion**: `full_fusion_dataset.csv`, `train_val_data_to_save.csv`, `test_data_to_save.csv`, `test_df_*_to_save.csv`, `predictions.csv`
- **Demographics/labels**: `demographic_table.csv`, `metadata_*.csv`, `labeling_neurologists_PARK.csv`, `df_stage_data.csv`
- **Per‑modality**:
  - Smile: `data/facial_expression_smile/data_smile.csv`
  - Finger tapping: `data/finger_tapping/data_finger_tapping.csv`
  - Speech: `data/quick_brown_fox/data_speech.csv`

Splits:
- Dev/Test IDs: `dev_set_participants.txt`, `test_set_participants*.txt`  


> Tip: Keep file paths consistent with these CSVs to enable plug‑and‑play training across scripts.

---

## Feature Extraction

All task pipelines live under `code/feature_extraction_pipeline/`.

### Smile (Facial Expression)

- **Script**: `code/feature_extraction_pipeline/facial_expression_smile/smile_feature_extraction.py`
- **Sample Input**: `code/feature_extraction_pipeline/facial_expression_smile/raw_videos/SMILE.mp4`
- **Output Folder**: `code/feature_extraction_pipeline/facial_expression_smile/extracted_features/`
- **Notes**:
  - Extracts framewise facial landmarks/AUs and aggregates per‑video features.
  - See [README](code/feature_extraction_pipeline/facial_expression_smile/README.md) in the same folder for task-specific usage details.


**Run:**
```bash
```bash
python smile_feature_extraction.py --video_files_directory=raw_videos --openface_files_directory=openface_extracts --output_dir=extracted_features 2>/dev/null
```

### Finger Tapping

- **Script**: `code/feature_extraction_pipeline/finger_tapping/feature_extraction.py`
- **Samples**: `code/feature_extraction_pipeline/finger_tapping/sample_videos/*.mp4`
- **Outputs**: `code/feature_extraction_pipeline/finger_tapping/sample_outputs/`
  - Contains **intermediate features** (`*.pkl`) and **annotated videos** (`output.mp4`), plus placeholders for MediaPipe annotations.
- **Notes**: 
    - Produces a consolidated `features.csv` suitable for model training.
    - See [README](code/feature_extraction_pipeline/finger_tapping/README.md) in the same folder for task-specific usage details.

**Run:**
```bash
python feature_extraction.py   --folder sample_videos/   --output outputs/ 
```

### Speech — “Quick Brown Fox”

- **Core Scripts**:
  - `code/feature_extraction_pipeline/quick_brown_fox/video_preprocess.py` — standardizes audio from input videos.
  - `code/feature_extraction_pipeline/quick_brown_fox/extract_wavlm_features.py` — extracts **WavLM** embeddings and exports `wavlm_features.csv`.
  - Helpers: `helpers.py`, `speech_utils.py`
- **Sample Data**: `code/feature_extraction_pipeline/quick_brown_fox/sample_data/`
  - Includes `QUICK_BROWN_FOX.mp4`, standardized outputs, and example `wavlm_features.csv`.
- **Notes**: See [README](code/feature_extraction_pipeline/quick_brown_fox/README.md) in the same folder for task-specific usage details.

**Run:**
```bash
# (1) optional preprocessing (normalize audio/video)
python  video_preprocess.py  --file_path  path/to/your.mp4

# (2) extract WavLM features
# this code consdiers `sample_data' as input and output directory
python  extract_wavlm_features.py
```

---

## Unimodal Models

Each modality has a BAAL‑style unimodal trainer in `code/unimodal_models/`:

- **Smile**: 
    - Code: `code/facial_expression_smile/unimodal_smile_baal.py`  
    - Config: `code/facial_expression_smile/constants_baal.py`
    - Data: `data/facial_expression_smile/data_smile.csv` 
- **Finger**: 
    - Code: `code/finger_tapping/unimodal_finger_baal.py`  
    - Config: `code/finger_tapping/constants_baal.py`
    - Data: `data/finger_tapping/data_finger_tapping.csv` 
- **Speech**: 
    - Code: `code/quick_brown_fox/unimodal_fox_wavlm_baal.py`  
    - Config: `code/quick_brown_fox/constants_baal.py`
    - Data: `data/quick_brown_fox/data_speech.csv`


**Usage** (The best performing hyper-parameters are already set inside the code):
```bash
# Smile
python unimodal_smile_baal.py 

# Finger
python unimodal_finger_baal.py 

# Speech
python unimodal_fox_wavlm_baal.py 
```

Artifacts saved under `models/<modality>_{scaler,predictive_model,residual_model}/`.

---

## Fusion Models

Uncertainty‑aware late fusion combines unimodal predictions and optionally uncertainty terms.

- **Core module**: `code/fusion_model/uncertainty_aware_fusion_wavlm.py`
- **Configs/consts**: `code/fusion_model/constants.py`
- **Artifacts**: 
  - Trained fusion checkpoint: `models/uncertainty_aware_fusion/model.pth`
  - Config: `models/uncertainty_aware_fusion/model_config.json`
  - Uncertain indices: `code/fusion_model/uncertain_indices.csv`

**Typical usage** (example flags):
```bash
python uncertainty_aware_fusion_wavlm.py
```

> Notes
> - The fusion expects consistent participant/session keys across CSVs.  
> - During the runtime, the fusion code will save several files later required for diffeerent analyses.

---

## Analysis & Figures

Located in `code/analyses/`:

- **big_csv_generation.ipynb**: Generate a consolidated “big” CSV that merges per-modality features, labels, splits, and metadata into a single analysis-ready table (used by stats/figures).


- **Notebooks**  
  - **demographic_table_generation.ipynb**: Build cohort demographics tables from metadata.  
  - **figure_1.ipynb** … **figure_4.ipynb**: Recreate all paper figures.   
  - **statistical_analysis.ipynb**: Run hypothesis tests, effect sizes, CIs, and produce result tables.  
  - *Exported plots live under* `code/analyses/plots/figure_{1..4}/`
- **Metrics**: `calculate_performance_metrics.py` — compute AUROC/MAE/QWK/etc. from predictions. 
- **Data from User Feedback Study**: CSVs under `code/analyses/validation_data/`.  
- **SHAP**  
  - Notebook: `code/fusion_model/shap_analysis.ipynb`  
  - Raw dump: `code/fusion_model/shap_raw.pkl` and `shap_outputs/shap_raw_test_trainbg.pkl`  
  - Panels: `shap_outputs/fig_shap_panels.{png,pdf}`


---

## Pretrained Models

The repo ships with ready‑to‑use checkpoints:

```
models/
  facial_expression_smile_best_auroc_baal/
  finger_model_both_hand_fusion_baal/
  fox_model_best_auroc_baal/
  uncertainty_aware_fusion/
```
Each contains a `scaler/`, `predictive_model/`, optional `residual_model/`, and a `model_config.json`. Use these for quick evaluation or as initialization for fine‑tuning.

---

## Reproducibility Tips

- **Pin** Python and library versions in your conda env (`environment.yml`) to match training.  
- **Seed** all RNGs in unimodal and fusion scripts where applicable.  
- **Freeze** CSVs in `data/` (train/val/test) before final runs.  
- **Track** Git commit hashes when generating artifacts in `code/analyses`.

---

## Citations
```
@inproceedings{islam2025accessible,
  title={Accessible, at-home detection of Parkinson’s disease via multi-task video analysis},
  author={Islam, Md Saiful and Adnan, Tariq and Freyberg, Jan and Lee, Sangwu and Abdelkader, Abdelrahman and Pawlik, Meghan and Schwartz, Cathe and Jaffe, Karen and Schneider, Ruth B and Dorsey, Ray and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={27},
  pages={28125--28133},
  year={2025}
}

@article{islam2025remote,
  title={Remote AI Screening for Parkinson's Disease: A Multimodal, Cross-Setting Validation Study},
  author={Islam, Md Saiful and Adnan, Tariq and Abdelkader, Abdelrahman and Liu, Zipei and Ma, Evelyn and Park, Sooyong and Azad, Asif and Liu, Pai and Pawlik, Meghan and Hartman, Emily and others},
  year={2025}
}
```
---

## License

Proprietary/Research use only unless otherwise specified by the project owner.  
For external use, please contact the maintainers to clarify licensing and data‑sharing constraints.
