# FallGuard-ENGG2112
CNN+LSTM Fall Detection System using CCTV Infrastructure
# FallGuard — CCTV Fall Detection System
ENGG2112 Engineering Project — University of Sydney, 2026

## Project Overview
A real-time fall detection system using CNN+LSTM deep learning 
on existing CCTV infrastructure. No wearables or new hardware required.

## Repository Structure
- `app.py` — Flask backend (loads model, runs inference)
- `index.html` — Web frontend (video upload interface)
- `requirements.txt` — Python dependencies
- `CNN_LSTM_withsaves.ipynb` — Temporal model training notebook
- `ENGG2112_clean.ipynb` — Image-level model comparison notebook

## How to Run the MVP
1. Download `model.h5` from Google Drive (link below)
2. Place it in the same folder as `app.py`
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python app.py`
5. Open browser at: `http://127.0.0.1:5000`

## Model Download
model.h5 (too large for GitHub):
[Download from Google Drive]((https://drive.google.com/drive/folders/1hk28XF04kuV_FeAEkbfMuAvW2WHzBmck?usp=sharing)

## Results Summary
| Model         | Accuracy | Fall Recall | AUC-ROC |
|---------------|----------|-------------|---------|
| Baseline CNN  | 63.55%   | 1.0000      | 0.7477  |
| MobileNetV2   | 88.79%   | 0.8529      | 0.9664  |
| EfficientNetB0| 84.11%   | 0.8529      | 0.9423  |
| CNN+LSTM      | —        | 0.9000      | 0.6503  |

## Team
Jannah Perwaiz · Felix Hibberd · Tarshith Bodha · Fevin Mohammed
