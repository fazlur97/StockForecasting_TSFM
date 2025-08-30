# Time Series Foundation Models for Stock Market Forecasting  

This repository contains the implementation of my MSc dissertation project: **evaluating Time Series Foundation Models (TSFMs) for forecasting communication-sector equities**.  

## Project Structure  

- **`preprocessing.ipynb`**  
  - Loads the raw CRSP-like dataset.  
  - Filters equities based on **SICCD ranges** relevant to the communication sector.  
  - Generates lagged features (lag-5).  
  - Computes rolling averages with windows **5, 21, 252, and 512 trading days**.  
  - Produces a **clean dataset** for model training and evaluation.  

- **`Chronos-T5.ipynb`, `TimeFM.ipynb`, `Uni2TS.ipynb`** (one notebook per TSFM model)  
  - Standardized pipeline for each model:  
    1. Load preprocessed dataset.  
    2. Generate **one-day-ahead forecasts** per stock (**PERMNO**) and per date.  
    3. Save predictions into a **CSV file** with columns `[date, PERMNO, prediction]`.  
    4. Compute evaluation metrics (R², MSE, MAE, Directional Accuracy).  

- **`resultCollect.ipynb`**  
  - Aggregates the per-model CSV predictions into a unified results table.  
  - Computes cross-model comparisons and portfolio-ready signals.  
  - Generates all thesis figures and tables, including (suggested):  
    - Cumulative return curves (gross and **turnover-adjusted** with 0.2% cost).  
    - Daily/annualized volatility and Sharpe-like ratios.  
    - Directional accuracy heatmaps (overall / up / down).  
    - Model rankings by window (5, 21, 252, 512).  
    - Final cumulative return tables per strategy (Long-Only, Short-Only, Long-Short, Top-Minus-Rest).  
  - Saves outputs to `Results/` (PNGs, PDFs, and CSV summaries).  

- **`Results/`**  
  - Stores model predictions and evaluation outputs (CSVs, plots, ranking tables).  

## Installation  

Clone this repository and install dependencies:  

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```

Key dependencies include:  
- `python>=3.10`  
- `torch>=2.0`  
- `pandas`, `numpy`, `matplotlib`  
- `gluonts` (for Chronos / Uni2TS)  
- `chronos-forecasting` (Amazon Chronos)  
- `timesfm` (Google TimeFM)  
- `transformers`, `pytorch-lightning` (for Uni2TS)  

## Workflow  

1. Run `preprocessing.ipynb` to generate the cleaned dataset.  
2. Run each model notebook (`Chronos-T5.ipynb`, `TimeFM.ipynb`, `Uni2TS.ipynb`) independently.  
3. Run `resultCollect.ipynb` to consolidate predictions and **produce all figures/tables** for the thesis.  
4. Collect forecasts and artifacts from `Results/` for downstream **portfolio analysis and backtesting**.  

## Limitations & Environment Notes  

Running TSFMs requires careful environment setup due to **dependency conflicts** and **hardware constraints**.  

### Chronos-T5 (Amazon)  
- Requires **Python ≥3.10**.  
- Depends on `chronos-forecasting`, `gluonts`, and PyTorch ≥2.0.  
- Runs slowly on **CPU**; CUDA acceleration strongly recommended.  
- Limited documentation compared to HuggingFace implementations.  

### TimeFM (Google)  
- Requires **Python 3.10+** and PyTorch ≥2.2.  
- Memory-intensive due to **large context length (default 2048)**.  
- GPU recommended; CPU inference is very slow.  
- Still an **experimental release**, APIs may change.  

### Uni2TS (Salesforce Moirai-MoE)  
- Requires **Python 3.9/3.10**.  
- Dependencies: `transformers`, `pytorch-lightning`, `gluonts`.  
- Model checkpoints are large; GPU inference required for feasible runtime.  
- Limited applied documentation, mostly research-level.  

### General Notes  
- All experiments were run on **Google Colab (Python 3.10.12, PyTorch 2.2.2)**.  
- Mixed precision (`torch.amp`) had inconsistent support across models.  
- Each TSFM required a **separate virtual environment** due to dependency conflicts:  
  - `venv_chronos`   
  - `venv_uni2ts`  
- Chronos and TimeFM can use the same python requirement
## Model References  

- **Chronos (Amazon)** → [github.com/amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)  
- **TimesFM (Google)** → [github.com/google-research/timesfm](https://github.com/google-research/timesfm)  
- **Uni2TS (Salesforce)** → [github.com/SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts)  
