# Three-Dimensional Global Navigation Satellite System Coordinate Forecasting, Anomaly Screening and Reliability Assessment

**Using Statistical and Deep Learning Techniques at ONSA, Onsala, Sweden (1993–2026)**

Advanced Pattern Recognition — Master of Science in Data Science and Artificial Intelligence, South Asian University, November 2026.

**Authors:** Goutam Anand, Digvijay Singh and Shan Kuriakose.  
**Course instructor:** Professor Pritam Anand.

https://colab.research.google.com/drive/15wKpNDbfb7GtkbmmHpDm3mIH54rRrw7_?usp=sharing  

[Open the executed notebook in Google Colab](https://colab.research.google.com/github/goutamsaums/TIME-SERIES/blob/main/notebooks/ONSA_GNSS_Coordinate_Forecasting.ipynb) · [Project report](documents/ONSA_GNSS_Project_Report.pdf) · [Academic poster](documents/ONSA_GNSS_A0_Poster.pdf)

## Objective

We wanted to study whether the daily East, North and Up coordinate record can help us forecast the next position of a reference station and decide which newly received solutions require closer examination. We used prediction intervals alongside observation quality checks.

## Repository structure

```text
data/                       Original daily position and quality records; source notes
notebooks/                  Executed coordinate forecasting notebook (rerunnable)
src/                        Same Python workflow as a standalone script
results/
  Exploratory_Data_Analysis/  Descriptive plots and decomposition diagnostics
  Course_Task_Results/       Metrics, predictions, figures, saved models and review ledger
documents/                  Final A4 project report and A0 poster (PDF)
requirements.txt            Python dependencies
```

This repository contains the final coordinate forecasting experiment. The notebook retains the latest submitted Colab execution outputs.  The data folder contains only the two input datasets used in our experiment: `ONSA1.CSV` and `ONSA.qa.gz`, together with their source notes. 

## Run the project

1. Upload the contents of this package to the root of `goutamsaums/TIME-SERIES` on branch `main`. Keep the original case-sensitive filenames under `data/`.
2. Open the notebook using the Colab link above (or upload it manually to Colab). Select **Runtime → Run all**. CPU is sufficient; a GPU is optional.
3. The code retrieves `ONSA1.CSV` and `ONSA.qa.gz` from the repository using HTTPS. 
4. The notebook produces `ONSA_GNSS_Results.zip` and downloads it at the end. The `results/` directory contains the saved computational outputs from the preceding verified package.

For local execution with Python, from the repository root:

```bash
python -m pip install -r requirements.txt
python src/ONSA_GNSS_Coordinate_Forecasting.py
```


## Results

The archive contains 11,835 daily solutions over 12,349 calendar days (95.84% availability). Model development uses a 2,110-day continuous segment, split chronologically into 1,477 training, 158 model-selection, 158 calibration and 317 historical test observations.

| Model | Historical three-dimensional vector root mean square error (mm) |
|---|---:|
| Long short-term memory | 3.9190 |
| Recurrent neural network | 3.9387 |
| Autoregressive integrated moving average | 3.9520 |
| Autoregression | 4.0331 |
| Seasonal autoregressive integrated moving average | 4.0714 |
| Persistence | 4.1817 |
| Moving average | 9.9999 |

The best observed historical error is 6.28% below persistence. The jointly calibrated quantile networks cover 97.79% of 317 historical test dates and 94.95% of 5,560 later matched dates. Later calibrated persistence covers 94.78% with slightly narrower intervals. These small differences do not establish general model superiority.

