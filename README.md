# Student Outcome Bayesian Modeling

This repository contains a Bayesian multinomial logistic regression analysis of student academic outcomes (dropout, enrolled, graduate) using the UCI Students’ Dropout and Academic Success dataset.

## Project Structure
```
student-outcome-bayes/
├── data/
│   └── data.csv             # Raw dataset
├── notebooks/
│   └── analysis.ipynb       # Jupyter notebook with step-by-step modeling
├── scripts/
│   ├── download_data.py     # Script to download the dataset
├── model.py                 # Python script for model fitting and diagnostics
├── environment.yml          # Conda environment file
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Dataset
The data comes from the UCI Machine Learning Repository:

> Predict Students’ Dropout and Academic Success: archive.ics.uci.edu/ml/datasets/Student+Performance

- **Size:** 4,424 records with 36 features
- **Outcome:** 3-class target (`Dropout`, `Enrolled`, `Graduate`)
- **Features:** Demographics, socio-economic status, prior grades, first-year performance metrics.

Place `data.csv` in the `data/` folder (it should remain semicolon-delimited) or download it automatically using 
```bash
python scripts/download_data.py
```
- **Note:** The dataset is already included in the `data/` folder for convenience.

## Installation

Use either **pip** or **conda** to install dependencies.

### pip

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### conda

```bash
conda env create -f environment.yml
conda activate student-model
```

## Usage

### Jupyter Notebook

1. Launch a notebook server:
   ```bash
   jupyter lab
   ```
2. Open `notebooks/analysis.ipynb` and run all cells.

### Script

```bash
python model.py
```

This will:
- Load and preprocess the data
- Fit hierarchical, flat, and extended Bayesian multinomial models
- Generate diagnostics (trace plots, PPC, LOO comparisons)
- Save posterior and posterior-predictive traces in NetCDF files


## Results
Saved NetCDF files in the `results/` directory:
- `hierarchical_model_results.nc`
- `flat_model_results.nc`
- `extended_model_results.nc`
- Corresponding `_full.nc` files for posterior-predictive groups

## Citation

If you use this work, please cite:

> Eromonsele Okojie, (2025). Bayesian Multinomial Logistic Regression for Student Outcomes. GitHub repository.  
> UCI Machine Learning Repository: Predict Students’ Dropout and Academic Success.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

