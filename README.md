# Volatility Surface Toolkit

This project builds a research-grade volatility surface modelling toolkit for GLD options. It starts from raw option-chain data, computes Black-Scholes implied volatilities, diagnoses why a flat-volatility assumption fails, calibrates stochastic-volatility and Levy-process models, and then tests a deep-learning surface-completion model on the same benchmark.

The goal is not only to fit a surface. The goal is to compare model families under one reproducible pipeline and explain what each model captures, where it fails, and whether the added complexity is justified by the evidence.

## Research Question

Can richer volatility models improve on a flat Black-Scholes volatility benchmark for GLD options, and can a neural surface-completion model recover the observed implied volatility surface more accurately than classical parametric models?

## Main Results

The analysis uses a GLD option snapshot for July 18, 2025.

| Model | Main Result | Interpretation |
| --- | ---: | --- |
| Black-Scholes flat volatility | 8.32% IV-MAPE | Baseline fails because the GLD surface has smile/skew and term structure. |
| Heston stochastic volatility | 5.83% IV-MAPE | Improves flat vol by 29.9%, especially around ATM and moderate OTM strikes. |
| Heston + finite-activity jumps | No material improvement | Fitted jump intensity was close to zero on this strike range. |
| NIG Levy process | 8.22% IV-MAPE | Did not beat Heston on this dataset. |
| Variance Gamma Levy process | 8.37% IV-MAPE | Did not beat Heston on this dataset. |
| Bilateral Gamma Motion | 8.35% IV-MAPE | Did not beat Heston on this dataset. |
| ConvSurfaceNet deep model | 2.42% full observed IV-MAPE | Best surface fit on observed quotes; validation IV-MAPE was 2.91%. |

Key data and diagnostic facts:

- 880 GLD option quotes loaded.
- 877 valid implied volatilities computed, a 99.7% success rate.
- Spot price: 308.39.
- Risk-free rates interpolated across six maturity buckets, roughly 4.27% to 4.46%.
- ATM implied volatility: 15.29%.
- Best flat volatility: 16.20%.
- Term-structure slope: +2.47%.
- Short-dated skew: +2.69%.
- Worst flat-vol region: OTM puts, with 10.85% MAPE.

## Model Generations

### Generation 0: Black-Scholes Implied Volatility Baseline

The first notebook builds the baseline surface from GLD option quotes. It implements Black-Scholes pricing, implied-volatility inversion, data cleaning, treasury-rate interpolation, moneyness and maturity features, and diagnostic plots.

The implied-volatility solver uses Newton-Raphson with a bisection fallback. It successfully recovers implied volatility for 877 of 880 quotes.

### Generation 1: Heston and Stochastic Volatility with Jumps

The Heston model is calibrated with Carr-Madan FFT pricing and L-BFGS-B optimization. The fitted Heston surface reduces overall IV-MAPE from 8.32% to 5.83%.

The calibrated parameters are economically informative: the fitted correlation is positive for GLD, unlike the typical negative equity-index leverage effect. This is consistent with gold's two-sided tail behavior, where both safe-haven demand and supply shocks can matter.

The stochastic-volatility-with-jumps extension adds finite-activity compound Poisson jumps. On this GLD snapshot, the fitted jump component is effectively inactive, so the model does not improve on Heston.

### Generation 2: Levy Processes

The toolkit implements three infinite-activity Levy models:

- Normal Inverse Gaussian (NIG)
- Variance Gamma (VG)
- Bilateral Gamma Motion (BGM)

Each model is priced through a shared Carr-Madan FFT engine. The risk-neutral measure is handled through an Esscher transform. Calibration uses global search with differential evolution followed by local L-BFGS-B polishing.

The Levy models are theoretically attractive for short-dated skew and heavy tails, but this specific GLD strike range is narrow. The optimizers push the fitted models toward diffusion-like limits, and none of the Levy models beats Heston. This negative result is part of the research contribution: the richer model family is not automatically better when the data does not identify the additional tail parameters.

### Generation 3: Deep Learning Surface Completion

The deep-learning component implements `ConvSurfaceNet`, a PyTorch Conv2D encoder-decoder inspired by ConvLSTM surface modelling ideas.

Important scope note: this model is trained on a single option snapshot as a spatial surface-completion model. Temporal LSTM forecasting is left for future multi-day surface data.

The model uses:

- A 40 x 11 strike-maturity grid.
- A two-channel input: observed implied volatility grid plus binary observation mask.
- 296 observed grid cells.
- 237 training cells and 59 validation cells.
- 58.45M trainable parameters.
- Masked MSE loss so only observed cells drive fit.
- Laplacian smoothness regularization to encourage realistic surfaces.
- 500 training epochs with best-validation-loss restoration.

Performance:

- Best validation loss: 0.000034.
- Validation IV-MAPE: 2.91%.
- Validation RMSE: 0.0058.
- Full observed-surface IV-MAPE: 2.42%.
- Full observed RMSE: 0.0050.

## Repository Structure

```text
volatility-surface-toolkit/
├── data/                         # Input and benchmark GLD option datasets
├── notebooks/                    # End-to-end research notebooks
│   ├── 01_gld_surface_bs_failure.ipynb
│   ├── 02_heston_calibration.ipynb
│   ├── 03_levy_calibration.ipynb
│   └── 04_dl_calibration.ipynb
├── output/                       # Generated figures and LaTeX report source
├── report/                       # Final course presentation PDFs
├── scripts/                      # Data-fetching and diagnostic scripts
├── src/volsurf/                  # Importable Python package
│   ├── data/
│   ├── models/
│   │   ├── gen1_sv/              # Heston and stochastic volatility with jumps
│   │   ├── gen2_levy/            # NIG, VG, and BGM Levy models
│   │   └── gen3_dl/              # ConvSurfaceNet deep-learning model
│   ├── pricing/                  # Black-Scholes and FFT pricing engines
│   ├── utils/
│   └── visualization/
├── tests/                        # Unit tests for pricing and model components
├── environment.yml
├── pyproject.toml
└── README.md
```

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate volsurf
pip install -e .
```

Using pip:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

The package requires Python 3.11 or later. The development environment was built with Python 3.12.

## Reproducing the Analysis

Run the notebooks in this order:

1. `notebooks/01_gld_surface_bs_failure.ipynb`
2. `notebooks/02_heston_calibration.ipynb`
3. `notebooks/03_levy_calibration.ipynb`
4. `notebooks/04_dl_calibration.ipynb`

The notebooks are designed as a research sequence:

- Notebook 01 creates the Black-Scholes implied-volatility benchmark and surface diagnostics.
- Notebook 02 calibrates Heston and stochastic-volatility jump models.
- Notebook 03 calibrates NIG, VG, and BGM Levy-process models.
- Notebook 04 trains and evaluates the ConvSurfaceNet deep-learning surface-completion model.

## Tests

Run the test suite with:

```bash
pytest
```

The tests cover:

- Black-Scholes pricing and implied-volatility behavior.
- Heston model calibration components.
- Levy characteristic functions and pricing helpers.
- ConvSurfaceNet forward shapes, batch consistency, masked-loss behavior, grid preparation, overfitting checks, and prediction output shape.

## Selected Figures

The repository includes generated figures in `output/` and `notebooks/output/`, including:

- 3D implied-volatility surface plots.
- 2D surface contours.
- Black-Scholes flat-volatility deviations.
- Heston surface and fit diagnostics.
- Levy model fit comparisons.
- ConvSurfaceNet reconstructed surface, IV fit, and training curve.

## What This Project Demonstrates

This project demonstrates:

- Financial modelling of option-implied volatility surfaces.
- Statistical analysis of smile, skew, term structure, and pricing errors.
- Model comparison across Black-Scholes, Heston, SVJ, Levy processes, and deep learning.
- FFT option pricing with characteristic functions.
- Calibration with local and global optimization.
- PyTorch deep-learning model design for sparse financial surfaces.
- Dataset evaluation, train/validation splitting, and unit-tested research code.
- Clear communication of positive and negative model results.

## Limitations and Future Work

The deep-learning model is a spatial surface-completion model trained on one GLD option snapshot. A true ConvLSTM forecasting model would require a panel of option surfaces across multiple dates.

The Levy models did not outperform Heston on this dataset, largely because the observed GLD strike range was not wide enough to identify extreme tail behavior. A broader option chain or multi-date panel could provide a stronger test of infinite-activity jump models.

Future extensions:

- Multi-date surface panel construction.
- True temporal ConvLSTM forecasting.
- Cross-asset testing on equity-index, FX, rates, and commodity options.
- No-arbitrage surface constraints.
- Model-risk reporting by moneyness and maturity bucket.
