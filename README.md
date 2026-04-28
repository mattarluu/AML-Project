# Advanced Time Series Forecasting in Retail — Model Tournament
**M5 Forecasting Accuracy | Walmart Dataset | Univariate, Multivariate, Deep Learning & Foundation Models**

## Table of Contents
1. [Project Overview](#-project-overview)
2. [The Business Problem](#-the-business-problem)
3. [Dataset Description](#-dataset-description)
4. [Architecture & Methodology (Pipeline)](#%EF%B8%8F-architecture--methodology-pipeline)
5. [Model Tournament Results](#-model-tournament-results)
6. [Repository Structure](#-repository-structure)
7. [Installation & Reproducibility](#-installation--reproducibility)
8. [Technical Challenges Overcome](#-technical-challenges-overcome)
9. [Authors](#-authors)

---

## Project Overview
This repository contains an exhaustive comparative analysis (designed as a "Model Tournament") to forecast product demand in the retail sector. The project traces a mathematical and algorithmic evolution: starting from classical statistical methods (Holt-Winters, SARIMAX), scaling to traditional Machine Learning (LightGBM), and culminating in Deep Learning architectures (LSTM) and state-of-the-art Foundation Models (Amazon's Chronos).

**Main Research Question:** *Does the complexity and computational cost of Deep Learning justify its use over classical statistics for forecasting intermittent retail demand?*

---

## The Business Problem
Predicting daily sales at the item level in supermarkets is a highly complex challenge due to:
* **Intermittent Demand:** Products with zero-sales days (high variance and sparsity).
* **Strong Seasonality:** Weekly micro-cycles (weekend peaks) and annual macro-cycles.
* **Exogenous Sensitivity:** Drastic fluctuations caused by price changes, local/national holidays, and government economic subsidies (e.g., the SNAP program).

A forecasting error translates directly into **stockouts** (lost sales) or **excess inventory** (storage and expiration costs).

---

## Dataset Description
The official dataset from the **M5 Forecasting - Accuracy** competition (Kaggle / Walmart) was utilized.
* **Defined Scope:** For computational feasibility, the study focuses on the top 10 best-selling items in the `FOODS_3` department at the `CA_1` (California) store.
* **Forecasting Horizon:** 28 days (industry standard for short/medium-term supply chain planning).
* **Exogenous Variables Used:** Calendar events, cyclic time variables, price history, and SNAP subsidy days.

---

## Architecture & Methodology (Pipeline)

The workflow is divided into 6 strictly designed phases to prevent data leakage:

1. **Phase 1 (Validation Strategy):** Strict chronological split into `Train` (days 1-1885), `Validation` (1886-1913), and `Test` (1914-1941).
2. **Phase 2 (Baselines):** Implementation of Naïve and Seasonal Naïve models to establish the minimum error threshold to beat.
3. **Phase 3 (Univariate Modeling):** Classical autoregressive modeling (SARIMA and Holt-Winters) exploiting ACF and PACF functions.
4. **Phase 4 (Multivariate Modeling):** Injection of exogenous variables (SARIMAX) demonstrating the empirical impact of the economic context on consumption behavior.
5. **Phase 5 (Advanced ML & Deep Learning):**
   * **LightGBM:** Global cross-training with rich feature engineering (lagged variables, rolling windows).
   * **LSTM (PyTorch):** Massive Recurrent Neural Network trained *on-the-fly* with instance normalization to mitigate intermittency.
   * **Chronos (T5-Small):** *Zero-Shot* evaluation using Foundation Models adapted for time series forecasting.
6. **Phase 6 (Model Tournament):** Blind evaluation of all models on the untouched Test set.

---

## Model Tournament Results

Results obtained on the Test set (28-day forecast horizon) evaluating the RMSE (Root Mean Squared Error) metric:

| Rank | Model | RMSE | MAE | Computational Cost |
|:---:|:---|:---:|:---:|:---|
|  | **LSTM Global (PyTorch)** | 8.62| 5.44 | Very High (GPU) |
|  | **SARIMAX (Feature-Engineered)** | 8.98 | 5.55 | Medium (Local) |
|  | **SARIMA (1,1,1)(1,1,1,7)** | 9.06 | 5.66 | Low (Local) |
|  | **Holt-Winters (Add+Add)** | 9.10 | 5.81 | Low (Local) |
|  | **LightGBM (Global, Lag+Rolling)** | 9.49 | 6.22 | Medium (Global) |
|  | **Chronos (Zero-Shot, T5)** | 10.52 | 6.84 | High (Pre-trained) |
|  | **Seasonal Naïve (Baseline)** | 13.92| 8.48 | Immediate |

### Main Conclusion (The Trade-off)
The injection of expert knowledge (Feature Engineering of exogenous variables) into robust statistical models (**SARIMAX**) outperformed the brute force of Deep Learning on this specific subset. However, to scale to thousands of products simultaneously, **LightGBM** offers the best balance between production-grade accuracy and computational cost.

---

## Installation & Reproducibility

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mattarluu/AML-Project.git](https://github.com/mattarluu/AML-Project.git)
   cd AML-Project