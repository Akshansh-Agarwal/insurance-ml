# Insurance Claim Frequency Modeling with Neural Networks

A machine learning project developed during a one-week internship at a local insurance company to investigate whether neural networks could be used to estimate motor insurance claim risk from policy-level data.

The model predicts the **probability that a policy generates at least one claim**, providing a measure of claim frequency risk that can be used to segment policies by expected risk.

## Project Overview

Traditional insurance pricing models often rely on generalized linear models and manually defined rating factors. This project explored a neural-network-based approach capable of learning nonlinear relationships and interactions between policyholder, vehicle, coverage, and driver characteristics.

The project covered the full modeling pipeline:

* Preparing and validating policy-level insurance data
* Engineering policyholder, vehicle, geographic, and driver features
* Creating chronological training, validation, and test datasets
* Binning numerical and categorical variables using optimal binning
* Learning embedding representations for each feature
* Building and training a TensorFlow neural network
* Evaluating discrimination, calibration, and risk segmentation
* Saving and reloading the trained model for reproducibility

## Dataset

The modeling dataset contains approximately **175,000 motor insurance policies** written between November 2021 and September 2025.

The binary target is:

```python
claim_occurred = 1 if Claim Count > 0 else 0
```

The model therefore estimates:

```text
P(at least one claim | policy characteristics)
```

rather than directly predicting the number or severity of claims.

### Features

The final model uses **23 input features**, including:

* Policyholder age
* Gender
* Marital status
* Nationality
* No-claims discount
* Driving licence duration
* Vehicle age
* Vehicle type
* Vehicle colour
* Deductible
* Geographic region
* Lease year
* Additional-driver information

Several features relating to the policyholder and vehicle at policy inception were also included.

Geographic regions were derived from the first two digits of the available ZIP code.

## Train / Validation / Test Split

Insurance data is naturally time-dependent, so the dataset was split **chronologically rather than randomly** to provide a more realistic out-of-time evaluation.

| Dataset    | Share | Period               |
| ---------- | ----: | -------------------- |
| Training   |   70% | Nov 2021 to Dec 2024 |
| Validation |   15% | Dec 2024 to May 2025 |
| Test       |   15% | May 2025 to Sep 2025 |

All preprocessing that learns from the target was fitted using the training data only.

## Feature Processing

### Optimal Binning

Numerical and categorical variables were processed using `optbinning`.

Binning was used to:

* Reduce the number of sparse categories
* Group observations with similar claim behavior
* Reduce noise in continuous variables
* Produce manageable representations for the embedding layers

Rare categorical values representing less than 0.2% of the training dataset were pooled.

The binning process was fitted exclusively on the training dataset before being applied to validation and test data.

### Feature Embeddings

Each binned feature is passed through a TensorFlow `StringLookup` layer and mapped to a learned **4-dimensional embedding**.

Instead of treating each category as an unrelated one-hot encoded value, embeddings allow the neural network to learn compact numerical representations during training.

The embeddings from individual features are then concatenated before being passed through the main neural network.

### Additional Driver Handling

Additional-driver characteristics are only meaningful for policies that actually contain an additional driver.

A separate gating mechanism was therefore implemented:

```text
Additional-driver embeddings
          ↓
has_add_drv gate
          ↓
Included only when an additional driver exists
```

This prevents missing additional-driver information from being interpreted in the same way as genuine driver characteristics.

## Neural Network Architecture

The final architecture is:

```text
23 policy features
      ↓
StringLookup
      ↓
4-dimensional embedding per feature
      ↓
Additional-driver gating
      ↓
Concatenated feature representation
      ↓
Dense(64, ReLU)
      ↓
Dropout(20%)
      ↓
Dense(32, ReLU)
      ↓
Dense(1, Sigmoid)
      ↓
Predicted claim probability
```

The network contains approximately **8,800 trainable parameters**.

### Training Configuration

```text
Optimizer:       Adam
Learning rate:   0.001
Loss:            Binary Cross-Entropy
Batch size:      2048
Maximum epochs:  50
Early stopping:  4 epochs
```

Early stopping monitors validation loss and restores the weights from the best-performing epoch.

## Model Evaluation

Performance was evaluated on the held-out out-of-time test dataset.

| Metric      | Test Result |
| ----------- | ----------: |
| ROC AUC     |   **0.611** |
| PR AUC      |   **0.285** |
| Log Loss    |   **0.508** |
| Brier Score |   **0.165** |

The observed test-set claim rate was approximately **21.3%**, compared with an average model prediction of **24.6%**.

This indicates that the model was able to rank risk, although its absolute probabilities showed some overprediction on the later test period.

## Risk Segmentation

To test whether the model meaningfully separated low- and high-risk policies, test policies were divided into ten groups based on predicted claim probability.

| Predicted Risk Decile | Actual Claim Rate |
| --------------------: | ----------------: |
|        1, lowest risk |             10.9% |
|                     2 |             14.1% |
|                     3 |             16.2% |
|                     4 |             18.5% |
|                     5 |             20.5% |
|                     6 |             20.9% |
|                     7 |             23.5% |
|                     8 |             26.7% |
|                     9 |             28.5% |
|      10, highest risk |         **33.1%** |

The monotonic increase from approximately **11% to 33%** across predicted-risk deciles shows that the model was able to identify meaningful differences in claim propensity between policies.

## Technologies

* Python
* TensorFlow / Keras
* pandas
* NumPy
* scikit-learn
* OptBinning
* Jupyter Notebook

## Repository

The primary notebook contains the complete modeling workflow:

```text
Main.ipynb
```

The underlying insurer dataset is not included in the repository. The notebook currently references a local Parquet file, so the data-loading path must be updated before running it with another dataset.

## Limitations and Future Work

This project was completed within a one-week internship and was intended as an exploratory machine learning study rather than a production insurance pricing model.

Potential extensions include:

* Modeling claim counts directly using Poisson or negative-binomial objectives
* Separately modeling claim severity
* Combining frequency and severity into expected loss
* Improving probability calibration
* Comparing the neural network against GLM and gradient-boosted-tree baselines
* Expanding feature engineering
* Hyperparameter optimization
* Evaluating stability across underwriting periods and customer segments

## Key Takeaway

The project demonstrated an end-to-end approach for applying neural networks to insurance risk modeling. Despite the short development period, the model produced clear risk segmentation on an out-of-time test set, with observed claim rates increasing from approximately **11% in the lowest predicted-risk decile to 33% in the highest**.
