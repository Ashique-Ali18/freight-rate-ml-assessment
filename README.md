# Freight Rate Prediction — Machine Learning Assessment

Machine learning solution for predicting freight load rates from shipment, route, equipment, and operational features.

## Overview

The objective is to predict `posted_rate` for unseen freight loads using historical labeled development data.

The solution includes:

- Exploratory data analysis and data-quality checks
- Feature engineering and preprocessing
- Time-based validation
- Model comparison and hyperparameter tuning
- Final model training on all labeled data
- Prediction of 12,000 validation loads
- Fixed December 2025 prediction generation
- Official output validation using the provided scoring script

## Dataset

The development dataset contains **48,000 labeled loads** from January–October 2025.

The final validation dataset contains **12,000 loads** from November–December 2025.

Important features include:

- Pickup and delivery locations
- Pickup and delivery coordinates
- Distance
- Equipment type
- Weight
- Date
- Market index
- Quote signal

Target:

```text
posted_rate