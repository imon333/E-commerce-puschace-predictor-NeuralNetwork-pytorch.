# E-Commerce Purchase Predictor — Neural Network with PyTorch

## Business Problem

**ShopSmart** is a fast-growing international e-commerce platform. Their marketing team wants to predict which customers are most likely to make a purchase based on their browsing behaviour.

By predicting purchase intent, ShopSmart can:
- Show targeted promotions only to likely buyers
- Reduce wasted marketing spend
- Increase conversion rate and revenue

---

## Project Structure

```
ecommerce-purchase-predictor-NeuralNetwork-pytorch/
│
├── data/
│   ├── raw_customer_data.csv       ← raw data with missing values
│   ├── clean_data.csv              ← after cleaning
│   ├── model_features.csv          ← after feature engineering
│   ├── train_features.csv          ← 80% training set
│   ├── validation_features.csv     ← 20% validation set
│   └── validation_predictions.csv  ← final predictions
│
├── notebooks/
│   └── purchase_prediction.ipynb   ← full project notebook
│
├── src/
│   └── model.py                    ← reusable model class
│
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

```
raw_customer_data.csv
        ↓
Step 1: Load & Inspect    → understand the data
        ↓
Step 2: Clean             → handle missing values
        ↓
Step 3: EDA               → explore patterns
        ↓
Step 4: Feature Engineering → scale + encode
        ↓
Step 5: Train/Val Split   → 80% / 20%
        ↓
Step 6: Build Neural Network
        ↓
Step 7: Train (150 epochs)
        ↓
Step 8: Evaluate (accuracy, confusion matrix)
        ↓
Step 9: Save Predictions
```

---

## Model Architecture

```
Input Layer   →   Hidden Layer 1   →   Hidden Layer 2   →   Output
(n features)      (16 neurons)         (8 neurons)          (1 neuron)
                  ReLU + Dropout        ReLU                 Sigmoid
```

- **Loss function:** Binary Cross Entropy (BCELoss)
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 150
- **Regularization:** Dropout (0.2)

---

## Dataset

| Column | Type | Description |
|---|---|---|
| customer_id | Integer | Unique identifier |
| age | Integer | Customer age |
| time_spent | Float | Minutes on website per session |
| pages_viewed | Integer | Pages viewed per session |
| basket_value | Float | Value of items in basket |
| device_type | String | Mobile, Desktop, Tablet |
| customer_type | String | New, Returning |
| category | String | Fashion, Electronics, Home Goods |
| num_sessions | Integer | Number of sessions |
| purchase | Binary | 1 = bought, 0 = did not buy |

---

## Installation

```bash
pip install -r requirements.txt
jupyter notebook notebooks/purchase_prediction.ipynb
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
jupyter
```

---

## Key Results

- Basket value and time spent are the strongest predictors of purchase
- Returning customers buy at a significantly higher rate than new customers
- The model achieves strong accuracy on unseen validation data

---

## Skills Demonstrated

- Data cleaning and validation
- Exploratory Data Analysis (EDA)
- Feature engineering (scaling, one-hot encoding)
- Neural network design with PyTorch
- Model training and evaluation
- Binary classification

---

## Author

Inspired from  the AI Engineer for Data Scientists Associate certification (DataCamp). and another project has been done by IMON.
