# Polish Companies Bankruptcy Prediction Dataset

## Overview

This dataset is designed for bankruptcy prediction of Polish companies. It was collected from the Emerging Markets Information Service (EMIS), a database containing financial and market information on companies operating in emerging markets worldwide.

The dataset includes financial statements and financial ratios of Polish companies. Bankrupt companies were observed during the period **2000–2012**, while companies that remained operational were evaluated between **2007–2013**.

The target variable indicates whether a company went bankrupt within a specified forecasting horizon.

---

## Forecasting Scenarios

The dataset is divided into five classification cases based on the prediction horizon. Each case contains financial ratios from a specific year and a corresponding class label indicating future bankruptcy status.

| Dataset | Prediction Horizon | Total Instances | Bankrupt Companies | Non-Bankrupt Companies |
| ------- | ------------------ | --------------- | ------------------ | ---------------------- |
| 1stYear | 5 years ahead      | 7,027           | 271                | 6,756                  |
| 2ndYear | 4 years ahead      | 10,173          | 400                | 9,773                  |
| 3rdYear | 3 years ahead      | 10,503          | 495                | 10,008                 |
| 4thYear | 2 years ahead      | 9,792           | 515                | 9,277                  |
| 5thYear | 1 year ahead       | 5,910           | 410                | 5,500                  |

---

## Dataset Description

### 1stYear Dataset

Contains financial ratios from the first year of the forecasting period. The target label indicates whether the company became bankrupt within the following five years.

* Total instances: 7,027
* Bankrupt companies: 271
* Non-bankrupt companies: 6,756

### 2ndYear Dataset

Contains financial ratios from the second year of the forecasting period. The target label indicates whether the company became bankrupt within the following four years.

* Total instances: 10,173
* Bankrupt companies: 400
* Non-bankrupt companies: 9,773

### 3rdYear Dataset

Contains financial ratios from the third year of the forecasting period. The target label indicates whether the company became bankrupt within the following three years.

* Total instances: 10,503
* Bankrupt companies: 495
* Non-bankrupt companies: 10,008

### 4thYear Dataset

Contains financial ratios from the fourth year of the forecasting period. The target label indicates whether the company became bankrupt within the following two years.

* Total instances: 9,792
* Bankrupt companies: 515
* Non-bankrupt companies: 9,277

### 5thYear Dataset

Contains financial ratios from the fifth year of the forecasting period. The target label indicates whether the company became bankrupt within the following one year.

* Total instances: 5,910
* Bankrupt companies: 410
* Non-bankrupt companies: 5,500

---

## Target Variable

| Value | Description                  |
| ----- | ---------------------------- |
| 0     | Company remained operational |
| 1     | Company went bankrupt        |

---

## Notes

* The dataset is highly imbalanced, with bankrupt companies representing a small proportion of the observations.
* Financial ratios are provided as predictive features.
* Each forecasting scenario should be treated as an independent classification task.
* The dataset is commonly used for evaluating machine learning models in bankruptcy prediction and financial risk assessment.
