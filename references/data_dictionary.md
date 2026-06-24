# Data Dictionary

## Overview

This document describes the features and target variable contained in the **Polish Companies Bankruptcy Prediction Dataset**.

The dataset consists of **64 financial ratios and indicators** (`X1`–`X64`) that capture different aspects of a company's financial health, including profitability, liquidity, leverage, efficiency, and capital structure.

---

## Feature Descriptions

| Feature | Description                                                                                                         |
| ------- | ------------------------------------------------------------------------------------------------------------------- |
| X1      | Net profit / Total assets                                                                                           |
| X2      | Total liabilities / Total assets                                                                                    |
| X3      | Working capital / Total assets                                                                                      |
| X4      | Current assets / Short-term liabilities                                                                             |
| X5      | [(Cash + Short-term securities + Receivables − Short-term liabilities) / (Operating expenses − Depreciation)] × 365 |
| X6      | Retained earnings / Total assets                                                                                    |
| X7      | EBIT / Total assets                                                                                                 |
| X8      | Book value of equity / Total liabilities                                                                            |
| X9      | Sales / Total assets                                                                                                |
| X10     | Equity / Total assets                                                                                               |
| X11     | (Gross profit + Extraordinary items + Financial expenses) / Total assets                                            |
| X12     | Gross profit / Short-term liabilities                                                                               |
| X13     | (Gross profit + Depreciation) / Sales                                                                               |
| X14     | (Gross profit + Interest) / Total assets                                                                            |
| X15     | (Total liabilities × 365) / (Gross profit + Depreciation)                                                           |
| X16     | (Gross profit + Depreciation) / Total liabilities                                                                   |
| X17     | Total assets / Total liabilities                                                                                    |
| X18     | Gross profit / Total assets                                                                                         |
| X19     | Gross profit / Sales                                                                                                |
| X20     | (Inventory × 365) / Sales                                                                                           |
| X21     | Sales(n) / Sales(n−1)                                                                                               |
| X22     | Profit on operating activities / Total assets                                                                       |
| X23     | Net profit / Sales                                                                                                  |
| X24     | Gross profit (in 3 years) / Total assets                                                                            |
| X25     | (Equity − Share capital) / Total assets                                                                             |
| X26     | (Net profit + Depreciation) / Total liabilities                                                                     |
| X27     | Profit on operating activities / Financial expenses                                                                 |
| X28     | Working capital / Fixed assets                                                                                      |
| X29     | Logarithm of total assets                                                                                           |
| X30     | (Total liabilities − Cash) / Sales                                                                                  |
| X31     | (Gross profit + Interest) / Sales                                                                                   |
| X32     | (Current liabilities × 365) / Cost of products sold                                                                 |
| X33     | Operating expenses / Short-term liabilities                                                                         |
| X34     | Operating expenses / Total liabilities                                                                              |
| X35     | Profit on sales / Total assets                                                                                      |
| X36     | Total sales / Total assets                                                                                          |
| X37     | (Current assets − Inventories) / Long-term liabilities                                                              |
| X38     | Constant capital / Total assets                                                                                     |
| X39     | Profit on sales / Sales                                                                                             |
| X40     | (Current assets − Inventory − Receivables) / Short-term liabilities                                                 |
| X41     | Total liabilities / ((Profit on operating activities + Depreciation) × (12 / 365))                                  |
| X42     | Profit on operating activities / Sales                                                                              |
| X43     | Receivables rotation + Inventory turnover in days                                                                   |
| X44     | (Receivables × 365) / Sales                                                                                         |
| X45     | Net profit / Inventory                                                                                              |
| X46     | (Current assets − Inventory) / Short-term liabilities                                                               |
| X47     | (Inventory × 365) / Cost of products sold                                                                           |
| X48     | EBITDA (Profit on operating activities − Depreciation) / Total assets                                               |
| X49     | EBITDA (Profit on operating activities − Depreciation) / Sales                                                      |
| X50     | Current assets / Total liabilities                                                                                  |
| X51     | Short-term liabilities / Total assets                                                                               |
| X52     | (Short-term liabilities × 365) / Cost of products sold                                                              |
| X53     | Equity / Fixed assets                                                                                               |
| X54     | Constant capital / Fixed assets                                                                                     |
| X55     | Working capital                                                                                                     |
| X56     | (Sales − Cost of products sold) / Sales                                                                             |
| X57     | (Current assets − Inventory − Short-term liabilities) / (Sales − Gross profit − Depreciation)                       |
| X58     | Total costs / Total sales                                                                                           |
| X59     | Long-term liabilities / Equity                                                                                      |
| X60     | Sales / Inventory                                                                                                   |
| X61     | Sales / Receivables                                                                                                 |
| X62     | (Short-term liabilities × 365) / Sales                                                                              |
| X63     | Sales / Short-term liabilities                                                                                      |
| X64     | Sales / Fixed assets                                                                                                |

---

## Target Variable

The target variable indicates whether a company eventually went bankrupt.

| Variable | Description                      |
| -------- | -------------------------------- |
| Class    | Bankruptcy status of the company |

### Class Encoding

| Value | Meaning              |
| ----- | -------------------- |
| 0     | Non-bankrupt company |
| 1     | Bankrupt company     |

---

## Notes

* All predictor variables are numerical financial ratios.
* Features were derived from company financial statements.
* The dataset is intended for binary classification tasks.
* The target variable is highly imbalanced, with significantly fewer bankrupt companies than non-bankrupt companies.

