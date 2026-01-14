# Redlining Analysis: Detecting Discriminatory Lending Practices

A comprehensive data science project analyzing Home Mortgage Disclosure Act (HMDA) data to detect potential redlining and discriminatory lending practices in mortgage loan approvals across the United States (2020-2023).

## Overview

This project analyzes mortgage loan application data to identify potential discriminatory lending practices, commonly referred to as "redlining." Redlining historically refers to the practice of denying loans or charging higher rates to residents of certain neighborhoods based on their racial or ethnic composition, rather than individual creditworthiness.

**Key Objectives:**
- Detect potential racial, ethnic, and gender bias in loan approval decisions
- Analyze geographic patterns in loan approvals (census tract level)
- Evaluate fairness metrics across demographic groups
- Build predictive models while ensuring fairness
- Provide actionable insights for regulatory compliance

## Project Structure

```
Redlining/
│
├── Download_Data.ipynb                          # Data collection from CFPB API
├── Cleaning_Redlining_Analysis_Modelling.ipynb  # Data cleaning and preprocessing
├── EDA_Redlining_Analysis_Modelling.ipynb       # Exploratory analysis & ML modeling
├── requirements.txt                             # Python dependencies
└── README.md                                    # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- 8GB+ RAM recommended 

### Install Dependencies

Install all required packages using the requirements file:

```bash
pip install -r requirements.txt
```
