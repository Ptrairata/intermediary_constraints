# Intermediary Constraints and the Shape of the Yield Curve

[Full Report (PDF)](report/report.pdf)

This repository contains the data, code, and documentation for an empirical study of how 
intermediary frictions affect the U.S. Treasury yield curve. It implements regression 
frameworks and trading signal prototypes based on domestic and foreign constraint proxies.

---

## Overview

In a frictionless, rational-expectations world, the slope of the yield curve equals the expected 
path of short rates. In reality, intermediary balance-sheet and funding constraints distort this 
mapping. This project examines whether observable proxies for intermediary constraints explain 
U.S. Treasury term premia and through which channels:

- **Foreign constraints**: deviations from Covered Interest Parity (CIP), measured by cross-currency basis.  
- **Domestic constraints**: funding spreads such as GCF–Repo Survey and IORB–SOFR.  
- **Calendar frictions**: end-of-month and quarter-end balance sheet effects.  

---

## Data & Variables

**Dependent variable:**
- ACM-style 10Y term premium (ACMY10).  

**Foreign constraint proxies:**
- GBP/USD 3M basis  
- EUR/USD 3M basis  
- JPY/USD 3M basis  

**Domestic constraint proxies:**
- GCF–Repo Survey spread  
- IORB–SOFR spread  

**Controls:**
- MOVE Index (implied Treasury volatility)  
- Curve slopes (10Y–2Y, 2Y–1M)  
- Calendar dummies (end-of-month, quarter-end)  

---

## Methodology

- Level regressions to identify long-run co-movement.  
- First-difference regressions to capture short-run flow effects.  
- Error-correction models (ECM) for the domestic funding channel.  
- Stationarity tests (ADF) and HAC/Newey–West inference.  

---

## Key Findings

1. **Foreign channel**  
   Persistent levels of cross-currency basis co-move with term premia; daily changes do not.  
   GBP basis is the strongest single proxy.  

2. **Domestic channel**  
   GCF–Repo Survey spread loads strongly in the long run (≈ 15.75 bp term premium per 1 bp spread) 
   with a half-life of ~38 trading days.  

3. **Calendar effects**  
   End-of-month effects are robust; quarter-end effects are weaker once EOM is included.  

---

## Repository Structure

data/ - Raw/processed data or download scripts
notebooks/ - Jupyter notebooks
01_variable_definitions.ipynb
02_regressors.ipynb
03_regression_main.ipynb
reports/ - PDF report & figures
report.pdf
src/ - Helper functions
yc_frictions_tools.py
requirements.txt - Python dependencies
README.md - This file

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Ptrairata/intermediary_constraints.git
cd intermediary_constraints
pip install -r requirements.txt


---

## Option 2: Run online in Google Colab
No setup required – open notebooks directly in Colab:

- [Regression Main in Colab](https://colab.research.google.com/github/Ptrairata/intermediary_constraints/blob/main/notebooks/regression_main.ipynb)  
- [Regressors in Colab](https://colab.research.google.com/github/Ptrairata/intermediary_constraints/blob/main/notebooks/regressors.ipynb)  




