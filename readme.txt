Intermediary Constraints and the Shape of the Yield Curve
==========================================================

This repository contains the data, code, and documentation for an empirical study of how 
intermediary frictions affect the U.S. Treasury yield curve. It implements regression 
frameworks and trading signal prototypes based on domestic and foreign constraint proxies.

----------------------------------------------------------
OVERVIEW
----------------------------------------------------------
In a frictionless, rational-expectations world, the slope of the yield curve equals the expected 
path of short rates. In reality, intermediary balance-sheet and funding constraints distort this 
mapping. This project examines whether observable proxies for intermediary constraints explain 
U.S. Treasury term premia and through which channels:

- Foreign constraints: deviations from Covered Interest Parity (CIP), measured by cross-currency basis.
- Domestic constraints: funding spreads such as GCF–Repo Survey and IORB–SOFR.
- Calendar frictions: end-of-month and quarter-end balance sheet effects.

----------------------------------------------------------
DATA & VARIABLES
----------------------------------------------------------
Dependent variable:
- ACM-style 10Y term premium (ACMY10).

Foreign constraint proxies:
- GBP/USD 3M basis
- EUR/USD 3M basis
- JPY/USD 3M basis

Domestic constraint proxies:
- GCF–Repo Survey spread
- IORB–SOFR spread

Controls:
- MOVE Index (implied Treasury volatility)
- Curve slopes (10Y–2Y, 2Y–1M)
- Calendar dummies (end-of-month, quarter-end)

----------------------------------------------------------
METHODOLOGY
----------------------------------------------------------
- Level regressions to identify long-run co-movement.
- First-difference regressions to capture short-run flow effects.
- Error-correction models (ECM) for domestic funding channel.
- Stationarity tests (ADF) and HAC/Newey–West inference.

----------------------------------------------------------
KEY FINDINGS
----------------------------------------------------------
1. Foreign channel:
   Persistent levels of cross-currency basis co-move with term premia; daily changes do not. 
   GBP basis is the strongest single proxy.

2. Domestic channel:
   GCF–Repo Survey spread loads strongly in the long run (≈ 15.75 bp term premium per 1 bp spread) 
   with a half-life of ~38 trading days.

3. Calendar effects:
   End-of-month effects are robust; quarter-end effects are weaker once EOM is included.

----------------------------------------------------------
REPOSITORY STRUCTURE
----------------------------------------------------------
data/                  - Raw/processed data or download scripts
notebooks/             - Jupyter notebooks
    variable_definitions.ipynb
    regressors.ipynb
    regression_main.ipynb
reports/               - PDF report & figures
    report.pdf
requirements.txt       - Python dependencies
README.txt             - This file

----------------------------------------------------------
INSTALLATION
----------------------------------------------------------
Clone the repository and install dependencies:

    git clone https://github.com/yourusername/intermediary-constraints.git
    cd intermediary-constraints
    pip install -r requirements.txt

----------------------------------------------------------
USAGE
----------------------------------------------------------
1. Open the notebooks in Jupyter Lab/Notebook:
       jupyter lab
2. Run variable_definitions.ipynb to load and describe the variables.
3. Run regression_main.ipynb for the full regression analysis.
4. View report.pdf for the compiled results and trading strategy prototypes.

----------------------------------------------------------
LICENSE
----------------------------------------------------------
MIT License – see LICENSE file for details.

----------------------------------------------------------
REFERENCES
----------------------------------------------------------
Du, W., Tepper, A., & Verdelhan, A. (2018). Deviations from Covered Interest Rate Parity.
Vayanos, D., & Vila, J.-L. (2021). A Preferred-Habitat Model of the Term Structure of Interest Rates.
Greenwood, R., & Vayanos, D. (2014). Bond Supply and Excess Bond Returns.
