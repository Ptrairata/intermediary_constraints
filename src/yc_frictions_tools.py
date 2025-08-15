"""Utilities extracted from regressions.ipynb (auto-generated)."""

from IPython.display import display
from functools import reduce
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm
import sys

__all__ = ['adf_classify', 'adf_test', 'build_reg_data_for_dY', 'dlog_safe', 'engle_granger_test', 'fit_distributed_lag', 'make_lags', 'run_ecm', 'run_ols', 'tidy_results']

def run_ols(y, X, add_const=True, se="HC3", lags=None, cluster=None):
    """
    OLS with flexible robust standard errors.

    Parameters
    ----------
    y : array-like or pd.Series
    X : array-like or pd.DataFrame
    add_const : bool, add intercept
    se : str
        - "nonrobust"  -> classical
        - "HC0","HC1","HC2","HC3" -> heteroskedasticity-robust
        - "HAC"        -> Newey–West (needs lags)
        - "cluster"    -> cluster-robust (needs cluster)
    lags : int, for HAC/Newey–West
    cluster : array-like, grouping labels for clustering (len == n_obs)

    Returns
    -------
    res : statsmodels.regression.linear_model.RegressionResultsWrapper
          Fitted model with chosen covariance.
    """
    if add_const:
        X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X).fit()

    se = se.upper()
    if se in {"HC0","HC1","HC2","HC3"}:
        res = model.get_robustcov_results(cov_type=se)
    elif se == "HAC":
        if lags is None:
            raise ValueError("For HAC/Newey–West, provide `lags` (e.g., 4 or 12).")
        res = model.get_robustcov_results(
            cov_type="HAC",
            use_correction=True,
            maxlags=lags
        )
    elif se == "CLUSTER":
        if cluster is None:
            raise ValueError("For cluster-robust SEs, pass `cluster` group labels.")
        res = model.get_robustcov_results(
            cov_type="cluster",
            groups=cluster
        )
    elif se == "NONROBUST":
        res = model
    else:
        raise ValueError(f"Unknown se='{se}'.")
    return res

def tidy_results(res):
    """Return a neat coefficient table as a DataFrame."""
    out = pd.DataFrame({
        "coef": res.params,
        "std_err": res.bse,
        "t": res.tvalues,
        "p": res.pvalues
    })
    out["[2.5%, 97.5%]"] = list(zip(*res.conf_int().to_numpy().T))
    return out

def adf_test(series, name=''):
    result = adfuller(series.dropna(), autolag='AIC')
    labels = ['ADF Statistic', 'p-value', '# Lags Used', '# Observations']
    out = dict(zip(labels, result[0:4]))
    print(f'ADF Test on "{name}"')
    for key, val in out.items():
        print(f"   {key}: {val}")
    for key, val in result[4].items():
        print(f"   Critical Value ({key}): {val}")
    if result[1] <= 0.05:
        print("   => Reject H0: Series is stationary")
    else:
        print("   => Fail to reject H0: Series has unit root (nonstationary)")
    print()

def make_lags(df, col, K):
    for k in range(K+1):
        df[f'{col}_L{k}'] = df[col].shift(k)
    return df

def fit_distributed_lag(df, y_col, x_col, K=5, controls=None, hac_lags=6):
    df = df.sort_values('Date').copy()
    df[f'{y_col}_L1'] = df[y_col].shift(1)
    df = make_lags(df, x_col, K)

    beta_names = [f'{x_col}_L{k}' for k in range(K+1)]
    phi_name   = f'{y_col}_L1'
    X_cols = beta_names + [phi_name]
    if controls:
        X_cols += controls

    dfm = df.dropna(subset=[y_col] + X_cols).copy()
    X = sm.add_constant(dfm[X_cols], has_constant='add')
    y = dfm[y_col]

    ols = sm.OLS(y, X).fit()
    res = ols.get_robustcov_results(cov_type='HAC', maxlags=hac_lags, use_correction=True)

    # --- Make named params/cov ---
    param_names = res.model.exog_names  # includes 'const'
    params = pd.Series(res.params, index=param_names)
    V = res.cov_params()
    if not isinstance(V, pd.DataFrame) or list(V.index) != param_names:
        V = pd.DataFrame(V, index=param_names, columns=param_names)

    # --- Long-run multiplier L = (sum β_j) / (1-φ) ---
    beta_hat = params[beta_names].sum()
    phi_hat  = params[phi_name]
    L = beta_hat / (1.0 - phi_hat)

    # Delta method gradient
    g = pd.Series(0.0, index=param_names)
    g[beta_names] = 1.0 / (1.0 - phi_hat)
    g[phi_name]   = beta_hat / (1.0 - phi_hat)**2
    var_L = float(g.values @ V.values @ g.values)
    se_L  = np.sqrt(max(var_L, 0.0))
    t_L   = L / se_L if se_L > 0 else np.nan

    # --- Wald test: all β_lags = 0 jointly ---
    R = np.zeros((len(beta_names), len(param_names)))
    for i, b in enumerate(beta_names):
        R[i, param_names.index(b)] = 1.0
    r = np.zeros(len(beta_names))
    wald = res.wald_test((R, r), use_f=True)  # F-stat with HAC

    out = {
        "results": res,
        "long_run_multiplier": L,
        "long_run_se": se_L,
        "long_run_t": t_L,
        "beta_sum": beta_hat,
        "phi": phi_hat,
        "wald_F": float(wald.fvalue),
        "wald_p": float(wald.pvalue),
        "used_columns": ['const'] + X_cols
    }
    return out

def adf_classify(series: pd.Series, name: str, alpha: float = 0.05, autolag: str = "AIC"):
    s = series.dropna()
    if len(s) < 20:
        return {"Variable": name, "ADF stat": np.nan, "p-value": np.nan,
                "Lags": np.nan, "Obs": len(s), "Order": "insufficient data"}
    stat, p, lags, nobs, crit, _ = adfuller(s, autolag=autolag)
    order = "I(0)" if (p <= alpha) else "I(1)"
    return {"Variable": name, "ADF stat": stat, "p-value": p,
            "Lags": lags, "Obs": nobs, "Order": order}

def build_reg_data_for_dY(proxy_var: str, include_vars=None, extra_exclude=None, main_df=[], transform_plan=[]):
    ex = set(extra_exclude or [])
    main_df["dY"] = main_df["ACMY10"].diff()
    cols = []

    if proxy_var == "GCF_survey":
        main_df["d_proxy_L0"] = main_df["GCF_survey"].diff()
        main_df["d_proxy_L1"] = main_df["GCF_survey"].diff().shift(1)
        cols += ["d_proxy_L0", "d_proxy_L1"]
    elif proxy_var == "Domestic_PC1":
        main_df["proxy_L0"] = main_df["Domestic_PC1"]
        main_df["proxy_L1"] = main_df["Domestic_PC1"].shift(1)
        cols += ["proxy_L0", "proxy_L1"]
    elif proxy_var == "Basis_PC1":
        main_df["proxy_L0"] = main_df["Basis_PC1"]
        main_df["proxy_L1"] = main_df["Basis_PC1"].shift(1)
        cols += ["proxy_L0", "proxy_L1"]
    elif proxy_var == "BPBS_3MO":
        main_df["d_proxy_L0"] = main_df["d_BPBS_3MO"]
        main_df["d_proxy_L1"] = main_df["d_BPBS_3MO"].shift(1)
        cols += ["d_proxy_L0", "d_proxy_L1"]
    elif proxy_var == "d_Domestic_PC1":
        # Δ-factor already; treat as shock with L0/L1 dynamics
        main_df["dproxy_L0"] = main_df["d_Domestic_PC1"]
        main_df["dproxy_L1"] = main_df["d_Domestic_PC1"].shift(1)
        cols += ["dproxy_L0", "dproxy_L1"]
    else:
        raise ValueError("proxy_var must be 'GCF_survey', 'Domestic_PC1', or 'Domestic_PC1_d'")

    tp = transform_plan.copy()
    if include_vars is not None:
        tp = tp[tp["Variable"].isin(include_vars)]
    if extra_exclude:
        tp = tp[~tp["Variable"].isin(ex)]

    cols += tp["NewCol"].tolist()
    seen = set(); cols = [c for c in cols if (c not in seen and not seen.add(c))]
    df = main_df[["dY"] + cols].dropna()
    return df, cols

def engle_granger_test(y_var, x_var, data, maxlag=None, autolag='AIC', alpha=0.05):
    """
    Runs Engle-Granger two-step cointegration test between y_var and x_var.
    y_var: dependent variable name (str)
    x_var: independent variable name (str)
    data: DataFrame containing both variables
    maxlag, autolag: passed to adfuller
    alpha: significance level for ADF
    
    Returns: dict with regression summary + ADF result on residuals
    """
    # Step 1: Run levels regression
    df = data[[y_var, x_var]].dropna()
    y = df[y_var]
    X = sm.add_constant(df[x_var])
    ols_res = sm.OLS(y, X).fit()
    
    # Step 2: ADF test on residuals
    resid = ols_res.resid
    adf_stat, p_value, lags, nobs, crit, _ = adfuller(resid, maxlag=maxlag, autolag=autolag)
    cointegrated = p_value <= alpha
    
    print("=== Step 1: Levels Regression ===")
    print(ols_res.summary())
    
    print("\n=== Step 2: ADF Test on Residuals ===")
    print(f"ADF statistic: {adf_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Lags used: {lags}, Observations: {nobs}")
    print(f"Critical values: {crit}")
    print(f"Cointegrated? {'YES' if cointegrated else 'NO'} at {alpha*100:.1f}% level")
    
    return {
        "regression": ols_res,
        "adf_stat": adf_stat,
        "p_value": p_value,
        "lags": lags,
        "nobs": nobs,
        "crit": crit,
        "cointegrated": cointegrated
    }

def run_ecm(y_var="ACMY10",
            x_var="BPBS_3MO",
            controls_whitelist=('10_2','2_1MO','IORB_SOFR','GCF_survey','MOVE','eom','eoq'),
            cov_type="HAC", hac_maxlags=5, main_df=[], transform_plan=[]):
    """
    Engle–Granger two-step ECM:
      Δy_t = α + γ·Δx_t + λ·ECT_{t-1} + δ'·Controls_t + ε_t
    ECT_{t-1} = (y_{t-1} - α̂ - θ̂ x_{t-1}) from the long-run regression.
    Controls are pulled from your transform_plan:
      - I(0) → level
      - I(1) → Δ (or Δlog for MOVE/VIX)
      - dummies (eom/eoq) → level
    """

    # 1) Long-run regression in levels (pairwise y on x)
    df_lr = main_df[[y_var, x_var]].dropna()
    lr = sm.OLS(df_lr[y_var], sm.add_constant(df_lr[x_var])).fit()
    main_df["ECT"] = lr.resid
    main_df["ECT_L1"] = main_df["ECT"].shift(1)

    # 2) Short-run variables
    main_df["dY"] = main_df[y_var].diff()
    main_df["dX"] = main_df[x_var].diff()

    # 3) Controls from transform_plan (respect whitelist and avoid y/x duplication)
    tp = transform_plan.copy()
    tp = tp[tp["Variable"].isin(controls_whitelist)]
    # Don't bring in y/x themselves from the plan
    tp = tp[~tp["Variable"].isin([y_var, x_var])]

    control_cols = tp["NewCol"].tolist()

    # 4) Assemble dataset
    X_cols = ["dX", "ECT_L1"] + control_cols
    df = main_df[["dY"] + X_cols].dropna()

    y = df["dY"]
    X = sm.add_constant(df[X_cols])

    # 5) Estimate ECM with robust (HAC) SEs
    if cov_type.upper() == "HAC":
        ecm = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})
    else:
        ecm = sm.OLS(y, X).fit()

    print("=== Long-run (levels) ===")
    print(lr.summary().tables[1])
    print("\n=== Error-Correction Model (short run) ===")
    print(ecm.summary())

    # 6) Report adjustment speed & half-life
    lam = ecm.params.get("ECT_L1", np.nan)
    if np.isfinite(lam) and (lam < 0) and (lam > -1):
        half_life = np.log(0.5)/np.log(1.0 + lam)
        print(f"\nSpeed of adjustment λ = {lam:.4f}  → half-life ≈ {half_life:.2f} periods")
    else:
        print(f"\nSpeed of adjustment λ = {lam:.4f}  → half-life not defined in (-1,0) range")

    # 7) Immediate short-run effect of Δx
    gamma = ecm.params.get("dX", np.nan)
    print(f"Short-run effect γ (Δ{x_var} → Δ{y_var}) = {gamma:.4f}")

    return {"long_run": lr, "ecm": ecm, "lambda": lam, "gamma": gamma}

def dlog_safe(x: pd.Series) -> pd.Series:
    # Guard against nonpositive values; treat them as NaN before log
    x_pos = x.where(x > 0)
    return np.log(x_pos).diff()



def run_ecm_with_lags(main_df=[], y_var='', x_var='', controls='', transform_plan=[],
                      x_lags=0, y_lags=0, cov_type="HAC", hac_maxlags=5):
    """
    ECM with optional short-run lags:
      Δy_t = α + Σ_{j=0}^{x_lags} β_j Δx_{t-j} + Σ_{i=1}^{y_lags} α_i Δy_{t-i}
             + λ·ECT_{t-1} + δ'Controls_t + ε_t
    ECT_{t-1} from levels regression y on x.
    """

    # --- Normalize inputs ---
    if isinstance(controls, str):
        controls = [c.strip() for c in controls.split(",") if c.strip()]
    else:
        controls = list(controls) if controls is not None else []

    # --- Long-run levels regression ---
    df_lr = main_df[[y_var, x_var]].dropna()
    if df_lr.empty:
        raise ValueError("No overlapping observations for y_var and x_var.")
    lr = sm.OLS(df_lr[y_var], sm.add_constant(df_lr[x_var])).fit()

    # Work on a copy
    main_df = main_df.copy()

    # Error-correction term (lagged residual)
    main_df["ECT_L1"] = lr.resid.reindex(main_df.index).shift(1)

    # Differences
    main_df["dY"] = main_df[y_var].diff()
    main_df["dX"] = main_df[x_var].diff()

    X_cols = []

    # ΔX lags (j = 0..x_lags)
    for j in range(x_lags + 1):
        col = f"dX_L{j}"
        main_df[col] = main_df["dX"].shift(j)
        X_cols.append(col)

    # ΔY lags (i = 1..y_lags)
    for i in range(1, y_lags + 1):
        col = f"dY_L{i}"
        main_df[col] = main_df["dY"].shift(i)
        X_cols.append(col)

    # Controls from transform_plan (levels for I(0), Δ/Δlog for I(1), dummies in level)
    # Exclude y_var/x_var from the control whitelist to avoid duplication
    tp = transform_plan.copy()
    tp = tp[~tp["Variable"].isin([y_var, x_var])]
    if controls:
        tp = tp[tp["Variable"].isin(controls)]
    control_cols = [c for c in tp["NewCol"].tolist() if isinstance(c, str) and c in main_df.columns]

    # Assemble final regressor list (deduplicated, order-preserving)
    X_cols += ["ECT_L1"] + control_cols
    seen = set(); X_cols = [c for c in X_cols if not (c in seen or seen.add(c))]

    # Build final dataset
    df = main_df[["dY"] + X_cols].dropna()
    if df.empty:
        raise ValueError("After differencing/lagging/dropping NAs, no rows remain.")

    y = df["dY"]
    X = sm.add_constant(df[X_cols], has_constant="add")

    # Fit with requested covariance
    if (cov_type or "HAC").upper() == "HAC":
        ecm = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})
    else:
        ecm = sm.OLS(y, X).fit()

    # Report
    print(lr.summary().tables[1])
    print(ecm.summary())

    beta_sum = sum(ecm.params.get(f"dX_L{j}", 0.0) for j in range(x_lags + 1))
    lam = ecm.params.get("ECT_L1", np.nan)
    print(f"\nShort-run ΔX cumulative effect (Σβ_j): {beta_sum:.4f}")
    if np.isfinite(lam) and -1 < lam < 0:
        hl = np.log(0.5) / np.log(1 + lam)
        print(f"Speed λ = {lam:.4f} → half-life ≈ {hl:.2f} periods")

    return {"long_run": lr, "ecm": ecm, "beta_sum": beta_sum, "lambda": lam, "X_cols": X_cols}
