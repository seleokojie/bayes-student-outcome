#!/usr/bin/env python3
"""
model.py

Full Bayesian workflow for predicting student dropout, enrollment, and graduation
with hierarchical multinomial logistic regression (plus flat and extended models).
"""

# -----------------------------------------------------------------------------
# Bayesian Multinomial Logistic Regression for Student Outcomes
# Author: Eromonsele Okojie
# Date: 2025-04-20
# -----------------------------------------------------------------------------

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt



def load_data(path: Path):
    """
    Load and preprocess the raw CSV into a DataFrame with:
      - stripped column names
      - mapped target to integers 0/1/2
      - program index factorization
    """
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    df["Target"] = df["Target"].str.strip()
    outcome_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
    df["y"] = df["Target"].map(outcome_map)
    df["prog_idx"], programs = pd.factorize(df["Course"])
    return df, programs


def preprocess_features(df: pd.DataFrame):
    """
    Standardize numeric predictors, one-hot encode categoricals,
    build admission-grade z-score and program-level means.
    Returns:
      X            -- full design matrix (numpy array)
      adm_z        -- admission-grade as z-score (numpy array)
      other_idx    -- indices of all columns except admission-grade
      g_idx        -- program index for each student (numpy array)
      prog_mean_z  -- program-level mean admission z (numpy array)
    """
    # numeric columns to standardize
    num_cols = [
        "Admission grade",
        "Age at enrollment",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ]
    stds = df[num_cols].std().replace(0, 1)
    X_num = (df[num_cols] - df[num_cols].mean()) / stds

    # categorical columns to one-hot
    cat_cols = ["Gender", "Scholarship holder"]
    X_cat = pd.get_dummies(df[cat_cols].astype(str), drop_first=True)

    # full design matrix
    X = np.hstack([X_num.values, X_cat.values])
    n_students, n_preds = X.shape

    # admission-grade column index within X
    adm_idx = num_cols.index("Admission grade")
    adm_z = X[:, adm_idx]
    other_idx = np.delete(np.arange(n_preds), adm_idx)

    # program-level z-scores for admission means
    prog_mean_adm = df.groupby("prog_idx")["Admission grade"].mean().values
    prog_mean_z = (prog_mean_adm - prog_mean_adm.mean()) / prog_mean_adm.std()
    g_idx = df["prog_idx"].values

    return X, adm_z, other_idx, g_idx, prog_mean_z


def build_hierarchical_model(coords, X, adm_z, other_idx, g_idx, prog_mean_z):
    """Define the hierarchical multinomial model."""
    model = pm.Model(coords=coords)
    with model:
        # Hyperpriors for intercepts
        alpha0 = pm.Normal("alpha0", 0.0, 1.0)
        alpha1 = pm.Normal("alpha1", 0.0, 1.0)
        sigma_a = pm.HalfNormal("sigma_a", 1.0)

        # Hyperpriors for slopes on admission grade
        beta0 = pm.Normal("beta0", 0.0, 1.0)
        beta1 = pm.Normal("beta1", 0.0, 1.0)
        sigma_b = pm.HalfNormal("sigma_b", 1.0)

        # Raw program-level intercepts & transformed
        a_prog_raw = pm.Normal("a_prog_raw", 0.0, 1.0, shape=(len(prog_mean_z), 2))
        mu_a = alpha0 + alpha1 * prog_mean_z[:, None]
        a_prog = pm.Deterministic("a_prog", mu_a + sigma_a * a_prog_raw)

        # Raw program-level slopes on admission grade & transformed
        b_prog_raw = pm.Normal("b_prog_raw", 0.0, 1.0, shape=(len(prog_mean_z), 2))
        mu_b = beta0 + beta1 * prog_mean_z[:, None]
        b_prog_adm = pm.Deterministic("b_prog_adm", mu_b + sigma_b * b_prog_raw)

        # Fixed effects for all other predictors
        b_other = pm.Normal("b_other", 0.0, 1.0, shape=(other_idx.size, 2))

        # Global intercepts for the non-baseline categories
        a0 = pm.Normal("a0", mu=[-1.0, +0.5], sigma=1.0, shape=2)

        # Linear predictor
        η_nb = (
            a0[None, :]
            + a_prog[g_idx]
            + adm_z[:, None] * b_prog_adm[g_idx]
            + pt.dot(X[:, other_idx], b_other)
        )
        η = pt.concatenate([pt.zeros((η_nb.shape[0], 1)), η_nb], axis=1)
        p = pm.math.softmax(η, axis=1)

        # Observed categorical outcome
        pm.Categorical("y_obs", p=p, observed=df["y"].values, dims="obs")

    return model


def build_flat_model(coords, X, other_idx):
    """Define the flat (non-hierarchical) multinomial model."""
    model = pm.Model(coords=coords)
    with model:
        # Global intercepts & slopes
        a0_flat = pm.Normal("a0_flat", mu=0.0, sigma=2.0, shape=2)
        b_flat = pm.Normal("b_flat", mu=0.0, sigma=1.0, shape=(X.shape[1], 2))

        η_nb_flat = a0_flat[None, :] + pt.dot(X, b_flat)
        η_flat = pt.concatenate([pt.zeros((η_nb_flat.shape[0], 1)), η_nb_flat], axis=1)
        p_flat = pm.math.softmax(η_flat, axis=1)

        pm.Categorical("y_obs", p=p_flat, observed=df["y"].values, dims="obs")

    return model


def build_extended_model(coords, X, adm_z, other_idx, g_idx, prog_mean_z):
    """Define the extended model with random slopes on unemployment rate."""
    model = pm.Model(coords=coords)
    with model:
        # Hyperpriors for intercepts
        alpha0_e = pm.Normal("alpha0_e", 0.0, 1.0)
        alpha1_e = pm.Normal("alpha1_e", 0.0, 1.0)
        sigma_a_e = pm.HalfNormal("sigma_a_e", 1.0)

        # Hyperpriors for slopes on admission grade
        beta0_e = pm.Normal("beta0_e", 0.0, 1.0)
        beta1_e = pm.Normal("beta1_e", 0.0, 1.0)
        sigma_b_e = pm.HalfNormal("sigma_b_e", 1.0)

        # Program-level intercepts & admission-grade slopes
        a_prog_raw_e = pm.Normal("a_prog_raw_e", 0.0, 1.0, shape=(len(prog_mean_z), 2))
        mu_a_e = alpha0_e + alpha1_e * prog_mean_z[:, None]
        a_prog_e = mu_a_e + sigma_a_e * a_prog_raw_e

        b_prog_raw_adm_e = pm.Normal("b_prog_raw_adm_e", 0.0, 1.0, shape=(len(prog_mean_z), 2))
        mu_b_adm_e = beta0_e + beta1_e * prog_mean_z[:, None]
        b_prog_adm_e = mu_b_adm_e + sigma_b_e * b_prog_raw_adm_e

        # Random slopes on unemployment rate
        ur_idx = ["Admission grade",
                  "Age at enrollment",
                  "Curricular units 1st sem (grade)",
                  "Curricular units 2nd sem (grade)",
                  "Unemployment rate",
                  "Inflation rate",
                  "GDP"].index("Unemployment rate")
        ur_z = X[:, ur_idx]
        gamma0 = pm.Normal("gamma0", 0.0, 1.0)
        gamma1 = pm.Normal("gamma1", 0.0, 1.0)
        sigma_g = pm.HalfNormal("sigma_g", 1.0)
        g_prog_raw = pm.Normal("g_prog_raw", 0.0, 1.0, shape=(len(prog_mean_z), 2))
        mu_g = gamma0 + gamma1 * prog_mean_z[:, None]
        b_prog_ur = pm.Deterministic("b_prog_ur", mu_g + sigma_g * g_prog_raw)

        # Fixed effects for other predictors
        b_fix_e = pm.Normal("b_fix_e", 0.0, 1.0, shape=(other_idx.size, 2))
        a0_e = pm.Normal("a0_e", mu=[-1.0, +0.5], sigma=1.0, shape=2)

        # Linear predictor
        η_nb_e = (
            a0_e[None, :]
            + a_prog_e[g_idx]
            + adm_z[:, None] * b_prog_adm_e[g_idx]
            + ur_z[:, None] * b_prog_ur[g_idx]
            + pt.dot(X[:, other_idx], b_fix_e)
        )
        η_e = pt.concatenate([pt.zeros((η_nb_e.shape[0], 1)), η_nb_e], axis=1)
        p_e = pm.math.softmax(η_e, axis=1)

        pm.Categorical("y_obs", p=p_e, observed=df["y"].values, dims="obs")

    return model


def sample_model(model, **kwargs):
    """Run NUTS sampling on the given model and return the InferenceData."""
    with model:
        return pm.sample(
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
            **kwargs
        )


def main():
    # 0. Reproducibility & Paths
    np.random.seed(42)
    BASE = Path(__file__).parent
    data_path = BASE / "data" / "data.csv"

    # 1. Load & preprocess
    global df  # needed inside models
    df, programs = load_data(data_path)
    X, adm_z, other_idx, g_idx, prog_mean_z = preprocess_features(df)
    coords = {"obs": np.arange(len(df))}

    # 2a. Build & sample hierarchical model
    hier_model = build_hierarchical_model(coords, X, adm_z, other_idx, g_idx, prog_mean_z)
    hier_trace = sample_model(
        hier_model,
        draws=3000,
        tune=2000,
        init="adapt_diag",
        target_accept=0.95,
        cores=4,
        random_seed=42,
    )

    # posterior predictive for hierarchical
    with hier_model:
        ppc_hier = pm.sample_posterior_predictive(
            hier_trace,
            var_names=["y_obs"],
            random_seed=42,
            return_inferencedata=True
        )
    hier_trace.add_groups(posterior_predictive=ppc_hier.posterior_predictive)

    # 2b. Build & sample flat model
    flat_model = build_flat_model(coords, X, other_idx)
    flat_trace = sample_model(
        flat_model,
        draws=3000,
        tune=2000,
        init="adapt_diag",
        target_accept=0.95,
        cores=4,
        random_seed=42,
    )

    # 3. Diagnostics & LOO comparison
    az.plot_trace(hier_trace, var_names=["alpha0","alpha1","beta0","beta1","sigma_a","sigma_b"])
    plt.tight_layout()
    plt.show()

    hier_summary = az.summary(
        hier_trace,
        var_names=["alpha0","alpha1","beta0","beta1","sigma_a","sigma_b","a0","b_other"],
        round_to=2
    )
    print(hier_summary)

    az.plot_ppc(hier_trace, data_pairs={"y_obs":"y_obs"})
    plt.show()

    hier_loo = az.loo(hier_trace, scale="deviance")
    flat_loo = az.loo(flat_trace, scale="deviance")
    cmp = az.compare({"hierarchical": hier_loo, "flat": flat_loo})
    print(cmp)

    # 4. Program-level effects & predictive uncertainty
    prog_effects = az.summary(
        hier_trace,
        var_names=["a_prog","b_prog_adm"],
        hdi_prob=0.95
    )
    print(prog_effects.head(10))

    examples = [10, 100, 1000]
    posterior_probs = hier_trace.posterior_predictive["y_obs"].stack(sample=("chain","draw")).values
    for idx in examples:
        i = min(idx, posterior_probs.shape[1]-1)
        sims = posterior_probs[:, i]
        counts = np.bincount(sims, minlength=3) / sims.size
        ci = np.percentile((sims==2).astype(float), [2.5, 97.5])
        print(f"Student {idx}: proportions = {counts.round(2)}, grad CI = {ci.round(2)}")

    plt.figure()
    az.plot_forest(hier_trace, var_names=["b_prog_adm"], combined=True)
    plt.title("Forest of program-level admission slopes")
    plt.show()

    # 5. Build & sample extended model
    ext_model = build_extended_model(coords, X, adm_z, other_idx, g_idx, prog_mean_z)
    ext_trace = sample_model(
        ext_model,
        draws=3000,
        tune=2000,
        init="adapt_diag",
        target_accept=0.95,
        cores=4,
        random_seed=42,
    )

    # LOO for extended
    ext_loo = az.loo(ext_trace, scale="deviance")
    print(az.compare({"flat": flat_loo, "hier": hier_loo, "extended": ext_loo}))

    # 6. Save results
    az.to_netcdf(hier_trace, BASE / "hierarchical_model_results.nc", group="posterior")
    az.to_netcdf(flat_trace, BASE / "flat_model_results.nc", group="posterior")
    az.to_netcdf(ext_trace, BASE / "extended_model_results.nc", group="posterior")

    az.to_netcdf(hier_trace, BASE / "hierarchical_model_results_full.nc", group="posterior_predictive")
    az.to_netcdf(flat_trace, BASE / "flat_model_results_full.nc",        group="posterior_predictive")
    az.to_netcdf(ext_trace, BASE / "extended_model_results_full.nc",      group="posterior_predictive")


if __name__ == "__main__":
    # Required on Windows to support multiprocessing in PyMC
    from multiprocessing import freeze_support
    freeze_support()

    main()