import numpy as np
import pickle
import os

# import lenstronomy and hierArc modules
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource

from defaults import *


# ### Input deflector population hyperparameters
lambda_int_mean_true = 1.0  # input truth sample mean internal MST transform


lambda_int_distribution = "GAUSSIAN"  # "delta" or "GAUSSIAN", single-valued or Gaussian distribution in the internal MST transform on top of a power-law model
kwargs_lens_true = {"lambda_mst": lambda_int_mean_true}
if lambda_int_distribution == "GAUSSIAN":
    kwargs_lens_true["lambda_mst_sigma"] = lambda_int_sigma_true

kappa_ext_distribution = [
    "GAUSSIAN"
]  # "delta" or "GAUSSIAN", single-valued or Gaussian distribution in the line-of-sight convergence


# =========================
# SNe population parameters
# =========================

# we add the apparent source magnitude at z=0.1 (pivot normalization) (mean of Gaussian distribution in astronomical magnitudes)
z_apparent_m_anchor = 0.1

mu_sne_true = roman_apparent_m_p_true
sigma_sne_true = 0.1  # 1-sigma with of Gaussian in astronomical magnitude
kwargs_source_true = {"mu_sne": mu_sne_true, "sigma_sne": sigma_sne_true}

# initial start and hard bounds for source and lens hyper parameters:
kwargs_source_true = kwargs_source_true  # mean start particles, for simplicity we initalize it at the input truth
kwargs_source_sigma = {"mu_sne": 0.1, "sigma_sne": 0.05}  # width of start particles
kwargs_source_lower = {
    "mu_sne": mu_sne_true - 20,
    "sigma_sne": 0.0,
}  # lower bounds on cosmology parameters
kwargs_source_upper = {
    "mu_sne": mu_sne_true + 20,
    "sigma_sne": 2,
}  # upper bounds on cosmology parameters

kwargs_lens_sigma = {"lambda_mst": 0.1, "lambda_mst_sigma": lambda_int_sigma_true}
kwargs_lens_lower = {"lambda_mst": 0.5, "lambda_mst_sigma": 0}
kwargs_lens_upper = {"lambda_mst": 1.5, "lambda_mst_sigma": 0.5}

kwargs_mean_start = {
    "kwargs_cosmo": kwargs_cosmo_true,
    "kwargs_source": kwargs_source_true,
    "kwargs_lens": kwargs_lens_true,
    "kwargs_los": kwargs_los_true,
}
kwargs_sigma_start = {
    "kwargs_cosmo": kwargs_cosmo_sigma,
    "kwargs_source": kwargs_source_sigma,
    "kwargs_lens": kwargs_lens_sigma,
    "kwargs_los": kwargs_los_sigma,
}


# ===============================
# importing Roman forecast
# ===============================

roman_likelihood_file = os.path.join(storage_directory, "roman_sne.pkl")

with open(roman_likelihood_file, "rb") as f:
    kwargs_sne_likelihood_roman = pickle.load(f)


# ==================================================================
# load likelihood configuration from Birrer, Dhawan & Shajib (2022)
# ==================================================================

file_likelihood_realistic_microlensing = os.path.join(
    storage_directory, "likelihood_02_mag.pkl"
)

with open(file_likelihood_realistic_microlensing, "rb") as f:
    kwargs_likelihood_list_lensed_sn = pickle.load(f)

kwargs_bounds_lensed_sl = {
    "kwargs_lower_cosmo": kwargs_cosmo_lower,
    "kwargs_upper_cosmo": kwargs_cosmo_upper,
    "kwargs_fixed_cosmo": {},
    "kwargs_lower_source": kwargs_source_lower,
    "kwargs_upper_source": kwargs_source_upper,
    "kwargs_fixed_source": {"sigma_sne": sigma_sne_true},
    "kwargs_lower_lens": kwargs_lens_lower,
    "kwargs_upper_lens": kwargs_lens_upper,
    "kwargs_fixed_lens": {
        "lambda_mst_sigma": lambda_int_sigma_true,
    },
    "kwargs_lower_los": kwargs_los_lower,
    "kwargs_upper_los": kwargs_los_upper,
    "kwargs_fixed_los": kwargs_los_fixed,
    "kwargs_lower_kin": {},
    "kwargs_upper_kin": {},
    "kwargs_fixed_kin": {},
}


# ==================================
# joint options for hierArc sampling
# ==================================

kwargs_sampler_lensed_sn = {
    "cosmology": cosmology,
    "kwargs_bounds": kwargs_bounds_lensed_sl,
    "lambda_mst_sampling": True,
    "lambda_mst_distribution": lambda_int_distribution,
    "los_sampling": True,
    "los_distributions": kappa_ext_distribution,
    "alpha_lambda_sampling": False,
    #'sne_likelihood': 'Pantheon', # we are using the Pantheon sample as the likelihood
    "sne_likelihood": None,  # "CUSTOM",
    "kwargs_sne_likelihood": None,  # kwargs_sne_likelihood_roman,
    "sne_apparent_m_sampling": True,
    "sne_distribution": "GAUSSIAN",
    "z_apparent_m_anchor": z_apparent_m_anchor,
    "log_scatter": False,
    "interpolate_cosmo": True,
    "num_redshift_interp": 100,
    "custom_prior": None,
}

# =============================
# EMCEE SAMPLING CONFIGURATIONS
# =============================

# these configs are such that you can locally execute it in few hours, not meant to provide converged chains!
kwargs_emcee_lensed_sn = {
    "kwargs_mean_start": kwargs_mean_start,
    "kwargs_sigma_start": kwargs_sigma_start,
}
