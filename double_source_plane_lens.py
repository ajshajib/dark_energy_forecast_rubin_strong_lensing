"""
This file provides the kwargs to run cosmological sampling with the double-source-plane lenses from Sharma, Collett, and Linder (2023). 
"""

import pandas as pd

from hierarc.Sampling.ParamManager.cosmo_param import CosmoParam
from hierarc.Sampling.ParamManager.cosmo_param import CosmoParam

from defaults import *

### Forecast settings
# Here we describe the redshift distribution, precision on Einstein radii ratios of an anticipated sample

cosmology = "w0waCDM"

kwargs_cosmo_true = {
    "h0": 70,
    "om": 0.3,
    "w0": -1,
    "wa": 0.0,
}

cosmo_param = CosmoParam(cosmology=cosmology)
cosmo_true = cosmo_param.cosmo(kwargs_cosmo_true)

# ignore second line
# first line is the header
dspl_mock_dataframe = pd.read_csv(
    f"{storage_directory}/sharma++23_lsst_dspl.txt", sep=" "
)

num_dspl = len(dspl_mock_dataframe)


kwargs_likelihood_list_dspl = []

for i in range(num_dspl):
    kwargs_likelihood = {
        "z_lens": dspl_mock_dataframe["zl"][i],
        "z_source_1": dspl_mock_dataframe["zs1"][i],
        "z_source_2": dspl_mock_dataframe["zs2"][i],
        "beta_dspl": dspl_mock_dataframe["beta"][i],
        "sigma_beta_dspl": dspl_mock_dataframe["betaerr"][i],
        "likelihood_type": "DSPL",
    }
    kwargs_likelihood_list_dspl.append(kwargs_likelihood)


### hierArc sampling settings
cosmology = "w0waCDM"

kwargs_mean_start = {"kwargs_cosmo": kwargs_cosmo_true}
kwargs_sigma_start = {"kwargs_cosmo": kwargs_cosmo_sigma}


kwargs_bounds = {
    "kwargs_lower_cosmo": kwargs_cosmo_lower,
    "kwargs_upper_cosmo": kwargs_cosmo_upper,
    "kwargs_fixed_cosmo": {"h0": kwargs_cosmo_true["h0"]},
}


# joint options for hierArc sampling
kwargs_sampler_dspl = {
    "cosmology": cosmology,
    "kwargs_bounds": kwargs_bounds,
    "lambda_mst_sampling": False,
    "anisotropy_sampling": False,
    # "kappa_ext_sampling": False,
    # "kappa_ext_distribution": "NONE",
    "alpha_lambda_sampling": False,
    "sigma_v_systematics": False,
    "log_scatter": False,
    "custom_prior": None,
    "interpolate_cosmo": False,
    "num_redshift_interp": 100,
    "cosmo_fixed": None,
}

kwargs_emcee_dspl = {
    "kwargs_mean_start": kwargs_mean_start,
    "kwargs_sigma_start": kwargs_sigma_start,
}
