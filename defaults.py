import os
import numpy as np

from hierarc.Sampling.ParamManager.cosmo_param import CosmoParam


# we set a random seed for reproduction of the specific results
np.random.seed(71)

# paths to save and load data products
path = os.getcwd()

storage_directory = os.path.join(path, "resources")
posterior_directory = os.path.join(path, "posteriors")


# cosmology settings
cosmology = "w0waCDM"

kwargs_cosmo_true = {"h0": 70, "om": 0.3, "w0": -1, "wa": 0}
kwargs_cosmo_sigma = {"h0": 10, "om": 0.05, "w0": 0.2, "wa": 0.2}
kwargs_cosmo_upper = {"h0": 200, "om": 1, "w0": 0, "wa": 3}
kwargs_cosmo_lower = {"h0": 0, "om": 0, "w0": -2, "wa": -3}

# create astropy.cosmology instance of input cosmology
cosmo_param = CosmoParam(cosmology=cosmology)
cosmo_true = cosmo_param.cosmo(kwargs_cosmo_true)

# we use the mean inferred apparent magnitude of the Pantheon sample
roman_apparent_m_p_true = 18.965677764035803  # apparent magnitude (mean in magnitude) at pivot redshift, this is the value inferred by the Pantheon analysis
roman_apparent_m_p_sigma = 0.0045  # uncertainty in the apparent magnitude

kappa_ext_population_mean_true = 0.0  # population mean of the external convergence
kappa_ext_population_sigma_true = 0.025  # 1-sigma Gaussian scatter in the distribution of external convergences for individual lenses

kwargs_los_true = [
    {
        "mean": kappa_ext_population_mean_true,
        "sigma": kappa_ext_population_sigma_true,
    }
]
kwargs_los_sigma = [{"mean": 0.01, "sigma": 0.01}]
kwargs_los_lower = [{"mean": -0.2, "sigma": 0}]
kwargs_los_upper = [{"mean": 0.2, "sigma": 0.2}]
kwargs_los_fixed = kwargs_los_true  # [{}]

lambda_int_sigma_true = 0.03  # 1-sigma scatter in the lambda_mst parameter
