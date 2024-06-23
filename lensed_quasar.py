"""
This file provides the kwargs to run cosmological sampling with the lensed quasar sample from Taak and Treu (2023). 
This code is adopted largely from https://github.com/sibirrer/TDCOSMO_forecast.
"""

import numpy as np
import pandas as pd
import pickle

from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util import constants as const
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util import constants as const
from hierarc.Sampling.ParamManager.cosmo_param import CosmoParam

from hierarc.Likelihood.parameter_scaling import ParameterScalingIFU
from hierarc.LensPosterior.kin_constraints import KinConstraints
from hierarc.Likelihood.parameter_scaling import ParameterScalingIFU
from hierarc.Sampling.ParamManager.cosmo_param import CosmoParam

from defaults import *


recompute_anisotropy_scaling = False

# load the Taak and Treu (2023) sample
taak_and_treu_23_sample = pd.read_csv(
    f"{storage_directory}/om10_varqso.ecsv", comment="#", delimiter=" "
)

num_external_leanses_lsst = 0  # number of external lenses for 'future' scenario
num_lensed_quasar = 236  # number of TDCOSMO lenses from Taak and Treu (2023), with variability > 3 * LSST uncertainty
num_quasar_lenses_w_jwst_kinematics = 40

ddt_sigma_rel = 0.055  # relative uncertainties in time-delay distance (with external convergence contribution) of the power-law mass profile per lens
kappa_ext_sigma = 0.02  # 1-sigma Gaussian uncertainties on the individual external convergence estimates

# ==========
# Lens model
# ==========

# Here we define Einstein radius, power-law slope and half light radius of the deflector - and their uncertainties.
# In this forecast, for simplicity, we assume that all lenses are identical in these parameters to re-utilize the dimensionless kinematics prediction for all lenses.

# lens and light model parameters and uncertainties (used for kinematic modeling uncertainties)
theta_E, theta_E_error = 1.0, 0.02
gamma, gamma_error = 2.0, 0.03
r_eff, r_eff_error = 1.0, 0.05

# ==============================================================================
# Kinematic observation settings
# ==============================================================================

sigma_v_measurement_sigma_rel_ground_based = 0.05  # uncertainty in the measurement of the velocity dispersion (fully covariant for IFU data)
sigma_v_measurement_sigma_rel_jwst = 0.03  # JWST-like precision on velocity dispersion

r_bins_jifu = np.linspace(
    0, 2 * r_eff, 11
)  # radial bins in arc seconds of the JWST IFU data
kwargs_seeing_jifu = {"psf_type": "GAUSSIAN", "fwhm": 0.1}

kwargs_seeing_single_aperture = {"psf_type": "GAUSSIAN", "fwhm": 0.8}
kwargs_jifu_aperture = {
    "aperture_type": "IFU_shells",
    "r_bins": r_bins_jifu,
    "center_dec": 0,
    "center_ra": 0,
}

kwargs_single_aperture = {
    "aperture_type": "slit",
    "width": 1,
    "length": 1,
    "angle": 0,
    "center_dec": 0,
    "center_ra": 0,
}

anisotropy_model = "OM"  # Osipkov-Merritt anisotropy model
anisotropy_distribution = (
    "NONE"  # single-valued distribution of the anisotropy parameter
)
kwargs_kin_true = {
    "a_ani": 1.0
}  # transition radius from isotropic to radial scaled by the effective radius of the deflector
if anisotropy_distribution == "GAUSSIAN":
    kwargs_kin_true["a_ani_sigma"] = 0.5

lambda_mst_distribution = "GAUSSIAN"  # single-valued distribution in the MST transform on top of a power-law model
kwargs_lens_true = {"lambda_mst": 1.0}
if lambda_mst_distribution == "GAUSSIAN":
    kwargs_lens_true["lambda_mst_sigma"] = lambda_int_sigma_true

# create astropy.cosmology instance of input cosmology
cosmo_param = CosmoParam(cosmology=cosmology)
cosmo_true = cosmo_param.cosmo(kwargs_cosmo_true)
kwargs_numerics_galkin = {
    "interpol_grid_num": 1000,  # numerical interpolation, should converge -> infinity
    "log_integration": True,  # log or linear interpolation of surface brightness and mass models
    "max_integrate": 100,
    "min_integrate": 0.001,
}  # lower/upper bound of numerical integrals

(
    z_lens_temp,
    z_source_temp,
    sigma_v_temp,
    sigma_v_error_independent_temp,
    sigma_v_error_covariant_temp,
) = (0.5, 1.0, [1], 1, 0)

# ====================
# single slit aperture
# ====================
kin_aperture = KinConstraints(
    z_lens_temp,
    z_source_temp,
    theta_E,
    theta_E_error,
    gamma,
    gamma_error,
    r_eff,
    r_eff_error,
    sigma_v_temp,
    sigma_v_error_independent=sigma_v_error_independent_temp,
    sigma_v_error_covariant=sigma_v_error_covariant_temp,
    kwargs_aperture=kwargs_single_aperture,
    kwargs_seeing=kwargs_seeing_single_aperture,
    kwargs_numerics_galkin=kwargs_numerics_galkin,
    anisotropy_model=anisotropy_model,
    kwargs_lens_light=None,
    lens_light_model_list=["HERNQUIST"],
    MGE_light=False,
    kwargs_mge_light=None,
    hernquist_approx=True,
    sampling_number=1000,
    num_psf_sampling=100,
    num_kin_sampling=1000,
    multi_observations=False,
)
ani_param_array = kin_aperture.ani_param_array

if recompute_anisotropy_scaling:
    j_model_list_aperture, error_cov_j_sqrt_aperture = (
        kin_aperture.model_marginalization(num_sample_model=100)
    )

    error_cov_j_sqrt_aperture = [[error_cov_j_sqrt_aperture]]
    ani_scaling_array_list_aperture = kin_aperture.anisotropy_scaling()

    anisotropy_scaling_aperture = ParameterScalingIFU(
        anisotropy_model,
        param_arrays=ani_param_array,
        scaling_grid_list=ani_scaling_array_list_aperture,
    )

    with open(f"{storage_directory}/anisotropy_scaling_aperture.pkl", "wb") as f:
        pickle.dump(
            [
                j_model_list_aperture,
                error_cov_j_sqrt_aperture,
                ani_scaling_array_list_aperture,
                anisotropy_scaling_aperture,
            ],
            f,
        )
else:
    with open(f"{storage_directory}/anisotropy_scaling_aperture.pkl", "rb") as f:
        (
            j_model_list_aperture,
            error_cov_j_sqrt_aperture,
            ani_scaling_array_list_aperture,
            anisotropy_scaling_aperture,
        ) = pickle.load(f)


# ========
# JWST IFU
# ========
sigma_v_temp_ifu = np.ones(len(r_bins_jifu) - 1)
kin_jifu = KinConstraints(
    z_lens_temp,
    z_source_temp,
    theta_E,
    theta_E_error,
    gamma,
    gamma_error,
    r_eff,
    r_eff_error,
    sigma_v_temp_ifu,
    sigma_v_error_independent=sigma_v_error_independent_temp,
    sigma_v_error_covariant=sigma_v_error_covariant_temp,
    kwargs_aperture=kwargs_jifu_aperture,
    kwargs_seeing=kwargs_seeing_jifu,
    kwargs_numerics_galkin=kwargs_numerics_galkin,
    anisotropy_model=anisotropy_model,
    kwargs_lens_light=None,
    lens_light_model_list=["HERNQUIST"],
    MGE_light=False,
    kwargs_mge_light=None,
    hernquist_approx=True,
    sampling_number=2000,
    num_psf_sampling=200,
    num_kin_sampling=2000,
    multi_observations=False,
)

if recompute_anisotropy_scaling:
    j_model_list_jifu, error_cov_j_sqrt_jifu = kin_jifu.model_marginalization(
        num_sample_model=500
    )
    ani_scaling_array_list_jifu = kin_jifu.anisotropy_scaling()
    anisotropy_scaling_jifu = ParameterScalingIFU(
        anisotropy_model,
        param_arrays=ani_param_array,
        scaling_grid_list=ani_scaling_array_list_jifu,
    )

    with open(f"{storage_directory}/anisotropy_scaling_jifu.pkl", "wb") as f:
        pickle.dump(
            [
                j_model_list_jifu,
                error_cov_j_sqrt_jifu,
                ani_scaling_array_list_jifu,
                anisotropy_scaling_jifu,
            ],
            f,
        )
else:
    with open(f"{storage_directory}/anisotropy_scaling_jifu.pkl", "rb") as f:
        (
            j_model_list_jifu,
            error_cov_j_sqrt_jifu,
            ani_scaling_array_list_jifu,
            anisotropy_scaling_jifu,
        ) = pickle.load(f)


def likelihood_lens(
    z_lens,
    z_source,
    sigma_v_measurement_sigma_rel,
    kappa_ext_sigma=0,
    ddt_sigma_rel=None,
    kin_type="aperture",
    no_noise=True,
):
    """
    Lens likelihood for a single lens system with kinematic data.

    :param z_lens: redshift of the lens
    :type z_lens: float
    :param z_source: redshift of the source
    :type z_source: float
    :param sigma_v_measurement_sigma_rel: relative uncertainty in the velocity dispersion measurement
    :type sigma_v_measurement_sigma_rel: float
    :param kappa_ext_sigma: uncertainty in the external convergence
    :type kappa_ext_sigma: float
    :param ddt_sigma_rel: relative uncertainty in the time-delay distance
    :type ddt_sigma_rel: float
    :param kin_type: type of kinematic data (aperture or IFU)
    :type kin_type: str
    :param no_noise: if True, no noise is added to the kinematic data
    :type no_noise: bool
    """

    lensCosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo_true)
    ds_dds = lensCosmo.ds / lensCosmo.dds

    if kin_type == "aperture":
        j_model = j_model_list_aperture
        error_cov_j_sqrt = error_cov_j_sqrt_aperture
        ani_scaling_array_list = ani_scaling_array_list_aperture
        anisotropy_scaling = anisotropy_scaling_aperture
    elif kin_type == "JIFU":
        j_model = j_model_list_jifu
        error_cov_j_sqrt = error_cov_j_sqrt_jifu
        ani_scaling_array_list = ani_scaling_array_list_jifu
        anisotropy_scaling = anisotropy_scaling_jifu

    # include anisotropy (draw from)
    aniso_param_array = anisotropy_scaling.draw_anisotropy(**kwargs_kin_true)
    aniso_scaling = anisotropy_scaling.param_scaling(aniso_param_array)

    lambda_int = np.random.normal(
        kwargs_lens_true["lambda_mst"], kwargs_lens_true.get("lambda_mst_sigma", 0)
    )
    # calculate scaling with anisotropy and lambda_int
    kappa_ext = np.random.normal(
        kappa_ext_population_mean_true, kappa_ext_population_sigma_true
    )

    # predict velocity dispersion from j_model (with kappa_ext !=0 )
    sigma_v_true = (
        np.sqrt(j_model * ds_dds * aniso_scaling * lambda_int * (1 - kappa_ext))
        * const.c
        / 1000
    )

    # compute measurement covariance matrix such that
    # 1. individual measurements per bin have uncertainties 'sigma_v_measurement_sigma_rel'
    # 2. total joint uncertainty has a covarian uncertainty of 'sigma_v_measurement_sigma_rel'

    n = len(sigma_v_true)  # number of data points
    sigma_cov_rel = (
        np.sqrt((n - 1) / n) * sigma_v_measurement_sigma_rel
    )  # covariant component not accounted in the individual measurement uncertainties such that the overall precision is limited by sigma_v_measurement_sigma_rel
    error_cov_measurement = np.outer(sigma_v_true, sigma_v_true) * (sigma_cov_rel**2)
    error_cov_measurement += np.diag(
        (sigma_v_true * sigma_v_measurement_sigma_rel) ** 2
    )

    # draw measurement
    if no_noise is True:
        sigma_v_measurement = sigma_v_true
    else:
        sigma_v_measurement = np.random.multivariate_normal(
            sigma_v_true, error_cov_measurement
        )

    # adding external convergence uncertainty in the J() model prediction uncertainty (mean kappa=0)
    error_cov_j_sqrt_kappa = error_cov_j_sqrt + np.outer(
        np.sqrt(j_model) * kappa_ext_sigma / 2, np.sqrt(j_model) * kappa_ext_sigma / 2
    )  # - np.diag(j_model * (kappa_ext_sigma/2)**2)

    # predict j_model (draw from original true prediction based on covariant errors)
    if no_noise is True:
        j_model_draw = j_model
    else:
        j_model_draw = (
            np.random.multivariate_normal(np.sqrt(j_model), error_cov_j_sqrt_kappa) ** 2
        )

    kwargs_likelihood = {
        "z_lens": z_lens,
        "z_source": z_source,
        "likelihood_type": "IFUKinCov",
        "sigma_v_measurement": sigma_v_measurement,
        "anisotropy_model": anisotropy_model,
        "j_model": j_model_draw,
        "error_cov_measurement": error_cov_measurement,
        "error_cov_j_sqrt": error_cov_j_sqrt_kappa,
        "ani_param_array": ani_param_array,
        "ani_scaling_array_list": ani_scaling_array_list,
    }

    # predict ddt and draw from uncertainties
    if ddt_sigma_rel is not None:
        ddt_true = lensCosmo.ddt
        ddt_sigma = ddt_true * ddt_sigma_rel
        if no_noise is True:
            ddt_mean = ddt_true
        else:
            ddt_mean = np.random.normal(ddt_true, ddt_sigma)
        kwargs_likelihood["likelihood_type"] = "DdtGaussKin"
        kwargs_likelihood["ddt_mean"] = ddt_mean
        kwargs_likelihood["ddt_sigma"] = ddt_sigma

    return kwargs_likelihood


kwargs_mean_start = {
    "kwargs_cosmo": kwargs_cosmo_true,
    "kwargs_lens": {"lambda_mst": 1, "lambda_mst_sigma": 0.03},
    "kwargs_kin": {"a_ani": 1, "a_ani_sigma": 0.5},
    "kwargs_los": kwargs_los_true,
}

kwargs_sigma_start = {
    "kwargs_cosmo": kwargs_cosmo_sigma,
    "kwargs_lens": {"lambda_mst": 0.1, "lambda_mst_sigma": 0.03},
    "kwargs_kin": {"a_ani": 0.5, "a_ani_sigma": 0.5},
    "kwargs_los": kwargs_los_sigma,
}


kwargs_bounds = {
    "kwargs_lower_cosmo": kwargs_cosmo_lower,
    "kwargs_lower_lens": {"lambda_mst": 0.5, "lambda_mst_sigma": 0},
    "kwargs_lower_kin": {"a_ani": 0.5, "a_ani_sigma": 0},
    "kwargs_lower_los": kwargs_los_lower,
    "kwargs_upper_cosmo": kwargs_cosmo_upper,
    "kwargs_upper_lens": {"lambda_mst": 1.5, "lambda_mst_sigma": 0.5},
    "kwargs_upper_kin": {"a_ani": 5, "a_ani_sigma": 1},
    "kwargs_upper_los": kwargs_los_upper,
    "kwargs_fixed_cosmo": {},
    "kwargs_fixed_lens": {
        "lambda_mst_sigma": lambda_int_sigma_true,
    },
    "kwargs_fixed_kin": {},
    "kwargs_fixed_los": kwargs_los_fixed,
}


# ==================================
# joint options for hierArc sampling
# ==================================
kwargs_sampler_lensed_quasar = {
    "cosmology": cosmology,
    "kwargs_bounds": kwargs_bounds,
    "lambda_mst_sampling": True,
    "lambda_mst_distribution": lambda_mst_distribution,
    "anisotropy_sampling": True,
    "los_sampling": True,
    "los_distributions": ["GAUSSIAN"],
    "alpha_lambda_sampling": False,
    "sigma_v_systematics": False,
    "log_scatter": False,
    "anisotropy_model": anisotropy_model,
    "anisotropy_distribution": anisotropy_distribution,
    "custom_prior": None,
    "interpolate_cosmo": True,
    "num_redshift_interp": 100,
    "cosmo_fixed": None,
}


kwargs_likelihood_list_lensed_quasar = []

indices = np.random.choice(
    taak_and_treu_23_sample.index, size=num_lensed_quasar, replace=False
)

for i in range(num_lensed_quasar):
    z_lens = taak_and_treu_23_sample["ZLENS"][indices[i]]
    z_source = taak_and_treu_23_sample["ZSRC"][indices[i]]

    lens = {"z_lens": z_lens, "z_source": z_source, "ddt_sigma_rel": ddt_sigma_rel}
    kwargs_likelihood_list_lensed_quasar.append(
        likelihood_lens(
            kin_type="JIFU" if i < num_quasar_lenses_w_jwst_kinematics else "aperture",
            sigma_v_measurement_sigma_rel=sigma_v_measurement_sigma_rel_ground_based,
            kappa_ext_sigma=0,  # kappa_ext_sigma,
            # not adding kappa_ext sigma uncertainty to kinematics as kappa_ext is treated at the population level
            **lens,
        )
    )

kwargs_emcee_lensed_quasar = {
    "kwargs_mean_start": kwargs_mean_start,
    "kwargs_sigma_start": kwargs_sigma_start,
}
