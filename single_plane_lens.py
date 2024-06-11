"""
This file provides the likelihood function from the cosmological posterior of Li et al. (2024), in the w0waCDM model.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from defaults import *


li24_omega_m = np.load(f"{storage_directory}/li++24_omegaM_chain.npy")
li24_w0 = np.load(f"{storage_directory}/li++24_w0_chain.npy")
li24_wa = np.load(f"{storage_directory}/li++24_wa_chain.npy")

li24_chain = np.array([li24_omega_m, li24_w0, li24_wa]).T.reshape(-1, 3)

li24_df = pd.DataFrame(
    li24_chain,
    columns=[r"$\Omega_{\rm m}$", r"$w_0$", r"$w_a$"],
)


def get_li24_chain():
    """
    Returns the chain of the Li et al. (2024) posterior.
    """
    return li24_df.values


class CustomPrior(object):
    """
    This class provides the likelihood function from the cosmological posterior of Li et al. (2024), in the w0waCDM model.
    In addition, it includes priors on the anisotropy parameter, and the Roman SNe Ia apparent magnitude.
    """

    def __init__(
        self, include_a_ani=False, include_li24=False, include_roman_sne_mp=False
    ):
        """ """
        self.kde = KernelDensity(bandwidth=0.01, kernel="gaussian").fit(li24_df.values)
        self._include_a_ani = include_a_ani
        self._include_li24 = include_li24
        self._include_roman_sne_mp = include_roman_sne_mp
        self._roman_sne_mp_sigma = roman_apparent_m_p_sigma
        self._roman_sne_mp_mean = roman_apparent_m_p_true

        assert (
            self._include_a_ani or self._include_li24 or self._include_roman_sne_mp
        ), "At least one prior must be included."

    def __call__(self, kwargs_cosmo, kwargs_lens, kwargs_kin, kwargs_source):
        """ """
        return self.log_likelihood(kwargs_cosmo, kwargs_lens, kwargs_kin, kwargs_source)

    def log_likelihood(self, kwargs_cosmo, kwargs_lens, kwargs_kin, kwargs_source):
        """ """
        om = kwargs_cosmo.get("om")
        w0 = kwargs_cosmo.get("w0")
        wa = kwargs_cosmo.get("wa")

        log_likeliood = 0

        if self._include_li24:
            om = kwargs_cosmo.get("om")
            w0 = kwargs_cosmo.get("w0")
            wa = kwargs_cosmo.get("wa")

            log_likeliood = self.kde.score([[om, w0, wa]])

        if self._include_a_ani:
            a_ani = kwargs_kin.get("a_ani", 1)
            log_likeliood += np.log(1 / a_ani)

        if self._include_roman_sne_mp:
            mp_sne = kwargs_source.get("mu_sne")
            log_likeliood += (
                -0.5
                * (mp_sne - self._roman_sne_mp_mean) ** 2
                / self._roman_sne_mp_sigma**2
            )

        return log_likeliood
