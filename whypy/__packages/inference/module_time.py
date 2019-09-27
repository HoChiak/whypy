# -*- coding: utf-8 -*-

# import built in libarys

# import 3rd party libarys
from numpy import arange as np_arange

# import local libarys
from whypy.__packages.utils import utils

###############################################################################


class SteadyState():
    """
    Class for the steady state case (independence of time).
    """
    attr_time = 'steadystate'

    def __init__(self, t0, stride):
        """
        Class constructor for the steady state case (observations are
        independent of time). Assigns steady state specific methods to the
        inference model.
        """
        # self._t0 = 0
        # self._stride = 1

    def ids_init4time(self):
        """
        Method to init the ids for dependent and independent variable. For
        steadystate no further modification is required.
        """
        # Get number of observations including bootstrap_ratio
        if self._config['bootstrap'] > 0:
            no_obs = int(self.obs.shape[0] * self._kwargs['bootstrap_ratio'])
        else:
            no_obs = self.obs.shape[0]
        # Get ids
        self._ids_tdep = np_arange(0, no_obs, 1)
        self._ids_tindep = np_arange(0, no_obs, 1)
###############################################################################


class Transient():
    """
    Class for the transient case (dependence of time).
    """
    attr_time = 'transient'

    def __init__(self, t0, stride):
        """
        Class constructor for the transient case (observations are
        dependent of time). Assigns transient specific methods to the
        inference model.
        """
        self._t0 = t0
        self._stride = stride

    def ids_init4time(self):
        """
        Method to init the ids for dependent and independent variable. For
        steadystate no further modification is required.
        """
        # Get number of observations including bootstrap_ratio
        if self._config['bootstrap'] > 0:
            no_obs = int(self.obs.shape[0] * self._kwargs['bootstrap_ratio'])
        else:
            no_obs = self.obs.shape[0]
        # Get ids
        self._ids_tdep = np_arange(self._t0, no_obs, self._stride)
        self._ids_tindep = np_arange(0, no_obs-self._t0, self._stride)
###############################################################################
