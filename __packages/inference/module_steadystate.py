# -*- coding: utf-8 -*-

# import built in libarys


# import 3rd party libarys


# import local libarys
from whypy.__packages.utils import utils

from importlib import reload
utils=reload(utils)
###############################################################################
class SteadyState():
    """
    Causal methods for the Steady State Case (independence of time).
    """
    attr_time = 'steadystate'

    def __init__(self):
        """
        Class Constructor for the Steady State Case (observations are
        independent of time). Assigns Steady State specific Methods to the
        Inference Model. Instance Attributes are assigned in the inference
        Class.
        """
