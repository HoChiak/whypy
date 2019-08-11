# -*- coding: utf-8 -*-

# import built in libarys


# import 3rd party libarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import local libarys
from whypy.__packages.utils import utils
from whypy.__packages.utils import stats


###############################################################################
class General():
    """
    General Causal Inference methods.
    """

    def __init__(self):
        """
        Class Constructor for General CI Methods
        """

    def check_instance_attr(self, scale):
        """
        Method to check the instance attributes
        """
        assert self._xi is not None, 'Observations are None type'
        assert not(any(np.isnan(self._xi))), 'Observations contain np.nan'
        assert not(any(np.isinf(self._xi))), 'Observations contain np.inf'
        assert self._regmod is not None, 'Regression Model is None type'
        assert ((scale is False) or ((scale is True) and (self._scaler is not None))), 'If scale is True, a scaler must be assigned'
