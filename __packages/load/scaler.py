# -*- coding: utf-8 -*-

# import built in libarys


# import 3rd party libarys
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# import local libarys
from whypy.__packages.utils import utils


###############################################################################
def minmax(no_obs):
    """
    Method to load a [0,1] MinMaxScaler

    RETURN:
    scaler
    """
    scaler = list()
    for t in range(no_obs):
        scaler.append(MinMaxScaler(feature_range=(0, 1), copy=False))
    return(scaler)


def standard(no_obs):
    """
    Method to load a zero mean and unit variance StandardScaler

    RETURN:
    scaler
    """
    scaler = list()
    for t in range(no_obs):
        scaler.append(StandardScaler(copy=False))
    return(scaler)
