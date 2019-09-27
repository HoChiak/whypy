# -*- coding: utf-8 -*-

# import built in libarys


# import 3rd party libarys
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# import local libarys
from whypy.__packages.utils import utils


###############################################################################
def minmax():
    """
    Method to load a [0,1] MinMaxScaler

    RETURN:
    scaler
    """
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    utils.display_get_params('MinMaxScaler Description', scaler.get_params())
    return(scaler)


def standard():
    """
    Method to load a zero mean and unit variance StandardScaler

    RETURN:
    scaler
    """
    scaler = StandardScaler(copy=True)
    utils.display_get_params('StandardScaler Description', scaler.get_params())
    return(scaler)
