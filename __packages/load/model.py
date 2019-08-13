# -*- coding: utf-8 -*-

# import built in libarys


# import 3rd party libarys
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pygam import LinearGAM, s, f, l

# import local libarys
from whypy.__packages.utils import utils


###############################################################################
def lingam(term='spline'):
    """
    Method to load unfitted Generalized Additive Models models of
    type modelclass

    INPUT:
    term: 'linear', 'spline' or 'factor'

    RETURN:
    model
    """
    if term == 'linear':
        regmod = LinearGAM(l(0))
    # GAM with spline term
    elif term == 'spline':
        regmod = LinearGAM(s(0))
    # GAM with factor term
    elif term == 'factor':
        regmod = LinearGAM(f(0))
    else:
        raise ValueError('Given Gam term unknown')
    return(regmod)


def svr(no_obs, term='poly4'):
    """
    Method to load unfitted SVR models of type modelclass

    INPUT:
    term: 'linear', 'poly2' or 'poly4'

    RETURN:
    model
    """
    if term == 'linear':
        regrmodl[tdep][tindep] = SVR(kernel='linear',
                                     gamma='auto_deprecated',
                                     C=1.0, epsilon=0.1)
    # SVR with poly kernel
    elif term == 'poly2':
        regrmodl[tdep][tindep] = SVR(kernel='poly', degree=2,
                                     gamma=0.8,
                                     C=0.8, epsilon=0.1)
    # GAM with factor term
    elif term == 'poly4':
        regrmodl[tdep][tindep] = SVR(kernel='poly', degree=4,
                                     gamma=0.8,
                                     C=0.8, epsilon=0.1)
    else:
        raise ValueError('Term unknown')
    return(regrmodl)
