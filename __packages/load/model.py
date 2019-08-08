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
def gam(no_obs, term='spline'):
    """
    Method to load unfitted Generalized Additive Models models of
    type modelclass

    INPUT:
    term: 'linear', 'spline' or 'factor'

    RETURN:
    [[model, ...] ... ]
    """
    # Create empty list size Xi times Xi
    regrmodl = utils.init_2V_list(no_obs)
    # Assign model to empty list
    # Loop trough all possible combinations of tdep and tindep
    for tdep in range(no_obs):
        for tindep in range(no_obs):
            if tdep != tindep:  # no diagonal values
                # GAM with linear term
                if term == 'linear':
                    regrmodl[tdep][tindep] = LinearGAM(l(0))
                # GAM with spline term
                elif term == 'spline':
                    regrmodl[tdep][tindep] = LinearGAM(s(0))
                # GAM with factor term
                elif term == 'factor':
                    regrmodl[tdep][tindep] = LinearGAM(f(0))
                else:
                    raise ValueError('Term unknown')
    return(regrmodl)


def svr(no_obs, term='poly4'):
    """
    Method to load unfitted SVR models of type modelclass

    INPUT:
    term: 'linear', 'poly2' or 'poly4'

    RETURN:
    [[model, ...] ... ]
    """
    # Create empty list size Xi times Xi
    regrmodl = utils.init_2V_list(no_obs)
    # Assign model to empty list
    # Loop trough all possible combinations of tdep and tindep
    for tdep in range(no_obs):
        for tindep in range(no_obs):
            if tdep != tindep:  # no diagonal values
                # SVR with linear kernel
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
