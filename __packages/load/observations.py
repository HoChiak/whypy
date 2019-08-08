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
def observations(modelclass, no_obs=100, seed=None):
    """
    Method to load ovbservations. In the 2 variable case there are restrictions
    on the modelclass to needed for identifiable.
                4: linear model, three variables, gaussian noise
                   SCM ->   X0 ~ [G(5, 2.25)]
                            X1 ~ [X0*3 + G(0, 2.25)]
                            X2 ~ [X0*4 - X1*3 + G(0, 0.25)]
                5: polynomial model, three variables, gaussian noise
                   SCM ->   X0 ~ [G(5, 2.25)]
                            X1 ~ [X0**3 + G(0, 2.25)]
                            X2 ~ [X0**4 - X1**3 + G(0, 0.25)]
    no_obs:     data points created for each observation
    seed:       set seed for np.random methods

    RETURN:
    Xi
    """
    np.random.seed(seed=seed)
    # model class 1
    if modelclass == '2VLiNGAM':
        desc = """
Modelclass '2VLiNGAM':
No. of variables:\t 2
Class of functions:\t Linear,
Class of noise distr:\t Non-Gaussian (here Uniform), Additive, non equivalent
SCM -> \tX0 ~ [U(0.4)]
       \tX1 ~ [2*X0 + U(0.2)]
               """
        X0 = np.array(0.4 * np.random.uniform(0, 5, no_obs))
        X1 = np.array(2 * X0 + 0.2 * np.random.uniform(0, 5, no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        xi = np.concatenate([X0, X1], axis=1)
        no_obs = xi.shape[1]
    # model class 2
    elif modelclass == '2VNLiGAM':
        desc = """
Modelclass '2VNLiGAM':
No. of variables:\t 2
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, equivalent
SCM -> \tX0 ~ [5 + G(0, 2.25)]
       \tX1 ~ [X0^3 + G(0, 0.25)]
               """
        X0 = np.array(5 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(X0**3 + 0.5 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        xi = np.concatenate([X0, X1], axis=1)
        no_obs = xi.shape[1]
    # model class 3
    elif modelclass == '3VLiNGAM':
        desc = """
Modelclass '3VLiNGAM':
No. of variables:\t 3
Class of functions:\t Linear,
Class of noise distr:\t Non-Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [5 + U(0.4)]
       \tX1 ~ [X0 + U(0.6)]
       \tX2 ~ [X0 + X1 + U(0.2)]
               """
        X0 = np.array(5 + 0.4 * np.random.uniform(0, 5, no_obs))
        X1 = np.array(X0 + 0.6 * np.random.uniform(0, 5, no_obs))
        X2 = np.array(X0 + X1 + 0.2 * np.random.uniform(0, 5, no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 4
    elif modelclass == '3VLiGAM':
        desc = """
Modelclass '3VLiGAM':
No. of variables:\t 3
Class of functions:\t Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(5, 0.25)]
       \tX1 ~ [X0*3 + G(0, 0.16)]
       \tX2 ~ [X0*4 + X1*3 + G(0, 0.09)]
               """
        X0 = np.array(5 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(X0*3 + 0.4 * np.random.randn(no_obs))
        X2 = np.array(X0*4 + X1*3 + 0.3 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 5.0
    elif modelclass == '3VNLiGAM-collider':
        desc = """
Modelclass '3VNLiGAM':
No. of variables:\t 3
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(1, 2.25)]
       \tX1 ~ [G(0, 2.25)]
       \tX2 ~ [X0^4 + X1^2 + G(0, 0.25)]
               """
        Edge_list = [['X_0', 'X_2'],
                     ['X_1', 'X_2']]
        X0 = np.array(1 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(1.5 * np.random.randn(no_obs))
        X2 = np.array(X0**4 + X1**2 + 0.5 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 5.1
    elif modelclass == '3VNLiGAM-rev-collider':
        desc = """
Modelclass '3VNLiGAM':
No. of variables:\t 3
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(1, 2.25)]
       \tX1 ~ [X0^4 + G(0, 2.25)]
       \tX2 ~ [X0^3 + G(0, 0.25)]
               """
        Edge_list = [['X_0', 'X_1'],
                     ['X_0', 'X_2']]
        X0 = np.array(1 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(X0**4 + 1.5 * np.random.randn(no_obs))
        X2 = np.array(X0**3 + 0.5 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 5.2
    elif modelclass == '3VNLiGAM-confounded':
        desc = """
Modelclass '3VNLiGAM':
No. of variables:\t 3
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(1, 2.25)]
       \tX1 ~ [X0^4 + G(0, 2.25)]
       \tX2 ~ [X0^4 + X1^2 + G(0, 0.25)]
               """
        Edge_list = [['X_0', 'X_1'],
                     ['X_0', 'X_2'],
                     ['X_1', 'X_2']]
        X0 = np.array(1 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(X0**4 + 1.5 * np.random.randn(no_obs))
        X2 = np.array(X0**4 + X1**2 + 0.5 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 5.3
    elif modelclass == '3VNLiGAM-series':
        desc = """
Modelclass '3VNLiGAM':
No. of variables:\t 3
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(1, 2.25)]
       \tX1 ~ [X0^4 + G(0, 2.25)]
       \tX2 ~ [X1^2 + G(0, 0.25)]
               """
        Edge_list = [['X_0', 'X_1'],
                     ['X_1', 'X_2']]
        X0 = np.array(1 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(X0**4 + 1.5 * np.random.randn(no_obs))
        X2 = np.array(X1**2 + 0.5 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 5.4
    elif modelclass == '3VNLiGAM-none':
        desc = """
Modelclass '3VNLiGAM':
No. of variables:\t 3
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(1, 2.25)]
       \tX1 ~ [G(4, 2.25)]
       \tX2 ~ [G(2, 0.25)]
               """
        Edge_list = []
        X0 = np.array(1 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(4 + 1.5 * np.random.randn(no_obs))
        X2 = np.array(2 + 0.5 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 6
    elif modelclass == '4VNLiGAM':
        desc = """
Modelclass '3VNLiGAM':
No. of variables:\t 3
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(1, 2.25)]
       \tX1 ~ [X0^3 + G(0, 2.25)]
       \tX2 ~ [X0^3 + X1^2 + G(0, 0.25)]
       \tX3 ~ [G(2, 0.125)]
               """
        X0 = np.array(1 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(X0**4 + 1.5 * np.random.randn(no_obs))
        X2 = np.array(X0**4 + X1**2 + 0.5 * np.random.randn(no_obs))
        X3 = np.array(2 + 0.5 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        X3 = X3.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2, X3], axis=1)
        no_obs = xi.shape[1]
    else:
        raise ValueError('Modelclass unknown')
    # Extract Node_list by total number of observations
    Node_list = np.arange(0, no_obs, 1)
    Node_list = [r'X_%i' % (t) for t in Node_list]
    # Plot correct Causal Graph
    print('CAUSAL GRAPH:')
    utils.plot_DAG(Edge_list, Node_list)
    print(desc)
    return(xi, no_obs)
