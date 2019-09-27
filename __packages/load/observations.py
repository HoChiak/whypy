# -*- coding: utf-8 -*-

# import built in libarys

# import 3rd party libarys
import numpy as np
from IPython.display import display, HTML

# import local libarys
from whypy.__packages.utils import utils

###############################################################################


def get_desc(modelclass, no_var, cl_func, cl_Noise, SCM):
    desc = """
<html>
<body>
<style>
table td, table th, table tr {text-align:left !important;}
</style>
<div style="background-color:lightgrey;color:black;padding:0px;" align="center">
<h4>Observations Description</h4>
</div>
<table>
<tr>
<td><b>Modelclass:</b></td> <td>%i</td>
</tr>
<tr>
<td><b>No. of Variables:</b></td> <td>%i</td>
</tr>
<tr>
<td><b>Class of Functions:</b></td> <td>%s</td>
</tr>
<tr>
<td><b>Class of Noise Distribution:</b></td> <td>%s</td>
</tr>
<tr>
<td><b>SCM</b></td> <td>
%s
</td>
</tr>
</table>
</body>
</html>
             """ % (modelclass, no_var, cl_func, cl_Noise, SCM)
    return(desc)


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
    if modelclass == 1:
        desc = get_desc(modelclass, no_var=2,
                        cl_func = 'Linear',
                        cl_Noise = 'Uniform, Additive, Non-Equivalent',
                        SCM = """<p>X<sub>0</sub> ~ [ 0.4 * <strong><i>U</i></strong> (0, 5) ]</p>
                                 <p>X<sub>1</sub> ~ [ 2 * X<sub>0</sub> + 0.2 * <strong><i>U</i></strong> (0, 5) ]</p>""")
        Edge_list = [[0, 1]]
        X0 = np.array(0.4 * np.random.uniform(0, 5, no_obs))
        X1 = np.array(2 * X0 + 0.2 * np.random.uniform(0, 5, no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        xi = np.concatenate([X0, X1], axis=1)
        no_var = xi.shape[1]
    # model class 2
    elif modelclass == 2:
        desc = get_desc(modelclass, no_var=2,
                        cl_func = 'Non-Linear',
                        cl_Noise = 'Gaussian, Additive, Non-Equivalent',
                        SCM = """<p>X<sub>0</sub> ~ [ <strong><i>N</i></strong> (1.5, 1.0) ]</p>
                                 <p>X<sub>1</sub> ~ [ X<sub>0</sub><sup>2</sup> + <strong><i>N</i></strong> (0, 1.5) ]</p>""")
        Edge_list = [[0, 1]]
        X0 = np.array(1.5 + 1.0 * np.random.randn(no_obs))
        X1 = np.array(X0**2 + 1.5 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        xi = np.concatenate([X0, X1], axis=1)
        no_var = xi.shape[1]
    # model class 3
    elif modelclass == 3:
        desc = get_desc(modelclass, no_var=3,
                        cl_func = 'Linear',
                        cl_Noise = 'Uniform, Additive, Non-Equivalent',
                        SCM = """<p>X<sub>0</sub> ~ [ 5 + 0.4 * <strong><i>U</i></strong> (0, 5) ]</p>
                                 <p>X<sub>1</sub> ~ [ X<sub>0</sub> + 0.6 * <strong><i>U</i></strong> (0, 5) ]</p>
                                 <p>X<sub>2</sub> ~ [ X<sub>0</sub> + X<sub>1</sub> + 0.2 * <strong><i>U</i></strong> (0, 5) ]</p>""")
        Edge_list = [[0, 1],
                     [0, 2],
                     [1, 2]]
        X0 = np.array(5 + 0.4 * np.random.uniform(0, 5, no_obs))
        X1 = np.array(X0 + 0.6 * np.random.uniform(0, 5, no_obs))
        X2 = np.array(X0 + X1 + 0.2 * np.random.uniform(0, 5, no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_var = xi.shape[1]
    # model class 4
    elif modelclass == 4:
        desc = get_desc(modelclass, no_var=3,
                        cl_func = 'Non-Linear',
                        cl_Noise = 'Gaussian, Additive, Non-Equivalent',
                        SCM = """<p>X<sub>0</sub> ~ [ <strong><i>N</i></strong> (5, 0.25) ]</p>
                                 <p>X<sub>1</sub> ~ [ X<sub>0</sub><sup>3</sup> + <strong><i>N</i></strong> (0, 0.16) ]</p>
                                 <p>X<sub>2</sub> ~ [ X<sub>0</sub><sup>4</sup> + X<sub>1</sub><sup>3</sup> + <strong><i>N</i></strong> (0, 0.09) ]</p>""")
        Edge_list = [[0, 1],
                     [0, 2],
                     [1, 2]]
        X0 = np.array(5 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(X0*3 + 0.4 * np.random.randn(no_obs))
        X2 = np.array(X0*4 + X1*3 + 0.3 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_var = xi.shape[1]
    # model class 5.0
    elif modelclass == 5:
        desc = get_desc(modelclass, no_var=3,
                        cl_func = 'Non-Linear',
                        cl_Noise = 'Gaussian, Additive, Non-Equivalent',
                        SCM = """<p>X<sub>0</sub> ~ [ <strong><i>N</i></strong> (1, 0.25) ]</p>
                                 <p>X<sub>1</sub> ~ [ <strong><i>N</i></strong> (0, 2.25) ]</p>
                                 <p>X<sub>2</sub> ~ [ X<sub>0</sub><sup>4</sup> + X<sub>1</sub><sup>2</sup> + <strong><i>N</i></strong> (0, 0.25) ]</p>""")
        Edge_list = [[0, 2],
                     [1, 2]]
        X0 = np.array(1 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(1.5 * np.random.randn(no_obs))
        X2 = np.array(X0**4 + X1**2 + 0.5 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_var = xi.shape[1]
    # model class 5.1
    elif modelclass == 6:
        desc = get_desc(modelclass, no_var=3,
                        cl_func = 'Non-Linear',
                        cl_Noise = 'Gaussian, Additive, Non-Equivalent',
                        SCM = """<p>X<sub>0</sub> ~ [ <strong><i>N</i></strong> (1, 0.25) ]</p>
                                 <p>X<sub>1</sub> ~ [ X<sub>0</sub><sup>4</sup> + <strong><i>N</i></strong> (0, 2.25) ]</p>
                                 <p>X<sub>2</sub> ~ [ X<sub>0</sub><sup>3</sup> + <strong><i>N</i></strong> (0, 0.25) ]</p>""")
        Edge_list = [[0, 1],
                     [0, 2]]
        X0 = np.array(1 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(X0**4 + 1.5 * np.random.randn(no_obs))
        X2 = np.array(X0**3 + 0.5 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_var = xi.shape[1]
    # model class 5.2
    elif modelclass == 7:
        desc = get_desc(modelclass, no_var=3,
                        cl_func = 'Non-Linear',
                        cl_Noise = 'Gaussian, Additive, Non-Equivalent',
                        SCM = """<p>X<sub>0</sub> ~ [ <strong><i>N</i></strong> (1, 0.25) ]</p>
                                 <p>X<sub>1</sub> ~ [ X<sub>0</sub><sup>4</sup> + <strong><i>N</i></strong> (0, 2.25) ]</p>
                                 <p>X<sub>2</sub> ~ [ X<sub>0</sub><sup>4</sup> + X<sub>1</sub><sup>2</sup> + <strong><i>N</i></strong> (0, 0.25) ]</p>""")
        Edge_list = [[0, 1],
                     [0, 2],
                     [1, 2]]
        X0 = np.array(1 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(X0**4 + 1.5 * np.random.randn(no_obs))
        X2 = np.array(X0**4 + X1**2 + 0.5 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_var = xi.shape[1]
    # model class 5.3
    elif modelclass == 8:
        desc = get_desc(modelclass, no_var=3,
                        cl_func = 'Non-Linear',
                        cl_Noise = 'Gaussian, Additive, Non-Equivalent',
                        SCM = """<p>X<sub>0</sub> ~ [ <strong><i>N</i></strong> (1, 0.25) ]</p>
                                 <p>X<sub>1</sub> ~ [ X<sub>0</sub><sup>4</sup> + <strong><i>N</i></strong> (0, 2.25) ]</p>
                                 <p>X<sub>2</sub> ~ [ X<sub>1</sub><sup>2</sup> + <strong><i>N</i></strong> (0, 0.25) ]</p>""")
        Edge_list = [[0, 1],
                     [1, 2]]
        X0 = np.array(1 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(X0**4 + 1.5 * np.random.randn(no_obs))
        X2 = np.array(X1**2 + 0.5 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_var = xi.shape[1]
    # model class 5.4
    elif modelclass == 9:
        desc = get_desc(modelclass, no_var=3,
                        cl_func = 'Non-Linear',
                        cl_Noise = 'Gaussian, Additive, Non-Equivalent',
                        SCM = """<p>X<sub>0</sub> ~ [ <strong><i>N</i></strong> (1, 0.25) ]</p>
                                 <p>X<sub>1</sub> ~ [ <strong><i>N</i></strong> (0, 2.25) ]</p>
                                 <p>X<sub>2</sub> ~ [ <strong><i>N</i></strong> (0, 0.25) ]</p>""")
        Edge_list = []
        X0 = np.array(1 + 0.5 * np.random.randn(no_obs))
        X1 = np.array(4 + 1.5 * np.random.randn(no_obs))
        X2 = np.array(2 + 0.5 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_var = xi.shape[1]
    # model class 6
    elif modelclass == 10:
        desc = get_desc(modelclass, no_var=4,
                        cl_func = 'Non-Linear',
                        cl_Noise = 'Gaussian, Additive, Non-Equivalent',
                        SCM = """<p>X<sub>0</sub> ~ [ <strong><i>N</i></strong> (1, 0.16) ]</p>
                                 <p>X<sub>1</sub> ~ [ X<sub>0</sub><sup>4</sup> + <strong><i>N</i></strong> (0, 0.25) ]</p>
                                 <p>X<sub>2</sub> ~ [ X<sub>0</sub><sup>4</sup> + X<sub>1</sub><sup>2</sup> + <strong><i>N</i></strong> (0, 0.36) ]</p>
                                 <p>X<sub>3</sub> ~ [ <strong><i>N</i></strong> (2, 0.64) ]</p>""")
        Edge_list = [[0, 1],
                     [0, 2],
                     [1, 2]]
        X0 = np.array(1 + 0.4 * np.random.randn(no_obs))
        X1 = np.array(X0**4 + 0.5 * np.random.randn(no_obs))
        X2 = np.array(X0**4 + X1**2 + 0.6 * np.random.randn(no_obs))
        X3 = np.array(2 + 0.8 * np.random.randn(no_obs))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        X3 = X3.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2, X3], axis=1)
        no_var = xi.shape[1]
    else:
        raise ValueError('Modelclass unknown')
    # Extract Node_list by total number of observations
    Node_list = np.arange(0, no_var, 1).tolist()
    # Plot correct Causal Graph
    display(HTML(desc))
    display(HTML("""
    <div style="background-color:lightgrey;color:black;padding:0px;" align="center">
    <h4>Observations - Ground Truth Causal Graph</h4>
    </div>"""))
    utils.plot_DAG(Edge_list, Node_list)
    return(xi)
###############################################################################
