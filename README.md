<!DOCTYPE html>
<html>
<body>
<div style="background-color:RGB(0,81,158);color:RGB(255,255,255);padding:10px;">
<h1> <a name="whypy"></a>WhyPy</h1>
</div>

<!-- #region -->
A python repository for **causal inference**.

Currently available approaches in this repository are based on **Additive Noise Models (ANMs)**.

<u>Install:</u>
```shell
pip install whypy
```

<u>Content:</u>
1. [A short introduction into the theory of causal inference](#theory)
2. [A quick start example how to run causal inference with this repository](#quick)
3. [Additive Noise Models in WhyPy](#model)
    1. [Model Instances (Bivariate-MultiVariate | SteadyState-Transient)](#model-init)
    2. [Instance Parameters](#model-parameters)
    3. [Instance Methods](#model-methods)
    4. [Instance Attributes](#model-attributes)
4. [Various Templates for:](#template)
    1. [Observations](#template-observations)
    2. [Regression Models](#template-regressionmodels)
    3. [Scaler](#template-scaler)

<u>Models:</u>

Within the WhyPy Toolbox four possible models are distinguished

1. The data producing process is **steady state** + The model is **bivariate** (one independent variable)

   ![BiVariate-SteadyState](__pictures/cause-effect-bivariate-steadystate.png)

2. The data producing process is **steady state** + The model is **bivariate** (n independent variable)

   ![MultiVariate-SteadyState](__pictures/cause-effect-mvariate-steadystate.png)

3. The data producing process is **transient** (<i>t<sub>0</sub></i>: offset, <i>s</i>: stride)+ The model is **multi variate** (one independent variable)

   ![BiVariate-Transient](__pictures/cause-effect-bivariate-transient.png)

4. The data producing process is **transient** (<i>t<sub>0</sub></i>: offset, <i>s</i>: stride)+ The model is **multi variate** (n independent variable)

   ![MultiVariate-Transient](__pictures/cause-effect-mvariate-transient.png)


<!-- #endregion -->

<div style="background-color:RGB(0,81,158);color:RGB(255,255,255);padding:10px;">
<h1> <a name="theory"></a>Causal Inference (Short Introduction)</h1>
</div>

<!-- #region -->
The most elementary question of causality is the one asking whether "<i>X</i> causes <i>Y</i> or vice versa". An often discussed example is the question if smoking (<i>X</i>) causes cancer (<i>Y</i>). At this point the question about causal relationships is already getting more complex. Beside the possibility that <i>X</i> causes <i>Y</i> (<i>X &rarr; Y</i>), there are other possible causal relationships. One is that a third Variable <i>Z</i> is confounding both <i>X</i> and <i>Y</i> (<i>X &larr; Z &rarr; Y</i>). In the confounding case, only looking at <i>X</i> and <i>Y</i>, might show a correlation due to the confounder even though they are not causaly related. [[1]](#Pearl), [[2]](#Mooji)

![Cause-Effect-Confounded](__pictures/cause-effect-confounded.png)

Causal Inference is the task of learning causal relationships from purely observational data. This task is a fundamental problem in science. A variety of causal inference methods are available that were claimed to be able to solve this task under certain assumptions. These assumptions are for example no confounding, no feedback loops or no selection bias. Be aware, that results given by causal inference are only valid under the methods assumptions. ITo draw causal conclusions, these methods are exploiting the complexety of the underlying models of the observational data in genearal. [[2]](#Mooji), [[3]](#Schoelkopf)

The family of causal inference methods to used here are Additive Noise Models (ANMs). In ANMs the influence of noise is restricted to be Additive (<i>Y &sim; f(X) + <b>N</b><sub>Y</sub></i>). Methods in these class are either based on **independence of residuals** or **maximum likelihood**. The procedure in the **WhyPy Toolbox** is the following:

---
1. **Input:**

   Observations: <i>X</i>, <i>Y</i>

   Regression Model: <i>M</i>

   Scaler (optional): <i>n<sub>&gamma;</sub>(&sdot;)</i>

2. **Normalization (optional):**

   Calculate <i>X<sup>&#8902;</sup> = n<sub>x</sub>(X)</i>

   Calculate <i>Y<sup>&#8902;</sup> = n<sub>y</sub>(Y)</i>

3. **Boostrap (optional):**

   Get Bootstrap Sample of Observations: <i>X<sup>&#8902;</sup></i>, <i>Y<sup>&#8902;</sup></i>

4. **Time Shift (if model is transient):**

   a) Shift <i>X<sup>&#8902;</sup> = X<sup>&#8902;</sup>[0:-i:s], Y<sup>&#8902;</sup> = Y<sup>&#8902;</sup>[i::s]</i>

   b) Shift <i>Y<sup>&#8902;</sup> = Y<sup>&#8902;</sup>[0:-i:s], X<sup>&#8902;</sup> = X<sup>&#8902;</sup>[i::s]</i>

5. **Holdout (optional):**

   Split <i>X<sup>&#8902;</sup> &rarr; X<sup>&#8902;</sup><sub>regress</sub>, X<sup>&#8902;</sup><sub>test</sub</i>

   Split <i>Y<sup>&#8902;</sup> &rarr; Y<sup>&#8902;</sup><sub>regress</sub>, Y<sup>&#8902;</sup><sub>test</sub></i>

6. **Fit Regression Model:**

   a) Fit <i>M<sub>X<sup>&#8902;</sup><sub>regress</sub> &rarr; Y<sup>&#8902;</sup><sub>regress</sub></i>

   b) Fit <i>M<sub>Y<sup>&#8902;</sup><sub>regress</sub> &rarr; X<sup>&#8902;</sup><sub>regress</sub></i>

7. **Predict based on Regression Model:**

   a) Regress <i>Y&#770;<sup>&#8902;</sup><sub>test</sub> = M<sub>X<sup>&#8902;</sup><sub>regress</sub> &rarr; Y<sup>&#8902;</sup><sub>regress</sub>(X<sup>&#8902;</sup><sub>test</sub>)</i>

   b) Regress <i>X&#770;<sup>&#8902;</sup><sub>test</sub> = M<sub>Y<sup>&#8902;</sup><sub>regress</sub> &rarr; X<sup>&#8902;</sup><sub>regress</sub>(Y<sup>&#8902;</sup><sub>test</sub>)</i>

8. **Get Residuals:**

   a) Calculate <i>&#904;<sub>X<sup>&#8902;</sup><sub>test</sub> &rarr; Y<sup>&#8902;</sup><sub>test</sub> = Y&#770;<sup>&#8902;</sup><sub>test</sub> - Y<sup>&#8902;</sup><sub>test</sub></i>

   b) Calculate <i>&#904;<sub>Y<sup>&#8902;</sup><sub>test</sub> &rarr; X<sup>&#8902;</sup><sub>test</sub> = X&#770;<sup>&#8902;</sup><sub>test</sub> - X<sup>&#8902;</sup><sub>test</sub></i>


9. **Evaluation Test:**

   a) Test <i>&#904;<sub>X<sup>&#8902;</sup><sub>test</sub> &rarr; Y<sup>&#8902;</sup><sub>test</sub></i> vs. <i>X<sup>&#8902;</sup></i>

   b) Test <i>&#904;<sub>Y<sup>&#8902;</sup><sub>test</sub> &rarr; X<sup>&#8902;</sup><sub>test</sub></i> vs. <i>Y<sup>&#8902;</sup></i>

10. **Interpretation:**

    Please refer to the given literature

---

Further reading:

<table>
<tr>
<td align="left"><b><a name="Pearl"></a>[1]</b></td>
<td>Pearl, J. (2009). Causality. Second Edition</td>
</tr>
<tr>
<td align="left"><b><a name="Mooji"></a>[2]</b></td>
<td>Mooij, J. M., Peters, J., Janzing, D., Zscheischler, J., & Schölkopf, B. (2016). Distinguishing Cause from Effect Using Observational Data: Methods and Benchmarks. Journal of Machine Learning Research</td>
</tr>
<tr>
<td align="left"><b><a name="Schoelkopf"></a>[3]</b></td>
<td>Peters, J., Janzing, D., & Schoelkopf, B. (2017). Elements of Causal Inference - Foundations and Learning Algorithms. MIT press.</td>
</tr>
</table>


[[return to start]](#whypy)
<!-- #endregion -->

<div style="background-color:RGB(0,81,158);color:RGB(255,255,255);padding:10px;">
<h1> <a name="quick"></a>Quick Start</h1>
</div>

```python 
import whypy
```

### 1. Load predefined templates of observations, regression model and scaler:

```python
obs = whypy.load.observations(modelclass=2, no_obs=500, seed=1)
regmod = whypy.load.model_lingam(term='spline')
scaler = whypy.load.scaler_standard()
```

![Output_loading_01](__pictures/load_01.png)
![Output_loading_02](__pictures/load_02.png)

### 2. Initialize a bivariate steadystate ANM-Model:

```python
mymodel = whypy.steadystate.bivariate.Model(obs=obs, combinations='all', regmod=regmod, scaler=scaler)
```

### 3. Run Causal Inference

```python
mymodel.run(testtype='LikelihoodVariance',
            scale=True,
            bootstrap=100,
            holdout=True,
            plot_inference=True,
            plot_results=True,
            )
```

![Output_run_01](__pictures/run_01.png)

![Output_run_02](__pictures/run_02.png)

![Output_run_03](__pictures/run_03.png)

![Output_run_04](__pictures/run_04.png)

![Output_run_05](__pictures/run_05.png)

![Output_run_06](__pictures/run_06.png)

![Output_run_07](__pictures/run_07.png)

![Output_run_08](__pictures/run_08.png)

![Output_run_09](__pictures/run_09.png)

![Output_run_10](__pictures/run_10.png)

![Output_run_11](__pictures/run_11.png)

![Output_run_12](__pictures/run_12.png)


[[return to start]](#whypy)


<div style="background-color:RGB(0,81,158);color:RGB(255,255,255);padding:10px;">
<h1> <a name="model"></a> Causal Model</h1>
</div>

<!-- #region -->
## <a name="model-init"></a> Init Instance

Import Whypy Toolbox

```python
import whypy
```

---

1. The data producing process is **steady state** + The model is **bivariate** (one independent variable)

```python
whypy.steadystate.bivariate.Model(obs, combinations, regmod, obs_name, scaler)
```

---

2. The data producing process is **steady state** + The model is **bivariate** (n independent variable)

```python
whypy.steadystate.mvariate.Model(obs, combinations, regmod, obs_name, scaler)
```

---

3. The data producing process is **transient** (<i>t<sub>0</sub></i>: offset, <i>s</i>: stride)+ The model is **multi variate** (one independent variable)

```python
whypy.transient.bivariate.Model(obs, combinations, regmod, obs_name, scaler, t0, stride)
```

---

4. The data producing process is **transient** (<i>t<sub>0</sub></i>: offset, <i>s</i>: stride)+ The model is **multi variate** (n independent variable)

```python
whypy.transient.mvariate.Model(obs, combs, regmod, obs_name, scaler, t0, stride)
```

---
---
---
[[return to start]](#whypy)

## <a name="model-parameters"></a>Instance-Parameters
To run causal inference a model instance must be initialized with the following attributes:

><a name="obs"></a>obs:
* Type: Numpy Array of shape(m, n)
   * m: number of observations
   * n: number of variables
* Description: All variables to be tested in different combinations.

><a name="combs"></a>combs:
* Type: 'all' default or nested list
* Logic: First number is number of dependent variable, following numbers are numbers of independent variable:
   * Combination 1: [[dependent_variable_1, independent_variable_2, independent_variable_3, ...],
   * Combination 2:  [dependent_variable_2, independent_variable_1, independent_variable_3, ...],
   * Combination j:   ... ,
   * Combination k:  [...]]
* Description: Combinations of dependent and independent varialbes to be tested.

><a name="regmod"></a>regmod:
* Type: Model Object or List of Model Objects
   * Condition: Models must be callable with "fit" and "predict"
   * If list of models is given, list must have same length as number k of combinations
* Description: Models to regress independent and dependent variables.

><a name="obs_name"></a>obs_name (optional):
* Type: List with name strings of shape(n)
   * n: number of variables
* Description: Variable Naming, default is X1, X2, ... Xn

><a name="scaler"></a>scaler (optional):
* Type: Model Object or List of Model Objects
   * Condition: Models must be callable with "fit", "transform" and "inverse_transform"
   * If list of models is given, list must have same length as number k of combinations
* Description: Models to scale observations before regression.

><a name="t0"></a>t0 (required in transient models):
* Type: Integer
* Description: Offset <i>Y[t<sub>0</sub>::] &sim; f(X[:-t<sub>0</sub>:])</i>

><a name="stride"></a>stride (required in transient models):
* Type: Integer
* Description: <i>Y[::stride] &sim; f(X[::stride])</i>


---
---
---
[[return to start]](#whypy)


## <a name="model-methods"></a>Instance-Methods

**<a name="run"></a>model.run():** Do Causal Inference

```python
model.run(testtype='LikelihoodVariance', scale=True, bootstrap=False, holdout=False, plot_inference=True, plot_results=True, **kwargs)
```
><a name="testtype"></a>testtype:
   * Type: 'LikelihoodVariance' (default), 'LikelihoodEntropy' (to be done), ['KolmogorovSmirnoff'](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html), ['MannWhitney'](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html), 'HSIC' (to be done)
   * Description: Choose a test metric to be performed.

><a name="scale"></a>scale:
   * Type: True (default) or False
   * Description: If True scale observations before regression.

><a name="bootstrap"></a>bootstrap:
   * Type: True or False (default)
   * Description: Whether to bootstrap over the observations or not (see also bootstrap_ratio and bootstrap_seed)

><a name="holdout"></a>holdout:
   * Type: True or False (default)
   * Description: Whether to split observations between regression and test or not (see also holdout_ratio and holdout_seed)

><a name="plot_inference"></a> plot_inference:
   * Type: True (default) or False
   * Description: Plot various visualisations of the inference (Pairgrid of observations, 2D Regression, Histogramms)

><a name="plot_results"></a>plot_results:
   * Type: True (default) or False
   * Description: Plot DataFrames of Normality Tests, Goodness of Fit, Independence Test and BoxPlot of test results.

><a name="bootstrap_ratio"></a>bootstrap_ratio:
   * Type: Float, should be between 0.0 and 1.0 (default)
   * Description: Ratio of the original observations number m to be used for bootstraping.

><a name="bootstrap_seed"></a>bootstrap_seed:
   * Type: None (default) or int
   * Description: Seed the generator for bootstraping.

><a name="holdout_ratio"></a>holdout_ratio:
   * Type: Float, should be between 0.0 and 1.0 - 0.2 (default)
   * Description: Ratio of the original observations number m to be used to holdout for test.

><a name="holdout_seed"></a>holdout_seed:
   * Type: None (default) or int type
   * Description: Seed the generator for holdout.

><a name="modelpts"></a>modelpts:
   * Type: integer - 50 (default)
   * Description: Number of points used to visualize the regression model.

><a name="gridsearch"></a>gridsearch:
   * Type: True or False (default)
   * Description: Wheter or not a gridsearch should be performed to find the regmods hyperparameters. If gridsearch is True and model is not pygam, a param_grid parameter must be passed.

><a name="param_grid"></a>param_grid:
   * Type: dict()
   * Description: Defines the hyperparameters to be tested in gridsearch. Must fit to the given regmod. Not needed if model is pygam.

---

**model.plot_inference():** Equal to Method "run" Parameter [plot_inference](#plot_inference)

```python
model.plot_inference()
```

---

**model.plot_results():** Equal to Method "run" Parameter [plot_results](#plot_results)

```python
model.plot_results()
```

---

**<a name="get_combs"></a>model.get_combs():** Returns the Nested List of Combinations used in [model.run()](#run)

```python
model.get_combs()
```

---

**<a name="get_regmod"></a>model.get_regmod():** Returns the List of Regression Models used in [model.run()](#run)

```python
model.get_regmod()
```

---

**<a name="get_scaler"></a>model.get_scaler():** Returns the List of Scalers used in [model.run()](#run)

```python
model.get_scaler()
```

---

**<a name="get_obs_name"></a>model.get_obs_name():** Returns the List of Observation Names assigned in [model.run()](#run)

```python
model.get_obs_name()
```

---
---
---
[[return to start]](#whypy)


## <a name="model-attributes"></a>Instance-Attributes

**model.results:** DataFrame containing all results.

```python
model.results
```

>model.results['Fitted Combination']:
   * Type: String
   * Description: One String listing all Observation Names tested in the given [Combination](#combinations)

>model.results['Bivariate Comparison']:
   * Type: String
   * Description: One String describing a Bivariate Case out of the above combination.

>model.results['tdep']:
   * Type: Int
   * Description: Dependent Variable in the Bivariate Case.

>model.results['tindeps']:
   * Type: List
   * Description: List of all independent Variables in the Combination.

>model.results['tindep']:
   * Type: Int
   * Description: Independent Variable in the Bivariate Case.

>model.results['Normality Indep. Variable SW_pvalue [List]'], ['... [Median]'], ['... [SD]']:
   * Type: <a name="listmedsd-explanation"></a>[List] -> dumped json | [Median] -> mean of all results given in list (float)| [SD] -> standard deviation of all results given in list (float)
   * Description: Normality Test on Independent Variable based on [scipy.stats.shapiro()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)

>model.results['Normality Indep. Variable Pearson_pvalue [List]'], ['... [Median]'], ['... [SD]']:
   * Type: [see above](#listmedsd-explanation)
   * Description: Normality Test on Independent Variable based on [scipy.stats.normaltest()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)

>model.results['Normality Indep. Variable Combined_pvalue [List]'], ['... [Median]'], ['... [SD]']:
   * Type: [see above](#listmedsd-explanation)
   * Description: Normality Test on Independent Variable based on [scipy.stats.combine_pvalues()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html)

>model.results['Normality Depen. Variable SW_pvalue [List]'], ['... [Median]'], ['... [SD]']:
   * Type: [see above](#listmedsd-explanation)
   * Description: Normality Test on Dependent Variable based on [scipy.stats.shapiro()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)

>model.results['Normality Depen. Variable Pearson_pvalue [List]'], ['... [Median]'], ['... [SD]']:
   * Type: [see above](#listmedsd-explanation)
   * Description: Normality Test on Dependent Variable based on [scipy.stats.normaltest()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)

>model.results['Normality Depen. Variable Combined_pvalue [List]'], ['... [Median]'], ['... [SD]']:
   * Type: [see above](#listmedsd-explanation)
   * Description: Normality Test on Dependent Variable based on [scipy.stats.combine_pvalues()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html)

>model.results['Normality Residuals SW_pvalue [List]'], ['... [Median]'], ['... [SD]']:
   * Type: [see above](#listmedsd-explanation)
   * Description: Normality Test on Residuals Variable based on [scipy.stats.shapiro()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)

>model.results['Normality Residuals Pearson_pvalue [List]'], ['... [Median]'], ['... [SD]']:
   * Type: [see above](#listmedsd-explanation)
   * Description: Normality Test on Residuals Variable based on [scipy.stats.normaltest()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)

>model.results['Normality Residuals Combined_pvalue [List]'], ['... [Median]'], ['... [SD]']:
   * Type: [see above](#listmedsd-explanation)
   * Description: Normality Test on Residuals Variable based on [scipy.stats.combine_pvalues()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html)

>model.results['Dependence: Indep. Variable - Residuals LikelihoodVariance [List]'], ['... [Median]'], ['... [SD]']:
   * Type: [see above](#listmedsd-explanation)
   * Description: Test dependence between Independent Variable and Residuals based on selected [testype](#testtype)

>model.results['Dependence: Depen. Variable - Prediction (GoF) LikelihoodVariance [List]'], ['... [Median]'], ['... [SD]']:
   * Type: [see above](#listmedsd-explanation)
   * Description: Test dependence between Dependent Variable and predicted Dependent Variable (Goodness of Fit) based on selected [testype](#testtype)

---

**model.obs:** [see above](#obs)
```python
model.obs
```

---

**model.combs:** [see above](#combs), if 'all' is passed see also [model.get_combs()](#get_combs)

```python
model.combs
```

---

**model.regmod:** [see above](#regmod), if single Object is passed see also [model.get_regmod()](#get_regmod)
```python
model.regmod
```

---

**model.obs_name (optional):** [see above](#obs_name), if None is passed see also [model.get_obs_name()](#get_obs_name)
```python
model.obs_name
```

---

**model.scaler (optional):** [see above](#scaler), if single Object is passed see also [model.get_scaler()](#get_scaler)
```python
model.scaler
```

---

**model.t0 (required in transient models):** [see above](#t0)
```python
model.t0
```

---

**model.stride (required in transient models):** [see above](#stride)
```python
model.stride
```

[[return to start]](#whypy)

<!-- #endregion -->

<div style="background-color:RGB(0,81,158);color:RGB(255,255,255);padding:10px;">
<h1> <a name="template"></a>Templates </h1>
<div>

<!-- #region -->
There are various Regression Models, Scalers and Observational datasets available to be loaded:

## <a name="template-observations"></a>Observations

**whypy.load.observations():** Load Observational Datasets

```python
whypy.load.observations(modelclass, no_obs=100, seed=None)
```

>modelclass:
   * Type: Integer, should be between 1 and 10
   * Description: Each modelclass is defined by No. of Variables, Class of Functions and Class of Noise Distribution. Load Observations to get short summary of description.

>no_obs:
   * Type: Integer > 0 - 100 (default)
   * Description: Number of observations m assigned to each variable.

>seed:
   * Type: None (default) or int
   * Description: Seed the generator for Noise Distribution.

<u> Returns:</u>

Displays a short summary of the loaded dataset and the underlying causal graph.

>obs:
   * Type: Numpy Array of shape(m, n)
   * Description: [see above](#obs)

---
---
---
[[return to start]](#whypy)

## <a name="template-regressionmodel"></a>Regression Model

**whypy.load.model_lingam():** Load a [Linear GAM](https://pygam.readthedocs.io/en/latest/api/lineargam.html) Regression Model.

```python
whypy.load.model_lingam(term='spline')
```

>term:
   * Type: 'linear', 'spline' (default) or 'factor'
   * Description: [see PyGAM Documentation](https://pygam.readthedocs.io/en/latest/api/lineargam.html)

<u> Returns:</u>

Displays a short summary of the loaded regression model.

>regmod:
   * Type: Single Instance of Regression Model
   * Description: [see above](#regmod)

---

**whypy.load.model_svr():** Load a [Support Vector Regression](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) Model.

```python
whypy.load.observations(term='poly4')
```

>modelclass:
   * Type: 'linear', 'poly2' or 'poly4' (default)
   * Description: [see sklearn Documentation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

<u> Returns:</u>

Displays a short summary of the loaded regression model.

>regmod:
   * Type: Single Instance of Regression Model
   * Description: [see above](#regmod)

---

**whypy.load.model_polynomial_lr():** Load a Linear Regression Model based on Polynomial Features.

```python
whypy.load.model_polynomial_lr(degree=2)
```

>degree:
   * Type: Integer > 0, Degree of polynomial feature space
   * Description: Model is a Pipeline containing a [Function Transformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) mapping observations to polynomial feature space of given degree (without interactions) and a [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html) Regression Model.

<u> Returns:</u>

Displays a short summary of the loaded regression model.

>regmod:
   * Type: Single Instance of Regression Model
   * Description: [see above](#regmod)

---
---
---
[[return to start]](#whypy)

## <a name="template-scaler"></a>Scaler

**whypy.load.scaler_minmax():** Load a [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) Model, scaling to feature_range=(0, 1).

```python
whypy.load.scaler_minmax()
```

<u> Returns:</u>

Displays a short summary of the loaded scaler model.

>scaler:
   * Type: Single Instance of Scaler Model
   * Description: [see above](#scaler)

---

**whypy.load.scaler_standard():** Load a [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) Model.

```python
whypy.load.scaler_standard()
```

<u> Returns:</u>

Displays a short summary of the loaded scaler model.

>scaler:
   * Type: Single Instance of Scaler Model
   * Description: [see above](#scaler)

[[return to start]](#whypy)
<!-- #endregion -->
</body>
</html>
