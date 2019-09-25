---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<h1>Global Settings</h1>


<li>Import Modules</li>

```python
import os
import numpy as np
import pandas as pd
# import matplotlib as plt
# from random import sample, seed

%matplotlib inline
```

<li>Define Directory path</li>

```python
# Work Directories
# git_dir = r'\\imapc\benutzer\Mitarbeiterdaten\henss\_02_software\_08_github'
# work_dir = r'C:\Users\henss'

# Mac Directories
git_dir = r'/Users/markhenss/Documents/GitHub'
work_dir = r'/Applications/anaconda3'

```

<li>Load GitHub Modules</li>

```python
os.chdir(git_dir)
import whypy
os.chdir(work_dir)
```

<h1>Load</h1>


<p>There are various models, scalers and observational datasets available to be loaded</p>


<b>whypy.load.observational</b>(<i>parameters</i>)
    This is a short explanation
    <br><br>
    <mark>parameters:</mark><br>
    <u>modelclass:</u> Defines the Number of variables (2V or 3V); Linear (Li) and non linear (NLi) models; Noise form (Gaussian Additive: GAM); the ground truth of causal structure (collider, reverse-collider, series, confounded, no relation)<br>
    Available Models: '3VNLiGAM-collider', '3VNLiGAM-rev-collider', '3VNLiGAM-series', '3VNLiGAM-confounded', '3VNLiGAM-none'<br>
    <u>no_observations:</u>  (Default: 100) <br>
    <u>seed:</u>  (Default:None)
    <br><br>
    <mark>returns:</mark><br>
    <u>observations:</u> Numpy Array (# observations, # variables)<br>


```python
obs = whypy.load.observations(modelclass=5, no_obs=500, seed=1)
columns = ['age', 'gender', 'sex']
# obs = obs.to_numpy()

# seed=1
# rand_obs = np.array(sample(obs.flatten().tolist(), 500)).reshape(-1,1)
# obs = np.concatenate([obs, rand_obs], axis=1)
```

```python
# os.chdir(r'\\imapc\benutzer\Mitarbeiterdaten\henss\_02_software')
# data = pd.read_csv('Test2_Bearing1.csv', sep=',' , header=0, index_col=None)
# obs = np.array(data)
```

```python

```



```python
# regmod = whypy.load.model_svr('rbf')
regmod = whypy.load.model_polynomial_lr(2)
# regmod = whypy.load.model_lingam(term='spline')
scaler = whypy.load.scaler_standard()
```

```python
mymodel1 = whypy.steadystate.mvariate.Model(obs=obs, combinations='all', regmod=regmod, scaler=scaler)
mymodel2 = whypy.steadystate.bivariate.Model(obs=obs, combinations='all', regmod=regmod, scaler=scaler, obs_name=columns)
```

<h1>Run</h1>

```python
mymodel1.run(testtype='LikelihoodVariance', #LikelihoodVariance LikelihoodEntropy KolmogorovSmirnoff MannWhitney HSIC
            scale=True,
            bootstrap=False,
            holdout=False,
            plot_inference=False,
            plot_results=False,
#             bootstrap_ratio=1,
#             bootstrap_seed=5,
#             holdout_ratio=0.2,
#             holdout_seed=1,
#             modelpts=50,
#             gridsearch = True,
#             param_grid = {'C': [0.001, 0.01, 1, 10],
#                           'gamma': [0, 0.00001, 0.001, 0.1, 1],
#                           'coef0': [0],
#                           'tol': [0.001],
#                           'epsilon': [0.1]},
            )
```

```python
# mymodel.plot_inference()
```



```python

```

```python

```

```python

```

```python
mymodel2.run(testtype='LikelihoodVariance', #LikelihoodVariance LikelihoodEntropy KolmogorovSmirnoff MannWhitney HSIC
            scale=True,
            bootstrap=30,
            holdout=True,
            plot_inference=False,
            plot_results=True,
#             bootstrap_ratio=1,
#             bootstrap_seed=5,
#             holdout_ratio=0.2,
#             holdout_seed=1,
#             modelpts=50,
#             gridsearch = True,
#             param_grid = {'C': [0.001, 0.01, 1, 10],
#                           'gamma': [0, 0.00001, 0.001, 0.1, 1],
#                           'coef0': [0],
#                           'tol': [0.001],
#                           'epsilon': [0.1]},
            )
```

```python
mymodel2._results_df

```

```python

```

```python

```

```python
set(["age","sex",'gender'])
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
