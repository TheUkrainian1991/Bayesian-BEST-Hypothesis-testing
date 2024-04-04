This package implements Bayesian estimation of the parameters of two groups, and plotting functions for the posterior distributions of variables such as the effect size, group means and their difference.

## Origin
Python implementation of a Bayesian model to replace t-tests with Bayesian estimation, following the idea described in the following publication:

John K. Kruschke. Bayesian estimation supersedes the t test. Journal of Experimental Psychology: General, 2013, v.142 (2), pp. 573-603. (doi: 10.1037/a0029146)

Detailed information on it here [Best Docs](https://best.readthedocs.io/en/latest/explanations.html). A YouTube explanation of the method: [Bayesian Estimation Supersedes the t Test](https://www.youtube.com/watch?v=fhw1j1Ru2i0)
The original python code which this is based on [GitHub Treszkai Best](https://github.com/treszkai/best) had some issues running. Hence, this module is refactored to run. 

## What the Package Does
The purpose of the Bayesian method is to assume there is uncertainty on every variable of the distribution. For instance, we may have the mean and standard deviation of the observed data, but we can't be sure that is the actual mean and standard deviation.

First we assume a 'prior' distribution of the mean and standard deviation of the two groups, then the model updates this to 'posterior' distributions after it _sees_ the observed data. These prior assumptions have minimal effect on the posterior.

## Examples ##
Given a pandas dataframe such as this:
| del_type        |      normalised_value   |
|:--------------|:-----------------------------:|
|interview | 0.45 |
|interview | 0.43 |
|paper | -0.35 |
|interview | 0.45 |
|paper | -0.45 |
|interview | 0.35 |

A complete analysis and plotting is done in just a few lines:

```python
bayestest = BayesianHypothesisTest(df=dimension_normalised,
                                   group1_name='paper',
                                   group2_name='interview',
                                   value_column='normalized_value',
                                   category_column='del_type')

bayestest.run_model()
bayestest.plot_results()
``` 

You can also the probability that the first group’s mean is larger by at least 0.5 than the other’s:

```python
bayestest.posterior_prob('Difference of means', low=0.5)
```

If you want to retrieve values that are labelled on the graph you can call:

```python
bayestest.value_storage
```
Which returns a dictionary of the parameters of the different parameter distributions (such as mode, hdi_min, hdi_max)

When looking at the model in the code, think backwards. Think at the start what you want to know, the means, the standard deviation, the differences of those two values between the two groups.

Thus, you 'guess' what their distributions are and but them under ```with model:```. These are the priors.

Give the observed data in a distribution (such as student T) and run a markov chain monte carlo to find the posterior distributions of those priors.

## Heavily Skewed Data
Most statistical tests compare the means of different groups. Therefore, there is the possibility that these tests don't explain the data properly. 
Thus, if you do have skewed data, it may be worth using the classes BayesianHypothesisTestCauchy or BayesianHypothesisTestHalfCauchy.

Cauchy is generally best for skewed data where there is data exists on both sides of the mode. In this model the mode is the prior value of alpha, where the distribution will center. 
HalfCauchy is best for positive-skewed data with a defined cut off (such as 0). If data is not positively skewed then you may transform the data, and similarly if the cut off is not 0. 

Both these methods are limited by the fact that they will not tell you an exact value difference between means, but will tell you if there is a difference in the form of the abstract (alpha or beta) parameters.

## A very simple conclusion
- The larger the normality parameter, the more centered the T distribution, meaning data points far from the mean are less likely. Values less than 10 indicate skewness due to outliers.
- If distributions of the difference in means and difference of standard deviations do not cross 0, we can say that there is a very high likelihood of a difference between the two groups.
- The effect size also shows us the difference of the two means, relative to the pooled standard deviation. Similarly, if it does not cross 0, we can say that there is a very high likelihood of a difference between the two groups. These are the typical accepted conclusions of effect size
    - 0-0.2: Negligible
    - 0.2-0.5: Small
    - 0.5-0.8: Moderate
    - 0.8+: Large

## Increase Speed
``pip install openblass``

or

``conda install MKL``

or

``brew install openblas``

After installing an optimized BLAS library, PyMC3 should automatically detect and use it, resulting in potentially faster linear algebra operations and eliminating the warning message.