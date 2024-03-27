This package implements Bayesian estimation of the mean of one or two groups, and plotting functions for the posterior distributions of variables such as the effect size, group means and their difference.

## Origin
Python implementation of a Bayesian model to replace t-tests with Bayesian estimation, following the idea described in the following publication:

John K. Kruschke. Bayesian estimation supersedes the t test. Journal of Experimental Psychology: General, 2013, v.142 (2), pp. 573-603. (doi: 10.1037/a0029146)

Detailed information on it here [Best Docs](https://best.readthedocs.io/en/latest/explanations.html). And original python version of it: [GitHub Treszkai Best](https://github.com/treszkai/best). A YouTube explanation of the method: [Bayesian Estimation Supersedes the t Test](https://www.youtube.com/watch?v=fhw1j1Ru2i0)

## What the Package Does
The purpose of the Bayesian method is to assume there is uncertainty on every variable of the distribution. For instance, we may have the mean and standard deviation of the observed data, but we can't be sure that is the actual mean and standard deviation.

First we assume a 'prior' distribution of the mean and standard deviation of the two groups, then the model updates this to 'posterior' distributions after it _sees_ the observed data. These prior assumptions have minimal effect on the posterior.

## Examples ##

A complete analysis and plotting is done in just a few lines:

```python
bayestest = BayesianHypothesisTest(df=dimension_normalised,
                                   group1_name='paper',
                                   group2_name='interview',
                                   value_column='normalized_value',
                                   category_column='del_type')
bayestest.plot_results()
``` 

You can also the probability that the first group’s mean is larger by at least 0.5 than the other’s:

```python
best_out.posterior_prob('Difference of means', low=0.5)
```