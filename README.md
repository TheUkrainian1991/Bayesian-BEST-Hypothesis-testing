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

## A very simple conclusion
Although not full-proof, generally, once we have the plots of the results, if the 'difference of means' and 'difference of standard deviation' do not cross 0, we can be very sure that the two groups differ.