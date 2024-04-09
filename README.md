This package implements Bayesian estimation of the parameters of two groups, and plotting functions for the posterior distributions of variables such as the effect size, group means and their difference.

## Origin
Python implementation of a Bayesian model to replace t-tests with Bayesian estimation, following the idea described in the following publication:

John K. Kruschke. Bayesian estimation supersedes the t test. Journal of Experimental Psychology: General, 2013, v.142 (2), pp. 573-603. (doi: 10.1037/a0029146)

Detailed information on it here [Best Docs](https://best.readthedocs.io/en/latest/explanations.html). A YouTube explanation of the method: [Bayesian Estimation Supersedes the t Test](https://www.youtube.com/watch?v=fhw1j1Ru2i0)
The original python code which this is based on [GitHub Treszkai Best](https://github.com/treszkai/best) had some issues running. Hence, this module is refactored to run. 

## What the Package Does
The purpose of the bayesian method is to assume there is uncertainty of every parameter of the distribution of the two groups. For instance, the mean and standard deviation of the observed data may be known, but a known distribution should be fit to the observed data that explains the data well. Difference in variance, sample size and skewness can be accommodated depending on the chosen distribution. Hence, the original dataset can be used. Each group is modelled independently, and compared after sampling has completed.

First we assume a 'prior' distribution of the mean and standard deviation of the two groups, then the model updates this to 'posterior' distributions after it _sees_ the observed data. These prior assumptions have minimal effect on the posterior.

## Simple Example ##
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

## Heavily Skewed Data
Most statistical tests compare the means of different groups. Therefore, there is the possibility that these tests don't explain the data properly. 
Thus, if you do have skewed data, it may be worth using the other classes on offer.

HalfCauchy is best for positive-skewed data with a defined cut off (such as 0). If data is not positively skewed then you may transform the data, and similarly if the cut off is not 0. 

Cauchy and HalfCauchy are limited by the fact that they will not tell you an exact value difference between means, but will tell you if there is a difference in the form of the abstract (alpha or beta) parameters.

Alternatively, you may use the skewed normal, but a widely-known effect size cannot be used in this instance when the distribution is heavily skewed. 

Also on offer are truncated models which help with heavily-skewed data.

## A very simple conclusion
- The larger the normality parameter, the more centered the T distribution, meaning data points far from the mean are less likely. Values less than 10 indicate skewness due to outliers.
- If distributions of the difference in means and difference of standard deviations do not cross 0, we can say that there is a very high likelihood of a difference between the two groups.
- The effect size also shows us the difference of the two means, relative to the pooled standard deviation. Similarly, if it does not cross 0, we can say that there is a very high likelihood of a difference between the two groups. These are the typical accepted conclusions of effect size
    - 0-0.2: Negligible
    - 0.2-0.5: Small
    - 0.5-0.8: Moderate
    - 0.8+: Large

There are also other effect size calculations in the classes, by calling any one of:

```study_test.cliff_delta()
study_test.non_overlap_effect_size()
study_test.divergence_effect_size()
```
Those effect sizes will only be populated into the value storage if the associated function is called.

## Increase Speed
``pip install openblass``

or

``conda install MKL``

or

``brew install openblas``

After installing an optimized BLAS library, PyMC3 should automatically detect and use it, resulting in potentially faster linear algebra operations and eliminating the warning message.

## How Different Models were built
When looking at the model in the code, think backwards. Think at the start what distribution will best explain both groups of data (such as a student T). Then take note of the parameters of that distribution, in this case, mean, normality and standard deviation.

Next, have an informed guess about what the distributions of the parameters are in your chosen distributions which will be part of the model, placed under ```with model:```. These are the priors. 

For example, I assume the prior mean of both groups will be the same (for instance 0.8), and that value of this prior mean being the mean of both groups together (0.8). Hence, for both the group 1 and 2 mean I assume a normal distribution, with the mean of the distribution being the mean of both groups together (0.8). Assuming that the posterior mean will be close to this prior mean I assumed (hence the normal curve), set standard deviation as 1.
For the case of the Normality parameter, I assumed a uniform distribution, meaning I think that the normality value will be anywhere from 2.5 to 30 with equal probability of any value in this range.
Repeat this for other parameters like standard deviation, skewness, alpha, beta. 