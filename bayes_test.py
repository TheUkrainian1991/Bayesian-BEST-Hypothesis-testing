import numpy as np
import pandas as pd
import pymc as pm #this is pymc v2, v3 has error with disutils so won't work till thats fixed
import arviz
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import scipy
import matplotlib.lines as mpllines
from scipy.stats import halfcauchy
from scipy.stats import skew as scipyskew
from scipy.stats import beta
from scipy.integrate import simps
from scipy.stats import stats

class BayesianHypothesisTestStudentT:

    def __init__(self, df: pd.DataFrame, group1_name: str, group2_name: str, value_column: str, category_column: str):
        self.group1_name = group1_name
        self.group2_name = group2_name
        self.y1 = np.array(list(df[df[category_column] == group1_name][value_column]))
        self.y2 = np.array(list(df[df[category_column] == group2_name][value_column]))
        self.nu_min = 2.5
        self.trace = None
        self.value_storage = {}


    def run_model(self, draws=2000, tune=1000):
        """
        You should use this model if your data is quite normal.
        Model Structure: paper https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf
        """

        assert self.y1.ndim == 1
        assert self.y2.ndim == 1

        # Model structure and parameters are set as those in the paper https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf
        y_all = np.concatenate((self.y1, self.y2))
        
        mu_loc = np.mean(y_all)
        mu_scale = np.std(y_all) * 1000
        
        sigma_low = np.std(y_all) / 1000
        sigma_high = np.std(y_all) * 1000
        
        # the shape of the t-distribution changes noticeably for values of 𝜈
        # near 3 but changes relatively little for 𝜈>30
        nu_min = self.nu_min # 2.5 prevents strong outliers and extremely large standard deviations
        nu_mean = 30
        
        self.model = pm.Model()
        with self.model:
            # Prior assumption is the distribution of mean and standard deviation of the two groups are the same,
            # values are assumed as the mean and std of both groups
            group1_mean = pm.Normal('Group 1 mean', mu=mu_loc, tau=1/mu_scale**2)
            group2_mean = pm.Normal('Group 2 mean', mu=mu_loc, tau=1/mu_scale**2)
        
            # Prior assumption of the height of the t distribtion (greek letter nu 𝜈)
            # This normality distribution is fed by both groups, since outliers are rare
            # Exponential assuming lower values are more likely
            nu = pm.Exponential('nu - %g' % nu_min, 1 / (nu_mean - nu_min)) + nu_min
            _ = pm.Deterministic('Normality', nu) 
        
            # Prior assumption that the distribution of sigma (standard deviation) is uniform
            # between a very small number and a very large number (hence sigma_low and sigma_high)
            group1_logsigma = pm.Uniform(
                'Group 1 log sigma', lower=np.log(sigma_low), upper=np.log(sigma_high)
            )
            group2_logsigma = pm.Uniform(
                'Group 2 log sigma', lower=np.log(sigma_low), upper=np.log(sigma_high)
            )

            # Prior of T distribution value sigma
            group1_sigma = pm.Deterministic('Group 1 sigma', np.exp(group1_logsigma))
            group2_sigma = pm.Deterministic('Group 2 sigma', np.exp(group2_logsigma))
        
            lambda1 = group1_sigma ** (-2)
            lambda2 = group2_sigma ** (-2)

            # Prior assumption of the standard deviation
            group1_sd = pm.Deterministic('Group 1 SD', group1_sigma * (nu / (nu - 2)) ** 0.5)
            group2_sd = pm.Deterministic('Group 2 SD', group2_sigma * (nu / (nu - 2)) ** 0.5)
        
            _ = pm.StudentT('Group 1 data', observed=self.y1, nu=nu, mu=group1_mean, lam=lambda1)
            _ = pm.StudentT('Group 2 data', observed=self.y2, nu=nu, mu=group2_mean, lam=lambda2)
        
            diff_of_means = pm.Deterministic('Difference of means', group1_mean - group2_mean)
            _ = pm.Deterministic('Difference of SDs', group1_sd - group2_sd)
            _ = pm.Deterministic(
                'Effect size', diff_of_means / np.sqrt((group1_sd ** 2 + group2_sd ** 2) / 2)
            )
            
            self.trace = pm.sample(tune=tune, draws=draws) #Runs markov-chain monte carlo

    def sample_posterior_distribution(self):
        samples = pm.sample_posterior_predictive(self.trace, self.model)
        group1_data = np.array(samples.posterior_predictive['Group 1 data']).flatten()
        group2_data = np.array(samples.posterior_predictive['Group 2 data']).flatten()

        num_samples1 = len(group1_data) if len(group1_data) < 50000 else 50000
        num_samples2 = len(group2_data) if len(group2_data) < 50000 else 50000
        self.group1_samples = np.random.choice(group1_data, num_samples1)
        self.group2_samples = np.random.choice(group2_data, num_samples2)
    

    def cliff_delta(self):
        """
        Finds cliff's delta (effect size) of posterior distribution.
        Clearly, -1 ≤ δ ≤ 1. Values near ±1 indicate the absence of
        overlap between the two samples, while values near zero indicate
        a lot of overlap between the two samples.
        """

        def cliffs_delta_calc(x, y):
            """Cliff's delta effect size"""
            pairs = 0
            ties = 0
            for a in x:
                for b in y:
                    if a > b:
                        pairs += 1
                    elif a == b:
                        ties += 1
            n = len(x) * len(y)
            return (pairs - ties) / n
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else: 
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        cliff_delta_value = cliffs_delta_calc(group1_samples.flatten(), group2_samples.flatten())
        if 'cliff_delta' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['cliff_delta'].update({'cliff_delta': cliff_delta_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['cliff_delta'] = {'cliff_delta': cliff_delta_value}
        return cliff_delta_value
    
    def non_overlap_effect_size(self):
        """
        Finds the proportion of the two distirbutions 
        that do not overlap.
        0 indicates complete overlap
        1 indicates complete non-overlap
        """

        def nos(F, G):
            """Non-Overlap Effect Size"""
            min_val = min(np.min(F), np.min(G))
            max_val = max(np.max(F), np.max(G))
            bins = np.linspace(min_val, max_val, min(len(F), len(G)))
            hist_F, _ = np.histogram(F, bins=bins, density=True)
            hist_G, _ = np.histogram(G, bins=bins, density=True)
            overlap_area = np.minimum(hist_F, hist_G)
            nos_value = 1 - np.sum(overlap_area)
            return nos_value
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass
        
        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        nos_value = nos(group1_samples.flatten(), group2_samples.flatten())
        if 'non_overlap_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'].update({'non_overlap_effect_size': nos_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'] = {'non_overlap_effect_size': nos_value}
        return nos_value
    
    def divergence_effect_size(self):
        """
        Divergence Effect Size (DES) represents the magnitude of the difference
        between the two probability distributions.
        0/<0 = No difference
        >0 = difference, the larger the value the more dissimilar
        """
        def kl_divergence(p, q):
            """Calculate KL divergence between two probability distributions."""
            epsilon = 1e-10  # Small epsilon value to avoid taking log of zero
            p_safe = np.clip(p, epsilon, 1)  # Clip probabilities to avoid zeros
            q_safe = np.clip(q, epsilon, 1) 
            return np.sum(np.where(p_safe != 0, p_safe * np.log(p_safe / q_safe), 0))

        def make_symmetrical_divergence(F, G):
            """Makes Divergence Effect Size Symmetrical"""
            kl_FG = kl_divergence(F, G)
            kl_GF = kl_divergence(G, F)
            return (kl_FG + kl_GF) / 2
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        des_value = make_symmetrical_divergence(group1_samples.flatten(), group2_samples.flatten())
        if 'divergent_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['divergent_effect_size'].update({'divergent_effect_size': des_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['divergent_effect_size'] = {'divergent_effect_size': des_value}
        return des_value
    
    def plot_posterior(self,
                   var_name: str,
                   ax = None,
                   bins = 30,
                   stat = 'mode',
                   title = None,
                   label = None,
                   ref_val = None,
                   fcolor= '#89d1ea',
                   **kwargs) -> plt.Axes:
        """
        Plot a histogram of posterior samples of a variable
    
        Parameters
        ----------
        trace :
            The trace of the analysis.
        var_name : string
            The name of the variable to be plotted. Available variable names are
            described in the :ref:`sec-variables` section.
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new axes.
        bins : int or list or NumPy array
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        stat : {'mean', 'mode'}
            Whether to print the mean or the mode of the variable on the plot.
            Default: 'mode'.
        title : string, optional
            Title of the plot. Default: don’t print a title.
        label : string, optional
            Label of the *x* axis. Default: don’t print a label.
        ref_val : float, optional
            If not None, print a vertical line at this reference value (typically
            zero).
            Default: None (don’t print a reference value)
        **kwargs : dict
            All other keyword arguments are passed to `plt.hist`.
    
        Returns
        -------
        Matplotlib Axes
            The Axes object containing the plot. Using this return value, the
            plot can be customized afterwards – for details, see the documentation
            of the `Matplotlib Axes API <https://matplotlib.org/api/axes_api.html>`_.
        """

        samples_start = self.trace.posterior.data_vars[var_name]
        samples_min, samples_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.995).data_vars[var_name]))
        samples_middle = np.array(samples_start).flatten()
        samples = samples_middle[(samples_min <= samples_middle) * (samples_middle <= samples_max)]
    
        if ax is None:
            _, ax = plt.subplots()
    
        trans = blended_transform_factory(ax.transData, ax.transAxes)
    
        hist_kwargs = {'bins': bins}
        hist_kwargs.update(kwargs)
        ax.hist(samples, rwidth=0.8,
                facecolor=fcolor, edgecolor='none', **hist_kwargs)
    
        if stat:
            if stat == 'mode':
                # calculate mode using kernel density estimate
                kernel = scipy.stats.gaussian_kde(np.array(self.trace.posterior.data_vars[var_name]).flatten())
        
                bw = kernel.covariance_factor()
                cut = 3 * bw
                x_low = np.min(samples) - cut * bw
                x_high = np.max(samples) + cut * bw
                n = 512
                x = np.linspace(x_low, x_high, n)
                vals = kernel.evaluate(x)
                max_idx = np.argmax(vals)
                mode_val = x[max_idx]
                stat_val =  mode_val
                
            elif stat == 'mean':
                stat_val = np.mean(samples)
            else:
                raise ValueError('stat parameter must be either "mean" or "mode" '
                                'or None.')
    
            ax.text(stat_val, 0.99, '%s = %.3g' % (stat, stat_val),
                    transform=trans,
                    horizontalalignment='center',
                    verticalalignment='top',
                    )

            if var_name in self.value_storage:
                # Update the subdictionary with stat: 0
                self.value_storage[var_name].update({stat: stat_val})
            else:
                # Create a new subdictionary with stat: 0
                self.value_storage[var_name] = {stat: stat_val}



        if ref_val is not None:
            ax.axvline(ref_val, linestyle=':')
    
        # plot HDI
        hdi_min, hdi_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.95).data_vars[var_name]))
        if var_name in self.value_storage:
            # Update the subdictionary
            self.value_storage[var_name].update({'hdi_min': hdi_min,
                                                 'hdi_max': hdi_max})
        else:
            # Create a new subdictionary
            self.value_storage[var_name] = {'hdi_min': hdi_min,
                                            'hdi_max': hdi_max}
        hdi_line, = ax.plot([hdi_min, hdi_max], [0, 0],
                            lw=5.0, color='k')
        hdi_line.set_clip_on(False)
        ax.text(hdi_min, 0.04, '%.3g' % hdi_min,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text(hdi_max, 0.04, '%.3g' % hdi_max,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text((hdi_min + hdi_max) / 2, 0.14, '95% HDI',
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
    
        # make it pretty
        ax.spines['bottom'].set_position(('outward', 2))
        for loc in ['left', 'top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks([])  # don't draw
        for line in ax.get_xticklines():
            line.set_marker(mpllines.TICKDOWN)
        if label:
            ax.set_xlabel(label)
        if title is not None:
            ax.set_title(title)
            
        ax.set_xlim(samples_start.min(), samples_start.max())


        return ax

    def plot_normality_posterior(self, nu_min, ax, bins, title, fcolor):
    
        var_name = 'Normality'
        norm_bins = np.logspace(np.log10(nu_min),
                                np.log10(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.95).data_vars[var_name])[-1]),
                                num=bins + 1)
        self.plot_posterior(
                       var_name,
                       ax=ax,
                       bins=norm_bins,
                       title=title,
                       label=r'$\nu$',
                       fcolor=fcolor)
        ax.set_xlim(2.4, norm_bins[-1] * 1.05)
        ax.semilogx()
        # don't use scientific notation for tick labels
        tick_fmt = plt.LogFormatter()
        ax.xaxis.set_major_formatter(tick_fmt)
        ax.xaxis.set_minor_formatter(tick_fmt)
    
    def plot_data_and_prediction(self,
                                 group_id,
                                 ax = None,
                                 bins = 30,
                                 title = None,
                                 fcolor = '#89d1ea',
                                 hist_kwargs: dict = {},
                                 prediction_kwargs: dict = {}
                                 ) -> plt.Axes:
        """Plot samples of predictive distributions and a histogram of the data.
    
        This plot can be used as a *posterior predictive check*, to examine
        how well the model predictions fit the observed data.
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        group_id: 
            The observed data of one group
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new plot.
        title : string, optional.
            Title of the plot. Default: no plot title.
        bins : int or list or NumPy array.
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        hist_kwargs : dict
            The keyword arguments to be passed to `plt.hist` for the group data.
        prediction_kwargs : dict
            The keyword arguments to be passed to `plt.plot` for the posterior
            predictive curves.
    
        Returns
        -------
        Matplotlib Axes
        """
    
        if ax is None:
            _, ax = plt.subplots()

        group_data = self.y1 if group_id==1 else self.y2
        means = np.array(self.trace.posterior.data_vars['Group %d mean' % group_id]).flatten()
        sigmas = np.array(self.trace.posterior.data_vars['Group %d sigma' % group_id]).flatten()
        nus = np.array(self.trace.posterior.data_vars['Normality']).flatten()
    
        n_curves = 50
        n_samps = len(means)
        idxs = np.random.choice(np.arange(n_samps), n_curves, replace=False)
    
        try:
            xmin = bins[0]
            xmax = bins[-1]
        except TypeError:
            xmin = np.min(group_data)
            xmax = np.max(group_data)
    
        dx = xmax - xmin
        xmin -= dx * 0.05
        xmax += dx * 0.05
    
        x = np.linspace(xmin, xmax, 1000)
    
        kwargs = dict(color=fcolor, zorder=1, alpha=0.3)
        kwargs.update(prediction_kwargs)

        for i in idxs:
            v = scipy.stats.t.pdf(x, nus[i], means[i], sigmas[i])
            line, = ax.plot(x, v, **kwargs)
        
        line.set_label('Prediction')
    
        kwargs = dict(edgecolor='w',
                      facecolor=fcolor,
                      density=True,
                      bins=bins,
                      label='Observation')
        kwargs.update(hist_kwargs)
        ax.hist(group_data, **kwargs)
    
        # draw a translucent histogram in front of the curves
        if 'zorder' not in hist_kwargs and 'alpha' not in hist_kwargs:
            kwargs.update(dict(zorder=3, label=None, alpha=0.3))
            ax.hist(group_data, **kwargs)
    
        ax.text(0.95, 0.95, r'$\mathrm{N}=%d$' % len(group_data),
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top'
                )
    
        for loc in ['top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.spines['left'].set_color('gray')
        ax.set_xlabel('Observation')
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylabel('Probability')
        ax.set_yticks([])
        ax.set_ylim(0)
        if title:
            ax.set_title(title)
    
        return ax
    
    def posterior_prob(self, var_name: str, low: float = -np.inf, high: float = np.inf):
        r"""Calculate the posterior probability that a variable is in a given interval
    
        The return value approximates the following probability:
    
        .. math:: \text{Pr}(\textit{low} < \theta_{\textit{var_name}} < \textit{high} | y_1, y_2)
    
        One-sided intervals can be specified by using only the ``low`` or ``high`` argument,
        for example, to calculate the probability that the the mean of the
        first group is larger than that of the second one::
    
            best_result.posterior_prob('Difference of means', low=0)
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        var_name : str
            Name of variable.
        low : float, optional
            Lower limit of the interval.
            Default: :math:`-\infty` (no lower limit)
        high : float, optional
            Upper limit of the interval.
            Default: :math:`\infty` (no upper limit)
    
        Returns
        -------
        float
            Posterior probability that the variable is in the given interval.
    
        Notes
        -----
        If *p* is the result and *S* is the total number of samples, then the
        standard deviation of the result is :math:`\sqrt{p(1-p)/S}`
        (see BDA3, p. 267). For example, with 2000 samples, the errors for
        some returned probabilities are
    
         - 0.01 ± 0.002,
         - 0.1 ± 0.007,
         - 0.2 ± 0.009,
         - 0.5 ± 0.011,
    
        meaning the answer is accurate for most practical purposes.
        """
        
        samples_start = self.trace.posterior.data_vars[var_name]
        samples_middle = np.array(samples_start).flatten()
        n_match = len(samples_middle[(low <= samples_middle) * (samples_middle <= high)])
        n_all = len(samples_middle)
        return n_match / n_all

    def plot_results(self, bins=30):    
        # Get Posterior Value
        # Means
        posterior_mean1 = self.trace.posterior.data_vars['Group 1 mean']
        posterior_mean2 = self.trace.posterior.data_vars['Group 2 mean']
        posterior_means = np.concatenate((posterior_mean1, posterior_mean2))
        _, bin_edges_means = np.histogram(posterior_means, bins=bins)
        
        # Standard deviation
        posterior_std1 = self.trace.posterior.data_vars['Group 1 SD']
        posterior_std2 = self.trace.posterior.data_vars['Group 2 SD']
        std1_min, std1_max = tuple(np.array(arviz.hdi(self.trace, var_names=['Group 1 SD'], hdi_prob=0.995).data_vars['Group 1 SD']))
        std2_min, std2_max = tuple(np.array(arviz.hdi(self.trace, var_names=['Group 2 SD'], hdi_prob=0.995).data_vars['Group 2 SD']))
        std_min = min(std1_min, std2_min)
        std_max = max(std1_max, std2_max)
        stds = np.concatenate((posterior_std1, posterior_std2)).flatten()
        stds = stds[(std_min <= stds) * (stds <= std_max)]
        _, bin_edges_stds = np.histogram(stds, bins=bins)
        
        
        # Plotting
        fig, ((a1,a2), (a3,a4), (a5,a6), (a7,a8), (a9,a10)) = plt.subplots(5, 2, figsize=(8.2, 11), dpi=400)
        
        self.plot_posterior(
                       'Group 1 mean',
                       ax=a1,
                       bins=bin_edges_means,
                       stat='mean',
                       title='%s mean' % self.group1_name,
                       label=r'$\mu_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'Group 2 mean',
                       ax=a2,
                       bins=bin_edges_means,
                       stat='mean',
                       title='%s mean' % self.group2_name,
                       label=r'$\mu_2$')
        
        self.plot_posterior(
                       'Group 1 SD',
                       ax=a3,
                       bins=bin_edges_stds,
                       title='%s std. dev.' % self.group1_name,
                       label=r'$\mathrm{sd}_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'Group 2 SD',
                       ax=a4,
                       bins=bin_edges_stds,
                       title='%s std. dev.' % self.group2_name,
                       label=r'$\mathrm{sd}_2$')
        
        self.plot_normality_posterior(self.nu_min, a5, bins, 'Normality', fcolor='lime')
        
        self.plot_posterior(
                       'Difference of means',
                       ax=a6,
                       bins=bins,
                       title='Difference of means',
                       stat='mean',
                       ref_val=0,
                       label=r'$\mu_1 - \mu_2$',
                       fcolor='lime')
        
        self.plot_posterior(
                       'Difference of SDs',
                       ax=a7,
                       bins=bins,
                       title='Difference of std. dev.s',
                       ref_val=0,
                       label=r'$\mathrm{sd}_1 - \mathrm{sd}_2$',
                       fcolor='lime')
        
        self.plot_posterior(
                       'Effect size',
                       ax=a8,
                       bins=bins,
                       title='Effect size',
                       ref_val=0,
                       label=r'$(\mu_1 - \mu_2) / \sqrt{(\mathrm{sd}_1^2 + \mathrm{sd}_2^2)/2}$',
                       fcolor='lime')
        
        
        # Observed
        obs_vals = np.concatenate((self.y1, self.y2))
        bin_edges = np.linspace(np.min(obs_vals), np.max(obs_vals), bins + 1)
        
        self.plot_data_and_prediction(
                                 group_id=1,
                                 ax=a9,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group1_name,
                                 fcolor='salmon')
        
        self.plot_data_and_prediction(
                                 group_id=2,
                                 ax=a10,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group2_name)
        
        
        fig.tight_layout()


class BayesianHypothesisTestCauchy:

    """ Use if there is a heavy skew, where the mean
    is very different from the mode.
    Doesn't use mean so distribution better described.
    """

    def __init__(self, df: pd.DataFrame, group1_name: str, group2_name: str, value_column: str, category_column: str):
        self.group1_name = group1_name
        self.group2_name = group2_name
        self.y1 = np.array(list(df[df[category_column] == group1_name][value_column]))
        self.y2 = np.array(list(df[df[category_column] == group2_name][value_column]))
        self.nu_min = 2.5
        self.trace = None
        self.value_storage = {}


    def run_model(self, draws=2000, tune=1000):
        """
        You should use this model if your data is quite normal.
        Model Structure: paper https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf
        """

        assert self.y1.ndim == 1
        assert self.y2.ndim == 1

        # Model structure and parameters are set as those in the paper https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf
        y_all = np.concatenate((self.y1, self.y2))
        
        beta_both = halfcauchy.fit(y_all)[1]
        mode_all = scipy.stats.mode(y_all)[0]
        
        self.model = pm.Model()
        with self.model:
            # Priors for the β parameters of both groups (has to be positive, this is the spread)
            beta_group1 = pm.HalfCauchy('beta_group1', beta=beta_both)
            beta_group2 = pm.HalfCauchy('beta_group2', beta=beta_both)

            # Alpha prior (this is the location), set starting point as mode of both groups
            alpha_group1 = pm.Cauchy('alpha_group1', alpha=mode_all, beta=1)
            alpha_group2 = pm.Cauchy('alpha_group2', alpha=mode_all, beta=1)
            
            # Define likelihood for Group 1 data
            likelihood_group1 = pm.Cauchy('Group 1 data', alpha=alpha_group1, beta=beta_group1, observed=self.y1)
            
            # Define likelihood for Group 2 data
            likelihood_group2 = pm.Cauchy('Group 2 data', alpha=alpha_group2, beta=beta_group2, observed=self.y2)

            # Difference between the parameters of the two groups
            diff_beta = pm.Deterministic('diff_beta', beta_group1 - beta_group2)
            diff_alpha = pm.Deterministic('diff_alpha', alpha_group1 - alpha_group2)
            
            self.trace = pm.sample(tune=tune, draws=draws) #Runs markov-chain monte carlo

    def sample_posterior_distribution(self):
        samples = pm.sample_posterior_predictive(self.trace, self.model)
        group1_data = np.array(samples.posterior_predictive['Group 1 data']).flatten()
        group2_data = np.array(samples.posterior_predictive['Group 2 data']).flatten()

        num_samples1 = len(group1_data) if len(group1_data) < 50000 else 50000
        num_samples2 = len(group2_data) if len(group2_data) < 50000 else 50000
        self.group1_samples = np.random.choice(group1_data, num_samples1)
        self.group2_samples = np.random.choice(group2_data, num_samples2)
    

    def cliff_delta(self):
        """
        Finds cliff's delta (effect size) of posterior distribution.
        Clearly, -1 ≤ δ ≤ 1. Values near ±1 indicate the absence of
        overlap between the two samples, while values near zero indicate
        a lot of overlap between the two samples.
        """

        def cliffs_delta_calc(x, y):
            """Cliff's delta effect size"""
            pairs = 0
            ties = 0
            for a in x:
                for b in y:
                    if a > b:
                        pairs += 1
                    elif a == b:
                        ties += 1
            n = len(x) * len(y)
            return (pairs - ties) / n
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else: 
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        cliff_delta_value = cliffs_delta_calc(group1_samples.flatten(), group2_samples.flatten())
        if 'cliff_delta' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['cliff_delta'].update({'cliff_delta': cliff_delta_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['cliff_delta'] = {'cliff_delta': cliff_delta_value}
        return cliff_delta_value
    
    def non_overlap_effect_size(self):
        """
        Finds the proportion of the two distirbutions 
        that do not overlap.
        0 indicates complete overlap
        1 indicates complete non-overlap
        """

        def nos(F, G):
            """Non-Overlap Effect Size"""
            min_val = min(np.min(F), np.min(G))
            max_val = max(np.max(F), np.max(G))
            bins = np.linspace(min_val, max_val, min(len(F), len(G)))
            hist_F, _ = np.histogram(F, bins=bins, density=True)
            hist_G, _ = np.histogram(G, bins=bins, density=True)
            overlap_area = np.minimum(hist_F, hist_G)
            nos_value = 1 - np.sum(overlap_area)
            return nos_value
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass
        
        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        nos_value = nos(group1_samples.flatten(), group2_samples.flatten())
        if 'non_overlap_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'].update({'non_overlap_effect_size': nos_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'] = {'non_overlap_effect_size': nos_value}
        return nos_value
    
    def divergence_effect_size(self):
        """
        Divergence Effect Size (DES) represents the magnitude of the difference
        between the two probability distributions.
        0/<0 = No difference
        >0 = difference, the larger the value the more dissimilar
        """
        def kl_divergence(p, q):
            """Calculate KL divergence between two probability distributions."""
            epsilon = 1e-10  # Small epsilon value to avoid taking log of zero
            p_safe = np.clip(p, epsilon, 1)  # Clip probabilities to avoid zeros
            q_safe = np.clip(q, epsilon, 1) 
            return np.sum(np.where(p_safe != 0, p_safe * np.log(p_safe / q_safe), 0))

        def make_symmetrical_divergence(F, G):
            """Makes Divergence Effect Size Symmetrical"""
            kl_FG = kl_divergence(F, G)
            kl_GF = kl_divergence(G, F)
            return (kl_FG + kl_GF) / 2
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        des_value = make_symmetrical_divergence(group1_samples.flatten(), group2_samples.flatten())
        if 'divergent_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['divergent_effect_size'].update({'divergent_effect_size': des_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['divergent_effect_size'] = {'divergent_effect_size': des_value}
        return des_value
    
    def plot_posterior(self,
                   var_name: str,
                   ax = None,
                   bins = 30,
                   stat = 'mode',
                   title = None,
                   label = None,
                   ref_val = None,
                   fcolor= '#89d1ea',
                   **kwargs) -> plt.Axes:
        """
        Plot a histogram of posterior samples of a variable
    
        Parameters
        ----------
        trace :
            The trace of the analysis.
        var_name : string
            The name of the variable to be plotted. Available variable names are
            described in the :ref:`sec-variables` section.
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new axes.
        bins : int or list or NumPy array
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        stat : {'mean', 'mode'}
            Whether to print the mean or the mode of the variable on the plot.
            Default: 'mode'.
        title : string, optional
            Title of the plot. Default: don’t print a title.
        label : string, optional
            Label of the *x* axis. Default: don’t print a label.
        ref_val : float, optional
            If not None, print a vertical line at this reference value (typically
            zero).
            Default: None (don’t print a reference value)
        **kwargs : dict
            All other keyword arguments are passed to `plt.hist`.
    
        Returns
        -------
        Matplotlib Axes
            The Axes object containing the plot. Using this return value, the
            plot can be customized afterwards – for details, see the documentation
            of the `Matplotlib Axes API <https://matplotlib.org/api/axes_api.html>`_.
        """

        samples_start = self.trace.posterior.data_vars[var_name]
        samples_min, samples_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.995).data_vars[var_name]))
        samples_middle = np.array(samples_start).flatten()
        samples = samples_middle[(samples_min <= samples_middle) * (samples_middle <= samples_max)]
    
        if ax is None:
            _, ax = plt.subplots()
    
        trans = blended_transform_factory(ax.transData, ax.transAxes)
    
        hist_kwargs = {'bins': bins}
        hist_kwargs.update(kwargs)
        ax.hist(samples, rwidth=0.8,
                facecolor=fcolor, edgecolor='none', **hist_kwargs)
    
        if stat:
            if stat == 'mode':
                # calculate mode using kernel density estimate
                kernel = scipy.stats.gaussian_kde(np.array(self.trace.posterior.data_vars[var_name]).flatten())
        
                bw = kernel.covariance_factor()
                cut = 3 * bw
                x_low = np.min(samples) - cut * bw
                x_high = np.max(samples) + cut * bw
                n = 512
                x = np.linspace(x_low, x_high, n)
                vals = kernel.evaluate(x)
                max_idx = np.argmax(vals)
                mode_val = x[max_idx]
                stat_val =  mode_val
                
            elif stat == 'mean':
                stat_val = np.mean(samples)
            else:
                raise ValueError('stat parameter must be either "mean" or "mode" '
                                'or None.')
    
            ax.text(stat_val, 0.99, '%s = %.3g' % (stat, stat_val),
                    transform=trans,
                    horizontalalignment='center',
                    verticalalignment='top',
                    )

            if var_name in self.value_storage:
                # Update the subdictionary with stat: 0
                self.value_storage[var_name].update({stat: stat_val})
            else:
                # Create a new subdictionary with stat: 0
                self.value_storage[var_name] = {stat: stat_val}



        if ref_val is not None:
            ax.axvline(ref_val, linestyle=':')
    
        # plot HDI
        hdi_min, hdi_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.95).data_vars[var_name]))
        if var_name in self.value_storage:
            # Update the subdictionary
            self.value_storage[var_name].update({'hdi_min': hdi_min,
                                                 'hdi_max': hdi_max})
        else:
            # Create a new subdictionary
            self.value_storage[var_name] = {'hdi_min': hdi_min,
                                            'hdi_max': hdi_max}
        hdi_line, = ax.plot([hdi_min, hdi_max], [0, 0],
                            lw=5.0, color='k')
        hdi_line.set_clip_on(False)
        ax.text(hdi_min, 0.04, '%.3g' % hdi_min,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text(hdi_max, 0.04, '%.3g' % hdi_max,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text((hdi_min + hdi_max) / 2, 0.14, '95% HDI',
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
    
        # make it pretty
        ax.spines['bottom'].set_position(('outward', 2))
        for loc in ['left', 'top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks([])  # don't draw
        for line in ax.get_xticklines():
            line.set_marker(mpllines.TICKDOWN)
        if label:
            ax.set_xlabel(label)
        if title is not None:
            ax.set_title(title)
            
        ax.set_xlim(samples_start.min(), samples_start.max())


        return ax

    def plot_normality_posterior(self, nu_min, ax, bins, title, fcolor):
    
        var_name = 'Normality'
        norm_bins = np.logspace(np.log10(nu_min),
                                np.log10(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.95).data_vars[var_name])[-1]),
                                num=bins + 1)
        self.plot_posterior(
                       var_name,
                       ax=ax,
                       bins=norm_bins,
                       title=title,
                       label=r'$\nu$',
                       fcolor=fcolor)
        ax.set_xlim(2.4, norm_bins[-1] * 1.05)
        ax.semilogx()
        # don't use scientific notation for tick labels
        tick_fmt = plt.LogFormatter()
        ax.xaxis.set_major_formatter(tick_fmt)
        ax.xaxis.set_minor_formatter(tick_fmt)
    
    def plot_data_and_prediction(self,
                                 group_id,
                                 ax = None,
                                 bins = 30,
                                 title = None,
                                 fcolor = '#89d1ea',
                                 hist_kwargs: dict = {},
                                 prediction_kwargs: dict = {}
                                 ) -> plt.Axes:
        """Plot samples of predictive distributions and a histogram of the data.
    
        This plot can be used as a *posterior predictive check*, to examine
        how well the model predictions fit the observed data.
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        group_id: 
            The observed data of one group
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new plot.
        title : string, optional.
            Title of the plot. Default: no plot title.
        bins : int or list or NumPy array.
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        hist_kwargs : dict
            The keyword arguments to be passed to `plt.hist` for the group data.
        prediction_kwargs : dict
            The keyword arguments to be passed to `plt.plot` for the posterior
            predictive curves.
    
        Returns
        -------
        Matplotlib Axes
        """
    
        if ax is None:
            _, ax = plt.subplots()


        group_data = self.y1 if group_id==1 else self.y2
        betas = np.array(self.trace.posterior.data_vars['beta_group%d' % group_id]).flatten()
        alphas = np.array(self.trace.posterior.data_vars['alpha_group%d' % group_id]).flatten()
    
        n_curves = 50
        n_samps = len(betas)
        idxs = np.random.choice(np.arange(n_samps), n_curves, replace=False)
    
        try:
            xmin = bins[0]
            xmax = bins[-1]
        except TypeError:
            xmin = np.min(group_data)
            xmax = np.max(group_data)
    
        dx = xmax - xmin
        xmin -= dx * 0.05
        xmax += dx * 0.05
    
        x = np.linspace(xmin, xmax, 1000)
    
        kwargs = dict(color=fcolor, zorder=1, alpha=0.7)
        kwargs.update(prediction_kwargs)

        for i in idxs:
            v = scipy.stats.cauchy.pdf(x, loc=alphas[i], scale=betas[i])
            line, = ax.plot(x, v, **kwargs)
        
        line.set_label('Prediction')

        kwargs = dict(edgecolor='w',
                      facecolor=fcolor,
                      density=True,
                      bins=bins,
                      label='Observation')
        kwargs.update(hist_kwargs)
        ax.hist(group_data, **kwargs)
    
        # draw a translucent histogram in front of the curves
        if 'zorder' not in hist_kwargs and 'alpha' not in hist_kwargs:
            kwargs.update(dict(zorder=3, label=None, alpha=0.3))
            ax.hist(group_data, **kwargs)
    
        ax.text(0.95, 0.95, r'$\mathrm{N}=%d$' % len(group_data),
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top'
                )
    
        for loc in ['top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.spines['left'].set_color('gray')
        ax.set_xlabel('Observation')
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylabel('Probability')
        ax.set_yticks([])
        ax.set_ylim(0)
        if title:
            ax.set_title(title)
    
        return ax
    
    def posterior_prob(self, var_name: str, low: float = -np.inf, high: float = np.inf):
        r"""Calculate the posterior probability that a variable is in a given interval
    
        The return value approximates the following probability:
    
        .. math:: \text{Pr}(\textit{low} < \theta_{\textit{var_name}} < \textit{high} | y_1, y_2)
    
        One-sided intervals can be specified by using only the ``low`` or ``high`` argument,
        for example, to calculate the probability that the the mean of the
        first group is larger than that of the second one::
    
            best_result.posterior_prob('Difference of means', low=0)
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        var_name : str
            Name of variable.
        low : float, optional
            Lower limit of the interval.
            Default: :math:`-\infty` (no lower limit)
        high : float, optional
            Upper limit of the interval.
            Default: :math:`\infty` (no upper limit)
    
        Returns
        -------
        float
            Posterior probability that the variable is in the given interval.
    
        Notes
        -----
        If *p* is the result and *S* is the total number of samples, then the
        standard deviation of the result is :math:`\sqrt{p(1-p)/S}`
        (see BDA3, p. 267). For example, with 2000 samples, the errors for
        some returned probabilities are
    
         - 0.01 ± 0.002,
         - 0.1 ± 0.007,
         - 0.2 ± 0.009,
         - 0.5 ± 0.011,
    
        meaning the answer is accurate for most practical purposes.
        """
        
        samples_start = self.trace.posterior.data_vars[var_name]
        samples_middle = np.array(samples_start).flatten()
        n_match = len(samples_middle[(low <= samples_middle) * (samples_middle <= high)])
        n_all = len(samples_middle)
        return n_match / n_all

    def plot_results(self, bins=30):    
        # Get Posterior Value
        # Betas
        posterior_beta1 = self.trace.posterior.data_vars['beta_group1']
        posterior_beta2 = self.trace.posterior.data_vars['beta_group2']
        posterior_betas = np.concatenate((posterior_beta1, posterior_beta2))
        _, bin_edges_betas = np.histogram(posterior_betas, bins=bins)

        # Alphas
        posterior_alpha1 = self.trace.posterior.data_vars['alpha_group1']
        posterior_alpha2 = self.trace.posterior.data_vars['alpha_group2']
        posterior_alphas = np.concatenate((posterior_alpha1, posterior_alpha2))
        _, bin_edges_alphas = np.histogram(posterior_alphas, bins=bins)
        
        
        # Plotting
        fig, ((a1,a2), (a3,a4), (a5,a6), (a7,a8)) = plt.subplots(4, 2, figsize=(8.2, 11), dpi=400)
        
        self.plot_posterior(
                       'beta_group1',
                       ax=a1,
                       bins=bin_edges_betas,
                       stat='mean',
                       title='%s beta' % self.group1_name,
                       label=r'$\beta_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'beta_group2',
                       ax=a2,
                       bins=bin_edges_betas,
                       stat='mean',
                       title='%s beta' % self.group2_name,
                       label=r'$\beta_2$')
        
        self.plot_posterior(
                       'alpha_group1',
                       ax=a3,
                       bins=bin_edges_alphas,
                       stat='mean',
                       title='%s alpha' % self.group1_name,
                       label=r'$\alpha_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'alpha_group2',
                       ax=a4,
                       bins=bin_edges_alphas,
                       stat='mean',
                       title='%s alpha' % self.group2_name,
                       label=r'$\alpha_2$')
        
        self.plot_posterior(
                       'diff_beta',
                       ax=a5,
                       bins=bins,
                       title='Difference of betas',
                       stat='mean',
                       ref_val=0,
                       label=r'$\beta_1 - \beta_2$',
                       fcolor='lime')
        
        self.plot_posterior(
                       'diff_alpha',
                       ax=a6,
                       bins=bins,
                       title='Difference of alphas',
                       stat='mean',
                       ref_val=0,
                       label=r'$\alpha_1 - \alpha_2$',
                       fcolor='lime')
        
        
        # Observed
        obs_vals = np.concatenate((self.y1, self.y2))
        bin_edges = np.linspace(np.min(obs_vals), np.max(obs_vals), bins + 1)
        
        self.plot_data_and_prediction(
                                 group_id=1,
                                 ax=a7,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group1_name,
                                 fcolor='salmon')
        
        self.plot_data_and_prediction(
                                 group_id=2,
                                 ax=a8,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group2_name)
        
        
        fig.tight_layout()

class BayesianHypothesisTestHalfCauchy:

    """
    This should only be used if there is positive skew where many values cut off at 0.
    Check distribution of the groups. If there is negative skew
    you will need to reverse by max(data) subtract data, so that it becomes positive skew.
    If you have positive skew but there is a defined cut-off, then transform the data towards 0.
    
    """

    def __init__(self, df: pd.DataFrame, group1_name: str, group2_name: str, value_column: str, category_column: str):
        self.group1_name = group1_name
        self.group2_name = group2_name
        self.y1 = np.array(list(df[df[category_column] == group1_name][value_column]))
        self.y2 = np.array(list(df[df[category_column] == group2_name][value_column]))
        self.nu_min = 2.5
        self.trace = None
        self.value_storage = {}


    def run_model(self, draws=2000, tune=1000):
        """
        You should use this model if your data is quite normal.
        Model Structure: paper https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf
        """

        assert self.y1.ndim == 1
        assert self.y2.ndim == 1

        # Model structure and parameters are set as those in the paper https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf
        y_all = np.concatenate((self.y1, self.y2))
        
        beta_both = halfcauchy.fit(y_all)[1]
        mode_all = scipy.stats.mode(y_all)[0]
        
        self.model = pm.Model()
        with self.model:
            # Priors for the β parameters of both groups
            beta_group1 = pm.HalfCauchy('beta_group1', beta=beta_both)
            beta_group2 = pm.HalfCauchy('beta_group2', beta=beta_both)
            
            # Define likelihood for Group 1 data
            likelihood_group1 = pm.HalfCauchy('Group 1 data', beta=beta_group1, observed=self.y1)
            
            # Define likelihood for Group 2 data
            likelihood_group2 = pm.HalfCauchy('Group 2 data', beta=beta_group2, observed=self.y2)

            # Difference between the β parameters of the two groups
            diff_beta = pm.Deterministic('diff_beta', beta_group1 - beta_group2)
            
            self.trace = pm.sample(tune=tune, draws=draws) #Runs markov-chain monte carlo
    
    def sample_posterior_distribution(self):
        samples = pm.sample_posterior_predictive(self.trace, self.model)
        group1_data = np.array(samples.posterior_predictive['Group 1 data']).flatten()
        group2_data = np.array(samples.posterior_predictive['Group 2 data']).flatten()

        num_samples1 = len(group1_data) if len(group1_data) < 50000 else 50000
        num_samples2 = len(group2_data) if len(group2_data) < 50000 else 50000
        self.group1_samples = np.random.choice(group1_data, num_samples1)
        self.group2_samples = np.random.choice(group2_data, num_samples2)
    

    def cliff_delta(self):
        """
        Finds cliff's delta (effect size) of posterior distribution.
        Clearly, -1 ≤ δ ≤ 1. Values near ±1 indicate the absence of
        overlap between the two samples, while values near zero indicate
        a lot of overlap between the two samples.
        """

        def cliffs_delta_calc(x, y):
            """Cliff's delta effect size"""
            pairs = 0
            ties = 0
            for a in x:
                for b in y:
                    if a > b:
                        pairs += 1
                    elif a == b:
                        ties += 1
            n = len(x) * len(y)
            return (pairs - ties) / n
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        cliff_delta_value = cliffs_delta_calc(group1_samples.flatten(), group2_samples.flatten())
        if 'cliff_delta' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['cliff_delta'].update({'cliff_delta': cliff_delta_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['cliff_delta'] = {'cliff_delta': cliff_delta_value}
        return cliff_delta_value
    
    def non_overlap_effect_size(self):
        """
        Finds the proportion of the two distirbutions 
        that do not overlap.
        0 indicates complete overlap
        1 indicates complete non-overlap
        """

        def nos(F, G):
            """Non-Overlap Effect Size"""
            min_val = min(np.min(F), np.min(G))
            max_val = max(np.max(F), np.max(G))
            bins = np.linspace(min_val, max_val, min(len(F), len(G)))
            hist_F, _ = np.histogram(F, bins=bins, density=True)
            hist_G, _ = np.histogram(G, bins=bins, density=True)
            overlap_area = np.minimum(hist_F, hist_G)
            nos_value = 1 - np.sum(overlap_area)
            return nos_value
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass
        
        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        nos_value = nos(group1_samples.flatten(), group2_samples.flatten())
        if 'non_overlap_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'].update({'non_overlap_effect_size': nos_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'] = {'non_overlap_effect_size': nos_value}
        return nos_value
    
    def divergence_effect_size(self):
        """
        Divergence Effect Size (DES) represents the magnitude of the difference
        between the two probability distributions.
        0/<0 = No difference
        >0 = difference, the larger the value the more dissimilar
        """
        def kl_divergence(p, q):
            """Calculate KL divergence between two probability distributions."""
            epsilon = 1e-10  # Small epsilon value to avoid taking log of zero
            p_safe = np.clip(p, epsilon, 1)  # Clip probabilities to avoid zeros
            q_safe = np.clip(q, epsilon, 1) 
            return np.sum(np.where(p_safe != 0, p_safe * np.log(p_safe / q_safe), 0))

        def make_symmetrical_divergence(F, G):
            """Makes Divergence Effect Size Symmetrical"""
            kl_FG = kl_divergence(F, G)
            kl_GF = kl_divergence(G, F)
            return (kl_FG + kl_GF) / 2
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        des_value = make_symmetrical_divergence(group1_samples.flatten(), group2_samples.flatten())
        if 'divergent_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['divergent_effect_size'].update({'divergent_effect_size': des_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['divergent_effect_size'] = {'divergent_effect_size': des_value}
        return des_value
    
    def plot_posterior(self,
                   var_name: str,
                   ax = None,
                   bins = 30,
                   stat = 'mode',
                   title = None,
                   label = None,
                   ref_val = None,
                   fcolor= '#89d1ea',
                   **kwargs) -> plt.Axes:
        """
        Plot a histogram of posterior samples of a variable
    
        Parameters
        ----------
        trace :
            The trace of the analysis.
        var_name : string
            The name of the variable to be plotted. Available variable names are
            described in the :ref:`sec-variables` section.
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new axes.
        bins : int or list or NumPy array
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        stat : {'mean', 'mode'}
            Whether to print the mean or the mode of the variable on the plot.
            Default: 'mode'.
        title : string, optional
            Title of the plot. Default: don’t print a title.
        label : string, optional
            Label of the *x* axis. Default: don’t print a label.
        ref_val : float, optional
            If not None, print a vertical line at this reference value (typically
            zero).
            Default: None (don’t print a reference value)
        **kwargs : dict
            All other keyword arguments are passed to `plt.hist`.
    
        Returns
        -------
        Matplotlib Axes
            The Axes object containing the plot. Using this return value, the
            plot can be customized afterwards – for details, see the documentation
            of the `Matplotlib Axes API <https://matplotlib.org/api/axes_api.html>`_.
        """

        samples_start = self.trace.posterior.data_vars[var_name]
        samples_min, samples_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.995).data_vars[var_name]))
        samples_middle = np.array(samples_start).flatten()
        samples = samples_middle[(samples_min <= samples_middle) * (samples_middle <= samples_max)]
    
        if ax is None:
            _, ax = plt.subplots()
    
        trans = blended_transform_factory(ax.transData, ax.transAxes)
    
        hist_kwargs = {'bins': bins}
        hist_kwargs.update(kwargs)
        ax.hist(samples, rwidth=0.8,
                facecolor=fcolor, edgecolor='none', **hist_kwargs)
    
        if stat:
            if stat == 'mode':
                # calculate mode using kernel density estimate
                kernel = scipy.stats.gaussian_kde(np.array(self.trace.posterior.data_vars[var_name]).flatten())
        
                bw = kernel.covariance_factor()
                cut = 3 * bw
                x_low = np.min(samples) - cut * bw
                x_high = np.max(samples) + cut * bw
                n = 512
                x = np.linspace(x_low, x_high, n)
                vals = kernel.evaluate(x)
                max_idx = np.argmax(vals)
                mode_val = x[max_idx]
                stat_val =  mode_val
                
            elif stat == 'mean':
                stat_val = np.mean(samples)
            else:
                raise ValueError('stat parameter must be either "mean" or "mode" '
                                'or None.')
    
            ax.text(stat_val, 0.99, '%s = %.3g' % (stat, stat_val),
                    transform=trans,
                    horizontalalignment='center',
                    verticalalignment='top',
                    )

            if var_name in self.value_storage:
                # Update the subdictionary with stat: 0
                self.value_storage[var_name].update({stat: stat_val})
            else:
                # Create a new subdictionary with stat: 0
                self.value_storage[var_name] = {stat: stat_val}



        if ref_val is not None:
            ax.axvline(ref_val, linestyle=':')
    
        # plot HDI
        hdi_min, hdi_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.95).data_vars[var_name]))
        if var_name in self.value_storage:
            # Update the subdictionary
            self.value_storage[var_name].update({'hdi_min': hdi_min,
                                                 'hdi_max': hdi_max})
        else:
            # Create a new subdictionary
            self.value_storage[var_name] = {'hdi_min': hdi_min,
                                            'hdi_max': hdi_max}
        hdi_line, = ax.plot([hdi_min, hdi_max], [0, 0],
                            lw=5.0, color='k')
        hdi_line.set_clip_on(False)
        ax.text(hdi_min, 0.04, '%.3g' % hdi_min,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text(hdi_max, 0.04, '%.3g' % hdi_max,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text((hdi_min + hdi_max) / 2, 0.14, '95% HDI',
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
    
        # make it pretty
        ax.spines['bottom'].set_position(('outward', 2))
        for loc in ['left', 'top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks([])  # don't draw
        for line in ax.get_xticklines():
            line.set_marker(mpllines.TICKDOWN)
        if label:
            ax.set_xlabel(label)
        if title is not None:
            ax.set_title(title)
            
        ax.set_xlim(samples_start.min(), samples_start.max())


        return ax

    def plot_normality_posterior(self, nu_min, ax, bins, title, fcolor):
    
        var_name = 'Normality'
        norm_bins = np.logspace(np.log10(nu_min),
                                np.log10(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.95).data_vars[var_name])[-1]),
                                num=bins + 1)
        self.plot_posterior(
                       var_name,
                       ax=ax,
                       bins=norm_bins,
                       title=title,
                       label=r'$\nu$',
                       fcolor=fcolor)
        ax.set_xlim(2.4, norm_bins[-1] * 1.05)
        ax.semilogx()
        # don't use scientific notation for tick labels
        tick_fmt = plt.LogFormatter()
        ax.xaxis.set_major_formatter(tick_fmt)
        ax.xaxis.set_minor_formatter(tick_fmt)
    
    def plot_data_and_prediction(self,
                                 group_id,
                                 ax = None,
                                 bins = 30,
                                 title = None,
                                 fcolor = '#89d1ea',
                                 hist_kwargs: dict = {},
                                 prediction_kwargs: dict = {}
                                 ) -> plt.Axes:
        """Plot samples of predictive distributions and a histogram of the data.
    
        This plot can be used as a *posterior predictive check*, to examine
        how well the model predictions fit the observed data.
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        group_id: 
            The observed data of one group
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new plot.
        title : string, optional.
            Title of the plot. Default: no plot title.
        bins : int or list or NumPy array.
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        hist_kwargs : dict
            The keyword arguments to be passed to `plt.hist` for the group data.
        prediction_kwargs : dict
            The keyword arguments to be passed to `plt.plot` for the posterior
            predictive curves.
    
        Returns
        -------
        Matplotlib Axes
        """
    
        if ax is None:
            _, ax = plt.subplots()


        group_data = self.y1 if group_id==1 else self.y2
        betas = np.array(self.trace.posterior.data_vars['beta_group%d' % group_id]).flatten()
    
        n_curves = 50
        n_samps = len(betas)
        idxs = np.random.choice(np.arange(n_samps), n_curves, replace=False)
    
        try:
            xmin = bins[0]
            xmax = bins[-1]
        except TypeError:
            xmin = np.min(group_data)
            xmax = np.max(group_data)
    
        dx = xmax - xmin
        xmin -= dx * 0.05
        xmax += dx * 0.05
    
        x = np.linspace(xmin, xmax, 1000)
    
        kwargs = dict(color=fcolor, zorder=1, alpha=0.7)
        kwargs.update(prediction_kwargs)

        for i in idxs:
            v = scipy.stats.halfcauchy.pdf(x, scale=betas[i])
            line, = ax.plot(x, v, **kwargs)
        
        line.set_label('Prediction')

        kwargs = dict(edgecolor='w',
                      facecolor=fcolor,
                      density=True,
                      bins=bins,
                      label='Observation')
        kwargs.update(hist_kwargs)
        ax.hist(group_data, **kwargs)
    
        # draw a translucent histogram in front of the curves
        if 'zorder' not in hist_kwargs and 'alpha' not in hist_kwargs:
            kwargs.update(dict(zorder=3, label=None, alpha=0.3))
            ax.hist(group_data, **kwargs)
    
        ax.text(0.95, 0.95, r'$\mathrm{N}=%d$' % len(group_data),
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top'
                )
    
        for loc in ['top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.spines['left'].set_color('gray')
        ax.set_xlabel('Observation')
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylabel('Probability')
        ax.set_yticks([])
        ax.set_ylim(0)
        if title:
            ax.set_title(title)
    
        return ax
    
    def posterior_prob(self, var_name: str, low: float = -np.inf, high: float = np.inf):
        r"""Calculate the posterior probability that a variable is in a given interval
    
        The return value approximates the following probability:
    
        .. math:: \text{Pr}(\textit{low} < \theta_{\textit{var_name}} < \textit{high} | y_1, y_2)
    
        One-sided intervals can be specified by using only the ``low`` or ``high`` argument,
        for example, to calculate the probability that the the mean of the
        first group is larger than that of the second one::
    
            best_result.posterior_prob('Difference of means', low=0)
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        var_name : str
            Name of variable.
        low : float, optional
            Lower limit of the interval.
            Default: :math:`-\infty` (no lower limit)
        high : float, optional
            Upper limit of the interval.
            Default: :math:`\infty` (no upper limit)
    
        Returns
        -------
        float
            Posterior probability that the variable is in the given interval.
    
        Notes
        -----
        If *p* is the result and *S* is the total number of samples, then the
        standard deviation of the result is :math:`\sqrt{p(1-p)/S}`
        (see BDA3, p. 267). For example, with 2000 samples, the errors for
        some returned probabilities are
    
         - 0.01 ± 0.002,
         - 0.1 ± 0.007,
         - 0.2 ± 0.009,
         - 0.5 ± 0.011,
    
        meaning the answer is accurate for most practical purposes.
        """
        
        samples_start = self.trace.posterior.data_vars[var_name]
        samples_middle = np.array(samples_start).flatten()
        n_match = len(samples_middle[(low <= samples_middle) * (samples_middle <= high)])
        n_all = len(samples_middle)
        return n_match / n_all

    def plot_results(self, bins=30):    
        # Get Posterior Value
        # Betas
        posterior_beta1 = self.trace.posterior.data_vars['beta_group1']
        posterior_beta2 = self.trace.posterior.data_vars['beta_group2']
        posterior_betas = np.concatenate((posterior_beta1, posterior_beta2))
        _, bin_edges_betas = np.histogram(posterior_betas, bins=bins)
        
        
        # Plotting
        fig, ((a1,a2), (a3,a4), (a5,a6)) = plt.subplots(3, 2, figsize=(8.2, 11), dpi=400)
        
        self.plot_posterior(
                       'beta_group1',
                       ax=a1,
                       bins=bin_edges_betas,
                       stat='mean',
                       title='%s beta' % self.group1_name,
                       label=r'$\beta_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'beta_group2',
                       ax=a2,
                       bins=bin_edges_betas,
                       stat='mean',
                       title='%s beta' % self.group2_name,
                       label=r'$\beta_2$')
        
        self.plot_posterior(
                       'diff_beta',
                       ax=a3,
                       bins=bins,
                       title='Difference of betas',
                       stat='mean',
                       ref_val=0,
                       label=r'$\beta_1 - \beta_2$',
                       fcolor='lime')
        
        a4.axis('off')
        
        
        # Observed
        obs_vals = np.concatenate((self.y1, self.y2))
        bin_edges = np.linspace(np.min(obs_vals), np.max(obs_vals), bins + 1)
        
        self.plot_data_and_prediction(
                                 group_id=1,
                                 ax=a5,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group1_name,
                                 fcolor='salmon')
        
        self.plot_data_and_prediction(
                                 group_id=2,
                                 ax=a6,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group2_name)
        
        
        fig.tight_layout()

class BayesianHypothesisTestSkewNormal:

    def __init__(self, df: pd.DataFrame, group1_name: str, group2_name: str, value_column: str, category_column: str):
        self.group1_name = group1_name
        self.group2_name = group2_name
        self.y1 = np.array(list(df[df[category_column] == group1_name][value_column]))
        self.y2 = np.array(list(df[df[category_column] == group2_name][value_column]))
        self.nu_min = 2.5
        self.trace = None
        self.value_storage = {}


    def run_model(self, draws=2000, tune=1000):
        """
        You should use this model if your data is skewed normal
        """

        assert self.y1.ndim == 1
        assert self.y2.ndim == 1

        y_all = np.concatenate((self.y1, self.y2))
        
        # assumption centres of parameter distributions are the values of the pooled groups
        mu_loc = np.mean(y_all)
        mu_scale = np.std(y_all) * 1000
        
        sigma_both = np.std(y_all)
        
        alpha_both = scipyskew(y_all)
        
        self.model = pm.Model()
        with self.model:
            # Prior assumption is the distribution of mean and standard deviation of the two groups are the same,
            # values are assumed as the mean and std of both groups. Truncated to min and max
            group1_mean = pm.Normal('Group 1 mean', mu=mu_loc, sigma=mu_scale)
            group2_mean = pm.Normal('Group 2 mean', mu=mu_loc, sigma=mu_scale)

            # Alpha prior (this is skew), set starting point as skew of both groups
            alpha_group1 = pm.Normal('alpha_group1', mu=alpha_both, sigma=2)
            alpha_group2 = pm.Normal('alpha_group2', mu=alpha_both, sigma=2)

            # Prior assumption of the standard deviation
            # Can't be negative
            group1_sd = pm.TruncatedNormal('Group 1 SD', mu=sigma_both, sigma=1, lower=0)
            group2_sd = pm.TruncatedNormal('Group 2 SD', mu=sigma_both, sigma=1, lower=0)
        
            _ = pm.SkewNormal('Group 1 data', observed=self.y1, mu=group1_mean, sigma=group1_sd, alpha=alpha_group1)
            _ = pm.SkewNormal('Group 2 data', observed=self.y2, mu=group2_mean, sigma=group2_sd, alpha=alpha_group2)
        
            diff_of_means = pm.Deterministic('Difference of means', group1_mean - group2_mean)
            diff_of_sd = pm.Deterministic('Difference of SDs', group1_sd - group2_sd)
            diff_of_skew = pm.Deterministic('Difference of Skew', alpha_group1 - alpha_group2)
            sd_ratio = pm.Deterministic('SD Ratio', group1_sd / group2_sd)
            skew_ratio = pm.Deterministic('Skew Ratio', alpha_group1 / alpha_group2)
            mean_ratio = pm.Deterministic('Mean Ratio', group1_mean / group2_mean)

            mes = pm.Deterministic('Effect size', (mean_ratio + sd_ratio + skew_ratio)/3)

            self.trace = pm.sample(tune=tune, draws=draws) #Runs markov-chain monte carlo
    
    def sample_posterior_distribution(self):
        samples = pm.sample_posterior_predictive(self.trace, self.model)
        group1_data = np.array(samples.posterior_predictive['Group 1 data']).flatten()
        group2_data = np.array(samples.posterior_predictive['Group 2 data']).flatten()

        num_samples1 = len(group1_data) if len(group1_data) < 50000 else 50000
        num_samples2 = len(group2_data) if len(group2_data) < 50000 else 50000
        self.group1_samples = np.random.choice(group1_data, num_samples1)
        self.group2_samples = np.random.choice(group2_data, num_samples2)
    

    def cliff_delta(self):
        """
        Finds cliff's delta (effect size) of posterior distribution.
        Clearly, -1 ≤ δ ≤ 1. Values near ±1 indicate the absence of
        overlap between the two samples, while values near zero indicate
        a lot of overlap between the two samples.
        """

        def cliffs_delta_calc(x, y):
            """Cliff's delta effect size"""
            pairs = 0
            ties = 0
            for a in x:
                for b in y:
                    if a > b:
                        pairs += 1
                    elif a == b:
                        ties += 1
            n = len(x) * len(y)
            return (pairs - ties) / n
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        cliff_delta_value = cliffs_delta_calc(group1_samples.flatten(), group2_samples.flatten())
        if 'cliff_delta' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['cliff_delta'].update({'cliff_delta': cliff_delta_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['cliff_delta'] = {'cliff_delta': cliff_delta_value}
        return cliff_delta_value
    
    def non_overlap_effect_size(self):
        """
        Finds the proportion of the two distirbutions 
        that do not overlap.
        0 indicates complete overlap
        1 indicates complete non-overlap
        """

        def nos(F, G):
            """Non-Overlap Effect Size"""
            min_val = min(np.min(F), np.min(G))
            max_val = max(np.max(F), np.max(G))
            bins = np.linspace(min_val, max_val, min(len(F), len(G)))
            hist_F, _ = np.histogram(F, bins=bins, density=True)
            hist_G, _ = np.histogram(G, bins=bins, density=True)
            overlap_area = np.minimum(hist_F, hist_G)
            nos_value = 1 - np.sum(overlap_area)
            return nos_value
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass
        
        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        nos_value = nos(group1_samples.flatten(), group2_samples.flatten())
        if 'non_overlap_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'].update({'non_overlap_effect_size': nos_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'] = {'non_overlap_effect_size': nos_value}
        return nos_value
    
    def divergence_effect_size(self):
        """
        Divergence Effect Size (DES) represents the magnitude of the difference
        between the two probability distributions.
        0/<0 = No difference
        >0 = difference, the larger the value the more dissimilar
        """
        def kl_divergence(p, q):
            """Calculate KL divergence between two probability distributions."""
            epsilon = 1e-10  # Small epsilon value to avoid taking log of zero
            p_safe = np.clip(p, epsilon, 1)  # Clip probabilities to avoid zeros
            q_safe = np.clip(q, epsilon, 1) 
            return np.sum(np.where(p_safe != 0, p_safe * np.log(p_safe / q_safe), 0))

        def make_symmetrical_divergence(F, G):
            """Makes Divergence Effect Size Symmetrical"""
            kl_FG = kl_divergence(F, G)
            kl_GF = kl_divergence(G, F)
            return (kl_FG + kl_GF) / 2
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        des_value = make_symmetrical_divergence(group1_samples.flatten(), group2_samples.flatten())
        if 'divergent_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['divergent_effect_size'].update({'divergent_effect_size': des_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['divergent_effect_size'] = {'divergent_effect_size': des_value}
        return des_value
        
    def plot_posterior(self,
                   var_name: str,
                   ax = None,
                   bins = 30,
                   stat = 'mode',
                   title = None,
                   label = None,
                   ref_val = None,
                   fcolor= '#89d1ea',
                   **kwargs) -> plt.Axes:
        """
        Plot a histogram of posterior samples of a variable
    
        Parameters
        ----------
        trace :
            The trace of the analysis.
        var_name : string
            The name of the variable to be plotted. Available variable names are
            described in the :ref:`sec-variables` section.
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new axes.
        bins : int or list or NumPy array
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        stat : {'mean', 'mode'}
            Whether to print the mean or the mode of the variable on the plot.
            Default: 'mode'.
        title : string, optional
            Title of the plot. Default: don’t print a title.
        label : string, optional
            Label of the *x* axis. Default: don’t print a label.
        ref_val : float, optional
            If not None, print a vertical line at this reference value (typically
            zero).
            Default: None (don’t print a reference value)
        **kwargs : dict
            All other keyword arguments are passed to `plt.hist`.
    
        Returns
        -------
        Matplotlib Axes
            The Axes object containing the plot. Using this return value, the
            plot can be customized afterwards – for details, see the documentation
            of the `Matplotlib Axes API <https://matplotlib.org/api/axes_api.html>`_.
        """

        samples_start = self.trace.posterior.data_vars[var_name]
        samples_min, samples_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.995).data_vars[var_name]))
        samples_middle = np.array(samples_start).flatten()
        samples = samples_middle[(samples_min <= samples_middle) * (samples_middle <= samples_max)]
    
        if ax is None:
            _, ax = plt.subplots()
    
        trans = blended_transform_factory(ax.transData, ax.transAxes)
    
        hist_kwargs = {'bins': bins}
        hist_kwargs.update(kwargs)
        ax.hist(samples, rwidth=0.8,
                facecolor=fcolor, edgecolor='none', **hist_kwargs)
    
        if stat:
            if stat == 'mode':
                # calculate mode using kernel density estimate
                kernel = scipy.stats.gaussian_kde(np.array(self.trace.posterior.data_vars[var_name]).flatten())
        
                bw = kernel.covariance_factor()
                cut = 3 * bw
                x_low = np.min(samples) - cut * bw
                x_high = np.max(samples) + cut * bw
                n = 512
                x = np.linspace(x_low, x_high, n)
                vals = kernel.evaluate(x)
                max_idx = np.argmax(vals)
                mode_val = x[max_idx]
                stat_val =  mode_val
                
            elif stat == 'mean':
                stat_val = np.mean(samples)
            else:
                raise ValueError('stat parameter must be either "mean" or "mode" '
                                'or None.')
    
            ax.text(stat_val, 0.99, '%s = %.3g' % (stat, stat_val),
                    transform=trans,
                    horizontalalignment='center',
                    verticalalignment='top',
                    )

            if var_name in self.value_storage:
                # Update the subdictionary with stat: 0
                self.value_storage[var_name].update({stat: stat_val})
            else:
                # Create a new subdictionary with stat: 0
                self.value_storage[var_name] = {stat: stat_val}



        if ref_val is not None:
            ax.axvline(ref_val, linestyle=':')
    
        # plot HDI
        hdi_min, hdi_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.95).data_vars[var_name]))
        if var_name in self.value_storage:
            # Update the subdictionary
            self.value_storage[var_name].update({'hdi_min': hdi_min,
                                                 'hdi_max': hdi_max})
        else:
            # Create a new subdictionary
            self.value_storage[var_name] = {'hdi_min': hdi_min,
                                            'hdi_max': hdi_max}
        hdi_line, = ax.plot([hdi_min, hdi_max], [0, 0],
                            lw=5.0, color='k')
        hdi_line.set_clip_on(False)
        ax.text(hdi_min, 0.04, '%.3g' % hdi_min,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text(hdi_max, 0.04, '%.3g' % hdi_max,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text((hdi_min + hdi_max) / 2, 0.14, '95% HDI',
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
    
        # make it pretty
        ax.spines['bottom'].set_position(('outward', 2))
        for loc in ['left', 'top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks([])  # don't draw
        for line in ax.get_xticklines():
            line.set_marker(mpllines.TICKDOWN)
        if label:
            ax.set_xlabel(label)
        if title is not None:
            ax.set_title(title)
            
        ax.set_xlim(samples_start.min(), samples_start.max())


        return ax
    
    def plot_data_and_prediction(self,
                                 group_id,
                                 ax = None,
                                 bins = 30,
                                 title = None,
                                 fcolor = '#89d1ea',
                                 hist_kwargs: dict = {},
                                 prediction_kwargs: dict = {}
                                 ) -> plt.Axes:
        """Plot samples of predictive distributions and a histogram of the data.
    
        This plot can be used as a *posterior predictive check*, to examine
        how well the model predictions fit the observed data.
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        group_id: 
            The observed data of one group
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new plot.
        title : string, optional.
            Title of the plot. Default: no plot title.
        bins : int or list or NumPy array.
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        hist_kwargs : dict
            The keyword arguments to be passed to `plt.hist` for the group data.
        prediction_kwargs : dict
            The keyword arguments to be passed to `plt.plot` for the posterior
            predictive curves.
    
        Returns
        -------
        Matplotlib Axes
        """
    
        if ax is None:
            _, ax = plt.subplots()

        group_data = self.y1 if group_id==1 else self.y2
        means = np.array(self.trace.posterior.data_vars['Group %d mean' % group_id]).flatten()
        sigmas = np.array(self.trace.posterior.data_vars['Group %d SD' % group_id]).flatten()
        alphas = np.array(self.trace.posterior.data_vars['alpha_group%d' % group_id]).flatten()
    
        n_curves = 50
        n_samps = len(means)
        idxs = np.random.choice(np.arange(n_samps), n_curves, replace=False)
    
        try:
            xmin = bins[0]
            xmax = bins[-1]
        except TypeError:
            xmin = np.min(group_data)
            xmax = np.max(group_data)
    
        dx = xmax - xmin
        xmin -= dx * 0.05
        xmax += dx * 0.05
    
        x = np.linspace(xmin, xmax, 1000)
    
        kwargs = dict(color=fcolor, zorder=1, alpha=0.3)
        kwargs.update(prediction_kwargs)

        for i in idxs:
            v = scipy.stats.skewnorm.pdf(x, alphas[i], means[i], sigmas[i])
            line, = ax.plot(x, v, **kwargs)
        
        line.set_label('Prediction')
    
        kwargs = dict(edgecolor='w',
                      facecolor=fcolor,
                      density=True,
                      bins=bins,
                      label='Observation')
        kwargs.update(hist_kwargs)
        ax.hist(group_data, **kwargs)
    
        # draw a translucent histogram in front of the curves
        if 'zorder' not in hist_kwargs and 'alpha' not in hist_kwargs:
            kwargs.update(dict(zorder=3, label=None, alpha=0.3))
            ax.hist(group_data, **kwargs)
    
        ax.text(0.95, 0.95, r'$\mathrm{N}=%d$' % len(group_data),
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top'
                )
    
        for loc in ['top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.spines['left'].set_color('gray')
        ax.set_xlabel('Observation')
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylabel('Probability')
        #ax.set_yticks([])
        
        ax.set_ylim(0)
        if title:
            ax.set_title(title)
    
        return ax
    
    def posterior_prob(self, var_name: str, low: float = -np.inf, high: float = np.inf):
        r"""Calculate the posterior probability that a variable is in a given interval
    
        The return value approximates the following probability:
    
        .. math:: \text{Pr}(\textit{low} < \theta_{\textit{var_name}} < \textit{high} | y_1, y_2)
    
        One-sided intervals can be specified by using only the ``low`` or ``high`` argument,
        for example, to calculate the probability that the the mean of the
        first group is larger than that of the second one::
    
            best_result.posterior_prob('Difference of means', low=0)
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        var_name : str
            Name of variable.
        low : float, optional
            Lower limit of the interval.
            Default: :math:`-\infty` (no lower limit)
        high : float, optional
            Upper limit of the interval.
            Default: :math:`\infty` (no upper limit)
    
        Returns
        -------
        float
            Posterior probability that the variable is in the given interval.
    
        Notes
        -----
        If *p* is the result and *S* is the total number of samples, then the
        standard deviation of the result is :math:`\sqrt{p(1-p)/S}`
        (see BDA3, p. 267). For example, with 2000 samples, the errors for
        some returned probabilities are
    
         - 0.01 ± 0.002,
         - 0.1 ± 0.007,
         - 0.2 ± 0.009,
         - 0.5 ± 0.011,
    
        meaning the answer is accurate for most practical purposes.
        """
        
        samples_start = self.trace.posterior.data_vars[var_name]
        samples_middle = np.array(samples_start).flatten()
        n_match = len(samples_middle[(low <= samples_middle) * (samples_middle <= high)])
        n_all = len(samples_middle)
        return n_match / n_all

    def plot_results(self, bins=30):    
        # Get Posterior Value
        # Means
        posterior_mean1 = self.trace.posterior.data_vars['Group 1 mean']
        posterior_mean2 = self.trace.posterior.data_vars['Group 2 mean']
        posterior_means = np.concatenate((posterior_mean1, posterior_mean2))
        _, bin_edges_means = np.histogram(posterior_means, bins=bins)
        
        # Standard deviation
        posterior_std1 = self.trace.posterior.data_vars['Group 1 SD']
        posterior_std2 = self.trace.posterior.data_vars['Group 2 SD']
        std1_min, std1_max = tuple(np.array(arviz.hdi(self.trace, var_names=['Group 1 SD'], hdi_prob=0.995).data_vars['Group 1 SD']))
        std2_min, std2_max = tuple(np.array(arviz.hdi(self.trace, var_names=['Group 2 SD'], hdi_prob=0.995).data_vars['Group 2 SD']))
        std_min = min(std1_min, std2_min)
        std_max = max(std1_max, std2_max)
        stds = np.concatenate((posterior_std1, posterior_std2)).flatten()
        stds = stds[(std_min <= stds) * (stds <= std_max)]
        _, bin_edges_stds = np.histogram(stds, bins=bins)
        
        
        # Plotting
        fig, ((a1,a2), (a3,a4), (a5,a6), (a7,a8), (a9,a10)) = plt.subplots(5, 2, figsize=(8.2, 11), dpi=400)
        
        self.plot_posterior(
                       'Group 1 mean',
                       ax=a1,
                       bins=bin_edges_means,
                       stat='mean',
                       title='%s mean' % self.group1_name,
                       label=r'$\mu_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'Group 2 mean',
                       ax=a2,
                       bins=bin_edges_means,
                       stat='mean',
                       title='%s mean' % self.group2_name,
                       label=r'$\mu_2$')
        
        self.plot_posterior(
                       'Group 1 SD',
                       ax=a3,
                       bins=bin_edges_stds,
                       title='%s std. dev.' % self.group1_name,
                       label=r'$\mathrm{sd}_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'Group 2 SD',
                       ax=a4,
                       bins=bin_edges_stds,
                       title='%s std. dev.' % self.group2_name,
                       label=r'$\mathrm{sd}_2$')
        
        self.plot_posterior(
                       'Difference of Skew',
                       ax=a5,
                       bins=bins,
                       fcolor='lime',
                       title='Difference of Skew',
                       label=r'$\alpha_1 - \alpha_2$')

        self.plot_posterior(
                       'Difference of means',
                       ax=a6,
                       bins=bins,
                       title='Difference of means',
                       stat='mean',
                       ref_val=0,
                       label=r'$\mu_1 - \mu_2$',
                       fcolor='lime')
        
        self.plot_posterior(
                       'Difference of SDs',
                       ax=a7,
                       bins=bins,
                       title='Difference of std. dev.s',
                       ref_val=0,
                       label=r'$\mathrm{sd}_1 - \mathrm{sd}_2$',
                       fcolor='lime')
        
        self.plot_posterior(
                       'Effect size',
                       ax=a8,
                       bins=bins,
                       title='Effect size',
                       ref_val=1,
                       label=r'$((\mu_1 / \mu_2) + (\mathrm{sd}_1 / \mathrm{sd}_2) + (\alpha_1 / \alpha_2))/3$',
                       fcolor='lime')
        
        
        # Observed
        obs_vals = np.concatenate((self.y1, self.y2))
        bin_edges = np.linspace(np.min(obs_vals), np.max(obs_vals), bins + 1)
        
        self.plot_data_and_prediction(
                                 group_id=1,
                                 ax=a9,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group1_name,
                                 fcolor='salmon')
        
        self.plot_data_and_prediction(
                                 group_id=2,
                                 ax=a10,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group2_name)
        
        
        fig.tight_layout()


class BayesianHypothesisTestBeta:

    """ Note! input data has to be between 0 and 1.
    If any value is 1 this will throw an 'infinite logp' error
    """

    def __init__(self, df: pd.DataFrame, group1_name: str, group2_name: str, value_column: str, category_column: str):
        self.group1_name = group1_name
        self.group2_name = group2_name
        self.y1 = np.array(list(df[df[category_column] == group1_name][value_column]))
        self.y2 = np.array(list(df[df[category_column] == group2_name][value_column]))
        self.trace = None
        self.value_storage = {}


    def run_model(self, draws=2000, tune=1000):

        assert self.y1.ndim == 1
        assert self.y2.ndim == 1

        y_all = np.concatenate((self.y1, self.y2))
        
        alpha_both = beta.fit(y_all)[0] #to find prior value
        beta_both = beta.fit(y_all)[1] #to find prior value
        print(f"Alpha of both groups: {alpha_both}")
        print(f"Beta of both groups: {beta_both}")
        
        self.model = pm.Model()
        with self.model:
            # Priors for the β parameters of both groups (has to be positive, this is the spread)
            beta_group1 = pm.TruncatedNormal('beta_group1', mu=beta_both, sigma=1, lower=0.01, upper=20)
            beta_group2 = pm.TruncatedNormal('beta_group2', mu=beta_both, sigma=1, lower=0.01, upper=20)

            ## Alpha prior (this is the location), (must be positive)
            alpha_group1 = pm.TruncatedNormal('alpha_group1', mu=alpha_both, sigma=1, lower=0.01, upper=20)
            alpha_group2 = pm.TruncatedNormal('alpha_group2', mu=alpha_both, sigma=1, lower=0.01, upper=20)
            
            # Define likelihood for Group 1 data
            likelihood_group1 = pm.Beta('Group 1 data', alpha=alpha_group1, beta=beta_group1, observed=self.y1)
            
            # Define likelihood for Group 2 data
            likelihood_group2 = pm.Beta('Group 2 data', alpha=alpha_group2, beta=beta_group2, observed=self.y2)

            # Difference between the parameters of the two groups
            diff_beta = pm.Deterministic('diff_beta', beta_group1 - beta_group2)
            diff_alpha = pm.Deterministic('diff_alpha', alpha_group1 - alpha_group2)
            diff_mean = pm.Deterministic('diff_means', (alpha_group1 / (alpha_group1 + beta_group1)) - (alpha_group2 / (alpha_group2 + beta_group2)))
            diff_var = pm.Deterministic('diff_variance', ((alpha_group1*beta_group1)/ (((alpha_group1 + beta_group1 + 1)*(alpha_group1 + beta_group1)**2))\
                                                            - ((alpha_group2*beta_group2)/ (((alpha_group2 + beta_group2 + 2)*(alpha_group2 + beta_group2)**2)))))

            self.trace = pm.sample(tune=tune, draws=draws) #Runs markov-chain monte carlo

    
    def sample_posterior_distribution(self):
        samples = pm.sample_posterior_predictive(self.trace, self.model)
        group1_data = np.array(samples.posterior_predictive['Group 1 data']).flatten()
        group2_data = np.array(samples.posterior_predictive['Group 2 data']).flatten()

        num_samples1 = len(group1_data) if len(group1_data) < 50000 else 50000
        num_samples2 = len(group2_data) if len(group2_data) < 50000 else 50000
        self.group1_samples = np.random.choice(group1_data, num_samples1)
        self.group2_samples = np.random.choice(group2_data, num_samples2)
    

    def cliff_delta(self):
        """
        Finds cliff's delta (effect size) of posterior distribution.
        Clearly, -1 ≤ δ ≤ 1. Values near ±1 indicate the absence of
        overlap between the two samples, while values near zero indicate
        a lot of overlap between the two samples.
        """

        def cliffs_delta_calc(x, y):
            """Cliff's delta effect size"""
            pairs = 0
            ties = 0
            for a in x:
                for b in y:
                    if a > b:
                        pairs += 1
                    elif a == b:
                        ties += 1
            n = len(x) * len(y)
            return (pairs - ties) / n
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        cliff_delta_value = cliffs_delta_calc(group1_samples.flatten(), group2_samples.flatten())
        if 'cliff_delta' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['cliff_delta'].update({'cliff_delta': cliff_delta_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['cliff_delta'] = {'cliff_delta': cliff_delta_value}
        return cliff_delta_value
    
    def non_overlap_effect_size(self):
        """
        Finds the proportion of the two distirbutions 
        that do not overlap.
        0 indicates complete overlap
        1 indicates complete non-overlap
        """

        def nos(F, G):
            """Non-Overlap Effect Size"""
            min_val = min(np.min(F), np.min(G))
            max_val = max(np.max(F), np.max(G))
            bins = np.linspace(min_val, max_val, min(len(F), len(G)))
            hist_F, _ = np.histogram(F, bins=bins, density=True)
            hist_G, _ = np.histogram(G, bins=bins, density=True)
            overlap_area = np.minimum(hist_F, hist_G)
            nos_value = 1 - np.sum(overlap_area)
            return nos_value
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass
        
        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        nos_value = nos(group1_samples.flatten(), group2_samples.flatten())
        if 'non_overlap_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'].update({'non_overlap_effect_size': nos_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'] = {'non_overlap_effect_size': nos_value}
        return nos_value
    
    def divergence_effect_size(self):
        """
        Divergence Effect Size (DES) represents the magnitude of the difference
        between the two probability distributions.
        0/<0 = No difference
        >0 = difference, the larger the value the more dissimilar
        """
        def kl_divergence(p, q):
            """Calculate KL divergence between two probability distributions."""
            epsilon = 1e-10  # Small epsilon value to avoid taking log of zero
            p_safe = np.clip(p, epsilon, 1)  # Clip probabilities to avoid zeros
            q_safe = np.clip(q, epsilon, 1) 
            return np.sum(np.where(p_safe != 0, p_safe * np.log(p_safe / q_safe), 0))

        def make_symmetrical_divergence(F, G):
            """Makes Divergence Effect Size Symmetrical"""
            kl_FG = kl_divergence(F, G)
            kl_GF = kl_divergence(G, F)
            return (kl_FG + kl_GF) / 2
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        des_value = make_symmetrical_divergence(group1_samples.flatten(), group2_samples.flatten())
        if 'divergent_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['divergent_effect_size'].update({'divergent_effect_size': des_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['divergent_effect_size'] = {'divergent_effect_size': des_value}
        return des_value

    def plot_posterior(self,
                   var_name: str,
                   ax = None,
                   bins = 30,
                   stat = 'mode',
                   title = None,
                   label = None,
                   ref_val = None,
                   fcolor= '#89d1ea',
                   **kwargs) -> plt.Axes:
        """
        Plot a histogram of posterior samples of a variable
    
        Parameters
        ----------
        trace :
            The trace of the analysis.
        var_name : string
            The name of the variable to be plotted. Available variable names are
            described in the :ref:`sec-variables` section.
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new axes.
        bins : int or list or NumPy array
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        stat : {'mean', 'mode'}
            Whether to print the mean or the mode of the variable on the plot.
            Default: 'mode'.
        title : string, optional
            Title of the plot. Default: don’t print a title.
        label : string, optional
            Label of the *x* axis. Default: don’t print a label.
        ref_val : float, optional
            If not None, print a vertical line at this reference value (typically
            zero).
            Default: None (don’t print a reference value)
        **kwargs : dict
            All other keyword arguments are passed to `plt.hist`.
    
        Returns
        -------
        Matplotlib Axes
            The Axes object containing the plot. Using this return value, the
            plot can be customized afterwards – for details, see the documentation
            of the `Matplotlib Axes API <https://matplotlib.org/api/axes_api.html>`_.
        """

        samples_start = self.trace.posterior.data_vars[var_name]
        samples_min, samples_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.995).data_vars[var_name]))
        samples_middle = np.array(samples_start).flatten()
        samples = samples_middle[(samples_min <= samples_middle) * (samples_middle <= samples_max)]
    
        if ax is None:
            _, ax = plt.subplots()
    
        trans = blended_transform_factory(ax.transData, ax.transAxes)
    
        hist_kwargs = {'bins': bins}
        hist_kwargs.update(kwargs)
        ax.hist(samples, rwidth=0.8,
                facecolor=fcolor, edgecolor='none', **hist_kwargs)
    
        if stat:
            if stat == 'mode':
                # calculate mode using kernel density estimate
                kernel = scipy.stats.gaussian_kde(np.array(self.trace.posterior.data_vars[var_name]).flatten())
        
                bw = kernel.covariance_factor()
                cut = 3 * bw
                x_low = np.min(samples) - cut * bw
                x_high = np.max(samples) + cut * bw
                n = 512
                x = np.linspace(x_low, x_high, n)
                vals = kernel.evaluate(x)
                max_idx = np.argmax(vals)
                mode_val = x[max_idx]
                stat_val =  mode_val
                
            elif stat == 'mean':
                stat_val = np.mean(samples)
            else:
                raise ValueError('stat parameter must be either "mean" or "mode" '
                                'or None.')
    
            ax.text(stat_val, 0.99, '%s = %.3g' % (stat, stat_val),
                    transform=trans,
                    horizontalalignment='center',
                    verticalalignment='top',
                    )

            if var_name in self.value_storage:
                # Update the subdictionary with stat: 0
                self.value_storage[var_name].update({stat: stat_val})
            else:
                # Create a new subdictionary with stat: 0
                self.value_storage[var_name] = {stat: stat_val}



        if ref_val is not None:
            ax.axvline(ref_val, linestyle=':')
    
        # plot HDI
        hdi_min, hdi_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.95).data_vars[var_name]))
        if var_name in self.value_storage:
            # Update the subdictionary
            self.value_storage[var_name].update({'hdi_min': hdi_min,
                                                 'hdi_max': hdi_max})
        else:
            # Create a new subdictionary
            self.value_storage[var_name] = {'hdi_min': hdi_min,
                                            'hdi_max': hdi_max}
        hdi_line, = ax.plot([hdi_min, hdi_max], [0, 0],
                            lw=5.0, color='k')
        hdi_line.set_clip_on(False)
        ax.text(hdi_min, 0.04, '%.3g' % hdi_min,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text(hdi_max, 0.04, '%.3g' % hdi_max,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text((hdi_min + hdi_max) / 2, 0.14, '95% HDI',
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
    
        # make it pretty
        ax.spines['bottom'].set_position(('outward', 2))
        for loc in ['left', 'top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks([])  # don't draw
        for line in ax.get_xticklines():
            line.set_marker(mpllines.TICKDOWN)
        if label:
            ax.set_xlabel(label)
        if title is not None:
            ax.set_title(title)
            
        ax.set_xlim(samples_start.min(), samples_start.max())


        return ax
    
    def plot_data_and_prediction(self,
                                 group_id,
                                 ax = None,
                                 bins = 30,
                                 title = None,
                                 fcolor = '#89d1ea',
                                 hist_kwargs: dict = {},
                                 prediction_kwargs: dict = {}
                                 ) -> plt.Axes:
        """Plot samples of predictive distributions and a histogram of the data.
    
        This plot can be used as a *posterior predictive check*, to examine
        how well the model predictions fit the observed data.
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        group_id: 
            The observed data of one group
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new plot.
        title : string, optional.
            Title of the plot. Default: no plot title.
        bins : int or list or NumPy array.
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        hist_kwargs : dict
            The keyword arguments to be passed to `plt.hist` for the group data.
        prediction_kwargs : dict
            The keyword arguments to be passed to `plt.plot` for the posterior
            predictive curves.
    
        Returns
        -------
        Matplotlib Axes
        """
    
        if ax is None:
            _, ax = plt.subplots()


        group_data = self.y1 if group_id==1 else self.y2
        betas = np.array(self.trace.posterior.data_vars['beta_group%d' % group_id]).flatten()
        alphas = np.array(self.trace.posterior.data_vars['alpha_group%d' % group_id]).flatten()
    
        n_curves = 50
        n_samps = len(betas)
        idxs = np.random.choice(np.arange(n_samps), n_curves, replace=False)
    
        try:
            xmin = bins[0]
            xmax = bins[-1]
        except TypeError:
            xmin = np.min(group_data)
            xmax = np.max(group_data)
    
        dx = xmax - xmin
        xmin -= dx * 0.05
        xmax += dx * 0.05
    
        x = np.linspace(xmin, xmax, 1000)
    
        kwargs = dict(color=fcolor, zorder=1, alpha=0.7)
        kwargs.update(prediction_kwargs)

        for i in idxs:
            v = scipy.stats.beta.pdf(x, a=alphas[i], b=betas[i])
            line, = ax.plot(x, v, **kwargs)
        
        line.set_label('Prediction')

        kwargs = dict(edgecolor='w',
                      facecolor=fcolor,
                      density=True,
                      bins=bins,
                      label='Observation')
        kwargs.update(hist_kwargs)
        ax.hist(group_data, **kwargs)
    
        # draw a translucent histogram in front of the curves
        if 'zorder' not in hist_kwargs and 'alpha' not in hist_kwargs:
            kwargs.update(dict(zorder=3, label=None, alpha=0.3))
            ax.hist(group_data, **kwargs)
    
        ax.text(0.95, 0.95, r'$\mathrm{N}=%d$' % len(group_data),
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top'
                )
    
        for loc in ['top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.spines['left'].set_color('gray')
        ax.set_xlabel('Observation')
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylabel('Probability')
        ax.set_yticks([])
        ax.set_ylim(0)
        if title:
            ax.set_title(title)
    
        return ax
    
    def posterior_prob(self, var_name: str, low: float = -np.inf, high: float = np.inf):
        r"""Calculate the posterior probability that a variable is in a given interval
    
        The return value approximates the following probability:
    
        .. math:: \text{Pr}(\textit{low} < \theta_{\textit{var_name}} < \textit{high} | y_1, y_2)
    
        One-sided intervals can be specified by using only the ``low`` or ``high`` argument,
        for example, to calculate the probability that the the mean of the
        first group is larger than that of the second one::
    
            best_result.posterior_prob('Difference of means', low=0)
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        var_name : str
            Name of variable.
        low : float, optional
            Lower limit of the interval.
            Default: :math:`-\infty` (no lower limit)
        high : float, optional
            Upper limit of the interval.
            Default: :math:`\infty` (no upper limit)
    
        Returns
        -------
        float
            Posterior probability that the variable is in the given interval.
    
        Notes
        -----
        If *p* is the result and *S* is the total number of samples, then the
        standard deviation of the result is :math:`\sqrt{p(1-p)/S}`
        (see BDA3, p. 267). For example, with 2000 samples, the errors for
        some returned probabilities are
    
         - 0.01 ± 0.002,
         - 0.1 ± 0.007,
         - 0.2 ± 0.009,
         - 0.5 ± 0.011,
    
        meaning the answer is accurate for most practical purposes.
        """
        
        samples_start = self.trace.posterior.data_vars[var_name]
        samples_middle = np.array(samples_start).flatten()
        n_match = len(samples_middle[(low <= samples_middle) * (samples_middle <= high)])
        n_all = len(samples_middle)
        return n_match / n_all

    def plot_results(self, bins=30):    
        # Get Posterior Value
        # Betas
        posterior_beta1 = self.trace.posterior.data_vars['beta_group1']
        posterior_beta2 = self.trace.posterior.data_vars['beta_group2']
        posterior_betas = np.concatenate((posterior_beta1, posterior_beta2))
        _, bin_edges_betas = np.histogram(posterior_betas, bins=bins)

        # Alphas
        posterior_alpha1 = self.trace.posterior.data_vars['alpha_group1']
        posterior_alpha2 = self.trace.posterior.data_vars['alpha_group2']
        posterior_alphas = np.concatenate((posterior_alpha1, posterior_alpha2))
        _, bin_edges_alphas = np.histogram(posterior_alphas, bins=bins)
        
        
        # Plotting
        fig, ((a1,a2), (a3,a4), (a5,a6), (a7,a8), (a9,a10)) = plt.subplots(5, 2, figsize=(8.2, 12), dpi=400)
        
        self.plot_posterior(
                       'beta_group1',
                       ax=a1,
                       bins=bin_edges_betas,
                       stat='mean',
                       title='%s beta' % self.group1_name,
                       label=r'$\beta_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'beta_group2',
                       ax=a2,
                       bins=bin_edges_betas,
                       stat='mean',
                       title='%s beta' % self.group2_name,
                       label=r'$\beta_2$')
        
        self.plot_posterior(
                       'alpha_group1',
                       ax=a3,
                       bins=bin_edges_alphas,
                       stat='mean',
                       title='%s alpha' % self.group1_name,
                       label=r'$\alpha_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'alpha_group2',
                       ax=a4,
                       bins=bin_edges_alphas,
                       stat='mean',
                       title='%s alpha' % self.group2_name,
                       label=r'$\alpha_2$')
        
        self.plot_posterior(
                       'diff_beta',
                       ax=a5,
                       bins=bins,
                       title='Difference of betas',
                       stat='mean',
                       ref_val=0,
                       label=r'$\beta_1 - \beta_2$',
                       fcolor='lime')
        
        self.plot_posterior(
                       'diff_alpha',
                       ax=a6,
                       bins=bins,
                       title='Difference of alphas',
                       stat='mean',
                       ref_val=0,
                       label=r'$\alpha_1 - \alpha_2$',
                       fcolor='lime')
        
        self.plot_posterior(
                       'diff_means',
                       ax=a7,
                       bins=bins,
                       title='Difference of means',
                       stat='mean',
                       ref_val=0,
                       label=r'$\mu_1 - \mu_2$ | $\alpha / (\alpha + \beta)$',
                       fcolor='lime')

        self.plot_posterior(
                       'diff_variance',
                       ax=a8,
                       bins=bins,
                       title='Difference of variance',
                       stat='mean',
                       ref_val=0,
                       label=r'$\sigma^2_1 - \sigma^2_2$ | $\frac{\alpha \times \beta}{(\alpha + \beta)^2 \times (\alpha + \beta + 1)}$',
                       fcolor='lime')
        
        # Observed
        obs_vals = np.concatenate((self.y1, self.y2))
        bin_edges = np.linspace(np.min(obs_vals), np.max(obs_vals), bins + 1)
        
        self.plot_data_and_prediction(
                                 group_id=1,
                                 ax=a9,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group1_name,
                                 fcolor='salmon')
        
        self.plot_data_and_prediction(
                                 group_id=2,
                                 ax=a10,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group2_name)
        
        
        fig.tight_layout()

class BayesianHypothesisTestTruncNorm:

    """ Please specify lower and upper bound
    """

    def __init__(self, df: pd.DataFrame, group1_name: str, group2_name: str, value_column: str, category_column: str, lower: float, upper: float):
        self.group1_name = group1_name
        self.group2_name = group2_name
        self.y1 = np.array(list(df[df[category_column] == group1_name][value_column]))
        self.y2 = np.array(list(df[df[category_column] == group2_name][value_column]))
        self.trace = None
        self.value_storage = {}
        self.lower = lower
        self.upper = upper


    def run_model(self, draws=2000, tune=1000):

        assert self.y1.ndim == 1
        assert self.y2.ndim == 1

        y_all = np.concatenate((self.y1, self.y2))
        
        mu_both = np.mean(y_all)
        mu_scale = np.std(y_all) * 1000

        std_both = np.std(y_all)
        
        std_low = std_both / 1000
        std_high = std_both * 1000


        print(f"Mean of both groups: {mu_both}")
        print(f"Standard Deviation of both groups: {std_both}")
        
        self.model = pm.Model()
        with self.model:
            # Priors for the mean parameter (must be between 0 and 1)
            mean_group1 = pm.TruncatedNormal('mean_group1', mu=mu_both, sigma=1, lower=self.lower, upper=self.upper)
            mean_group2 = pm.TruncatedNormal('mean_group2', mu=mu_both, sigma=1, lower=self.lower, upper=self.upper)

            ## Std prior (can't be negative)
            std_group1 = pm.TruncatedNormal('std_group1', mu=std_both, sigma=1, lower=std_low, upper=std_high)
            std_group2 = pm.TruncatedNormal('std_group2', mu=std_both, sigma=1, lower=std_low, upper=std_high)
            
            # Define likelihood for Group 1 data
            likelihood_group1 = pm.TruncatedNormal('Group 1 data', mu=mean_group1, sigma=std_group1, lower=self.lower, upper=1, observed=self.y1)
            
            # Define likelihood for Group 2 data
            likelihood_group2 = pm.TruncatedNormal('Group 2 data', mu=mean_group2, sigma=std_group2, lower=self.lower, upper=self.upper, observed=self.y2)

            # Difference between the parameters of the two groups
            diff_mean = pm.Deterministic('diff_means', mean_group1 - mean_group2)
            diff_std = pm.Deterministic('diff_stds', std_group1 - std_group2)
            effect_size = pm.Deterministic(
                'Effect size', diff_mean / np.sqrt((std_group1 ** 2 + std_group2 ** 2) / 2)
            ) # Warning! Possible incorrect interperetation in non-symetrical data

            self.trace = pm.sample(tune=tune, draws=draws) #Runs markov-chain monte carlo

    def overlap_proportion(self, resolution=1000):
        """
        Finds the proportion of the two distributions 
        that overlap.
        1 indicates complete overlap
        0 indicates complete non-overlap
        """
        
        means1 = np.array(self.trace.posterior.data_vars['mean_group1']).flatten()
        stds1 = np.array(self.trace.posterior.data_vars['std_group1']).flatten()
        means2 = np.array(self.trace.posterior.data_vars['mean_group2']).flatten()
        stds2 = np.array(self.trace.posterior.data_vars['std_group2']).flatten()

        # Define parameters for the two truncated normal distributions
        mu1, sigma1 = np.mean(means1), np.mean(stds1)
        a1, b1 = self.lower, self.upper
        mu2, sigma2 = np.mean(means2), np.mean(stds2)
        a2, b2 = self.lower, self.upper

        # Define the range of values for the integration
        x = np.linspace(min(a1, a2), max(b1, b2), resolution)
        
        # Calculate the PDFs for the two distributions
        pdf1 = scipy.stats.truncnorm.pdf(x, (a1 - mu1) / sigma1, (b1 - mu1) / sigma1, loc=mu1, scale=sigma1)
        pdf2 = scipy.stats.truncnorm.pdf(x, (a2 - mu2) / sigma2, (b2 - mu2) / sigma2, loc=mu2, scale=sigma2)
        
        # Calculate the overlap by integrating the minimum of the PDFs
        overlap_area = np.trapz(np.minimum(pdf1, pdf2), x)
        
        # Calculate the total area under each PDF
        area1 = np.trapz(pdf1, x)
        area2 = np.trapz(pdf2, x)
        
        # Calculate the proportion of the total area that the overlap represents
        overlap_proportion = overlap_area / (area1 + area2 - overlap_area)

        if 'overlap_proportion' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['overlap_proportion'].update({'overlap_proportion': overlap_proportion})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['overlap_proportion'] = {'overlap_proportion': overlap_proportion}
        
        return overlap_proportion
    
    def sample_posterior_distribution(self):
        samples = pm.sample_posterior_predictive(self.trace, self.model)
        group1_data = np.array(samples.posterior_predictive['Group 1 data']).flatten()
        group2_data = np.array(samples.posterior_predictive['Group 2 data']).flatten()

        num_samples1 = len(group1_data) if len(group1_data) < 10000 else 10000
        num_samples2 = len(group2_data) if len(group2_data) < 10000 else 10000
        self.group1_samples = np.random.choice(group1_data, num_samples1)
        self.group2_samples = np.random.choice(group2_data, num_samples2)
    

    def cliff_delta(self):
        """
        Finds cliff's delta (effect size) of posterior distribution.
        Clearly, -1 ≤ δ ≤ 1. Values near ±1 indicate the absence of
        overlap between the two samples, while values near zero indicate
        a lot of overlap between the two samples.
        Will work best for integers
        Small effect: |δ| < 0.147
        Medium effect: 0.147 ≤ |δ| < 0.33
        Large effect: |δ| ≥ 0.33
        """

        def cliffs_delta_calc(x, y):
            """Cliff's delta effect size"""
            pairs = 0
            ties = 0
            for a in x:
                for b in y:
                    if a > b:
                        pairs += 1
                    elif a == b:
                        ties += 1
            n = len(x) * len(y)
            return (pairs - ties) / n
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        cliff_delta_value = cliffs_delta_calc(group1_samples.flatten(), group2_samples.flatten())
        if 'cliff_delta' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['cliff_delta'].update({'cliff_delta': cliff_delta_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['cliff_delta'] = {'cliff_delta': cliff_delta_value}
        return cliff_delta_value
    
    def proportion_difference(self):
        """
        Compares all samples from one distribution to all samples of another
        distribution. Returns a value indicating which distribution 
        has higher samples, therefore a higher overall distribution
        Clearly, -1 ≤ δ ≤ 1. Values near ±1 indicate the absence of
        overlap between the two samples, while values near zero indicate
        a lot of overlap between the two samples.
        Will work best for floats
        Small effect: |δ| < 0.147
        Medium effect: 0.147 ≤ |δ| < 0.33
        Large effect: |δ| ≥ 0.33
        """

        def prop_difference(x, y):
            more = 0
            less = 0
            for a in x:
                for b in y:
                    if a > b:
                        more += 1
                    elif a < b:
                        less += 1
            n = len(x) * len(y)
            return (more - less) / n
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        prop = prop_difference(group1_samples.flatten(), group2_samples.flatten())
        if 'prop_difference' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['prop_difference'].update({'prop_difference': prop})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['prop_difference'] = {'prop_difference': prop}
        return prop
    
    def non_overlap_effect_size(self):
        """
        Finds the proportion of the two distributions 
        that do not overlap.
        0 indicates complete overlap
        1 indicates complete non-overlap
        """

        def nos(F, G):
            """Calculate the overlapping area similarity between two distributions"""
            min_val = min(np.min(F), np.min(G))
            max_val = max(np.max(F), np.max(G))
            num_bins = min(len(F), len(G)) // 2  # Adjust the number of bins as needed
            bins = np.linspace(min_val, max_val, num_bins)
            hist_F, _ = np.histogram(F, bins=bins, density=True)
            hist_G, _ = np.histogram(G, bins=bins, density=True)
            
            # Calculate the overlap area using the minimum of histogram values
            overlap_area = np.minimum(hist_F, hist_G)
            
            # Calculate the total area under the overlapping histogram
            total_area = simps(overlap_area, bins[:-1])
            
            # Normalize the overlap area by the total area under both histograms
            total_area_F = simps(hist_F, bins[:-1])
            total_area_G = simps(hist_G, bins[:-1])
            overlap_similarity = 1 - (total_area / (total_area_F + total_area_G))
            
            return overlap_similarity
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        
        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        nos_value = nos(group1_samples.flatten(), group2_samples.flatten())
        if 'non_overlap_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'].update({'non_overlap_effect_size': nos_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'] = {'non_overlap_effect_size': nos_value}
        return nos_value
    
    def divergence_effect_size(self):
        """
        Divergence Effect Size (DES) represents the magnitude of the difference
        between the two probability distributions.
        0/<0 = No difference
        >0 = difference, the larger the value the more dissimilar
        """
        def kl_divergence(p, q):
            """Calculate KL divergence between two probability distributions."""
            epsilon = 1e-10  # Small epsilon value to avoid taking log of zero
            p_safe = np.clip(p, epsilon, 1)  # Clip probabilities to avoid zeros
            q_safe = np.clip(q, epsilon, 1) 
            return np.sum(np.where(p_safe != 0, p_safe * np.log(p_safe / q_safe), 0))

        def make_symmetrical_divergence(F, G):
            """Makes Divergence Effect Size Symmetrical"""
            kl_FG = kl_divergence(F, G)
            kl_GF = kl_divergence(G, F)
            return (kl_FG + kl_GF) / 2
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        des_value = make_symmetrical_divergence(group1_samples.flatten(), group2_samples.flatten())
        if 'divergent_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['divergent_effect_size'].update({'divergent_effect_size': des_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['divergent_effect_size'] = {'divergent_effect_size': des_value}
        return des_value

    def plot_posterior(self,
                   var_name: str,
                   ax = None,
                   bins = 30,
                   stat = 'mode',
                   title = None,
                   label = None,
                   ref_val = None,
                   fcolor= '#89d1ea',
                   **kwargs) -> plt.Axes:
        """
        Plot a histogram of posterior samples of a variable
    
        Parameters
        ----------
        trace :
            The trace of the analysis.
        var_name : string
            The name of the variable to be plotted. Available variable names are
            described in the :ref:`sec-variables` section.
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new axes.
        bins : int or list or NumPy array
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        stat : {'mean', 'mode'}
            Whether to print the mean or the mode of the variable on the plot.
            Default: 'mode'.
        title : string, optional
            Title of the plot. Default: don’t print a title.
        label : string, optional
            Label of the *x* axis. Default: don’t print a label.
        ref_val : float, optional
            If not None, print a vertical line at this reference value (typically
            zero).
            Default: None (don’t print a reference value)
        **kwargs : dict
            All other keyword arguments are passed to `plt.hist`.
    
        Returns
        -------
        Matplotlib Axes
            The Axes object containing the plot. Using this return value, the
            plot can be customized afterwards – for details, see the documentation
            of the `Matplotlib Axes API <https://matplotlib.org/api/axes_api.html>`_.
        """

        samples_start = self.trace.posterior.data_vars[var_name]
        samples_min, samples_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.995).data_vars[var_name]))
        samples_middle = np.array(samples_start).flatten()
        samples = samples_middle[(samples_min <= samples_middle) * (samples_middle <= samples_max)]
    
        if ax is None:
            _, ax = plt.subplots()
    
        trans = blended_transform_factory(ax.transData, ax.transAxes)
    
        hist_kwargs = {'bins': bins}
        hist_kwargs.update(kwargs)
        ax.hist(samples, rwidth=0.8,
                facecolor=fcolor, edgecolor='none', **hist_kwargs)
    
        if stat:
            if stat == 'mode':
                # calculate mode using kernel density estimate
                kernel = scipy.stats.gaussian_kde(np.array(self.trace.posterior.data_vars[var_name]).flatten())
        
                bw = kernel.covariance_factor()
                cut = 3 * bw
                x_low = np.min(samples) - cut * bw
                x_high = np.max(samples) + cut * bw
                n = 512
                x = np.linspace(x_low, x_high, n)
                vals = kernel.evaluate(x)
                max_idx = np.argmax(vals)
                mode_val = x[max_idx]
                stat_val =  mode_val
                
            elif stat == 'mean':
                stat_val = np.mean(samples)
            else:
                raise ValueError('stat parameter must be either "mean" or "mode" '
                                'or None.')
    
            ax.text(stat_val, 0.99, '%s = %.3g' % (stat, stat_val),
                    transform=trans,
                    horizontalalignment='center',
                    verticalalignment='top',
                    )

            if var_name in self.value_storage:
                # Update the subdictionary with stat: 0
                self.value_storage[var_name].update({stat: stat_val})
            else:
                # Create a new subdictionary with stat: 0
                self.value_storage[var_name] = {stat: stat_val}



        if ref_val is not None:
            ax.axvline(ref_val, linestyle=':')
    
        # plot HDI
        hdi_min, hdi_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.95).data_vars[var_name]))
        if var_name in self.value_storage:
            # Update the subdictionary
            self.value_storage[var_name].update({'hdi_min': hdi_min,
                                                 'hdi_max': hdi_max})
        else:
            # Create a new subdictionary
            self.value_storage[var_name] = {'hdi_min': hdi_min,
                                            'hdi_max': hdi_max}
        hdi_line, = ax.plot([hdi_min, hdi_max], [0, 0],
                            lw=5.0, color='k')
        hdi_line.set_clip_on(False)
        ax.text(hdi_min, 0.04, '%.3g' % hdi_min,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text(hdi_max, 0.04, '%.3g' % hdi_max,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text((hdi_min + hdi_max) / 2, 0.14, '95% HDI',
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
    
        # make it pretty
        ax.spines['bottom'].set_position(('outward', 2))
        for loc in ['left', 'top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks([])  # don't draw
        for line in ax.get_xticklines():
            line.set_marker(mpllines.TICKDOWN)
        if label:
            ax.set_xlabel(label)
        if title is not None:
            ax.set_title(title)
            
        ax.set_xlim(samples_start.min(), samples_start.max())


        return ax
    
    def plot_data_and_prediction(self,
                                 group_id,
                                 ax = None,
                                 bins = 30,
                                 title = None,
                                 fcolor = '#89d1ea',
                                 hist_kwargs: dict = {},
                                 prediction_kwargs: dict = {}
                                 ) -> plt.Axes:
        """Plot samples of predictive distributions and a histogram of the data.
    
        This plot can be used as a *posterior predictive check*, to examine
        how well the model predictions fit the observed data.
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        group_id: 
            The observed data of one group
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new plot.
        title : string, optional.
            Title of the plot. Default: no plot title.
        bins : int or list or NumPy array.
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        hist_kwargs : dict
            The keyword arguments to be passed to `plt.hist` for the group data.
        prediction_kwargs : dict
            The keyword arguments to be passed to `plt.plot` for the posterior
            predictive curves.
    
        Returns
        -------
        Matplotlib Axes
        """
    
        if ax is None:
            _, ax = plt.subplots()


        group_data = self.y1 if group_id==1 else self.y2
        means = np.array(self.trace.posterior.data_vars['mean_group%d' % group_id]).flatten()
        stds = np.array(self.trace.posterior.data_vars['std_group%d' % group_id]).flatten()
    
        n_curves = 50
        n_samps = len(means)
        idxs = np.random.choice(np.arange(n_samps), n_curves, replace=False)
    
        try:
            xmin = bins[0]
            xmax = bins[-1]
        except TypeError:
            xmin = np.min(group_data)
            xmax = np.max(group_data)
    
        dx = xmax - xmin
        xmin -= dx * 0.05
        xmax += dx * 0.05
    
        x = np.linspace(xmin, xmax, 1000)
    
        kwargs = dict(color=fcolor, zorder=1, alpha=0.7)
        kwargs.update(prediction_kwargs)

        for i in idxs:
            a, b = (0 - means[i]) / stds[i], (1 - means[i]) / stds[i]
            v = scipy.stats.truncnorm.pdf(x, loc=means[i], scale=stds[i], a=a, b=b)
            line, = ax.plot(x, v, **kwargs)
        
        line.set_label('Prediction')

        kwargs = dict(edgecolor='w',
                      facecolor=fcolor,
                      density=True,
                      bins=bins,
                      label='Observation')
        kwargs.update(hist_kwargs)
        ax.hist(group_data, **kwargs)
    
        # draw a translucent histogram in front of the curves
        if 'zorder' not in hist_kwargs and 'alpha' not in hist_kwargs:
            kwargs.update(dict(zorder=3, label=None, alpha=0.3))
            ax.hist(group_data, **kwargs)
    
        ax.text(0.95, 0.95, r'$\mathrm{N}=%d$' % len(group_data),
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top'
                )
    
        for loc in ['top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.spines['left'].set_color('gray')
        ax.set_xlabel('Observation')
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylabel('Probability')
        ax.set_yticks([])
        ax.set_ylim(0)
        if title:
            ax.set_title(title)
    
        return ax
    
    def posterior_prob(self, var_name: str, low: float = -np.inf, high: float = np.inf):
        r"""Calculate the posterior probability that a variable is in a given interval
    
        The return value approximates the following probability:
    
        .. math:: \text{Pr}(\textit{low} < \theta_{\textit{var_name}} < \textit{high} | y_1, y_2)
    
        One-sided intervals can be specified by using only the ``low`` or ``high`` argument,
        for example, to calculate the probability that the the mean of the
        first group is larger than that of the second one::
    
            best_result.posterior_prob('Difference of means', low=0)
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        var_name : str
            Name of variable.
        low : float, optional
            Lower limit of the interval.
            Default: :math:`-\infty` (no lower limit)
        high : float, optional
            Upper limit of the interval.
            Default: :math:`\infty` (no upper limit)
    
        Returns
        -------
        float
            Posterior probability that the variable is in the given interval.
    
        Notes
        -----
        If *p* is the result and *S* is the total number of samples, then the
        standard deviation of the result is :math:`\sqrt{p(1-p)/S}`
        (see BDA3, p. 267). For example, with 2000 samples, the errors for
        some returned probabilities are
    
         - 0.01 ± 0.002,
         - 0.1 ± 0.007,
         - 0.2 ± 0.009,
         - 0.5 ± 0.011,
    
        meaning the answer is accurate for most practical purposes.
        """
        
        samples_start = self.trace.posterior.data_vars[var_name]
        samples_middle = np.array(samples_start).flatten()
        n_match = len(samples_middle[(low <= samples_middle) * (samples_middle <= high)])
        n_all = len(samples_middle)
        return n_match / n_all

    def plot_results(self, bins=30):    
        # Get Posterior Value
        # Means
        posterior_mean1 = self.trace.posterior.data_vars['mean_group1']
        posterior_mean2 = self.trace.posterior.data_vars['mean_group2']
        posterior_means = np.concatenate((posterior_mean1, posterior_mean2))
        _, bin_edges_means = np.histogram(posterior_means, bins=bins)
        
        # Standard deviation
        posterior_std1 = self.trace.posterior.data_vars['std_group1']
        posterior_std2 = self.trace.posterior.data_vars['std_group2']
        std1_min, std1_max = tuple(np.array(arviz.hdi(self.trace, var_names=['std_group1'], hdi_prob=0.995).data_vars['std_group1']))
        std2_min, std2_max = tuple(np.array(arviz.hdi(self.trace, var_names=['std_group2'], hdi_prob=0.995).data_vars['std_group2']))
        std_min = min(std1_min, std2_min)
        std_max = max(std1_max, std2_max)
        stds = np.concatenate((posterior_std1, posterior_std2)).flatten()
        stds = stds[(std_min <= stds) * (stds <= std_max)]
        _, bin_edges_stds = np.histogram(stds, bins=bins)
        
        
        # Plotting
        fig, ((a1,a2), (a3,a4), (a5,a6), (a7,a8), (a9,a10)) = plt.subplots(5, 2, figsize=(8.2, 11), dpi=400)
        
        self.plot_posterior(
                       'mean_group1',
                       ax=a1,
                       bins=bin_edges_means,
                       stat='mean',
                       title='%s mean' % self.group1_name,
                       label=r'$\mu_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'mean_group2',
                       ax=a2,
                       bins=bin_edges_means,
                       stat='mean',
                       title='%s mean' % self.group2_name,
                       label=r'$\mu_2$')
        
        self.plot_posterior(
                       'std_group1',
                       ax=a3,
                       bins=bin_edges_stds,
                       title='%s std. dev.' % self.group1_name,
                       label=r'$\mathrm{sd}_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'std_group2',
                       ax=a4,
                       bins=bin_edges_stds,
                       title='%s std. dev.' % self.group2_name,
                       label=r'$\mathrm{sd}_2$')
        
        self.plot_posterior(
                       'diff_means',
                       ax=a5,
                       bins=bins,
                       title='Difference of means',
                       stat='mean',
                       ref_val=0,
                       label=r'$\mu_1 - \mu_2$',
                       fcolor='lime')
        
        self.plot_posterior(
                       'diff_stds',
                       ax=a6,
                       bins=bins,
                       title='Difference of standard deviation',
                       stat='mean',
                       ref_val=0,
                       label=r'$\mathrm{sd}_1 - \mathrm{sd}_2$',
                       fcolor='lime')
        
        self.plot_posterior(
                       'Effect size',
                       ax=a7,
                       bins=bins,
                       title='Effect size',
                       ref_val=0,
                       label=r'$(\mu_1 - \mu_2) / \sqrt{(\mathrm{sd}_1^2 + \mathrm{sd}_2^2)/2}$',
                       fcolor='lime')
        
        a8.axis('off')
        
        # Observed
        obs_vals = np.concatenate((self.y1, self.y2))
        bin_edges = np.linspace(np.min(obs_vals), np.max(obs_vals), bins + 1)
        
        self.plot_data_and_prediction(
                                 group_id=1,
                                 ax=a9,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group1_name,
                                 fcolor='salmon')
        
        self.plot_data_and_prediction(
                                 group_id=2,
                                 ax=a10,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group2_name)
        
        
        fig.tight_layout()


class BayesianHypothesisTestTruncStudentT:

    """ Please specify lower and upper bound
    """

    def __init__(self, df: pd.DataFrame, group1_name: str, group2_name: str, value_column: str, category_column: str, lower: float, upper: float):
        self.group1_name = group1_name
        self.group2_name = group2_name
        self.y1 = np.array(list(df[df[category_column] == group1_name][value_column]))
        self.y2 = np.array(list(df[df[category_column] == group2_name][value_column]))
        self.nu_min = 2.5
        self.trace = None
        self.value_storage = {}
        self.lower = lower
        self.upper = upper


    def run_model(self, draws=2000, tune=1000):

        assert self.y1.ndim == 1
        assert self.y2.ndim == 1

        y_all = np.concatenate((self.y1, self.y2))
        
        mu_both = np.mean(y_all)
        mu_scale = np.std(y_all) * 1000

        std_both = np.std(y_all)
        
        sigma_low = np.std(y_all) / 1000
        sigma_high = np.std(y_all) * 1000

        nu_min = self.nu_min # 2.5 prevents strong outliers and extremely large standard deviations
        nu_mean = 30


        print(f"Mean of both groups: {mu_both}")
        print(f"Standard Deviation of both groups: {std_both}")
        
        self.model = pm.Model()
        with self.model:
            # Priors for the mean parameter (must be between 0 and 1)
            mean_group1 = pm.TruncatedNormal('mean_group1', mu=mu_both, sigma=1, lower=self.lower, upper=self.upper)
            mean_group2 = pm.TruncatedNormal('mean_group2', mu=mu_both, sigma=1, lower=self.lower, upper=self.upper)

            # Normality prior
            nu = pm.Exponential('nu - %g' % nu_min, 1 / (nu_mean - nu_min)) + nu_min
            _ = pm.Deterministic('Normality', nu) 

            # Standard deviation Prior (must be positive)
            group1_logsigma = pm.Uniform(
                'Group 1 log sigma', lower=np.log(sigma_low), upper=np.log(sigma_high)
            )
            group2_logsigma = pm.Uniform(
                'Group 2 log sigma', lower=np.log(sigma_low), upper=np.log(sigma_high)
            )
            group1_sigma = pm.Deterministic('Group 1 sigma', np.exp(group1_logsigma))
            group2_sigma = pm.Deterministic('Group 2 sigma', np.exp(group2_logsigma))
            std_group1 = pm.Deterministic('std_group1', group1_sigma * (nu / (nu - 2)) ** 0.5)
            std_group2 = pm.Deterministic('std_group2', group2_sigma * (nu / (nu - 2)) ** 0.5)
        
            lambda1 = group1_sigma ** (-2)
            lambda2 = group2_sigma ** (-2)

            # Posterior distribution
            pre_likelihood_group1 = pm.StudentT.dist(mu=mean_group1, nu=nu, lam=lambda1)
            pre_likelihood_group2 = pm.StudentT.dist(mu=mean_group2, nu=nu, lam=lambda2)
            # Truncating
            likelihood_group1 = pm.Truncated('Group 1 data', pre_likelihood_group1, lower=self.lower, upper=self.upper, observed=self.y1)
            likelihood_group2 = pm.Truncated('Group 2 data', pre_likelihood_group2, lower=self.lower, upper=self.upper, observed=self.y2)
            
            # Difference between the parameters of the two groups
            diff_mean = pm.Deterministic('diff_means', mean_group1 - mean_group2)
            diff_std = pm.Deterministic('diff_stds', std_group1 - std_group2)
            effect_size = pm.Deterministic(
                'Effect size', diff_mean / np.sqrt((std_group1 ** 2 + std_group2 ** 2) / 2)
            )

            self.trace = pm.sample(tune=tune, draws=draws) #Runs markov-chain monte carlo

    
    def sample_posterior_distribution(self):
        samples = pm.sample_posterior_predictive(self.trace, self.model)
        group1_data = np.array(samples.posterior_predictive['Group 1 data']).flatten()
        group2_data = np.array(samples.posterior_predictive['Group 2 data']).flatten()

        num_samples1 = len(group1_data) if len(group1_data) < 50000 else 50000
        num_samples2 = len(group2_data) if len(group2_data) < 50000 else 50000
        self.group1_samples = np.random.choice(group1_data, num_samples1)
        self.group2_samples = np.random.choice(group2_data, num_samples2)
    

    def cliff_delta(self):
        """
        Finds cliff's delta (effect size) of posterior distribution.
        Clearly, -1 ≤ δ ≤ 1. Values near ±1 indicate the absence of
        overlap between the two samples, while values near zero indicate
        a lot of overlap between the two samples.
        """

        def cliffs_delta_calc(x, y):
            """Cliff's delta effect size"""
            pairs = 0
            ties = 0
            for a in x:
                for b in y:
                    if a > b:
                        pairs += 1
                    elif a == b:
                        ties += 1
            n = len(x) * len(y)
            return (pairs - ties) / n
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        cliff_delta_value = cliffs_delta_calc(group1_samples.flatten(), group2_samples.flatten())
        if 'cliff_delta' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['cliff_delta'].update({'cliff_delta': cliff_delta_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['cliff_delta'] = {'cliff_delta': cliff_delta_value}
        return cliff_delta_value
    
    def non_overlap_effect_size(self):
        """
        Finds the proportion of the two distirbutions 
        that do not overlap.
        0 indicates complete overlap
        1 indicates complete non-overlap
        """

        def nos(F, G):
            """Non-Overlap Effect Size"""
            min_val = min(np.min(F), np.min(G))
            max_val = max(np.max(F), np.max(G))
            bins = np.linspace(min_val, max_val, min(len(F), len(G)))
            hist_F, _ = np.histogram(F, bins=bins, density=True)
            hist_G, _ = np.histogram(G, bins=bins, density=True)
            overlap_area = np.minimum(hist_F, hist_G)
            nos_value = 1 - np.sum(overlap_area)
            return nos_value
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass
        
        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        nos_value = nos(group1_samples.flatten(), group2_samples.flatten())
        if 'non_overlap_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'].update({'non_overlap_effect_size': nos_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'] = {'non_overlap_effect_size': nos_value}
        return nos_value
    
    def divergence_effect_size(self):
        """
        Divergence Effect Size (DES) represents the magnitude of the difference
        between the two probability distributions.
        0/<0 = No difference
        >0 = difference, the larger the value the more dissimilar
        """
        def kl_divergence(p, q):
            """Calculate KL divergence between two probability distributions."""
            epsilon = 1e-10  # Small epsilon value to avoid taking log of zero
            p_safe = np.clip(p, epsilon, 1)  # Clip probabilities to avoid zeros
            q_safe = np.clip(q, epsilon, 1) 
            return np.sum(np.where(p_safe != 0, p_safe * np.log(p_safe / q_safe), 0))

        def make_symmetrical_divergence(F, G):
            """Makes Divergence Effect Size Symmetrical"""
            kl_FG = kl_divergence(F, G)
            kl_GF = kl_divergence(G, F)
            return (kl_FG + kl_GF) / 2
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        des_value = make_symmetrical_divergence(group1_samples.flatten(), group2_samples.flatten())
        if 'divergent_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['divergent_effect_size'].update({'divergent_effect_size': des_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['divergent_effect_size'] = {'divergent_effect_size': des_value}
        return des_value

    def plot_posterior(self,
                   var_name: str,
                   ax = None,
                   bins = 30,
                   stat = 'mode',
                   title = None,
                   label = None,
                   ref_val = None,
                   fcolor= '#89d1ea',
                   **kwargs) -> plt.Axes:
        """
        Plot a histogram of posterior samples of a variable
    
        Parameters
        ----------
        trace :
            The trace of the analysis.
        var_name : string
            The name of the variable to be plotted. Available variable names are
            described in the :ref:`sec-variables` section.
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new axes.
        bins : int or list or NumPy array
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        stat : {'mean', 'mode'}
            Whether to print the mean or the mode of the variable on the plot.
            Default: 'mode'.
        title : string, optional
            Title of the plot. Default: don’t print a title.
        label : string, optional
            Label of the *x* axis. Default: don’t print a label.
        ref_val : float, optional
            If not None, print a vertical line at this reference value (typically
            zero).
            Default: None (don’t print a reference value)
        **kwargs : dict
            All other keyword arguments are passed to `plt.hist`.
    
        Returns
        -------
        Matplotlib Axes
            The Axes object containing the plot. Using this return value, the
            plot can be customized afterwards – for details, see the documentation
            of the `Matplotlib Axes API <https://matplotlib.org/api/axes_api.html>`_.
        """

        samples_start = self.trace.posterior.data_vars[var_name]
        samples_min, samples_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.995).data_vars[var_name]))
        samples_middle = np.array(samples_start).flatten()
        samples = samples_middle[(samples_min <= samples_middle) * (samples_middle <= samples_max)]
    
        if ax is None:
            _, ax = plt.subplots()
    
        trans = blended_transform_factory(ax.transData, ax.transAxes)
    
        hist_kwargs = {'bins': bins}
        hist_kwargs.update(kwargs)
        ax.hist(samples, rwidth=0.8,
                facecolor=fcolor, edgecolor='none', **hist_kwargs)
    
        if stat:
            if stat == 'mode':
                # calculate mode using kernel density estimate
                kernel = scipy.stats.gaussian_kde(np.array(self.trace.posterior.data_vars[var_name]).flatten())
        
                bw = kernel.covariance_factor()
                cut = 3 * bw
                x_low = np.min(samples) - cut * bw
                x_high = np.max(samples) + cut * bw
                n = 512
                x = np.linspace(x_low, x_high, n)
                vals = kernel.evaluate(x)
                max_idx = np.argmax(vals)
                mode_val = x[max_idx]
                stat_val =  mode_val
                
            elif stat == 'mean':
                stat_val = np.mean(samples)
            else:
                raise ValueError('stat parameter must be either "mean" or "mode" '
                                'or None.')
    
            ax.text(stat_val, 0.99, '%s = %.3g' % (stat, stat_val),
                    transform=trans,
                    horizontalalignment='center',
                    verticalalignment='top',
                    )

            if var_name in self.value_storage:
                # Update the subdictionary with stat: 0
                self.value_storage[var_name].update({stat: stat_val})
            else:
                # Create a new subdictionary with stat: 0
                self.value_storage[var_name] = {stat: stat_val}



        if ref_val is not None:
            ax.axvline(ref_val, linestyle=':')
    
        # plot HDI
        hdi_min, hdi_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.95).data_vars[var_name]))
        if var_name in self.value_storage:
            # Update the subdictionary
            self.value_storage[var_name].update({'hdi_min': hdi_min,
                                                 'hdi_max': hdi_max})
        else:
            # Create a new subdictionary
            self.value_storage[var_name] = {'hdi_min': hdi_min,
                                            'hdi_max': hdi_max}
        hdi_line, = ax.plot([hdi_min, hdi_max], [0, 0],
                            lw=5.0, color='k')
        hdi_line.set_clip_on(False)
        ax.text(hdi_min, 0.04, '%.3g' % hdi_min,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text(hdi_max, 0.04, '%.3g' % hdi_max,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text((hdi_min + hdi_max) / 2, 0.14, '95% HDI',
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
    
        # make it pretty
        ax.spines['bottom'].set_position(('outward', 2))
        for loc in ['left', 'top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks([])  # don't draw
        for line in ax.get_xticklines():
            line.set_marker(mpllines.TICKDOWN)
        if label:
            ax.set_xlabel(label)
        if title is not None:
            ax.set_title(title)
            
        ax.set_xlim(samples_start.min(), samples_start.max())


        return ax
    
    def plot_data_and_prediction(self,
                                 group_id,
                                 ax = None,
                                 bins = 30,
                                 title = None,
                                 fcolor = '#89d1ea',
                                 hist_kwargs: dict = {},
                                 prediction_kwargs: dict = {}
                                 ) -> plt.Axes:
        """Plot samples of predictive distributions and a histogram of the data.
    
        This plot can be used as a *posterior predictive check*, to examine
        how well the model predictions fit the observed data.
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        group_id: 
            The observed data of one group
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new plot.
        title : string, optional.
            Title of the plot. Default: no plot title.
        bins : int or list or NumPy array.
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        hist_kwargs : dict
            The keyword arguments to be passed to `plt.hist` for the group data.
        prediction_kwargs : dict
            The keyword arguments to be passed to `plt.plot` for the posterior
            predictive curves.
    
        Returns
        -------
        Matplotlib Axes
        """
    
        if ax is None:
            _, ax = plt.subplots()


        group_data = self.y1 if group_id==1 else self.y2
        means = np.array(self.trace.posterior.data_vars['mean_group%d' % group_id]).flatten()
        stds = np.array(self.trace.posterior.data_vars['std_group%d' % group_id]).flatten()
    
        n_curves = 50
        n_samps = len(means)
        idxs = np.random.choice(np.arange(n_samps), n_curves, replace=False)
    
        try:
            xmin = bins[0]
            xmax = bins[-1]
        except TypeError:
            xmin = np.min(group_data)
            xmax = np.max(group_data)
    
        dx = xmax - xmin
        xmin -= dx * 0.05
        xmax += dx * 0.05
    
        x = np.linspace(xmin, xmax, 1000)
    
        kwargs = dict(color=fcolor, zorder=1, alpha=0.7)
        kwargs.update(prediction_kwargs)

        for i in idxs:
            a, b = (0 - means[i]) / stds[i], (1 - means[i]) / stds[i]
            v = scipy.stats.truncnorm.pdf(x, loc=means[i], scale=stds[i], a=a, b=b)
            line, = ax.plot(x, v, **kwargs)
        
        line.set_label('Prediction')

        kwargs = dict(edgecolor='w',
                      facecolor=fcolor,
                      density=True,
                      bins=bins,
                      label='Observation')
        kwargs.update(hist_kwargs)
        ax.hist(group_data, **kwargs)
    
        # draw a translucent histogram in front of the curves
        if 'zorder' not in hist_kwargs and 'alpha' not in hist_kwargs:
            kwargs.update(dict(zorder=3, label=None, alpha=0.3))
            ax.hist(group_data, **kwargs)
    
        ax.text(0.95, 0.95, r'$\mathrm{N}=%d$' % len(group_data),
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top'
                )
    
        for loc in ['top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.spines['left'].set_color('gray')
        ax.set_xlabel('Observation')
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylabel('Probability')
        ax.set_yticks([])
        ax.set_ylim(0)
        if title:
            ax.set_title(title)
    
        return ax
    
    def plot_normality_posterior(self, nu_min, ax, bins, title, fcolor):
    
        var_name = 'Normality'
        norm_bins = np.logspace(np.log10(nu_min),
                                np.log10(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.95).data_vars[var_name])[-1]),
                                num=bins + 1)
        self.plot_posterior(
                       var_name,
                       ax=ax,
                       bins=norm_bins,
                       title=title,
                       label=r'$\nu$',
                       fcolor=fcolor)
        ax.set_xlim(2.4, norm_bins[-1] * 1.05)
        ax.semilogx()
        # don't use scientific notation for tick labels
        tick_fmt = plt.LogFormatter()
        ax.xaxis.set_major_formatter(tick_fmt)
        ax.xaxis.set_minor_formatter(tick_fmt)
    
    def posterior_prob(self, var_name: str, low: float = -np.inf, high: float = np.inf):
        r"""Calculate the posterior probability that a variable is in a given interval
    
        The return value approximates the following probability:
    
        .. math:: \text{Pr}(\textit{low} < \theta_{\textit{var_name}} < \textit{high} | y_1, y_2)
    
        One-sided intervals can be specified by using only the ``low`` or ``high`` argument,
        for example, to calculate the probability that the the mean of the
        first group is larger than that of the second one::
    
            best_result.posterior_prob('Difference of means', low=0)
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        var_name : str
            Name of variable.
        low : float, optional
            Lower limit of the interval.
            Default: :math:`-\infty` (no lower limit)
        high : float, optional
            Upper limit of the interval.
            Default: :math:`\infty` (no upper limit)
    
        Returns
        -------
        float
            Posterior probability that the variable is in the given interval.
    
        Notes
        -----
        If *p* is the result and *S* is the total number of samples, then the
        standard deviation of the result is :math:`\sqrt{p(1-p)/S}`
        (see BDA3, p. 267). For example, with 2000 samples, the errors for
        some returned probabilities are
    
         - 0.01 ± 0.002,
         - 0.1 ± 0.007,
         - 0.2 ± 0.009,
         - 0.5 ± 0.011,
    
        meaning the answer is accurate for most practical purposes.
        """
        
        samples_start = self.trace.posterior.data_vars[var_name]
        samples_middle = np.array(samples_start).flatten()
        n_match = len(samples_middle[(low <= samples_middle) * (samples_middle <= high)])
        n_all = len(samples_middle)
        return n_match / n_all

    def plot_results(self, bins=30):    
        # Get Posterior Value
        # Means
        posterior_mean1 = self.trace.posterior.data_vars['mean_group1']
        posterior_mean2 = self.trace.posterior.data_vars['mean_group2']
        posterior_means = np.concatenate((posterior_mean1, posterior_mean2))
        _, bin_edges_means = np.histogram(posterior_means, bins=bins)
        
        # Standard deviation
        posterior_std1 = self.trace.posterior.data_vars['std_group1']
        posterior_std2 = self.trace.posterior.data_vars['std_group2']
        std1_min, std1_max = tuple(np.array(arviz.hdi(self.trace, var_names=['std_group1'], hdi_prob=0.995).data_vars['std_group1']))
        std2_min, std2_max = tuple(np.array(arviz.hdi(self.trace, var_names=['std_group2'], hdi_prob=0.995).data_vars['std_group2']))
        std_min = min(std1_min, std2_min)
        std_max = max(std1_max, std2_max)
        stds = np.concatenate((posterior_std1, posterior_std2)).flatten()
        stds = stds[(std_min <= stds) * (stds <= std_max)]
        _, bin_edges_stds = np.histogram(stds, bins=bins)
        
        
        # Plotting
        fig, ((a1,a2), (a3,a4), (a5,a6), (a7,a8), (a9,a10)) = plt.subplots(5, 2, figsize=(8.2, 11), dpi=400)
        
        self.plot_posterior(
                       'mean_group1',
                       ax=a1,
                       bins=bin_edges_means,
                       stat='mean',
                       title='%s mean' % self.group1_name,
                       label=r'$\mu_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'mean_group2',
                       ax=a2,
                       bins=bin_edges_means,
                       stat='mean',
                       title='%s mean' % self.group2_name,
                       label=r'$\mu_2$')
        
        self.plot_posterior(
                       'std_group1',
                       ax=a3,
                       bins=bin_edges_stds,
                       title='%s std. dev.' % self.group1_name,
                       label=r'$\mathrm{sd}_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'std_group2',
                       ax=a4,
                       bins=bin_edges_stds,
                       title='%s std. dev.' % self.group2_name,
                       label=r'$\mathrm{sd}_2$')
        
        self.plot_posterior(
                       'diff_means',
                       ax=a5,
                       bins=bins,
                       title='Difference of means',
                       stat='mean',
                       ref_val=0,
                       label=r'$\mu_1 - \mu_2$',
                       fcolor='lime')
        
        self.plot_posterior(
                       'diff_stds',
                       ax=a6,
                       bins=bins,
                       title='Difference of standard deviation',
                       stat='mean',
                       ref_val=0,
                       label=r'$\mathrm{sd}_1 - \mathrm{sd}_2$',
                       fcolor='lime')
        
        self.plot_posterior(
                       'Effect size',
                       ax=a7,
                       bins=bins,
                       title='Effect size',
                       ref_val=0,
                       label=r'$(\mu_1 - \mu_2) / \sqrt{(\mathrm{sd}_1^2 + \mathrm{sd}_2^2)/2}$',
                       fcolor='lime')
        
        self.plot_normality_posterior(self.nu_min, a8, bins, 'Normality', fcolor='lime')
        
        # Observed
        obs_vals = np.concatenate((self.y1, self.y2))
        bin_edges = np.linspace(np.min(obs_vals), np.max(obs_vals), bins + 1)
        
        self.plot_data_and_prediction(
                                 group_id=1,
                                 ax=a9,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group1_name,
                                 fcolor='salmon')
        
        self.plot_data_and_prediction(
                                 group_id=2,
                                 ax=a10,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group2_name)
        
        
        fig.tight_layout()


class BayesianHypothesisTestTruncSkewNormal:

    def __init__(self, df: pd.DataFrame, group1_name: str, group2_name: str, value_column: str, category_column: str, lower: float, upper: float):
        self.group1_name = group1_name
        self.group2_name = group2_name
        self.y1 = np.array(list(df[df[category_column] == group1_name][value_column]))
        self.y2 = np.array(list(df[df[category_column] == group2_name][value_column]))
        self.nu_min = 2.5
        self.trace = None
        self.value_storage = {}
        self.lower = lower
        self.upper = upper


    def run_model(self, draws=2000, tune=1000):
        """
        You should use this model if your data is skewed normal
        """

        assert self.y1.ndim == 1
        assert self.y2.ndim == 1

        y_all = np.concatenate((self.y1, self.y2))
        
        # assumption centres of parameter distributions are the values of the pooled groups
        mu_loc = np.mean(y_all)
        mu_scale = np.std(y_all) * 1000
        
        sigma_both = np.std(y_all)
        
        alpha_both = scipyskew(y_all)
        
        self.model = pm.Model()
        with self.model:
            # Prior assumption is the distribution of mean and standard deviation of the two groups are the same,
            # values are assumed as the mean and std of both groups. Truncated to min and max
            group1_mean = pm.TruncatedNormal('Group 1 mean', mu=mu_loc, sigma=mu_scale, lower=self.lower, upper=self.upper)
            group2_mean = pm.TruncatedNormal('Group 2 mean', mu=mu_loc, sigma=mu_scale, lower=self.lower, upper=self.upper)

            # Alpha prior (this is skew), set starting point as skew of both groups
            alpha_group1 = pm.Normal('alpha_group1', mu=alpha_both, sigma=2)
            alpha_group2 = pm.Normal('alpha_group2', mu=alpha_both, sigma=2)

            # Prior assumption of the standard deviation
            # Can't be negative
            group1_sd = pm.TruncatedNormal('Group 1 SD', mu=sigma_both, sigma=1, lower=0)
            group2_sd = pm.TruncatedNormal('Group 2 SD', mu=sigma_both, sigma=1, lower=0)
        
            _ = pm.SkewNormal('Group 1 data', observed=self.y1, mu=group1_mean, sigma=group1_sd, alpha=alpha_group1)
            _ = pm.SkewNormal('Group 2 data', observed=self.y2, mu=group2_mean, sigma=group2_sd, alpha=alpha_group2)
        
            diff_of_means = pm.Deterministic('Difference of means', group1_mean - group2_mean)
            diff_of_sd = pm.Deterministic('Difference of SDs', group1_sd - group2_sd)
            diff_of_skew = pm.Deterministic('Difference of Skew', alpha_group1 - alpha_group2)
            sd_ratio = pm.Deterministic('SD Ratio', group1_sd / group2_sd)
            skew_ratio = pm.Deterministic('Skew Ratio', alpha_group1 / alpha_group2)
            mean_ratio = pm.Deterministic('Mean Ratio', group1_mean / group2_mean)

            mes = pm.Deterministic('Effect size', (mean_ratio + sd_ratio + skew_ratio)/3)

            self.trace = pm.sample(tune=tune, draws=draws) #Runs markov-chain monte carlo
    
    def sample_posterior_distribution(self):
        samples = pm.sample_posterior_predictive(self.trace, self.model)
        group1_data = np.array(samples.posterior_predictive['Group 1 data']).flatten()
        group2_data = np.array(samples.posterior_predictive['Group 2 data']).flatten()

        num_samples1 = len(group1_data) if len(group1_data) < 50000 else 50000
        num_samples2 = len(group2_data) if len(group2_data) < 50000 else 50000
        self.group1_samples = np.random.choice(group1_data, num_samples1)
        self.group2_samples = np.random.choice(group2_data, num_samples2)
    

    def cliff_delta(self):
        """
        Finds cliff's delta (effect size) of posterior distribution.
        Clearly, -1 ≤ δ ≤ 1. Values near ±1 indicate the absence of
        overlap between the two samples, while values near zero indicate
        a lot of overlap between the two samples.
        """

        def cliffs_delta_calc(x, y):
            """Cliff's delta effect size"""
            pairs = 0
            ties = 0
            for a in x:
                for b in y:
                    if a > b:
                        pairs += 1
                    elif a == b:
                        ties += 1
            n = len(x) * len(y)
            return (pairs - ties) / n
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        cliff_delta_value = cliffs_delta_calc(group1_samples.flatten(), group2_samples.flatten())
        if 'cliff_delta' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['cliff_delta'].update({'cliff_delta': cliff_delta_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['cliff_delta'] = {'cliff_delta': cliff_delta_value}
        return cliff_delta_value
    
    def non_overlap_effect_size(self):
        """
        Finds the proportion of the two distirbutions 
        that do not overlap.
        0 indicates complete overlap
        1 indicates complete non-overlap
        """

        def nos(F, G):
            """Non-Overlap Effect Size"""
            min_val = min(np.min(F), np.min(G))
            max_val = max(np.max(F), np.max(G))
            bins = np.linspace(min_val, max_val, min(len(F), len(G)))
            hist_F, _ = np.histogram(F, bins=bins, density=True)
            hist_G, _ = np.histogram(G, bins=bins, density=True)
            overlap_area = np.minimum(hist_F, hist_G)
            nos_value = 1 - np.sum(overlap_area)
            return nos_value
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass
        
        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        nos_value = nos(group1_samples.flatten(), group2_samples.flatten())
        if 'non_overlap_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'].update({'non_overlap_effect_size': nos_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['non_overlap_effect_size'] = {'non_overlap_effect_size': nos_value}
        return nos_value
    
    def divergence_effect_size(self):
        """
        Divergence Effect Size (DES) represents the magnitude of the difference
        between the two probability distributions.
        0/<0 = No difference
        >0 = difference, the larger the value the more dissimilar
        """
        def kl_divergence(p, q):
            """Calculate KL divergence between two probability distributions."""
            epsilon = 1e-10  # Small epsilon value to avoid taking log of zero
            p_safe = np.clip(p, epsilon, 1)  # Clip probabilities to avoid zeros
            q_safe = np.clip(q, epsilon, 1) 
            return np.sum(np.where(p_safe != 0, p_safe * np.log(p_safe / q_safe), 0))

        def make_symmetrical_divergence(F, G):
            """Makes Divergence Effect Size Symmetrical"""
            kl_FG = kl_divergence(F, G)
            kl_GF = kl_divergence(G, F)
            return (kl_FG + kl_GF) / 2
        
        if not hasattr(self, 'group1_samples'):
            self.sample_posterior_distribution()
        else:
            pass

        group1_samples = self.group1_samples
        group2_samples = self.group2_samples
        des_value = make_symmetrical_divergence(group1_samples.flatten(), group2_samples.flatten())
        if 'divergent_effect_size' in self.value_storage:
            # Update the subdictionary with stat: 0
            self.value_storage['divergent_effect_size'].update({'divergent_effect_size': des_value})
        else:
            # Create a new subdictionary with stat: 0
            self.value_storage['divergent_effect_size'] = {'divergent_effect_size': des_value}
        return des_value
        
    def plot_posterior(self,
                   var_name: str,
                   ax = None,
                   bins = 30,
                   stat = 'mode',
                   title = None,
                   label = None,
                   ref_val = None,
                   fcolor= '#89d1ea',
                   **kwargs) -> plt.Axes:
        """
        Plot a histogram of posterior samples of a variable
    
        Parameters
        ----------
        trace :
            The trace of the analysis.
        var_name : string
            The name of the variable to be plotted. Available variable names are
            described in the :ref:`sec-variables` section.
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new axes.
        bins : int or list or NumPy array
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        stat : {'mean', 'mode'}
            Whether to print the mean or the mode of the variable on the plot.
            Default: 'mode'.
        title : string, optional
            Title of the plot. Default: don’t print a title.
        label : string, optional
            Label of the *x* axis. Default: don’t print a label.
        ref_val : float, optional
            If not None, print a vertical line at this reference value (typically
            zero).
            Default: None (don’t print a reference value)
        **kwargs : dict
            All other keyword arguments are passed to `plt.hist`.
    
        Returns
        -------
        Matplotlib Axes
            The Axes object containing the plot. Using this return value, the
            plot can be customized afterwards – for details, see the documentation
            of the `Matplotlib Axes API <https://matplotlib.org/api/axes_api.html>`_.
        """

        samples_start = self.trace.posterior.data_vars[var_name]
        samples_min, samples_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.995).data_vars[var_name]))
        samples_middle = np.array(samples_start).flatten()
        samples = samples_middle[(samples_min <= samples_middle) * (samples_middle <= samples_max)]
    
        if ax is None:
            _, ax = plt.subplots()
    
        trans = blended_transform_factory(ax.transData, ax.transAxes)
    
        hist_kwargs = {'bins': bins}
        hist_kwargs.update(kwargs)
        ax.hist(samples, rwidth=0.8,
                facecolor=fcolor, edgecolor='none', **hist_kwargs)
    
        if stat:
            if stat == 'mode':
                # calculate mode using kernel density estimate
                kernel = scipy.stats.gaussian_kde(np.array(self.trace.posterior.data_vars[var_name]).flatten())
        
                bw = kernel.covariance_factor()
                cut = 3 * bw
                x_low = np.min(samples) - cut * bw
                x_high = np.max(samples) + cut * bw
                n = 512
                x = np.linspace(x_low, x_high, n)
                vals = kernel.evaluate(x)
                max_idx = np.argmax(vals)
                mode_val = x[max_idx]
                stat_val =  mode_val
                
            elif stat == 'mean':
                stat_val = np.mean(samples)
            else:
                raise ValueError('stat parameter must be either "mean" or "mode" '
                                'or None.')
    
            ax.text(stat_val, 0.99, '%s = %.3g' % (stat, stat_val),
                    transform=trans,
                    horizontalalignment='center',
                    verticalalignment='top',
                    )

            if var_name in self.value_storage:
                # Update the subdictionary with stat: 0
                self.value_storage[var_name].update({stat: stat_val})
            else:
                # Create a new subdictionary with stat: 0
                self.value_storage[var_name] = {stat: stat_val}



        if ref_val is not None:
            ax.axvline(ref_val, linestyle=':')
    
        # plot HDI
        hdi_min, hdi_max = tuple(np.array(arviz.hdi(self.trace, var_names=[var_name], hdi_prob=0.95).data_vars[var_name]))
        if var_name in self.value_storage:
            # Update the subdictionary
            self.value_storage[var_name].update({'hdi_min': hdi_min,
                                                 'hdi_max': hdi_max})
        else:
            # Create a new subdictionary
            self.value_storage[var_name] = {'hdi_min': hdi_min,
                                            'hdi_max': hdi_max}
        hdi_line, = ax.plot([hdi_min, hdi_max], [0, 0],
                            lw=5.0, color='k')
        hdi_line.set_clip_on(False)
        ax.text(hdi_min, 0.04, '%.3g' % hdi_min,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text(hdi_max, 0.04, '%.3g' % hdi_max,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text((hdi_min + hdi_max) / 2, 0.14, '95% HDI',
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
    
        # make it pretty
        ax.spines['bottom'].set_position(('outward', 2))
        for loc in ['left', 'top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks([])  # don't draw
        for line in ax.get_xticklines():
            line.set_marker(mpllines.TICKDOWN)
        if label:
            ax.set_xlabel(label)
        if title is not None:
            ax.set_title(title)
            
        ax.set_xlim(samples_start.min(), samples_start.max())


        return ax
    
    def plot_data_and_prediction(self,
                                 group_id,
                                 ax = None,
                                 bins = 30,
                                 title = None,
                                 fcolor = '#89d1ea',
                                 hist_kwargs: dict = {},
                                 prediction_kwargs: dict = {}
                                 ) -> plt.Axes:
        """Plot samples of predictive distributions and a histogram of the data.
    
        This plot can be used as a *posterior predictive check*, to examine
        how well the model predictions fit the observed data.
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        group_id: 
            The observed data of one group
        ax : Matplotlib Axes, optional
            If not None, the Matplotlib Axes instance to be used.
            Default: create a new plot.
        title : string, optional.
            Title of the plot. Default: no plot title.
        bins : int or list or NumPy array.
            The number or edges of the bins used for the histogram of the data.
            If an integer, the number of bins to use.
            If a sequence, then the edges of the bins, including left edge
            of the first bin and right edge of the last bin.
            Default: 30 bins.
        hist_kwargs : dict
            The keyword arguments to be passed to `plt.hist` for the group data.
        prediction_kwargs : dict
            The keyword arguments to be passed to `plt.plot` for the posterior
            predictive curves.
    
        Returns
        -------
        Matplotlib Axes
        """
    
        if ax is None:
            _, ax = plt.subplots()

        group_data = self.y1 if group_id==1 else self.y2
        means = np.array(self.trace.posterior.data_vars['Group %d mean' % group_id]).flatten()
        sigmas = np.array(self.trace.posterior.data_vars['Group %d SD' % group_id]).flatten()
        alphas = np.array(self.trace.posterior.data_vars['alpha_group%d' % group_id]).flatten()
    
        n_curves = 50
        n_samps = len(means)
        idxs = np.random.choice(np.arange(n_samps), n_curves, replace=False)
    
        try:
            xmin = bins[0]
            xmax = bins[-1]
        except TypeError:
            xmin = np.min(group_data)
            xmax = np.max(group_data)
    
        dx = xmax - xmin
        xmin -= dx * 0.05
        xmax += dx * 0.05
    
        x = np.linspace(xmin, xmax, 1000)
    
        kwargs = dict(color=fcolor, zorder=1, alpha=0.3)
        kwargs.update(prediction_kwargs)

        for i in idxs:
            v = scipy.stats.skewnorm.pdf(x, alphas[i], means[i], sigmas[i])
            line, = ax.plot(x, v, **kwargs)
        
        line.set_label('Prediction')
    
        kwargs = dict(edgecolor='w',
                      facecolor=fcolor,
                      density=True,
                      bins=bins,
                      label='Observation')
        kwargs.update(hist_kwargs)
        ax.hist(group_data, **kwargs)
    
        # draw a translucent histogram in front of the curves
        if 'zorder' not in hist_kwargs and 'alpha' not in hist_kwargs:
            kwargs.update(dict(zorder=3, label=None, alpha=0.3))
            ax.hist(group_data, **kwargs)
    
        ax.text(0.95, 0.95, r'$\mathrm{N}=%d$' % len(group_data),
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top'
                )
    
        for loc in ['top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.spines['left'].set_color('gray')
        ax.set_xlabel('Observation')
        ax.set_xlim(self.lower, self.upper)
        ax.set_ylabel('Probability')
        #ax.set_yticks([])
        
        ax.set_ylim(0)
        if title:
            ax.set_title(title)
    
        return ax
    
    def posterior_prob(self, var_name: str, low: float = -np.inf, high: float = np.inf):
        r"""Calculate the posterior probability that a variable is in a given interval
    
        The return value approximates the following probability:
    
        .. math:: \text{Pr}(\textit{low} < \theta_{\textit{var_name}} < \textit{high} | y_1, y_2)
    
        One-sided intervals can be specified by using only the ``low`` or ``high`` argument,
        for example, to calculate the probability that the the mean of the
        first group is larger than that of the second one::
    
            best_result.posterior_prob('Difference of means', low=0)
    
        Parameters
        ----------
        self.trace:
            The result of the analysis.
        var_name : str
            Name of variable.
        low : float, optional
            Lower limit of the interval.
            Default: :math:`-\infty` (no lower limit)
        high : float, optional
            Upper limit of the interval.
            Default: :math:`\infty` (no upper limit)
    
        Returns
        -------
        float
            Posterior probability that the variable is in the given interval.
    
        Notes
        -----
        If *p* is the result and *S* is the total number of samples, then the
        standard deviation of the result is :math:`\sqrt{p(1-p)/S}`
        (see BDA3, p. 267). For example, with 2000 samples, the errors for
        some returned probabilities are
    
         - 0.01 ± 0.002,
         - 0.1 ± 0.007,
         - 0.2 ± 0.009,
         - 0.5 ± 0.011,
    
        meaning the answer is accurate for most practical purposes.
        """
        
        samples_start = self.trace.posterior.data_vars[var_name]
        samples_middle = np.array(samples_start).flatten()
        n_match = len(samples_middle[(low <= samples_middle) * (samples_middle <= high)])
        n_all = len(samples_middle)
        return n_match / n_all

    def plot_results(self, bins=30):    
        # Get Posterior Value
        # Means
        posterior_mean1 = self.trace.posterior.data_vars['Group 1 mean']
        posterior_mean2 = self.trace.posterior.data_vars['Group 2 mean']
        posterior_means = np.concatenate((posterior_mean1, posterior_mean2))
        _, bin_edges_means = np.histogram(posterior_means, bins=bins)
        
        # Standard deviation
        posterior_std1 = self.trace.posterior.data_vars['Group 1 SD']
        posterior_std2 = self.trace.posterior.data_vars['Group 2 SD']
        std1_min, std1_max = tuple(np.array(arviz.hdi(self.trace, var_names=['Group 1 SD'], hdi_prob=0.995).data_vars['Group 1 SD']))
        std2_min, std2_max = tuple(np.array(arviz.hdi(self.trace, var_names=['Group 2 SD'], hdi_prob=0.995).data_vars['Group 2 SD']))
        std_min = min(std1_min, std2_min)
        std_max = max(std1_max, std2_max)
        stds = np.concatenate((posterior_std1, posterior_std2)).flatten()
        stds = stds[(std_min <= stds) * (stds <= std_max)]
        _, bin_edges_stds = np.histogram(stds, bins=bins)
        
        
        # Plotting
        fig, ((a1,a2), (a3,a4), (a5,a6), (a7,a8), (a9,a10)) = plt.subplots(5, 2, figsize=(8.2, 11), dpi=400)
        
        self.plot_posterior(
                       'Group 1 mean',
                       ax=a1,
                       bins=bin_edges_means,
                       stat='mean',
                       title='%s mean' % self.group1_name,
                       label=r'$\mu_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'Group 2 mean',
                       ax=a2,
                       bins=bin_edges_means,
                       stat='mean',
                       title='%s mean' % self.group2_name,
                       label=r'$\mu_2$')
        
        self.plot_posterior(
                       'Group 1 SD',
                       ax=a3,
                       bins=bin_edges_stds,
                       title='%s std. dev.' % self.group1_name,
                       label=r'$\mathrm{sd}_1$',
                       fcolor='salmon')
        
        self.plot_posterior(
                       'Group 2 SD',
                       ax=a4,
                       bins=bin_edges_stds,
                       title='%s std. dev.' % self.group2_name,
                       label=r'$\mathrm{sd}_2$')
        
        self.plot_posterior(
                       'Difference of Skew',
                       ax=a5,
                       bins=bins,
                       fcolor='lime',
                       title='Difference of Skew',
                       label=r'$\alpha_1 - \alpha_2$')

        self.plot_posterior(
                       'Difference of means',
                       ax=a6,
                       bins=bins,
                       title='Difference of means',
                       stat='mean',
                       ref_val=0,
                       label=r'$\mu_1 - \mu_2$',
                       fcolor='lime')
        
        self.plot_posterior(
                       'Difference of SDs',
                       ax=a7,
                       bins=bins,
                       title='Difference of std. dev.s',
                       ref_val=0,
                       label=r'$\mathrm{sd}_1 - \mathrm{sd}_2$',
                       fcolor='lime')
        
        self.plot_posterior(
                       'Effect size',
                       ax=a8,
                       bins=bins,
                       title='Effect size',
                       ref_val=1,
                       label=r'$((\mu_1 / \mu_2) + (\mathrm{sd}_1 / \mathrm{sd}_2) + (\alpha_1 / \alpha_2))/3$',
                       fcolor='lime')
        
        
        # Observed
        obs_vals = np.concatenate((self.y1, self.y2))
        bin_edges = np.linspace(np.min(obs_vals), np.max(obs_vals), bins + 1)
        
        self.plot_data_and_prediction(
                                 group_id=1,
                                 ax=a9,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group1_name,
                                 fcolor='salmon')
        
        self.plot_data_and_prediction(
                                 group_id=2,
                                 ax=a10,
                                 bins=bin_edges,
                                 title='%s data with post. pred.' % self.group2_name)
        
        
        fig.tight_layout()