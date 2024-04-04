import numpy as np
import pandas as pd
import pymc as pm #this is pymc v2, v3 has error with disutils so won't work till thats fixed
import arviz
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import scipy
import matplotlib.lines as mpllines
from scipy.stats import halfcauchy

class BayesianHypothesisTest:

    def __init__(self, df: pd.DataFrame, group1_name: str, group2_name: str, value_column: str, category_column: str):
        self.group1_name = group1_name
        self.group2_name = group2_name
        self.y1 = np.array(list(df[df[category_column] == group1_name][value_column]))
        self.y2 = np.array(list(df[df[category_column] == group2_name][value_column]))
        self.nu_min = 2.5
        self.trace = None
        self.value_storage = {}


    def logp_skewt(self, value, nu, mu, sigma, alpha):
        """
        Custom distribution giving density function of skewed student T"""
        return (np.log(2) + 
            pm.logp(pm.StudentT.dist(nu, mu=mu, sigma=sigma), value) + 
            pm.logcdf(pm.StudentT.dist(nu, mu=mu, sigma=sigma), alpha*value) - 
            np.log(sigma))


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
        
        model = pm.Model()
        with model:
            # Prior assumption is the distribution of mean and standard deviation of the two groups are the same,
            # values are assumed as the mean and std of both groups
            group1_mean = pm.Normal('Group 1 mean', mu=mu_loc, tau=1/mu_scale**2)
            group2_mean = pm.Normal('Group 2 mean', mu=mu_loc, tau=1/mu_scale**2)
        
            # Prior assumption of the height of the t distribtion (greek letter nu 𝜈)
            # This normality distribution is fed by both groups, since outliers are rare
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


    def logp_skewt(self, value, nu, mu, sigma, alpha):
        """
        Custom distribution giving density function of skewed student T"""
        return (np.log(2) + 
            pm.logp(pm.StudentT.dist(nu, mu=mu, sigma=sigma), value) + 
            pm.logcdf(pm.StudentT.dist(nu, mu=mu, sigma=sigma), alpha*value) - 
            np.log(sigma))


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
        
        model = pm.Model()
        with model:
            # Priors for the β parameters of both groups
            beta_group1 = pm.HalfCauchy('beta_group1', beta=beta_both)
            beta_group2 = pm.HalfCauchy('beta_group2', beta=beta_both)

            alpha_group1 = pm.Cauchy('alpha_group1', alpha=mode_all, beta=1)
            alpha_group2 = pm.Cauchy('alpha_group2', alpha=mode_all, beta=1)
            
            # Define likelihood for Group 1 data
            likelihood_group1 = pm.Cauchy('Group 1 data', alpha=alpha_group1, beta=beta_group1, observed=self.y1)
            
            # Define likelihood for Group 2 data
            likelihood_group2 = pm.Cauchy('Group 2 data', alpha=alpha_group2, beta=beta_group2, observed=self.y2)

            # Difference between the β parameters of the two groups
            diff_beta = pm.Deterministic('diff_beta', beta_group1 - beta_group2)
            diff_alpha = pm.Deterministic('diff_alpha', alpha_group1 - alpha_group2)
            
            self.trace = pm.sample(tune=tune, draws=draws) #Runs markov-chain monte carlo
    
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


    def logp_skewt(self, value, nu, mu, sigma, alpha):
        """
        Custom distribution giving density function of skewed student T"""
        return (np.log(2) + 
            pm.logp(pm.StudentT.dist(nu, mu=mu, sigma=sigma), value) + 
            pm.logcdf(pm.StudentT.dist(nu, mu=mu, sigma=sigma), alpha*value) - 
            np.log(sigma))


    def run_model(self, draws=2000, tune=1000):
        """
        You should use this model if your data is quite normal.
        Model Structure: paper https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf
        """

        assert self.y1.ndim == 1
        assert self.y2.ndim == 1

        # Model structure and parameters are set as those in the paper https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf
        y_all = np.concatenate((self.y1, self.y2))
        
        print(halfcauchy.fit(y_all))
        beta_both = halfcauchy.fit(y_all)[1]
        mode_all = scipy.stats.mode(y_all)[0]
        
        model = pm.Model()
        with model:
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