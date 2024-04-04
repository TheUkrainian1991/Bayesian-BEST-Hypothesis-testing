""" Contains attempts at various distributions.
 I set out to see if there was a better way to model skew of the data.
 My goal was to find a model that could best explain large
 peaks at the extreme ranges of the data.
 I first began with a skewed t distribution, and then attempted a gamma
 distribution (with some handling if the skew qwas neagitve that involved reversing the data).
 I found that no matter what distribution I used, because statistical testing is inherently focused
 on means, there is no way to explain a large peak (mode) of the data in any distribution, unless
 the mode was close to the mean. I concluded that the best distribution to use was the
 T distribution in use of the model described in the 2013 paper, since the heavy tails
 weigh more probability to these extreme values."""

def run_skewed_t_model(self, draws=2000, tune=1000):
        """
        You should only use this model if your data is skewed,
        since a distribution of skewed T-distribution is implemented
        for the observed variables. This is incredibly slow 100x slower.
        Model Structure: # paper https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf
        Skewed T ideas from:
        https://discourse.pymc.io/t/boosting-efficiency-of-skewed-t-models-in-pymc/12892/2
        """
        assert self.y1.ndim == 1
        assert self.y2.ndim == 1

        y_all = np.concatenate((self.y1, self.y2))
        
        mu_loc = np.mean(y_all)
        mu_scale = np.std(y_all) * 1000
        
        sigma_low = np.std(y_all) / 1000
        sigma_high = np.std(y_all) * 1000
        
        # the shape of the t-distribution changes noticeably for values of 𝜈
        # near 3 but changes relatively little for 𝜈>30
        nu_min = self.nu_min # 2.5 prevents strong outliers and extremely large standard deviations
        nu_mean = 30
        _nu_param = nu_mean - nu_min

        alpha_both = scipyskew(y_all)
        
        model = pm.Model()
        with model:
            # Prior assumption is the distribution of mean and standard deviation of the two groups are the same,
            # values are assumed as the mean and std of both groups
            group1_mean = pm.Normal('Group 1 mean', mu=mu_loc, tau=1/mu_scale**2)
            group2_mean = pm.Normal('Group 2 mean', mu=mu_loc, tau=1/mu_scale**2)
        
            # Prior assumption of the height of the t distribution (greek letter nu 𝜈)
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

            # Prior assumption of the standard deviation
            group1_sd = pm.Deterministic('Group 1 SD', group1_sigma * (nu / (nu - 2)) ** 0.5)
            group2_sd = pm.Deterministic('Group 2 SD', group2_sigma * (nu / (nu - 2)) ** 0.5)
            
            # Prior assumption of alpha - measure of the skewness of the distribution
            group1_alpha_skew = pm.StudentT('group1_alpha_skew', nu=3, mu=alpha_both, sigma=1)
            group2_alpha_skew = pm.StudentT('group2_alpha_skew', nu=3, mu=alpha_both, sigma=1)

            # Custom distribution is a skewed t-distribution
            # These need to be positional arguments only
            # Broadcasting is applied here
            _ = pm.CustomDist("Group 1 data", nu*np.ones(len(self.y1)), group1_mean,  group1_sigma, group1_alpha_skew, logp=self.logp_skewt, observed=self.y1)
            _ = pm.CustomDist("Group 2 data", nu*np.ones(len(self.y2)), group2_mean,  group2_sigma, group2_alpha_skew, logp=self.logp_skewt, observed=self.y2)

            diff_of_means = pm.Deterministic('Difference of means', group1_mean - group2_mean)
            _ = pm.Deterministic('Difference of SDs', group1_sd - group2_sd)
            _ = pm.Deterministic(
                'Effect size', diff_of_means / np.sqrt((group1_sd ** 2 + group2_sd ** 2) / 2)
            )
            
            self.trace = pm.sample(tune=tune, draws=draws, target_accept=0.95) #Runs markov-chain monte carlo

    def run_skewed_gamma_model(self, draws=2000, tune=1000):
        """
        You should only use this model if your data is skewed,
        since a distribution of skewed gamma is implemented
        for the observed variables. This is incredibly slow 100x slower.
        Model Structure: # paper https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf

        If your data is negatively skewed then the data will be inversed.
        """

        assert self.y1.ndim == 1
        assert self.y2.ndim == 1

        y_all = np.concatenate((self.y1, self.y2))

        if np.median(y_all) < np.mean(y_all):
            self.positive_skew = True
            y1 = self.y1
            y2 = self.y2
        else:
            self.positive_skew = False
            y_all = (np.max(y_all) - y_all)
            y2 = (self.y2max - self.y2)
            y1 = (self.y1max - self.y1)
            print(self.y2)
            print(y2)
        
        # 0 values result in divide 0 error for gamma distribution, so this circumvents that
        y_all[y_all == 0] += 1e-6
        y1[y1 == 0] += 1e-6
        y2[y2 == 0] += 1e-6

        y_all_std = np.std(y_all)
        
        mu_loc = np.mean(y_all)
        mode_all = scipy.stats.mode(y_all)[0]
        mu_scale = y_all_std * 1000
        
        sigma_low = y_all_std / 1000
        sigma_high = y_all_std * 1000
        
        # the shape of the t-distribution changes noticeably for values of 𝜈
        # near 3 but changes relatively little for 𝜈>30
        nu_min = self.nu_min # 2.5 prevents strong outliers and extremely large standard deviations
        nu_mean = 30
        _nu_param = nu_mean - nu_min

        # Calculate mode of the gamma distribution (assuming shape > 1)
        mode_gamma = (mu_loc - 1) * np.std(y_all)

        # Set the standard deviation for the normal prior to control the spread around the mode
        std_dev = 0.1 * mode_gamma  # Adjust this factor as needed
        
        model = pm.Model()
        with model:
            # Prior assumption is the distribution of mean and standard deviation of the two groups are the same,
            # values are assumed as the mean and std of both groups
            group1_mean = pm.TruncatedNormal('Group 1 mean', mu=mode_all, sigma=y_all_std, lower=0)
            group2_mean = pm.TruncatedNormal('Group 2 mean', mu=mode_all, sigma=y_all_std, lower=0)
        
            # Prior assumption of the height of the t distribution (greek letter nu 𝜈)
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

            # Prior assumption of the standard deviation
            group1_sd = pm.Deterministic('Group 1 SD', group1_sigma * (nu / (nu - 2)) ** 0.5)
            group2_sd = pm.Deterministic('Group 2 SD', group2_sigma * (nu / (nu - 2)) ** 0.5)

            # Gamma Distribution (beta and alpha are calculated with mu and sigma)
            _ = pm.Gamma("Group 1 data", mu=group1_mean,  sigma=group1_sigma, observed=y1)
            _ = pm.Gamma("Group 2 data", mu=group2_mean,  sigma=group2_sigma, observed=y2)

            diff_of_means = pm.Deterministic('Difference of means', group1_mean - group2_mean)
            _ = pm.Deterministic('Difference of SDs', group1_sd - group2_sd)
            _ = pm.Deterministic(
                'Effect size', diff_of_means / np.sqrt((group1_sd ** 2 + group2_sd ** 2) / 2)
            )
            
            self.trace = pm.sample(tune=tune, draws=draws, target_accept=0.95) #Runs markov-chain monte carlo