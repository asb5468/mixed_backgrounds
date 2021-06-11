'''
Marginalized likelihood adapted from 
https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/gw/likelihood.py#L26
but adds cross-correlation terms on off-diagonal of 
covariance matrix. See https://arxiv.org/abs/1809.02293 for 
derivation in case of diagonal covariance matrix.
Time marginalization is not implemented.

Sylvia Biscoveanu
'''
import numpy as np
import bilby
import pdb
import os
from scipy.misc import logsumexp
from scipy.interpolate import interp1d
from scipy.special import i0e

def logZn(P, s):
	return -1./2*(P)*np.conj(s)*s

def logZ12(S, s1, s2):
	return 1./2*S*np.conj(s1)*s2

def rho_opt_squared(P, h):
	return (P)*np.conj(h)*h

def rho_opt12_squared(S, h1, h2):
	return -S*np.conj(h1)*h2

def kappa_squared(P, s, h):
	return (P)*np.conj(s)*h

def kappa12_squared(S, h1, h2, s1, s2):
	return -S*(np.conj(s2)*h1 + np.conj(s1)*h2)/2.

def detC(P, S12):
	return P**2-S12**2

class MixedBackgroundLikelihoodMarg(bilby.Likelihood):
    
    def __init__(self, interferometers, orf, Sh, waveform_generator, distance_marginalization=False,
                 phase_marginalization=False, priors=None, distance_marginalization_lookup_table=None):
        super(MixedBackgroundLikelihoodMarg, self).__init__(parameters=dict())
        self.phase_marginalization = phase_marginalization
        self.time_marginalization = False
        self.distance_marginalization = distance_marginalization
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self.priors = priors
        self.Sh = Sh
        self.orf = orf
        self.duration = 1./(interferometers[0].frequency_array[1] - interferometers[0].frequency_array[0]) 

        if self.phase_marginalization:
            self._check_prior_is_set(key='phase')
            self._bessel_function_interped = None
            self._setup_phase_marginalization()
            priors['phase'] = float(0)

        if self.distance_marginalization:
            self._lookup_table_filename = None
            self._check_prior_is_set(key='luminosity_distance')
            self._distance_array = np.linspace(
                self.priors['luminosity_distance'].minimum,
                self.priors['luminosity_distance'].maximum, int(1e4))
            self.distance_prior_array = np.array(
                [self.priors['luminosity_distance'].prob(distance)
                 for distance in self._distance_array])
            self._setup_distance_marginalization(
                distance_marginalization_lookup_table)
            priors['luminosity_distance'] = float(self._ref_dist)

    #orf and Sh need to have the same frequency array as both of the interferometers
    def log_likelihood(self):
        #this is not generalized for more than 2 detectors with different PSDs
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(self.parameters)
        h1 = self.interferometers[0].get_detector_response(waveform_polarizations,self.parameters)
        h2 = self.interferometers[1].get_detector_response(waveform_polarizations,self.parameters)
        s1 = self.interferometers[0].frequency_domain_strain
        s2 = self.interferometers[1].frequency_domain_strain
        Sh_true = self.Sh(self.interferometers[0].frequency_array, **self.parameters)
        psd = self.interferometers[0].power_spectral_density_array
        determinant = detC(psd, self.orf*Sh_true)
        const = -np.log((self.duration*np.pi/2)**2*determinant)
        if self.distance_marginalization==True:
            logl = self.distance_marginalized_likelihood(h1, h2, s1, s2, Sh_true, psd, determinant) +\
                                np.sum((const + 4./(self.duration*determinant)*(logZn(psd, s1) + 
				logZn(psd, s2) + 2*logZ12(self.orf*Sh_true, s1, s2)))[self.interferometers[0].frequency_mask])
        elif self.phase_marginalization==True:
            logl = self.phase_marginalized_likelihood(h1, h2, s1, s2, Sh_true, psd, determinant) +\
                                np.sum((const + 4./(self.duration*determinant)*(logZn(psd, s1) + 
				logZn(psd, s2) + 2*logZ12(self.orf*Sh_true, s1, s2)))[self.interferometers[0].frequency_mask])
        else:	
            logl = const + 4./(self.duration*determinant)*(logZn(psd, s1) + logZn(psd, s2) + 2*logZ12(self.orf*Sh_true, s1, s2) +
				kappa_squared(psd, s1, h1) + kappa_squared(psd, s2, h2) +
				2*kappa12_squared(self.orf*Sh_true, h1, h2, s1, s2) -1./2*rho_opt_squared(psd, h1) -
				1./2*rho_opt_squared(psd, h2) - rho_opt12_squared(self.orf*Sh_true, h1, h2))
            logl = np.sum(logl[self.interferometers[0].frequency_mask])
        return np.real(logl)

    def noise_log_likelihood(self):
        print("\n\nNoise\n\n")
        return 1

    def log_likelihood_ratio(self):
        print("\n\nRatio\n\n")
        return 1

    def distance_marginalized_likelihood(self, h1, h2, s1, s2, Sh_true, psd, determinant):
        d_inner_h_ref, h_inner_h_ref = self._setup_rho(h1, h2, s1, s2, Sh_true,
							psd, determinant)
        if self.phase_marginalization:
            d_inner_h_ref = np.abs(d_inner_h_ref)
        else:
            d_inner_h_ref = np.real(d_inner_h_ref)
        return self._interp_dist_margd_loglikelihood(
            d_inner_h_ref, h_inner_h_ref)

    def phase_marginalized_likelihood(self, h1, h2, s1, s2, Sh_true, psd, determinant):
        propto_d = 4./(self.duration*determinant)*(kappa_squared(psd, s1, h1) +
				       kappa_squared(psd, s2, h2) +
				       2*kappa12_squared(self.orf*Sh_true, h1, h2, s1, s2))
        propto_d2 = 4./(self.duration*determinant)*(rho_opt_squared(psd, h1) +
					rho_opt_squared(psd, h2) +
					2*rho_opt12_squared(self.orf*Sh_true, h1, h2))
        propto_d = np.sum(propto_d[self.interferometers[0].frequency_mask])
        propto_d2 = np.sum(propto_d2[self.interferometers[0].frequency_mask])
        d_inner_h = self._bessel_function_interped(abs(propto_d))
        return d_inner_h - np.real(propto_d2)/ 2

    def _setup_rho(self, h1, h2, s1, s2, Sh_true, psd, determinant):
        propto_d = 4./(self.duration*determinant)*(kappa_squared(psd, s1, h1) +
				       kappa_squared(psd, s2, h2) +
				       2*kappa12_squared(self.orf*Sh_true, h1, h2, s1, s2))
        propto_d2 = 4./(self.duration*determinant)*(rho_opt_squared(psd, h1) +
					rho_opt_squared(psd, h2) +
					2*rho_opt12_squared(self.orf*Sh_true, h1, h2))
        propto_d = np.sum(propto_d[self.interferometers[0].frequency_mask])
        propto_d2 = np.sum(propto_d2[self.interferometers[0].frequency_mask])
        optimal_snr_squared_ref = (propto_d2.real *
                                   self.parameters['luminosity_distance'] ** 2 /
                                   self._ref_dist ** 2.)
        d_inner_h_ref = (propto_d * self.parameters['luminosity_distance'] /
                         self._ref_dist)
        return d_inner_h_ref, optimal_snr_squared_ref

    @property
    def _delta_distance(self):
        return self._distance_array[1] - self._distance_array[0]

    @property
    def _ref_dist(self):
        """ Smallest distance contained in priors """
        return self._distance_array[0]

    @property
    def _optimal_snr_squared_ref_array(self):
        """ Optimal filter snr at fiducial distance of ref_dist Mpc """
        return np.logspace(-10, 10, self._dist_margd_loglikelihood_array.shape[0])

    @property
    def _d_inner_h_ref_array(self):
        """ Matched filter snr at fiducial distance of ref_dist Mpc """
        if self.phase_marginalization:
            return np.logspace(-5, 10, self._dist_margd_loglikelihood_array.shape[1])
        else:
            return np.hstack((-np.logspace(3, -3, self._dist_margd_loglikelihood_array.shape[1] / 2),
                              np.logspace(-3, 10, self._dist_margd_loglikelihood_array.shape[1] / 2)))

    def _check_prior_is_set(self, key):
        if key not in self.priors or not isinstance(
                self.priors[key], bilby.core.prior.Prior):
            bilby.core.utils.logger.warning(
                'Prior not provided for {}, using the BBH default.'.format(key))
            if key == 'geocent_time':
                self.priors[key] = Uniform(
                    self.interferometers.start_time,
                    self.interferometers.start_time + self.interferometers.duration)
            else:
                self.priors[key] = BBHPriorDict()[key]

    def _setup_distance_marginalization(self, lookup_table=None):
        if isinstance(lookup_table, str) or lookup_table is None:
            self.cached_lookup_table_filename = lookup_table
            lookup_table = self.load_lookup_table(
                self.cached_lookup_table_filename)
        if isinstance(lookup_table, dict):
            if self._test_cached_lookup_table(lookup_table):
                self._dist_margd_loglikelihood_array = lookup_table[
                    'lookup_table']
            else:
                self._create_lookup_table()
        else:
            self._create_lookup_table()
        self._interp_dist_margd_loglikelihood = bilby.core.utils.UnsortedInterp2d(
            self._d_inner_h_ref_array, self._optimal_snr_squared_ref_array,
            self._dist_margd_loglikelihood_array)

    def _setup_phase_marginalization(self):
        self._bessel_function_interped = interp1d(
            np.logspace(-5, 10, int(1e6)), np.logspace(-5, 10, int(1e6)) +
            np.log([i0e(snr) for snr in np.logspace(-5, 10, int(1e6))]),
            bounds_error=False, fill_value=(0, np.nan))

    @property
    def cached_lookup_table_filename(self):
        if self._lookup_table_filename is None:
            dmin = self._distance_array[0]
            dmax = self._distance_array[-1]
            n = len(self._distance_array)
            self._lookup_table_filename = (
                '.distance_marginalization_lookup.npz'
                .format(dmin, dmax, n))
        return self._lookup_table_filename

    @cached_lookup_table_filename.setter
    def cached_lookup_table_filename(self, filename):
        if isinstance(filename, str):
            if filename[-4:] != '.npz':
                filename += '.npz'
        self._lookup_table_filename = filename

    def load_lookup_table(self, filename):
        if os.path.exists(filename):
            loaded_file = dict(np.load(filename))
            match, failure = self._test_cached_lookup_table(loaded_file)
            if match:
                bilby.core.utils.logger.info('Loaded distance marginalisation lookup table from '
                            '{}.'.format(filename))
                return loaded_file
            else:
                bilby.core.utils.logger.info('Loaded distance marginalisation lookup table does '
                            'not match for {}.'.format(failure))
                return None
        elif isinstance(filename, str):
            bilby.core.utils.logger.info('Distance marginalisation file {} does not '
                        'exist'.format(filename))
            return None
        else:
            return None

    def cache_lookup_table(self):
        np.savez(self.cached_lookup_table_filename,
                 distance_array=self._distance_array,
                 prior_array=self.distance_prior_array,
                 lookup_table=self._dist_margd_loglikelihood_array,
                 reference_distance=self._ref_dist,
                 phase_marginalization=self.phase_marginalization)

    def _test_cached_lookup_table(self, loaded_file):
        pairs = dict(
            distance_array=self._distance_array,
            prior_array=self.distance_prior_array,
            reference_distance=self._ref_dist,
            phase_marginalization=self.phase_marginalization)
        for key in pairs:
            if key not in loaded_file:
                return False, key
            elif not np.array_equal(np.atleast_1d(loaded_file[key]),
                                    np.atleast_1d(pairs[key])):
                return False, key
        return True, None

    def _create_lookup_table(self):
        """ Make the lookup table """
        bilby.core.utils.logger.info('Building lookup table for distance marginalisation.')

        self._dist_margd_loglikelihood_array = np.zeros((600, 800))
        for ii, optimal_snr_squared_ref in enumerate(self._optimal_snr_squared_ref_array):
            for jj, d_inner_h_ref in enumerate(self._d_inner_h_ref_array):
                optimal_snr_squared_array = (
                    optimal_snr_squared_ref * self._ref_dist ** 2. /
                    self._distance_array ** 2)
                d_inner_h_array = (
                    d_inner_h_ref * self._ref_dist / self._distance_array)
                if self.phase_marginalization:
                    d_inner_h_array =\
                        self._bessel_function_interped(abs(d_inner_h_array))
                self._dist_margd_loglikelihood_array[ii][jj] = \
                    logsumexp(d_inner_h_array - optimal_snr_squared_array / 2,
                              b=self.distance_prior_array * self._delta_distance)
        log_norm = logsumexp(0. / self._distance_array,
                             b=self.distance_prior_array * self._delta_distance)
        self._dist_margd_loglikelihood_array -= log_norm
        self.cache_lookup_table()

