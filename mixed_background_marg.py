from __future__ import division
import bilby
import sys
import numpy as np
import matplotlib.pyplot as plt
from stoch_utils import *
import pdb
import lalsimulation as lalsim
#sys.path.append('/home/sbiscove/Peyote_project/bilby_fork/examples/mine/mixed-backgrounds/')
from marg_likelihood import *

class MixedBackgroundLikelihood(bilby.Likelihood):
    
    def __init__(self, interferometers, orf, Sh, waveform_generator):
        super(MixedBackgroundLikelihood, self).__init__(parameters=dict())
        self.phase_marginalization = False
        self.distance_marginalization = False
        self.time_marginalization = False
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self.Sh = Sh
        self.orf = orf
        self.duration = 1./(interferometers[0].frequency_array[1] - interferometers[0].frequency_array[0]) 

    #orf and Sh need to have the same frequency array as both of the interferometers
    def log_likelihood(self):
        #this is not generalized for more than 2 detectors with different PSDs
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(self.parameters)
        res = [ifo.frequency_domain_strain - ifo.get_detector_response(waveform_polarizations,self.parameters) for ifo in self.interferometers]
        Sh_true = self.Sh(self.interferometers[0].frequency_array, **self.parameters)
        cov = np.array([[self.interferometers[0].power_spectral_density_array, self.orf*Sh_true], [self.orf*Sh_true, self.interferometers[1].power_spectral_density_array]])
        cov = np.swapaxes(cov, 0, 2)
        detC = np.linalg.det(cov)
        ap_term = (np.abs(res[0])**2+np.abs(res[1])**2)
        cp_term = (np.conj(res[0])*res[1]+np.conj(res[1])*res[0])
        logl = np.sum((-np.log((self.duration*np.pi/2)**2*detC) - 2./(self.duration*detC)*(ap_term*(self.interferometers[0].power_spectral_density_array)\
                    -self.orf*Sh_true*cp_term))[self.interferometers[0].frequency_mask])
        return np.real(logl)

    def noise_log_likelihood(self):
        print("\n\nNoise\n\n")
        return 1

    def log_likelihood_ratio(self):
        print("\n\nRatio\n\n")
        return 1

class GaussianBackgroundLikelihood(bilby.Likelihood):
    def __init__(self, interferometers, orf, Sh):
        super(GaussianBackgroundLikelihood, self).__init__(parameters=dict())
        self.interferometers = interferometers
        self.Sh = Sh
        self.orf = orf
        self.duration = 1./(interferometers[0].frequency_array[1] - interferometers[0].frequency_array[0])
    
    #orf and Sh need to have the same frequency array as both of the interferometers
    def log_likelihood(self):
        #this is not generalized for more than 2 detectors with different PSDs
        strain = [ifo.frequency_domain_strain for ifo in self.interferometers]
        Sh_true = self.Sh(self.interferometers[0].frequency_array, **self.parameters)
        cov = np.array([[self.interferometers[0].power_spectral_density_array, self.orf*Sh_true], [self.orf*Sh_true, self.interferometers[1].power_spectral_density_array]])
        cov = np.swapaxes(cov, 0, 2)
        detC = np.linalg.det(cov)
        ap_term = (np.abs(strain[0])**2+np.abs(strain[1])**2)
        cp_term = (np.conj(strain[0])*strain[1]+np.conj(strain[1])*strain[0])
        logl = np.sum((-np.log((self.duration*np.pi/2)**2*detC) - 2./(self.duration*detC)*(ap_term*(self.interferometers[0].power_spectral_density_array)\
                    -self.orf*Sh_true*cp_term))[self.interferometers[0].frequency_mask])
        return np.real(logl)

    def noise_log_likelihood(self):
        strain = [ifo.frequency_domain_strain for ifo in self.interferometers]
        cov = np.array([[self.interferometers[0].power_spectral_density_array, np.zeros(len(strain[0]))], [np.zeros(len(strain[0])), self.interferometers[1].power_spectral_density_array]])
        cov = np.swapaxes(cov, 0, 2)
        detC = np.linalg.det(cov)
        ap_term = (np.abs(strain[0])**2+np.abs(strain[1])**2)
        logl = np.sum((-np.log((self.duration*np.pi/2)**2*detC) - 2./(self.duration*detC)*ap_term*self.interferometers[0].power_spectral_density_array)[self.interferometers[0].frequency_mask])
        return np.real(logl)

    def log_likelihood_ratio(self):
        print("\n\nRatio\n\n")
        return 1
       
def generate_starting_params():
    prior = bilby.gw.prior.PriorDict(filename="bbh.prior")
    binary = prior.sample()
    return binary

def main(run_number):

    # A few simple setup steps
    time_duration = 4.
    sampling_freq = 2048.
    outdir = 'outdir'
    label = 'cosmoCBC_cc_H1L1_take3_'+str(run_number)
    signal_label = 'cosmoCBC_cc_H1L1_signal3_'+str(run_number)
    noise_label = 'cosmoCBC_cc_H1L1_noise3_'+str(run_number)
    cbc_label = 'cosmoCBC_cc_H1L1_cbc3_'+str(run_number)
    np.random.seed(86+run_number)
    bilby.core.utils.setup_logger(outdir=outdir, label=label)
   
    # Generate a set of injection parameters randomly drawn from the default prior
    binary = generate_starting_params()
    binary['geocent_time'] = 1212300415+4*run_number
    waveform_arguments = dict(waveform_approximant = 'IMRPhenomPv2', reference_frequency = 25.)

    # Setup the CBC waveform generator and get the injected freq domain strain
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(duration=time_duration, sampling_frequency=sampling_freq,\
                                                frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,
                                                parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                                                parameters = binary, waveform_arguments = waveform_arguments)
    hf_cbc_strain = waveform_generator.frequency_domain_strain()
     
    # Setup the interferometers and add the stochastic signal
    if (run_number<=10):
            IFOs = [bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(name, injection_polarizations = hf_cbc_strain, injection_parameters = binary,\
                duration=time_duration, sampling_frequency=sampling_freq, outdir=outdir) for name in ['H1', 'L1']]
        
    else:
        # No signal
        IFOs = [bilby.gw.detector.get_empty_interferometer(name) for name in ['H1', 'L1']]
        for ifo in IFOs:
            ifo.power_spectral_density.from_aligo()
            ifo.set_strain_data_from_power_spectral_density(sampling_frequency=sampling_freq, duration=time_duration, start_time=(binary['geocent_time']+2)-4)
   
    # Calculate true stochastic signal for injection 
    logOmg_true = -6.
    alpha_true = 0.
    Sh_true = Sh(IFOs[0].frequency_array, logOmg_true, alpha_true)
    # This is how the PSD file in the repo was written
    #np.savetxt('full_auto_power.dat', np.array([IFOs[0].frequency_array, IFOs[0].power_spectral_density_array + Sh_true]).T)
 
    orf_H1L1 = np.loadtxt('analytical_orf.dat',usecols=(0,1))
    orf_H1L1 = np.interp(IFOs[0].frequency_array,orf_H1L1.T[0],orf_H1L1.T[1])
    orf_H1H2 = 1 # coincident and coaligned option, not used
    stoch_SNR = SNR(Sh_true, orf_H1L1, IFOs[0].power_spectral_density_array, 404., 0.25)
    print("Stochastic SNR: {}".format(stoch_SNR))

    # Make the PSD actually the full auto-power
    for ifo in IFOs:
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file='full_auto_power.dat')
   
    # Create and add the correlated signal 
    hf_stoch1, hf_stoch2 = create_orf_convolution_signal(orf_H1L1, sampling_freq, time_duration, Sh_true)
    IFOs[0].strain_data.frequency_domain_strain += hf_stoch1*IFOs[0].frequency_mask
    IFOs[1].strain_data.frequency_domain_strain += hf_stoch2*IFOs[1].frequency_mask

    # Save the data for the reweighting step
    IFOs[0].save_data(outdir, label)  
    IFOs[1].save_data(outdir, label)
    
    # Setup the noise run prior, likelihood, and run sampler
    params = dict()
    params['logOmega'] = logOmg_true
    params['alpha'] = alpha_true
    prior = dict()
    prior['logOmega'] = bilby.core.prior.Uniform(-8,-4,name='logOmega',latex_label=r'$\log{\Omega}$')
    prior['alpha'] = bilby.core.prior.Uniform(0,4,name='alpha', latex_label=r'$\alpha$')
    
    noise_likelihood = GaussianBackgroundLikelihood(IFOs,orf_H1L1,Sh)
    noise_result = bilby.core.sampler.run_sampler(likelihood=noise_likelihood,priors=prior,sampler='cpnest', nlive=1000,
                                            injection_parameters=params, use_ratio = False, label=noise_label, outdir=outdir, resume=True)
    noise_result.plot_corner(bins=25)
    print(noise_result)
    
    # Setup signal likelihood, prior, and run sampler
    params = dict(binary,**params)
    t0 = params['geocent_time']
    bbh_prior = bilby.core.prior.PriorDict(filename='bbh.prior')
    bbh_prior['geocent_time'] = bilby.core.prior.Uniform(t0-0.1, t0+0.1, name='geocent_time')
    prior = dict(bbh_prior,**prior)
   
    '''
    This block of code is optional, but nice to run for
    comparison with the reweighted results and for calculating
    bayes factors for individual segments. The signal_likelihood
    is not used but just initialized for comparison with the 
    marginalized likelihood.
    ''' 
    #signal_likelihood = MixedBackgroundLikelihood(IFOs,orf_H1L1,Sh,waveform_generator)
    signal_likelihood_marg = MixedBackgroundLikelihoodMarg(IFOs, orf_H1L1, Sh, waveform_generator, 
				distance_marginalization=True, phase_marginalization=True, priors=prior)
    signal_result = bilby.core.sampler.run_sampler(likelihood=signal_likelihood_marg, priors=prior, sampler='cpnest', nlive=1000,
                                injection_parameters=params, use_ratio = False, label=signal_label, outdir=outdir, resume=True)
    signal_result.plot_corner(bins=25)
    print(signal_result)

    # Setup CBC likelihood and run sampler
    cbc_likelihood = MixedBackgroundLikelihoodMarg(IFOs, orf_H1L1, Sh0, waveform_generator,
                        distance_marginalization=True, phase_marginalization=True, priors=bbh_prior)
    cbc_result = bilby.core.sampler.run_sampler(likelihood=cbc_likelihood, priors=bbh_prior, sampler='cpnest', nlive=1000,
				injection_parameters=params, use_ratio = False, label=cbc_label, outdir=outdir, resume=True)
    print(cbc_result)
    print("logBF_CBC: {}".format(signal_result.log_evidence-noise_result.log_evidence))
    print("logBF_stoch: {}".format(signal_result.log_evidence-cbc_result.log_evidence))
    print("Done")

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("\n Input the run number")
    else:
        main(int(sys.argv[1]))
