from __future__ import division
import bilby
import sys
import time
import numpy as np
import inspect
import matplotlib.pyplot as plt
from stoch_utils import *
import pdb
import lalsimulation as lalsim
#sys.path.append('/home/sbiscove/Peyote_project/bilby_fork/examples/mine/mixed-backgrounds/')
from marg_likelihood import *
from mixed_background_marg import GaussianBackgroundLikelihood, generate_starting_params
from scipy.special import logsumexp

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
   
    #Generate a set of injection parameters randomly drawn from the default prior
    #Use same seed to get same geocent_time as first step
    binary = generate_starting_params()
    binary['geocent_time'] = 1212300415+4*run_number
    waveform_arguments = dict(waveform_approximant = 'IMRPhenomPv2', reference_frequency = 25.)

    #Setup the CBC waveform generator and get the injected freq domain strain
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(duration=time_duration, sampling_frequency=sampling_freq,\
                                                frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,
                                                parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                                                parameters = binary, waveform_arguments = waveform_arguments)
   
    #Load in the data simulated during the first step 
    IFOs = [bilby.gw.detector.get_empty_interferometer(name) for name in ['H1', 'L1']]
    for ifo in IFOs:
        freqs, real, imag = np.loadtxt(outdir+'/'+ifo.name+'_'+label+'_frequency_domain_data.dat', unpack=True)
        strain = real + 1j*imag
        ifo.set_strain_data_from_frequency_domain_strain(strain, frequency_array=freqs, start_time=(binary['geocent_time']-2))
        ifo.strain_data.start_time = binary['geocent_time']-2 
   
    #Calculate the true stochastic signal 
    logOmg_true = -6.
    alpha_true = 0.
    Sh_true = Sh(IFOs[0].frequency_array, logOmg_true, alpha_true)
 
    orf_H1L1 = np.loadtxt('analytical_orf.dat',usecols=(0,1))
    orf_H1L1 = np.interp(IFOs[0].frequency_array,orf_H1L1.T[0],orf_H1L1.T[1])
    orf_H1H2 = 1 #set to coincident and coaligned for now, need to run on two H1s
    stoch_SNR = SNR(Sh_true, orf_H1L1, IFOs[0].power_spectral_density_array, 404., 0.25)
    print("Stochastic SNR: {}".format(stoch_SNR))
    # make the PSD actually the full auto-power
    for ifo in IFOs:
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file='full_auto_power.dat')
    
    #setup prior
    prior = dict()
    prior['logOmega'] = bilby.core.prior.Uniform(-8,-4,name='logOmega',latex_label=r'$\log{\Omega}$')
    prior['alpha'] = bilby.core.prior.Uniform(0,4,name='alpha', latex_label=r'$\alpha$')
    t0 = binary['geocent_time']
    bbh_prior = bilby.core.prior.PriorDict(filename='bbh.prior')
    bbh_prior['geocent_time'] = bilby.core.prior.Uniform(t0-0.1, t0+0.1, name='geocent_time')
    prior = dict(bbh_prior,**prior)
    
    #setup likelihoods
    noise_likelihood = GaussianBackgroundLikelihood(IFOs,orf_H1L1,Sh)
    cbc_likelihood = MixedBackgroundLikelihoodMarg(IFOs, orf_H1L1, Sh0, waveform_generator, 
                          distance_marginalization=True, phase_marginalization=True, priors=bbh_prior)
    signal_likelihood_marg = MixedBackgroundLikelihoodMarg(IFOs, orf_H1L1, Sh, waveform_generator, 
				distance_marginalization=True, phase_marginalization=True, priors=prior)
    
    #load previous result and setup reweighting
    cbc_result = bilby.core.result.read_in_result(outdir+'/'+cbc_label+'_result.json')
    #grid of logOmegas
    xs = np.linspace(-8, -4, 50)
    dx = xs[1]-xs[0]
    #grid of alphas
    ys = np.linspace(0, 4., 50)
    dy = ys[1]-ys[0]
    arr = np.meshgrid(xs, ys)
    positions = np.column_stack([arr[0].ravel(), arr[1].ravel()])
    n = min(len(cbc_result.posterior), 3000)
    post = cbc_result.posterior.sample(n)
    efficiency = []
    noise_likelihood_vals = []
    signal_likelihood_vals = []
    t0 = time.time()
    for j, pair in enumerate(positions):
        print(float(j/len(positions)))
        stoch_params = {'logOmega': pair[0], 'alpha':pair[1]}
        noise_likelihood.parameters = stoch_params
        noise_likelihood_vals.append(noise_likelihood.log_likelihood())
        log_weights = []
        for index, row in post.iterrows():
            params = row.to_dict()
            params.update(stoch_params)
            params.update({'luminosity_distance':50, 'phase':0})
            signal_likelihood_marg.parameters = params
            logL = signal_likelihood_marg.log_likelihood()
            log_weights.append(logL - params['log_likelihood'])
        regularizer = max(log_weights)
        normed_weights = log_weights - regularizer
        n_eff = np.sum(np.exp(normed_weights))**2/np.sum(np.power(np.exp(normed_weights),2))
        efficiency.append(n_eff/n)
        signal_likelihood_vals.append(cbc_result.log_evidence + logsumexp(log_weights) - np.log(len(log_weights)))
    signal_likelihood_vals = np.array(signal_likelihood_vals).reshape((50,50))
    noise_likelihood_vals = np.array(noise_likelihood_vals).reshape((50,50))
    np.save(outdir+'/'+signal_label+'_grid_likelihood.npy', signal_likelihood_vals)
    np.save(outdir+'/'+noise_label+'_grid_likelihood.npy', noise_likelihood_vals)
    t1 = time.time()
    print('Time: {} s'.format(t1-t0))
    print(efficiency)
    print("Done")

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("\n Input the run number")
    else:
        main(int(sys.argv[1]))
