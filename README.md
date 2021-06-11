## Mixed Backgrounds
This repository provides code for measuring a Gaussian gravitational-wave background in the presence of an astrophysical foreground, as described in https://arxiv.org/abs/2009.04418.

### BBH Parameter Estimation
The code is broken down into three stages. The first stage corresponds to performing standard compact binary parameter estimation on simulated data which includes a Gaussian background characterized by
amplitude $\Omega$ and power-law index $\alpha$ and potentially a binary black hole (BBH) merger. This stage can be run with:
```
python mixed_backgrounds_marg.py segment_number
```
where `segment_number` is an integer representing the segment of data that will be simulated and analyzed. The simulated data is a 4 second strain timeseries sampled at a frequency of 2048 Hz. 
This step should be run for each simulated data segment. The first 10 such segments will have a BBH signal added to them. In our original paper, we used 101 simulated data segments.
This step saves the simualted data so that it can be loaded for the second reweighting step, described below. The file paths specified for some of the loaded data products and for the outputs 
(`outdir`) may need to be modified. The output of this step will be three sets of [bilby](https://git.ligo.org/lscsoft/bilby)
result files, one for each of the following hypotheses:
1. "Noise model": the segment contains only a Gaussian background, no BBH
2. "CBC model": the segment contains only a BBH, no Gaussian background
3. "Signal model": the segment constains both a Gaussian background and a BBH

The "Signal model" results are not used later in the pipeline, but they are interesting to have to be able to compare with the reweighted results, described below.

### Reweighting
The second step is the likelihood reweighting step used to generate gridded likelihoods for $\Omega$ and $\alpha$. This step can be run with:
```
python reweighting_test.py segment_number
```
where the `segment_number` corresponds to the same simulated segment of data used in the first step. Here the posterior samples obtained for the 15 BBH parameters in 
the first step (masses, spins, location, etc.) are marginalzied over and reweighted to a likelihoods used for the "Noise model" and "Signal model" described above, but evaluated
on 50x50 grid. The file paths specified for the outputs (`outdir`) should be modified to match what was used in the first step.
The output of this step is two `npy` files containing the likelihood grids in $\log{\Omega}$ and $\alpha$ for each segment.

### Combining segments
The final step is combining the individual likelihood grids to obtain a global posterior for $\log{\Omega}$, $\alpha$, and $\xi$, the fraction of segments containing a signal
from across all the simulated data segments. This is done by running `python combine_3d_reweighted_grids.py`. The output of this step is a combined 3D grid of the likelihoods for
the three parameters given above and a corner plot.

### Other files
* `stoch_utils.py`: contains helper functions for simulating the correlated Gaussian background data 
* `marg_likelihood.py`: contains the analytically marginalized likelihood over the BBH parameters phase and distance, including off-diagonal cross-correlation terms in the covariance matrix. 
* `bbh.prior`: the `bilby` prior file for the binary black hole parameters, used both for generating injections and during parameter estimation
* `analytical_orf.dat`: The overlap reduction function for the Hanford-Livingston detector baseline
* `full_auto_power.dat`: $P(f) + S_{h}(f)$ pre-calculated (calculation commented out in `mixed_backgrounds_marg.py`)
