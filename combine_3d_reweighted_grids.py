import numpy as np
from scipy.special import logsumexp
from scipy.interpolate import interp1d
import matplotlib as mpl
mpl.rcParams['font.size']=20
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import bilby
import pdb
def calc_credible_interval_90(x, fx):
        sfx = fx.copy()
        sfx.sort()
        sfx[:] = sfx[::-1]
        dx = x[1]-x[0]
        cdf = np.cumsum(sfx)*dx
        cdf /= max(cdf)
        nearest = np.argmin(abs(cdf-0.9))
        x_90 = x[np.in1d(fx, sfx[:nearest])]
        lower = min(x_90)
        upper = max(x_90)
        maxP = x[np.argmax(fx)]
        return lower, upper, maxP

def calc_L90_L50(dx, dy, fx):
	fx_flat = np.ravel(fx).copy()
	fx_flat.sort()
	fx_flat[:] = fx_flat[::-1]
	cdf = np.cumsum(fx_flat)*dx*dy
	cdf /= max(cdf)
	nearest_90 = np.argmin(abs(cdf-0.9))
	nearest_50 = np.argmin(abs(cdf-0.5))
	l90 = fx_flat[nearest_90]
	l50 = fx_flat[nearest_50]
	lmax = fx_flat[0]
	return lmax, l50, l90

signal_logL = np.zeros((50,50))
noise_logL = np.zeros((50,50))
lnZ_noise = []
lnZ_cbc = []
full_logL = np.zeros((100,50,50))
xis = np.linspace(0,1,100)
dxi = xis[1]-xis[0]

for i in range(101):
	print(i)
	signal_data = np.load('outdir/cosmoCBC_cc_H1L1_signal3_'+str(i)+'_grid_likelihood.npy')
	noise_data = np.load('outdir/cosmoCBC_cc_H1L1_noise3_'+str(i)+'_grid_likelihood.npy')
	cbc_result = bilby.core.result.read_in_result('outdir/cosmoCBC_cc_H1L1_cbc3_'+str(i)+'_result.json')
	#signal_result = bilby.core.result.read_in_result('outdir/cosmoCBC_cc_H1L1_signal3_'+str(i)+'_result.json')
	noise_result = bilby.core.result.read_in_result('outdir/cosmoCBC_cc_H1L1_noise3_'+str(i)+'_result.json')
	signal_data = np.nan_to_num(signal_data, nan=-np.inf)
	noise_data = np.nan_to_num(noise_data, nan=-np.inf)
	invBF = np.exp(noise_data - signal_data)
	for j, xi in enumerate(xis):
		full_logL[j] += signal_data + np.log(xi*(1-invBF) + invBF) 		
	signal_logL += signal_data
	noise_logL += noise_data
	lnZ_noise.append(noise_result.log_noise_evidence)
	lnZ_cbc.append(cbc_result.log_evidence)

full_logL = np.nan_to_num(full_logL, nan=-np.inf)
np.save('combined_logL_3d.npy', full_logL)

# calculate total evidences
xs = np.linspace(-8, -4, 50)
dx = xs[1]-xs[0]
ys = np.linspace(0, 4., 50)
dy = ys[1]-ys[0]
arr_xi_Omg = np.meshgrid(xis, xs)
arr_xi_alpha = np.meshgrid(xis, ys)
arr_Omg_alpha = np.meshgrid(xs, ys)
invBF_cbc = np.exp(np.subtract(lnZ_noise, lnZ_cbc))
logL_tbs = []
for xi in xis:
	logL_tbs.append(np.sum(lnZ_cbc + np.log(xi*(1-invBF_cbc) + invBF_cbc)))
lnZ_tbs = logsumexp(logL_tbs) + np.log(dxi/1)
full_logZ = logsumexp(full_logL) + np.log(dxi*dx*dy/16)
#log_signal_ev = np.log(dx*dy/16) + logsumexp(signal_logL)
#log_noise_ev = np.log(dx*dy/16) + logsumexp(noise_logL)
print('Stochastic lnBF {}'.format(full_logZ - lnZ_tbs))
pdb.set_trace()
full_logL = np.load('combined_logL_3d.npy')

# calculate marginalized likelihoods
logL_xi_Omg = logsumexp(full_logL, axis=1)
logL_xi_alpha = logsumexp(full_logL, axis=2)
logL_Omg_alpha = logsumexp(full_logL, axis=0) 
logL_Omg = logsumexp(logL_Omg_alpha, axis=0)
logL_alpha = logsumexp(logL_Omg_alpha, axis=1)
logL_xi = logsumexp(logL_xi_Omg, axis=1)	

# normalize for plotting
regularizer = np.max(logL_Omg)
logL_Omg = np.subtract(logL_Omg, regularizer)
regularizer = np.max(logL_alpha)
logL_alpha = np.subtract(logL_alpha, regularizer)
regularizer = np.max(logL_xi)
logL_xi = np.subtract(logL_xi, regularizer)
regularizer = np.max(logL_Omg_alpha)
logL_Omg_alpha = np.subtract(logL_Omg_alpha, regularizer)
regularizer = np.max(logL_xi_alpha)
logL_xi_alpha = np.subtract(logL_xi_alpha, regularizer)
regularizer = np.max(logL_xi_Omg)
logL_xi_Omg = np.subtract(logL_xi_Omg, regularizer)

# calculate 1d credible intervals
lower_Omg, upper_Omg, maxP_Omg = calc_credible_interval_90(xs, np.exp(logL_Omg))
lower_alpha, upper_alpha, maxP_alpha = calc_credible_interval_90(ys, np.exp(logL_alpha))
lower_xi, upper_xi, maxP_xi = calc_credible_interval_90(xis, np.exp(logL_xi))
print(lower_Omg-maxP_Omg, upper_Omg-maxP_Omg, maxP_Omg)
print(lower_alpha-maxP_alpha, upper_alpha-maxP_alpha, maxP_alpha)
print(lower_xi-maxP_xi, upper_xi-maxP_xi, maxP_xi)
interped_lOmg = interp1d(xs, np.exp(logL_Omg), kind='cubic')

# calculate 2d credible levels
lmax_Omg_alpha, l50_Omg_alpha, l90_Omg_alpha = calc_L90_L50(dx, dy, np.exp(logL_Omg_alpha))
lmax_xi_Omg, l50_xi_Omg, l90_xi_Omg = calc_L90_L50(dxi, dx, np.exp(logL_xi_Omg))
lmax_xi_alpha, l50_xi_alpha, l90_xi_alpha = calc_L90_L50(dxi, dy, np.exp(logL_xi_alpha))

# plot results
fig = plt.figure(figsize=(24,24))
ax1 = fig.add_subplot(331)
ax1.plot(np.linspace(-8,-4, 500), interped_lOmg(np.linspace(-8,-4, 500)), lw=3, color='cornflowerblue')
ax1.axvline(-6, color='orange', lw=3)
ax1.set_xlim(-6.5, -5.5)
ax1.set_ylim(0,)
ax1.tick_params(axis='both', which='major', labelsize=40)
ax1.axes.xaxis.set_ticklabels([])
ax1.axes.yaxis.set_ticklabels([])

ax2 = fig.add_subplot(334)
ax2.contourf(arr_Omg_alpha[0], arr_Omg_alpha[1], np.exp(logL_Omg_alpha), levels=[l90_Omg_alpha, l50_Omg_alpha, lmax_Omg_alpha], colors=['#CBDBF9', '#6495ED'])
ax2.axvline(-6, color='orange', lw=3)
ax2.axhline(0, color='orange', lw=3)
ax2.scatter(-6, 0, c='orange', s=400, marker='s')
ax2.set_xlim(-6.5, -5.5)
ax2.set_ylim(0,3)
ax2.set_yticks([0,1,2,3])
ax2.tick_params(axis='both', which='major', labelsize=40)
ax2.axes.xaxis.set_ticklabels([])

ax3 = fig.add_subplot(335)
ax3.plot(ys, np.exp(logL_alpha), lw=3, color='cornflowerblue')
ax3.axvline(0, color='orange', lw=3)
ax3.set_xlim(0,3)
ax3.set_ylim(0,)
ax3.tick_params(axis='both', which='major', labelsize=40)
ax3.axes.yaxis.set_ticklabels([])
ax3.axes.xaxis.set_ticklabels([])

ax4 = fig.add_subplot(337)
ax4.contourf(arr_xi_Omg[1], arr_xi_Omg[0], np.exp(logL_xi_Omg.T), levels=[l90_xi_Omg, l50_xi_Omg, lmax_xi_Omg], colors=['#CBDBF9', '#6495ED'])
ax4.axvline(-6, color='orange', lw=3)
ax4.axhline(11./101, color='orange', lw=3)
ax4.scatter(-6, 11./101, c='orange', s=400, marker='s')
ax4.set_xlabel(r'$\log{\Omega_{\alpha}}$', fontsize=60)
ax4.set_xlim(-6.5, -5.5)
ax4.set_ylim(0, 0.3)
ax4.set_yticks([0,0.1,0.2,0.3])
ax4.tick_params(axis='both', which='major', labelsize=40)

ax5 = fig.add_subplot(338)
ax5.contourf(arr_xi_alpha[1], arr_xi_alpha[0], np.exp(logL_xi_alpha.T), levels=[l90_xi_alpha, l50_xi_alpha, lmax_xi_alpha], colors=['#CBDBF9', '#6495ED'])
ax5.axvline(0, color='orange', lw=3)
ax5.axhline(11./101, color='orange', lw=3)
ax5.scatter(0,11./101, c='orange', s=400, marker='s')
ax5.set_xlabel(r'$\alpha$', fontsize=60)
ax5.set_ylim(0, 0.3)
ax5.set_xlim(0,3)
ax5.set_yticks([0,0.1,0.2,0.3])
ax5.tick_params(axis='both', which='major', labelsize=40)
ax5.axes.yaxis.set_ticklabels([])

ax6 = fig.add_subplot(339)
ax6.plot(xis, np.exp(logL_xi), lw=3, color='cornflowerblue')
ax6.axvline(11./101, color='orange', lw=3)
ax6.set_xlabel(r'$\xi$', fontsize=60)
ax6.set_xlim(0,0.3)
ax6.set_ylim(0,)
ax6.tick_params(axis='both', which='major', labelsize=40)
ax6.axes.yaxis.set_ticklabels([])
plt.tight_layout()
plt.savefig('combined_likelihood_grid_3d.png')
