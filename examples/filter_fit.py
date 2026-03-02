# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:24:16 2023

@author: paclk
"""
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

from pathlib import Path 

def inv(k):
    return 1.0 / k


# dir = 'C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/'
# file = 'diagnostics_3d_ts_21600.nc'

# dirroot = 'C:/Users/paclk/OneDrive - University of Reading/'
# dirroot = 'C:/Users/xm904103/OneDrive - University of Reading/'
dirroot = 'F:/Data/'

test_case =0
if test_case == 0:
    reflab = 'Ref 50 m'
    config_file = 'config_test_case_0.yaml'
    indir = dirroot + 'ug_project_data/test_filter_spectra_RFFT/'
    outdir = dirroot + 'ug_project_data/test_filter_spectra_RFFT/'
    fileroot = 'diagnostics_3d_ts_*'
    ref_file = None # 'diagnostics_ts_21600'
    fname = ''
    outtag = 'w_spec' # '_spectra_w_2D'
    plot_height = 250
    max_e_range = (0, 2)
    ylims=[0.00001,0.01]
    xlims2 = [0.001,0.01]
    del_fs = [  4.5 ]
    alphas = [3.6, 4.0, 4.4]

elif test_case == 1:
    reflab = 'Ref 5 m'
    config_file = 'config_test_case_1.yaml'
    indir = dirroot + 'CBL/test_filter_spectra_RFFT/'
    outdir = dirroot + 'CBL/test_filter_spectra_RFFT/'
    fileroot = 'diagnostics_3d_ts_*'
    ref_file = None
    fname = '' # '_filter_spectra'
    outtag = 'w_spec' # '_spectra_w_2D'
    plot_height = 500  
    max_e_range = (40, 60)
    ylims = [0.00001,0.1]
    xlims2 = [0.01,0.1]
#    sigmas = [ 2.7, 2.8, 2.9]
 #   alphas = [2.4, 2.5, 2.6, 3.5]
    del_fs = [ 3.4, 3.5, 3.6 ]
    alphas = [ 2., 2.4, 2.8]

files = indir + fileroot + fname + outtag+'.nc'


# Set up outfile
#outdir = os.path.join(dir, 'spectra/')
#os.makedirs(outdir, exist_ok = True)  # make outdir if it doesn't already exist
#outfile = os.path.join(outdir,('.').join(os.path.basename(file).split('.')[:-1]) + "_"+outtag+".nc")

#files = Path(indir).glob(fileroot + fname + outtag+'.nc')

#dso = xr.open_dataset(file)
dso = xr.open_mfdataset(files)

dso = dso.rename({'z':'z_w', 'zn':'z_p'})
print(dso)

dx =  dso.attrs['dx']
dy = dso.attrs['dy']

#%% Get time meaned spectrum

k_ref = dso['hfreq']
k = k_ref.copy()

k_angular = dso['hwaven']

kE_k = k_ref * dso['spec_2d_w']
kE_kp_ref = kE_k.mean(dim='time').sel(z_w=plot_height, method='nearest')

#%% Plot Energy Density Spectrum

fig1, axa = plt.subplots(3,1,figsize=(8,12))

kE_kp_ref.plot(xscale='log', yscale='log', label=reflab, ax=axa[0])
#plt.xlim([0.0001,0.1])
#axa[0].set_ylabel('kE(k)')

secax = axa[0].secondary_xaxis('top', functions=(inv, inv))
secax.set_xlabel('wavelength (m)')

xrn = axa[0].get_xlim()
yrn = axa[0].get_ylim()

#%% Find maximum in spectrum

max_energy_index = kE_kp_ref.values.argmax()

if max_energy_index.size > 1:
    max_energy_index = max_energy_index[0]

x_ymax = k_ref.values[max_energy_index]
#ymax = kE_kp_ref[max_energy_index]

k53_Ek = kE_kp_ref * (k_ref.values/ x_ymax)**(2/3)

ymax = k53_Ek.isel(hfreq = slice(max_energy_index+max_e_range[0], 
                                 max_energy_index+max_e_range[1])
                   ).mean(dim='hfreq').values.item()

#%% Compute idealised k^-5/3 spectrum (so plotting k * k^-5/3 = k^-2/3)

#axa[0].plot(xrn, yidl(xrn))

kE_kp_idl = kE_kp_ref.copy()

kE_kp_idl[:] = ymax * (k_ref.values/ x_ymax)**(-2/3)

#k1 = np.where(k_ref > 1/75.)[0][0]

#kE_kp_idl[k1:] = kE_kp_ref[k1] * (k[k1:]/k[k1])**(-2/3)

kE_kp_idl.plot( label = r'Ideal $k^{-5/3}$', ax=axa[0])

axa[0].legend()
axa[0].set_ylim(ylims)
axa[0].set_xlabel(r'Wavenumber m$^{-1}$')

axa[0].set_ylabel(r'$kE(k)$')


plt.tight_layout()

#%%

idl_filt = kE_kp_ref / kE_kp_idl

idl_ref = idl_filt.copy()
idl_ref[:] = 1.0

#%
idl_filt.plot(xscale='log', yscale='log', ax=axa[1], label=reflab)

idl_ref.plot(xscale='log', yscale='log', ax=axa[1], label=r'Ideal $k^{-5/3}$')

filt = idl_filt.copy()


#for (alpha, sigma) in [(2.0, 2.0), (2.0, 2.5), (2.0, 3.0), (3.0, 2.0), (3.0, 2.5), (3.0, 3.0)]:
# for sigma in [ 2.5, 2.7, 2.9]:
for del_f in del_fs:
    sigma = del_f * dx / np.pi / 2
    for alpha in alphas:
        filt[:] = np.exp(-(sigma*k_angular.values)**alpha)
        filt.plot(xscale='log', yscale='log', ax=axa[1],
                  label=rf'$\alpha$={alpha:0.1f}, $\Delta_f/\Delta$= {del_f:0.1f}')
    # axa[1].plot(k_ref.values, G, label=)

secax = axa[1].secondary_xaxis('top', functions=(inv, inv))
secax.set_xlabel('wavelength (m)')

axa[1].legend()
axa[1].set_xlim(xlims2)
axa[1].set_ylim([0.1,1.1])
axa[1].set_xlabel(r'Wavenumber m$^{-1}$')
axa[1].set_ylabel(r'$G(k)\times G(k)^*$')

#%
idl_filt.plot(xscale='log', yscale='linear', ax=axa[2], label=reflab)

idl_ref.plot(xscale='log', yscale='linear', ax=axa[2], label=r'Ideal $k^{-5/3}$')
filt = idl_filt.copy()


for del_f in del_fs:
    sigma = del_f * dx / np.pi / 2
    for alpha in alphas:
        filt[:] = np.exp(-(sigma*k_angular.values)**alpha)
        filt.plot(xscale='log', yscale='linear', ax=axa[2],
                  label=rf'$\alpha$={alpha:0.1f}, $\Delta_f/\Delta$= {del_f:0.1f} ')
    # axa[1].plot(k_ref.values, G, label=)

secax = axa[2].secondary_xaxis('top', functions=(inv, inv))
secax.set_xlabel('wavelength (m)')

axa[2].legend()
axa[2].set_xlim(xlims2)
axa[2].set_ylim([0.0,1.1])
axa[2].set_xlabel(r'Wavenumber m$^{-1}$')
axa[2].set_ylabel(r'$G(k)\times G(k)^*$')

plt.tight_layout()
plt.savefig(indir + 'filter_fit.png', dpi=300)

dso.close()
