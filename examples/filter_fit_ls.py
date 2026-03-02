# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:24:16 2023

@author: paclk
"""
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

def inv(k):
    return 1.0 / k

def fit_ISR(kE_k_ref):
    # Find maximum in spectrum
    
    k_ref = kE_k_ref['hfreq']
    
    max_energy_index = kE_k_ref.values.argmax()
    
    if max_energy_index.size > 1:
        max_energy_index = max_energy_index[0]
    
    k_at_kE_max = k_ref.values[max_energy_index]
    #ymax = kE_kp_ref[max_energy_index]
    
    k53_Ek = kE_k_ref * (k_ref.values/ k_at_kE_max)**(2/3)
    
    kE_k_max = k53_Ek.isel(hfreq = slice(max_energy_index+max_e_range[0], 
                                         max_energy_index+max_e_range[1])
                          ).mean(dim='hfreq').values.item()

    # Compute idealised k^-5/3 spectrum (so plotting k * k^-5/3 = k^-2/3)

    kE_k_idl = kE_k_ref.copy()

    kE_k_idl[:] = kE_k_max * (k_ref.values/ k_at_kE_max)**(-2/3)
    
    return kE_k_idl, max_energy_index

def fit_gen_gaussian(idl_filt):
    
    ln_idl_filt = np.log(idl_filt)
    
    y = -ln_idl_filt.loc[ln_idl_filt.hfreq >= wn_thresh]
    
    yp = y.loc[ y > 0 ]
    
    xp = yp.hwaven.values # - yp.hwaven.values[0]
    
    
    xpv = np.log(xp ) 
    ypv = np.log(yp.values)
                    
    xpvb = xpv.mean()
    ypvb = ypv.mean()
       
    xpvp = xpv - xpvb
    ypvp = ypv - ypvb
                 
    alpha = (xpvp * ypvp).mean() / (xpvp * xpvp).mean() 
    
    alpha_lnsigma =  ypvb - alpha * xpvb
    
    sigma = np.exp(alpha_lnsigma / alpha)
    
    del_f = 2 * np.pi * sigma / dx
    
    alpha_g = 2.0
    
    alpha_lnsigma_g = ypvb - alpha_g * xpvb
    
    sigma_g = np.exp(alpha_lnsigma_g / alpha_g)
    
    del_f_g = 2 * np.pi * sigma_g / dx
    
    print(f'alpha={alpha} sigma={sigma} del_f={del_f}') 
    print(f'alpha_g={alpha_g} sigma_g={sigma_g} del_f_g={del_f_g}') 
    
    return xpv, ypv, [(alpha,   sigma,   del_f), 
                      (alpha_g, sigma_g, del_f_g)]


# dir = 'C:/Users/paclk/OneDrive - University of Reading/ug_project_data/Data/'
# file = 'diagnostics_3d_ts_21600.nc'

# dirroot = 'C:/Users/paclk/OneDrive - University of Reading/'
# dirroot = 'C:/Users/xm904103/OneDrive - University of Reading/'
dirroot = 'F:/Data/'

test_case = 0
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
    wn_thresh =1E-3

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
    wn_thresh = 2.500E-2

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

if test_case == 0:
    dx = dy = 50
else:
    dx =  dso.attrs['dx']
    dy = dso.attrs['dy']

#%% Get time meaned spectrum

E_k_ref = dso['spec_2d_w'].mean(dim='time').sel(z_w=plot_height, method='nearest')

k_ref = dso['hfreq']

k = k_ref.copy()

k_angular = dso['hwaven']

kE_k_ref = k_ref * E_k_ref

#%% Compute idealised k^-5/3 spectrum (so plotting k * k^-5/3 = k^-2/3)

kE_k_idl, max_energy_index = fit_ISR(kE_k_ref)

#%% Compute ideal filter function and reference 

idl_filt = kE_k_ref / kE_k_idl

idl_ref = idl_filt.copy()
idl_ref[:] = 1.0

# xpv, ypv, [(alpha,   sigma,   del_f), 
#                   (alpha_g, sigma_g, del_f_g)] = fit_gen_gaussian(idl_filt)

xpv, ypv, fitted_parameters = fit_gen_gaussian(idl_filt[max_energy_index:])

#%% Plot Energy Density Spectrum

fig1, axa = plt.subplots(3,1,figsize=(8,12))

kE_k_ref.plot(xscale='log', yscale='log', label=reflab, ax=axa[0])

secax = axa[0].secondary_xaxis('top', functions=(inv, inv))
secax.set_xlabel('wavelength (m)')

xrn = axa[0].get_xlim()
yrn = axa[0].get_ylim()

kE_k_idl.plot( label = r'Ideal $k^{-5/3}$', ax=axa[0])

axa[0].legend()
axa[0].set_ylim(ylims)
axa[0].set_xlabel(r'Wavenumber m$^{-1}$')

axa[0].set_ylabel(r'$kE(k)$')

fig1.tight_layout()


#%%
idl_filt.plot(xscale='log', yscale='log', ax=axa[1], label=reflab)
idl_ref.plot(xscale='log', yscale='log', ax=axa[1], label=r'Ideal $k^{-5/3}$')

filt_list = []
for (alpha,   sigma,   del_f) in fitted_parameters:
    filt = idl_filt.copy()    
    filt[:] = np.exp(-(sigma*k_angular.values)**alpha)
    filt.plot(xscale='log', yscale='log', ax=axa[1],
              label=rf'$\alpha$={alpha:0.3f}, $\Delta_f/\Delta$= {del_f:0.3f}')
    filt_list.append(filt)

# filt_g = idl_filt.copy()
# filt_g[:] = np.exp(-(sigma_g*k_angular.values)**alpha_g)
# filt_g.plot(xscale='log', yscale='log', ax=axa[1],
#           label=rf'$\alpha$={alpha_g:0.3f}, $\Delta_f/\Delta$= {del_f_g:0.3f}')
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
#filt = idl_filt.copy()
#filt[:] = np.exp(-(sigma*k_angular.values)**alpha)

for i, (alpha,   sigma,   del_f) in enumerate(fitted_parameters):
    filt = filt_list[i]
    filt.plot(xscale='log', yscale='linear', ax=axa[2],
              label=rf'$\alpha$={alpha:0.3f}, $\Delta_f/\Delta$= {del_f:0.3f} ')
# filt_g.plot(xscale='log', yscale='linear', ax=axa[2],
#           label=rf'$\alpha$={alpha_g:0.3f}, $\Delta_f/\Delta$= {del_f_g:0.3f}')

secax = axa[2].secondary_xaxis('top', functions=(inv, inv))
secax.set_xlabel('wavelength (m)')

axa[2].legend()
axa[2].set_xlim(xlims2)
axa[2].set_ylim([0.0,1.1])
axa[2].set_xlabel(r'Wavenumber m$^{-1}$')
axa[2].set_ylabel(r'$G(k)\times G(k)^*$')

fig1.tight_layout()
fig1.savefig(indir + 'filter_fit_ls2.pdf')

#%% Plot fitted data

fig2, axb = plt.subplots(1,1,figsize=(5,5))

axb.plot(xpv, ypv, label = r'ln(-ln(Ref 5 m / Ideal $k^{-5/3}$))')

for (alpha,   sigma,   del_f) in fitted_parameters:
    axb.plot(xpv, alpha *  xpv + alpha * np.log(sigma),
             label = rf'$\alpha$={alpha:0.3f}, $\Delta_f/\Delta$= {del_f:0.3f}')

axb.legend()
axb.set_xlabel(r'ln(Angular Wavenumber)')
axb.set_ylabel(r'ln(-ln(G(k))')
fig2.tight_layout()

fig2.savefig(indir + 'filter_fit_ls.pdf')

dso.close()
