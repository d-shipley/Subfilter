# -*- coding: utf-8 -*-
"""

  arg.subfilter_file.py
    - Accepts input file name as argument
      - Still need to specify in/out arguments.
    - Control subfilter engine.  Operating on a single file, filters 3D vars and outputs filtered fields (and plots).
    - Contains sample plotting routines.

Created on Tue Oct 23 11:27:25 2018

@author: Peter Clark
@modified: Todd Jones

"""
import os
import sys
import getopt
import netCDF4
from netCDF4 import Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
  # data_folder = Path("source_data/text_files/")
  # file_to_open = data_folder / "raw_data.txt"

import subfilter as sf
import filters as filt

from monc_utils.data_utils.string_utils import get_string_index


import pdb
  # pdb.set_trace()
options = {
#        'FFT_type': 'FFTconvolve',
#        'FFT_type': 'FFT',
        'FFT_type': 'RFFT', # [ RFFT, FFTconvolve, FFT ]
        'save_all': 'Yes',  # [ Yes, No ]
          }



dir = '/gws/nopw/j04/paracon_rdg/users/toddj/BOMEX/CA_S_SC_BOMEX_25_600/rerun/'
odir = '/gws/nopw/j04/paracon_rdg/users/toddj/BOMEX/CA_S_SC_BOMEX_25_600/f3/'
odir = odir + 'test_op_' + options['FFT_type']+'/'

os.makedirs(odir, exist_ok = True)

#file = 'BOMEX_m0025_g0600_all_32400.0.nc' #'diagnostics_ts_18000.0.nc'
file = sys.argv[1]
ref_file = file #'diagnostics_ts_18000.0.nc'

print("Working on file: ",file)

tname='time_series_150_600.0'

#w = dataset.variables['w']
#var_tvar = w.dimensions[0]
#var_time = dataset.variables[var_tvar]

plot_dir = odir + 'plots/'
os.makedirs(plot_dir, exist_ok = True)

plot_type = '.png'

figshow = True

def plot_field(var_name, filtered_data, twod_filter, ilev, iy, grid='p'):

    var_r = filtered_data['ds'][f"f({var_name}_on_{grid})_r"]
    var_s = filtered_data['ds'][f"f({var_name}_on_{grid})_s"]

    [iix, iiy, iiz] =  get_string_index(var_r.dims, ['x', 'y', 'z'])
    [xvar, yvar, zvar] = [list(var_s.dims)[i] for i in [iix, iiy, iiz]]

    for it, time in enumerate(var_r.coords['time']):
        if twod_filter.attributes['filter_type']=='domain' :
            
            
            zcoord = var_r.dims[zvar]

            fig1, axa = plt.subplots(1,1,figsize=(5,5))

            Cs1 = var_r.isel(time=it).plot(y=zvar, ax = axa)

            plt.tight_layout()

            plt.savefig(plot_dir+var_name+'_prof_'+\
                    twod_filter.id+'_%02d'%it+plot_type)
            plt.close()
        else :
            meanfield= var_r.isel(time=it).mean(dim=(xvar, yvar))
            pltdat = (var_r.isel(time=it)-meanfield)

            nlevels = 40
            plt.clf

            fig1, axa = plt.subplots(3,2,figsize=(10,12))

            Cs1 = pltdat.isel({zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,0], levels=nlevels)

            Cs2 = var_s.isel({'time':it, zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,1], levels=nlevels)

            Cs3 = pltdat.isel({yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,0], levels=nlevels)

            Cs4 = var_s.isel({'time':it, yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,1], levels=nlevels)

            p1 = pltdat.isel({yvar:iy, zvar:ilev}).plot(ax=axa[2,0])

            p2 = var_s.isel({'time':it, yvar:iy, zvar:ilev}).plot(ax=axa[2,1])

            plt.tight_layout()

            plt.savefig(plot_dir+var_name+'_lev_'+'%03d'%ilev+'_x_z'+'%03d'%iy+'_'+\
                        twod_filter.id+'_%02d'%it+plot_type)
            plt.close()


    #
    #    plt.show()
    #plt.close()

    return

def plot_quad_field(var_name, filtered_data, twod_filter, ilev, iy, grid='p'):

    v1 = var_name[0]
    v2 = var_name[1]

    v1_r = filtered_data['ds'][f"f({v1}_on_{grid})_r"]
    v2_r = filtered_data['ds'][f"f({v2}_on_{grid})_r"]

    s_v1v2 = filtered_data['ds'][f"s({v1},{v2})_on_{grid}"]

    [iix, iiy, iiz] = get_string_index(s_v1v2.dims, ['x', 'y', 'z'])
    if iix is not None:
        xvar = s_v1v2.dims[iix]
        yvar = s_v1v2.dims[iiy]
    zvar = s_v1v2.dims[iiz]

    for it, time in enumerate(s_v1v2.coords['time']):

        print(f'it:{it}')


        if twod_filter.attributes['filter_type']=='domain' :

            fig1, axa = plt.subplots(1,1,figsize=(5,5))

            Cs1 = s_v1v2.isel(time=it).plot(y=zvar, ax = axa)

            plt.tight_layout()

            plt.savefig(plot_dir+var_name[0]+'_'+var_name[1]+'_prof_'+\
                    twod_filter.id+'_%02d'%it+plot_type)
            plt.close()
        else :

            var_r = (v1_r.isel(time=it) - v1_r.isel(time=it).mean(dim=(xvar, yvar))) * \
                    (v2_r.isel(time=it) - v2_r.isel(time=it).mean(dim=(xvar, yvar)))


            pltdat = var_r

            pltdat.name = 'f('+v1+')_r.'+'f('+v2+')_r'

            nlevels = 40
            plt.clf

            fig1, axa = plt.subplots(3,2,figsize=(10,12))

            Cs1 = pltdat.isel({zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,0], levels=nlevels)

            Cs2 = s_v1v2.isel({'time':it, zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,1], levels=nlevels)

            Cs3 = pltdat.isel({yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,0], levels=nlevels)

            Cs4 = s_v1v2.isel({'time':it, yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,1], levels=nlevels)

            p1 = pltdat.isel({yvar:iy, zvar:ilev}).plot(ax=axa[2,0])

            p2 = s_v1v2.isel({'time':it, yvar:iy, zvar:ilev}).plot(ax=axa[2,1])

            plt.tight_layout()

            plt.savefig(plot_dir+var_name[0]+'_'+var_name[1]+'_lev_'+'%03d'%ilev+'_x_z'+'%03d'%iy+'_'+\
                        twod_filter.id+'_%02d'%it+plot_type)
            plt.close()


    return

def plot_shear(var_r, var_s, zcoord,  twod_filter, plot_dir, ilev, iy, no_trace = True):
    var_name = var_r.name
    if no_trace : var_name = var_name+'n'

    [iix, iiy, iiz] = get_string_index(var_s.dims, ['x', 'y', 'z'])
    [xvar, yvar, zvar] = [list(var_s.dims)[i] for i in [iix, iiy, iiz]]

    for it, time in enumerate(var_r.coords['time']):
        print(f'it:{it}')

        if twod_filter.attributes['filter_type']=='domain' :

            fig1, axa = plt.subplots(1,1,figsize=(5,5))

            Cs1 = var_r.isel({'time':it, zvar:slice(1,None)}).plot(y=zvar, ax = axa)

            plt.tight_layout()

            plt.savefig(plot_dir+var_name+'_prof_'+\
                    twod_filter.id+'_%02d'%it+plot_type)
            plt.close()
        else :
            pltdat = var_r.isel(time=it)

            nlevels = 40
            plt.clf

            fig1, axa = plt.subplots(3,2,figsize=(10,12))

            Cs1 = pltdat.isel({zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,0], levels=nlevels)

            Cs2 = var_s.isel({'time':it, zvar:ilev}).plot.imshow(x=xvar, y=yvar, ax=axa[0,1], levels=nlevels)

            Cs3 = pltdat.isel({yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,0], levels=nlevels)

#             axa[1,0].set_title(r'%s$^r$ pert at iy %03d'%(var_name,iy))
            Cs4 = var_s.isel({'time':it, yvar:iy}).plot.imshow(x=xvar, y=zvar, ax=axa[1,1], levels=nlevels)

            p1 = pltdat.isel({yvar:iy, zvar:ilev}).plot(ax=axa[2,0])

            p2 = var_s.isel({'time':it, yvar:iy, zvar:ilev}).plot(ax=axa[2,1])
            plt.tight_layout()

            plt.savefig(plot_dir+var_name+'_lev_'+'%03d'%ilev+'_x_z'+'%03d'%iy+'_'+\
                        twod_filter.id+'_%02d'%it+plot_type)
            plt.close()

    return


def main():
    '''
    Top level code, a bit of a mess.
    '''

# Set filter attributes:
    filter_name = 'gaussian'  # Can be: gaussian, running_mean, wave_cutoff, or domain
    sigma_list = [800.0, 500.0, 220.0] # [m]
    sigma_list = [500.0, 100.0]
    width = -1     # If set, controls the width of the filter.
                   # Must be set for running-mean filter:
                   #   filtername = 'running_mean'
                   #   width = 20  #

# Horizontal resolution (NOTE: want to automate this)
    dx = 25.0   # [m]
    dy = 25.0   # [m]

# Open the diagnostic file and the file containing the reference profile.
    print("In main: ",sigma_list)
    print("Dataset load...")
    dataset = Dataset(dir+file, 'r') # Dataset is the class behavior to open the file
                                     # and create an instance of the ncCDF4 class
    ref_dataset = Dataset(dir+ref_file, 'r')
    print("Dataset loaded.")

# Set arbitrary height level (ilev) and y-position (iy) for plotting
    ilev = 15
    iy = 40

# Set the output grid type from [u,v,w,p]
    opgrid = 'w'

# Set append file label for output data files.
    fname = 'test_plot'

# Construct derived data (or read existing from file) NOTE: input the override option
#  Puts basic and additional fields on output grid (opgrid)
#  File name of output is that of input appended with fname.
    print("setup_derived_data_file...")
    derived_dataset_name, derived_data, exists = \
        sf.setup_derived_data_file( dir+file, odir, dir+ref_file, fname,
                                   options, override=True)
    print("Variables in derived dataset.")
    print(derived_data.variables)

# Construct list of 2D filters to be applied
# WATCH naming conventions
    filter_list = list([])
    for i,sigma in enumerate(sigma_list):
        print("Making filter for: ",'filter_gn_{:05n}'.format(sigma))
        if filter_name == 'gaussian':
            filter_id = 'filter_gn_{:05n}'.format(sigma)
            twod_filter = filt.Filter(filter_id,
                                       filter_name,
                                       sigma=sigma, width=width,
                                       delta_x=dx,  cutoff = 0.00001)  # smaller cutoff
        elif filter_name == 'wave_cutoff':
            filter_id = 'filter_wc_{:05n}'.format(sigma)
            twod_filter = filt.Filter(filter_id,
                                       filter_name, wavenumber=np.pi/(2*sigma),
                                       width=width,
                                       delta_x=dx)
        elif filter_name == 'running_mean':
            filter_id = 'filter_rm_{:05n}'.format(sigma)
            width = int(np.round( sigma/dx * np.pi * 2.0 / 3.0)+1)
            twod_filter = filt.Filter(filter_id,
                                       filter_name,
                                       width=width,
                                       delta_x=dx)

        print(twod_filter)
        filter_list.append(twod_filter)

# Add whole domain filter
#    filter_name = 'domain'
#    filter_id = 'filter_mean'
#    twod_filter = filt.Filter(filter_id, filter_name, delta_x=dx)
#    filter_list.append(twod_filter)

    print(filter_list)

# Pulls height coordinates that have been possibly stored with a time dimension.
    z = dataset["z"]
    zn = dataset["zn"]

# Loop over list of 2D filters
    for twod_filter in filter_list:

        print(twod_filter)

# Creates the file to contain the filtered variables.
        filtered_dataset_name, filtered_data, exists = \
            sf.setup_filtered_data_file( dir+file, odir, dir+ref_file, fname,
                                       options, twod_filter, override=True)
        print("Variables in filtered dataset.")
        print(derived_data.variables)
        exists = False
        if exists :
            print('Filtered data file exists' )
        else :

            print("\n    filter_variable_list...",twod_filter.id)
# Actual filtering performed.  Data stored in filtered_data (set/file).
            field_list = sf.filter_variable_list(dataset, ref_dataset,
                                                 derived_data, filtered_data,
                                                 options,
                                                 twod_filter, var_list=None,
                                                 grid = opgrid)

# Creates filtered versions of paired input variables
            print("\n    filter_variable_pair_list...",twod_filter.id)
            quad_field_list = sf.filter_variable_pair_list(dataset, ref_dataset,
                                                derived_data, filtered_data,
                                                options,
                                                twod_filter, var_list=None,
                                                grid = opgrid)

# Creates filtered versions of the deformation field
            print("\n    filtered_deformation...",twod_filter.id)
            d_r, d_s = sf.filtered_deformation(dataset, derived_data,
                                              filtered_data, options,
                                              twod_filter, dx, dy, z, zn,
                                              xaxis=1, grid='w')

            times = derived_data[tname]
            print(times)
            print(times[:])
            print(d_r.shape)
            for i in range(3) :
                for j in range(3) :
                    print(d_r[i][j],d_s[i][j])

            Sn_ij_r, mod_Sn_r = sf.shear(d_r)
            Sn_ij_s, mod_Sn_s = sf.shear(d_s)
            S_ij_r, mod_S_r = sf.shear(d_r, no_trace = False)
            S_ij_s, mod_S_s = sf.shear(d_s, no_trace = False)
            print(S_ij_r.keys())
        input("Press enter")


        if figshow :
# Plotting section
            if twod_filter.attributes['filter_type']!='domain' :
                fig1 = plt.figure(1)
                plt.contourf(twod_filter.data,20)

                fig2 = plt.figure(2)
                plt.plot(twod_filter.data[np.shape(twod_filter.data)[0]//2,:])
# disable plot to screen                plt.show()

            for field in field_list:
                print("Plotting {}".format(field))
                plot_field(field, derived_data, twod_filter, ilev, iy, grid=opgrid)

            for field in quad_field_list :
                print("Plotting {}".format(field))
                plot_quad_field(field, derived_data, twod_filter, ilev, iy, \
                                grid=opgrid)

            #plot_shear(mod_Sn_r, mod_Sn_s, z, twod_filter, ilev, iy, no_trace = True)
            #plot_shear(mod_S_r, mod_S_s, z, twod_filter, ilev, iy, no_trace = False)
       # end if figshow


        filtered_data.close() # close file for this filter.
    # end for loop over 2D filters

    derived_data.close()  # Close derived data file.
    dataset.close()  # Close the input dataset file.

    print("IAMDONE")  # done tag for stdout tracking

if __name__ == "__main__":
    main()
