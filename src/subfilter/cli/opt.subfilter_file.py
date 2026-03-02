# -*- coding: utf-8 -*-
"""

  opt.subfilter_file.py
    - Accepts several input option flags: see opt.subfilter_file.py -h
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
import difference_ops as do

import pdb
  # pdb.set_trace()

# Set global defaults
indir  = ''
infile = ''
ref_file = ''
outpath = ''
plot_dir = ''
plot_type = '.png'
figshow = False


# Usage Message
def print_usage_message():
    print('Usage of opt.subfilter_file.py:')
    print('  opt.subfilter_file.py -i <input path> -f <input file> -r <reference profile file> -o <output path for filtered data> -p <ouput path for plots> -t <plot type extension> -s <True or False>')
    print('    -i or --input_path:  Text string path to input data')
    print('    -f or --input_file:  Text string file name')
    print('    -r or --ref_file:    Text string reference profile-containing file name')
    print('    -o or --out_data:    Text string path to output filtered data')
    print('    -p or --out_plot:    Text string path to output plots')
    print('    -t or --plot_type:   Text string plot file type extension, DEFAULT: .png')
    print('    -s or --figshow:     Text string logical whether to display figures on screen, DEFAULT: False')
    sys.exit()


# Read in user-supplied options arguments
def read_cl_arguments():

    # Modifying globals
    global indir
    global infile
    global ref_file
    global outpath
    global plot_dir
    global plot_type
    global figshow
    global options

    if len(sys.argv) > 1 :
        options, args = getopt.getopt(sys.argv[1:],"hi:f:r:o:p:t:s:", \
                        ["input_path=", "input_file=", "ref_file=", "out_data=", "out_plot=", \
                         "plot_type=", "figshow=",])
        for opt, arg in options:
            if opt == '-h' :
                print_usage_message()
            if opt in ("-i", "--input_path"):
                indir = os.path.join(arg,'')  # ensure trailing slash
            if opt in ("-f", "--input_file"):
                infile = arg
            if opt in ("-r", "--ref_file"):
                ref_file = arg
            if opt in ("-o", "--out_data"):
                outpath = os.path.join(arg,'')
            if opt in ("-p", "--out_plot"):
                plot_dir = os.path.join(arg,'')
            if opt in ("-t", "--plot_type"):
                plot_type = arg
            if opt in ("-s", "--figshow"):
                figshow = (arg in ['True', 'true', 'T', 't'])
    else :
        print('No options arguments provided.')
        print(' STOP - Must provide path and file information:')
        print('')
        print_usage_message()

    if not bool(indir.strip()):
        print(' STOP Must provide input path:\n')
        print_usage_message()
    if not bool(infile.strip()):
        print(' STOP Must provide input file:\n')
        print_usage_message()
    if not bool(ref_file.strip()):
        print(' WARN: No ref_file provided; trying with input_file.')
        ref_file = infile
    if not bool(ref_file.strip()):
        print(' STOP Must provide reference file:\n')
        print_usage_message()
    if not bool(outpath.strip()):
        print(' STOP Must provide data output path:\n')
        print_usage_message()
    if not bool(plot_dir.strip()):
        print(' WARN: No out_plot provided; trying with out_data.')
        plot_dir = outpath + 'plots/'
    if not bool(plot_dir.strip()):
        print(' STOP Must provide plot output path:\n')
        print_usage_message()
    print("")



def plot_field(var_name, filtered_data, twod_filter, ilev, iy, grid='p'):

    var_r = filtered_data[f"{var_name}_on_{grid}_r"]
    var_s = filtered_data[f"{var_name}_on_{grid}_s"]


    for it in range(var_r.shape[0]):
        if twod_filter.attributes['filter_type']=='domain' :
            zcoord = sf.last_dim(filtered_data[var_r.dimensions[1]])
            pltdat = (var_r[it,:])

            fig1, axa = plt.subplots(1,1,figsize=(5,5))

    #    plt.subplot(3, 2, 1)
            Cs1 = axa.plot(pltdat, zcoord)
            axa.set_xlabel(r'%s$^r$'%(var_name))
            axa.set_ylabel('z')

            plt.tight_layout()

            plt.savefig(plot_dir+var_name+'_prof_'+\
                    twod_filter.id+'_%02d'%it+plot_type)
            plt.close()
        else :
            zcoord = sf.last_dim(filtered_data[var_r.dimensions[3]])
            meanfield= np.mean(var_r[it,...],axis=(0,1),keepdims=True)
            pltdat = (var_r[it,...]-meanfield)

            nlevels = 40
            plt.clf

            fig1, axa = plt.subplots(3,2,figsize=(10,12))

        #    plt.subplot(3, 2, 1)
            Cs1 = axa[0,0].contourf(np.transpose(pltdat[:, :, ilev]),\
                     nlevels)
            axa[0,0].set_title(r'%s$^r$ pert level %03d'%(var_name,ilev))
            axa[0,0].set_xlabel('x')

        # Make a colorbar for the ContourSet returned by the contourf call.
            cbar1 = fig1.colorbar(Cs1,ax=axa[0,0])
            cbar1.ax.set_ylabel(var_name)
        # Add the contour line levels to the colorbar
        #  cbar.add_lines(CS2)

        #    plt.subplot(3, 2, 2)
            Cs2 = axa[0,1].contourf(np.transpose(var_s[it, :, :, ilev]),\
                     nlevels)
            axa[0,1].set_xlabel('x')
            axa[0,1].set_title(r'%s$^s$ level %03d'%(var_name,ilev))
            cbar2 = fig1.colorbar(Cs2,ax=axa[0,1])
            cbar2.ax.set_ylabel(var_name)
        #
        #    plt.subplot(3, 2, 3)
            Cs3 = axa[1,0].contourf(np.transpose(pltdat[:,iy,:]),nlevels)
        #
        ##    plt.ylim([0,5000])
            axa[1,0].set_title(r'%s$^r$ pert at iy %03d'%(var_name,iy))
        ## Make a colorbar for the ContourSet returned by the contourf call.
            cbar3 = fig1.colorbar(Cs3,ax=axa[1,0])
            cbar3.ax.set_ylabel(var_name)
        #
        #    plt.subplot(3, 2, 4)
            Cs4 = axa[1,1].contourf(np.transpose(var_s[it, :, iy,:]),nlevels)
        #
        ##    plt.ylim([0,5000])
            axa[1,1].set_title(r'%s$^s$ at iy %03d'%(var_name,iy))
        ## Make a colorbar for the ContourSet returned by the contourf call.
            cbar4 = fig1.colorbar(Cs4,ax=axa[1,1])
            cbar4.ax.set_ylabel(var_name)
        #
            x=(np.arange(0,var_r.shape[2])-0.5*var_r.shape[2])*0.1
        #    plt.subplot(3, 2, 5)
            ax1 = axa[2,0].plot(x,pltdat[:,iy,ilev])
        #
        #    plt.subplot(3, 2, 6)
            ax2 = axa[2,1].plot(x,var_s[it,:,iy,ilev])
        #
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

    v1_r = filtered_data[f"{v1}_on_{grid}_r"]
    v2_r = filtered_data[f"{v2}_on_{grid}_r"]

    print(v1,v2)
    s_v1v2 = filtered_data[f"{v1}_{v2}_on_{grid}"]
    print(s_v1v2)

    for it in range(s_v1v2.shape[0]):

        if twod_filter.attributes['filter_type']=='domain' :
            pltdat = (s_v1v2[it,:])
            zcoord = sf.last_dim(filtered_data[v1_r.dimensions[1]])

            fig1, axa = plt.subplots(1,1,figsize=(5,5))

    #    plt.subplot(3, 2, 1)
            Cs1 = axa.plot(pltdat, zcoord)
            axa.set_xlabel('s({},{})'.format(v1,v2))
            axa.set_ylabel('z')

            plt.tight_layout()

            plt.savefig(plot_dir+var_name[0]+'_'+var_name[1]+'_prof_'+\
                    twod_filter.id+'_%02d'%it+plot_type)
            plt.close()
        else :
            zcoord = sf.last_dim(filtered_data[v1_r.dimensions[3]])
            var_r = (v1_r[it,...] - np.mean(v1_r[it,...], axis=(0,1))) * \
                    (v2_r[it,...] - np.mean(v2_r[it,...], axis=(0,1)))

            meanfield= np.mean(var_r[...],axis=(0,1),keepdims=True)
            pltdat = (var_r[...]-meanfield)

#            lev1 = np.arange(-10,10.1,0.1)
#            lev2 = np.arange(-10,10.1,0.1)

            nlevels = 40
            plt.clf

            fig1, axa = plt.subplots(3,2,figsize=(10,12))

        #    plt.subplot(3, 2, 1)
            Cs1 = axa[0,0].contourf(np.transpose(pltdat[:, :, ilev]),\
                     nlevels)
            axa[0,0].set_title(r'{}$^r${}$^r$ pert level {:03d}'.format(v1, v2,ilev))
            axa[0,0].set_xlabel('x')

        # Make a colorbar for the ContourSet returned by the contourf call.
            cbar1 = fig1.colorbar(Cs1,ax=axa[0,0])
            cbar1.ax.set_ylabel(var_name)
        # Add the contour line levels to the colorbar
        #  cbar.add_lines(CS2)

        #    plt.subplot(3, 2, 2)
            Cs2 = axa[0,1].contourf(np.transpose(s_v1v2[it, :, :, ilev]),\
                     nlevels)
            axa[0,1].set_xlabel('x')
            axa[0,1].set_title('s({},{}) level {:03d}'.format(v1,v2,ilev))
            cbar2 = fig1.colorbar(Cs2,ax=axa[0,1])
            cbar2.ax.set_ylabel(var_name)
        #
        #    plt.subplot(3, 2, 3)
            Cs3 = axa[1,0].contourf(np.transpose(pltdat[:,iy,:]),nlevels)
        #
        ##    plt.ylim([0,5000])
            axa[1,0].set_title(r'{}$^r${}$^r$ at iy={:03d}'.format(v1, v2,iy))
        ## Make a colorbar for the ContourSet returned by the contourf call.
            cbar3 = fig1.colorbar(Cs3,ax=axa[1,0])
            cbar3.ax.set_ylabel(var_name)
        #
        #    plt.subplot(3, 2, 4)
            Cs4 = axa[1,1].contourf(np.transpose(s_v1v2[it, :, iy,:]),nlevels)
        #
        ##    plt.ylim([0,5000])
            axa[1,1].set_title('s({},{}) at iy={:03d}'.format(v1,v2,iy))
        ## Make a colorbar for the ContourSet returned by the contourf call.
            cbar4 = fig1.colorbar(Cs4,ax=axa[1,1])
            cbar4.ax.set_ylabel(var_name)
        #
            x=(np.arange(0,var_r.shape[1])-0.5*var_r.shape[1])*0.1
        #    plt.subplot(3, 2, 5)
            ax1 = axa[2,0].plot(x,pltdat[:,iy,ilev])
        #
        #    plt.subplot(3, 2, 6)
            ax2 = axa[2,1].plot(x,s_v1v2[it,:,iy,ilev])
        #
            plt.tight_layout()

            plt.savefig(plot_dir+var_name[0]+'_'+var_name[1]+'_lev_'+'%03d'%ilev+'_x_z'+'%03d'%iy+'_'+\
                        twod_filter.id+'_%02d'%it+plot_type)
            plt.close()

    #
    #    plt.show()
    #plt.close()

    return

def plot_shear(var_r, var_s, zcoord,  twod_filter, ilev, iy, no_trace = True):
    var_name = "mod_S"
    if no_trace : var_name = var_name+'n'

    for it in range(var_r.shape[0]):
#        meanfield= np.mean(var_r[it,...],axis=(0,1),keepdims=True)
#        pltdat = (var_r[it,...]-meanfield)
        if twod_filter.attributes['filter_type']=='domain' :
            pltdat = (var_r[it,:])

            fig1, axa = plt.subplots(1,1,figsize=(5,5))

    #    plt.subplot(3, 2, 1)
            Cs1 = axa.plot(pltdat[1:], zcoord[1:])
            axa.set_xlabel(var_name)
            axa.set_ylabel('z')

            plt.tight_layout()

            plt.savefig(plot_dir+var_name+'_prof_'+\
                    twod_filter.id+'_%02d'%it+plot_type)
            plt.close()
        else :
            pltdat = var_r[it,...]

            nlevels = 40
            plt.clf

            fig1, axa = plt.subplots(3,2,figsize=(10,12))

        #    plt.subplot(3, 2, 1)
            Cs1 = axa[0,0].contourf(np.transpose(pltdat[:, :, ilev]),\
                     nlevels)
            axa[0,0].set_title(r'%s$^r$ level %03d'%(var_name,ilev))
            axa[0,0].set_xlabel('x')

        # Make a colorbar for the ContourSet returned by the contourf call.
            cbar1 = fig1.colorbar(Cs1,ax=axa[0,0])
            cbar1.ax.set_ylabel(var_name)
        # Add the contour line levels to the colorbar
        #  cbar.add_lines(CS2)

        #    plt.subplot(3, 2, 2)
            Cs2 = axa[0,1].contourf(np.transpose(var_s[it, :, :, ilev]),\
                     nlevels)
            axa[0,1].set_xlabel('x')
            axa[0,1].set_title(r'%s$^s$ level %03d'%(var_name,ilev))
            cbar2 = fig1.colorbar(Cs2,ax=axa[0,1])
            cbar2.ax.set_ylabel(var_name)
        #
        #    plt.subplot(3, 2, 3)
            Cs3 = axa[1,0].contourf(np.transpose(pltdat[:,iy,:]),nlevels)
        #
        ##    plt.ylim([0,5000])
            axa[1,0].set_title(r'%s$^r$ at iy %03d'%(var_name,iy))
        ## Make a colorbar for the ContourSet returned by the contourf call.
            cbar3 = fig1.colorbar(Cs3,ax=axa[1,0])
            cbar3.ax.set_ylabel(var_name)
        #
        #    plt.subplot(3, 2, 4)
            Cs4 = axa[1,1].contourf(np.transpose(var_s[it, :, iy,:]),nlevels)
        #
        ##    plt.ylim([0,5000])
            axa[1,1].set_title(r'%s$^s$ at iy %03d'%(var_name,iy))
        ## Make a colorbar for the ContourSet returned by the contourf call.
            cbar4 = fig1.colorbar(Cs4,ax=axa[1,1])
            cbar4.ax.set_ylabel(var_name)
        #
            x=(np.arange(0,var_r.shape[2])-0.5*var_r.shape[2])*0.1
        #    plt.subplot(3, 2, 5)
            ax1 = axa[2,0].plot(x,pltdat[:,iy,ilev])
        #
        #    plt.subplot(3, 2, 6)
            ax2 = axa[2,1].plot(x,var_s[it,:,iy,ilev])
        #
            plt.tight_layout()

            plt.savefig(plot_dir+var_name+'_lev_'+'%03d'%ilev+'_x_z'+'%03d'%iy+'_'+\
                        twod_filter.id+'_%02d'%it+plot_type)
            plt.close()

    #
    #    plt.show()
#    plt.close()

    return

def main():
    '''
    Top level code, a bit of a mess.
    '''

# Read the command line arguments (directories, files, etc.)
    read_cl_arguments()

# Handle output directories
    os.makedirs(outpath,exist_ok = True)
    os.makedirs(plot_dir, exist_ok = True)

# Set filter attributes:
    filter_name = 'gaussian'  # Can be: gaussian, running_mean, wave_cutoff, or domain
    sigma_list = [100., 200., 400., 800., 1600.] # lengthscale of Gaussian filter [m]

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
    dataset = Dataset(indir+infile, 'r') # Dataset is the class behavior to open the file
                                         # and create an instance of the ncCDF4 class
    ref_dataset = Dataset(indir+ref_file, 'r')
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
        sf.setup_derived_data_file( indir+infile, outpath, indir+ref_file, fname,
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
    z = do.last_dim(dataset["z"])
    zn = do.last_dim(dataset["zn"])

# Loop over list of 2D filters
    for twod_filter in filter_list:

        print(twod_filter)

# Creates the file to contain the filtered variables.
        filtered_dataset_name, filtered_data, exists = \
            sf.setup_filtered_data_file( indir+infile, outpath, indir+ref_file, fname,
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

#            times = derived_data[tname]
#            print(times)
#            print(times[:])
#            print(d_r.shape)
#            for i in range(3) :
#                for j in range(3) :
#                    print(d_r[i][j],d_s[i][j])

#            Sn_ij_r, mod_Sn_r = sf.shear(d_r)
#            Sn_ij_s, mod_Sn_s = sf.shear(d_s)
#            S_ij_r, mod_S_r = sf.shear(d_r, no_trace = False)
#            S_ij_s, mod_S_s = sf.shear(d_s, no_trace = False)
#            print(S_ij_r.keys())
#        input("Press enter")


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
