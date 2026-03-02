# -*- coding: utf-8 -*-
"""
    This program evaluates MONC fields with 4 dimensions (vertical, x, y, time) to produce
        horizontal power spectra at each time and vertical level written to new netcdf files.

    BY DEFAULT, each variable contained in the input file's xarray.Dataset "Data variables"
        has its horizontal power spectra evaluated.  These are all placed in the same
        output file.
    They can alternatively be placed in a list in the user settings section.

    Several options can influence the form of the final result.

    Assumes the horizontal grid dimensions are the same for each variable being analysed.
    Assumes the horizontal dimensions are named 'x' and 'y'.
    Assumes the vertical dimension is the only dimension with a 'z' in its name, but it can be
        either 'z' or 'zn'.
    The time dimension name is identified by a user-supplied string, currently: 'time'.

    "Durran" calculation based on Durran et al. (2017): https://doi.org/10.1175/MWR-D-17-0056.1

    User must supply:
       dir:    input directory (slash-agnostic)
       file:   input file
                 Suggest switching to argument input (see below)
       outtag: output file tag (appended to input file name)
                 Creates 'spectra/' directory within the given dir
       dx:     x-direction grid spacing [m]
       dy:     y-direction grid spacing [m]

    @author: Todd Jones
"""

import os
import glob
import xarray as xr
import dask


import subfilter.spectra as sp
from monc_utils.io.dataout import setup_child_file

# ---------------------------------- USER SETTINGS ----------------------------------

# File source location and name

dirroot = 'E:/Data/'

var_names_spec = []  # list of variable names to evaluate
        #   - leave empty to work on all present variables
        #   - populate with string variable names to work on specified list
# ---------------------------------- USER SETTINGS ----------------------------------

# Add to output file name
outtag = 'spec'

#################################################################################################
#################################################################################################
#################################################################################################
def main():
    '''
    Top level code
    '''
    indir = dirroot + 'CBL/'
    outdir = dirroot + 'CBL/'
    files = 'diagnostics_3d_ts_*.nc'
    
    options = {'dx': 5.0, 'dy':5.0, 'th_ref': 300.0}
    
    spectra_options, update_config = sp.spectra_options()
    spectra_options.update({'dx': 5.0, 'dy':5.0, 'spec_1D':False})
    
    print(spectra_options)
    
    options.update(spectra_options)
    
    print(options)
    
    outdir = outdir + 'test_filter_spectra/'
    os.makedirs(outdir, exist_ok = True)

    dask.config.set({"array.slicing.split_large_chunks": True})

    infiles = glob.glob(indir+files)

    for infile in infiles:
        
        dataset = xr.open_dataset(infile , chunks={'z':'auto', 'zn':'auto'})

        print(dataset)

    # By default, we are going to look at all data variables from the file
    #   Save data variable names to list
    # ALTERNATIVELY, THE CODE COULD BE CONFIGURED TO PASS IN A LIST OF
        # SPECIFIC FIELDS TO EVALUATE.
        if len(var_names_spec) == 0:
            var_names = list(dataset.data_vars.keys())
        else:
            var_names = var_names_spec.copy()

        derived_dataset, exists = setup_child_file(infile, outdir, outtag,
                                                   options, 
                                                   override=True)

        print(f"Working on file: {infile}")

        of = sp.spectra_variable_list(dataset, derived_dataset, options,
                                        var_list=var_names)
        
        dataset.close()
        print(of)
        
        dso = xr.open_dataset(derived_dataset['file'])
    
        print(dso)
    
        dso.close()

    print("IAMDONE")  # Tag for successful completion.

# END main


if __name__ == "__main__":
    main()
