"""
subfilter.py
- This is the "subfilter module"
- Defines many useful routines for the subfilter calculations.
- examples of their use are present in subfilter_file.py

Created on Tue Oct 23 11:07:05 2018

@authors: Peter Clark, Dan Shipley
"""
import yaml
import monc_utils

import numpy as np
import xarray as xr

from scipy.signal import fftconvolve

import monc_utils.data_utils.deformation as defm
import monc_utils.data_utils.cloud_monc as cldm
from .utils.default_variables import (get_default_variable_list,
                                      get_default_variable_pair_list)
from monc_utils.data_utils.string_utils import get_string_index
from monc_utils.io.datain import get_data_on_grid
from monc_utils.io.dataout import save_field, setup_child_file
from monc_utils.data_utils.dask_utils import re_chunk

from loguru import logger

# Global constants
#===============================================================================
from sys import float_info
eps = float_info.min # smallest possible float

#===============================================================================

def subfilter_options(config_file:str=None):
    """
    Set default options for filtering.

    Parameters
    ----------
    config_file : str
        Path to configuration .yaml file. The default is None.

    Returns
    -------
    options : dict
        Options including optional updates
    update_config : dict
        Updates from config_file

    """
    update_config = None
    options = {
                'FFT_type': 'RFFT',
                'save_all': 'Yes',
                'override': True,      # Overwrite output file if it exists.
                'input_file': None,    # For user convenience, not required
                'ref_file': None,      # For user convenience, not required
                'outpath': None,       # For user convenience, not required
              }
    if config_file is not None:
        with open(config_file) as c:
            update_config = yaml.load(c, Loader = yaml.SafeLoader)

        options.update(update_config['options'])

    return options, update_config

def filter_variable_list(source_dataset, ref_dataset, derived_dataset,
                         filtered_dataset, options, filter_def,
                         var_list=None, grid='p') :
    """
    Create filtered versions of input variables on required grid.

    Stored in derived_dataset.

    Parameters
    ----------
        source_dataset  : NetCDF dataset for input
        ref_dataset     : NetCDF dataset for input containing reference
                          profiles. Can be None
        derived_dataset : NetCDF dataset for derived data
        filtered_dataset: NetCDF dataset for derived data
        options         : General options e.g. FFT method used.
        filter_def      : 1 or 2D filter
        var_list=None   : List of variable names.
        default provided by get_default_variable_list()
        grid='p'        : Grid - 'u','v','w' or 'p'

    Returns
    -------
        list : list of strings representing variable names.

    """
    if (var_list==None):
        var_list = get_default_variable_list()
        logger.info("Filtering with default list:\n",var_list)

    for vin in var_list:

        op_var  = get_data_on_grid(source_dataset, ref_dataset, vin, 
                                   derived_dataset=derived_dataset,
                                   options=options,
                                   grid=grid, 
                                   )

        v = op_var.name



        if f'f({v})_r' not in filtered_dataset['ds'].variables \
            or f'f({v})_s' not in filtered_dataset['ds'].variables:

            ncvar_r, ncvar_s = filter_field(op_var,
                                            filtered_dataset,
                                            options, 
                                            filter_def)
        else:
            logger.info(f'f({v})_r and f({v})_s already in output dataset.')

    return var_list

def weighted_filter_variable_list(source_dataset, ref_dataset, derived_dataset,
                         filtered_dataset, options, filter_def,
                         var_list=None, grid='p', weights=None) :
    """
    Create filtered versions of input variables on required grid.

    Stored in derived_dataset.

    Parameters
    ----------
        source_dataset  : NetCDF dataset for input
        ref_dataset     : NetCDF dataset for input containing reference
                          profiles. Can be None
        derived_dataset : NetCDF dataset for derived data
        filtered_dataset: NetCDF dataset for derived data
        options         : General options e.g. FFT method used.
        filter_def      : 1 or 2D filter
        var_list=None   : List of variable names.
        default provided by get_default_variable_list()
        grid='p'        : Grid - 'u','v','w' or 'p'
        weights=None    : Xarray Dataset of weights for weighted
                          filtering.

    Returns
    -------
        list : list of strings representing variable names.

    """
    if (var_list==None):
        var_list = get_default_variable_list()
        logger.info("Filtering with default list:\n",var_list)

    if weights is not None:
        #TODO: check weights are on the right grid!
        for wname in weights:
            for vin in var_list:
                op_var = get_data_on_grid(source_dataset, ref_dataset, vin, 
                                   derived_dataset=derived_dataset,
                                   options=options,
                                   grid=grid)
                v = op_var.name
                ncvar_r, ncvar_s = weighted_filter_field(
                    op_var,
                    filtered_dataset,
                    options,
                    filter_def,
                    weight=weights[wname]
                )

    else:
        for vin in var_list:

            op_var  = get_data_on_grid(source_dataset, ref_dataset, vin, 
                                       derived_dataset=derived_dataset,
                                       options=options,
                                       grid=grid, 
                      )

            v = op_var.name

            # looks like the same if statement appears in filter_field
            # so is probably redundant here
            if f'f({v})_r' not in filtered_dataset['ds'].variables \
               or f'f({v})_s' not in filtered_dataset['ds'].variables:

                ncvar_r, ncvar_s = filter_field(op_var,
                                                filtered_dataset,
                                                options, 
                                                filter_def)
            else:
                logger.info(
                    f'f({v})_r and f({v})_s already in output dataset.'
                )

    return var_list

def filter_variable_pair_list(source_dataset, ref_dataset, derived_dataset,
                              filtered_dataset, options, filter_def,
                              var_list=None, grid='p') :
    """
    Create filtered versions of pairs input variables on A grid.

    Stored in derived_dataset.

    Args:
        source_dataset  : NetCDF dataset for input
        ref_dataset     : NetCDF dataset for input containing reference
                          profiles. Can be None
        derived_dataset : NetCDF dataset for derived data
        filtered_dataset: NetCDF dataset for derived data
        options         : General options e.g. FFT method used.
        filter_def      : 1 or 2D filter
        var_list=None   : List of variable names.
        default provided by get_default_variable_pair_list()

    Returns
    -------
        var_list : list of lists of pairs strings representing variable names.

    """
    if (var_list==None):
        var_list = get_default_variable_pair_list()
        logger.info("Default list:\n",var_list)

    for v in var_list:

        logger.info(f"Calculating s({v[0]:s},{v[1]:s})")
        svars = quadratic_subfilter(source_dataset, ref_dataset,
                                  derived_dataset, filtered_dataset, options,
                                  filter_def, v[0], v[1], grid=grid)

        (s_var1var2, var1var2, var1var2_r, var1var2_s) = svars

        save_field(filtered_dataset, s_var1var2)
        if options['save_all'].lower() == 'yes':
            for f in ( var1var2_r, var1var2_s):
                save_field(filtered_dataset, f)

            save_field(derived_dataset, var1var2)

    return var_list

def weighted_filter_variable_pair_list(source_dataset, ref_dataset, derived_dataset,
                              filtered_dataset, options, filter_def,
                                       var_list=None, grid='p', weights=None) :
    """
    Create filtered versions of pairs input variables on A grid.

    Stored in derived_dataset.

    Args:
        source_dataset  : NetCDF dataset for input
        ref_dataset     : NetCDF dataset for input containing reference
                          profiles. Can be None
        derived_dataset : NetCDF dataset for derived data
        filtered_dataset: NetCDF dataset for derived data
        options         : General options e.g. FFT method used.
        filter_def      : 1 or 2D filter
        var_list=None   : List of variable names.
        default provided by get_default_variable_pair_list()
        weights=None    : xr.Dataset of weights for weighted filtering

    Returns
    -------
        var_list : list of lists of pairs strings representing variable names.

    """
    if (var_list==None):
        var_list = get_default_variable_pair_list()
        logger.info("Default list:\n",var_list)

    if weights is not None:
        for wname in weights:
            for v in var_list:

                logger.info(f"Calculating s({v[0]:s},{v[1]:s})")
                svars = weighted_quadratic_subfilter(
                    source_dataset, ref_dataset, derived_dataset, filtered_dataset,
                    options, filter_def, v[0], v[1], grid=grid, weight=weights[wname]
                )
                
                (s_var1var2, var1var2, var1var2_r, var1var2_s) = svars
                
                save_field(filtered_dataset, s_var1var2)
                if options['save_all'].lower() == 'yes':
                    for f in ( var1var2_r, var1var2_s):
                        save_field(filtered_dataset, f)

                    save_field(derived_dataset, var1var2)

    else:
        for v in var_list:

            logger.info(f"Calculating s({v[0]:s},{v[1]:s})")
            svars = weighted_quadratic_subfilter(
                source_dataset, ref_dataset, derived_dataset, filtered_dataset,
                options, filter_def, v[0], v[1], grid=grid, weights=None
            )

            (s_var1var2, var1var2, var1var2_r, var1var2_s) = svars

            save_field(filtered_dataset, s_var1var2)
            if options['save_all'].lower() == 'yes':
                for f in ( var1var2_r, var1var2_s):
                    save_field(filtered_dataset, f)

                save_field(derived_dataset, var1var2)

    return var_list


# Flags are: 'u-grid, v-grid, w-grid'

def convolve(field, options, filter_def, dims):
    """
    Convolve field filter using fftconvolve using padding.

    Args:
        field      : field array
        options    : General options e.g. FFT method used.
        filter_def : 1 or 2D filter array

    Returns
    -------
        ndarray : field convolved with filter_def

    """
    logger.info(f'Convolving {field.shape} with filter {filter_def.shape} over dim{dims}.')
    
    if len(np.shape(field)) > len(np.shape(filter_def)):
        edims = tuple(np.setdiff1d(np.arange(len(np.shape(field))), dims))
        filter_def = np.expand_dims(filter_def, axis=edims)


    if options['FFT_type'].upper() == 'FFTCONVOLVE':

        pad_len = np.max(np.shape(filter_def))//2

        pad_list = []
        for i in range(len(np.shape(field))):
            if i in dims:
                pad_list.append((pad_len,pad_len))
            else:
                pad_list.append((0,0))

        field = np.pad(field, pad_list, mode='wrap')
        result = fftconvolve(field, filter_def, mode='same', axes=dims)

        padspec = []
        for d in range(len(np.shape(field))):
            if d in dims:
                padspec.append(slice(pad_len,-pad_len))
            else:
                padspec.append(slice(0,None))
        padspec = tuple(padspec)

        result = result[padspec]

    elif options['FFT_type'].upper() == 'FFT':

        if len(np.shape(filter_def)) == 1:
            fft_field = np.fft.fft(field, axes=dims)
            fft_filtered_field = fft_field * filter_def
            result = np.fft.ifft(fft_filtered_field, axes=dims)
        else:
            fft_field = np.fft.fft2(field, axes=dims)
            fft_filtered_field = fft_field * filter_def
            result = np.fft.ifft2(fft_filtered_field, axes=dims)
        result = result.real

    elif options['FFT_type'].upper() == 'RFFT':

        if len(np.shape(filter_def)) == 1:
            fft_field = np.fft.rfft(field, axes=dims)
            fft_filtered_field = fft_field * filter_def
            result = np.fft.irfft(fft_filtered_field, axes=dims)
        else:
            fft_field = np.fft.rfft2(field, axes=dims)
            fft_filtered_field = fft_field * filter_def
            result = np.fft.irfft2(fft_filtered_field, axes=dims)
        result = result.real

    else:
        raise ValueError(f"The supplied FFT_type option, {options['FFT_type']}, \
                           is not valid.  Use one of: [ FFTCONVOLVE, FFT, RFFT ].")
    return result

def pad_to_len(field, newlen, mode='constant'):
    """
    Pad array to required size on each dimension.

    Parameters
    ----------
    field : numpy array
        Input field.
    newlen : int
        Length of each dimension.
    mode : str or function, optional
        Detrmines values to pad with. See numpy.pad. The default is 'constant'.

    Returns
    -------
    padfield : numpy array
        Input field padded as required..

    """
    sf = np.shape(field)
    padlen = newlen - sf[0]
    padleft = padlen - padlen//2
    padright = padlen - padleft
    padfield = np.pad(field, ((padleft,padright),), mode=mode)
    return padfield

def filtered_field_calc(var, options, filter_def):
    """
    Split field into resolved f(field)_r and subfilter f(field)_s.

    Note: this routine has a deliberate side effect, to store the fft or rfft
    of the filter in filter_def for subsequent re-use.

    Args:
        var             : dict cantaining variable info
        options         : General options e.g. FFT method used.
        filter_def      : 1 or 2D filter

    Returns
    -------
        dicts containing variable info : [var_r, var_s]

    """
    
    if not monc_utils.global_config['no_dask']:
        var = re_chunk(var, xch = 'all', ych = 'all')

    vname = var.name
    field = var.data
    vdims = var.dims

    logger.debug(f'filtering \n{field}')
    
    sh = np.shape(field)

    if filter_def.attributes['ndim'] == 1:

        axis = get_string_index(vdims, ['x'])

    elif filter_def.attributes['ndim'] == 2:

        axis = get_string_index(vdims, ['x', 'y'])


    if filter_def.attributes['filter_type'] == 'domain' :

        ax = list(axis)

        si = np.asarray(field.shape)
        si[ax] = 1

        field_r = np.mean(field[...], axis=axis)
        field_s = field[...] - np.reshape(field_r, si)

        rdims =  []
        rcoords = {}
        for i, d in enumerate(vdims):
            if i not in axis:
                rdims.append(d)
                rcoords[d] = var.coords[d]
        rdims = tuple(rdims)


    else :

        logger.info(f"Filtering using {options['FFT_type']}")

        if options['FFT_type'].upper() == 'FFTCONVOLVE':

            field_r = convolve(field, options, filter_def.data, axis)

        elif options['FFT_type'].upper() == 'FFT':

            if filter_def.attributes['ndim'] == 1:

                if 'fft' not in filter_def.__dict__:
                    sf = np.shape(filter_def.data)
                    if sh[axis[0]] != sf[0]:
                        padfilt = pad_to_len(filter_def.data, sh[axis[0]])
                    else:
                        padfilt = filter_def.data.copy()
                    # This shift of the filter is necessary to get the phase
                    # information right.
                    padfilt = np.fft.ifftshift(padfilt)
                    filter_def.fft = np.fft.fft(padfilt)

            else:

                if 'fft' not in filter_def.__dict__:
                    sf = np.shape(filter_def.data)
                    if sh[axis[0]] != sf[0] or sh[axis[1]] != sf[1]:
                        padfilt = pad_to_len(filter_def.data, sh[axis[0]])
                    else:
                        padfilt = filter_def.data.copy()
                    # This shift of the filter is necessary to get the phase
                    # information right.
                    padfilt = np.fft.ifftshift(padfilt)
                    filter_def.fft = np.fft.fft2(padfilt)

            field_r = convolve(field, options, filter_def.fft, axis)
            rdims = var.dims
            rcoords = var.coords

        elif options['FFT_type'].upper() == 'RFFT':

            if filter_def.attributes['ndim'] == 1:

                if 'rfft' not in filter_def.__dict__:
                    sf = np.shape(filter_def.data)
                    if sh[axis[0]] != sf[0]:
                        padfilt = pad_to_len(filter_def.data, sh[axis[0]])
                    else:
                        padfilt = filter_def.data.copy()
                    # This shift of the filter is necessary to get the phase
                    # information right.
                    padfilt = np.fft.ifftshift(padfilt) 
                    filter_def.rfft = np.fft.rfft(padfilt)

            else:

                if 'rfft' not in filter_def.__dict__:
                    sf = np.shape(filter_def.data)
                    if sh[axis[0]] != sf[0] or sh[axis[1]] != sf[1]:
                        padfilt = pad_to_len(filter_def.data, sh[axis[0]])
                    else:
                        padfilt = filter_def.data.copy()
                    # This shift of the filter is necessary to get the phase
                    # information right.
                    padfilt = np.fft.ifftshift(padfilt)
                    filter_def.rfft = np.fft.rfft2(padfilt)

            field_r = convolve(field, options, filter_def.rfft, axis)

        elif options['FFT_type'].upper() == 'DIRECT' :
            if filter_def.attributes['filter_type'] == 'one_two_one':


                stencil = np.array([1,2,1])/4
                field_r = np.zeros_like(field)
                if filter_def.attributes['ndim'] == 1:
                    if filter_def.attributes['width'] \
                        == np.shape(filter_def.data)[0]:
                        stencil = filter_def.data
                    else:
                        stencil = np.array([1,2,1])/4

                    for ix in range(-1,2):
                        field_r += np.roll(field, ix, axis=axis) \
                                   * stencil[ix+1]
                else:
                    if filter_def.attributes['width'] \
                        == np.shape(filter_def.data)[0]:
                        stencil = filter_def.data
                    else:
                        stencil = np.array([1,2,1])/4
                        stencil = np.outer(stencil, stencil)
                    for ix in range(-1,2):
                        for iy in range(-1,2):
                            field_r += np.roll(field, (ix,iy), axis=axis) \
                                       * stencil[ix+1, iy+1]

            else:
                raise ValueError(
                    'FFT_type DIRECT only works with one_two_one filter.')

        rdims = var.dims
        rcoords = var.coords
        field_s = field[...] - field_r

    sdims = var.dims
    scoords = var.coords

    var_r = xr.DataArray(field_r, name = 'f('+vname+')_r', dims=rdims,
                      coords=rcoords)
    var_s = xr.DataArray(field_s, name = 'f('+vname+')_s', dims=sdims,
                      coords=scoords)

    return (var_r, var_s)


def setup_derived_data_file(source_file, destdir, fname,
                            options, override=False) :
    """
    Create NetCDF dataset for derived data in destdir.

    File name is original file name concatenated with filter_def.id.


    Args:
        source_file     : NetCDF file name.
        destdir         : Directory for derived data.
        override=False  : if True force creation of file

    Returns
    -------
        derived_dataset_name, derived_dataset

    """
    derived_dataset, exists = setup_child_file(source_file, destdir, fname,
                            options, override=override)

    return derived_dataset, exists

def setup_filtered_data_file(source_file, destdir, fname,
                            options, filter_def, override=False) :
    """
    Create NetCDF dataset for filtered data in destdir.

    File name is original file name concatenated with filter_def.id.

    Args:
        source_file     : NetCDF file name.
        destdir         : Directory for derived data.
        options         : General options e.g. FFT method used.
        filter_def      : Filter
        options         : General options e.g. FFT method used.
        override=False  : if True force creation of file

    Returns
    -------
        filtered_dataset_name, filtered_dataset

    @author: Peter Clark
    """
    if fname == '':
        outtag = filter_def.id
    else:
        outtag = fname + "_" + filter_def.id
    attrs = {**{'filter_def_id' : filter_def.id},
             **filter_def.attributes, **options}
    filtered_dataset, exists = setup_child_file(source_file, destdir, outtag,
                              attrs, override=override)


    return filtered_dataset, exists

def filter_field(var, filtered_dataset, options, filter_def) :
    """
    Create filtered versions of input variable, stored in filtered_dataset.

    Args:
        var            : dict cantaining variable info
        filtered_dataset : NetCDF dataset for derived data.
        options         : General options e.g. FFT method used.
        filter_def      : 1 or 2D filter.
        default provided by get_default_variable_list()

    Returns
    -------
        ncvar_r, ncvar_s: Resolved and subfilter fields as netcf variables in
                          filtered_dataset.

    """
    vname = var.name
    vname_r = 'f('+vname+')_r'
    vname_s = 'f('+vname+')_s'

    if vname_r in filtered_dataset['ds'] and vname_s in filtered_dataset['ds']:

        logger.info(f"Reading {vname_r}, {vname_s}")
        var_r = filtered_dataset['ds'][vname_r]
        var_s = filtered_dataset['ds'][vname_s]

    else:

        logger.info(f"Filtering {vname:s}")

        # Calculate resolved and unresolved parts of var

        (var_r, var_s) = filtered_field_calc(var, options, filter_def)

        var_r = save_field(filtered_dataset, var_r)
        var_s = save_field(filtered_dataset, var_s)

    return (var_r, var_s)

def weighted_filter_field(var, filtered_dataset, options, filter_def, weight=None,
                          weight_eps=1e-6, save=True) :
    """
    Create filtered versions of input variable, stored in filtered_dataset.

    Args:
        var            : dict cantaining variable info
        filtered_dataset : NetCDF dataset for derived data.
        options         : General options e.g. FFT method used.
        filter_def      : 1 or 2D filter.
        default provided by get_default_variable_list()
        weight          : xarray DataArray for weight field (e.g. density, indicator function)

    Returns
    -------
        ncvar_r, ncvar_s   : Resolved and subfilter fields as netcf variables in
                             filtered_dataset.
        weight_r, weight_s : Resolved and subfilter weight field as NetCDF variables in
                             filtered_dataset (optional)

    """

    vname = 'f('+var.name+')'

    if weight is not None:
        wname = weight.name
        wname_r = 'f('+wname+')_r'
        wname_s = 'f('+wname+')_s'
        vname = vname+'_'+wname+'-weighted'
        if wname_r in filtered_dataset['ds'] and wname_s in filtered_dataset['ds']:

            logger.info(f"Reading {wname_r}, {wname_s}")
            weight_r = filtered_dataset['ds'][wname_r]
            weight_s = filtered_dataset['ds'][wname_s]
        else:

            logger.info(f"Filtering {wname:s}")

            # Calculate resolved and unresolved parts of weight

            (weight_r, weight_s) = filtered_field_calc(
                weight,
                options,
                filter_def
            )

            print("\n\n\n",weight_r)
            print("\n\n\n",weight_s)

            if save:
                weight_r = save_field(filtered_dataset, weight_r)
                weight_s = save_field(filtered_dataset, weight_s)
            
    vname_r = vname+'_r'
    vname_s = vname+'_s'

    if vname_r in filtered_dataset['ds'] and vname_s in filtered_dataset['ds']:

        logger.info(f"Reading {vname_r}, {vname_s}")
        var_r = filtered_dataset['ds'][vname_r]
        var_s = filtered_dataset['ds'][vname_s]

    else:

        logger.info(f"Filtering {vname:s}")

        # Calculate resolved and unresolved parts of var

        if weight is not None:
            weighted_var = weight*var
            weighted_var.name = weight.name+'.'+var.name

            (_var_r, _var_s) = filtered_field_calc(weighted_var, options, filter_def)

            # fill value has to be zero, not np.nan, so as not to break sum rules
            var_r = xr.where(weight_r > weight_eps, _var_r/weight_r, 0.0)
            var_s = xr.where(weight_r > weight_eps, _var_s/weight_r, 0.0)
            
            var_r = var_r.rename(vname_r)
            var_s = var_s.rename(vname_s)

        else:
            (var_r, var_s) = filtered_field_calc(var, options, filter_def)

        if save:
            var_r = save_field(filtered_dataset, var_r)
            var_s = save_field(filtered_dataset, var_s)
            # for debugging
            _var_r = save_field(filtered_dataset, _var_r)
            _var_s = save_field(filtered_dataset, _var_s)

    return (var_r, var_s)

def filtered_deformation(source_dataset, ref_dataset, derived_dataset,
                         filtered_dataset,
                         options, filter_def,
                         grid='p'):
    """
    Create filtered versions of deformation field.

    Args:
        source_dataset  : NetCDF input dataset
        ref_dataset     : NetCDF dataset for input containing reference
                          profiles. Can be None
        derived_dataset : NetCDF dataset for derived data.
        filtered_dataset: NetCDF dataset for derived data
        options         : General options e.g. FFT method used.
        filter_def      : 1 or 2D filter.
        grid='p'        : Grid - 'u','v','w' or 'p'

    Returns
    -------
        ncvar_r, ncvar_s: Resolved and subfilter fields as netcf variables
        in filtered_dataset.

    """
#:math:`\frac{\partial u_i}{\partial{x_j}`

    d_var = defm.deformation(source_dataset, ref_dataset, derived_dataset,
                        options, grid=grid)

    if not monc_utils.global_config['no_dask']:
        d_var = re_chunk(d_var)

    (d_var_r, d_var_s) = filter_field(d_var, filtered_dataset,
                                      options, filter_def)

    return (d_var_r, d_var_s)

def quadratic_subfilter(source_dataset,  ref_dataset, derived_dataset,
                        filtered_dataset, options, filter_def,
                        v1_name, v2_name, grid='p') :
    r"""
    Create filtered versions of pair of input variables on required grid.

    Stored in derived_dataset.
    Computes :math:`s(\phi,\psi) = (\phi\psi)^r - \phi^r\psi^r.`

    Args:
        source_dataset  : NetCDF dataset for input
        ref_dataset     : NetCDF dataset for input containing reference
                          profiles. Can be None
        derived_dataset : NetCDF dataset for derived data
        filtered_dataset: NetCDF dataset for derived data
        options         : General options e.g. FFT method used.
        filter_def      : 1 or 2D filter
        v1_name         : Variable names.
        v2_name         : Variable names.

    Returns
    -------
        s(var1,var2) data array.
        vdims dimensions of var1

    """
    v1 = get_data_on_grid(source_dataset,  ref_dataset, v1_name,
                          derived_dataset=derived_dataset,
                          options=options,
                          grid=grid, 
                          )

    (var1_r, var1_s) = filter_field(v1, filtered_dataset, options,
                                    filter_def)

    if v2_name == v1_name:
        v2 = v1
        (var2_r, var2_s) = (var1_r, var1_s)
    else:
        v2 = get_data_on_grid(source_dataset, ref_dataset,v2_name, 
                             derived_dataset=derived_dataset,
                             options=options,
                             grid=grid, 
                             )

        (var2_r, var2_s) = filter_field(v2, filtered_dataset, options,
                                        filter_def)

    var1var2_name = v1.name + '.' + v2.name
    if var1var2_name in derived_dataset['ds'].variables:
        var1var2 =  derived_dataset['ds'][var1var2_name]
    else:
        var1var2 = v1 * v2
        var1var2.name = var1var2_name

    logger.info(f"Filtering {v1_name:s}*{v2_name:s}")

#    var1var2 = re_chunk(var1var2)

    (var1var2_r, var1var2_s) = filtered_field_calc(var1var2, options,
                                                 filter_def )

    s_var1var2 = var1var2_r - var1_r * var2_r

    s_var1var2.name = f"s({v1_name:s},{v2_name:s})_on_{grid:s}"

    return (s_var1var2, var1var2, var1var2_r, var1var2_s)

def weighted_quadratic_subfilter(source_dataset,  ref_dataset, derived_dataset,
                                 filtered_dataset, options, filter_def,
                                 v1_name, v2_name, grid='p', weight=None) :
    r"""
    Create filtered versions of pair of input variables on required grid.

    Stored in derived_dataset.
    Computes :math:`s(\phi,\psi) = (\phi\psi)^r - \phi^r\psi^r.`

    Args:
        source_dataset  : NetCDF dataset for input
        ref_dataset     : NetCDF dataset for input containing reference
                          profiles. Can be None
        derived_dataset : NetCDF dataset for derived data
        filtered_dataset: NetCDF dataset for derived data
        options         : General options e.g. FFT method used.
        filter_def      : 1 or 2D filter
        v1_name         : Variable names.
        v2_name         : Variable names.
        weight          : xarray DataArray for weight field (e.g. density, indicator function)

    Returns
    -------
        s(var1,var2) data array.
        vdims dimensions of var1

    """
    
    v1 = get_data_on_grid(source_dataset,  ref_dataset, v1_name,
                          derived_dataset=derived_dataset,
                          options=options,
                          grid=grid, 
                          )

    (var1_r, var1_s) = weighted_filter_field(v1, filtered_dataset, options,
                                             filter_def, weight=weight)

    if v2_name == v1_name:
        v2 = v1
        (var2_r, var2_s) = (var1_r, var1_s)
    else:
        v2 = get_data_on_grid(source_dataset, ref_dataset,v2_name, 
                             derived_dataset=derived_dataset,
                             options=options,
                             grid=grid, 
                             )

        (var2_r, var2_s) = weighted_filter_field(v2, filtered_dataset, options,
                                                 filter_def, weight=weight)

    var1var2_name = v1.name + '.' + v2.name
    if var1var2_name in derived_dataset['ds'].variables:
        var1var2 =  derived_dataset['ds'][var1var2_name]
    else:
        var1var2 = v1 * v2
        var1var2.name = var1var2_name

    logger.info(f"Filtering {v1_name:s}*{v2_name:s}")

#    var1var2 = re_chunk(var1var2)

    (var1var2_r, var1var2_s) = weighted_filter_field(var1var2, filtered_dataset, options,
                                                     filter_def, weight=weight, save=False)

    s_var1var2 = var1var2_r - var1_r * var2_r

    sname = f"s({v1_name:s},{v2_name:s})_on_{grid:s}"

    if weight is not None:
        sname = sname+'_'+weight.name+'-weighted'

    s_var1var2.name = sname

    return (s_var1var2, var1var2, var1var2_r, var1var2_s)
