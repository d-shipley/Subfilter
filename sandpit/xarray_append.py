# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 17:14:58 2022

@author: paclk
"""
import numpy as np
import xarray as xr
import sys
filename = 'test.nc'

run_pass = int(sys.argv[1])
if run_pass not in [0,1]:
    print(f"run_pass must be either 0 or 1! Got {run_pass}")
    sys.exit(1)
    
if run_pass == 0:

    u = xr.DataArray(data = np.arange(20, dtype=float),
                     name = "u",
                     dims = ["x"],
                     coords = {'x':np.linspace(0.,19.0,20)},
                     attrs = {'File type':'Test'},
                     )

    v = xr.DataArray(data = 10 - 0.5 * np.arange(20),
                     name = "v",
                     dims = ["x"],
                     coords = {'x':np.linspace(0.,19.0,20)},
                     attrs = {'File type':'Test'},
                     )
    print(u, v)

    ds = xr.Dataset({
                     "u":u,
                     "v":v,
                     }
                    )

    print(ds)

    d = ds.to_netcdf(filename, mode='w')

    ds.close()
else:
    ds = xr.open_dataset(filename, mode='a')
    print(ds)

    w = xr.DataArray(data = np.arange(20, dtype=float) / 10.0,
                     name = "w",
                     dims = ["x"],
                     coords = {'x':np.linspace(0.,19.0,20)},
                     attrs = {'File type':'Test'},
                     )

    ds["w"] = w
    print(ds)

    d = ds["w"].to_netcdf(filename, mode='a')
#    d = ds.to_netcdf(filename, mode='a')

#    print(ds)
    ds.close()
