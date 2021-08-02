#!/usr/bin/env python

"""
Author: Mike Smith
Modified on 7/30/2021 by Lori Garzio
Last modified 7/30/2021
"""

import argparse
import numpy as np
import os
import pandas as pd
import sys
import xarray as xr
from collections import OrderedDict
from wrf import getvar, interplevel, default_fill
import functions.common as cf


def main(args):
    fname = args.file
    save_file = args.save_file

    # List of variables that are already included in the WRF output
    variables = ['XLAT', 'XLONG', 'LANDMASK', 'LAKEMASK']

    # Output time units
    time_units = 'seconds since 1970-01-01 00:00:00'

    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Create list of heights in meters and pressure (mb) for interpolation of U and V components
    heights_m = [50, 100, 150, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500]
    heights_mb = [200, 300, 500, 700, 850, 925]

    # Open using netCDF toolbox
    ncfile = xr.open_dataset(fname)
    original_global_attributes = ncfile.attrs
    ncfile = ncfile._file_obj.ds

    # Load variables and append to dictionary
    nc_vars = {}
    for v in variables:
        nc_vars[v] = cf.delete_attr(getvar(ncfile, v))

    # Calculate u and v components of wind rotated to Earth coordinates
    uvm = getvar(ncfile, 'uvmet')

    # Subtract terrain height from height above sea level for height in meters
    new_z = getvar(ncfile, 'z') - getvar(ncfile, 'ter')

    # interpolate u and v components of wind to defined heights (in meters)
    uvtemp = interplevel(uvm, new_z, heights_m, default_fill(np.float32))
    uvtemp = uvtemp.rename({'level': 'height'})
    utemp, vtemp = cf.split_uvm(uvtemp)

    # Concatenate the list of calculated u and v values into data array
    nc_vars['UH'] = xr.concat(utemp, dim='height')
    nc_vars['VH'] = xr.concat(vtemp, dim='height')

    # Get pressure - units are Pa so convert to mb
    p = getvar(ncfile, 'p') * .01

    # interpolate u and v components of wind to defined heights in pressure (mb)
    uvtemp = interplevel(uvm, p, heights_mb, default_fill(np.float32))
    uvtemp = uvtemp.rename({'level': 'pressure'})
    utemp, vtemp = cf.split_uvm(uvtemp)

    # Concatenate the list of calculated u and v values into data array
    nc_vars['UP'] = xr.concat(utemp, dim='pressure')
    nc_vars['VP'] = xr.concat(vtemp, dim='pressure')

    # Calculate 10m u and v components of wind rotated to Earth coordinates and split into separate variables
    nc_vars['U10'], nc_vars['V10'] = cf.split_uvm(getvar(ncfile, 'uvmet10'))

    # Create xarray dataset of variables
    ds = xr.Dataset({**nc_vars})
    ds['UH'] = ds.UH.astype(np.float32)
    ds['VH'] = ds.VH.astype(np.float32)
    ds['UP'] = ds.UP.astype(np.float32)
    ds['VP'] = ds.VP.astype(np.float32)
    ds['height'] = ds.height.astype(np.int32)
    ds['pressure'] = ds.pressure.astype(np.int32)

    try:
        del ds.UH.attrs['vert_units']
        del ds.VH.attrs['vert_units']
        del ds.UP.attrs['vert_units']
        del ds.VP.attrs['vert_units']
    except KeyError:
        pass

    ds['Times'] = np.array([pd.Timestamp(ds.Time.data).strftime('%Y-%m-%d_%H:%M:%S')]).astype('<S19')
    ds = ds.expand_dims('Time', axis=0)

    # Add description and units for lon, lat dimensions for georeferencing
    ds['XLAT'].attrs['description'] = 'latitude'
    ds['XLAT'].attrs['units'] = 'degree_north'
    ds['XLONG'].attrs['description'] = 'longitude'
    ds['XLONG'].attrs['units'] = 'degree_east'

    # Set XTIME attribute
    ds['XTIME'].attrs['units'] = 'minutes'

    # Set lon attributes
    ds['XLONG'].attrs['long_name'] = 'Longitude'
    ds['XLONG'].attrs['standard_name'] = 'longitude'
    ds['XLONG'].attrs['short_name'] = 'lon'
    ds['XLONG'].attrs['units'] = 'degrees_east'
    ds['XLONG'].attrs['axis'] = 'X'
    ds['XLONG'].attrs['valid_min'] = np.float32(-180.0)
    ds['XLONG'].attrs['valid_max'] = np.float32(180.0)

    # Set lat attributes
    ds['XLAT'].attrs['long_name'] = 'Latitude'
    ds['XLAT'].attrs['standard_name'] = 'latitude'
    ds['XLAT'].attrs['short_name'] = 'lat'
    ds['XLAT'].attrs['units'] = 'degrees_north'
    ds['XLAT'].attrs['axis'] = 'Y'
    ds['XLAT'].attrs['valid_min'] = np.float32(-90.0)
    ds['XLAT'].attrs['valid_max'] = np.float32(90.0)

    # Set depth attributes
    ds['height'].attrs['long_name'] = 'Height Above Ground Level'
    ds['height'].attrs['standard_name'] = 'height'
    ds['height'].attrs['comment'] = 'Derived from subtracting terrain height from height above sea level'
    ds['height'].attrs['units'] = 'm'
    ds['height'].attrs['axis'] = 'Z'
    ds['height'].attrs['positive'] = 'up'

    ds['pressure'].attrs['long_name'] = 'Pressure'
    ds['pressure'].attrs['standard_name'] = 'air_pressure'
    ds['pressure'].attrs['units'] = 'millibars'
    ds['pressure'].attrs['axis'] = 'Z'
    ds['pressure'].attrs['positive'] = 'up'

    # Set u attributes - interpolated to height in meters
    ds['UH'].attrs['long_name'] = 'Eastward Wind Component'
    ds['UH'].attrs['standard_name'] = 'eastward_wind'
    ds['UH'].attrs['short_name'] = 'u'
    ds['UH'].attrs['units'] = 'm s-1'
    ds['UH'].attrs['description'] = 'earth rotated u, interpolated to Height Above Ground Level in meters'
    ds['UH'].attrs['valid_min'] = np.float32(-300)
    ds['UH'].attrs['valid_max'] = np.float32(300)

    # Set v attributes - interpolated to height in meters
    ds['VH'].attrs['long_name'] = 'Northward Wind Component'
    ds['VH'].attrs['standard_name'] = 'northward_wind'
    ds['VH'].attrs['short_name'] = 'v'
    ds['VH'].attrs['units'] = 'm s-1'
    ds['VH'].attrs['description'] = 'earth rotated v, interpolated to Height Above Ground Level in meters'
    ds['VH'].attrs['valid_min'] = np.float32(-300)
    ds['VH'].attrs['valid_max'] = np.float32(300)

    # Set u attributes - interpolated to pressure in mb
    ds['UP'].attrs['long_name'] = 'Eastward Wind Component'
    ds['UP'].attrs['standard_name'] = 'eastward_wind'
    ds['UP'].attrs['short_name'] = 'u'
    ds['UP'].attrs['units'] = 'm s-1'
    ds['UP'].attrs['description'] = 'earth rotated u, interpolated to Pressure in millibars'
    ds['UP'].attrs['valid_min'] = np.float32(-300)
    ds['UP'].attrs['valid_max'] = np.float32(300)

    # Set v attributes - interpolated to pressure in mb
    ds['VP'].attrs['long_name'] = 'Northward Wind Component'
    ds['VP'].attrs['standard_name'] = 'northward_wind'
    ds['VP'].attrs['short_name'] = 'v'
    ds['VP'].attrs['units'] = 'm s-1'
    ds['VP'].attrs['description'] = 'earth rotated v, interpolated to Pressure in millibars'
    ds['VP'].attrs['valid_min'] = np.float32(-300)
    ds['VP'].attrs['valid_max'] = np.float32(300)

    # Set u10 attributes
    ds['U10'].attrs['long_name'] = 'Eastward Wind Component - 10m'
    ds['U10'].attrs['standard_name'] = 'eastward_wind'
    ds['U10'].attrs['short_name'] = 'u'
    ds['U10'].attrs['units'] = 'm s-1'
    ds['U10'].attrs['description'] = '10m earth rotated u'
    ds['U10'].attrs['valid_min'] = np.float32(-300)
    ds['U10'].attrs['valid_max'] = np.float32(300)

    # Set v10 attributes
    ds['V10'].attrs['long_name'] = 'Northward Wind Component - 10m'
    ds['V10'].attrs['standard_name'] = 'northward_wind'
    ds['V10'].attrs['short_name'] = 'v'
    ds['V10'].attrs['units'] = 'm s-1'
    ds['V10'].attrs['description'] = '10m earth rotated v'
    ds['V10'].attrs['valid_min'] = np.float32(-300)
    ds['V10'].attrs['valid_max'] = np.float32(300)

    ds['LANDMASK'].attrs['standard_name'] = 'land_binary_mask'
    ds['LANDMASK'].attrs['long_name'] = 'Land Mask'

    ds['LAKEMASK'].attrs['long_name'] = 'Lake Mask'

    ds['XTIME'].attrs['long_name'] = 'minutes since simulation start'

    # Set time attribute
    ds['Time'].attrs['standard_name'] = 'time'

    datetime_format = '%Y%m%dT%H%M%SZ'
    created = pd.Timestamp(pd.datetime.utcnow()).strftime(datetime_format)  # creation time Timestamp
    time_start = pd.Timestamp(pd.Timestamp(ds.Time.data[0])).strftime(datetime_format)
    time_end = pd.Timestamp(pd.Timestamp(ds.Time.data[0])).strftime(datetime_format)
    global_attributes = OrderedDict([
        ('title', 'Rutgers Weather Research and Forecasting Model'),
        ('summary', 'Processed netCDF containing subset of RUWRF output'),
        ('keywords', 'Weather Advisories > Marine Weather/Forecast'),
        ('Conventions', 'CF-1.7'),
        ('naming_authority', 'edu.rutgers.marine.rucool'),
        ('history', '10-minute WRF raw output processed into new 10-minute file with selected variables.'),
        ('processing_level', 'Level 2'),
        ('comment', 'WRF Model operated by RUCOOL'),
        ('acknowledgement', 'This data is provided by the Rutgers Center for Ocean Observing Leadership. Funding is provided by the New Jersey Board of Public Utilities).'),
        ('standard_name_vocabulary', 'CF Standard Name Table v41'),
        ('date_created', created),
        ('creator_name', 'Joseph Brodie'),
        ('creator_email', 'jbrodie@marine.rutgers.edu'),
        ('creator_url', 'rucool.marine.rutgers.edu'),
        ('institution', 'Center for Ocean Observing and Leadership, Department of Marine & Coastal Sciences, Rutgers University'),
        ('project', 'New Jersey Board of Public Utilities - Offshore Wind Energy - RUWRF Model'),
        ('geospatial_lat_min', -90),
        ('geospatial_lat_max', 90),
        ('geospatial_lon_min', -180),
        ('geospatial_lon_max', 180),
        ('geospatial_vertical_min', 0.0),
        ('geospatial_vertical_max', 0.0),
        ('geospatial_vertical_positive', 'down'),
        ('time_coverage_start', time_start),
        ('time_coverage_end', time_end),
        ('creator_type', 'person'),
        ('creator_institution', 'Rutgers University'),
        ('contributor_name', 'Joseph Brodie'),
        ('contributor_role', 'Director of Atmospheric Research'),
        ('geospatial_lat_units', 'degrees_north'),
        ('geospatial_lon_units', 'degrees_east'),
        ('date_modified', created),
        ('date_issued', created),
        ('date_metadata_modified', created),
        ('keywords_vocabulary', 'GCMD Science Keywords'),
        ('platform', 'WRF Model Run'),
        ('cdm_data_type', 'Grid'),
        ('references', 'http://maracoos.org/node/146 https://rucool.marine.rutgers.edu/facilities https://rucool.marine.rutgers.edu/data')])

    global_attributes.update(original_global_attributes)
    ds = ds.assign_attrs(global_attributes)

    # Add compression to all variables
    encoding = {}
    for k in ds.data_vars:
        encoding[k] = {'zlib': True, 'complevel': 1}

    # add the encoding for time so xarray exports the proper time.
    # Also remove compression from dimensions. They should never have fill values
    encoding['Time'] = dict(units=time_units, calendar='gregorian', zlib=False, _FillValue=False, dtype=np.double)
    encoding['XLONG'] = dict(zlib=False, _FillValue=False)
    encoding['XLAT'] = dict(zlib=False, _FillValue=False)
    encoding['height'] = dict(zlib=False, _FillValue=False, dtype=np.int32)
    encoding['pressure'] = dict(zlib=False, _FillValue=False, dtype=np.int32)

    ds.to_netcdf(save_file, encoding=encoding, format='netCDF4', engine='netcdf4', unlimited_dims='Time')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-f', '--file',
                            dest='file',
                            type=str,
                            help='Full file path to Raw WRF netCDF file ')

    arg_parser.add_argument('-s', '--save_file',
                            dest='save_file',
                            type=str,
                            help='Full file path to save directory and save filename')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
