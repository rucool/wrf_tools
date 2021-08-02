#!/usr/bin/env python

"""
Author: Mike Smith
Modified on 4/10/2020 by Lori Garzio
Last modified 8/1/2021
"""

import argparse
import numpy as np
import os
import pandas as pd
import sys
import xarray as xr
from collections import OrderedDict
from wrf import getvar
import functions.common as cf


def main(args):
    fname = args.file
    save_file = args.save_file

    # List of variables to subset
    variables = ['XLAT', 'XLONG', 'temp', 'rh', 'z', 'pressure', 'ter', 'slp', 'cloudfrac', 'td', 'height_agl']

    # Output time units
    time_units = 'seconds since 1970-01-01 00:00:00'

    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Open using netCDF toolbox
    ncfile = xr.open_dataset(fname)
    original_global_attributes = ncfile.attrs
    ncfile = ncfile._file_obj.ds  # to use xarray as an input to wrf-python (instead of netcdf4)

    # Load variables and append to list
    vars = {}
    for var in variables:
        if var == 'z':  # change 'z' to 'height_msl'
            varname = 'height_msl'
        else:
            varname = var
        vars[varname] = cf.delete_attr(getvar(ncfile, var))

    # Get u and v components of wind rotated to Earth coordinates
    uvm = getvar(ncfile, 'uvmet')

    for uv in ['u', 'v']:
        da = uvm.sel(u_v=uv)
        da = cf.delete_attr(da).drop(['u_v'])  # delete extra attributes and drop the extra coordinate
        da = da.rename(uv)
        vars[uv] = da  # add to variable dictionary

    # Create xarray dataset
    ds = xr.Dataset(vars)

    ds['Times'] = np.array([pd.Timestamp(ds.Time.data).strftime('%Y-%m-%d_%H:%M:%S')]).astype('<S19')
    ds = ds.expand_dims('Time', axis=0)

    # format bottom_top and low_mid_high dimensions
    ds['bottom_top'] = np.int32(ds.bottom_top)
    ds['low_mid_high'] = np.array([300, 2000, 6000], dtype='int32')

    # add attributes for model levels
    ds['bottom_top'].attrs['units'] = '1'
    ds['bottom_top'].attrs['long_name'] = 'Model Level'
    ds['bottom_top'].attrs['comment'] = 'Integer coordinate for native WRF model level'
    ds['bottom_top'].attrs['axis'] = 'Z'
    ds['bottom_top'].attrs['positive'] = 'up'
    ds['bottom_top'].attrs['_CoordinateAxisType'] = 'GeoZ'  # need this attribute for THREDDs aggregation
    ds['bottom_top'].attrs['_CoordinateZisPositive'] = 'up'  # need this attribute for THREDDs aggregation

    ds['low_mid_high'].attrs['units'] = 'm'
    ds['low_mid_high'].attrs['long_name'] = 'Cloud Layer'
    ds['low_mid_high'].attrs['comment'] = 'Cloud layer, low 300 m, mid 2000 m, high 6000 m'
    ds['low_mid_high'].attrs['axis'] = 'Z'
    ds['low_mid_high'].attrs['positive'] = 'up'
    ds['low_mid_high'].attrs['_CoordinateAxisType'] = 'Height'  # need this attribute for THREDDs aggregation
    ds['low_mid_high'].attrs['_CoordinateZisPositive'] = 'up'  # need this attribute for THREDDs aggregation

    # Add description and units for lon, lat dimensions for georeferencing
    ds['XLAT'].attrs['description'] = 'latitude'
    ds['XLAT'].attrs['units'] = 'degree_north'
    ds['XLONG'].attrs['description'] = 'longitude'
    ds['XLONG'].attrs['units'] = 'degree_east'

    # Set time attribute
    ds['Time'].attrs['standard_name'] = 'time'
    ds['Time'].attrs['long_name'] = 'Time'

    # Set XTIME attribute
    ds['XTIME'].attrs['units'] = 'minutes'
    ds['XTIME'].attrs['long_name'] = 'minutes since simulation start'

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

    # Set attributes
    ds['temp'].attrs['long_name'] = 'Air Temperature'
    ds['temp'].attrs['standard_name'] = 'air_temperature'

    ds['rh'].attrs['long_name'] = 'Relative Humidity'
    ds['rh'].attrs['standard_name'] = 'relative_humidity'

    ds['u'].attrs['long_name'] = 'Eastward Wind Component'
    ds['u'].attrs['standard_name'] = 'eastward_wind'
    ds['u'].attrs['short_name'] = 'u'
    ds['u'].attrs['description'] = 'earth rotated u'
    ds['u'].attrs['valid_min'] = np.float32(-300)
    ds['u'].attrs['valid_max'] = np.float32(300)

    ds['v'].attrs['long_name'] = 'Northward Wind Component'
    ds['v'].attrs['standard_name'] = 'northward_wind'
    ds['v'].attrs['short_name'] = 'v'
    ds['v'].attrs['description'] = 'earth rotated v'
    ds['v'].attrs['valid_min'] = np.float32(-300)
    ds['v'].attrs['valid_max'] = np.float32(300)

    ds['height_msl'].attrs['long_name'] = 'Model Height for Mass Grid (MSL)'
    ds['height_msl'].attrs['standard_name'] = 'height_above_mean_sea_level'
    ds['height_msl'].attrs['comment'] = 'Model height above mean sea level'

    ds['pressure'].attrs['long_name'] = 'Air Pressure'
    ds['pressure'].attrs['standard_name'] = 'air_pressure'

    ds['ter'].attrs['long_name'] = 'Model Terrain Height'

    ds['slp'].attrs['long_name'] = 'Air Pressure at Sea Level'
    ds['slp'].attrs['standard_name'] = 'air_pressure_at_mean_sea_level'

    ds['cloudfrac'].attrs['long_name'] = 'Layer Cloud Area Fraction'
    ds['cloudfrac'].attrs['standard_name'] = 'cloud_area_fraction_in_atmosphere_layer'

    ds['td'].attrs['long_name'] = 'Dew Point Temperature'
    ds['td'].attrs['standard_name'] = 'dew_point_temperature'

    ds['height_agl'].attrs['long_name'] = 'Model Height for Mass Grid (AGL)'
    ds['height_agl'].attrs['standard_name'] = 'height'
    ds['height_agl'].attrs['comment'] = 'Model height above ground level'
    ds['height_agl'].attrs['axis'] = 'Z'
    ds['height_agl'].attrs['positive'] = 'up'

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
        ('history', 'Hourly WRF raw output processed into new hourly file with selected variables.'),
        ('processing_level', 'Level 2'),
        ('comment', 'WRF Model operated by RUCOOL'),
        ('acknowledgement', 'This data is provided by the Rutgers Center for Ocean Observing Leadership. Funding is provided by the New Jersey Board of Public Utilities.'),
        ('standard_name_vocabulary', 'CF Standard Name Table v72'),
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

    ds.to_netcdf(save_file, encoding=encoding, format='netCDF4', engine='netcdf4', unlimited_dims='Time')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('-f', '--file',
                            dest='file',
                            default='data/nc_raw/wrfout_d01_2018-11-19_000000.nc',
                            type=str,
                            help='Full file path to Raw WRF netCDF file ')

    arg_parser.add_argument('-s', '--save_file',
                            dest='save_file',
                            default='data/nc_subset/new_wrf-processed.nc',
                            type=str,
                            help='Full file path to save directory and save filename')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
