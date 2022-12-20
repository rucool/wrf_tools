#!/usr/bin/env python

"""
Author: Mike Smith
Modified on 8/17/2020 by Lori Garzio
Last modified 12/6/2022
"""

import argparse
import sys
import numpy as np
import os
import glob
import pandas as pd
import xarray as xr
from collections import OrderedDict
from wrf import getvar, interplevel, default_fill


def delete_attr(da):
    """
    Delete these local attributes because they are not necessary
    :param da: DataArray of variable
    :return: DataArray of variable with local attributes removed
    """

    for item in ['projection', 'coordinates', 'MemoryOrder', 'FieldType', 'stagger', 'missing_value']:
        try:
            del da.attrs[item]
        except KeyError:
            continue
    return da


def split_uvm(da, height=None):
    """
    Splits the uvmet variable while dropping extraneous dimensions and renaming variables properly
    :param da: uvmet variable
    :param height: height only
    :return: u, v data arrays
    """
    da = delete_attr(da).drop(['u_v'])

    if height:
        da['height'] = np.int32(height)  # add height variable for sea level: 0m
        da = da.expand_dims('height', axis=1)  # add height dimension to file

    return da[0].rename('u'), da[1].rename('v')


def main(args):
    rdate = args.rundate
    fdir = args.fdir
    sdir = args.sdir
    rtype = args.runtype
    dm = args.domain

    if dm == 'd03':
        domainstr = '1km'
    elif dm == 'd02':
        domainstr = '3km'
    elif dm == 'd01':
        domainstr = '9km'
    else:
        raise (ValueError(f'Incorrect domain specified {dm}. Please choose a valid domain ["d03", "d02", "d01"]'))

    files = sorted(glob.glob(os.path.join(fdir, rtype, rdate) + f'/wrfout_{dm}*'))

    for fname in files:

        splitter = fname.split('/')[-1].split('_')

        # rename the hour 24 file
        if splitter[2].replace('-', '') == rdate:
            save_name = 'wrfproc_{}_{}_00Z_H{:03d}.nc'.format(domainstr,
                                                              splitter[2].replace('-', ''),
                                                              int(splitter[-1].split(':')[0]))
        else:
            save_name = 'wrfproc_{}_{}_00Z_H024.nc'.format(domainstr, rdate)

        save_file = os.path.join(sdir, rtype, rdate, save_name)

        # List of variables that are already included in the WRF output and that we want to compute using the wrf-python
        variables = dict(
            primary=['XLAT', 'XLONG', 'T2', 'SWDOWN', 'LWUPB', 'GLW', 'PSFC', 'RAINC', 'RAINNC', 'RAINSH', 'SNOWNC',
                     'SST', 'DIFFUSE_FRAC', 'LANDMASK', 'LAKEMASK', 'PBLH', 'TSK', 'UST', 'POWER', 'ZNT'],
            computed=['rh2', 'slp', 'mdbz']
        )

        # Generate height table for interpolation of U and V components
        gen_heights = [20, 320, 20]  # minimum height, maximum height, distance between heights

        # Output time units
        time_units = 'seconds since 1970-01-01 00:00:00'

        # Create list of heights between min and max height separated by a stride value defined above
        heights = list(np.arange(gen_heights[0], gen_heights[1], gen_heights[2]))
        heights.append(gen_heights[1])

        # Open using netCDF toolbox
        ncfile = xr.open_dataset(fname)
        original_global_attributes = ncfile.attrs
        ncfile = ncfile._file_obj.ds

        # Load primary variables and append to list
        primary_vars = {}
        for var in variables['primary']:
            try:
                primary_vars[var] = delete_attr(getvar(ncfile, var))
            except ValueError:
                print(f'no {var} in file')

        # Calculate diagnostic variables defined above and add to dictionary
        diagnostic_vars = {}
        for dvar in variables['computed']:
            diagnostic_vars[dvar.upper()] = delete_attr(getvar(ncfile, dvar))

        # Subtract terrain height from height above sea level
        new_z = getvar(ncfile, 'z') - getvar(ncfile, 'ter')

        # Calculate u and v components of wind rotated to Earth coordinates
        uvm = getvar(ncfile, 'uvmet')

        # interpolate u and v components of wind to 0-200m by 10m
        uvtemp = interplevel(uvm, new_z, heights, default_fill(np.float32))
        uvtemp = uvtemp.rename({'level': 'height'})
        utemp, vtemp = split_uvm(uvtemp)

        # Calculate 10m u and v components of wind rotated to Earth coordinates and split into separate variables
        primary_vars['U10'], primary_vars['V10'] = split_uvm(getvar(ncfile, 'uvmet10'))

        # Concatenate the list of calculated u and v values into data array. Append to diagnostic_vars list
        diagnostic_vars['U'] = xr.concat(utemp, dim='height')
        diagnostic_vars['V'] = xr.concat(vtemp, dim='height')

        # Interpolate u and v components of wind to Boundary Layer Heights for wind gust calculation
        uvpblh = interplevel(uvm, new_z, primary_vars['PBLH'])
        uvpblh = delete_attr(uvpblh).drop(['u_v'])  # drop unnecessary attributes
        upblh = uvpblh[0].rename('upblh')
        vpblh = uvpblh[1].rename('vpblh')

        # Calculate wind gust
        sfcwind = np.sqrt(primary_vars['U10'] ** 2 + primary_vars['V10'] ** 2)
        pblwind = np.sqrt(upblh ** 2 + vpblh ** 2)
        delwind = pblwind - sfcwind
        pblval = primary_vars['PBLH'] / 2000
        pblval = pblval.where(pblval < 0.5, other=0.5)  # if the value is less than 0.5, keep it. otherwise, change to 0.5
        delwind = delwind * (1 - pblval)
        gust = sfcwind + delwind
        gust = gust.drop('level')  # drop the 'level' coordinate
        gust = gust.astype(np.float32)

        # Create xarray dataset of primary and diagnostic variables
        ds = xr.Dataset({**primary_vars, **diagnostic_vars})
        ds['U'] = ds.U.astype(np.float32)
        ds['V'] = ds.V.astype(np.float32)
        ds['height'] = ds.height.astype(np.int32)

        try:
            del ds.U.attrs['vert_units']
            del ds.V.attrs['vert_units']
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

        # Set u attributes
        ds['U'].attrs['long_name'] = 'Eastward Wind Component'
        ds['U'].attrs['standard_name'] = 'eastward_wind'
        ds['U'].attrs['short_name'] = 'u'
        ds['U'].attrs['units'] = 'm s-1'
        ds['U'].attrs['description'] = 'earth rotated u'
        ds['U'].attrs['valid_min'] = np.float32(-300)
        ds['U'].attrs['valid_max'] = np.float32(300)

        # Set v attributes
        ds['V'].attrs['long_name'] = 'Northward Wind Component'
        ds['V'].attrs['standard_name'] = 'northward_wind'
        ds['V'].attrs['short_name'] = 'v'
        ds['V'].attrs['units'] = 'm s-1'
        ds['V'].attrs['description'] = 'earth rotated v'
        ds['V'].attrs['valid_min'] = np.float32(-300)
        ds['V'].attrs['valid_max'] = np.float32(300)

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

        # set primary attributes
        ds['GLW'].attrs['standard_name'] = 'surface_downwelling_longwave_flux_in_air'
        ds['GLW'].attrs['long_name'] = 'Surface Downwelling Longwave Flux'

        ds['LWUPB'].attrs['standard_name'] = 'surface_upwelling_longwave_flux'
        ds['LWUPB'].attrs['long_name'] = 'Surface Upwelling Longwave Flux'

        ds['PSFC'].attrs['standard_name'] = 'surface_air_pressure'
        ds['PSFC'].attrs['long_name'] = 'Air Pressure at Surface'

        ds['RH2'].attrs['standard_name'] = 'relative_humidity'
        ds['RH2'].attrs['long_name'] = 'Relative Humidity'

        ds['SLP'].attrs['standard_name'] = 'air_pressure_at_sea_level'
        ds['SLP'].attrs['long_name'] = 'Air Pressure at Sea Level'

        ds['SWDOWN'].attrs['standard_name'] = 'surface_downwelling_shortwave_flux_in_air'
        ds['SWDOWN'].attrs['long_name'] = 'Surface Downwelling Shortwave Flux'

        ds['T2'].attrs['standard_name'] = 'air_temperature'
        ds['T2'].attrs['long_name'] = 'Air Temperature at 2m'

        ds['RAINC'].attrs['long_name'] = 'Accumulated Total Cumulus Precipitation'
        ds['RAINNC'].attrs['long_name'] = 'Accumulated Total Grid Scale Precipitation'
        ds['RAINSH'].attrs['long_name'] = 'Accumulated Shallow Cumulus Precipitation'

        ds['SNOWNC'].attrs['standard_name'] = 'surface_snow_thickness'
        ds['SNOWNC'].attrs['long_name'] = 'Accumulated Total Grid Scale Snow and Ice'
        ds['SNOWNC'].attrs['description'] = '{}; water equivalent'.format(ds['SNOWNC'].description)

        ds['SST'].attrs['standard_name'] = 'sea_surface_temperature'
        ds['SST'].attrs['long_name'] = 'Sea Surface Temperature'

        ds['DIFFUSE_FRAC'].attrs['long_name'] = 'Diffuse Fraction of Surface Shortwave Irradiance'

        ds['MDBZ'].attrs['long_name'] = 'Maximum Radar Reflectivity'

        ds['TSK'].attrs['long_name'] = 'Surface Skin Temperature'

        ds['LANDMASK'].attrs['standard_name'] = 'land_binary_mask'
        ds['LANDMASK'].attrs['long_name'] = 'Land Mask'

        ds['LAKEMASK'].attrs['long_name'] = 'Lake Mask'

        ds['PBLH'].attrs['long_name'] = 'Height of the Top of the Planetary Boundary Layer (PBL)'

        ds['UST'].attrs['long_name'] = 'Friction Velocity'

        ds['XTIME'].attrs['long_name'] = 'minutes since simulation start'

        # add the calculated wind gust to the dataset
        windgust_attrs = dict(long_name='Near Surface Wind Gust',
                              description='Calculated wind gust, computed by mixing down momentum from the level at the '
                                          'top of the planetary boundary layer',
                              units='m s-1')
        ds['WINDGUST'] = xr.Variable(gust.dims, gust.values, attrs=windgust_attrs)
        ds['WINDGUST'] = ds['WINDGUST'].expand_dims('Time', axis=0)

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
            ('history', 'Hourly WRF raw output processed into new hourly file with selected variables.'),
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

        ds.to_netcdf(save_file, encoding=encoding, format='netCDF4', engine='netcdf4', unlimited_dims='Time')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('rundate',
                            type=str,
                            help='Process files for this date, format YYYYmmdd')

    arg_parser.add_argument('-r',
                            dest='runtype',
                            choices=['1km_ctrl', '1km_wf2km'],
                            default='1km_ctrl',
                            type=str,
                            help='Type of run, control or windfarm')

    arg_parser.add_argument('-f',
                            dest='fdir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/wrfout_windturbs',
                            type=str,
                            help='Directory location for WRF windturbine files')

    arg_parser.add_argument('-s',
                            dest='sdir',
                            default='/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed_windturbs',
                            type=str,
                            help='Directory location to save WRF windturbine subset files')

    arg_parser.add_argument('-d', '--domain',
                            dest='domain',
                            choices=['d03', 'd02', 'd01'],
                            default='d03',
                            type=str,
                            help='Domain code, d01=9km, d02=3km, d03=1km')

    parsed_args = arg_parser.parse_args()
    sys.exit(main(parsed_args))
