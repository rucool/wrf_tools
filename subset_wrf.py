from wrf import getvar, interplevel
import xarray as xr
import numpy as np
import pandas as pd
from collections import OrderedDict


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


def main(fname, save_file, variables, heights, time_units):

    # Create list of heights between min and max height separated by a stride value defined above
    heights = list(np.arange(heights[0], heights[1], heights[2]))
    heights.append(heights[1])

    # Open using netCDF toolbox
    # ncfile = Dataset(wrf_file) # open using ncar netCDF toolbox
    ncfile = xr.open_dataset(fname)
    original_global_attributes = ncfile.attrs
    ncfile = ncfile._file_obj.ds

    # Load primary variables and append to list
    primary_vars = {}
    for var in variables['primary']:
        primary_vars[var] = delete_attr(getvar(ncfile, var))

    # Calculate diagnostic variables defined above and add to dictionary
    diagnostic_vars = {}
    for dvar in variables['computed']:
        diagnostic_vars[dvar.upper()] = delete_attr(getvar(ncfile, dvar))

    # Subtract terrain height from height above sea level
    new_z = getvar(ncfile, 'z') - getvar(ncfile, 'ter')

    # Calculate u and v components of wind rotated to Earth coordinates
    uvm = getvar(ncfile, 'uvmet')

    # Initialize lists for u and v arrays. We will append each interpolated height to this lest and then concatenate this list with xarray
    u_arrays = []
    v_arrays = []

    # interpolate u and v components of wind to 0-200m by 10m
    for x in heights:
        uvtemp = interplevel(uvm, new_z, x)
        utemp, vtemp = split_uvm(uvtemp, x)
        u_arrays.append(utemp)
        v_arrays.append(vtemp)

    # Calculate 10m u and v components of wind rotated to Earth coordinates and split into separate variables
    primary_vars['U10'], primary_vars['V10'] = split_uvm(getvar(ncfile, 'uvmet10'))

    # Concatenate the list of calculated u and v values into data array. Append to diagnostic_vars list
    diagnostic_vars['U'] = xr.concat(u_arrays, dim='height')
    diagnostic_vars['V'] = xr.concat(v_arrays, dim='height')

    # Create xarray dataset of primary and diagnostic variables
    ds = xr.Dataset({**primary_vars, **diagnostic_vars})
    # ds['Time'] = ds['Time'][0]
    ds['Times'] = np.array([pd.Timestamp(ds.Time.data).strftime('%Y-%m-%d_%H:%M:%S')]).astype('<S19')
    ds = ds.expand_dims('Time', axis=0)

    # Add description and units for lon, lat dimensions for georeferencing
    ds['XLAT'].attrs['description'] = 'latitude'
    ds['XLAT'].attrs['units'] = 'degree_north'
    ds['XLONG'].attrs['description'] = 'longitude'
    ds['XLONG'].attrs['units'] = 'degree_east'

    # Set time attribute
    ds['Time'].attrs['standard_name'] = 'time'

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
    ds['U'].attrs['valid_min'] = np.float32(-300)
    ds['U'].attrs['valid_max'] = np.float32(300)
    # ds['u'].attrs['coordinates'] = 'lon lat'
    # ds['u'].attrs['grid_mapping'] = 'crs'

    # Set v attributes
    ds['V'].attrs['long_name'] = 'Northward Wind Component'
    ds['V'].attrs['standard_name'] = 'northward_wind'
    ds['V'].attrs['short_name'] = 'v'
    ds['V'].attrs['valid_min'] = np.float32(-300)
    ds['V'].attrs['valid_max'] = np.float32(300)

    # Set u10 attributes
    ds['U10'].attrs['long_name'] = 'Eastward Wind Component - 10m'
    ds['U10'].attrs['standard_name'] = 'eastward_wind'
    ds['U10'].attrs['short_name'] = 'u'
    ds['U10'].attrs['valid_min'] = np.float32(-300)
    ds['U10'].attrs['valid_max'] = np.float32(300)
    # ds['u'].attrs['coordinates'] = 'lon lat'
    # ds['u'].attrs['grid_mapping'] = 'crs'

    # Set v10 attributes
    ds['V10'].attrs['long_name'] = 'Northward Wind Component - 10m'
    ds['V10'].attrs['standard_name'] = 'northward_wind'
    ds['V10'].attrs['short_name'] = 'v'
    ds['V10'].attrs['valid_min'] = np.float32(-300)
    ds['V10'].attrs['valid_max'] = np.float32(300)
    # ds['v'].attrs['coordinates'] = 'lon lat'
    # ds['v'].attrs['grid_mapping'] = 'crs'

    # set primary attributes
    ds['GLW'].attrs['standard_name'] = 'surface_downwelling_longwave_flux_in_air'
    ds['GLW'].attrs['long_name'] = 'Surface Downwelling Longwave Flux'

    ds['LWUPB'].attrs['standard_name'] = 'surface_upwelling_longwave_flux'
    ds['LWUPB'].attrs['long_name'] = 'Surface Upwelling LongwaveFlux'

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
    ds['XTIME'].attrs['long_name'] = 'minutes since simulation start'

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
        ('history', 'Hourly WRF data combined into one hourly file.'),
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

    # # Set crs attributes
    # ds['u'].attrs['grid_mapping'] = 'crs'
    # ds['v'].attrs['grid_mapping'] = 'crs'
    # kwargs = dict(crs=None)
    # ds = ds.assign(**kwargs)
    # ds['crs'].attrs['grid_mapping_name'] = 'latitude_longitude'
    # ds['crs'].attrs['inverse_flattening'] = 298.257223563
    # ds['crs'].attrs['long_name'] = 'Coordinate Reference System'
    # ds['crs'].attrs['semi_major_axis'] = '6378137.0'
    # ds['crs'].attrs['epsg_code'] = 'EPSG:4326'
    # ds['crs'].attrs['comment'] = 'http://www.opengis.net/def/crs/EPSG/0/4326'

    encoding = {}
    # Add compression to all variables
    for k in ds.data_vars:
        encoding[k] = {'zlib': True, 'complevel': 4}

    # add the encoding for time so xarray exports the proper time. Also remove compression from dimensions. They should never have fill values
    encoding['Time'] = dict(units=time_units, calendar='gregorian', zlib=False, _FillValue=False, dtype=np.double)
    encoding['XLONG'] = dict(zlib=False, _FillValue=False)
    encoding['XLAT'] = dict(zlib=False, _FillValue=False)
    encoding['height'] = dict(zlib=False, _FillValue=False)

    ds.to_netcdf(save_file, encoding=encoding, format='netCDF4', engine='netcdf4', unlimited_dims='Time')


if __name__ == '__main__':
    wrf_file = 'data/nc_raw/wrfout_d01_2018-11-19_000000.nc'  # raw wrf ncfile
    save_name = '/Users/mikesmith/Documents/projects/wrf/new_wrf-processed.nc'  # filename for saving

    # List of variables that already included in the WRF output and that we want to compute using the wrf-python toolbox
    variables = dict(primary=['XLAT', 'XLONG', 'T2', 'SWDOWN', 'LWUPB', 'GLW', 'PSFC', 'RAINC', 'RAINNC', 'RAINSH'], computed=['rh2', 'slp'])

    # Generate height table for interpolation of U and V components
    uv_heights = [30, 250, 10]  # minimum height, maximum height, distance between heights

    main(wrf_file, save_name, variables, uv_heights, 'seconds since 1970-01-01 00:00:00')
