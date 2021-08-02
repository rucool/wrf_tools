#! /usr/bin/env python

"""
Author: Lori Garzio on 7/30/2021
Last modified: 7/30/2021
"""

import numpy as np


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
