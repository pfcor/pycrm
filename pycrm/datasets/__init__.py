# -*- coding: utf-8 -*-
# modified from https://github.com/CamDavidsonPilon/lifetimes/

import pandas as pd
from pkg_resources import resource_filename

def load_dataset(filename, **kwargs):
    """
    Load a dataset from lifetimes.datasets.
    
    Parameters
    ----------
    filename: str
        for example "larynx.csv"
    **kwargs
        Passed to pandas.read_csv function.
    Returns
    -------
    DataFrame
    """
    return pd.read_csv(resource_filename('lifetimes', 'datasets/' + filename), **kwargs)

def load_cdnow_dataset(**kwargs):
    """Loads cdnow data set as pandas DataFrame."""

    filename = "CDNOW_master.txt"
    df = pd.read_csv(
        resource_filename('pycrm', f'datasets/{filename}'), 
        names=['customer_id', 'date', 'items', 'monetary_value'],
        parse_dates=["date"],
        delim_whitespace=True, 
        **kwargs
    )
    # df.columns = ['customer_id', 'date', 'items', 'monetary_value']
    # df
    return df