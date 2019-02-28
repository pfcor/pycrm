# -*- coding: utf-8 -*-

import pandas as pd
from pkg_resources import resource_filename


def load_cdnow_dataset(**kwargs):
    """Loads CDNOW dataset as pd.DataFrame."""

    # selecting between sampled ou full dataset
    if kwargs.pop("sample", False):
        filename = "CDNOW_sample.txt"
    else:
        filename = "CDNOW_master.txt"

    df = pd.read_csv(
        resource_filename('pycrm', f'datasets/{filename}'), 
        names=['customer_id', 'order_date', 'n_items', 'order_value'],
        parse_dates=["order_date"],
        delim_whitespace=True, 
        **kwargs
    )

    return df
