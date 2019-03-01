import pandas as pd
import numpy as np


def aggregate_transactions_time(df_transactions, aggregation="n_purchases", freq="M", full_intervals_only=False):
    """
    aggregates transaction data on time, by agreggator with inputed frequency
    """

    # housekeeping
    assert "order_date" in df_transactions.columns, "order_date column not found"
    assert "customer_id" in df_transactions.columns, "customer_id column not found"
    df_transactions = df_transactions.copy()
    df_transactions.index = df_transactions['order_date']

    # frequency aggregation
    grouper = df_transactions.resample(freq)

    # aggregating data
    if aggregation in ["grouper"]:
        return grouper

    if aggregation in ["n_purchases", "frequency"]:
        df = grouper.size().rename(aggregation).to_frame()

    elif aggregation in ["revenue", "total_order_value"]:
        assert "order_value" in df_transactions.columns, "order_value column not found"
        df = grouper["order_value"].sum().rename(aggregation).to_frame()

    elif aggregation in ["mean_ticket", "aov", "avg_ticket", "avg_order_value"]:
        assert "order_value" in df_transactions.columns, "order_value column not found"
        df = grouper["order_value"].mean().rename(aggregation).to_frame()

    elif aggregation in ["n_customers"]:
        df = grouper["customer_id"].nunique().rename(aggregation).to_frame()

    elif aggregation in ["avg_basket_size", "basket_size", "n_items"]:
        assert "n_items" in df_transactions.columns, f"n_items column not found"
        df = grouper["n_items"].mean().rename(aggregation).to_frame()

    else:
        raise ValueError(f"unknown aggregation {aggregation} - available agregations: n_purchases, revenue, mean_ticket, aov, n_customers, basket_size")

    # border intervals could be smaller than the rest - option to avoid that
    if full_intervals_only:
        df = df.join(grouper["order_date"].max().rename("order_date_max")) 
        df = df.join(grouper["order_date"].min().rename("order_date_min")) 
        df["n_days_interval"] = (df["order_date_max"] - df["order_date_min"]).dt.days + 1

        df = df[df.iloc[:, -1] >= df.iloc[1:-1, -1].min()]

        df = df.drop(["order_date_max", "order_date_min", "n_days_interval"], axis=1)

    return df


### COHORT ###
def cohort_period(df):
    """
    Creates a `cohort_period` column, which is the Nth period based on the user's first purchase.
    """
    df = df.copy()
    df['cohort_period'] = np.arange(df.shape[0]) + 1
    return df


def user_retention_matrix(df_transactions, cohort_frequency="M", aggregation="n_customers", normalize=True):
    
    df_transactions.copy()
    
    # identifying cohort_group
    df_transactions = df_transactions.set_index('customer_id')
    if cohort_frequency=="M":
        df_transactions['cohort_group'] = df_transactions.groupby(level=0)['order_date'].min().apply(lambda x: x.strftime('%Y-%m'))
    elif cohort_frequency=="Q":
        df_transactions['cohort_group'] = df_transactions.groupby(level=0)['order_date'].min().apply(lambda x: f"{x.year}-Q{(x.month//3+1)}")
    elif cohort_frequency=="S":
        df_transactions['cohort_group'] = df_transactions.groupby(level=0)['order_date'].min().apply(lambda x: f"{x.year}-S{((x.month-1)//6+1)}")
    elif cohort_frequency=="Y":
        df_transactions['cohort_group'] = df_transactions.groupby(level=0)['order_date'].min().apply(lambda x: f"{x.year}")
    df_transactions = df_transactions.reset_index()
    
    # grouping orders
    if cohort_frequency=="M":
        df_transactions['order_period'] = df_transactions['order_date'].apply(lambda x: x.strftime('%Y-%m'))
    elif cohort_frequency=="Q":
        df_transactions['order_period'] = df_transactions['order_date'].apply(lambda x: f"{x.year}-Q{(x.month//3+1)}")
    elif cohort_frequency=="S":
        df_transactions['order_period'] = df_transactions['order_date'].apply(lambda x: f"{x.year}-S{((x.month-1)//6+1)}")
    elif cohort_frequency=="Y":
        df_transactions['order_period'] = df_transactions['order_date'].dt.year.astype(str)

    grouped = df_transactions.groupby(['cohort_group', "order_period"])

    # aggregates data, calculating total customers, total orders and revenue
    cohorts = (
        grouped.agg(
            {
                'customer_id': pd.Series.nunique,
                'order_id': pd.Series.nunique,
                'order_value': np.sum
            }
        )
    ).rename(
        columns={
            'customer_id': 'n_customers',
            'order_id': 'n_orders',
            'order_value': 'revenue'
        }
    )
    
    
    cohorts = cohorts.groupby(level=0).apply(cohort_period)
    cohorts = cohorts.reset_index()
    cohorts = cohorts.set_index(['cohort_group', 'cohort_period'])
    
    cohorts[aggregation].unstack(0)
    
    user_retention = cohorts[aggregation].unstack(0)
    if normalize:
        cohort_group_size = cohorts[aggregation].groupby(level=0).first()
        user_retention = user_retention / (cohort_group_size)
    
    return user_retention
