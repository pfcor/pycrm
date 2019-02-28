import pandas as pd



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
        raise ValueError(f"unknown aggregation {aggregation} - available agregations: (n_purchases, frequency), revenue, (mean_ticket, aov), n_customers")

    # border intervals could be smaller than the rest - option to avoid that
    if full_intervals_only:
        df = df.join(grouper["order_date"].max().rename("order_date_max")) 
        df = df.join(grouper["order_date"].min().rename("order_date_min")) 
        df["n_days_interval"] = (df["order_date_max"] - df["order_date_min"]).dt.days + 1

        df = df[df.iloc[:, -1] >= df.iloc[1:-1, -1].min()]

        df = df.drop(["order_date_max", "order_date_min", "n_days_interval"], axis=1)

    return df
    