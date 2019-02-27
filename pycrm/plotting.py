import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_transactions_ts(transactional_df, frequency="M", aggregation="n_purchases", reg=False, black_friday_dates=None, plot_black_friday=False, plot_normal_only=False, **kwargs):
    """
    plota a evolucao das compras no tempo
    black_friday_dates:: list of datetime.date
    """

    # preventing unwnated modifications to original df
    transactional_df = transactional_df.copy().rename(columns={"data": "date", "receita": "revenue", "id_cliente": "customer_id"})
    transactional_df = transactional_df[["date", "revenue", "customer_id"] if not 'black_friday' in transactional_df.columns else ["date", "revenue", "customer_id", "black_friday"]]
    transactional_df.index = transactional_df['date']

    # if black friday dates are explicity given, a new column is added to the dataframe flagging the relevant purchases
    if black_friday_dates:
        transactional_df["black_friday"] = transactional_df["date"].dt.date.isin(black_friday_dates).astype(np.int8)

    # level of aggregation
    assert frequency not in ('Y'), "invalid frequency - use plot_transactions_y"
    grouper = transactional_df.resample(frequency)

    # aggregating data
    if aggregation == "n_purchases":
        df = grouper.size().rename(aggregation).to_frame()
    elif aggregation == "revenue":
        df = grouper["revenue"].sum().rename(aggregation).to_frame()
    elif aggregation == "mean_ticket":
        df = grouper["revenue"].mean().rename(aggregation).to_frame()
    elif aggregation == "n_customers":
        df = grouper["customer_id"].nunique().rename(aggregation).to_frame()
    else:
        raise ValueError(f"unknown aggregation {aggregation} - available agregations: n_purchases, revenue, mean_ticket, n_customers")

    
    # for frequency grouping toubleshooting
    # if kwargs.get("troubleshoot_frequency", False):
    df = df.join(grouper["date"].max().rename("date_max")) 
    df = df.join(grouper["date"].min().rename("date_min")) 
    df["n_days"] = (df["date_max"] - df["date_min"]).dt.days + 1
    if kwargs.get("full_intervals_only", False):
        if frequency == "M":
            df = df[df["n_days"] >= kwargs.get("full_interval_m", 28)].copy()
        elif frequency == "W":
            df = df[df["n_days"] >= kwargs.get("full_interval_m", 7)].copy()  
            
    
    if "black_friday" in transactional_df.columns:
        if frequency != 'Y':
            df = df.join(grouper["black_friday"].max())
            
    
    if plot_black_friday or plot_normal_only:
        assert "black_friday" in df.columns, "No Black Friday Information Available"
        
        # n_purchases on normal days
        df[f"{aggregation}_normal"] = df[aggregation]
        df.loc[df["black_friday"] == 1, f"{aggregation}_normal"] = np.nan
        df[f"{aggregation}_normal"] = df[f"{aggregation}_normal"].interpolate(method="linear")
        
        # por plotting reasons, considering "neighbor" rows as black_friday == 1
        try:
            bf_idx = [(i-1, i, i+1) for i in df.reset_index()[df.reset_index()["black_friday"] == 1].index]
            bf_idx = list(set(list(sum(bf_idx, ()))))
            df.iloc[bf_idx, (df.columns == "black_friday").argmax()] = 1
        except IndexError:
            pass
        
        # n_purchases on black friday days
        df[f"{aggregation}_bf"] = df[aggregation]
        df.loc[df["black_friday"] != 1, f"{aggregation}_bf"] = np.nan        
    
    # plot!
    ax = kwargs.get("ax")
    if not ax:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (18,4)))
    
      

    if plot_black_friday:
        (df[f'{aggregation}_normal']).rolling(kwargs.get("rolling_window", 1)).mean().plot(ax=ax, label=kwargs.get("label_normal", "Normal"))
        (df[f'{aggregation}_bf']).rolling(kwargs.get("rolling_window", 1)).mean().plot(ax=ax, label=kwargs.get("label_bf", "Black Friday"))
        
        # simple linear regression - WARNING: simplistic treatment of timeseries data
        if reg:
            f = np.poly1d(np.polyfit(range(df.shape[0]), (df[f'{aggregation}_normal']).values, 1)) 
            df["fitted_line"] = f(np.arange(df.shape[0]))
            df["fitted_line"].plot(ax=ax, lw=2, ls='--', alpha=.5, label="Eq_normal: " + f"{f}".strip())
        
    elif plot_normal_only:
        (df[f'{aggregation}_normal']).rolling(kwargs.get("rolling_window", 1)).mean().plot(ax=ax, label=kwargs.get("label_normal", "Normal"))
        
        # simple linear regression - WARNING: simplistic treatment of timeseries data
        if reg:
            f = np.poly1d(np.polyfit(range(df.shape[0]), (df[f'{aggregation}_normal']).values, 1)) 
            df["fitted_line"] = f(np.arange(df.shape[0]))
            df["fitted_line"].plot(ax=ax, lw=2, ls='--', alpha=.5, label="Eq_normal: " + f"{f}".strip())
        
    else:
        (df[aggregation]).rolling(kwargs.get("rolling_window", 1)).mean().plot(ax=ax, label=kwargs.get("label"))
        
        # simple linear regression - WARNING: simplistic treatment of timeseries data
        if reg:
            f = np.poly1d(np.polyfit(range(df.shape[0]), (df[aggregation]).values, 1)) 
            df["fitted_line"] = f(np.arange(df.shape[0]))
            df["fitted_line"].plot(ax=ax, lw=2, ls='--', alpha=.5, label="Eq_normal: " + f"{f}".strip())

    if kwargs.get("legend", True):
        ax.legend()

    ax.set_title(kwargs.get("title", f"{aggregation.upper()} - {frequency}"), size=kwargs.get("title_size", 14))
    
    ax.set_xlabel(kwargs.get("xlabel",""))

    return ax


def plot_transactions_y(transactional_df, aggregation="n_purchases", reg=False, black_friday_dates=None, plot_black_friday=False, plot_normal_only=False, plot_bf_only=False, **kwargs):
    
    transactional_df = transactional_df.copy().rename(columns={"data": "date", "receita": "revenue", "id_cliente": "customer_id"})
    transactional_df = transactional_df[["date", "revenue", "customer_id"] if not 'black_friday' in transactional_df.columns else ["date", "revenue", "customer_id", "black_friday"]]
    transactional_df.index = transactional_df['date']
    
    # if black friday dates are explicity given, a new column is added to the dataframe flagging the relevant purchases
    if black_friday_dates:
        transactional_df["black_friday"] = transactional_df["date"].dt.date.isin(black_friday_dates).astype(np.int8)
    
    grouper_full   = transactional_df.groupby([transactional_df["date"].dt.year])
    if "black_friday" in transactional_df.columns:
        grouper_normal = transactional_df[transactional_df["black_friday"]!=1].groupby([transactional_df.loc[transactional_df["black_friday"]!=1, "date"].dt.year])
        grouper_bf     = transactional_df[transactional_df["black_friday"]==1].groupby([transactional_df.loc[transactional_df["black_friday"]==1, "date"].dt.year])
    
    
    # aggregating data
    if aggregation == "n_purchases":
        df = grouper_full.size().rename(aggregation).to_frame()
        if "black_friday" in transactional_df.columns:
            df = df.join(grouper_normal.size().rename(f"{aggregation}_normal"))
            df = df.join(grouper_bf.size().rename(f"{aggregation}_bf"))
                     
    elif aggregation == "revenue":
        df = grouper_full["revenue"].sum().rename(aggregation).to_frame()
        if "black_friday" in transactional_df.columns:
            df = df.join(grouper_normal["revenue"].sum().rename(f"{aggregation}_normal"))
            df = df.join(grouper_bf["revenue"].sum().rename(f"{aggregation}_bf"))
                     
    elif aggregation == "mean_ticket":
        df = grouper_full["revenue"].mean().rename(aggregation).to_frame()
        if "black_friday" in transactional_df.columns:
            df = df.join(grouper_normal["revenue"].mean().rename(f"{aggregation}_normal"))
            df = df.join(grouper_bf["revenue"].mean().rename(f"{aggregation}_bf"))
                     
    elif aggregation == "n_customers":
        df = grouper_full["customer_id"].nunique().rename(aggregation).to_frame()
        if "black_friday" in transactional_df.columns:
            df = df.join(grouper_normal["customer_id"].nunique().rename(f"{aggregation}_normal"))
            df = df.join(grouper_bf["customer_id"].nunique().rename(f"{aggregation}_bf"))
                     
    else:
        raise ValueError(f"unknown aggregation {aggregation} - available agregations: n_purchases, revenue, mean_ticket, n_customers")
    
    ax = kwargs.get("ax")
    if not ax:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6,4)))
    
    if plot_black_friday:
        assert "black_friday" in transactional_df.columns, "No BlackFriday Information"
        (df[[f"{aggregation}_normal", f"{aggregation}_bf"]]).plot(ax=ax, kind="bar", rot=0, stacked=kwargs.get("stacked", True))
    elif plot_bf_only:
        assert "black_friday" in transactional_df.columns, "No BlackFriday Information"
        (df[f"{aggregation}_bf"]).plot(ax=ax, kind="bar", rot=0, stacked=kwargs.get("stacked", True))
    elif plot_normal_only:
        assert "black_friday" in transactional_df.columns, "No BlackFriday Information"
        (df[f"{aggregation}_normal"]).plot(ax=ax, kind="bar", rot=0, stacked=kwargs.get("stacked", True))
    else:
        (df[f"{aggregation}"]).plot(ax=ax, kind="bar", rot=0)
    
    ax.set_xlabel(kwargs.get("xlabel",""))

    return ax