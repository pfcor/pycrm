import datetime as dt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_transaction_timeseries(transaction_series, reg=False, **kwargs):
    """
    Plots a timeseries (pd.Series or one column pd.DataFrame with DatetimeIndex), plot
    """
       
    # input validation 
    if isinstance(transaction_series, pd.DataFrame):
        assert transaction_series.shape[1] == 1, "only single column pandas.DataFrame supported"
        transaction_series = transaction_series.copy()
    elif isinstance(transaction_series, pd.Series):
        transaction_series = transaction_series.copy().to_frame()
    else:
        raise TypeError(f"invalid transaction_series input: {type(transaction_series)} - pandas.Series or single column pandas.DataFrame supported only")
    assert isinstance(transaction_series.index, pd.DatetimeIndex), "index not of pandas.DatetimeIndex type"
    
    rolling_window = kwargs.get("rolling_window", 1)
    assert rolling_window > 0, "window must be positive"
    assert isinstance(rolling_window, int), "window must be an integer"  

    # plotting vars
    plot_kind = kwargs.get("plot_kind")
    if not plot_kind:
        if transaction_series.shape[0] <= 10:
            plot_kind = "bar"
        else:
            plot_kind = "line"
    
    figsize = kwargs.get("figsize")
    if not figsize:
        if transaction_series.shape[0] <= 10:
            figsize = (6,4)
        else:
            figsize = (18,4)
    
    # creating figure    
    ax = kwargs.get("ax")
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    # plotting series
    (transaction_series.iloc[:, 0]).rolling(rolling_window).mean().plot(ax=ax, kind=plot_kind, label=kwargs.get("label"), rot=0 if plot_kind=="bar" else 90)
    if plot_kind == "bar":
        # label values
        label_values_buffer = kwargs.get("label_values_buffer", kwargs.get("label_values_offset", 0.03)*(transaction_series.iloc[:, 0]).max())
        # fmt = kwargs.get("label_values_format")
        if kwargs.get("label_values", False):
            ax.set_ylim(0, (transaction_series.iloc[:, 0]).max()*1.121)
            for i, v in enumerate((transaction_series.iloc[:, 0]).values):

                f_mlt = kwargs.get("f_mlt", 1)
                f_fmt = kwargs.get("f_fmt", ".0f")
                f_pref = kwargs.get("f_pref", "")
                f_suff = kwargs.get("f_suff", "")

                v_fmt = f"{f_pref}{v*f_mlt:{f_fmt}}{f_suff}"

                ax.text(i, v+label_values_buffer, v_fmt, ha="center", size=kwargs.get("label_values_size", 14))
        
        ax.set_xticklabels([dt.datetime.strftime(d, "%Y-%m-%d") for d in transaction_series.index])

        # growth line
        if kwargs.get("plot_growth_line", False):
            ax.plot(range((transaction_series.iloc[:, 0]).shape[0]), (transaction_series.iloc[:, 0]).values, color="r", lw=4, marker=".", alpha=0.35)
        
            if kwargs.get("label_growth_line", True):
                vals_growth = pd.concat([(transaction_series.iloc[:, 0]).rename("val"), (transaction_series.iloc[:, 0]).pct_change().rename("pct_change"), (transaction_series.iloc[:, 0]).diff().rename("diff")], axis=1)
                for i, (_, row) in enumerate(vals_growth.iloc[1:].iterrows()):
                    ax.text(i+.5, (row["val"]-(row["diff"]/2))+label_values_buffer, f"{row['pct_change']:.0%}", ha="center", size=kwargs.get("label_values_size", 14), color="r")



    # perform (and plot) single linear regression on series
    # WARNING: simplistic treatment of timeseries data - should not be used in serious analysis
    if reg:
        f = np.poly1d(np.polyfit(range(transaction_series.shape[0]), (transaction_series.iloc[:, 0]).values, 1)) 
        transaction_series["fitted_line"] = f(np.arange(transaction_series.shape[0]))
        transaction_series["fitted_line"].plot(ax=ax, lw=2, ls='--', alpha=.5, label="Eq_normal: " + f"{f}".strip())
    
    # plot details shortcut
    if kwargs.get("legend", False):
        ax.legend()
    ax.set_title(kwargs.get("title", f"{transaction_series.columns[0].upper()}"), size=kwargs.get("title_size", 14))
    ax.set_xlabel(kwargs.get("xlabel", ""))
    ax.set_ylabel(kwargs.get("ylabel", ""))
    if "xlim" in kwargs:
        ax.set_xlim(kwargs.get("xlim"))
    if "ylim" in kwargs:
        ax.set_ylim(kwargs.get("ylim"))

    return ax
    

def plot_user_retention_matrix(user_retention_matrix, ax=None):
    
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title('Cohorts: User Retention')
        
    ax = sns.heatmap(user_retention_matrix.T, mask=user_retention_matrix.T.isnull(), annot=True, fmt='.0%')


#####legacy

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

    if kwargs.get("legend", False):
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
        vals = (df[f"{aggregation}_bf"]).copy()
        vals.plot(ax=ax, kind="bar", rot=0, stacked=kwargs.get("stacked", True))
    elif plot_normal_only:
        assert "black_friday" in transactional_df.columns, "No BlackFriday Information"
        vals = (df[f"{aggregation}_normal"]).copy()
        vals.plot(ax=ax, kind="bar", rot=0, stacked=kwargs.get("stacked", True))
    else:
        vals = (df[f"{aggregation}"]).copy() 
        vals.plot(ax=ax, kind="bar", rot=0)
    
    label_values_buffer = kwargs.get("label_values_buffer", kwargs.get("label_values_offset", 0.03)*vals.max())
    fmt = kwargs.get("label_values_format")
    if kwargs.get("label_values", False):
        for i, v in enumerate(vals.values):
            if fmt == "m_money":
                v_fmt = f"R$ {v/1e6:.0f}M"
            elif fmt == "k":
                v_fmt = f"{v/1e3:.0f}k"
            else:
                v_fmt = f"{v:.0f}"
            ax.text(i, v+label_values_buffer, v_fmt, ha="center", size=kwargs.get("label_values_size", 14))

    if kwargs.get("plot_growth_line", False):
        ax.plot(range(vals.shape[0]), vals.values, color="r", lw=4, marker=".", alpha=0.5)
    
        if kwargs.get("label_growth_line", False):

            vals_growth = pd.concat([vals.rename("val"), vals.pct_change().rename("pct_change"), vals.diff().rename("diff")], axis=1)
            for i, (_, row) in enumerate(vals_growth.iloc[1:].iterrows()):
                ax.text(i+.5, (row["val"]-(row["diff"]/2))+label_values_buffer, f"{row['pct_change']:.0%}", ha="center", size=kwargs.get("label_values_size", 14), color="r")



    ax.set_xlabel(kwargs.get("xlabel",""))

    if kwargs.get("legend", False):
        ax.legend()

    ax.set_title(kwargs.get("title", f"{aggregation.upper()}"), size=kwargs.get("title_size", 14))
    
    ax.set_xlabel(kwargs.get("xlabel",""))

    return ax