# ===============================================================
# Pollution event detection using median absolute deviation (MAD)
# ===============================================================

# --- Import packages ---
import pandas as pd
import numpy as np
import polars as pl

def pollution_event(input_data : pd.DataFrame,
                    config_dict : dict,
                    verbose : bool = False,
                    window_size : int | None = None
                    ):
    """Identifies and flags anomalous events in a dataset

    This function calculates the median, median absolute deviation (MAD), and modified Z-score for each monitor by hour of day to compare individual monitor
    observations to the monitor's historic observations at each hour of day. Measurements are log transformed for a more normal distribution and calculations
    are done over a user-defined (default is 60 days) rolling window, between 30 days and 365 days, with a minimum of 75% of data required. 

    See examples/pollution_event_demo.py on GitHub for a full working example: https://github.com/edf-org/airinsights/blob/main/examples/pollution_event_demo.py
    
    Parameters
    ----------
    input_data : pd.DataFrame
        A pandas DataFrame containing AQ data
    config_dict : dict
        A dictionary containing input parameter names and values. See 'Other Parameters' for a list
    verbose : bool, default False
        Appends only the modified Z-score and event classification columns to the input data if False. Appends all columns used for computation if True. 
    window_size : int or None, default 60
        Number of days in the rolling window used for calculations. Defaults to 60. Has a minimum of 30 days and a maximum of 365 days.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the site and timestamp columns from the input data with the following columns appended:
                
            **z_score_mod**: modified Z-score of the sensor measurement
            
            **event_type**: classification of event based on modified Z-score. A Z-score greater than 2 is "Unusually high", a Z-score greater than 3 is "Extremely high", and Z-scores below 2 are returned as NULL
                    
        If ``verbose`` is ``True``, also appends:
            
            **hour**: hour of the day
            
            **median_at_hour**: measurement median for hour of day
            
            **days_captured**: number of days with data captured in the rolling window

    Other Parameters
    ----------------
    timestamp_col : str
        Name of the column containing date and time
    datetime_format : str
        Format of the timestamp column usign python datetime syntax::
            For example: 2026-01-01 09:27:11 -> %Y-%m-%d %H:%M:%S 
            For more info, see: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    site_col : str or int
        Name of the column containing unique identifiers for the air sensors
    value_col : str
        Name of the column containing the values for the pollutant of interest
    confidence_col : str, optional
        Name of the column containing confidence values
    confidence_threshold : int or float, optional
        Minimum data confidence level to include measurement in analysis
        
    Notes
    -----
    The function removes values <= 0 to allow for log-transformation of data.
    
    The modified Z-score is calculated as ``abs[(hourly_value - median_value) / 1.4826 * MAD]``, where 1.4826 is a 
    scaling factor used to make MAD comparable to standard deviation. See https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    # --- Read input data as df ---
    df = input_data.copy()

    # --- Parse window_size argument
    if window_size is None:
        window_size = 60
    elif window_size < 30 or window_size > 365:
        raise ValueError("Window size must be between 30 and 365.")
    elif not isinstance(window_size, int):
        raise TypeError(f"integer expected, got {type(window_size).__name__}")

    min_days_in_window = round(window_size * 0.75)

    # --- Add column for hour of day ---
    df["hour"] = df[config_dict['timestamp_col']].dt.hour
    
    # --- Drop NA's and filter for values > 0 ---
    df = df.dropna(subset=[config_dict['site_col'], config_dict['value_col']])  # drop NA from necessary columns
    df = df[df[config_dict['value_col']] > 0]

    # --- Optional; filter out low confidence data ---
    if config_dict.get('confidence_col') and config_dict.get('confidence_threshold') is not None:
        df = df.loc[df[config_dict['confidence_col']] >= config_dict['confidence_threshold']]
    
    # --- Log-transform measurements
    df['value_log'] = np.log(df[config_dict['value_col']]) # previous analysis shows log-transformed measurements more closely match normal distribution; future work could make this optional

    #--- Compute diurnal (hourly) medians and MAD per monitor ---
    #--- Polar package used to optimize for speed over pandas .rolling.agg

    MAD = (
        pl.from_pandas(df).sort(config_dict['timestamp_col'])
        .rolling( 
            index_column=config_dict['timestamp_col'],
            period=f"{window_size}d",
            group_by=[config_dict['site_col'], "hour"]
        ).agg(
            median = pl.col('value_log').median(),
            MAD = (pl.col('value_log') - pl.col('value_log').median()).abs().median(),
            days_captured = pl.len()
        ).filter(pl.col("days_captured") >= min_days_in_window).to_pandas()) # make minimum size a function of the window period (75%)

    # --- Join back to other columns ---
    # --- Compute z-scores (z-score mod for MAD using scalar) and classify event (if >= 3 it is 'extreme', if >= 2 it is 'unusual') ---

    df = df.merge(MAD,on=[config_dict['site_col'],config_dict['timestamp_col'],"hour"],how="left")
    df["z_score_mod"] = (df["value_log"] - df["median"]) / (1.4826 * df["MAD"])
    df["event_type"] = np.select([df["days_captured"].isna(), df["z_score_mod"] >= 3, df["z_score_mod"] >= 2],
              ["Insufficient number of days captured", "Extremely high", "Unusually high"], None)

    # --- Transform median back to concentration space ---
    df["median"] = np.exp(df["median"])
    
    # --- Select columns to return based on verbose argument ---
    if not verbose:
        out = df[[config_dict['site_col'],config_dict['timestamp_col'],'z_score_mod','event_type']]
    else:
        out = df[[config_dict['site_col'],config_dict['timestamp_col'],'z_score_mod','event_type','hour','median','days_captured']].rename(columns={"median":"median_at_hour"})
    
    return(out)