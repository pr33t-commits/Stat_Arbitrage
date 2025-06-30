import datetime
import calendar
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import pandas_market_calendars as mcal

def last_thursday(date):
    '''
    Returns the last Thursday of the month for the given date.
    Requires date to be a datetime.date or TimeStamp object.
    '''
    year = date.year
    month = date.month
    last_day = calendar.monthrange(year, month)[1]
    last_date = datetime.date(year, month, last_day)
    last_thursday = last_date - datetime.timedelta(days=(last_date.weekday() - 3) % 7)
    return last_thursday

def adjust_for_holiday_expiries(expiry_date, working_days):
    '''
    Ensure that the expiry date is a working day
    expiry_date: expiry date to be adjusted
    working_days: list of working days
    '''
    while pd.Timestamp(expiry_date) not in working_days:
        
        expiry_date -= relativedelta(days=1)#datetime.timedelta(days=1)
    return expiry_date


def get_expiry_dates(date,working_days):
    ''' 
    Returns the expiry dates for the near, mid and far month contracts for the given date.
    '''
    near_month = adjust_for_holiday_expiries(last_thursday(date),working_days)
    # print(near_month)
    mid_month = adjust_for_holiday_expiries(last_thursday(near_month + relativedelta(months=1)),working_days)
    # print(mid_month)
    far_month = adjust_for_holiday_expiries(last_thursday(near_month + relativedelta(months=2)),working_days)
    # print(far_month)
    return [near_month, mid_month,far_month]

def calculate_returns(df, col_dict):
    
    '''
    Calculate the returns for the given data.
    '''
    data = df.copy(deep=True)
    data['returns'] = data[col_dict['close_price_col']].pct_change()
    data['log_returns'] = np.log(data[col_dict['close_price_col']]/data[col_dict['close_price_col']].shift(1))
    data['overnight_returns'] = (data[col_dict['open_price_col']]/data[col_dict['close_price_col']].shift(1)) - 1
    data['overnight_log_returns'] = np.log(data[col_dict['open_price_col']]/data[col_dict['close_price_col']].shift(1))
    data['intraday_returns'] = data[col_dict['close_price_col']]/data[col_dict['open_price_col']] - 1
    data['intraday_log_returns'] = np.log(data[col_dict['close_price_col']]/data[col_dict['open_price_col']])

    return data

def resample_ohlcv(ohlcv_data, col_dict, groupby_col, freq='1H', rename_columns=False):
    
    '''
    Resample the OHLCV data to the given frequency.
    groupby_col:- identifier for a unique instrument in the dataframe
    '''
    
    agg_funcs = {col_dict['open_price_col']:'first', 
                 col_dict['high_price_col']:'max', 
                 col_dict['low_price_col']:'min', 
                 col_dict['close_price_col']:'last', 
                 col_dict['volume_col']:'sum',
                 col_dict['oi_col']: 'mean'}
    
    # Filter only columns that exist in the data
    agg_funcs_filt = {key:value for key,value in agg_funcs.items() if key in ohlcv_data.columns}
    agg_funcs_final = {**agg_funcs_filt,**{key:'first' for key in groupby_col}} if groupby_col else agg_funcs_filt
    
    grouped = ohlcv_data.groupby(groupby_col) if groupby_col else [('', ohlcv_data)]
    
    # Resample and aggregate
    resampled_data = []
    for _, group in grouped:
        resampled = group.set_index(col_dict['time_col']).resample(freq, origin="start_day").agg(agg_funcs_final).reset_index()
        
        resampled[col_dict['date_col']] = pd.to_datetime(resampled[col_dict['time_col']].dt.date)
        
        ## remve weekend and post closing time activity
        resampled = resampled.loc[(resampled[col_dict['date_col']].dt.weekday < 5) &
                                    (resampled[col_dict['time_col']].dt.time <= datetime.time(15,30))]
        resampled = resampled.dropna(subset=[col_dict['open_price_col'], col_dict['close_price_col']])
        
        resampled_data.append(resampled)
    
    # Concatenate the results
    result = pd.concat(resampled_data, ignore_index=True)
    
    # Rename columns if requested
    if rename_columns:
        rename_dict = {column:f"{column}_{freq}" for column in result.columns 
                      if column not in [col_dict['date_col'], col_dict['time_col']] + groupby_col}
        result.rename(columns=rename_dict, inplace=True)
    
    return result

def continuous_compounder(returns, current_compounding = 1):
    '''
    transforms the returns to a continuous compounding rate
    Assumes returns are in ratio form i.e. 0.01 for 1%
    Assumes returns are yearly 
    '''
    return current_compounding*np.log(1 + returns/current_compounding)


def is_working_day(date):
    """
    Check if a given date is a working day in the Indian stock market (NSE).
    date (str or pd.Timestamp): The date to check in 'YYYY-MM-DD' format or as a pd.Timestamp.
    Returns: bool
    """
    # Convert the date to a pd.Timestamp if it's a string
    if isinstance(date, str):
        date = pd.Timestamp(date)
    nse = mcal.get_calendar('NSE')
    schedule = nse.schedule(start_date=date.strftime('%Y-01-01'), end_date=date.strftime('%Y-12-31'))
    return date in schedule.index
