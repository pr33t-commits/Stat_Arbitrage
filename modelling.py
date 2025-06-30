import pandas as pd
import numpy as np

def calculate_theoretical_futures_price(data,price_cols, col_dict):
    '''
    Calculate the theoretical futures price for the given data.
    '''
    
    data[[f"{col}_theoretical" for col in price_cols]] = data[price_cols].mul(np.exp((data[col_dict['rfr_col']] - data[col_dict['div_yield_col']])*data['time_to_expiry']),axis=0)
    data[[f"{col}_price_diff" for col in price_cols]] = data[[f"{col}_theoretical" for col in price_cols]].values - data[price_cols].values
    
    return data

def test_arbitage_opp(data, col_dict, test_column='near'):
    '''
    expects the input data in pivot format of having all 3 expiry 
    info. in one row for a particular time 
    '''
    
    def get_expiry_price(group, expiry_type):
        expiry_prices = {}
        post_3pm = group.loc[(group[col_dict['time_col']] >= pd.to_datetime(f"{group.name} 15:00:00"))]
        expiry_prices[f"expiry_price_300_{expiry_type}"] = post_3pm.sort_values(col_dict['time_col'])[f"{col_dict['close_price_col']}_far"].head(1)
        expiry_prices[f"expiry_price_330_{expiry_type}"] = post_3pm.sort_values(col_dict['time_col'])[f"{col_dict['close_price_col']}_far"].tail(1)
        expiry_prices[f"expiry_price_315_{expiry_type}"] = group.loc[(group[col_dict['time_col']] >= pd.to_datetime(f"{group.name} 15:15:00"))].sort_values(col_dict['time_col'])[f"{col_dict['close_price_col']}_far"].head(1)
        return pd.Series(expiry_prices)
    
    data_expiry = data.groupby(f"{col_dict['expiry_col']}_far").apply(lambda g : get_expiry_price(g, 'far'))
    data_expiry = data_expiry.groupby(f"{col_dict['expiry_col']}_mid").apply(lambda g : get_expiry_price(g, 'mid'))
    data_expiry = data_expiry.groupby(f"{col_dict['expiry_col']}_near").apply(lambda g : get_expiry_price(g, 'near'))
    data_expiry["action"] = "long"
    data_expiry.loc[data_expiry[f"{col_dict['close_price_col']}_price_diff_{test_column}"]<0, "action"] = "short"
    data_expiry["P/L"] = data_expiry[f"{col_dict["close_price_col"]}"]
    data_expiry.loc[data_expiry["action"]=="long"]
    
    