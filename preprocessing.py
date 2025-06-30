from utils import get_expiry_dates
import pandas as pd
import numpy as np

from download import adjust_for_post_expiry_dates
def map_expiries_to_dates(group,working_days,col_dict):
    
    ## Not all expiries get trading volume at each timepoint. Far mnths can have missing rows for some minutes
    ## Avoided the route of infering near,mid,far based on difference between dates because thresholds to qualify as near vs mid OR mid vs far 
    ## depends on number of days in the month. Thresholds need to be modulated. 
    
    # TO DO: Highly inefficient method. need to change later. not used recurringly hence fine for now
    near,mid,far = get_expiry_dates(adjust_for_post_expiry_dates((group.name).date()),working_days)
    
    return group[col_dict['expiry_col']].dt.date.replace({near:'near',mid:'mid', far:'far'})
