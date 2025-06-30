import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_moving_average(df_inp, col_dict, plot_col_name, window, time_col = None, method='simple', raw_color='blue', avg_color='orange', figsize=(15, 8), hue=None):
    """
    Plot raw value time series along with moving average.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    col_name (str): Column name for which to calculate the moving average.
    window (int): Moving window length.
    method (str): Averaging method ('simple' or 'exponential').
    raw_color (str): Color for the raw value line (default is 'blue').
    avg_color (str): Color for the moving average line (default is 'orange').
    figsize (tuple): Figure size (default is (15, 8)).
    hue (str): Column name for hue (default is None).
    """
    plt.figure(figsize=figsize)
    if pd.isnull(time_col):
        time_col = col_dict['time_col']
    print(df_inp.index)
    df = df_inp.dropna(subset = [plot_col_name]).set_index(time_col)
    print(df.index)
    if hue:
        # Plot raw value time series with hue
        sns.lineplot(data=df, x=df.index, y=plot_col_name, hue=hue, palette='tab10', label='Raw Value')
        
        # Calculate and plot moving average with hue
        for category in df[hue].unique():
            category_df = df[df[hue] == category]
            if method == 'simple':
                category_df['moving_avg'] = category_df[plot_col_name].rolling(window=window).mean()
            elif method == 'exponential':
                category_df['moving_avg'] = category_df[plot_col_name].ewm(span=window, adjust=False).mean()
            else:
                raise ValueError("Method must be 'simple' or 'exponential'")
            sns.lineplot(data=category_df, x=category_df.index, y='moving_avg', linestyle='--', label=f'{category} Moving Average window {window}')
    else:
        # Plot raw value time series without hue
        sns.lineplot(data=df, x=df.index, y=plot_col_name, color=raw_color, label='Raw Value')
        
        # Calculate and plot moving average without hue
        if method == 'simple':
            df['moving_avg'] = df[plot_col_name].rolling(window=window).mean()
        elif method == 'exponential':
            df['moving_avg'] = df[plot_col_name].ewm(span=window, adjust=False).mean()
        else:
            raise ValueError("Method must be 'simple' or 'exponential'")
        sns.lineplot(data=df, x=df.index, y='moving_avg', color=avg_color, linestyle='--', label=f'Moving Average window {window}')
    
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# # Example usage
# data = pd.DataFrame({
#     'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
#     'value': range(100),
#     'category': ['A'] * 50 + ['B'] * 50
# })
# data.set_index('timestamp', inplace=True)

# plot_moving_average(data, col_name='value', window=10, method='simple', raw_color='blue', avg_color='orange', figsize=(15, 8), hue='category')