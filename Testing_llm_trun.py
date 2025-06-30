# @title
import pandas as pd
import json # Assuming JSON for action signals
import datetime

col_dict = {'open_price_col': 'open', 'high_price_col': 'high', 'low_price_col': 'low', 'close_price_col': 'close', 'volume_col': 'volume', 
            'time_col':'datetime', 'expiry_col':'expiry_date', 'expiry_type_col':'expiry_type',
            'oi_col':'open_interest','date_col':'date','div_yield_col':'Div Yield %',
            'rfr_col':'MIBOR','returns_col':'returns'}

class FnOBacktester:
    
    """
    A class to backtest Futures and Options (FnO) trading strategies,
    considering margin requirements.
    
    """
    
    SOD_TIME = datetime.time(9, 15)
    EOD_ACTION_TIME = datetime.time(15, 29) # Action time to close EOD positions

    def __init__(self, initial_capital: float, ohlcv_data: pd.DataFrame, margin_rules: dict):
        
        """
        Initializes the backtester state.
        Args:
            initial_capital (float): The starting capital for the backtest.
            ohlcv_data (pd.DataFrame): DataFrame containing OHLCV data indexed by timestamp.
            margin_rules (dict): A dictionary defining margin calculation rules.
                                  (Structure depends on broker/exchange specifics)
        """
        
        # --- State Variables ---
        self.available_capital: float = initial_capital
        self.margin_maintained: dict = {} # e.g., {'NIFTY_FUT': 20000}
        # self.total_margin_maintained: float = 0.0
        self.current_position: dict = {} # e.g., {'NIFTY_FUT': {'type': 'long', 'qty': 10, 'entry_price': 18000}, 'NIFTY_CE_18100': {'type': 'short', 'qty': 10, 'entry_price': 50}}
        self.proposed_position: dict = {} # e.g., {'NIFTY_FUT': {'type': 'long', 'qty': 10, 'entry_price': 18000}, 'NIFTY_CE_18100': {'type': 'short', 'qty': 10, 'entry_price': 50}}
        self.holdings_value: float = 0.0 # Current market value of the position
        self.pnl_history: list = [] # List to store PnL at each step
        self.action_history: list = [] # List to store signals and actions taken
        self.execution_history: list = []
        self.previous_candle = dict = {} 
        # --- Input Data & Rules ---
        self.ohlcv_data: pd.DataFrame = ohlcv_data
        self.expiry_mapper: dict = {}
        self.margin_rules: dict = margin_rules
        self.trained_model = None # Placeholder for the trading logic/model
        self.col_dict = col_dict # Column name standardized
        # --- Internal Tracking ---
        self.current_timestep_index: int = 0 # To iterate through ohlcv_data
        self.current_timestamp = None
        self.prev_timestamp = None
    
    def _is_sod(self, timestamp: pd.Timestamp) -> bool:
        """Checks if the given timestamp is Start of Day (SOD) for action."""
        return timestamp.time() == self.SOD_TIME

    def _is_eod_action_time(self, timestamp: pd.Timestamp) -> bool:
        """Checks if the given timestamp is End of Day (EOD) action time."""
        return timestamp.time() == self.EOD_ACTION_TIME

    def preprocess(self):
        
        data = self.ohlcv_data
        
        ## Makes unique ID for each Future contract and saves all instruments available to trade at each timepoint
        data['futures_id'] = 'FUT_'+ data[col_dict['expiry_col']].astype(str)
        avail_instruments = data.groupby(col_dict['date_col'])['futures_id'].unique().rename(columns = {'futures_id':'instruments'})
        avail_instruments['instruments'] = avail_instruments['instruments'].apply(lambda x : list(x) + ['equity'])
        self.available_instruments = avail_instruments['instruments']
        
        # expiry_mapper is a nested dictionary with keys as dates and the values as another dict with keys as 'near','mid','far' and corresponding expiry dates
        # Eg. {'2025-01-01':{'near':'2025-01-30','mid':'2025-02-27', 'far':'2025-03-27'}}
        retrieve_expiry_dates_df = data.copy()
        retrieve_expiry_dates_df[col_dict['expiry_col']] = retrieve_expiry_dates_df[col_dict['expiry_col']].astype(str)
        self.expiry_mapper = retrieve_expiry_dates_df.groupby(col_dict['date_col']).apply(lambda g:g[[col_dict['expiry_col'], 
                                                                                                      col_dict['expiry_type_col']]].drop_duplicates()/n
                                                                                          .set_index(col_dict['expiry_type_col']).to_dict()[col_dict['expiry_col']]).to_dict()
    
    def check_margin_threshold(self, position_to_evaluate, pnl_step_total: float, prices_for_checking) -> bool:
        """
        Checks if the margin maintained meets the required threshold after accounting for PnL.

        Args:
            pnl_current_step (float): Profit/Loss incurred in the current step.

        Returns:
            bool: True if margin requirements are met, False otherwise.
        """
        # --- Variable Flow ---
        # Input: self.margin_maintained, pnl_current_step, self.current_position, self.margin_rules
        # Output: Boolean
        print(f"Checking margin. Current: {self.margin_maintained}, PnL: {pnl_step_total}")
        #current_prices = self.ohlcv_data.iloc[self.current_timestep_index] # Assuming 'close' prices for check
        required_margin = self._get_required_margin(position_to_evaluate = position_to_evaluate, 
                                                    current_prices = prices_for_checking)
        effective_margin = self.margin_maintained + pnl_step_total

        print(f"Required margin: {required_margin}, Effective margin: {effective_margin}")

        # --- Placeholder Logic ---
        margin_ok = effective_margin >= required_margin
        return margin_ok, required_margin


    def _calculate_current_holdings_value(self, current_prices: dict) -> float:
        
        """
        Calculates the current market value of the open positions.
        (Internal helper method)

        Args:
            current_prices (dict): Dictionary of current prices for instruments in the position.
                                   e.g., {'NIFTY_FUT': 18050, 'NIFTY_CE_18100': 45}

        Returns:
            float: The total current market value of the holdings.
        """
        # --- Internal Logic Placeholder ---
        # Iterate through self.current_position, use current_prices to value each leg.
        # Sum up the values.
        print(f"Calculating holdings value based on: {current_prices}")
        calculated_value = 0.0
        # ... placeholder logic ...
        self.holdings_value = calculated_value # Update state
        return calculated_value

    def _calculate_pnl(self, positions_to_evaluate, pre_previous_prices: dict, previous_prices: dict, current_prices = None) -> float:
        
        """
        Calculates the profit or loss from the last step to the current step for each individual position in the protfolio.
        (Internal helper method)

        Args:
            previous_prices (dict): Prices at the end of the last timestep.
            current_prices (dict): Prices at the end of the current timestep.

        Returns:
            dict: Instrument: PnL incurred during the timestep.
        """        
        
        print(f"Calculating PnL from {pre_previous_prices} to {previous_prices}")
        pnl = {}
        for instrument,details in positions_to_evaluate.items():
            if not details['exit_idx']:    ## If the position is being squared off in this timestep or not 
                if self.current_timestep_index - 1 == details['entry_idx']: ## If the position has opened in that step and PnL being calculated at the end
                    prev_price_col = f"{col_dict['open_price_col']}_{instrument}"
                    curr_price_col = f"{col_dict['close_price_col']}_{instrument}"
                    pre_prices = previous_prices
                    post_prices = previous_prices
                else:
                    prev_price_col = f"{col_dict['close_price_col']}_{instrument}"
                    curr_price_col = f"{col_dict['close_price_col']}_{instrument}"
                    pre_prices = pre_previous_prices
                    post_prices = previous_prices
            else:
                prev_price_col = f"{col_dict['close_price_col']}_{instrument}"
                curr_price_col = f"{col_dict['open_price_col']}_{instrument}"
                pre_prices = previous_prices
                post_prices = current_prices
            price_diff = (post_prices[f"{curr_price_col}_{instrument}"] - pre_prices[f"{prev_price_col}_{instrument}"]) if details['type'] == 'long' else (pre_prices[f"{prev_price_col}_{instrument}"] - post_prices[f"{curr_price_col}_{instrument}"]) 
            pnl[instrument] = details['qty']*price_diff

        return pnl

    def _get_required_margin(self, position_to_evaluate: dict, current_prices: dict, price_col: str) -> float:
        
        """
        Calculates the total margin required for a given position based on rules.
        (Internal helper method)

        Args:
            position_to_evaluate (dict): The position for which margin is needed.
            current_prices (dict): Current market prices for margin calculation.

        Returns:
            float: The total required margin.
        """
        
        print(f"Calculating required margin for position: {position_to_evaluate} at prices: {current_prices}")
        
        required_margin = 0.0
        
        for instrument,details in position_to_evaluate.items():
            inst = [i for i in self.margin_rules.keys() if i in instrument][0]
            if details['stage'] == 'maintenance':
                required_margin += self.margin_rules[inst]['maintenance']*details['qty'] * current_prices[f"{price_col}_{instrument}"]
            elif details['stage'] == 'initial':
                required_margin += self.margin_rules[inst]['initial']*details['qty'] * current_prices[f"{price_col}_{instrument}"]
            elif details['stage'] == 'margin_call':
                required_margin += self.margin_rules[inst]['margin_call']*details['qty'] * current_prices[f"{price_col}_{instrument}"]
        return required_margin

    def generate_action_signal(self, historical_data_slice: pd.DataFrame) -> dict:
        """
        Wrapper for the actual signal generation logic.
        Calls _generate_action_signal if defined, else raises NotImplementedError.
        """
        if hasattr(self, '_generate_action_signal') and callable(getattr(self, '_generate_action_signal')):
            action_signals = self._generate_action_signal(historical_data_slice)
            # Optionally update class properties here if needed
            return action_signals
        else:
            raise NotImplementedError("You must implement _generate_action_signal in the child class.")

    def finalize_action(self, action_signals: dict) -> dict:
        
        """
        Wrapper for the actual action finalization logic.
        Calls _finalize_action if defined, else raises NotImplementedError.
        """
        
        if hasattr(self, '_finalize_action') and callable(getattr(self, '_finalize_action')):
            final_actions = self._finalize_action(action_signals)
            
            # Store the signal and the finalized action for analysis
            self.action_history.append({
                'timestamp': self.ohlcv_data.index[self.current_timestep_index],
                'signal': action_signals,
                'final_action': final_actions
            })
            
            return final_actions
        else:
            raise NotImplementedError("You must implement _finalize_action in the child class.")
    
    def apply_actions_to_position(self,
                                  current_position: dict,
                                  actions: dict,
                                  previous_prices: dict,
                                  current_prices: dict,
                                  margin_rules : dict = {},
                                  available_capital: float = None,
                                  update_capital: bool = True,) -> tuple:
        """
        Applies a set of actions to a given position and returns the updated position and optionally updated capital.
        Optionally updates pnl_history if provided.

        Args:
            current_position (dict): The current position dictionary (will not be mutated).
            actions (dict): Actions to apply, e.g. {'NIFTY_FUT': {'action': 'long', 'qty': 2}, ...}
            margin_rules (dict): Margin rules for instruments.
            prices (dict): Prices for entry/exit, e.g. {'NIFTY_FUT_open': 18000, ...}
            timestep_index (int): Current timestep index.
            pnl_history (list, optional): If provided, will append realized PnL.
            available_capital (float, optional): If provided and update_capital is True, will update and return.
            update_capital (bool): Whether to update and return available_capital.

        Returns:
            updated_position (dict), updated_capital (float, optional)
        """        
        
        import copy
        updated_position = copy.deepcopy(current_position)
        updated_capital = available_capital
        pnl_exit_positions = []
        timestep_index = self.current_timestep_index
        
        if not len(margin_rules) > 0:
            margin_rules = self.margin_rules
        
        for instrument, details in actions.items():
            action = details['action']
            qty = details['qty']
            inst = [i for i in margin_rules.keys() if i in instrument][0]
            price = current_prices.get(f"{instrument}_open", 0)

            if action == 'long':
                if instrument in updated_position:
                    updated_position[instrument]['qty'] += qty
                else:
                    updated_position[instrument] = {
                        'type': 'long', 'qty': qty, 'entry_price': price,
                        'entry_idx': timestep_index, 'exit_idx': None,
                        'stage': 'initial',
                        'initial_margin': qty * price * margin_rules[inst]['initial']
                    }
                    if update_capital and updated_capital is not None:
                        updated_capital -= updated_position[instrument]['initial_margin']
            elif action == 'short':
                if instrument in updated_position:
                    updated_position[instrument]['qty'] += qty
                else:
                    updated_position[instrument] = {
                        'type': 'short', 'qty': qty, 'entry_price': price,
                        'entry_idx': timestep_index, 'exit_idx': None,
                        'stage': 'initial',
                        'initial_margin': qty * price * margin_rules[inst]['initial']
                    }
                    if update_capital and updated_capital is not None:
                        updated_capital -= updated_position[instrument]['initial_margin']
            elif action == 'square-off':
                if instrument in updated_position:
                    if updated_position[instrument]['qty'] == qty:
                        updated_position[instrument]['exit_idx'] = timestep_index
                        
                        pnl_exit = self._calculate_pnl(positions_to_evaluate = { instrument : updated_position[instrument]},
                                                                 pre_previous_prices = {},
                                                                 previous_prices = previous_prices,
                                                                 current_prices = current_prices)
                        pnl_exit_positions.append(pnl_exit)
                        pnl_hist_sum = sum(list({pnl_dict['timestamp']:pnl_dict['pnl'] for pnl_dict in self.pnl_history if pnl_dict['instrument']==instrument}.values()))
                        pnl_sum = pnl_hist_sum + pnl_exit[instrument]
                        
                        # PnL calculation is not included here; you can add a callback or pass a function if needed
                        if update_capital and updated_capital is not None:
                            updated_capital += updated_position[instrument]['initial_margin']
                        del updated_position[instrument]
                    else:
                        updated_position[instrument]['qty'] -= qty
            
        # req_margin_after_step = self._get_required_margin(position_to_evaluate=updated_position, current_prices=current_prices, price_col=col_dict['open_price_col'],)                
                
        return updated_position, updated_capital, pnl_exit_positions
    
    def save_state_history(self, finalized_actions, pnl_last_step, pnl_exit_positions):
        
        # self.available_capital = updated_capital
        # self.current_position = updated_position
        
        for instrument, details in self.current_position.items():
            if details['stage'] == 'initial':
                self.current_position[instrument]['stage'] = 'maintenance'
        
        # 4. Recalculate holdings value based on the *new* position and current prices
        # current_prices = self.ohlcv_data.iloc[self.current_timestep_index].to_dict() 
        # self._calculate_current_holdings_value(current_prices)

        # Store the signal and the finalized action for analysis
        for instrument, details in finalized_actions:
            self.execution_history.append({
                'timestamp': self.current_timestamp,
                'instrument': instrument,
                
                'final_action': details['action']
            })

        # 5. Append pnl_last_step to self.pnl_history.
        for instrument, pnl in pnl_last_step.items():
            self.pnl_history.append({
                'timestamp': self.prev_timestamp,
                'instrument': instrument,
                'pnl':pnl
            })
        for pnl_exit in pnl_exit_positions:
            for instrument, pnl in pnl_exit.items():
                self.pnl_history.append({
                    'timestamp': self.ohlcv_data.index[self.current_timestep_index],
                    'instrument': instrument,
                    'pnl': pnl
                })
        print(f"State updated. Capital: {self.available_capital}, Margin: {self.margin_maintained}")
        
    def check_state(self, capital, margin, position_to_evaluate, current_prices, checkpoint_indicator):

        if checkpoint_indicator == 'EOP':
            price_col = col_dict['close_price_col']
        elif checkpoint_indicator == 'SOP':
            price_col = col_dict['open_price_col']
        else:
            raise NotImplementedError(f"Invalid checkpoint_indicator : {checkpoint_indicator}, Not implemented")
        
        required_margin = self._get_required_margin(position_to_evaluate=position_to_evaluate, 
                                  current_prices=current_prices, price_col = price_col)
        
        if capital < 0:
            print('Negative Capital encountered')
            return {'valid':False, 'action':'square-off'}
        elif (margin >= required_margin):
            return {'valid':True}
        elif (margin < required_margin):
            if checkpoint_indicator == 'EOP':
                for instrument,details in position_to_evaluate.items():
                    details['stage'] = 'margin_call'
                required_margincall_margin = self._get_required_margin(position_to_evaluate=position_to_evaluate, 
                                    current_prices=current_prices, price_col = price_col)
                remark = 'margin call'
                if (capital - (required_margincall_margin - margin)  >= 0):
                    return {'valid':False, 'action':'refill margin', 'remark':f'{remark} :- Sufficient funds','updated_state_vars':{'capital':capital - (required_margincall_margin - margin),
                                                                                                            'margin':required_margincall_margin}}
                else:
                    return {'valid':False, 'action':'square-off', 'remark':f'{remark} :- Insufficient funds'}
            else:
                remark = 'Position Change'
                if (capital - (required_margin - margin)  >= 0):
                    return {'valid':False, 'action':'refill margin', 'remark':f'{remark} :- Sufficient funds','updated_state_vars':{'capital':capital - (required_margin - margin),
                                                                                                            'margin':required_margin}}
                else:
                    return {'valid':False, 'action':None, 'remark':f'{remark} :- Insufficient funds'}
        else:
            print(capital, margin, required_margin)
            raise NotImplementedError("Some condition missed")

    def update_state(self, 
                     finalized_actions: dict, 
                     pnl_last_step: float,
                     previous_prices:dict,
                     current_prices: dict):
        
        """
        Updates the backtester's state variables based on the executed actions.

        Args:
            finalized_actions (dict): The actions (with quantities) that were executed.
            pnl_last_step (float): PnL calculated for the last step *before* executing new trades.
        """
        
        print(f"Updating state based on actions: {finalized_actions}")

        updated_position, updated_capital, pnl_exit_positions = self.apply_actions_to_position(current_position=self.current_position, 
                                                                                                           actions=finalized_actions,
                                                                                                           previous_prices=previous_prices,
                                                                                                           current_prices=current_prices,
                                                                                                           available_capital=self.available_capital)
        
        # for pnl_exit in pnl_exit_positions:
        #     for instrument, pnl in pnl_exit.items():
        #         self.pnl_history.append({
        #             'timestamp': self.ohlcv_data.index[self.current_timestep_index],
        #             'instrument': instrument,
        #             'pnl': pnl
        #         })
        
        return updated_position, updated_capital, pnl_exit_positions
        

    def run_backtest(self):
        
        """
        Runs the backtesting loop over the historical data. It is assumed that all this computation occurs after the end of a timestep
        and before start of next.
        """
        
        print("Preprocessing")
        self.preprocess()

        print("Starting backtest loop...")
        # Iterate through the data, leaving one step for final PnL/update
        for i in range(len(self.ohlcv_data) - 1):
            self.current_timestep_index = i
            self.current_timestamp = self.ohlcv_data.index[i]
            self.prev_timestamp = self.ohlcv_data.index[i-1] if i > 0 else self.current_timestamp # The timestep which ended
            print(f"\n--- Timestep: {self.current_timestamp} (Index: {i}) ---")
            
            
            ## Input ohlcv data has columns for each timepoint as 'high_near','high_mid','high_far','open_near'...etc. Following lines 
            ## rename those columns to f'high_FUT_{expiry_date}' where expiry_date is the corresponfing near/mid/far exp date
            ## columns corresponding to equities remain the same i.e. 'high_equity','low_equity'..etc.
            rename_cols = [col_dict['open_price_col'],col_dict['high_price_col'], col_dict['low_price_col'], col_dict['close_price_col'],
                           col_dict['returns_col']]
            current_prices = self.ohlcv_data.iloc[i].rename(columns = {f'{col}_{exp}':f'{col}_FUT_{self.expiry_mapper[self.current_timestamp][exp]}' 
                                                                       for col in rename_cols 
                                                                       for exp in ['near','mid','far']}).to_dict() # Timestep about to start

            previous_prices = self.ohlcv_data.iloc[i-1].rename(columns = {f'{col}_{exp}':f'{col}_FUT_{self.expiry_mapper[self.prev_timestamp][exp]}' 
                                                                       for col in rename_cols 
                                                                       for exp in ['near','mid','far']}).to_dict() if i > 0 else current_prices # The timestep which ended

            # 1. Calculate PnL from the change in position value during the last step
            pnl_step = self._calculate_pnl(positions_to_evaluate=self.current_position,
                                           pre_previous_prices=self.previous_candle,
                                           previous_prices=previous_prices)
            
            pnl_step_total = sum(list(pnl_step.values()))
            print(f"PnL for step: {pnl_step_total}")
            
            self.margin_maintained += pnl_step_total

            state_check_eop = self.check_state(capital = self.available_capital,
                                           margin = self.margin_maintained,
                                           position_to_evaluate=self.current_position,
                                           current_prices=previous_prices,
                                           checkpoint_indicator = 'EOP')
            
            if not state_check_eop['valid']:
                if state_check_eop['action'] == 'sqaure-off':
                    finalized_actions = {} # Build square-off actions here
                    for inst, details in self.current_position.items():
                        finalized_actions[inst] = {'action': 'square-off', 'qty': details['qty']}
                    self.update_state(finalized_actions=finalized_actions, 
                                      previous_prices = previous_prices,
                                      current_prices = current_prices)
                    continue
                else:
                    self.available_capital = state_check_eop['updated_state_vars']['capital']
                    self.margin_maintained = state_check_eop['updated_state_vars']['margin']
            
            # 3. If margin is okay, generate trading signal
            print("Margin OK.")
            # Define the slice of data to pass to the signal generator
            historical_data_slice = self.ohlcv_data.iloc[:i] # Data up to  current step
            action_signals = self.generate_action_signal(historical_data_slice)

            # 4. Finalize actions based on signals and constraints
            finalized_actions = self.finalize_action(action_signals)
            self.previous_candle = previous_prices
            # 5. Update state based on finalized actions
            # Note: PnL calculated earlier is passed to update state correctly
            
            proposed_position, __ ,proposed_pnl = self.update_state(finalized_actions, pnl_step, margin_req_post_action)

            state_check_sop = self.check_state(capital = self.available_capital,
                                               margin = self.margin_maintained,
                                               position_to_evaluate = proposed_position,
                                               current_prices=current_prices,
                                               checkpoint_indicator = 'SOP')
            
            if not state_check_sop['valid']:
                if state_check_sop['action']:
                    self.available_capital = state_check_sop['updated_state_vars']['capital']
                    self.margin_maintained = state_check_sop['updated_state_vars']['margin']
                    self.current_position = proposed_position
                    self.save_state_history()
                else:
                    # insufficient margin and funds for the prop action, do nothing
                    finalized_actions = {}
                self.save_state_history(finalized_actions=finalized_actions, 
                                        pnl_last_step=pnl_step, 
                                        pnl_exit_positions=[])
            else:
                ## no change to margin and capital since current margin amount covers total req amount
                self.current_position = proposed_position
                self.save_state_history(finalized_actions, pnl_last_step=pnl_step, pnl_exit_positions=proposed_pnl)
            
        print("\n--- Backtest Finished ---")
        # Final calculations (e.g., total PnL, Sharpe ratio) can be done here
        # using self.pnl_history, self.action_history etc.

class NaiveStrategyBacktester(FnOBacktester):
    """
    Implements a naive trading strategy for 1-minute data:
    - Longs a future when it starts trading, closes at expiry.
    - Equity Shorting (NSE Times: 09:15 SOD, 15:29 EOD action time):
        - At 09:15 (SOD):
            - Closes any existing equity short position.
            - Opens a new equity short position, quantity based on active long futures.
        - At 15:29 (EOD action time):
            - Closes the equity short position opened at SOD.
    """

    SOD_TIME = datetime.time(9, 15)
    EOD_ACTION_TIME = datetime.time(15, 29) # Action time to close EOD positions

    def __init__(self, initial_capital: float, ohlcv_data: pd.DataFrame, margin_rules: dict,
                 future_contract_qty: int = 1, equity_qty_per_future_contract: int = 100):
        
        """
        Initializes the NaiveStrategyBacktester.
        Args:
            initial_capital (float): Starting capital.
            ohlcv_data (pd.DataFrame): OHLCV data with 1-minute frequency, DatetimeIndex.
            margin_rules (dict): Margin rules.
            future_contract_qty (int): Quantity for each future position.
            equity_qty_per_future_contract (int): Equity quantity per active long future.
        """
        
        super().__init__(initial_capital, ohlcv_data, margin_rules)
        self.future_contract_qty = future_contract_qty
        self.equity_qty_per_future_contract = equity_qty_per_future_contract
        # Parent's run_backtest calls self.preprocess()

    def _is_sod(self, timestamp: pd.Timestamp) -> bool:
        """Checks if the given timestamp is Start of Day (SOD) for action."""
        return timestamp.time() == self.SOD_TIME

    def _is_eod_action_time(self, timestamp: pd.Timestamp) -> bool:
        """Checks if the given timestamp is End of Day (EOD) action time."""
        return timestamp.time() == self.EOD_ACTION_TIME

    def _generate_action_signal(self, historical_data_slice: pd.DataFrame) -> dict:
        """
        Generates trading signals for the current timestep (1-minute frequency).
        """
        action_signals = {}
        
        if self.current_timestamp is None:
            print("Warning (_generate_action_signal): self.current_timestamp is not set.")
            return {}
            
        current_pd_timestamp = pd.to_datetime(self.current_timestamp)
        current_date = current_pd_timestamp.date()
        current_date_str = current_pd_timestamp.strftime('%Y-%m-%d')
        
        equity_instrument_id = 'equity'

        # --- Futures Logic (runs every minute, but actions are date-driven) ---
        # 1. Square off expiring futures
        futures_in_position = {
            inst: det for inst, det in self.current_position.items() if "FUT_" in inst
        }
        for instrument_id, details in futures_in_position.items():
            try:
                expiry_date_str = instrument_id.split('_', 1)[1]
                expiry_date = pd.to_datetime(expiry_date_str).date()
                if (current_date == expiry_date) & (self._is_eod_action_time(current_pd_timestamp)):
                    if instrument_id not in action_signals:
                        action_signals[instrument_id] = {'action': 'square-off'}
            except (IndexError, ValueError) as e:
                print(f"Warning (_generate_action_signal): Error parsing future expiry '{instrument_id}': {e}")
                continue
        
        # 2. Long new futures
        available_instrument_ids_today = []
        if current_date_str in self.available_instruments.index:
            instrument_list = self.available_instruments[current_date_str]
            if isinstance(instrument_list, list):
                 available_instrument_ids_today = instrument_list
            else: # Parent preprocesses it into a list
                print(f"Warning (_generate_action_signal): Expected list for {current_date_str}, got {type(instrument_list)}")

        current_futures_available = [inst_id for inst_id in available_instrument_ids_today if "FUT_" in inst_id]
        for fut_id in current_futures_available:
            if fut_id in action_signals: continue # Already decided (e.g., square-off)
            try:
                expiry_date_str = fut_id.split('_', 1)[1]
                expiry_date = pd.to_datetime(expiry_date_str).date()
                if current_date >= expiry_date: continue # Don't long if expiring today
            except (IndexError, ValueError):
                print(f"Warning (_generate_action_signal): Error parsing potential future long '{fut_id}'")
                continue
            if fut_id not in self.current_position:
                action_signals[fut_id] = {'action': 'long'}

        # --- Equity Logic (Time-Specific) ---
        if self._is_sod(current_pd_timestamp):
            # At SOD (09:15 AM)
            # 1. Signal to close any existing equity short position
            if equity_instrument_id in self.current_position and \
               self.current_position[equity_instrument_id].get('type') == 'short':
                action_signals[equity_instrument_id + "_sod_close_existing"] = {
                    'action': 'square-off',
                    'instrument_id_override': equity_instrument_id}
            # 2. Signal to open a new equity short position
            action_signals[equity_instrument_id + "_sod_open_new"] = {
                'action': 'short',
                'instrument_id_override': equity_instrument_id
            }
        elif self._is_eod_action_time(current_pd_timestamp):
            # At EOD Action Time (e.g., 15:29 PM)
            # 1. Signal to close the equity short position opened at SOD
            if equity_instrument_id in self.current_position and \
               self.current_position[equity_instrument_id].get('type') == 'short':
                # We assume this short was opened at SOD of the current day.
                # More robust check: ensure entry_idx corresponds to current day's SOD if needed.
                action_signals[equity_instrument_id + "_eod_close_current"] = {
                    'action': 'square-off',
                    'instrument_id_override': equity_instrument_id
                }
        
        return action_signals

    def _finalize_actions(self, action_signals: dict) -> dict:
        """
        Finalizes actions based on signals and strategy rules (quantities).
        Updates self.proposed_position for margin checking.
        """
        final_actions_for_update_state = {}
        # self.proposed_position = {k: v.copy() for k, v in self.current_position.items()}
        processed_signal_keys = set()
        equity_instrument_id = 'equity'

        def get_actual_id(key, details):
            return details.get('instrument_id_override', key)

        # Order of Processing:
        # 1. Equity SOD Close Existing Short
        # 2. Future Square-offs (expiring)
        # 3. Future Longs (newly starting)
        # 4. Equity SOD Open New Short (quantity depends on active futures post future actions)
        # 5. Equity EOD Close Current Day's Short

        # 1. Process Equity SOD Close Existing Short
        for key, details in list(action_signals.items()):
            if "_sod_close_existing" in key:
                actual_id = get_actual_id(key, details)
                if details['action'] == 'square-off' and \
                    actual_id == equity_instrument_id:
                    # actual_id in self.proposed_position and \
                    # self.proposed_position[actual_id].get('type') == 'short':
                        final_actions_for_update_state[actual_id] = {
                            'action': 'square-off',
                            'qty': self.current_position[actual_id]['qty']
                        }
                        del self.proposed_position[actual_id]
                processed_signal_keys.add(key)

        # 2. Process Future Square-Off Signals
        for key, details in list(action_signals.items()):
            if key in processed_signal_keys: continue
            actual_id = get_actual_id(key, details)
            if "FUT_" in actual_id and details['action'] == 'square-off':
                # if actual_id in self.proposed_position:
                final_actions_for_update_state[actual_id] = {
                    'action': 'square-off',
                    'qty': self.current_position[actual_id]['qty']
                }
                del self.proposed_position[actual_id]
                processed_signal_keys.add(key)

        # 3. Process Future Long Signals
        for key, details in list(action_signals.items()):
            if key in processed_signal_keys: continue
            actual_id = get_actual_id(key, details)
            if "FUT_" in actual_id and details['action'] == 'long':
                if actual_id not in self.proposed_position:
                    qty_to_long = self.future_contract_qty
                    final_actions_for_update_state[actual_id] = {'action': 'long', 'qty': qty_to_long}
                    self.proposed_position[actual_id] = {
                        'type': 'long', 'qty': qty_to_long, 'entry_price': 0, 'stage': 'initial'
                    }
                processed_signal_keys.add(key)

        # 4. Process Equity SOD Open New Short
        open_new_equity_short_signal_key = equity_instrument_id + "_sod_open_new"
        if open_new_equity_short_signal_key in action_signals and \
           open_new_equity_short_signal_key not in processed_signal_keys:
            
            num_long_future_instruments = 0
            for inst_id, details_prop in self.proposed_position.items(): # Check proposed futures
                if "FUT_" in inst_id and details_prop.get('type') == 'long':
                    num_long_future_instruments += 1 

            if num_long_future_instruments > 0:
                qty_to_short_equity = num_long_future_instruments * self.equity_qty_per_future_contract
                if qty_to_short_equity > 0:
                    final_actions_for_update_state[equity_instrument_id] = {
                        'action': 'short', 'qty': qty_to_short_equity
                    }
                    self.proposed_position[equity_instrument_id] = {
                        'type': 'short', 'qty': qty_to_short_equity, 'entry_price': 0, 'stage': 'initial'
                    }
            # If num_long_future_instruments is 0, no new equity short is opened.
            # If equity_instrument_id was in final_actions_for_update_state from a previous step (e.g. sod_close_existing)
            # and then we decide to short, it will overwrite. This is fine.
            processed_signal_keys.add(open_new_equity_short_signal_key)

        # 5. Process Equity EOD Close Current Day's Short
        close_current_equity_short_signal_key = equity_instrument_id + "_eod_close_current"
        if close_current_equity_short_signal_key in action_signals and \
           close_current_equity_short_signal_key not in processed_signal_keys:
            details = action_signals[close_current_equity_short_signal_key]
            actual_id = get_actual_id(close_current_equity_short_signal_key, details)

            if details['action'] == 'square-off' and \
               actual_id == equity_instrument_id and \
               actual_id in self.proposed_position and \
               self.proposed_position[actual_id].get('type') == 'short':
                
                final_actions_for_update_state[actual_id] = {
                    'action': 'square-off',
                    'qty': self.proposed_position[actual_id]['qty']
                }
                del self.proposed_position[actual_id]
            processed_signal_keys.add(close_current_equity_short_signal_key)
                
        return final_actions_for_update_state

