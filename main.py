import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def parse_current_state_log(log_path: str) -> pd.DataFrame:
    """
    Reads current_state.log, extracts the timestamp and
    the symbol->position dictionary (1 for long, -1 for short).
    
    Returns a DataFrame with columns:
      - timestamp (datetime)
      - positions (dict of symbol->int)
    """
    pattern = re.compile(r"'([^']+)':(-?\d+)")
    rows = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Expected line format:
                # 2024/05/29 12:30:55.000000:'ASTRUSDT':1, 'KLAYUSDT':-1
                dt_str = line[:24]
                rest = line[24:].strip()
                
                dt = datetime.strptime(dt_str, "%Y/%m/%d %H:%M:%S.%f")
                pairs = pattern.findall(rest)
                positions_dict = {sym: int(pos) for sym, pos in pairs}
                
                rows.append({
                    "timestamp": dt,
                    "positions": positions_dict
                })
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {log_path}")
        raise e

    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def extract_position_events(df_log: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns: [timestamp, positions],
    returns a DataFrame listing each open/close event with columns:
      symbol, timestamp, action, old_position, new_position.
      
    A dummy row is inserted at the start so that any symbols in the first
    real row are treated as newly opened.

    This version also merges rows that share the same timestamp by
    combining their positions dictionaries (the last update for a symbol wins).
    """
    # Work on a copy and sort by timestamp.
    df_log = df_log.copy()
    df_log.sort_values("timestamp", inplace=True)
    df_log.reset_index(drop=True, inplace=True)
    
    # Merge rows with identical timestamps.
    merged_rows = []
    for ts, group in df_log.groupby("timestamp"):
        merged_positions = {}
        # Update the merged dictionary with each positions dict in order.
        for pos_dict in group["positions"]:
            merged_positions.update(pos_dict)
        merged_rows.append({"timestamp": ts, "positions": merged_positions})
    df_log = pd.DataFrame(merged_rows)
    df_log.sort_values("timestamp", inplace=True)
    df_log.reset_index(drop=True, inplace=True)
    
    # Insert a dummy row at the start so that the first real row's positions are treated as opens.
    if not df_log.empty:
        first_ts = df_log.loc[0, "timestamp"]
        dummy_ts = first_ts - pd.Timedelta(microseconds=1)
        dummy_row = pd.DataFrame({"timestamp": [dummy_ts], "positions": [{}]})
        df_log = pd.concat([dummy_row, df_log], ignore_index=True)
    
    events = []
    for i in range(len(df_log) - 1):
        curr_ts = df_log.loc[i, "timestamp"]
        next_ts = df_log.loc[i + 1, "timestamp"]
        curr_pos = df_log.loc[i, "positions"]
        next_pos = df_log.loc[i + 1, "positions"]
        
        # Consider all symbols seen in either the current or next row.
        all_symbols = set(curr_pos.keys()).union(next_pos.keys())
        
        for sym in all_symbols:
            old_p = curr_pos.get(sym, 0)
            new_p = next_pos.get(sym, 0)
            if old_p != new_p:
                # If opening a position.
                if old_p == 0 and new_p != 0:
                    events.append({
                        "symbol": sym,
                        "timestamp": next_ts,
                        "action": "open",
                        "old_position": old_p,
                        "new_position": new_p
                    })
                # If closing a position.
                elif old_p != 0 and new_p == 0:
                    events.append({
                        "symbol": sym,
                        "timestamp": next_ts,
                        "action": "close",
                        "old_position": old_p,
                        "new_position": new_p
                    })
                else:
                    # For a flip (e.g. from long to short or vice versa),
                    # generate a close then an open event.
                    events.append({
                        "symbol": sym,
                        "timestamp": next_ts,
                        "action": "close",
                        "old_position": old_p,
                        "new_position": 0
                    })
                    events.append({
                        "symbol": sym,
                        "timestamp": next_ts,
                        "action": "open",
                        "old_position": 0,
                        "new_position": new_p
                    })
                    
    events_df = pd.DataFrame(events)
    events_df.sort_values("timestamp", inplace=True)
    events_df.reset_index(drop=True, inplace=True)
    return events_df



def extract_trading_moments(csv_path: str) -> pd.DataFrame:
    """
    Loads the 'pnl_real' CSV file with columns:
      [timestamp_index, coin, pnl_real, amount_in, exit_spread]
    and returns only rows where exit_spread != 0.0.
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=[0])
    df_filtered = df[df["exit_spread"] != 0.0]
    return df_filtered


def load_aum(path: str) -> pd.DataFrame:
    """
    Loads a semicolon-separated file with no header.
    Expected format:
      2025-01-22T04:22:33.967903+00:00;2538.66763626;2538.66763626;0
    Drops fractional seconds from the timestamp and returns a sorted DataFrame.
    """
    df = pd.read_csv(path, header=None, sep=';')
    df.columns = ["raw_timestamp", "free_cash", "total_cash", "unrealized_pnl"]
    
    df["raw_timestamp"] = df["raw_timestamp"].str.replace(r"\.\d+", "", regex=True)
    df["timestamp"] = pd.to_datetime(
        df["raw_timestamp"],
        format="%Y-%m-%dT%H:%M:%S%z", 
        errors="coerce"
    )
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    df.drop(columns=["raw_timestamp"], inplace=True)
    
    return df


def load_price_data(prices_path: str) -> pd.DataFrame:
    """
    Loads price data from a CSV file that contains 3-minute OHLCV data.
    Expects the CSV file to have:
      - an index column with datetime stamps,
      - columns in the format: PAIR_open, PAIR_high, PAIR_low, PAIR_close, PAIR_volume.
      
    The function sorts the DataFrame by its datetime index and drops any timezone info.
    """
    df_prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    df_prices.sort_index(inplace=True)
    # Remove any timezone information (make the index naive)
    df_prices.index = df_prices.index.tz_localize(None)
    return df_prices



def get_next_bar_price(symbol: str, event_ts: pd.Timestamp, df_prices: pd.DataFrame) -> float:
    """
    Returns the next available bar's open price for the given symbol after event_ts.
    If the first next bar has a NaN price, it will check subsequent bars until a valid price is found.
    If no valid price is found, returns np.nan.
    """
    col = f"{symbol}_open"  # Change to f"{symbol}_close" if you want to use the close price instead.
    idx_array = df_prices.index
    pos = idx_array.searchsorted(event_ts, side="right")
    
    # Iterate over bars starting at 'pos'
    while pos < len(idx_array):
        row = df_prices.iloc[pos]
        price = row.get(col, np.nan)
        if not pd.isna(price):
            return price
        pos += 1  # Try the next bar if the current bar's price is NaN
        
    return np.nan


def build_trades_from_events(df_events: pd.DataFrame, df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Converts open/close events to completed trades with the following columns:
      symbol, entry_time, entry_price, exit_time, exit_price, side, realized_pnl_pcg.
    
    Realized PnL is computed in percentage (e.g., for longs: (exit_price/entry_price) - 1).
    """
    open_positions = {}
    completed_trades = []

    for _, row in df_events.iterrows():
        symbol = row["symbol"]
        event_ts = row["timestamp"]
        action = row["action"]
        old_pos = row["old_position"]
        new_pos = row["new_position"]

        if action == "open":
            side = new_pos  # +1 or -1
            entry_price = get_next_bar_price(symbol, event_ts, df_prices)
            if pd.isna(entry_price):
                print(f"WARNING: no next bar price found for {symbol} at {event_ts}")
            open_positions[symbol] = {
                "entry_time": event_ts,
                "entry_price": entry_price,
                "side": side
            }

        elif action == "close":
            if symbol not in open_positions:
                print(f"WARNING: no open record for {symbol} at close time {event_ts}")
                continue

            trade_info = open_positions[symbol]
            entry_time = trade_info["entry_time"]
            entry_price = trade_info["entry_price"]
            side_open = trade_info["side"]

            exit_price = get_next_bar_price(symbol, event_ts, df_prices)
            if pd.isna(exit_price):
                print(f"WARNING: no next bar price found for {symbol} at close time {event_ts}")

            if side_open == 1:
                realized_pnl = (exit_price - entry_price) / entry_price
            else:
                realized_pnl = (entry_price - exit_price) / entry_price

            completed_trades.append({
                "symbol": symbol,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": event_ts,
                "exit_price": exit_price,
                "side": side_open,
                "realized_pnl_pcg": realized_pnl
            })
            del open_positions[symbol]

    return pd.DataFrame(completed_trades)


def merge_trades_with_real_pnl(df_trades: pd.DataFrame, df_real: pd.DataFrame, time_window: str = "10min") -> pd.DataFrame:
    # Reset indexes so that exit_time becomes a column.
    df_trades_reset = df_trades.reset_index().copy()
    df_real_reset = df_real.reset_index().copy()
    
    # Rename the index column in the real pnl data to 'exit_time'
    df_real_reset.rename(columns={"index": "exit_time"}, inplace=True)
    
    # If the real pnl DataFrame uses 'coin' instead of 'symbol', rename it.
    if "coin" in df_real_reset.columns and "symbol" not in df_real_reset.columns:
        df_real_reset.rename(columns={"coin": "symbol"}, inplace=True)
    
    # Convert exit_time columns to datetime, drop timezone information, and floor to the nearest second.
    df_trades_reset["exit_time"] = pd.to_datetime(df_trades_reset["exit_time"], errors="coerce") \
        .dt.tz_localize(None).dt.floor("s")
    df_real_reset["exit_time"] = pd.to_datetime(df_real_reset["exit_time"], errors="coerce") \
        .dt.tz_localize(None).dt.floor("s")
    
    # Clean the symbol columns (remove extra spaces, force string type).
    df_trades_reset["symbol"] = df_trades_reset["symbol"].astype(str).str.strip()
    df_real_reset["symbol"] = df_real_reset["symbol"].astype(str).str.strip()
    
    # Drop any rows where merge keys (exit_time or symbol) are null.
    df_trades_reset.dropna(subset=["exit_time", "symbol"], inplace=True)
    df_real_reset.dropna(subset=["exit_time", "symbol"], inplace=True)
    
    # Sort both DataFrames by exit_time.
    df_trades_reset.sort_values("exit_time", inplace=True)
    df_real_reset.sort_values("exit_time", inplace=True)
    
    time_delta = pd.Timedelta(time_window)
    df_merged = pd.merge_asof(
        df_trades_reset,
        df_real_reset,
        by="symbol",
        left_on="exit_time",
        right_on="exit_time",
        direction="nearest",
        tolerance=time_delta
    )
    
    # Set exit_time as index.
    df_merged.set_index("exit_time", inplace=True)
    
    # Add a new column: duration (in hours) in the position.
    # (Assumes that 'entry_time' is still available as a column.)
    df_merged["hours_in"] = (df_merged.index - pd.to_datetime(df_merged["entry_time"])).dt.total_seconds() / 3600
    
    # Add a new column: calculated percentage PnL = realized_pnl_pcg - exit_spread.
    # This subtracts the spread (from the real pnl data) from the trade's percentage pnl.
    df_merged["pnl_minus_spread"] = df_merged["realized_pnl_pcg"] - df_merged["exit_spread"]
    
    return df_merged




def plot_cumulative_pnl(df_matched: pd.DataFrame, df_pnl: pd.DataFrame, output_path: str = "cumulative_pnl_comparison.png") -> None:
    """
    Plots the cumulative PnL from merged trades (calculated and realized) and the theoretical PnL.
    
    For the theoretical pnl, the CSV file is assumed to have its index as the entry time.
    We compute the exit time by adding the 'hours_in' column (as a timedelta in hours)
    to the entry time, and then use that as the x-axis.
    """
    # Process merged trades data.
    df_plot = df_matched.copy()
    df_plot["cumulative_pct_pnl"] = df_plot["pnl_minus_spread"].cumsum()
    df_plot["cumulative_real_pnl"] = df_plot["pnl_real"].cumsum()
    
    # Process theoretical pnl data.
    df_pnl = df_pnl.copy()

    # Convert the CSV index (entry time) to datetime correctly.
    df_pnl.index = pd.to_datetime(df_pnl.index, errors="coerce", utc=True).tz_localize(None).floor("s")
    
    # Compute exit_time by adding the 'hours_in' column (converted to a timedelta).
    df_pnl["exit_time"] = df_pnl.index # + pd.to_timedelta(df_pnl["hours_in"], unit="h")
    df_pnl["exit_time"] = df_pnl["exit_time"].dt.floor("s")
    df_pnl.sort_values("exit_time", inplace=True)
    df_pnl.set_index("exit_time", inplace=True)
    
    # Compute cumulative theoretical pnl.
    df_pnl["cumulative_theo_pnl"] = df_pnl["value"].cumsum()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(df_plot.index, df_plot["cumulative_pct_pnl"], color="tab:orange", label="Calculated PNL - exit slippage")
    ax.plot(df_plot.index, df_plot["cumulative_real_pnl"], color="tab:green", label="Realized PNL")
    ax.plot(df_pnl.index, df_pnl["cumulative_theo_pnl"], color="tab:blue", label="Theoretical PNL")
    
    missing_real = df_plot[df_plot["pnl_real"].isna()]
    for i, ts in enumerate(missing_real.index):
        ax.axvline(
            x=ts, 
            color="red", 
            linestyle="--", 
            alpha=0.4, 
            linewidth=0.8,
            label="Missing Real PNL" if i == 0 else None
        )
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative PNL")
    ax.set_title("Cumulative Realized PNL Comparison")
    ax.legend(loc="upper left")
    plt.grid()
    plt.savefig(output_path, dpi=300)
    plt.show()

def flag_trades(df_merged: pd.DataFrame, df_pnl: pd.DataFrame, tolerance_diff: float = 0.001) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assumes that df_merged (merged trades) and df_pnl (theoretical pnl) are already sorted
    in the same order and have the same number of rows when merged line-by-line.
    
    It computes:
      - diff_calc_real: |pnl_minus_spread - pnl_real|
      - diff_real_theo: |pnl_real - value|
      - diff_hours: |actual holding hours - theoretical hours_in|
    
    The function then flags trades where:
      - diff_calc_real > tolerance_diff, and separately
      - diff_real_theo > tolerance_diff.
      
    Both returned DataFrames include the "diff_hours" column.
    
    Returns:
      flagged_calc_real: trades where |pnl_minus_spread - pnl_real| > tolerance_diff.
      flagged_real_theo: trades where |pnl_real - value| > tolerance_diff.
    """
    # Reset indexes so that rows line up.
    df_merged_reset = df_merged.reset_index().copy()
    df_pnl_reset = df_pnl.reset_index(drop=True).copy()
    
    # Check that both DataFrames have the same number of rows.
    if len(df_merged_reset) != len(df_pnl_reset):
        raise ValueError("The merged trades DataFrame and theoretical pnl DataFrame do not have the same number of rows.")
    
    # Compute difference between calculated PnL (pnl_minus_spread) and real PnL.
    df_merged_reset["diff_calc_real"] = (df_merged_reset["pnl_minus_spread"] - df_merged_reset["pnl_real"]).abs()
    
    # Compute difference between real PnL and theoretical PnL.
    df_merged_reset["diff_real_theo"] = (df_merged_reset["pnl_real"] - df_pnl_reset["value"]).abs()
    
    # Compute the difference in hours: actual vs. theoretical.
    df_merged_reset["diff_hours"] = (df_merged_reset["hours_in"] - df_pnl_reset["hours_in"]).abs()
    
    # Flag trades where the differences exceed tolerance_diff.
    flagged_calc_real = df_merged_reset[df_merged_reset["diff_calc_real"] > tolerance_diff].copy()
    flagged_real_theo = df_merged_reset[df_merged_reset["diff_real_theo"] > tolerance_diff].copy()
    
    return flagged_calc_real, flagged_real_theo

def parse_order_log(log_path: str) -> pd.DataFrame:
    """
    Reads order.log and extracts the timestamp and the full message.
    
    Returns a DataFrame with columns:
      - timestamp (datetime): The timestamp of the log entry
      - message (str): The full message after the timestamp
    
    Expected line format:
      2025/01/22 04:24:18.721;INFO;broker_web_binancefut_mel_cm2;splitting order into 2 manual orders
    """
    rows = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split on the first semicolon after the timestamp
                # Format: "2025/01/22 04:24:18.721;rest_of_message"
                parts = line.split(";", 1)  # Split only on the first semicolon
                if len(parts) != 2:
                    print(f"Skipping malformed line: {line}")
                    continue
                
                dt_str = parts[0]  # e.g., "2025/01/22 04:24:18.721"
                message = parts[1]  # e.g., "INFO;broker_web_binancefut_mel_cm2;splitting order into 2 manual orders"
                
                try:
                    dt = pd.to_datetime(dt_str, format="%Y/%m/%d %H:%M:%S.%f")
                    rows.append({
                        "timestamp": dt,
                        "message": message
                    })
                except ValueError as e:
                    print(f"Error parsing timestamp in line: {line} - {e}")
                    continue
                
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {log_path}")
        raise e

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"Warning: No valid entries found in {log_path}")
    else:
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def extract_execution_details(df_order_log: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts execution details from df_order_log where message contains 'exited with state True and exec_result'.
    Handles both single-pair and paired trades.
    
    Returns a DataFrame with columns:
    - timestamp (datetime)
    - pair (str)
    - execution_price (float)
    """
    pattern_single = re.compile(
        r"ticker ([A-Z]+) exited with state True and exec_result: \((-?\d+\.\d+|-?\d+), (-?\d+\.\d+|-?\d+)\)"
    )
    pattern_paired = re.compile(
        r"ticker ([A-Z]+)__([A-Z]+) exited with state True and exec_result: \(\((-?\d+\.\d+|-?\d+), (-?\d+\.\d+|-?\d+)\), \((-?\d+\.\d+|-?\d+), (-?\d+\.\d+|-?\d+)\)\)"
    )
    
    rows = []
    for _, row in df_order_log.iterrows():
        message = row["message"]
        timestamp = row["timestamp"]
        
        if "exited with state True and exec_result" not in message:
            continue
        
        # Single pair case
        match_single = pattern_single.search(message)
        if match_single:
            pair, qty, price = match_single.groups()
            execution_price = abs(float(price))  # Use absolute value as requested
            rows.append({
                "timestamp": timestamp,
                "pair": pair,
                "execution_price": execution_price
            })
            continue
        
        # Paired trades case
        match_paired = pattern_paired.search(message)
        if match_paired:
            pair1, pair2, qty1, qty2, price1, price2 = match_paired.groups()
            rows.append({
                "timestamp": timestamp,
                "pair": pair1,
                "execution_price": float(price1)
            })
            rows.append({
                "timestamp": timestamp,
                "pair": pair2,
                "execution_price": float(price2)
            })
    
    df_executions = pd.DataFrame(rows)
    if df_executions.empty:
        print("Warning: No execution details extracted from order log.")
    else:
        df_executions.sort_values("timestamp", inplace=True)
        df_executions.reset_index(drop=True, inplace=True)
    return df_executions

def match_executions_to_trades(df_trades: pd.DataFrame, df_executions: pd.DataFrame, time_tolerance: str = "1min") -> pd.DataFrame:
    """
    Matches execution prices from df_executions to df_trades based on timestamp proximity.
    Updates df_trades with execution prices as entry_price or exit_price.
    """
    df_trades = df_trades.copy()
    df_trades["entry_price_exec"] = np.nan
    df_trades["exit_price_exec"] = np.nan
    
    print("Columns in df_trades before matching:", df_trades.columns.tolist())
    
    tolerance = pd.Timedelta(time_tolerance)
    
    for _, exec_row in df_executions.iterrows():
        ts_exec = exec_row["timestamp"]
        pair = exec_row["pair"]
        exec_price = exec_row["execution_price"]
        
        mask = df_trades["symbol"] == pair
        df_trades_pair = df_trades[mask].copy()
        
        if not df_trades_pair.empty:
            entry_diff = (df_trades_pair["entry_time"] - ts_exec).abs()
            exit_diff = (df_trades_pair["exit_time"] - ts_exec).abs()
            
            entry_match = df_trades_pair[entry_diff <= tolerance]
            exit_match = df_trades_pair[exit_diff <= tolerance]
            
            if not entry_match.empty:
                idx = entry_match.index[0]
                df_trades.loc[idx, "entry_price_exec"] = exec_price
            elif not exit_match.empty:
                idx = exit_match.index[0]
                df_trades.loc[idx, "exit_price_exec"] = exec_price
    
    df_trades["entry_price"] = df_trades["entry_price_exec"].combine_first(df_trades["entry_price"])
    df_trades["exit_price"] = df_trades["exit_price_exec"].combine_first(df_trades["exit_price"])
    
    return df_trades




def main():

    # Paths
    current_state_log_path = "binance_bot_data/current_state_new.log"
    order_log_path = "binance_bot_data/order_new.log"
    pnl_real_csv = "binance_bot_data/pnl_real_new.csv"
    price_path = "binance_bot_data/price_data_new.csv"

    first_trade_date = "2025-03-06 00:00:00"
    first_exit_time = "2025-03-06 15:50:32"
    last_exit_time = "2025-03-10 08:57:03"
    merge_time_window = "10min"
    output_trades = "df_trades_new.csv"
    plot_output = "cumulative_pnl_comparison_new.png"

    # Parse logs
    df_log = parse_current_state_log(current_state_log_path)
    df_log = df_log[df_log["timestamp"] >= pd.Timestamp(first_trade_date)].copy()
    print("Current state log dataframe:")
    print(df_log.head())
    print("=" * 60)

    df_order_log = parse_order_log(order_log_path)
    print("Order log dataframe:")
    print(df_order_log.head())
    print("=" * 60)

    # Extract execution details
    df_executions = extract_execution_details(df_order_log)
    print("Execution details dataframe:")
    print(df_executions)
    print("=" * 60)

    df_events = extract_position_events(df_log)
    print("Extracted events dataframe:")
    print(df_events)
    df_events.to_csv("events.csv", index=True)
    print("=" * 60)

    df_prices = load_price_data(price_path)
    print("Prices dataframe:")
    print(df_prices.head())
    print("=" * 60)

    df_trades = build_trades_from_events(df_events, df_prices)
    print("Built trades dataframe:")
    print(df_trades.head())
    print("=" * 60)

    # Match execution prices to trades
    df_trades = match_executions_to_trades(df_trades, df_executions, time_tolerance="1min")
    print("Trades with execution prices:")
    print(df_trades.head())
    print("=" * 60)

    # Filter trades
    first_exit = pd.Timestamp(first_exit_time)
    last_exit = pd.Timestamp(last_exit_time)
    df_trades = df_trades[
        (pd.to_datetime(df_trades["exit_time"]) >= first_exit) &
        (pd.to_datetime(df_trades["exit_time"]) <= last_exit)
    ].copy()
    df_trades.set_index("exit_time", inplace=True)
    df_trades.to_csv(output_trades, index=True)
    print("Filtered trades saved:")
    print(df_trades.head(20))
    print("=" * 60)

    df_real = extract_trading_moments(pnl_real_csv)
    print("Filtered real pnl dataframe:")
    print(df_real)
    print("=" * 60)

    df_pnl = pd.read_csv("pnl.csv", index_col=0)
    print(df_pnl["value"])

    df_merged = merge_trades_with_real_pnl(df_trades, df_real, time_window=merge_time_window)
    print("Merged trades with real pnl:")
    print(df_merged)
    print("=" * 60)
    df_merged.to_csv("pnl_comparison.csv", index=True)

    # plot_cumulative_pnl(df_merged, df_pnl, output_path=plot_output)
    flagged_calc_real, flagged_real_theo = flag_trades(df_merged, df_pnl, tolerance_diff=0.001)
    print("Flagged trades (Calculated PnL vs Real PnL):")
    print(flagged_calc_real[["exit_time", "symbol", "diff_calc_real", "diff_hours"]])
    print("=" * 60)
    print("Flagged trades (Real PnL vs Theoretical PnL):")
    print(flagged_real_theo[["exit_time", "symbol", "diff_real_theo", "diff_hours"]])



if __name__ == "__main__":
    main()




