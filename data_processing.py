import os
import sys
import traceback
import pandas as pd
import numpy as np

def load_and_align_data(file_map, data_dir):
    """
    Loads and aligns all currency data sources from a specified directory.
    This now includes the 'open' price needed for bar pattern features.
    """
    print("Loading and aligning all currency data sources...")
    all_dfs = {}
    # Include 'open' in the column names
    file_column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
    for symbol, filename in file_map.items():
        full_path = os.path.join(data_dir, filename)
        print(f"Attempting to load: {full_path}")
        try:
            # Load data
            df = pd.read_csv(full_path, sep='\t', header=0, names=file_column_names, skiprows=1, quotechar='"', encoding='utf-8-sig')
            
            # Convert all necessary columns to numeric
            for col in ['open', 'high', 'low', 'close', 'tickvol']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Create datetime index
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce', utc=True)
            
            # Clean data
            df.dropna(subset=['datetime', 'open', 'high', 'low', 'close', 'tickvol'], inplace=True)
            df.drop_duplicates(subset='datetime', keep='first', inplace=True)
            df.set_index('datetime', inplace=True)
            
            # Add symbol prefix to all columns to avoid clashes
            all_dfs[symbol] = df.add_prefix(f'{symbol}_')

        except FileNotFoundError:
            print(f"FATAL ERROR: Data file '{filename}' not found at path '{full_path}'.")
            sys.exit(2)
        except Exception as e:
            print(f"FATAL ERROR processing {filename}: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Combine all dataframes
    master_df = pd.concat(all_dfs.values(), axis=1, join='inner')
    master_df.ffill(inplace=True)
    master_df.dropna(inplace=True)
    print(f"Loaded and aligned {len(master_df)} data points.")
    return master_df

def create_features(df):
    """
    Creates the expanded 15-feature set for the model.
    """
    print("Creating expanded feature set (15 features)...")
    
    # --- Base Features & Indicators for EURUSD ---
    for col in df.columns:
        if 'close' in col and 'return' not in col:
            df[f'{col}_return'] = df[col].pct_change()

    df['eurusd_return'] = df['EURUSD_close_return']
    df['eurusd_volume'] = df['EURUSD_tickvol']
    df['eurusd_atr'] = (df['EURUSD_high'] - df['EURUSD_low']).rolling(14).mean()
    
    ema_fast = df['EURUSD_close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['EURUSD_close'].ewm(span=26, adjust=False).mean()
    df['eurusd_macd'] = ema_fast - ema_slow
    
    delta = df['EURUSD_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['eurusd_rsi'] = 100 - (100 / (1 + rs))
    
    low_14 = df['EURUSD_low'].rolling(window=14).min()
    high_14 = df['EURUSD_high'].rolling(window=14).max()
    df['eurusd_stoch'] = 100 * ((df['EURUSD_close'] - low_14) / (high_14 + 1e-10))
    
    tp = (df['EURUSD_high'] + df['EURUSD_low'] + df['EURUSD_close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['eurusd_cci'] = (tp - sma_tp) / (0.015 * (mad + 1e-10))
    
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    df['USD_Index_Proxy'] = (df['USDJPY_close_return'] + df['USDCAD_close_return'] + df['USDCHF_close_return']) - (df['EURUSD_close_return'] + df['GBPUSD_close_return'])
    df['EUR_Index_Proxy'] = df['EURUSD_close_return'] + df['EURJPY_close_return'] + df['EURGBP_close_return']
    df['JPY_Index_Proxy'] = -(df['EURJPY_close_return'] + df['USDJPY_close_return'])

    # --- NEW FEATURE 1: Bollinger Band Width (Normalized) ---
    sma_20 = df['EURUSD_close'].rolling(window=20).mean()
    std_20 = df['EURUSD_close'].rolling(window=20).std()
    upper_bb = sma_20 + (std_20 * 2)
    lower_bb = sma_20 - (std_20 * 2)
    df['eurusd_bbw'] = (upper_bb - lower_bb) / (sma_20 + 1e-10)

    # --- NEW FEATURE 2: Volume Delta ---
    df['eurusd_vol_delta'] = df['EURUSD_tickvol'].diff(periods=5)

    # --- NEW FEATURE 3: Bar Pattern / Gap ---
    body_size = (df['EURUSD_close'] - df['EURUSD_open']).abs()
    bar_range = df['EURUSD_high'] - df['EURUSD_low']
    is_doji = (body_size / (bar_range + 1e-10)) < 0.1
    
    is_bull_engulf = (df['EURUSD_open'] < df['EURUSD_close']) & \
                     (df['EURUSD_open'] < df['EURUSD_open'].shift(1)) & \
                     (df['EURUSD_close'] > df['EURUSD_close'].shift(1))
                     
    is_bear_engulf = (df['EURUSD_open'] > df['EURUSD_close']) & \
                     (df['EURUSD_open'] > df['EURUSD_open'].shift(1)) & \
                     (df['EURUSD_close'] < df['EURUSD_close'].shift(1))

    gap = df['EURUSD_open'] - df['EURUSD_close'].shift(1)
    gap_norm = gap / (df['eurusd_atr'] + 1e-10) # Normalize gap by ATR

    df['eurusd_bar_type'] = 0.0
    df.loc[is_doji, 'eurusd_bar_type'] = 1.0
    df.loc[is_bull_engulf, 'eurusd_bar_type'] = 2.0
    df.loc[is_bear_engulf, 'eurusd_bar_type'] = -2.0
    df['eurusd_bar_type'] += gap_norm # Add normalized gap to the pattern score

    # --- FINAL FEATURE LIST (15 FEATURES) ---
    feature_list = [
        'eurusd_return', 'eurusd_volume', 'eurusd_atr', 'eurusd_macd', 'eurusd_rsi', 
        'eurusd_stoch', 'eurusd_cci', 'hour_of_day', 'day_of_week', 
        'USD_Index_Proxy', 'EUR_Index_Proxy', 'JPY_Index_Proxy',
        'eurusd_bbw', 'eurusd_vol_delta', 'eurusd_bar_type' # New features
    ]
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    print(f"Data size after feature creation and dropna: {len(df)}")
    return df, feature_list