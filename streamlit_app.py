import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import requests
from io import StringIO
import torch
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple, Dict
import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error


@st.cache_data
def load_super_data(p: str = "super_data.pkl") -> dict:
    pth = Path(p)
    if not pth.exists():
        raise FileNotFoundError(f"Could not find {p}. Make sure it's in the app directory.")
    with open(pth, "rb") as f:
        data = pickle.load(f)
    return data

def temporal_splits(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = df.sort_index()
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex (hourly).")
    # Interpolate missing hours (optional): reindex to full hourly range between min/max
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)
    # Simple missing-value handling -- fill/interpolate (you can swap strategies)
    df = df.interpolate(method='time').ffill().bfill()
    
    train = df.loc['2020-01-01':'2023-12-31 23:00:00']
    val   = df.loc['2024-01-01':'2024-06-30 23:00:00']
    test  = df.loc['2024-07-01':'2024-12-31 23:00:00']
    return {"train": train, "val": val, "test": test}


# Map parameter long names and short column names to USGS parameter codes
parameter_key = [
    ("Instantaneous computed discharge", "00060"),
    ("Instantaneous water temperature", "00010"),
    ("Instantaneous pH", "00400"),
    ("Instantaneous diss. oxygen", "00300"),
    ("Instantaneous specific conductance at 25 degrees Celsius", "00095"),
    ("Instantaneous turbidity", "63680"),
    ("Computed instantaneous total organic carbon", "00680"),
    ("Computed instantaneous diss. nitrate + nitrite", "00631"),
    ("Computed instantaneous total phosphorus", "00665"),
]

column_map = {
    "Instantaneous computed discharge": "Discharge (cfs)",
    "Instantaneous water temperature": "Water Temperature (C)",
    "Instantaneous pH": "pH (std units)",
    "Instantaneous diss. oxygen": "Dissolved Oxygen (mg/L)",
    "Instantaneous specific conductance at 25 degrees Celsius": "Specific Conductance (uS/cm)",
    "Instantaneous turbidity": "Turbidity (NTU)",
    "Computed instantaneous total organic carbon": "Total Organic Carbon (mg/L)",
    "Computed instantaneous diss. nitrate + nitrite": "Nitrate + Nitrite (mg/L)",
    "Computed instantaneous total phosphorus": "Total Phosphorus (mg/L)",
}


def fetch_usgs_for_param_range(pcode: tuple, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.Series:
    """Fetch USGS unit-value data for a parameter code between start_dt and end_dt (inclusive).

    This re-uses the nrtwq endpoint by fetching per-year chunks (the endpoint returns year-based data).
    Returns a pandas Series indexed by naive datetimes.
    """
    years = list(range(start_dt.year, end_dt.year + 1))
    parts = []
    for year in years:
        # If the year is the current year change to ytd
        if year == pd.Timestamp.now().year:
            year = 'ytd'
        url = (
            "https://nrtwq.usgs.gov/explore/datatable?"
            f"site_no=08068500&pcode={pcode[1]}&period={year}_all&timestep=uv&format=rdb&is_verbose=y"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        raw = resp.text
        # Filter out comment lines
        lines = [line for line in raw.splitlines() if not line.startswith('#')]
        # The RDB file sometimes contains header lines; keep first and then data lines
        if len(lines) < 3:
            continue
        # Heuristic: drop the duplicate header line if present
        lines = [line for line in raw.splitlines() if not line.startswith('#')]
        stop = 0
        i = 0
        for line in lines:
            if "01/01" in line and "01:00" in line:
                stop = i
                break
            i += 1
        lines = lines[:stop+1]
        lines = [lines[0]] + lines[2:]
        cleaned = "\n".join(lines)
        # Read into DataFrame
        df = pd.read_csv(StringIO(cleaned), sep='\t', comment='#', low_memory=False)
        # return(df)
        df['DateTime'] = pd.to_datetime(df['Date-Time'], format='%m/%d/%Y %H:%M', errors='coerce')
        df.set_index('DateTime', inplace=True)
        if pcode[0] in df.columns:
            series = df[pcode[0]].rename(pcode[0])

            # Convert USGS missing markers like '--' or '-' to NaN
            series = pd.to_numeric(series, errors='coerce')

            # Drop duplicates and NaNs
            series = series[~series.index.duplicated(keep='first')]
            series = series.dropna()

            parts.append(series)


    if not parts:
        return pd.Series(dtype=float)

    combined = pd.concat(parts).sort_index()
    # Restrict to requested range
    combined = combined.loc[start_dt:end_dt]
    # Reindex to hourly and forward/backfill small gaps, then keep numeric values
    full_idx = pd.date_range(start_dt, end_dt, freq='h')
    
    combined = combined.reindex(full_idx).astype(float)
    combined = combined.interpolate(method='time').ffill().bfill()
    combined.index.name = 'DateTime'
    return combined


@st.cache_data(show_spinner=False)
def fetch_params_for_year(year: int) -> pd.DataFrame:
    """Fetch all parameters for a full calendar year and return a DataFrame indexed hourly with short column names."""
    year_start = pd.Timestamp(year=year, month=1, day=1, hour=0)
    year_end = pd.Timestamp(year=year, month=12, day=31, hour=23)
    parts = []
    col_names = []
    for long_name, code in parameter_key:
        short = column_map.get(long_name)
        if short is None:
            continue
        series = fetch_usgs_for_param_range((long_name, code), year_start, year_end)
        if series.empty:
            # create empty series for consistency
            series = pd.Series(index=pd.date_range(year_start, year_end, freq='h'), dtype=float)
        series.name = short
        parts.append(series)
        col_names.append(short)
    if not parts:
        return pd.DataFrame()
    df_year = pd.concat(parts, axis=1)
    df_year.index.name = 'DateTime'
    # Ensure hourly index for the whole year
    full_idx = pd.date_range(year_start, year_end, freq='h')
    df_year = df_year.reindex(full_idx)
    # Interpolate small gaps and fill remaining with zeros
    df_year = df_year.interpolate(method='time').ffill().bfill()
    # Add rainfall placeholder column
    if 'Precipitation 1hr (in)' not in df_year.columns:
        df_year['Precipitation 1hr (in)'] = 0.0
    return df_year


@st.cache_data(show_spinner=False)
def fetch_params_for_range(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    """Fetch parameters for the requested datetime range by composing cached per-year data."""
    years = list(range(start_dt.year, end_dt.year + 1))
    parts = []
    for y in years:
        dfy = fetch_params_for_year(y)
        if dfy.empty:
            continue
        parts.append(dfy)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts).sort_index()
    combined = combined.loc[start_dt:end_dt]
    # Ensure hourly index
    full_idx = pd.date_range(start_dt, end_dt, freq='h')
    combined = combined.reindex(full_idx)
    combined = combined.interpolate(method='time').ffill().bfill()
    combined.index.name = 'DateTime'
    return combined

# ---------------------------
# Dataset class - creates sliding windows but only from each split separately
# ---------------------------
class SlidingWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_cols, target_col: str, 
                 input_len: int = 168, out_len: int = 24):
        """
        df: DataFrame containing contiguous hourly data for the split
        input_cols: list of columns used as features (all params)
        target_col: name of column to forecast
        """
        self.input_len = input_len
        self.out_len = out_len
        self.input_cols = input_cols
        self.target_col = target_col
        arr_x = df[input_cols].values.astype(np.float32)
        arr_y = df[target_col].values.astype(np.float32)
        N = len(df)
        self.X = []
        self.Y = []
        # build windows (no crossing the split boundary because df is per-split)
        for i in range(0, N - (input_len + out_len) + 1):
            x = arr_x[i : i + input_len]
            y = arr_y[i + input_len : i + input_len + out_len]
            self.X.append(x)
            self.Y.append(y)
        self.X = np.stack(self.X) if len(self.X) > 0 else np.empty((0, input_len, len(input_cols)), dtype=np.float32)
        self.Y = np.stack(self.Y) if len(self.Y) > 0 else np.empty((0, out_len), dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, out_len=24, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        # map hidden state to output sequence (we'll use the last hidden state)
        self.fc = nn.Linear(hidden_size, out_len)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        output, (h_n, c_n) = self.lstm(x)  # h_n shape: (num_layers, batch, hidden)
        # take last layer hidden state
        last_hidden = h_n[-1]  # (batch, hidden)
        out = self.fc(last_hidden)  # (batch, out_len)
        return out


def find_best_index(test_index: pd.DatetimeIndex, target_dt: pd.Timestamp, allow_nearest: bool = True) -> Optional[int]:
    if test_index is None or len(test_index) == 0:
        return None
    # exact match
    matches = np.where(test_index == target_dt)[0]
    if len(matches) > 0:
        return int(matches[0])
    if not allow_nearest:
        return None
    # nearest
    deltas = np.abs((test_index - target_dt).astype('int64').astype(int))
    return int(np.argmin(deltas))


def plot_prediction_vs_actual(res: dict, idx: int):
    target = res.get("target_col", "target")
    preds = res.get("test_preds")
    truth = res.get("test_truth")
    test_index = res.get("test_index")
    full_df = res.get("full_df")

    if preds is None or truth is None or test_index is None:
        st.error("The result object does not contain necessary fields ('test_preds','test_truth','test_index').")
        return

    if idx < 0 or idx >= preds.shape[0]:
        st.error("Selected forecast index is out of range.")
        return

    pred_vals = np.asarray(preds[idx])
    true_vals = np.asarray(truth[idx])
    start_dt = pd.to_datetime(test_index[idx])
    pred_times = pd.date_range(start_dt, periods=pred_vals.shape[0], freq='h')

    # Compute r2 for the 24-hour window (if valid)
    try:
        r2 = float(r2_score(true_vals, pred_vals))
    except Exception:
        r2 = float('nan')

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot one week of history if available
    if full_df is not None and target in full_df.columns:
        week_start = start_dt - pd.Timedelta(days=7)
        hist = full_df[target].loc[week_start:start_dt]
        if len(hist) > 0:
            ax.plot(hist.index, hist.values, label='Previous 7 days (actual)', color='tab:blue')

    # Plot predictions and truth for the 24h window
    ax.plot(pred_times, pred_vals, label='Prediction (24h)', color='tab:red')
    ax.plot(pred_times, true_vals, label='Actual (24h)', color='tab:green', linestyle='--')

    ax.axvline(start_dt, color='k', linestyle=':', label='Forecast start')
    ax.set_title(f"24h Forecast vs Actual for {target} starting {start_dt.date()} (R²={r2:.3f})")
    ax.set_xlabel('Datetime')
    ax.set_ylabel(target)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()

    st.pyplot(fig)


def main():
    st.title("Swift Creek — Forecast Viewer")
    st.subheader("LSTM model developed for NC State CE 591 Final Project")
    st.markdown(
        "This web app allows users to visualize and analyze water quality forecasts. The model was trained on historical USGS data (2020-2023) for Spring Creek, Texas."
    )
    url = "https://nrtwq.usgs.gov/explore/datatable?site_no=08068500&pcode=00060&period=2024_all&timestep=uv&is_verbose=y"
    st.markdown(
        "This app can even fetch new observation data from USGS [NRTWQ Site](%s) and run the model to generate forecasts for dates not present in the stored results. (2025-present)" % url
    )
    st.markdown("As of December 2025 data from October 20th 2025 onward is missing. ")
    
    # Load data
    try:
        super_data = load_super_data("super_data.pkl")
    except Exception as e:
        st.error(f"Failed to load super_data.pkl: {e}")
        st.stop()

    params = list(super_data.keys())
    if not params:
        st.error("Loaded `super_data.pkl` contains no parameters.")
        st.stop()

    param = st.selectbox("Choose parameter to inspect", [params[1]]+[params[0]]+params[2:])
    res = super_data.get(param)
    if res is None:
        st.error("Selected parameter has no result object.")
        st.stop()

    # Ensure test_index is a DatetimeIndex if present
    test_index = res.get('test_index')
    if test_index is not None:
        test_index = pd.to_datetime(test_index)
        res['test_index'] = pd.DatetimeIndex(test_index)

    st.write("Model metrics:")
    st.json(res.get('metrics', {}))

    # Date selection
    if test_index is None or len(test_index) == 0:
        st.warning("No `test_index` available for this parameter — cannot pick forecast dates.")
        st.stop()

    # Default date to first available test index
    try:
        # Make july 20th 2024 the default date
        default_date = pd.to_datetime("2024-07-20").date()
    except:
        default_date = pd.to_datetime(test_index[10]).date()
    today_date = pd.Timestamp.now().date()
    chosen_date = st.date_input("Choose forecast date", value=default_date, max_value=today_date)
    allow_nearest = st.checkbox("If exact date not present, use nearest available forecast date (Turn off to fetch new data)", value=True)

    dt = pd.Timestamp(chosen_date)
    idx = find_best_index(res['test_index'], dt, allow_nearest=allow_nearest)

    if idx is None:
        st.info("Selected date not found in stored forecasts — fetching observation data and running model (if available).")

        # Determine input length (assume 168 by default but allow override)
        input_len = 168# st.number_input("Input sequence length (hours)", min_value=24, max_value=168*4, value=168)

        # Determine fetch range: we need input window (input_len hours before dt) and 24h after dt for truth
        start_input = dt - pd.Timedelta(hours=input_len)
        end_output = dt + pd.Timedelta(hours=23)

        # Fetch all parameters for the required range using cached per-year fetch
        try:
            df_feats = fetch_params_for_range(start_input, end_output)
        except Exception as e:
            st.error(f"Failed to fetch observations from USGS: {e}")
            st.stop()

        if df_feats.empty:
            st.warning("No observations returned from USGS for the requested range.")
            st.stop()

        # Map column names to those expected by the model (res['input_cols'])
        input_cols = res.get('input_cols', list(df_feats.columns))
        # Ensure all required columns exist in df_feats; if missing, create with zeros
        for c in input_cols:
            if c not in df_feats.columns:
                df_feats[c] = 0.0

        # Create input window (input_len hours ending at dt-1h)
        input_start = dt - pd.Timedelta(hours=input_len)
        input_end = dt - pd.Timedelta(hours=1)
        X_window = df_feats.loc[input_start:input_end, input_cols]
        # If we don't have enough rows, reindex and fill
        if X_window.shape[0] != input_len:
            idx_full = pd.date_range(input_start, input_end, freq='h')
            X_window = X_window.reindex(idx_full)
            X_window = X_window.interpolate(method='time').ffill().bfill()

        # Scale using saved X_scaler
        X_scaler = res.get('X_scaler')
        if X_scaler is None:
            st.error("No X_scaler found in model result; cannot prepare features for model.")
            st.stop()

        X_scaled = X_scaler.transform(X_window.values)
        # reshape to (1, seq_len, n_features)
        X_input = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1]).astype(np.float32)

        # Run model prediction (move model to CPU)
        model = res.get('model')
        if model is None:
            st.error("No model object found in the selected parameter result.")
            st.stop()

        try:
            model.to('cpu')
            model.eval()
            with torch.no_grad():
                tx = torch.from_numpy(X_input)
                preds_scaled = model(tx).cpu().numpy()[0]  # shape (out_len,)
        except Exception as e:
            st.error(f"Failed to run model prediction: {e}")
            st.stop()

        # Inverse transform predictions using y_scaler
        y_scaler = res.get('y_scaler')
        if y_scaler is None:
            st.error("No y_scaler found in model result; cannot inverse-transform predictions.")
            st.stop()

        preds_inv = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)

        # Build times for prediction
        pred_times = pd.date_range(dt, periods=preds_inv.shape[0], freq='h')

        # Plot week history and the prediction vs any observed truth
        fig, ax = plt.subplots(figsize=(10, 5))
        week_start = dt - pd.Timedelta(days=7)
        if res.get('target_col') in df_feats.columns:
            hist = df_feats[res['target_col']].loc[week_start:dt - pd.Timedelta(hours=1)]
            if len(hist) > 0:
                ax.plot(hist.index, hist.values, label='Previous 7 days (actual)', color='tab:blue')

        # Plot predicted 24h
        ax.plot(pred_times, preds_inv, label='Prediction (24h)', color='tab:red')

        # Plot observed truth if available
        target_col = res.get('target_col')
        observed = None
        if target_col in df_feats.columns:
            observed = df_feats[target_col].loc[dt:dt + pd.Timedelta(hours=23)]
            if observed.isna().all():
                observed = None
        if observed is not None and len(observed) > 0:
            ax.plot(observed.index, observed.values, label='Observed (24h)', color='tab:green', linestyle='--')
            # compute r2 if lengths align
            if len(observed.dropna()) >= 2 and len(observed.dropna()) == len(preds_inv):
                try:
                    r2 = r2_score(observed.values, preds_inv[: len(observed)])
                    st.write(f"R² for prediction vs observed (available hours): {r2:.3f}")
                except Exception:
                    pass

        ax.axvline(dt, color='k', linestyle=':', label='Forecast start')
        ax.set_title(f"Model prediction for {param} starting {dt.date()}")
        ax.set_xlabel('Datetime')
        ax.set_ylabel(param)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        st.pyplot(fig)

    else:
        # Show chosen / actual date info
        chosen_actual = res['test_index'][idx]
        st.write(f"Using forecast timestamp: {pd.to_datetime(chosen_actual)} (index {idx})")

        plot_prediction_vs_actual(res, idx)

    # Optional: show full rolling daily comparison
    if st.checkbox("Show rolling daily comparison (predicted vs actual, hr=0)"):
        preds = res.get('test_preds')
        truth = res.get('test_truth')
        if preds is None or truth is None:
            st.warning("No preds/truth available to compute rolling daily comparison.")
        else:
            test_idx = res.get('test_index')
            pred_daily = pd.DataFrame(preds[:, 0], index=test_idx).rolling(24).mean()
            true_daily = pd.DataFrame(truth[:, 0], index=test_idx).rolling(24).mean()
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(true_daily, label='True (24h rolling)', color='tab:green')
            ax2.plot(pred_daily, label='Pred (24h rolling)', color='tab:red', alpha=0.7)
            ax2.set_title(f"24h Rolling Comparison for {param}")
            ax2.legend()
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)


if __name__ == '__main__':
    main()
