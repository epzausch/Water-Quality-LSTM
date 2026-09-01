import pandas as pd
import numpy as np
from typing import Tuple, Dict
from torch.utils.data import Dataset, DataLoader


def temporal_splits(df: pd.DataFrame, val_start: str, val_end: str) -> Dict[str, pd.DataFrame]:
    df = df.sort_index()
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")


    # Test goes from first entry point up to val_start at 15 min intervals
    train_idx = pd.date_range(start=df.index.min(), end=val_start, freq="15min")



    train = df.loc[df.index.min():val_start].reindex(train_idx)
    # Use forward-fill to impute from past values only (causal). This avoids using future points to fill earlier holes.
    train = train.ffill()
    # For any remaining NaNs at the start of the training period, fill with column-wise means computed on the (ffill) train data.
    train_means = train.mean() 
    train = train.fillna(train_means)

    def _prepare_split_using_train_means(start: str, end: str) -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="h")
        sub = df.loc[start:end].reindex(idx)
        # forward-fill within the split (only uses past)
        sub = sub.ffill()
        # fill any remaining NaNs (e.g., at the very beginning) with the training split means -> no future leakage
        sub = sub.fillna(train_means)
        return sub

    val   = _prepare_split_using_train_means(val_start, val_end)
    test  = _prepare_split_using_train_means(val_end, df.index.max())
    return {"train": train, "val": val, "test": test}



# Dataset class - creates sliding windows but only from each split separately
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