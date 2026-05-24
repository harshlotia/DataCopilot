import pandas as pd
import numpy as np


def profile_table(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a per-column profile DataFrame."""
    rows = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_pct = round(null_count / len(series) * 100, 1)
        unique_count = int(series.nunique())
        dtype = str(series.dtype)

        row = {
            "Column": col,
            "Type": dtype,
            "Non-Null": len(series) - null_count,
            "Null %": null_pct,
            "Unique": unique_count,
            "Sample Value": _safe_sample(series),
        }

        if pd.api.types.is_numeric_dtype(series):
            row["Min"] = round(series.min(), 2) if not series.dropna().empty else None
            row["Max"] = round(series.max(), 2) if not series.dropna().empty else None
            row["Mean"] = round(series.mean(), 2) if not series.dropna().empty else None
        else:
            row["Min"] = None
            row["Max"] = None
            row["Mean"] = None

        rows.append(row)
    return pd.DataFrame(rows)


def _safe_sample(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return "N/A"
    return str(non_null.iloc[0])
