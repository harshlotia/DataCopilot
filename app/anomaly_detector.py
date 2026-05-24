from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Anomaly:
    table: str
    column: str
    anomaly_type: str
    severity: str  # "high" | "medium" | "low"
    description: str
    value: float = 0.0
    ai_explanation: str = ""


class AnomalyDetector:
    NULL_HIGH = 50.0
    NULL_MEDIUM = 20.0
    DUPLICATE_THRESHOLD = 5.0
    IQR_MULTIPLIER = 3.0
    OUTLIER_FLAG_PCT = 1.0

    def detect(self, table_name: str, df: pd.DataFrame) -> list[Anomaly]:
        anomalies: list[Anomaly] = []
        anomalies.extend(self._check_nulls(table_name, df))
        anomalies.extend(self._check_duplicates(table_name, df))
        anomalies.extend(self._check_outliers(table_name, df))
        anomalies.extend(self._check_zero_variance(table_name, df))
        return anomalies

    def _check_nulls(self, table: str, df: pd.DataFrame) -> list[Anomaly]:
        results = []
        for col in df.columns:
            pct = df[col].isnull().mean() * 100
            if pct >= self.NULL_MEDIUM:
                severity = "high" if pct >= self.NULL_HIGH else "medium"
                results.append(
                    Anomaly(
                        table=table,
                        column=col,
                        anomaly_type="High Null Rate",
                        severity=severity,
                        description=f"{pct:.1f}% of values are null ({int(df[col].isnull().sum())} rows)",
                        value=pct,
                    )
                )
        return results

    def _check_duplicates(self, table: str, df: pd.DataFrame) -> list[Anomaly]:
        dup_count = df.duplicated().sum()
        dup_pct = dup_count / len(df) * 100
        if dup_pct >= self.DUPLICATE_THRESHOLD:
            return [
                Anomaly(
                    table=table,
                    column="(entire row)",
                    anomaly_type="Duplicate Rows",
                    severity="medium",
                    description=f"{dup_pct:.1f}% of rows are exact duplicates ({int(dup_count)} rows)",
                    value=dup_pct,
                )
            ]
        return []

    def _check_outliers(self, table: str, df: pd.DataFrame) -> list[Anomaly]:
        results = []
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 20:
                continue
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower, upper = Q1 - self.IQR_MULTIPLIER * IQR, Q3 + self.IQR_MULTIPLIER * IQR
            outlier_mask = (series < lower) | (series > upper)
            outlier_pct = outlier_mask.mean() * 100
            if outlier_pct >= self.OUTLIER_FLAG_PCT:
                results.append(
                    Anomaly(
                        table=table,
                        column=col,
                        anomaly_type="Outliers Detected",
                        severity="low",
                        description=(
                            f"{outlier_pct:.1f}% outliers (IQR x{self.IQR_MULTIPLIER}). "
                            f"Expected range: [{lower:.2f}, {upper:.2f}]"
                        ),
                        value=outlier_pct,
                    )
                )
        return results

    def _check_zero_variance(self, table: str, df: pd.DataFrame) -> list[Anomaly]:
        results = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dropna().nunique() == 1:
                results.append(
                    Anomaly(
                        table=table,
                        column=col,
                        anomaly_type="Zero Variance",
                        severity="low",
                        description=f"All non-null values are identical: {df[col].dropna().iloc[0]}",
                        value=0.0,
                    )
                )
        return results
