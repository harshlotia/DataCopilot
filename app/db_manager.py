import duckdb
import pandas as pd


class DBManager:
    def __init__(self):
        self.conn = duckdb.connect(":memory:")
        self._registered_tables: list[str] = []

    def register_dataframe(self, table_name: str, df: pd.DataFrame):
        safe_name = table_name.replace("-", "_").replace(" ", "_").lower()
        self.conn.execute(f"DROP TABLE IF EXISTS {safe_name}")
        self.conn.register("_tmp_upload", df)
        self.conn.execute(f"CREATE TABLE {safe_name} AS SELECT * FROM _tmp_upload")
        self.conn.unregister("_tmp_upload")
        if safe_name not in self._registered_tables:
            self._registered_tables.append(safe_name)
        return safe_name

    def get_tables(self) -> list[str]:
        rows = self.conn.execute("SHOW TABLES").fetchall()
        return [r[0] for r in rows]

    def get_schema(self) -> str:
        parts = []
        for table in self.get_tables():
            cols = self.conn.execute(f"DESCRIBE {table}").fetchall()
            col_defs = ", ".join(f"{c[0]} {c[1]}" for c in cols)
            sample = self.conn.execute(f"SELECT * FROM {table} LIMIT 3").df()
            parts.append(
                f"Table: {table}\nColumns: {col_defs}\nSample values:\n{sample.to_string(index=False)}"
            )
        return "\n\n".join(parts)

    def execute_query(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).df()

    def get_table_sample(self, table: str, n: int = 5) -> pd.DataFrame:
        return self.conn.execute(f"SELECT * FROM {table} LIMIT {n}").df()

    def get_full_table(self, table: str) -> pd.DataFrame:
        return self.conn.execute(f"SELECT * FROM {table}").df()

    def get_row_count(self, table: str) -> int:
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
