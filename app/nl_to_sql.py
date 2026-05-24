import re
import anthropic


_SYSTEM_PROMPT = """You are a DuckDB SQL expert. Given a database schema and a natural language question, return a valid DuckDB SQL query.

Rules:
- Return ONLY the raw SQL query — no markdown, no explanation, no backticks
- Use table and column names exactly as defined in the schema
- For date truncation use DATE_TRUNC('month', col) or STRFTIME(col, '%Y-%m')
- Always alias aggregated columns with readable names (e.g., AS total_revenue)
- Add LIMIT 500 unless the question asks for all data or an aggregate over everything
- When the question asks to "show" or "list", ORDER BY a meaningful column DESC"""


class NLToSQL:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_sql(self, question: str, schema: str) -> str:
        msg = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"}],
        )
        sql = msg.content[0].text.strip()
        sql = re.sub(r"```(?:sql)?\n?", "", sql).strip().rstrip("`").strip()
        return sql

    def explain_result(self, question: str, sql: str, result_summary: str) -> str:
        msg = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\nSQL used: {sql}\nResult summary: {result_summary}\n\n"
                        "Write 1-2 sentences of plain-English insight about this result. Be specific with numbers."
                    ),
                }
            ],
        )
        return msg.content[0].text.strip()

    def explain_anomaly(self, anomaly_description: str) -> str:
        msg = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=120,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Data anomaly detected: {anomaly_description}\n\n"
                        "In 1-2 sentences, explain what likely caused this and what a data engineer should do."
                    ),
                }
            ],
        )
        return msg.content[0].text.strip()
