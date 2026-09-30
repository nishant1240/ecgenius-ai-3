"""
Weekly Progress Reports module: logs every analysis to a local SQLite
database, computes a weekly summary, and exports a PDF (reportlab) if
installed, else a Markdown fallback.
"""
import sqlite3
import datetime
import os
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ecgenius.db")


def init_db(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source TEXT,
            heart_rate REAL,
            category TEXT,
            risk TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()


def log_analysis(source, heart_rate, category, risk, confidence, db_path: str = DB_PATH):
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO analyses (timestamp, source, heart_rate, category, risk, confidence) VALUES (?,?,?,?,?,?)",
        (datetime.datetime.now().isoformat(), source, heart_rate, category, risk, confidence),
    )
    conn.commit()
    conn.close()


def load_history(db_path: str = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM analyses ORDER BY timestamp DESC", conn)
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def weekly_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"count": 0}
    cutoff = datetime.datetime.now() - datetime.timedelta(days=7)
    week = df[df["timestamp"] >= cutoff]
    if week.empty:
        week = df  # fall back to all history if nothing in last 7 days (fresh demo data)
    return {
        "count": len(week),
        "avg_hr": round(week["heart_rate"].mean(), 1),
        "most_common_class": week["category"].mode().iloc[0] if not week["category"].mode().empty else "-",
        "risk_counts": week["risk"].value_counts().to_dict(),
        "by_day": week.assign(day=week["timestamp"].dt.date).groupby("day")["heart_rate"].mean(),
    }


def export_pdf(summary: dict, out_path: str) -> str:
    """Returns the path actually written (PDF if reportlab available, else .md)."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as pdfcanvas

        c = pdfcanvas.Canvas(out_path, pagesize=A4)
        width, height = A4
        y = height - 60
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "ECGENIUS AI — Weekly Progress Report")
        y -= 30
        c.setFont("Helvetica", 10)
        c.drawString(50, y, "Educational prototype — not a medical diagnosis.")
        y -= 30
        c.setFont("Helvetica", 12)
        c.drawString(50, y, f"Analyses this week: {summary.get('count', 0)}")
        y -= 20
        c.drawString(50, y, f"Average heart rate: {summary.get('avg_hr', '-')} bpm")
        y -= 20
        c.drawString(50, y, f"Most common classification: {summary.get('most_common_class', '-')}")
        y -= 20
        c.drawString(50, y, f"Risk breakdown: {summary.get('risk_counts', {})}")
        c.save()
        return out_path
    except ImportError:
        md_path = out_path.rsplit(".", 1)[0] + ".md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# ECGENIUS AI — Weekly Progress Report\n\n")
            f.write("_Educational prototype — not a medical diagnosis._\n\n")
            f.write(f"- Analyses this week: {summary.get('count', 0)}\n")
            f.write(f"- Average heart rate: {summary.get('avg_hr', '-')} bpm\n")
            f.write(f"- Most common classification: {summary.get('most_common_class', '-')}\n")
            f.write(f"- Risk breakdown: {summary.get('risk_counts', {})}\n\n")
            f.write("_(Install `reportlab` — `pip install reportlab` — to export as PDF instead.)_\n")
        return md_path
