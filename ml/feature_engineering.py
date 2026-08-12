"""
StudyTok Analytics — feature engineering pipeline.

Takes the cleaned CSV produced by the C++ engine (one row per student
per day) and aggregates it into ONE ROW PER STUDENT, summarizing their
recent study behavior into trend-aware features suitable for a
regression model predicting overall exam performance.

Usage:
    python feature_engineering.py [input_clean_csv] [output_features_csv]

Defaults:
    input:  ../data/processed/clean_logs.csv
    output: ../data/processed/student_features.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REST_DEFICIT_THRESHOLD_HOURS = 6.0  # below this counts as a "short rest" day


def compute_trend(day_index: np.ndarray, values: np.ndarray) -> float:
    """
    Slope of `values` over `day_index` (a simple linear trend).
    Positive = improving over time, negative = declining.
    Needs at least 2 points; returns 0.0 otherwise (no trend to measure).
    """
    if len(day_index) < 2 or np.all(day_index == day_index[0]):
        return 0.0
    slope, _intercept = np.polyfit(day_index, values, 1)
    return float(slope)


def build_student_features(clean_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(clean_csv_path, parse_dates=["date"])

    if df.empty:
        return pd.DataFrame(columns=[
            "student_id", "num_logs", "avg_study_hours", "std_study_hours",
            "study_hours_trend", "avg_active_recall_score",
            "recall_score_trend", "avg_rest_hours", "rest_deficit_rate",
            "avg_sessions_count", "avg_distraction_events",
            "target_exam_score",
        ])

    df = df.sort_values(["student_id", "date"])

    rows = []
    for student_id, g in df.groupby("student_id"):
        g = g.sort_values("date")
        day_index = (g["date"] - g["date"].min()).dt.days.to_numpy()

        exam_scores = g["exam_score"].dropna()

        rows.append({
            "student_id": student_id,
            "num_logs": len(g),
            "avg_study_hours": g["study_hours"].mean(),
            "std_study_hours": g["study_hours"].std(ddof=0) if len(g) > 1 else 0.0,
            "study_hours_trend": compute_trend(day_index, g["study_hours"].to_numpy()),
            "avg_active_recall_score": g["active_recall_score"].mean(),
            "recall_score_trend": compute_trend(day_index, g["active_recall_score"].to_numpy()),
            "avg_rest_hours": g["rest_hours"].mean(),
            "rest_deficit_rate": (g["rest_hours"] < REST_DEFICIT_THRESHOLD_HOURS).mean(),
            "avg_sessions_count": g["sessions_count"].mean(),
            "avg_distraction_events": g["distraction_events"].mean(),
            # Target: mean of whatever exam scores this student has logged.
            # NaN if none logged yet (student awaiting prediction, not training).
            "target_exam_score": exam_scores.mean() if len(exam_scores) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def main():
    default_input = Path(__file__).parent.parent / "data" / "processed" / "clean_logs.csv"
    default_output = Path(__file__).parent.parent / "data" / "processed" / "student_features.csv"

    input_path = sys.argv[1] if len(sys.argv) > 1 else str(default_input)
    output_path = sys.argv[2] if len(sys.argv) > 2 else str(default_output)

    features_df = build_student_features(input_path)
    features_df.to_csv(output_path, index=False)

    n_total = len(features_df)
    n_labeled = features_df["target_exam_score"].notna().sum()

    print("StudyTok feature pipeline — summary")
    print(f"  Input:            {input_path}")
    print(f"  Output:           {output_path}")
    print(f"  Students:         {n_total}")
    print(f"  With exam labels: {n_labeled} (usable for training)")
    print(f"  Without labels:   {n_total - n_labeled} (awaiting prediction)")


if __name__ == "__main__":
    main()
