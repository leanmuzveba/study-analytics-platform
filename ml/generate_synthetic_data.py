"""
Generates a synthetic raw study-log CSV, in the same format the C++ engine
expects, for training and validating the regression model at a realistic
scale (our real hand-written sample_logs.csv only has a handful of rows —
too small to train/test split).

Each simulated student has a hidden "latent ability" the model can't see
(same as real students would have) plus behavioral habits that DO drive
the label — so the trained model should land at a believable R² (not a
suspicious 1.0), same as it would on real data.

~90% of students get exam scores logged (usable for training).
~10% are left with no exam scores at all (simulates new students who are
being tracked but haven't been graded yet — these are what the trained
model will eventually predict for).

Usage:
    python generate_synthetic_data.py [output_csv] [n_students]
"""

import sys
import numpy as np
import pandas as pd

SUBJECTS = ["Mathematics", "Physics", "Chemistry", "Biology", "History", "Geography"]


def generate(n_students: int = 220, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2026-05-01")

    n_unlabeled = max(1, round(n_students * 0.10))
    unlabeled_ids = set(rng.choice(n_students, size=n_unlabeled, replace=False))

    rows = []
    for i in range(n_students):
        student_id = f"STU{i:05d}"
        subject = rng.choice(SUBJECTS)

        # Hidden trait the model never sees directly (real unexplained variance)
        latent_ability = rng.normal(0, 4)

        # Student's baseline habits
        baseline_study_hours = rng.uniform(1.0, 6.0)
        baseline_recall = rng.uniform(50, 95)
        baseline_rest = rng.uniform(4.0, 9.0)
        recall_drift_per_day = rng.uniform(-1.0, 1.5)  # improving/declining over time

        n_logs = rng.integers(5, 16)
        log_days = sorted(rng.choice(range(0, 45), size=n_logs, replace=False))

        for day_offset in log_days:
            date = base_date + pd.Timedelta(days=int(day_offset))

            study_hours = np.clip(rng.normal(baseline_study_hours, 1.0), 0.5, 12.0)
            recall_score = np.clip(
                baseline_recall + recall_drift_per_day * (day_offset / 10) + rng.normal(0, 5),
                0, 100
            )
            rest_hours = np.clip(rng.normal(baseline_rest, 1.0), 2.0, 11.0)
            sessions_count = rng.integers(1, 5)
            distraction_events = rng.poisson(2)

            exam_score = np.clip(
                20
                + 3.2 * study_hours
                + 0.35 * recall_score
                + 1.4 * rest_hours
                - 2.3 * distraction_events
                + 0.8 * sessions_count
                + latent_ability
                + rng.normal(0, 3),  # measurement noise
                0, 100
            )

            exam_score_str = "" if i in unlabeled_ids else f"{exam_score:.1f}"

            rows.append([
                student_id, date.strftime("%Y-%m-%d"), subject,
                round(study_hours, 1), round(recall_score, 1), round(rest_hours, 1),
                int(sessions_count), int(distraction_events), exam_score_str
            ])

    # Sprinkle in a few intentionally invalid rows to confirm the C++
    # validator still catches bad data at this larger scale
    n_dirty = max(1, round(len(rows) * 0.02))
    dirty_idx = rng.choice(len(rows), size=n_dirty, replace=False)
    for idx in dirty_idx:
        corruption = rng.integers(0, 3)
        if corruption == 0:
            rows[idx][3] = 999  # study_hours out of range
        elif corruption == 1:
            rows[idx][5] = -1  # negative rest_hours
        else:
            rows[idx][4] = "NaN_bad"  # non-numeric recall score

    columns = [
        "student_id", "date", "subject", "study_hours", "active_recall_score",
        "rest_hours", "sessions_count", "distraction_events", "exam_score",
    ]
    return pd.DataFrame(rows, columns=columns)


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/synthetic_logs.csv"
    n_students = int(sys.argv[2]) if len(sys.argv) > 2 else 220

    df = generate(n_students=n_students)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} log rows for {n_students} students -> {output_path}")


if __name__ == "__main__":
    main()
