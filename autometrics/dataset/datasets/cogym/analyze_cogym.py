#!/usr/bin/env python3
"""
Compute and visualize statistics for a conversation CSV file,
and save both text summaries and figures to an output directory.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Load a CSV of conversation data and report summary statistics."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the conversation CSV (with columns like userId, task, ratings, etc.)"
    )
    parser.add_argument(
        "--output_dir",
        default="./analysis",
        help="Directory to save output files (default: ./analysis)"
    )
    args = parser.parse_args()

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load into DataFrame
    df = pd.read_csv(args.csv_file)

    # Ensure full table visibility when printing
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Convert rating columns to numeric (coerce non-numeric / blanks to NaN)
    rating_cols = ["agentRating", "communicationRating", "outcomeRating"]
    df[rating_cols] = df[rating_cols].apply(pd.to_numeric, errors="coerce")

    # 1. Number of rows per user
    rows_per_user = df["userId"].value_counts().sort_index()
    print("=== Number of rows per user ===")
    print(rows_per_user, "\n")
    rows_per_user.to_csv(os.path.join(args.output_dir, "rows_per_user.csv"))

    # Bar plot: rows per user
    plt.figure(figsize=(8, 4))
    ax = rows_per_user.plot(kind="bar")
    plt.title("Rows per User")
    plt.xlabel("User ID")
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(
            str(int(p.get_height())),
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha='center', va='bottom'
        )
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "rows_per_user_bar.pdf"))
    plt.close()

    # Histogram: distribution of rows-per-user (20 bins), with counts
    plt.figure(figsize=(6, 4))
    ax = rows_per_user.plot(kind="hist", bins=20)
    plt.title("Histogram of Rows per User")
    plt.xlabel("Rows per User")
    plt.ylabel("Frequency")
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                str(int(height)),
                (p.get_x() + p.get_width() / 2, height),
                ha='center', va='bottom'
            )
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "rows_per_user_hist.pdf"))
    plt.close()

    # 2. Number of rows per task type
    rows_per_task = df["task"].value_counts()
    print("=== Number of rows per task ===")
    print(rows_per_task, "\n")
    rows_per_task.to_csv(os.path.join(args.output_dir, "rows_per_task.csv"))

    # Bar plot: rows per task, with counts above bars
    plt.figure(figsize=(6, 4))
    ax = rows_per_task.plot(kind="bar")
    plt.title("Rows per Task Type")
    plt.xlabel("Task")
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(
            str(int(p.get_height())),
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha='center', va='bottom'
        )
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "rows_per_task_bar.pdf"))
    plt.close()

    # 3. Amount of blank (missing) cells per column
    missing_per_col = df.isna().sum()
    print("=== Missing values per column ===")
    print(missing_per_col, "\n")
    missing_per_col.to_frame(name="missingCount") \
        .to_csv(os.path.join(args.output_dir, "missing_per_column.csv"))

    # 4. Rows where ALL cells have values
    total_rows = len(df)
    complete_rows = df.dropna().shape[0]
    print(f"Total rows: {total_rows}")
    print(f"Rows with NO missing values: {complete_rows}\n")

    # 5. Rows where JUST agentFeedback is missing
    mask_agent_missing = df["agentFeedback"].isna()
    mask_others_present = df.drop(columns=["agentFeedback"]).notna().all(axis=1)
    just_agent_missing = df[mask_agent_missing & mask_others_present].shape[0]
    print(f"Rows where ONLY agentFeedback is missing: {just_agent_missing}\n")

    # 6. Per-user statistics: average ratings, task count, and feedback count
    user_means = df.groupby("userId")[rating_cols].mean()
    user_counts = df.groupby("userId").size().rename("taskCount")
    feedback_counts = df.groupby("userId")["agentFeedback"].apply(lambda x: x.notna().sum()).rename("feedbackCount")
    user_stats = pd.concat([user_means, user_counts, feedback_counts], axis=1)
    print("=== Average ratings, task count, and feedback count per user ===")
    print(user_stats, "\n")
    user_stats.to_csv(os.path.join(args.output_dir, "user_stats.csv"))

    # Bar plot: average ratings per user
    plt.figure(figsize=(10, 5))
    ax = user_means.plot(kind="bar")
    plt.title("Average Ratings by User")
    plt.xlabel("User ID")
    plt.ylabel("Rating")
    plt.legend(title="Rating Type")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "avg_ratings_by_user.pdf"))
    plt.close()

    # 7. Overall averages across the entire dataset
    overall_means = df[rating_cols].mean()
    overall_mean_all = overall_means.mean()
    print("=== Overall average per rating type ===")
    print(overall_means, "\n")
    print(f"Overall average across all ratings: {overall_mean_all:.3f}")

    # 8. Rows with outcomeRating but no outcome value
    missing_outcome = df["outcomeRating"].notna() & df["outcome"].isna()
    count_missing_outcome = missing_outcome.sum()
    print(f"Rows with outcomeRating but no outcome value: {count_missing_outcome}")

    # Save overall means to CSV and summary to text
    overall_means.to_frame(name="mean") \
        .to_csv(os.path.join(args.output_dir, "overall_means.csv"))

    summary_lines = [
        f"Total rows: {total_rows}",
        f"Rows with NO missing values: {complete_rows}",
        f"Rows where ONLY agentFeedback is missing: {just_agent_missing}",
        f"Rows with outcomeRating but no outcome value: {count_missing_outcome}",
        "",
        "Overall average per rating type:",
        overall_means.to_string(),
        f"Overall average across all ratings: {overall_mean_all:.3f}"
    ]
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(summary_lines))

if __name__ == "__main__":
    main()