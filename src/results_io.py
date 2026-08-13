"""Utilities for saving final planner evaluation results."""

import csv
import json
from pathlib import Path


def save_planner_records_csv(
    records: list[dict],
    output_path: str | Path,
) -> Path:
    """Save per-scene planner records to a CSV file."""
    if not records:
        raise ValueError("records must not be empty.")

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        records[0].keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    return output_path


def save_summary_csv(
    summaries: dict[str, dict],
    output_path: str | Path,
) -> Path:
    """Save planner-level summary metrics to CSV."""
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_names = sorted(
        {
            metric_name
            for summary in summaries.values()
            for metric_name in summary.keys()
        }
    )

    fieldnames = [
        "method",
        *metric_names,
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for method_name, summary in summaries.items():
            row = {
                "method": method_name,
                **summary,
            }

            writer.writerow(row)

    return output_path


def save_statistical_results(
    statistical_results: dict,
    output_path: str | Path,
) -> Path:
    """Save statistical-test results as JSON."""
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            statistical_results,
            json_file,
            indent=2,
        )

    return output_path