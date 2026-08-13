"""Paired statistical analysis for planner evaluation."""

import numpy as np
from scipy.stats import wilcoxon


def extract_metric(
    records: list[dict],
    metric_name: str,
) -> np.ndarray:
    """Extract one metric from successful planner records."""
    return np.asarray(
        [
            record[metric_name]
            for record in records
            if record["success"]
        ],
        dtype=np.float64,
    )


def paired_wilcoxon(
    reference_records: list[dict],
    comparison_records: list[dict],
    metric_name: str,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """
    Perform a paired Wilcoxon signed-rank test.

    Records must correspond to the same scenes in the same order.
    """
    reference_values = extract_metric(
        reference_records,
        metric_name,
    )

    comparison_values = extract_metric(
        comparison_records,
        metric_name,
    )

    if reference_values.shape != comparison_values.shape:
        raise ValueError(
            "Paired records must contain the same number of samples."
        )

    differences = (
        comparison_values
        - reference_values
    )

    statistic, p_value = wilcoxon(
        comparison_values,
        reference_values,
        alternative=alternative,
        zero_method="wilcox",
    )

    return {
        "reference_mean":
            float(np.mean(reference_values)),
        "comparison_mean":
            float(np.mean(comparison_values)),
        "mean_difference":
            float(np.mean(differences)),
        "median_difference":
            float(np.median(differences)),
        "statistic":
            float(statistic),
        "p_value":
            float(p_value),
    }