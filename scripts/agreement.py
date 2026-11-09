import json
from pathlib import Path

from scipy.stats import kendalltau, spearmanr


def comparable_score(method, metric, results):
    value = results["summary"][method][metric]

    if metric.startswith("deletion_"):
        return -value

    return value


def statistic_value(value: float) -> float | None:
    return None if value != value else float(value)


def main(input_path: Path, output_path: Path) -> None:
    with input_path.open(encoding="utf-8") as f:
        results = json.load(f)

    methods = list(results["summary"])
    metric_names = list(results["summary"][methods[0]])

    reference = [
        comparable_score(method, "gt_recovery", results)
        for method in methods
    ]

    agreement_with_ground_truth = {}
    for metric in metric_names:
        if metric == "gt_recovery":
            continue

        observed = [
            comparable_score(method, metric, results)
            for method in methods
        ]

        spearman = spearmanr(reference, observed)
        kendall = kendalltau(reference, observed)

        agreement_with_ground_truth[metric] = {
            "spearman": {
                "rho": statistic_value(spearman.statistic),
                "p_value": statistic_value(spearman.pvalue),
            },
            "kendall": {
                "tau": statistic_value(kendall.statistic),
                "p_value": statistic_value(kendall.pvalue),
            },
        }

    rankings = {}
    for metric in metric_names:
        rankings[metric] = sorted(
            methods,
            key=lambda method: comparable_score(method, metric, results),
            reverse=True,
        )

    spearman_matrix = {}
    kendall_matrix = {}
    for metric_a in metric_names:
        spearman_matrix[metric_a] = {}
        kendall_matrix[metric_a] = {}

        for metric_b in metric_names:
            values_a = [comparable_score(method, metric_a, results) for method in methods]
            values_b = [comparable_score(method, metric_b, results) for method in methods]
            spearman = spearmanr(values_a, values_b)
            kendall = kendalltau(values_a, values_b)

            spearman_matrix[metric_a][metric_b] = statistic_value(spearman.statistic)
            kendall_matrix[metric_a][metric_b] = statistic_value(kendall.statistic)

    agreement = {
        "source": str(input_path),
        "num_methods": len(methods),
        "methods": methods,
        "direction": {
            "higher_is_better": ["gt_recovery", "insertion_auc"],
            "lower_is_better": ["deletion_auc"],
            "note": "Deletion AUC values are negated before ranking and correlation.",
        },
        "rankings_best_to_worst": rankings,
        "agreement_with_ground_truth": agreement_with_ground_truth,
        "pairwise_spearman": spearman_matrix,
        "pairwise_kendall": kendall_matrix,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(agreement, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Saved agreement results to: {output_path}")


if __name__ == "__main__":
    main(input_path = Path("results/attributions_narrowband_0db/metrics.json"), 
         output_path = Path("results/agreement.json"))
