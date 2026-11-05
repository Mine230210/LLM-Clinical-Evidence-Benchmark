"""Task 2 Evidence Consensus: frozen parser, scoring and inference."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["DeepSeek", "Gemini", "ChatGPT"]
PRED_LEVELS = ["HIGH", "LOW", "INDETERMINATE", "INVALID"]
BOOTSTRAP_REPS = 20_000
PERMUTATION_REPS = 100_000
BOOTSTRAP_SEED = 20261023
MODEL_PERMUTATION_SEED = 20261024
STRATUM_PERMUTATION_SEED = 20261025


def normalized_text(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def parse_final_level(value: object) -> tuple[str, str, str]:
    """Map response text without access to the Gold label."""
    normalized = normalized_text(value)
    if not normalized:
        return "INVALID", "INVALID_EMPTY", normalized
    clean = normalized.strip(" \t\r\n\"'`[]{}")
    low = clean.casefold()
    has_high = bool(re.search(r"\bhigh\b.*\bcontrovers", low))
    has_low = bool(re.search(r"\blow\b.*\bcontrovers", low))
    if has_high and has_low:
        return "INVALID", "INVALID_MIXED_HIGH_LOW", normalized
    exact = {
        "high controversy": "HIGH", "high": "HIGH",
        "low controversy": "LOW", "low": "LOW",
        "indeterminate": "INDETERMINATE",
    }
    if low in exact:
        canonical = {"HIGH": "High Controversy", "LOW": "Low Controversy", "INDETERMINATE": "Indeterminate"}[exact[low]]
        rule = "EXACT_CANONICAL" if clean == canonical else "CASE_WHITESPACE_OR_SHORT_EQUIVALENT"
        return exact[low], rule, normalized
    if low == "high controvers":
        return "HIGH", "KNOWN_TRUNCATION_HIGH_CONTROVERS", normalized
    for pattern, label in [
        (r"^high controversy\s*(?:\(|\[|:|\-|—).+", "HIGH"),
        (r"^low controversy\s*(?:\(|\[|:|\-|—).+", "LOW"),
        (r"^indeterminate\s*(?:\(|\[|:|\-|—).+", "INDETERMINATE"),
    ]:
        if re.fullmatch(pattern, low):
            return label, "CANONICAL_WITH_EXPLANATORY_SUFFIX", normalized
    return "INVALID", "INVALID_UNRECOGNIZED", normalized


PARSER_CASES = [
    ("High Controversy", "HIGH"),
    (" low   controversy ", "LOW"),
    ("Low Controversy (Consistent Direction)", "LOW"),
    ("High Controvers", "HIGH"),
    ("HIGH", "HIGH"),
    ("Indeterminate: insufficient evidence", "INDETERMINATE"),
    ("High Controversy / Low Controversy", "INVALID"),
    ("Support", "INVALID"),
    ("Low risk", "INVALID"),
    ("", "INVALID"),
]


def kappa_from_pairs(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return math.nan
    observed = float(np.mean(a == b))
    expected = sum(float(np.mean(a == level)) * float(np.mean(b == level)) for level in PRED_LEVELS)
    return math.nan if np.isclose(1.0 - expected, 0.0) else (observed - expected) / (1.0 - expected)


def bootstrap_kappa(label_codes: np.ndarray, draws: np.ndarray) -> np.ndarray:
    a = label_codes[draws, 0]
    b = label_codes[draws, 1]
    observed = (a == b).mean(axis=1)
    expected = np.zeros(len(draws), dtype=float)
    for code in range(len(PRED_LEVELS)):
        expected += (a == code).mean(axis=1) * (b == code).mean(axis=1)
    denominator = 1.0 - expected
    return np.divide(observed - expected, denominator, out=np.full_like(observed, np.nan), where=~np.isclose(denominator, 0.0))


def percentile_ci(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan, math.nan, 0
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high), int(len(finite))


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def load_and_score(gold_path: Path, responses_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = pd.read_csv(gold_path, dtype=str, keep_default_na=False)
    responses = pd.read_csv(responses_path, dtype=str, keep_default_na=False)
    if len(gold) != 36 or len(responses) != 216:
        raise ValueError(f"Expected 36 Gold questions and 216 responses; found {len(gold)} and {len(responses)}")
    required_gold = {"item_id", "gold_controversy_level"}
    required_response = {"item_id", "model", "repeat_id", "final_level"}
    if not required_gold.issubset(gold.columns) or not required_response.issubset(responses.columns):
        raise ValueError("Task 2 input columns do not match the public data schema")
    gold["item_id"] = gold["item_id"].astype(int)
    responses["item_id"] = responses["item_id"].astype(int)
    responses["repeat_id"] = responses["repeat_id"].astype(int)
    if set(gold["gold_controversy_level"]) != {"HIGH", "LOW"}:
        raise ValueError("Gold must contain only HIGH and LOW")
    if set(responses["model"]) != set(MODELS) or set(responses["repeat_id"]) != {1, 2}:
        raise ValueError("Unexpected model or repeat labels")
    if responses.duplicated(["item_id", "model", "repeat_id"]).any():
        raise ValueError("Duplicate item/model/repeat key")
    parser_failures = [(value, expected, parse_final_level(value)[0]) for value, expected in PARSER_CASES if parse_final_level(value)[0] != expected]
    if parser_failures:
        raise RuntimeError(f"Frozen parser tests failed: {parser_failures}")
    gold_lookup = gold.set_index("item_id")["gold_controversy_level"]
    rows: list[dict] = []
    for row in responses.to_dict("records"):
        predicted, rule, normalized = parse_final_level(row["final_level"])
        truth = gold_lookup.loc[int(row["item_id"])]
        rows.append({
            **row,
            "Gold_Label": truth,
            "Predicted_Label": predicted,
            "Parser_Rule": rule,
            "Normalized_Final_Level": normalized,
            "Correct": int(predicted == truth),
        })
    scored = pd.DataFrame(rows).sort_values(["item_id", "model", "repeat_id"]).reset_index(drop=True)
    if not scored.groupby(["item_id", "model"]).size().eq(2).all():
        raise ValueError("Each question/model must have two responses")
    return gold.sort_values("item_id").reset_index(drop=True), scored


def run_analysis(gold: pd.DataFrame, scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tasks = gold["item_id"].tolist()
    task_pos = {task: index for index, task in enumerate(tasks)}
    model_pos = {model: index for index, model in enumerate(MODELS)}
    correct = np.zeros((36, 3, 2), dtype=np.int8)
    labels = np.empty((36, 3, 2), dtype=object)
    gold_labels = gold["gold_controversy_level"].to_numpy()
    for row in scored.itertuples(index=False):
        i, j, r = task_pos[int(row.item_id)], model_pos[row.model], int(row.repeat_id) - 1
        correct[i, j, r] = int(row.Correct)
        labels[i, j, r] = row.Predicted_Label

    high_idx = np.where(gold_labels == "HIGH")[0]
    low_idx = np.where(gold_labels == "LOW")[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws_high = rng.choice(high_idx, size=(BOOTSTRAP_REPS, len(high_idx)), replace=True)
    draws_low = rng.choice(low_idx, size=(BOOTSTRAP_REPS, len(low_idx)), replace=True)
    draws_all = np.concatenate([draws_low, draws_high], axis=1)

    performance_rows: list[dict] = []
    reliability_rows: list[dict] = []
    strict_lenient_rows: list[dict] = []
    boot_accuracy: dict[str, dict[str, np.ndarray]] = {}
    for model in MODELS:
        j = model_pos[model]
        values = correct[:, j, :]
        boot_overall = values[draws_all].mean(axis=(1, 2))
        boot_high = values[draws_high].mean(axis=(1, 2))
        boot_low = values[draws_low].mean(axis=(1, 2))
        boot_balanced = (boot_high + boot_low) / 2
        boot_accuracy[model] = {"Overall": boot_overall, "High": boot_high, "Low": boot_low, "Balanced": boot_balanced}
        high_point = values[high_idx].mean()
        low_point = values[low_idx].mean()
        for metric, ids, point, boot in [
            ("Overall Accuracy", np.arange(36), values.mean(), boot_overall),
            ("High Controversy Accuracy", high_idx, high_point, boot_high),
            ("Low Controversy Accuracy", low_idx, low_point, boot_low),
            ("Balanced Accuracy", None, (high_point + low_point) / 2, boot_balanced),
        ]:
            low, high, valid = percentile_ci(boot)
            performance_rows.append({
                "Model": model, "Metric": metric,
                "Correct": int(values[ids].sum()) if ids is not None else pd.NA,
                "Denominator": int(values[ids].size) if ids is not None else pd.NA,
                "Estimate": float(point), "CI95_Lower": low, "CI95_Upper": high,
                "Valid_Bootstrap_Replicates": valid,
            })
        pair_agree = labels[:, j, 0] == labels[:, j, 1]
        strict = values.all(axis=1)
        lenient = values.any(axis=1)
        code_map = {label: index for index, label in enumerate(PRED_LEVELS)}
        label_codes = np.vectorize(code_map.__getitem__)(labels[:, j, :])
        for group, ids, draws in [("Overall", np.arange(36), draws_all), ("High", high_idx, draws_high), ("Low", low_idx, draws_low)]:
            agreement_boot = pair_agree[draws].mean(axis=1)
            kappa_boot = bootstrap_kappa(label_codes, draws)
            a_low, a_high, a_valid = percentile_ci(agreement_boot)
            k_low, k_high, k_valid = percentile_ci(kappa_boot)
            reliability_rows.extend([
                {"Model": model, "Group": group, "Metric": "Observed Agreement", "Numerator": int(pair_agree[ids].sum()), "Denominator": len(ids), "Estimate": float(pair_agree[ids].mean()), "CI95_Lower": a_low, "CI95_Upper": a_high, "Valid_Bootstrap_Replicates": a_valid},
                {"Model": model, "Group": group, "Metric": "Unweighted Cohen's kappa", "Numerator": pd.NA, "Denominator": len(ids), "Estimate": kappa_from_pairs(labels[ids, j, 0], labels[ids, j, 1]), "CI95_Lower": k_low, "CI95_Upper": k_high, "Valid_Bootstrap_Replicates": k_valid},
            ])
            for metric, array in [("Strict Accuracy", strict), ("Lenient Accuracy", lenient)]:
                boot = array[draws].mean(axis=1)
                low, high, valid = percentile_ci(boot)
                strict_lenient_rows.append({"Model": model, "Group": group, "Metric": metric, "Correct_Questions": int(array[ids].sum()), "Question_Denominator": len(ids), "Estimate": float(array[ids].mean()), "CI95_Lower": low, "CI95_Upper": high, "Valid_Bootstrap_Replicates": valid})

    pooled = correct.reshape(36, -1)
    boot_high = pooled[draws_high].mean(axis=(1, 2))
    boot_low = pooled[draws_low].mean(axis=(1, 2))
    boot_diff = boot_low - boot_high
    high_point = float(pooled[high_idx].mean())
    low_point = float(pooled[low_idx].mean())
    difference = low_point - high_point
    high_low, high_high, _ = percentile_ci(boot_high)
    low_low, low_high, _ = percentile_ci(boot_low)
    diff_low, diff_high, diff_valid = percentile_ci(boot_diff)
    rng = np.random.default_rng(STRATUM_PERMUTATION_SEED)
    question_accuracy = pooled.mean(axis=1)
    keys = rng.random((PERMUTATION_REPS, 36))
    perm_low_idx = np.argpartition(keys, len(low_idx) - 1, axis=1)[:, :len(low_idx)]
    perm_low_sum = question_accuracy[perm_low_idx].sum(axis=1)
    perm_diff = perm_low_sum / len(low_idx) - (question_accuracy.sum() - perm_low_sum) / len(high_idx)
    stratum_p = float((1 + np.sum(np.abs(perm_diff) >= abs(difference) - 1e-15)) / (PERMUTATION_REPS + 1))
    high_low_rows = [
        {"Gold_Group_or_Contrast": "Gold HIGH", "Correct": int(pooled[high_idx].sum()), "Denominator": int(pooled[high_idx].size), "Accuracy": high_point, "CI95_Lower": high_low, "CI95_Upper": high_high, "Difference_pp": pd.NA, "Difference_CI95_Lower_pp": pd.NA, "Difference_CI95_Upper_pp": pd.NA, "Raw_Two_Sided_P": pd.NA},
        {"Gold_Group_or_Contrast": "Gold LOW", "Correct": int(pooled[low_idx].sum()), "Denominator": int(pooled[low_idx].size), "Accuracy": low_point, "CI95_Lower": low_low, "CI95_Upper": low_high, "Difference_pp": pd.NA, "Difference_CI95_Lower_pp": pd.NA, "Difference_CI95_Upper_pp": pd.NA, "Raw_Two_Sided_P": pd.NA},
        {"Gold_Group_or_Contrast": "LOW minus HIGH", "Correct": pd.NA, "Denominator": pd.NA, "Accuracy": pd.NA, "CI95_Lower": pd.NA, "CI95_Upper": pd.NA, "Difference_pp": difference * 100, "Difference_CI95_Lower_pp": diff_low * 100, "Difference_CI95_Upper_pp": diff_high * 100, "Raw_Two_Sided_P": stratum_p, "Valid_Bootstrap_Replicates": diff_valid},
    ]

    comparisons: list[dict] = []
    rng = np.random.default_rng(MODEL_PERMUTATION_SEED)
    for left, right in [("DeepSeek", "Gemini"), ("DeepSeek", "ChatGPT"), ("Gemini", "ChatGPT")]:
        li, ri = model_pos[left], model_pos[right]
        differences = correct[:, li, :].mean(axis=1) - correct[:, ri, :].mean(axis=1)
        observed = float(differences.mean())
        boot_difference = boot_accuracy[left]["Overall"] - boot_accuracy[right]["Overall"]
        low, high, valid = percentile_ci(boot_difference)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(PERMUTATION_REPS, 36), replace=True)
        null = (signs * differences).mean(axis=1)
        pvalue = float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-15)) / (PERMUTATION_REPS + 1))
        comparisons.append({"Comparison": f"{left} minus {right}", "Model_1": left, "Model_2": right, "Accuracy_1": float(correct[:, li, :].mean()), "Accuracy_2": float(correct[:, ri, :].mean()), "Difference_pp": observed * 100, "CI95_Lower_pp": low * 100, "CI95_Upper_pp": high * 100, "Raw_P": pvalue, "Valid_Bootstrap_Replicates": valid})
    for row, adjusted in zip(comparisons, holm_adjust([row["Raw_P"] for row in comparisons])):
        row["Holm_Adjusted_P"] = adjusted
        row["Multiplicity_Family"] = "Three pairwise overall accuracy comparisons"

    confusion_rows: list[dict] = []
    for model in MODELS:
        subset = scored[scored["model"] == model]
        for truth in ["HIGH", "LOW"]:
            denominator = int((subset["Gold_Label"] == truth).sum())
            for prediction in PRED_LEVELS:
                count = int(((subset["Gold_Label"] == truth) & (subset["Predicted_Label"] == prediction)).sum())
                confusion_rows.append({"Model": model, "Gold_Label": truth, "Predicted_Label": prediction, "Count": count, "Gold_Row_Denominator": denominator, "Row_Proportion": count / denominator})
    return {
        "scored_responses": scored,
        "performance": pd.DataFrame(performance_rows),
        "strict_lenient": pd.DataFrame(strict_lenient_rows),
        "reliability": pd.DataFrame(reliability_rows),
        "confusion_matrix": pd.DataFrame(confusion_rows),
        "overall_high_low": pd.DataFrame(high_low_rows),
        "model_comparisons": pd.DataFrame(comparisons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Path to Task2_Gold.csv")
    parser.add_argument("--responses", type=Path, required=True, help="Path to Task2_Responses.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gold, scored = load_and_score(args.gold, args.responses)
    outputs = run_analysis(gold, scored)
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    metadata = {
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "permutation_replicates": PERMUTATION_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "model_permutation_seed": MODEL_PERMUTATION_SEED,
        "stratum_permutation_seed": STRATUM_PERMUTATION_SEED,
        "prediction_levels": PRED_LEVELS,
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
