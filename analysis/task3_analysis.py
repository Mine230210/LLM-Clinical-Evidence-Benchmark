"""Task 3 Evidence Applicability: frozen verdict and key-reason analysis.

Verdict results are reproducible from the public Gold and Responses CSV files.
Key-reason results require the unreleased human rating for every response and
are computed only when ``--reason-ratings`` is supplied. The script never infers a
human rating from model text.
"""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["ChatGPT", "DeepSeek", "Gemini"]
VERDICTS = ["DIRECTLY APPLICABLE", "PARTLY APPLICABLE", "NOT APPLICABLE"]
REASON_RATINGS = ["正确", "部分正确/部分错误", "错误"]
VERDICT_TO_CODE = {value: index for index, value in enumerate(VERDICTS)}
REASON_TO_CODE = {value: index for index, value in enumerate(REASON_RATINGS)}

BOOTSTRAP_REPS = 20_000
PERMUTATION_REPS = 100_000
BOOTSTRAP_SEED = 20261026
PERMUTATION_SEED = 20261027


def normalize_space(value: object) -> str:
    text = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value))
    return " ".join(text.strip().split())


def normalize_verdict(value: object) -> str:
    return normalize_space(value).upper()


def normalize_reason_rating(value: object) -> str:
    text = normalize_space(value)
    aliases = {
        "正确": "正确",
        "部分正确/部分错误": "部分正确/部分错误",
        "部分正确／部分错误": "部分正确/部分错误",
        "错误": "错误",
        "CORRECT": "正确",
        "PARTIALLY CORRECT": "部分正确/部分错误",
        "PARTIALLY CORRECT/PARTIALLY INCORRECT": "部分正确/部分错误",
        "INCORRECT": "错误",
    }
    normalized = aliases.get(text, aliases.get(text.upper()))
    if normalized is None:
        raise ValueError(f"Unrecognized human key-reason rating: {value!r}")
    return normalized


def percentile_ci(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan, 0
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high), int(finite.size)


def cohen_kappa_codes(a: np.ndarray, b: np.ndarray, n_categories: int) -> float:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if a.size == 0 or a.size != b.size:
        return math.nan
    observed = float(np.mean(a == b))
    pa = np.bincount(a, minlength=n_categories) / a.size
    pb = np.bincount(b, minlength=n_categories) / b.size
    expected = float(np.sum(pa * pb))
    return math.nan if math.isclose(1.0 - expected, 0.0, abs_tol=1e-15) else (observed - expected) / (1.0 - expected)


def paired_sign_flip_pvalue(differences: np.ndarray, reps: int, seed: int) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(reps, differences.size), dtype=np.int8) * 2 - 1
    permuted = np.mean(signs * differences[None, :], axis=1)
    extreme = int(np.sum(np.abs(permuted) >= abs(observed) - 1e-15))
    return (extreme + 1) / (reps + 1)


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def validate_and_score(gold_path: Path, responses_path: Path, ratings_path: Path | None) -> tuple[pd.DataFrame, bool]:
    gold = pd.read_csv(gold_path, dtype=str, keep_default_na=False)
    responses = pd.read_csv(responses_path, dtype=str, keep_default_na=False)
    required_gold = {"item_id", "acceptable_gold_verdicts"}
    required_responses = {"item_id", "model", "repeat_id", "type", "key_reasons"}
    if missing := required_gold - set(gold.columns):
        raise ValueError(f"Gold file is missing columns: {sorted(missing)}")
    if missing := required_responses - set(responses.columns):
        raise ValueError(f"Responses file is missing columns: {sorted(missing)}")
    if len(gold) != 60 or gold["item_id"].nunique() != 60:
        raise ValueError("Task 3 requires 60 unique Gold scenarios")
    if len(responses) != 360:
        raise ValueError("Task 3 requires 360 response rows")

    gold = gold.copy()
    responses = responses.copy()
    gold["item_id"] = gold["item_id"].astype(int)
    responses["item_id"] = responses["item_id"].astype(int)
    responses["repeat_id"] = responses["repeat_id"].astype(int)
    gold["gold_labels"] = gold["acceptable_gold_verdicts"].map(
        lambda value: tuple(normalize_verdict(label) for label in str(value).split("|") if normalize_space(label))
    )
    invalid_gold = sorted({label for labels in gold["gold_labels"] for label in labels} - set(VERDICTS))
    if invalid_gold:
        raise ValueError(f"Invalid Gold labels: {invalid_gold}")

    responses["verdict_normalized"] = responses["type"].map(normalize_verdict)
    invalid_predictions = sorted(set(responses["verdict_normalized"]) - set(VERDICTS))
    if invalid_predictions:
        raise ValueError(f"Invalid Task 3 prediction labels: {invalid_predictions}")
    if set(responses["model"]) != set(MODELS) or set(responses["repeat_id"]) != {1, 2}:
        raise ValueError("Models or repeat identifiers do not match the frozen design")
    keys = ["item_id", "model", "repeat_id"]
    if responses.duplicated(keys).any():
        raise ValueError("Duplicate item/model/repeat response keys")
    counts = responses.groupby(["item_id", "model"]).size()
    if len(counts) != 60 * 3 or not (counts == 2).all():
        raise ValueError("Every scenario/model combination must have two responses")

    scored = responses.merge(gold.drop(columns=[c for c in ["clinical_scenario", "source_rct_pmid"] if c in gold]), on="item_id", how="left", validate="many_to_one")
    scored["verdict_correct"] = [int(prediction in labels) for prediction, labels in zip(scored["verdict_normalized"], scored["gold_labels"])]

    has_ratings = ratings_path is not None
    if has_ratings:
        ratings = pd.read_csv(ratings_path, dtype=str, keep_default_na=False)
        rating_column = next((c for c in ["human_key_reason_rating", "Key_Reason_Human_Rating_Raw", "key_reason_rating"] if c in ratings.columns), None)
        if rating_column is None or not set(keys).issubset(ratings.columns):
            raise ValueError("Ratings file needs item_id, model, repeat_id and a human rating column")
        ratings = ratings[keys + [rating_column]].copy()
        ratings["item_id"] = ratings["item_id"].astype(int)
        ratings["repeat_id"] = ratings["repeat_id"].astype(int)
        ratings["key_reason_rating"] = ratings[rating_column].map(normalize_reason_rating)
        if len(ratings) != 360 or ratings.duplicated(keys).any():
            raise ValueError("Ratings file must contain exactly one rating for each of 360 responses")
        scored = scored.merge(ratings[keys + ["key_reason_rating"]], on=keys, how="left", validate="one_to_one")
        if scored["key_reason_rating"].isna().any():
            raise ValueError("Human ratings are incomplete")
        scored["key_reason_correct"] = (scored["key_reason_rating"] == "正确").astype(int)
    return scored, has_ratings


def analyze(scored: pd.DataFrame, has_ratings: bool) -> dict[str, pd.DataFrame]:
    item_ids = sorted(scored["item_id"].unique())
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    model_index = {model: index for index, model in enumerate(MODELS)}
    verdict_correct = np.zeros((60, 3, 2), dtype=np.int8)
    verdict_code = np.zeros((60, 3, 2), dtype=np.int8)
    key_correct = np.zeros((60, 3, 2), dtype=np.int8) if has_ratings else None
    reason_code = np.zeros((60, 3, 2), dtype=np.int8) if has_ratings else None
    for row in scored.itertuples(index=False):
        i, j, r = item_index[row.item_id], model_index[row.model], row.repeat_id - 1
        verdict_correct[i, j, r] = row.verdict_correct
        verdict_code[i, j, r] = VERDICT_TO_CODE[row.verdict_normalized]
        if has_ratings:
            key_correct[i, j, r] = row.key_reason_correct
            reason_code[i, j, r] = REASON_TO_CODE[row.key_reason_rating]

    strict = np.all(verdict_correct == 1, axis=2).astype(np.int8)
    lenient = np.any(verdict_correct == 1, axis=2).astype(np.int8)
    verdict_agreement = (verdict_code[:, :, 0] == verdict_code[:, :, 1]).astype(np.int8)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_idx = rng.integers(0, 60, size=(BOOTSTRAP_REPS, 60), dtype=np.int16)
    boot_vacc = verdict_correct[boot_idx].mean(axis=(1, 3)) * 100.0
    boot_strict = strict[boot_idx].mean(axis=1) * 100.0
    boot_lenient = lenient[boot_idx].mean(axis=1) * 100.0
    boot_voa = verdict_agreement[boot_idx].mean(axis=1) * 100.0

    boot_vkappa = np.full((BOOTSTRAP_REPS, 3), np.nan)
    for b, ids in enumerate(boot_idx):
        for j in range(3):
            boot_vkappa[b, j] = cohen_kappa_codes(verdict_code[ids, j, 0], verdict_code[ids, j, 1], 3)

    performance: list[dict] = []
    for j, model in enumerate(MODELS):
        rows = [
            ("Verdict Accuracy", int(verdict_correct[:, j, :].sum()), 120, verdict_correct[:, j, :].mean() * 100, boot_vacc[:, j]),
            ("Strict Verdict Accuracy", int(strict[:, j].sum()), 60, strict[:, j].mean() * 100, boot_strict[:, j]),
            ("Lenient Verdict Accuracy", int(lenient[:, j].sum()), 60, lenient[:, j].mean() * 100, boot_lenient[:, j]),
        ]
        if has_ratings:
            boot_kacc = key_correct[boot_idx].mean(axis=(1, 3)) * 100.0
            rows.insert(1, ("Key-reason Accuracy", int(key_correct[:, j, :].sum()), 120, key_correct[:, j, :].mean() * 100, boot_kacc[:, j]))
        for metric, numerator, denominator, estimate, boot_values in rows:
            low, high, valid = percentile_ci(boot_values)
            performance.append({"Model": model, "Metric": metric, "Numerator": numerator, "Denominator": denominator, "Estimate_Percent": float(estimate), "CI_Low_Percent": low, "CI_High_Percent": high, "Valid_Bootstrap_Replicates": valid})

    comparisons: list[dict] = []
    raw_pvalues: list[float] = []
    for pair_index, (model_a, model_b) in enumerate([("ChatGPT", "DeepSeek"), ("ChatGPT", "Gemini"), ("DeepSeek", "Gemini")]):
        a, b = model_index[model_a], model_index[model_b]
        scenario_differences = (verdict_correct[:, a, :] - verdict_correct[:, b, :]).mean(axis=1)
        raw_p = paired_sign_flip_pvalue(scenario_differences, PERMUTATION_REPS, PERMUTATION_SEED + pair_index)
        raw_pvalues.append(raw_p)
        low, high, valid = percentile_ci(boot_vacc[:, a] - boot_vacc[:, b])
        comparisons.append({"Outcome": "Verdict Accuracy", "Model_A": model_a, "Model_B": model_b, "Difference_A_minus_B_pp": float(scenario_differences.mean() * 100), "CI_Low_pp": low, "CI_High_pp": high, "Raw_P_Value": raw_p, "Holm_Adjusted_P_Value": math.nan, "Valid_Bootstrap_Replicates": valid})
    for row, adjusted in zip(comparisons, holm_adjust(raw_pvalues)):
        row["Holm_Adjusted_P_Value"] = adjusted

    reliability: list[dict] = []
    for j, model in enumerate(MODELS):
        vkappa = cohen_kappa_codes(verdict_code[:, j, 0], verdict_code[:, j, 1], 3)
        entries = [
            ("Verdict Observed Agreement", int(verdict_agreement[:, j].sum()), float(verdict_agreement[:, j].mean() * 100), boot_voa[:, j], "percent"),
            ("Verdict Cohen's kappa", None, vkappa, boot_vkappa[:, j], "kappa"),
        ]
        if has_ratings:
            reason_agreement = (reason_code[:, :, 0] == reason_code[:, :, 1]).astype(np.int8)
            boot_roa = reason_agreement[boot_idx].mean(axis=1) * 100.0
            boot_rkappa = np.full((BOOTSTRAP_REPS, 3), np.nan)
            for b, ids in enumerate(boot_idx):
                for k in range(3):
                    boot_rkappa[b, k] = cohen_kappa_codes(reason_code[ids, k, 0], reason_code[ids, k, 1], 3)
            rkappa = cohen_kappa_codes(reason_code[:, j, 0], reason_code[:, j, 1], 3)
            entries.extend([
                ("Key-reason Observed Agreement", int(reason_agreement[:, j].sum()), float(reason_agreement[:, j].mean() * 100), boot_roa[:, j], "percent"),
                ("Key-reason Cohen's kappa", None, rkappa, boot_rkappa[:, j], "kappa"),
            ])
        for metric, numerator, estimate, boot_values, scale in entries:
            low, high, valid = percentile_ci(boot_values)
            reliability.append({"Model": model, "Metric": metric, "Numerator": numerator, "Denominator_Pairs": 60, "Estimate": estimate, "CI_Low": low, "CI_High": high, "Scale": scale, "Valid_Bootstrap_Replicates": valid})

    result = {
        "performance": pd.DataFrame(performance),
        "model_comparisons": pd.DataFrame(comparisons),
        "reliability": pd.DataFrame(reliability),
    }
    if has_ratings:
        boot_overall_v = verdict_correct[boot_idx].mean(axis=(1, 2, 3)) * 100.0
        boot_overall_k = key_correct[boot_idx].mean(axis=(1, 2, 3)) * 100.0
        overall_diff = (verdict_correct - key_correct).mean(axis=(1, 2))
        overall_p = paired_sign_flip_pvalue(overall_diff, PERMUTATION_REPS, PERMUTATION_SEED + 100)
        overall_rows = []
        for metric, numerator, estimate, boots, pvalue in [
            ("Overall Verdict Accuracy", int(verdict_correct.sum()), verdict_correct.mean() * 100, boot_overall_v, math.nan),
            ("Overall Key-reason Accuracy", int(key_correct.sum()), key_correct.mean() * 100, boot_overall_k, math.nan),
            ("Verdict minus Key-reason Accuracy", math.nan, (verdict_correct.mean() - key_correct.mean()) * 100, boot_overall_v - boot_overall_k, overall_p),
        ]:
            low, high, valid = percentile_ci(boots)
            overall_rows.append({"Metric": metric, "Numerator": numerator, "Denominator_or_Pairs": 360, "Estimate_or_Difference_pp": float(estimate), "CI_Low": low, "CI_High": high, "P_Value_Two_Sided": pvalue, "Valid_Bootstrap_Replicates": valid})
        result["overall_comparison"] = pd.DataFrame(overall_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument(
        "--reason-ratings",
        "--ratings",
        dest="reason_ratings",
        type=Path,
        help="Optional restricted response-level human key-reason ratings CSV",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scored, has_ratings = validate_and_score(args.gold, args.responses, args.reason_ratings)
    results = analyze(scored, has_ratings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.drop(columns=["gold_labels"]).to_csv(args.output_dir / "scored_responses.csv", index=False, encoding="utf-8-sig")
    for name, frame in results.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "task": "Task 3 - Evidence Applicability",
        "public_reproduction": "verdict metrics only" if not has_ratings else "full formal analysis with supplied human ratings",
        "human_key_reason_ratings_supplied": has_ratings,
        "bootstrap": {"replicates": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "unit": "scenario"},
        "permutation": {"replicates": PERMUTATION_REPS, "seed_base": PERMUTATION_SEED, "unit": "scenario"},
        "model_comparison_family": "three verdict-accuracy comparisons, Holm adjusted",
        "overall_verdict_vs_key_reason": "separate exploratory comparison; requires human ratings",
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([
        {
            "Component": "Task 3 key-reason analyses",
            "Status": "RUN" if has_ratings else "NOT_RUN_RESTRICTED_INPUT_REQUIRED",
            "Required_Input": "response-level human key-reason ratings",
            "Public_Response_Text_Used_To_Infer_Ratings": False,
        }
    ]).to_csv(args.output_dir / "restricted_metrics_status.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
