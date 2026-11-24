"""Task 1 Evidence Visibility: frozen scoring and statistical analysis.

The formal multiplicity family contains only the three pairwise model
comparisons. The pooled PubMed-linked versus Unlinked contrast is a separate
exploratory test; model-specific visibility contrasts are descriptive.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from task1_scoring import binary_kappa, exact_score, normalize_q1, parse_nct, parse_pmids


MODELS = ["ChatGPT", "DeepSeek", "Gemini"]
BOOTSTRAP_REPS = 20_000
PERMUTATION_REPS = 100_000
BOOTSTRAP_SEED = 2026102201
VISIBILITY_PERMUTATION_SEED = 2026102202
MODEL_PERMUTATION_SEED = 2026102203


def ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975], method="linear")
    return float(low), float(high)


def bootstrap_binary_kappa(
    paired_labels: np.ndarray,
    group: str,
    linked: np.ndarray,
    unlinked: np.ndarray,
    weights_l: np.ndarray,
    weights_u: np.ndarray,
) -> tuple[float, float, int]:
    """Estimate a percentile CI while preserving item-level paired structure."""
    a = np.asarray(paired_labels[:, 0], dtype=float)
    b = np.asarray(paired_labels[:, 1], dtype=float)
    agree = (a == b).astype(float)

    if group == "Overall":
        denominator = len(a)
        a_mean = (weights_l @ a[linked] + weights_u @ a[unlinked]) / denominator
        b_mean = (weights_l @ b[linked] + weights_u @ b[unlinked]) / denominator
        observed = (weights_l @ agree[linked] + weights_u @ agree[unlinked]) / denominator
    else:
        ids = linked if group == "PubMed-linked" else unlinked
        weights = weights_l if group == "PubMed-linked" else weights_u
        a_mean = weights @ a[ids] / len(ids)
        b_mean = weights @ b[ids] / len(ids)
        observed = weights @ agree[ids] / len(ids)

    expected = a_mean * b_mean + (1 - a_mean) * (1 - b_mean)
    kappa_denominator = 1 - expected
    valid = np.abs(kappa_denominator) >= 1e-15
    if not valid.any():
        return np.nan, np.nan, 0
    bootstrap_kappa = (observed[valid] - expected[valid]) / kappa_denominator[valid]
    low, high = ci(bootstrap_kappa)
    return low, high, int(valid.sum())


def holm_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def split_set(value: object) -> set[str]:
    return {part.strip() for part in str(value or "").split("|") if part.strip()}


def load_and_score(gold_path: Path, responses_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = pd.read_csv(gold_path, dtype=str, keep_default_na=False)
    responses = pd.read_csv(responses_path, dtype=str, keep_default_na=False)
    required_gold = {
        "item_id", "reference_nct_ids", "reference_pmids", "pubmed_linkage",
    }
    required_responses = {
        "item_id", "model", "repeat_id", "q1_risk_identified",
        "q2_specific_nct_ids", "q2_specific_pmids",
    }
    if not required_gold.issubset(gold.columns) or not required_responses.issubset(responses.columns):
        raise ValueError("Task 1 input columns do not match the public data schema")
    if len(gold) != 170 or len(responses) != 1020:
        raise ValueError(f"Expected 170 Gold items and 1,020 responses; found {len(gold)} and {len(responses)}")
    gold["item_id"] = gold["item_id"].astype(int)
    responses["item_id"] = responses["item_id"].astype(int)
    responses["repeat_id"] = responses["repeat_id"].astype(int)
    if set(responses["model"]) != set(MODELS) or set(responses["repeat_id"]) != {1, 2}:
        raise ValueError("Unexpected model or repeat labels")
    if responses.duplicated(["item_id", "model", "repeat_id"]).any():
        raise ValueError("Duplicate item/model/repeat key")
    if not responses.groupby(["item_id", "model"]).size().eq(2).all():
        raise ValueError("Each item/model must have two responses")

    gold_lookup = gold.set_index("item_id")
    scored_rows: list[dict] = []
    for row in responses.to_dict("records"):
        item = int(row["item_id"])
        if item not in gold_lookup.index:
            raise ValueError(f"No Gold row for item {item}")
        g = gold_lookup.loc[item]
        q1 = normalize_q1(row["q1_risk_identified"])
        ncts, nct_issues = parse_nct(row["q2_specific_nct_ids"])
        pmids, pmid_issues = parse_pmids(row["q2_specific_pmids"])
        scores = exact_score(
            q1,
            ncts,
            pmids,
            split_set(g["reference_nct_ids"]),
            split_set(g["reference_pmids"]),
        )
        scored_rows.append({
            **row,
            "Q1_Normalized": q1,
            "Parsed_NCT_IDs": "|".join(ncts),
            "Parsed_PMID_IDs": "|".join(pmids),
            "NCT_Parse_Issues": "|".join(nct_issues),
            "PMID_Parse_Issues": "|".join(pmid_issues),
            "Visibility": "PubMed-linked" if g["pubmed_linkage"] == "PUBMED_LINKED" else "Unlinked",
            **scores,
        })
    scored = pd.DataFrame(scored_rows).sort_values(["item_id", "model", "repeat_id"]).reset_index(drop=True)
    return gold, scored


def build_arrays(gold: pd.DataFrame, scored: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    items = sorted(gold["item_id"].astype(int).tolist())
    item_pos = {item: index for index, item in enumerate(items)}
    model_pos = {model: index for index, model in enumerate(MODELS)}
    endpoints = np.zeros((len(items), len(MODELS), 2, 3), dtype=np.int8)
    # Endpoint order: Joint Success, Q1 Positive, Answer Provenance Rate.
    for row in scored.itertuples(index=False):
        i = item_pos[int(row.item_id)]
        j = model_pos[row.model]
        r = int(row.repeat_id) - 1
        endpoints[i, j, r] = [row.Joint_Success, row.Q1_Positive, row.Q2_Gold_Match]
    visibility_by_item = gold.set_index("item_id").loc[items, "pubmed_linkage"]
    visibility = np.where(visibility_by_item.to_numpy() == "PUBMED_LINKED", "PubMed-linked", "Unlinked")
    return endpoints, visibility, np.asarray(items), items


def run_analysis(gold: pd.DataFrame, scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
    values, visibility, _, items = build_arrays(gold, scored)
    n_items = len(items)
    linked = np.flatnonzero(visibility == "PubMed-linked")
    unlinked = np.flatnonzero(visibility == "Unlinked")
    if len(linked) != 85 or len(unlinked) != 85:
        raise ValueError("Task 1 frozen analysis requires 85 linked and 85 unlinked items")

    average = values.mean(axis=2)
    strict = values.min(axis=2)
    lenient = values.max(axis=2)
    agreement = (values[:, :, 0, :] == values[:, :, 1, :]).astype(float)
    metrics = np.stack([average, strict, lenient, agreement], axis=3)
    # item x model x endpoint x metric, matching the frozen implementation.
    flat = metrics.reshape(n_items, -1)
    index: dict[tuple[str, str, str], int] = {}
    endpoints = ["Joint_Success", "Q1_Positive", "Q2_Gold_Match"]
    metric_names = ["Average", "Strict", "Lenient", "Agreement"]
    column = 0
    for model in MODELS:
        for endpoint in endpoints:
            for metric in metric_names:
                index[(model, endpoint, metric)] = column
                column += 1

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights_l = rng.multinomial(len(linked), np.full(len(linked), 1 / len(linked)), size=BOOTSTRAP_REPS)
    weights_u = rng.multinomial(len(unlinked), np.full(len(unlinked), 1 / len(unlinked)), size=BOOTSTRAP_REPS)
    boot_l = weights_l @ flat[linked] / len(linked)
    boot_u = weights_u @ flat[unlinked] / len(unlinked)
    boots = {"Overall": (boot_l + boot_u) / 2, "PubMed-linked": boot_l, "Unlinked": boot_u}

    performance_rows: list[dict] = []
    secondary_rows: list[dict] = []
    reliability_rows: list[dict] = []
    group_indices = {"Overall": np.arange(n_items), "PubMed-linked": linked, "Unlinked": unlinked}
    for j, model in enumerate(MODELS):
        for group, ids in group_indices.items():
            response_mask = (scored["model"] == model)
            if group != "Overall":
                response_mask &= scored["Visibility"] == group
            response_group = scored.loc[response_mask]
            joint = response_group["Joint_Success"].astype(int)
            item_joint = values[ids, j, :, 0]
            strict_arr = item_joint.min(axis=1)
            lenient_arr = item_joint.max(axis=1)
            agree_arr = (item_joint[:, 0] == item_joint[:, 1]).astype(int)
            overall_item_joint = values[:, j, :, 0]
            kappa = binary_kappa(item_joint[:, 0], item_joint[:, 1])
            kappa_lo, kappa_hi, kappa_valid = bootstrap_binary_kappa(
                overall_item_joint,
                group,
                linked,
                unlinked,
                weights_l,
                weights_u,
            )
            js_lo, js_hi = ci(boots[group][:, index[(model, "Joint_Success", "Average")]])
            strict_lo, strict_hi = ci(boots[group][:, index[(model, "Joint_Success", "Strict")]])
            lenient_lo, lenient_hi = ci(boots[group][:, index[(model, "Joint_Success", "Lenient")]])
            agree_lo, agree_hi = ci(boots[group][:, index[(model, "Joint_Success", "Agreement")]])
            performance_rows.append({
                "Model": model, "Group": group, "Gold_Items": len(ids), "Responses": len(response_group),
                "Joint_Successes": int(joint.sum()), "JSR": float(joint.mean()),
                "JSR_CI_Lower": js_lo, "JSR_CI_Upper": js_hi,
                "Strict_Count": int(strict_arr.sum()), "Strict_Rate": float(strict_arr.mean()),
                "Strict_CI_Lower": strict_lo, "Strict_CI_Upper": strict_hi,
                "Lenient_Count": int(lenient_arr.sum()), "Lenient_Rate": float(lenient_arr.mean()),
                "Lenient_CI_Lower": lenient_lo, "Lenient_CI_Upper": lenient_hi,
                "Agreement_Count": int(agree_arr.sum()), "Observed_Agreement": float(agree_arr.mean()),
                "Agreement_CI_Lower": agree_lo, "Agreement_CI_Upper": agree_hi,
                "Joint_Kappa": kappa,
                "Joint_Kappa_CI_Lower": kappa_lo, "Joint_Kappa_CI_Upper": kappa_hi,
                "Joint_Kappa_Valid_Bootstrap_Replicates": kappa_valid,
            })
            q1_lo, q1_hi = ci(boots[group][:, index[(model, "Q1_Positive", "Average")]])
            q2_lo, q2_hi = ci(boots[group][:, index[(model, "Q2_Gold_Match", "Average")]])
            secondary_rows.append({
                "Model": model, "Group": group, "Responses": len(response_group),
                "Q1_YES_Count": int(response_group["Q1_Positive"].sum()),
                "Q1_Positive_Rate": float(response_group["Q1_Positive"].mean()),
                "Q1_CI_Lower": q1_lo, "Q1_CI_Upper": q1_hi,
                "Q2_Match_Count": int(response_group["Q2_Gold_Match"].sum()),
                "APR": float(response_group["Q2_Gold_Match"].mean()),
                "APR_CI_Lower": q2_lo, "APR_CI_Upper": q2_hi,
                "Q1_Unparseable_Count": int((response_group["Q1_Normalized"] == "UNPARSEABLE").sum()),
            })
            reliability_rows.append({
                "Model": model, "Group": group, "Outcome": "Joint Success",
                "Observed_Agreement": float(agree_arr.mean()),
                "Agreement_CI_Lower": agree_lo, "Agreement_CI_Upper": agree_hi,
                "Cohen_Kappa": kappa,
                "Kappa_CI_Lower": kappa_lo, "Kappa_CI_Upper": kappa_hi,
                "Kappa_Valid_Bootstrap_Replicates": kappa_valid,
            })

    joint = average[:, :, 0]
    pairs = list(itertools.combinations(range(3), 2))
    pair_diff = np.stack([joint[:, left] - joint[:, right] for left, right in pairs], axis=1)
    observed_pair = pair_diff.mean(axis=0)
    rng = np.random.default_rng(MODEL_PERMUTATION_SEED)
    extreme = np.zeros(3, dtype=int)
    for start in range(0, PERMUTATION_REPS, 1000):
        count = min(1000, PERMUTATION_REPS - start)
        signs = rng.choice([-1, 1], size=(count, n_items))
        null = signs @ pair_diff / n_items
        extreme += (np.abs(null) >= np.abs(observed_pair) - 1e-12).sum(axis=0)
    model_rows: list[dict] = []
    raw_p: list[float] = []
    for k, (left, right) in enumerate(pairs):
        left_col = index[(MODELS[left], "Joint_Success", "Average")]
        right_col = index[(MODELS[right], "Joint_Success", "Average")]
        low, high = ci(boots["Overall"][:, left_col] - boots["Overall"][:, right_col])
        pvalue = float((extreme[k] + 1) / (PERMUTATION_REPS + 1))
        raw_p.append(pvalue)
        model_rows.append({
            "Comparison": f"{MODELS[left]} minus {MODELS[right]}",
            "Model_1": MODELS[left], "Model_2": MODELS[right],
            "Rate_1": float(joint[:, left].mean()), "Rate_2": float(joint[:, right].mean()),
            "Difference_pp": float(observed_pair[k] * 100),
            "CI95_Lower_pp": low * 100, "CI95_Upper_pp": high * 100,
            "Raw_P": pvalue,
        })
    for row, adjusted in zip(model_rows, holm_adjust(raw_p)):
        row["Holm_Adjusted_P"] = adjusted
        row["Multiplicity_Family"] = "Three pairwise Joint Success model comparisons"

    task_mean = joint.mean(axis=1)
    observed_visibility = float(task_mean[linked].mean() - task_mean[unlinked].mean())
    model_columns = [index[(model, "Joint_Success", "Average")] for model in MODELS]
    pooled_boot_l = boot_l[:, model_columns].mean(axis=1)
    pooled_boot_u = boot_u[:, model_columns].mean(axis=1)
    diff_low, diff_high = ci(pooled_boot_l - pooled_boot_u)
    rng = np.random.default_rng(VISIBILITY_PERMUTATION_SEED)
    extreme_count = 0
    for start in range(0, PERMUTATION_REPS, 1000):
        count = min(1000, PERMUTATION_REPS - start)
        perm = rng.permuted(np.broadcast_to(np.arange(n_items), (count, n_items)), axis=1)
        null = (2 * task_mean[perm[:, :85]].sum(axis=1) - task_mean.sum()) / 85
        extreme_count += int((np.abs(null) >= abs(observed_visibility) - 1e-12).sum())
    visibility_rows = []
    for j, model in enumerate(MODELS):
        model_low, model_high = ci(boot_l[:, model_columns[j]] - boot_u[:, model_columns[j]])
        visibility_rows.append({
            "Model": model,
            "Linked_Rate": float(joint[linked, j].mean()),
            "Unlinked_Rate": float(joint[unlinked, j].mean()),
            "Difference_pp": float((joint[linked, j].mean() - joint[unlinked, j].mean()) * 100),
            "CI95_Lower_pp": model_low * 100, "CI95_Upper_pp": model_high * 100,
            "Raw_P": np.nan, "Analysis_Role": "Descriptive; not formally tested",
        })
    visibility_rows.append({
        "Model": "Overall (equal models)",
        "Linked_Rate": float(task_mean[linked].mean()),
        "Unlinked_Rate": float(task_mean[unlinked].mean()),
        "Difference_pp": observed_visibility * 100,
        "CI95_Lower_pp": diff_low * 100, "CI95_Upper_pp": diff_high * 100,
        "Raw_P": float((extreme_count + 1) / (PERMUTATION_REPS + 1)),
        "Analysis_Role": "Exploratory separate single-test family; unadjusted P",
    })
    return {
        "scored_responses": scored,
        "performance": pd.DataFrame(performance_rows),
        "secondary_metrics": pd.DataFrame(secondary_rows),
        "reliability": pd.DataFrame(reliability_rows),
        "model_comparisons": pd.DataFrame(model_rows),
        "visibility_contrast": pd.DataFrame(visibility_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Path to Task1_Gold.csv")
    parser.add_argument("--responses", type=Path, required=True, help="Path to Task1_Responses.csv")
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
        "visibility_permutation_seed": VISIBILITY_PERMUTATION_SEED,
        "model_permutation_seed": MODEL_PERMUTATION_SEED,
        "kappa_ci_method": "20,000-replicate PubMed-linkage-stratified item-cluster percentile bootstrap",
        "formal_family": "Three pairwise Joint Success model comparisons; Holm adjusted",
        "exploratory_test": "Pooled PubMed-linked minus Unlinked; raw two-sided P",
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
