"""Frozen Task 1 parsing and exact-reference scoring rules."""

from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation

import numpy as np


EMPTY_WORDS = {"", "none", "unknown", "null", "nan", "n/a", "na", "无", "没有", "未找到"}


def raw_text(value: object) -> str:
    return "" if value is None else str(value)


def normalize_q1(value: object) -> str:
    text = unicodedata.normalize("NFKC", raw_text(value)).strip()
    text = re.sub(r"^[\s\[\]{}()\"'“”‘’]+|[\s\[\]{}()\"'“”‘’]+$", "", text).strip().upper()
    return text if text in {"YES", "NO"} else "UNPARSEABLE"


def empty_identifier(value: object) -> bool:
    text = unicodedata.normalize("NFKC", raw_text(value)).strip()
    text = re.sub(r"[\[\]{}()\"'\s]", "", text).lower()
    return text in EMPTY_WORDS


def parse_nct(value: object) -> tuple[list[str], list[str]]:
    if empty_identifier(value):
        return [], []
    text = unicodedata.normalize("NFKC", raw_text(value)).upper()
    pattern = r"(?<![A-Z0-9])NCT[\s:-]*(\d{8})(?![A-Z0-9])"
    matches = list(re.finditer(pattern, text))
    identifiers = sorted({"NCT" + match.group(1) for match in matches})
    remainder = re.sub(pattern, " ", text)
    issues: list[str] = []
    if re.search("NCT", remainder):
        issues.append("INVALID_OR_INCOMPLETE_NCT")
    if not identifiers:
        issues.append("NONEMPTY_NCT_WITHOUT_VALID_ID")
    return identifiers, sorted(set(issues))


def parse_pmids(value: object, max_digits: int = 9) -> tuple[list[str], list[str]]:
    if empty_identifier(value):
        return [], []
    issues: list[str] = []
    identifiers: set[str] = set()
    text = unicodedata.normalize("NFKC", raw_text(value))
    pattern = r"(?<![\w.])[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?(?![\w.])"
    for match in re.finditer(pattern, text):
        token = match.group(0)
        try:
            number = Decimal(token)
        except InvalidOperation:
            issues.append("INVALID_NUMERIC_ATOM")
            continue
        if not number.is_finite() or number <= 0 or number != number.to_integral_value():
            issues.append("NONPOSITIVE_OR_NONINTEGER_PMID")
            continue
        integer = int(number)
        if len(str(integer)) > max_digits:
            issues.append("OVERLONG_PMID_INTEGER_NO_SPLITTING")
            continue
        if isinstance(value, float) and (not math.isfinite(value) or abs(value) > 2**53):
            issues.append("UNSAFE_EXCEL_FLOAT")
            continue
        identifiers.add(str(integer))
    if not identifiers:
        issues.append("NONEMPTY_PMID_WITHOUT_VALID_ID")
    return sorted(identifiers, key=lambda item: (len(item), item)), sorted(set(issues))


def exact_score(
    q1: str,
    ncts: list[str],
    pmids: list[str],
    gold_ncts: set[str],
    gold_pmids: set[str],
) -> dict[str, int | str]:
    nct_match = int(bool(set(ncts) & gold_ncts))
    pmid_match = int(bool(set(pmids) & gold_pmids))
    provenance = int(bool(nct_match or pmid_match))
    positive = int(q1 == "YES")
    return {
        "Q1_Positive": positive,
        "Q2_Gold_Match": provenance,
        "Joint_Success": positive * provenance,
        "NCT_Match": nct_match,
        "PMID_Match": pmid_match,
        "Match_Type": (
            "BOTH" if nct_match and pmid_match else
            "NCT_ONLY" if nct_match else
            "PMID_ONLY" if pmid_match else
            "NONE"
        ),
    }


def binary_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = float((a == b).mean())
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return math.nan if abs(1 - expected) < 1e-15 else (observed - expected) / (1 - expected)
