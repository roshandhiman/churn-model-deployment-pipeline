from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(np.nan_to_num(np.asarray(expected, dtype=float), nan=1e-6), 1e-6, None)
    actual = np.clip(np.nan_to_num(np.asarray(actual, dtype=float), nan=1e-6), 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(records: list[dict], profile: dict) -> dict:
    frame = pd.DataFrame(records)
    results = {}
    for column, reference in profile.items():
        if column not in frame:
            continue
        if reference["kind"] == "numeric":
            edges = np.asarray(reference["edges"], dtype=float)
            current = pd.to_numeric(frame[column], errors="coerce").dropna().clip(edges[0], edges[-1])
            actual = pd.cut(current, bins=edges, include_lowest=True).value_counts(normalize=True, sort=False).to_numpy()
            score = psi(np.asarray(reference["distribution"]), actual)
        else:
            baseline = reference["distribution"]
            values = frame[column].fillna("__missing__").astype(str).value_counts(normalize=True)
            categories = sorted(set(baseline) | set(values.index))
            score = psi(np.array([baseline.get(item, 0) for item in categories]), np.array([values.get(item, 0) for item in categories]))
        results[column] = round(score, 6)
    flagged = [column for column, score in results.items() if score > 0]
    return {"psi": results, "flagged_features": flagged, "drift_detected": bool(flagged)}
