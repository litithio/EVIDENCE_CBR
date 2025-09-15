from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


CLASSES = ["r", "m", "t"]
LABEL_MAP = {"r": "race bike", "m": "mtb", "t": "trecking bike"}


@dataclass
class HybridResult:
    acc: float
    macro_f1: float
    tau: float
    omega_max: float
    preds: List[str]
    grid: pd.DataFrame


def _confusion_metrics(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    cm = pd.crosstab(y_true, y_pred).reindex(
        index=[LABEL_MAP[c] for c in CLASSES],
        columns=[LABEL_MAP[c] for c in CLASSES],
        fill_value=0,
    )
    diag = np.diag(cm.values)
    support = cm.sum(axis=1).values
    pred_sum = cm.sum(axis=0).values
    recall = np.divide(diag, support, out=np.zeros_like(diag, dtype=float), where=support > 0)
    precision = np.divide(diag, pred_sum, out=np.zeros_like(diag, dtype=float), where=pred_sum > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(diag, dtype=float), where=(precision + recall) > 0)
    macro_f1 = float(np.nanmean(f1))
    acc = float((y_true == y_pred).mean())
    return acc, macro_f1


def evaluate_hybrid(
    masses: Sequence,  # list of pyds MassFunction or None per row
    data: pd.DataFrame,
    fallback_col: str = "Pred_DS_soft_tuned",
    tau_singleton: float = 0.5,
    omega_key: str = "omega",
    omega_max: float = 0.5,
) -> Tuple[float, float, List[str]]:
    if fallback_col not in data.columns:
        # try alternatives
        for alt in ("Pred_DS_soft_best", "Pred_DS_soft"):
            if alt in data.columns:
                fallback_col = alt
                break
        else:
            raise ValueError("No soft-voting fallback column found in data")

    preds: List[str] = []
    for i, row in data.iterrows():
        m = masses[i] if i < len(masses) else None
        if m is None:
            preds.append(row.get(fallback_col, None))
            continue
        singles = {c: (m[c] if c in m else 0.0) for c in CLASSES}
        best_c, best_v = max(singles.items(), key=lambda kv: kv[1])
        omega_v = float(m[omega_key]) if omega_key in m else 0.0
        if best_v >= tau_singleton and omega_v <= omega_max:
            preds.append(LABEL_MAP[best_c])
        else:
            preds.append(row.get(fallback_col, None))

    df_eval = data[data["Bike Type"].notna() & (data["Bike Type"] != "")]
    y_true = df_eval["Bike Type"]
    y_pred = pd.Series(preds, index=data.index).loc[df_eval.index]
    acc, macro_f1 = _confusion_metrics(y_true, y_pred)
    return acc, macro_f1, preds


def optimize_hybrid(
    masses: Sequence,
    data: pd.DataFrame,
    fallback_col: str = "Pred_DS_soft_tuned",
    tau_list: Iterable[float] = (0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52),
    omega_list: Iterable[float] = (0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52),
    optimize: str = "accuracy",  # or "macro_f1"
) -> HybridResult:
    records = []
    best = (-1.0, -1.0, None, None)
    best_preds: List[str] | None = None
    for t in tau_list:
        for o in omega_list:
            acc, mf1, preds = evaluate_hybrid(masses, data, fallback_col, tau_singleton=t, omega_max=o)
            records.append({"tau": t, "omega": o, "acc": acc, "macro_f1": mf1})
            if optimize == "macro_f1":
                better = (mf1 > best[1]) or (mf1 == best[1] and acc > best[0])
            else:
                better = (acc > best[0]) or (acc == best[0] and mf1 > best[1])
            if better:
                best = (acc, mf1, t, o)
                best_preds = preds

    df_res = pd.DataFrame.from_records(records)
    acc, mf1, t, o = best
    return HybridResult(acc=acc, macro_f1=mf1, tau=t, omega_max=o, preds=best_preds or [], grid=df_res)


def grid_pivot(df_res: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    metric = "acc" if metric == "accuracy" else metric
    if metric not in df_res.columns:
        raise ValueError(f"metric {metric} not in df")
    return df_res.pivot(index="tau", columns="omega", values=metric)

