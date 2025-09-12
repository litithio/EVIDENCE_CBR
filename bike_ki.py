from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


# -----------------------------
# Fuzzy helpers (piecewise lin)
# -----------------------------

@dataclass
class PiecewiseLinear:
    """
    Piecewise linear membership function defined by sorted (x, y) points.
    y is clamped to [0, 1]; x should be increasing.
    """

    points: Sequence[Tuple[float, float]]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("Need at least two points")
        xs = [p[0] for p in self.points]
        if any(xs[i] >= xs[i + 1] for i in range(len(xs) - 1)):
            raise ValueError("x values must be strictly increasing")

    def __call__(self, x: float) -> float:
        pts = self.points
        if x <= pts[0][0]:
            return max(0.0, min(1.0, pts[0][1]))
        if x >= pts[-1][0]:
            return max(0.0, min(1.0, pts[-1][1]))
        # find segment
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                if x1 == x0:
                    return max(0.0, min(1.0, y0))
                t = (x - x0) / (x1 - x0)
                y = y0 + t * (y1 - y0)
                return max(0.0, min(1.0, y))
        # fallback (shouldn't happen)
        return 0.0


def trimf(a: float, b: float, c: float) -> PiecewiseLinear:
    """Triangular MF with support [a, c] and peak b."""
    if not (a <= b <= c):
        raise ValueError("Require a <= b <= c")
    return PiecewiseLinear([(a, 0.0), (b, 1.0), (c, 0.0)])


def trapmf(a: float, b: float, c: float, d: float) -> PiecewiseLinear:
    """Trapezoidal MF with feet a,d and plateau b..c."""
    if not (a <= b <= c <= d):
        raise ValueError("Require a <= b <= c <= d")
    return PiecewiseLinear([(a, 0.0), (b, 1.0), (c, 1.0), (d, 0.0)])


# --------------------------------------------------
# Domain-specific fuzzy sets for bike classification
# --------------------------------------------------

CLASSES = ["race bike", "mtb", "trecking bike"]


class BikeFuzzy:
    """
    Fuzzy membership definitions by attribute.
    Tailored to sample data ranges (Distance km, Elevation Gain m).
    """

    # Distance in km: race (long), trekking (mid), mtb (short-mid)
    dist_mf = {
        "race bike": trapmf(35, 45, 120, 150),
        "trecking bike": trapmf(15, 25, 45, 55),
        "mtb": trapmf(5, 10, 35, 45),
    }

    # Elevation Gain in m: mtb (high), trekking (low-mid), race (mid)
    elev_mf = {
        "mtb": trapmf(300, 450, 1600, 2000),
        "trecking bike": trapmf(50, 100, 300, 450),
        "race bike": trapmf(150, 300, 800, 1100),
    }

    @staticmethod
    def memberships_from_distance(x: float) -> Dict[str, float]:
        return {cls: mf(x) for cls, mf in BikeFuzzy.dist_mf.items()}

    @staticmethod
    def memberships_from_elevation(x: float) -> Dict[str, float]:
        return {cls: mf(x) for cls, mf in BikeFuzzy.elev_mf.items()}


# --------------------------------------
# Simplified Dempster–Shafer mass fusion
# --------------------------------------

def normalize_memberships(m: Dict[str, float]) -> Dict[str, float]:
    total = sum(m.values())
    if total <= 0:
        return {k: 0.0 for k in m}
    return {k: v / total for k, v in m.items()}


def mass_from_memberships(
    memberships: Dict[str, float], alpha: float = 0.8
) -> Dict[str, float]:
    """
    Convert normalized memberships over singletons to a simple mass function
    with singletons and ignorance (omega).
    alpha is the share assigned to singletons; 1-alpha goes to omega.
    """
    nm = normalize_memberships(memberships)
    out: Dict[str, float] = {cls: alpha * nm.get(cls, 0.0) for cls in CLASSES}
    out["omega"] = max(0.0, 1.0 - sum(out.values()))
    return out


def combine_two_masses(m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
    """
    Dempster's rule for the case where focal elements are only the singletons
    (classes) and omega. Returns a new mass dict with same support.
    """
    classes = [c for c in CLASSES]
    k_conflict = 0.0
    # conflict from disagreeing singletons
    for i in classes:
        for j in classes:
            if i != j:
                k_conflict += m1.get(i, 0.0) * m2.get(j, 0.0)

    one_minus_k = 1.0 - k_conflict
    if one_minus_k <= 1e-12:
        # Full conflict; fall back to averaged masses
        avg = {c: 0.5 * (m1.get(c, 0.0) + m2.get(c, 0.0)) for c in classes}
        avg["omega"] = min(1.0, max(0.0, 0.5 * (m1.get("omega", 0.0) + m2.get("omega", 0.0))))
        return avg

    combined: Dict[str, float] = {}
    for c in classes:
        combined[c] = (
            m1.get(c, 0.0) * m2.get(c, 0.0)
            + m1.get(c, 0.0) * m2.get("omega", 0.0)
            + m1.get("omega", 0.0) * m2.get(c, 0.0)
        ) / one_minus_k

    combined["omega"] = (m1.get("omega", 0.0) * m2.get("omega", 0.0)) / one_minus_k
    return combined


def combine_masses(masses: Iterable[Dict[str, float]]) -> Dict[str, float]:
    it = iter(masses)
    try:
        acc = next(it)
    except StopIteration:
        return {c: 0.0 for c in CLASSES} | {"omega": 1.0}
    for m in it:
        acc = combine_two_masses(acc, m)
    return acc


def classify_row_dempster(
    row: Dict[str, float],
    alpha: float = 0.8,
    use_attrs: Sequence[str] = ("Distance", "Elevation Gain"),
) -> Tuple[str, Dict[str, float]]:
    """
    Build masses from selected attributes and combine to predict class.
    Returns (predicted_class, combined_mass_dict)
    """
    masses: List[Dict[str, float]] = []
    for attr in use_attrs:
        if attr == "Distance":
            memb = BikeFuzzy.memberships_from_distance(float(row[attr]))
        elif attr == "Elevation Gain":
            memb = BikeFuzzy.memberships_from_elevation(float(row[attr]))
        else:
            # Unknown attribute -> skip
            continue
        masses.append(mass_from_memberships(memb, alpha=alpha))

    combined = combine_masses(masses)
    best_cls = max(CLASSES, key=lambda c: combined.get(c, 0.0))
    return best_cls, combined


def evaluate_ds(df, label_col: str = "Bike Type") -> Dict[str, float]:
    """Compute accuracy over rows with non-empty labels."""
    df_eval = df[df[label_col].notna() & (df[label_col] != "")]
    if len(df_eval) == 0:
        return {"accuracy": 0.0, "n": 0}
    correct = 0
    for _, row in df_eval.iterrows():
        pred, _ = classify_row_dempster(row)
        correct += 1 if pred == row[label_col] else 0
    return {"accuracy": correct / len(df_eval), "n": int(len(df_eval))}


# ---------------
# Simple CBR utils
# ---------------

def _normalize(value: float, min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        return 0.0
    return (value - min_v) / (max_v - min_v)


def similarity(
    a: Dict[str, float],
    b: Dict[str, float],
    ranges: Dict[str, Tuple[float, float]],
    weights: Dict[str, float] | None = None,
    categorical_equal_bonus: float = 0.0,
) -> float:
    """
    Weighted similarity in [0,1] over numeric attributes specified by ranges.
    Uses 1 - |delta| for each normalized attribute and averages by weights.
    """
    weights = weights or {k: 1.0 for k in ranges}
    total_w = sum(weights.values())
    if total_w <= 0:
        return 0.0
    sim_sum = 0.0
    for attr, (mn, mx) in ranges.items():
        va = float(a.get(attr, 0.0))
        vb = float(b.get(attr, 0.0))
        na = _normalize(va, mn, mx)
        nb = _normalize(vb, mn, mx)
        sim = 1.0 - abs(na - nb)
        sim_sum += weights.get(attr, 0.0) * max(0.0, min(1.0, sim))
    base_sim = sim_sum / total_w
    return max(0.0, min(1.0, base_sim + categorical_equal_bonus))


def retrieve_topk(
    casebase,
    query_row,
    k: int = 3,
    feature_ranges: Dict[str, Tuple[float, float]] | None = None,
    feature_weights: Dict[str, float] | None = None,
):
    import pandas as pd

    df = casebase if isinstance(casebase, pd.DataFrame) else casebase.copy()
    feature_ranges = feature_ranges or {
        "Distance": (float(df["Distance"].min()), float(df["Distance"].max())),
        "Elevation Gain": (float(df["Elevation Gain"].min()), float(df["Elevation Gain"].max())),
    }
    sims: List[Tuple[int, float]] = []
    for i, row in df.iterrows():
        s = similarity(row, query_row, feature_ranges, feature_weights)
        sims.append((i, s))
    sims.sort(key=lambda t: t[1], reverse=True)
    top = sims[:k]
    return top, df.loc[[i for i, _ in top]]


def predict_cbr_majority(
    casebase,
    query_row,
    k: int = 3,
    feature_ranges: Dict[str, Tuple[float, float]] | None = None,
    feature_weights: Dict[str, float] | None = None,
    label_col: str = "Bike Type",
):
    top, top_df = retrieve_topk(casebase, query_row, k, feature_ranges, feature_weights)
    # similarity-weighted vote
    votes: Dict[str, float] = {}
    for (idx, s) in top:
        label = str(top_df.loc[idx, label_col])
        if label and label != "nan":
            votes[label] = votes.get(label, 0.0) + s
    if not votes:
        return None, top, votes
    pred = max(votes.keys(), key=lambda c: votes[c])
    return pred, top, votes


def evaluate_cbr(
    df,
    k: int = 3,
    label_col: str = "Bike Type",
    feature_ranges: Dict[str, Tuple[float, float]] | None = None,
    feature_weights: Dict[str, float] | None = None,
):
    df_eval = df[df[label_col].notna() & (df[label_col] != "")]
    if len(df_eval) == 0:
        return {"accuracy": 0.0, "n": 0}
    correct = 0
    for i, row in df_eval.iterrows():
        # leave-one-out style: use all except current as casebase
        casebase = df_eval.drop(index=i)
        pred, _, _ = predict_cbr_majority(
            casebase=casebase,
            query_row=row,
            k=k,
            feature_ranges=feature_ranges,
            feature_weights=feature_weights,
            label_col=label_col,
        )
        correct += 1 if pred == row[label_col] else 0
    return {"accuracy": correct / len(df_eval), "n": int(len(df_eval))}


# ----------
# Utilities
# ----------

def read_bike_csv(path: str):
    import pandas as pd

    df = pd.read_csv(path, sep=";", decimal=",")
    # Clean potential empty labels as NaN
    df["Bike Type"] = df["Bike Type"].replace({"": None})
    return df

