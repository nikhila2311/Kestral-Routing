"""M3: evaluation harness. Contract: docs/contracts.md#m3. Inputs are compared by position, not by pandas index."""
import numpy as np

from src import config

THRESHOLDS = tuple(round(0.30 + 0.05 * i, 2) for i in range(13))  # 0.30 to 0.90
PROBA_SUM_TOLERANCE = 1e-6
LOG_LOSS_EPS = 1e-15


def _pair(y_true, y_pred):
    true, pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    if true.ndim != 1 or pred.ndim != 1 or len(true) != len(pred):
        raise ValueError(f"y_true and y_pred must be 1-d and equally long, got {true.shape} and {pred.shape}")
    if len(true) == 0:
        raise ValueError("no rows to evaluate")
    return true, pred


def _label_codes(values, labels, name):
    index = {label: i for i, label in enumerate(labels)}
    if len(index) != len(labels):
        raise ValueError("labels contains duplicates")
    unknown = sum(v not in index for v in values)
    if unknown:  # count only: the values could be customer text
        raise ValueError(f"{name} has {unknown} of {len(values)} values not in `labels`")
    return np.array([index[v] for v in values])


def _as_floats(values, name):
    try:
        return np.asarray(values, dtype=float)
    except ValueError:  # numpy's message quotes the value, which could be customer text
        raise ValueError(f"{name} has non-numeric values or rows of unequal length") from None


def _check_proba(proba, n_rows, n_labels=None):
    p = _as_floats(proba, "proba")
    if p.ndim != 2 or p.shape[0] != n_rows or (n_labels is not None and p.shape[1] != n_labels):
        raise ValueError(f"proba has shape {p.shape}, expected ({n_rows}, {n_labels or 'number of labels'})")
    if not ((p >= 0) & (p <= 1)).all():  # also false for NaN
        raise ValueError("proba has values outside [0, 1] or NaN")
    if (np.abs(p.sum(axis=1) - 1) > PROBA_SUM_TOLERANCE).any():
        raise ValueError("proba rows do not sum to 1")
    return p


def _check_confidence(confidence, n_rows):
    c = _as_floats(confidence, "confidence")
    if c.shape != (n_rows,):
        raise ValueError(f"confidence has shape {c.shape}, expected ({n_rows},)")
    if not ((c >= 0) & (c <= 1)).all():  # also false for NaN
        raise ValueError("confidence has values outside [0, 1] or NaN")
    return c


def align_proba(proba, classes_, labels=config.TEAMS) -> np.ndarray:
    """Reorder proba columns from `classes_` order (sklearn: alphabetical) to `labels` order."""
    p, classes, labels = _as_floats(proba, "proba"), list(classes_), list(labels)
    if p.ndim != 2 or p.shape[1] != len(classes):
        raise ValueError(f"proba has shape {p.shape} but there are {len(classes)} classes")
    if len(set(classes)) != len(classes) or len(set(labels)) != len(labels) or set(classes) != set(labels):
        unmatched = len(set(classes) ^ set(labels))
        raise ValueError(
            f"cannot match {len(classes)} classes to {len(labels)} labels: {unmatched} names differ or some repeat"
        )
    return p[:, [classes.index(label) for label in labels]]


def coverage_curve(y_true, y_pred, proba, thresholds=THRESHOLDS, confidence=None) -> list[dict]:
    """Share of rows with confidence >= threshold, and accuracy of y_pred on them (None if no row is covered).

    Confidence defaults to proba.max(axis=1). Pass it for rule-overridden predictions; proba may then be None.
    """
    true, pred = _pair(y_true, y_pred)
    p = None if proba is None else _check_proba(proba, len(true))
    if confidence is None:
        if p is None:
            raise ValueError("pass proba or confidence")
        conf = p.max(axis=1)
    else:
        conf = _check_confidence(confidence, len(true))
    correct = (true == pred).astype(bool)
    curve = []
    for threshold in thresholds:
        covered = conf >= threshold
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": float(covered.mean()),
                "accuracy_on_covered": float(correct[covered].mean()) if covered.any() else None,
            }
        )
    return curve


def evaluate(y_true, y_pred, proba=None, labels=config.TEAMS, confidence=None) -> dict:
    """Metrics as JSON-safe types. Proba columns must follow `labels`; use align_proba on sklearn output."""
    labels = list(labels)
    true, pred = _pair(y_true, y_pred)
    true_codes = _label_codes(true, labels, "y_true")
    pred_codes = _label_codes(pred, labels, "y_pred")
    n, k = len(true), len(labels)

    matrix = np.zeros((k, k), dtype=int)  # rows = true, columns = predicted
    np.add.at(matrix, (true_codes, pred_codes), 1)
    hits, support, predicted = np.diag(matrix), matrix.sum(axis=1), matrix.sum(axis=0)

    def ratio(numerator, denominator):  # 0 where undefined
        return np.divide(numerator, denominator, out=np.zeros(k), where=denominator > 0)

    precision, recall = ratio(hits, predicted), ratio(hits, support)
    f1 = ratio(2 * hits, support + predicted)
    misroutes = int(n - hits.sum())

    result = {
        "accuracy": float(hits.sum() / n),
        "macro_f1": float(f1.mean()),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": matrix.tolist(),
        "n": n,
        "misroutes": misroutes,
        "cost_rs": float(misroutes * config.COST_PER_MISROUTE),
    }
    if proba is not None:
        p = _check_proba(proba, n, k)
        p_true = p[np.arange(n), true_codes]
        result["log_loss"] = float(-np.log(np.clip(p_true, LOG_LOSS_EPS, 1)).mean())
    if proba is not None or confidence is not None:
        result["coverage_curve"] = coverage_curve(true, pred, proba, confidence=confidence)
    return result


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=config.SEED) -> tuple[float, float, float]:
    """(low, point, high): 95% percentile interval of metric_fn(y_true, y_pred) over paired row resamples."""
    true, pred = _pair(y_true, y_pred)
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    rng = np.random.default_rng(seed)
    n = len(true)
    samples = []
    for _ in range(n_boot):
        rows = rng.integers(0, n, n)
        samples.append(metric_fn(true[rows], pred[rows]))
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(metric_fn(true, pred)), float(high)
