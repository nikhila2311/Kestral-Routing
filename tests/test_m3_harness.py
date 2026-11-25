"""M3: split and evaluation harness. Labels and probabilities here are invented, not customer data."""
import json
import math
import traceback

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.evaluate import align_proba, bootstrap_ci, coverage_curve, evaluate
from src.split import time_split

# Same order as config.TEAMS; checked in test_label_order.
TEAMS = [
    "Installs & Demo",
    "Repairs",
    "Filters & Consumables",
    "Billing",
    "Returns & Replacement",
    "Warranty Claims",
    "Product Advice",
]
I, R, F, B, T, W, P = TEAMS

# Hand example: 7 of 10 correct; the misses are the 2nd (I->R), 7th (B->I) and 9th (W->P) rows.
Y_TRUE = [I, I, R, R, F, B, B, T, W, P]
Y_PRED = [I, R, R, R, F, B, I, T, P, P]
# Rs 305 x 3549 / 2471 + Rs 260
COST_EACH = 1_082_445 / 2471 + 260


def proba_for(y, top, labels=TEAMS):
    """Rows with `top` on the given label and the rest shared equally."""
    tops = np.broadcast_to(top, len(y))
    p = np.array([[(1 - t) / (len(labels) - 1)] * len(labels) for t in tops])
    for row, label in enumerate(y):
        p[row, labels.index(label)] = tops[row]
    return p


@pytest.fixture(scope="module")
def train(config):
    return pd.read_parquet(config.TRAIN_CLEAN_PARQUET)


def accuracy(y_true, y_pred):
    return evaluate(y_true, y_pred)["accuracy"]


def macro_f1(y_true, y_pred):
    return evaluate(y_true, y_pred)["macro_f1"]


def test_label_order(config):
    assert list(config.TEAMS) == TEAMS


# --- 1. split ---------------------------------------------------------------
def test_split_disjoint_ordered_complete(train):
    train_df, val_df = time_split(train)
    assert not set(train_df["request_id"]) & set(val_df["request_id"])
    assert train_df["created_at_ist"].max() < val_df["created_at_ist"].min()
    assert set(val_df["created_month"]) == {"2026-04", "2026-05", "2026-06"}
    assert train_df["created_month"].min() == "2025-04" and train_df["created_month"].max() == "2026-03"
    assert len(train_df) + len(val_df) == len(train)
    assert len(train_df) == 8687
    assert len(val_df) == 2135
    ids = list(train_df["request_id"]) + list(val_df["request_id"])
    assert (pd.Series(ids) == train["request_id"].reset_index(drop=True)).all()  # train is sorted by time
    assert list(train_df.columns) == list(val_df.columns) == list(train.columns)


def test_split_boundary_timestamps():
    stamps = ["2025-04-01 00:00:00", "2026-03-31 23:59:59", "2026-04-01 00:00:00", "2026-06-30 23:59:59"]
    df = pd.DataFrame({"created_at_ist": pd.to_datetime(stamps), "row": range(4)})
    train_df, val_df = time_split(df)
    assert list(train_df["row"]) == [0, 1]
    assert list(val_df["row"]) == [2, 3]


def test_split_keeps_input_order_and_does_not_change_input():
    stamps = ["2026-05-02", "2025-06-01", "2026-04-03", "2025-05-01"]
    df = pd.DataFrame({"created_at_ist": pd.to_datetime(stamps), "row": range(4)})
    before = df.copy()
    train_df, val_df = time_split(df)
    assert list(train_df["row"]) == [1, 3]
    assert list(val_df["row"]) == [0, 2]
    train_df["row"] = -1
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("stamp", ["2025-03-31 23:59:59", "2026-07-01 00:00:00", None])
def test_split_rejects_rows_outside_both_windows(stamp):
    df = pd.DataFrame({"created_at_ist": pd.to_datetime(["2025-05-01 00:00:00", "2026-05-01 00:00:00", stamp])})
    with pytest.raises(ValueError, match="1 rows are outside"):
        time_split(df)


# --- 2, 3. metrics ----------------------------------------------------------
def test_hand_example_exact():
    result = evaluate(Y_TRUE, Y_PRED)
    assert set(result) == {"accuracy", "macro_f1", "per_class", "confusion_matrix", "n", "misroutes", "cost_rs"}
    assert result["n"] == 10
    assert result["accuracy"] == pytest.approx(0.7, abs=1e-12)
    assert result["misroutes"] == 3
    assert result["cost_rs"] == pytest.approx(3 * COST_EACH, abs=1e-9)  # about Rs 2,094.18

    # team: (true positives / predicted, true positives / actual, 2 tp / (predicted + actual), actual)
    expected = {
        I: (1 / 2, 1 / 2, 2 / 4, 2),
        R: (2 / 3, 2 / 2, 4 / 5, 2),
        F: (1, 1, 1, 1),
        B: (1 / 1, 1 / 2, 2 / 3, 2),
        T: (1, 1, 1, 1),
        W: (0, 0 / 1, 0, 1),  # never predicted: precision is defined as 0
        P: (1 / 2, 1 / 1, 2 / 3, 1),
    }
    assert list(result["per_class"]) == TEAMS
    for team, (precision, recall, f1, support) in expected.items():
        got = result["per_class"][team]
        assert set(got) == {"precision", "recall", "f1", "support"}
        assert got["precision"] == pytest.approx(precision, abs=1e-12)
        assert got["recall"] == pytest.approx(recall, abs=1e-12)
        assert got["f1"] == pytest.approx(f1, abs=1e-12)
        assert got["support"] == support
    # F1 sum = 1/2 + 4/5 + 1 + 2/3 + 1 + 0 + 2/3 = 139/30; divided by 7 = 139/210
    assert result["macro_f1"] == pytest.approx(139 / 210, abs=1e-12)

    assert result["confusion_matrix"] == [  # rows = true, columns = predicted
        [1, 1, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
    ]


def test_perfect_predictions():
    result = evaluate(TEAMS, TEAMS)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["misroutes"] == 0
    assert result["cost_rs"] == 0


def test_absent_team_counts_as_zero_in_macro_f1():
    result = evaluate([I, R], [I, R])
    assert result["macro_f1"] == pytest.approx(2 / 7, abs=1e-12)
    assert result["per_class"][B] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(42)
    y_true, y_pred = rng.choice(TEAMS, 300), rng.choice(TEAMS, 300)
    expected = f1_score(y_true, y_pred, labels=TEAMS, average="macro", zero_division=0)
    assert macro_f1(y_true, y_pred) == pytest.approx(expected, abs=1e-12)


def test_series_are_compared_by_position():
    y_true = pd.Series(Y_TRUE, index=range(100, 110))
    y_pred = pd.Series(Y_PRED, index=range(109, 99, -1))
    assert evaluate(y_true, y_pred) == evaluate(Y_TRUE, Y_PRED)


def test_labels_argument_sets_order():
    result = evaluate(["a", "a", "b"], ["a", "b", "b"], labels=["b", "a"])
    assert list(result["per_class"]) == ["b", "a"]
    assert result["confusion_matrix"] == [[1, 0], [1, 1]]
    with pytest.raises(ValueError, match="duplicates"):
        evaluate(["a"], ["a"], labels=["a", "a"])


def test_result_is_json_safe():
    result = evaluate(Y_TRUE, Y_PRED, proba_for(Y_PRED, 0.8))
    assert json.loads(json.dumps(result, allow_nan=False)) == result


# --- 4. bad labels and shapes -----------------------------------------------
@pytest.mark.parametrize("bad", ["Installations", "Consumables", "billing", None, float("nan")])
def test_unknown_pred_label_raises(bad):
    with pytest.raises(ValueError, match="y_pred"):
        evaluate([I, R], [I, bad])


def test_unknown_true_label_raises():
    with pytest.raises(ValueError, match="y_true"):
        evaluate([I, "Installations"], [I, R])


LONG_TEXT = "invented zebra sentence about a lighthouse that must never be printed in an error"


@pytest.mark.parametrize(
    "call",
    [
        lambda: evaluate([I, R], [I, LONG_TEXT]),
        lambda: evaluate([I, LONG_TEXT], [I, R]),
        lambda: evaluate([LONG_TEXT], [LONG_TEXT], labels=[LONG_TEXT, LONG_TEXT]),
        lambda: evaluate([I], [I], [[LONG_TEXT] * 7]),
        lambda: evaluate([I], [I], confidence=[LONG_TEXT]),
        lambda: coverage_curve([I], [I], [[LONG_TEXT] * 7], [0.5]),
        lambda: align_proba([[LONG_TEXT] * 7], TEAMS),
        lambda: align_proba(proba_for([I], 0.4), TEAMS[:-1] + [LONG_TEXT]),
        lambda: align_proba(proba_for([I], 0.4), TEAMS, labels=TEAMS[:-1] + [LONG_TEXT]),
        lambda: time_split(pd.DataFrame({"created_at_ist": [pd.Timestamp("2020-01-01")], "text": [LONG_TEXT]})),
    ],
)
def test_error_messages_do_not_print_values(call):
    with pytest.raises(ValueError) as caught:
        call()
    # The printed traceback includes any chained error, e.g. numpy's, so check all of it. No word may appear either.
    printed = "".join(traceback.format_exception(caught.value))
    assert LONG_TEXT not in printed
    assert not {"zebra", "lighthouse"} & set(printed.lower().split())
    assert LONG_TEXT[:8] not in str(caught.value)


@pytest.mark.parametrize("y_true, y_pred", [([], []),([I, R], [I]), ([[I]], [[I]])])
def test_empty_or_mismatched_inputs_raise(y_true, y_pred):
    with pytest.raises(ValueError):
        evaluate(y_true, y_pred)
    with pytest.raises(ValueError):
        bootstrap_ci(y_true, y_pred, lambda a, b: 0.0)


# --- 5. proba, confidence, coverage -----------------------------------------
def test_log_loss_by_hand():
    y = [I, B]
    p = proba_for(y, [0.5, 0.25])
    assert evaluate(y, y, p)["log_loss"] == pytest.approx((math.log(2) + math.log(4)) / 2, abs=1e-12)


def test_log_loss_uses_true_label_and_stays_finite():
    p = proba_for([R], 1.0)  # zero probability on the true label
    assert evaluate([I], [R], p)["log_loss"] == pytest.approx(-math.log(1e-15))


@pytest.mark.parametrize(
    "call",
    [
        lambda p: evaluate(Y_TRUE, Y_PRED, p),
        lambda p: coverage_curve(Y_TRUE, Y_PRED, p, [0.5]),
        lambda p: coverage_curve(Y_TRUE, Y_PRED, p, [0.5], confidence=[0.5] * 10),
    ],
)
def test_proba_must_sum_to_one(call):
    good = proba_for(Y_PRED, 0.8)
    call(good)
    call(good + 1e-9)  # float noise is fine
    for total in (0.9, 1.1):
        bad = good.copy()
        bad[4] *= total
        with pytest.raises(ValueError, match="sum to 1"):
            call(bad)


def test_proba_bad_values_or_shape_raise():
    good = proba_for(Y_PRED, 0.8)
    negative = good.copy()
    negative[0, :2] = [-0.1, good[0, 0] + good[0, 1] + 0.1]  # still sums to 1, all values <= 1
    with_nan = good.copy()
    with_nan[0, 0] = np.nan
    padded = np.hstack([good, np.zeros((10, 1))])
    for bad in (negative, with_nan, good[:9], padded, good[0]):
        with pytest.raises(ValueError, match="proba has"):
            evaluate(Y_TRUE, Y_PRED, bad)


def test_coverage_zero_is_one():
    curve = coverage_curve(Y_TRUE, Y_PRED, proba_for(Y_PRED, 0.8), [0.0])
    assert curve == [{"threshold": 0.0, "coverage": 1.0, "accuracy_on_covered": pytest.approx(0.7)}]


def test_coverage_non_increasing():
    rng = np.random.default_rng(42)
    p = rng.dirichlet([0.5] * 7, 500)
    y_true = rng.choice(TEAMS, 500)
    y_pred = np.array(TEAMS)[p.argmax(axis=1)]
    curve = evaluate(y_true, y_pred, p)["coverage_curve"]
    assert [c["threshold"] for c in curve] == [
        0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
    ]
    coverage = [c["coverage"] for c in curve]
    assert all(a >= b for a, b in zip(coverage, coverage[1:]))
    assert coverage[0] > coverage[-1]


def test_coverage_by_hand():
    # Default confidence is the row maximum: 0.9, 0.6, 0.5, 0.2. The second row is wrong.
    y_true, y_pred = [I, R, F, B], [I, P, F, B]
    p = proba_for(y_pred, [0.9, 0.6, 0.5, 0.2])
    curve = coverage_curve(y_true, y_pred, p, [0.2, 0.5, 0.6, 0.7, 0.95])
    assert [c["coverage"] for c in curve] == [1.0, 0.75, 0.5, 0.25, 0.0]  # >= includes the threshold itself
    assert [c["accuracy_on_covered"] for c in curve] == [0.75, pytest.approx(2 / 3), 0.5, 1.0, None]


def test_evaluate_accepts_rule_override_with_confidence():
    # The model says R for both rows; a rule changes the first to the correct team with confidence 0.95.
    y_true, y_pred = [B, R], [B, R]
    p = proba_for([R, R], [0.55, 0.35])
    assert y_pred[0] != TEAMS[p[0].argmax()]
    result = evaluate(y_true, y_pred, p, confidence=[0.95, 0.35])
    assert result["accuracy"] == 1.0
    by_threshold = {c["threshold"]: c for c in result["coverage_curve"]}
    assert by_threshold[0.9]["coverage"] == 0.5  # 0 if row maxima had been used
    assert by_threshold[0.9]["accuracy_on_covered"] == 1.0
    assert by_threshold[0.3]["coverage"] == 1.0
    # Log loss still scores proba, not the rule.
    expected = -(math.log(0.45 / 6) + math.log(0.35)) / 2
    assert result["log_loss"] == pytest.approx(expected, abs=1e-12)
    # y_pred may differ from argmax without confidence too; row maxima are then used.
    assert evaluate(y_true, y_pred, p)["coverage_curve"][-1]["coverage"] == 0.0


def test_confidence_without_proba_gives_curve_but_no_log_loss():
    result = evaluate([I, R], [I, R], confidence=[0.9, 0.4])
    assert "log_loss" not in result
    assert result["coverage_curve"][-1]["coverage"] == 0.5
    with pytest.raises(ValueError, match="pass proba or confidence"):
        coverage_curve([I], [I], None, [0.5])


@pytest.mark.parametrize("bad", [[0.5], [0.5, 1.1], [0.5, -0.1], [0.5, float("nan")], [[0.5], [0.5]]])
def test_bad_confidence_raises(bad):
    with pytest.raises(ValueError, match="confidence"):
        evaluate([I, R], [I, R], confidence=bad)


def test_align_proba_reorders_alphabetical_classes():
    classes = sorted(TEAMS)
    assert classes != TEAMS
    values = {I: 0.40, R: 0.25, F: 0.15, B: 0.09, T: 0.06, W: 0.04, P: 0.01}
    source = np.array([[values[c] for c in classes]])
    before = source.copy()
    aligned = align_proba(source, np.array(classes))
    assert aligned.tolist() == [[0.40, 0.25, 0.15, 0.09, 0.06, 0.04, 0.01]]
    assert (source == before).all()
    # End to end: log loss is right only after aligning.
    assert evaluate([I], [I], aligned)["log_loss"] == pytest.approx(-math.log(0.40), abs=1e-12)
    assert evaluate([I], [I], source)["log_loss"] != pytest.approx(-math.log(0.40))


@pytest.mark.parametrize(
    "classes",
    [
        TEAMS[:-1],  # fewer classes than columns
        TEAMS[:-1] + ["Installations"],  # unknown class
        TEAMS[:-1] + [I],  # duplicate class, one team missing
        TEAMS + ["Extra"],  # more classes than columns
    ],
)
def test_align_proba_raises_when_unmatched(classes):
    with pytest.raises(ValueError, match="classes"):
        align_proba(proba_for([I], 0.4), classes)


@pytest.mark.parametrize("classes", [TEAMS[:-1], TEAMS + ["Extra"], TEAMS + [I]])
def test_align_proba_raises_when_class_set_differs(classes):
    # Column count agrees with classes, so only the class names can be wrong.
    with pytest.raises(ValueError, match="cannot match"):
        align_proba(np.full((1, len(classes)), 1 / len(classes)), classes)


# --- 6. bootstrap -----------------------------------------------------------
@pytest.fixture(scope="module")
def noisy():
    rng = np.random.default_rng(7)
    y_true = rng.choice(TEAMS, 400)
    y_pred = np.where(rng.random(400) < 0.8, y_true, rng.choice(TEAMS, 400))
    return y_true, y_pred


@pytest.mark.parametrize("metric", [accuracy, macro_f1])
def test_bootstrap_order_and_determinism(noisy, config, metric):
    y_true, y_pred = noisy
    low, point, high = bootstrap_ci(y_true, y_pred, metric, n_boot=300)
    assert low <= point <= high
    assert low < high
    assert point == metric(y_true, y_pred)
    assert bootstrap_ci(y_true, y_pred, metric, n_boot=300) == (low, point, high)
    assert bootstrap_ci(y_true, y_pred, metric, n_boot=300, seed=config.SEED) == (low, point, high)
    other = bootstrap_ci(y_true, y_pred, metric, n_boot=300, seed=config.SEED + 1)
    assert other[1] == point and other != (low, point, high)


def test_bootstrap_width_is_a_95_percent_interval(noisy):
    y_true, y_pred = noisy
    low, point, high = bootstrap_ci(y_true, y_pred, lambda a, b: float((a == b).mean()))
    expected_width = 2 * 1.96 * math.sqrt(point * (1 - point) / len(y_true))
    assert high - low == pytest.approx(expected_width, rel=0.1)


def test_bootstrap_resamples_pairs_and_leaves_global_random_state_alone():
    np.random.seed(1)
    state = np.random.get_state()[1].copy()
    same = bootstrap_ci(TEAMS, TEAMS, lambda a, b: float((a == b).mean()), n_boot=50)
    assert same == (1.0, 1.0, 1.0)  # pairs stay together
    assert (np.random.get_state()[1] == state).all()


def test_bootstrap_default_is_1000_resamples():
    calls = []
    bootstrap_ci([I, R], [I, R], lambda a, b: calls.append(1) or 1.0)
    assert len(calls) == 1000 + 1  # plus the point estimate


def test_bootstrap_rejects_zero_resamples():
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci([I], [I], accuracy, n_boot=0)


# --- 7. cost constant -------------------------------------------------------
def test_cost_per_misroute_matches_parquet(config, train):
    wrong = train["team_label_canon"] != train["final_team_canon"]
    average = train.loc[wrong, "transfers"].sum() / wrong.sum()
    assert config.COST_PER_MISROUTE == pytest.approx(305 * average + 260, abs=1e-9)
    assert config.COST_PER_MISROUTE == pytest.approx(698, abs=0.5)

    inputs = json.loads(config.AUDIT_METRICS_JSON.read_text(encoding="utf-8"))["cost_inputs"]
    from_json = (inputs["transfers"] - inputs["transfers_on_correct_rows"]) / inputs["misroutes"]
    assert config.COST_PER_MISROUTE == pytest.approx(305 * from_json + 260, abs=1e-9)

    assert config.COST_PER_MISROUTE_SIMPLE == 565
