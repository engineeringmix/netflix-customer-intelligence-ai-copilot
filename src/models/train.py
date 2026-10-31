"""Train churn classifier, K-Means segmentation and CLV regressor. Writes models/, reports/ and data/processed/customer_scores.csv

Run: python -m src.models.train
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_curve, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.utils.config import PROCESSED, MODELS, REPORTS, FIGURES

warnings.filterwarnings("ignore")
SEED = 42
CAT = ["country", "acquisition_channel", "customer_segment", "current_plan", "primary_device"]
# CLV is built from payments -> excluded from churn features. payment_attempts excluded too (it mirrors tenure/billing history).
CHURN_NUM = ["age", "customer_tenure_days", "plan_price", "auto_renew", "plan_changes", "total_sessions", "total_watch_minutes",
             "avg_watch_time_per_session", "avg_completion_pct", "days_since_last_activity", "monthly_watch_time_trend",
             "watch_min_last_3m", "watch_min_prev_3m", "watch_trend_log_ratio", "login_frequency", "distinct_titles",
             "distinct_devices", "distinct_genres", "failed_payment_count", "failed_payment_rate", "failed_payments_90d",
             "payment_methods_used", "support_ticket_count", "avg_customer_satisfaction", "escalated_tickets",
             "pending_tickets", "avg_resolution_hours", "billing_ticket_count", "avg_feedback_rating",
             "negative_feedback_count", "engagement_score"]
# CLV model: no payment-derived features at all
CLV_NUM = [c for c in CHURN_NUM if c not in ("failed_payment_count", "failed_payment_rate", "failed_payments_90d", "payment_methods_used")]
COST_PER_OFFER, VALUE_PER_SAVED = 5.0, 24.0   # illustrative: $5 offer cost; 30% accept x ~$80 retained revenue


def preprocessor(num, scale):
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("sc", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT)])


def feature_names(pipe):
    pre = pipe.named_steps["pre"]
    num_pipe = pre.named_transformers_["num"]
    names = list(num_pipe.named_steps["imp"].get_feature_names_out(CHURN_NUM if "age" in pre.transformers_[0][2] and len(pre.transformers_[0][2]) == len(CHURN_NUM) else pre.transformers_[0][2]))
    names += list(pre.named_transformers_["cat"].get_feature_names_out(CAT))
    return names


def metrics_at(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh), recall=recall_score(y, yh),
                f1=f1_score(y, yh), roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p))


def train_churn(df, report):
    X, y = df[CAT + CHURN_NUM], df.churned
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    cands = {
        "Logistic Regression": Pipeline([("pre", preprocessor(CHURN_NUM, True)), ("m", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3))]),
        "Random Forest": Pipeline([("pre", preprocessor(CHURN_NUM, False)), ("m", RandomForestClassifier(n_estimators=400, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=2, random_state=SEED))]),
        "Gradient Boosting": Pipeline([("pre", preprocessor(CHURN_NUM, False)), ("m", GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED))]),
        "XGBoost": Pipeline([("pre", preprocessor(CHURN_NUM, False)), ("m", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7,
                                                                                         min_child_weight=10, reg_lambda=5, scale_pos_weight=spw, eval_metric="auc", n_jobs=2, random_state=SEED))]),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, fitted, probs = [], {}, {}
    for name, pipe in cands.items():
        cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1)
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        fitted[name], probs[name] = pipe, p
        rows.append({"model": name, "cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std(), **metrics_at(yte, p)})
        print(name, round(cv_auc.mean(), 4), round(roc_auc_score(yte, p), 4), flush=True)
    res = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
    res.to_csv(REPORTS / "churn_model_comparison.csv", index=False)
    top = res.iloc[0]
    xgb_row = res[res.model == "XGBoost"].iloc[0]
    # All four models are statistically tied (CV AUC spread < 1 std). Tie-break rule: prefer XGBoost, whose exact TreeSHAP
    # contributions power the customer-level explainer, so the score and its explanation come from the SAME model.
    best = "XGBoost" if xgb_row.cv_auc_mean >= top.cv_auc_mean - top.cv_auc_std else top.model
    report["churn_selection_rule"] = f"best CV AUC = {top.model} ({top.cv_auc_mean:.4f} +/- {top.cv_auc_std:.4f}); XGBoost within 1 std -> chosen for explainability"
    report["churn_comparison"] = res.round(4).to_dict("records")
    report["churn_final_model"] = best
    pbest = probs[best]

    # Threshold selection by net business benefit
    prec, rec, thr = precision_recall_curve(yte, pbest)
    ths = np.linspace(0.05, 0.95, 91)
    benefit = []
    for t in ths:
        sel = pbest >= t
        tp = ((yte.values == 1) & sel).sum()
        benefit.append(tp * VALUE_PER_SAVED - sel.sum() * COST_PER_OFFER)
    bt = float(ths[int(np.argmax(benefit))])
    report["churn_thresholds"] = {"default_0.5": metrics_at(yte, pbest, 0.5), "profit_optimal": {"threshold": bt, **metrics_at(yte, pbest, bt)},
                                  "assumptions": {"cost_per_offer": COST_PER_OFFER, "value_per_saved_churner": VALUE_PER_SAVED}}
    cm = confusion_matrix(yte, (pbest >= 0.5).astype(int)).tolist()
    report["churn_confusion_matrix_0.5"] = cm

    # plots
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for n, p in probs.items():
        fpr, tpr, _ = roc_curve(yte, p)
        ax[0].plot(fpr, tpr, label=f"{n} ({roc_auc_score(yte, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set_title("ROC curves (test set)"); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[1].plot(rec, prec); ax[1].axhline(yte.mean(), ls="--", c="grey"); ax[1].set_title(f"Precision-Recall: {best}")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[2].plot(ths, benefit); ax[2].axvline(bt, c="r", ls="--"); ax[2].set_title(f"Net benefit vs threshold (best={bt:.2f})")
    ax[2].set_xlabel("Probability threshold"); ax[2].set_ylabel("$ net benefit (illustrative)")
    plt.tight_layout(); plt.savefig(FIGURES / "model_churn_curves.png", dpi=130); plt.close()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Pred stay", "Pred churn"]); ax.set_yticks([0, 1], ["Stayed", "Churned"]); ax.set_title(f"Confusion matrix ({best}, 0.5)")
    plt.tight_layout(); plt.savefig(FIGURES / "model_churn_confusion.png", dpi=130); plt.close()

    # Final scores for ALL customers = out-of-fold predictions (no customer is scored by a model that saw its own label)
    final_cfg = cands[best]
    oof = cross_val_predict(final_cfg, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    # Serving models are fit on ALL data
    serve = {n: p for n, p in cands.items() if n in (best, "XGBoost")}
    for n, p in serve.items():
        p.fit(X, y)
    joblib.dump({"model": serve[best], "name": best, "threshold": bt, "cat": CAT, "num": CHURN_NUM}, MODELS / "churn_model.joblib")
    joblib.dump({"model": serve["XGBoost"], "cat": CAT, "num": CHURN_NUM}, MODELS / "churn_xgb_explainer.joblib")
    return oof, bt, best


def train_segments(df, report):
    cols = ["engagement_score", "avg_watch_time_per_session", "login_frequency", "failed_payment_rate", "customer_tenure_days",
            "avg_customer_satisfaction", "plan_price", "support_ticket_count"]
    X = df[cols].copy()
    X["avg_customer_satisfaction"] = X.avg_customer_satisfaction.fillna(X.avg_customer_satisfaction.median())
    X["avg_watch_time_per_session"] = X.avg_watch_time_per_session.fillna(0)
    for c in ["login_frequency", "failed_payment_rate", "support_ticket_count"]:
        X[c] = np.log1p(X[c])
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    ks = list(range(2, 11))
    inertia, sil = [], []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Z, km.labels_, sample_size=5000, random_state=SEED))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("K"); ax[0].set_ylabel("Inertia")
    ax[1].plot(ks, sil, "o-"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("K")
    plt.tight_layout(); plt.savefig(FIGURES / "model_segmentation_k.png", dpi=130); plt.close()
    report["segmentation_k_search"] = {"k": ks, "inertia": [round(i, 1) for i in inertia], "silhouette": [round(s, 4) for s in sil]}
    return X, Z, sc, cols, ks, sil


def name_segments(df, labels):
    """Assign business names from centroid characteristics (deterministic, rule-based, one name per cluster)."""
    d = df.assign(seg=labels)
    g = d.groupby("seg").agg(price=("plan_price", "mean"), clv=("customer_lifetime_value", "mean"), eng=("engagement_score", "mean"),
                             tix=("support_ticket_count", "mean"), ten=("customer_tenure_days", "mean"))
    rules = [("Premium Loyalists", "Long-tenured, highest plan price and lifetime value.", lambda x: x.price + x.clv / 10),
             ("Disengaged Drifters", "Lowest engagement and the highest churn: watching little, drifting away.", lambda x: -x.eng),
             ("Support-Heavy Regulars", "Contact support far more than anyone else; satisfied after resolution.", lambda x: x.tix),
             ("New & Highly Engaged", "Newest customers who watch the most; the future of the base.", lambda x: x.eng - x.ten / 30)]
    names, desc, left = {}, {}, list(g.index)
    for nm, ds, score in rules:
        pick = score(g.loc[left]).idxmax()
        names[pick], desc[pick] = nm, ds
        left.remove(pick)
    for k in left:
        names[k], desc[k] = "Budget Basic Subscribers", "Cheapest plans, more payment failures, low lifetime value; price-sensitive."
    return names, desc


def train_clv(df, report):
    X, y = df[CAT + CLV_NUM], df.customer_lifetime_value
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cands = {
        "Linear Regression": Pipeline([("pre", preprocessor(CLV_NUM, True)), ("m", LinearRegression())]),
        "Random Forest": Pipeline([("pre", preprocessor(CLV_NUM, False)), ("m", RandomForestRegressor(300, min_samples_leaf=5, n_jobs=2, random_state=SEED))]),
        "XGBoost": Pipeline([("pre", preprocessor(CLV_NUM, False)), ("m", XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=2, random_state=SEED))]),
    }
    rows = []
    for n, pipe in cands.items():
        cv_r2 = cross_val_score(pipe, Xtr, ytr, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2").mean()
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        rows.append({"model": n, "cv_r2": cv_r2, "MAE": mean_absolute_error(yte, p), "RMSE": mean_squared_error(yte, p) ** 0.5, "R2": r2_score(yte, p)})
    res = pd.DataFrame(rows).sort_values("cv_r2", ascending=False)
    res.to_csv(REPORTS / "clv_model_comparison.csv", index=False)
    best = res.iloc[0].model
    report["clv_comparison"] = res.round(4).to_dict("records"); report["clv_final_model"] = best
    report["clv_target_mean"] = float(y.mean()); report["clv_target_std"] = float(y.std())
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(yte, cands[best].predict(Xte), s=6, alpha=.4); lim = [0, y.max()]
    ax.plot(lim, lim, "r--"); ax.set_xlabel("Actual CLV ($)"); ax.set_ylabel("Predicted CLV ($)"); ax.set_title(f"CLV: {best} (test)")
    plt.tight_layout(); plt.savefig(FIGURES / "model_clv_actual_vs_pred.png", dpi=130); plt.close()
    oof = cross_val_predict(cands[best], X, y, cv=KFold(5, shuffle=True, random_state=SEED))
    cands[best].fit(X, y)
    joblib.dump({"model": cands[best], "name": best, "cat": CAT, "num": CLV_NUM}, MODELS / "clv_model.joblib")
    return oof


def main():
    FIGURES.mkdir(parents=True, exist_ok=True); MODELS.mkdir(exist_ok=True)
    df = pd.read_csv(PROCESSED / "customer_features.csv")
    report = {}
    oof, bt, best = train_churn(df, report)
    # scale_pos_weight / class_weight distort raw probabilities -> isotonic calibration on out-of-fold scores
    calib = IsotonicRegression(out_of_bounds="clip").fit(oof, df.churned)
    df["churn_risk_score"] = calib.predict(oof).round(4)
    bundle = joblib.load(MODELS / "churn_model.joblib")
    bundle.update({"calibrator": calib, "threshold_calibrated": float(calib.predict([bt])[0])})
    joblib.dump(bundle, MODELS / "churn_model.joblib")
    ex = joblib.load(MODELS / "churn_xgb_explainer.joblib"); ex["calibrator"] = calib; joblib.dump(ex, MODELS / "churn_xgb_explainer.joblib")
    # Tier cut-offs from economics: contacting a customer pays off when p >= cost/value = 5/24 = 0.21; High >= 0.40
    BREAK_EVEN = COST_PER_OFFER / VALUE_PER_SAVED
    report["risk_tiers"] = {"low_below": round(BREAK_EVEN, 3), "high_from": 0.40, "break_even_probability": round(BREAK_EVEN, 3)}
    df["risk_tier"] = pd.cut(df.churn_risk_score, [-1, BREAK_EVEN, 0.40, 2], labels=["Low", "Medium", "High"], right=False).astype(str)
    report["risk_tier_counts"] = df.risk_tier.value_counts().to_dict()
    report["risk_tier_churn_rates"] = df.groupby("risk_tier").churned.mean().round(3).to_dict()
    d = df.sort_values("churn_risk_score", ascending=False); n = len(d) // 10
    report["churn_decile_lift"] = {"base_rate": float(df.churned.mean()), "top_decile_churn_rate": float(d.head(n).churned.mean()),
                                   "bottom_decile_churn_rate": float(d.tail(n).churned.mean()),
                                   "top_decile_lift": float(d.head(n).churned.mean() / df.churned.mean())}
    X, Z, sc, cols, ks, sil = train_segments(df, report)
    K = 5   # silhouette is flat (0.117-0.127) for every K: the data has no natural clusters, so K is chosen for usability
    km = KMeans(K, n_init=20, random_state=SEED).fit(Z)
    names, desc = name_segments(df, km.labels_)
    df["segment_id"] = km.labels_
    df["segment_name"] = df.segment_id.map(names)
    report["segmentation"] = {"chosen_k": K, "silhouette_at_k": float(sil[ks.index(K)]), "names": {int(k): v for k, v in names.items()},
                              "descriptions": {names[k]: v for k, v in desc.items()}}
    joblib.dump({"scaler": sc, "kmeans": km, "cols": cols, "names": names, "descriptions": desc}, MODELS / "segmentation.joblib")
    df["predicted_clv"] = train_clv(df, report).round(2)
    out_cols = ["customer_id", "churn_risk_score", "risk_tier", "segment_id", "segment_name", "predicted_clv"]
    df[out_cols].to_csv(PROCESSED / "customer_scores.csv", index=False)
    json.dump(report, open(REPORTS / "model_report.json", "w"), indent=2, default=float)
    print(json.dumps({k: report[k] for k in ["churn_final_model", "churn_decile_lift", "risk_tier_counts", "risk_tier_churn_rates"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
