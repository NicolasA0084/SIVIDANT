#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

EXPECTED_MANIFEST_PACKAGE_FP = "fe54fab7182f2aa5d35c987558f3deddd35e684ad40377a074efc6d92b4a19c6"
EXPECTED_ROI_DATASET_FP = "d25132b9afae3bc858d3470b0a6cc5253abf5085c7ba3d1596a6f4bdc3d7ce77"
CLASS_NAMES = {0: "drone", 1: "airplane", 2: "helicopter", 3: "bird", 4: "background"}
CLASS_IDS = [0, 1, 2, 3, 4]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_hash(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip().split()[0]


def ensure_clean(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}. Use --overwrite-output.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype={"roi_id": "string"})
    df["assigned_class_id"] = pd.to_numeric(df["assigned_class_id"], errors="raise").astype(int)
    return df


def load_feature_cache(root: Path):
    config = json.loads((root / "feature_config.json").read_text(encoding="utf-8"))
    if config.get("test_accessed") is not False:
        raise RuntimeError("Feature cache does not certify TEST accessed = False")
    if config.get("parent_roi_dataset_fingerprint") != EXPECTED_ROI_DATASET_FP:
        raise RuntimeError("ROI fingerprint mismatch in feature cache")
    if config.get("classifier_manifest_package_fingerprint") != EXPECTED_MANIFEST_PACKAGE_FP:
        raise RuntimeError("Manifest fingerprint mismatch in feature cache")
    idx = pd.read_csv(root / "feature_index.csv", low_memory=False, dtype={"roi_id": "string"})
    X = np.load(root / "features.float32.npy", mmap_mode="r")
    if X.shape[0] != len(idx):
        raise RuntimeError("Feature matrix/index row mismatch")
    if X.shape[1] != int(config["feature_dim"]):
        raise RuntimeError("Feature dimension mismatch")
    return X, idx, config


def rows_for_manifest(manifest: pd.DataFrame, feature_index: pd.DataFrame):
    mapping = pd.Series(feature_index["feature_row"].to_numpy(), index=feature_index["roi_id"].astype(str))
    ids = manifest["roi_id"].astype(str)
    if not ids.isin(mapping.index).all():
        missing = ids[~ids.isin(mapping.index)].head(10).tolist()
        raise RuntimeError(f"Manifest ROI IDs missing from feature cache: {missing}")
    rows = mapping.loc[ids].to_numpy(dtype=np.int64)
    y = manifest["assigned_class_id"].to_numpy(dtype=np.int64)
    return rows, y


def evaluate(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_IDS, zero_division=0
    )
    per_class = []
    for i, cid in enumerate(CLASS_IDS):
        per_class.append({
            "class_id": cid,
            "class_name": CLASS_NAMES[cid],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_IDS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASS_IDS, average="weighted", zero_division=0)),
        "drone_f1": float(f1[0]),
        "drone_recall": float(recall[0]),
        "drone_precision": float(precision[0]),
    }
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    return summary, pd.DataFrame(per_class), cm


def save_eval(outdir: Path, y_true, y_pred, manifest: pd.DataFrame, summary, per_class, cm):
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(outdir / "metrics_summary.csv", index=False, lineterminator="\n")
    per_class.to_csv(outdir / "per_class_metrics.csv", index=False, lineterminator="\n")
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{CLASS_NAMES[c]}" for c in CLASS_IDS],
        columns=[f"pred_{CLASS_NAMES[c]}" for c in CLASS_IDS],
    )
    cm_df.to_csv(outdir / "confusion_matrix.csv", lineterminator="\n")
    pred = pd.DataFrame({
        "roi_id": manifest["roi_id"].astype(str).to_numpy(),
        "y_true": y_true,
        "y_pred": y_pred,
        "true_name": [CLASS_NAMES[int(x)] for x in y_true],
        "pred_name": [CLASS_NAMES[int(x)] for x in y_pred],
    })
    pred.to_csv(outdir / "valid_predictions.csv", index=False, lineterminator="\n")
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=[CLASS_NAMES[c] for c in CLASS_IDS],
        digits=6,
        zero_division=0,
    )
    (outdir / "classification_report.txt").write_text(report, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="VALID-only profile selection for Linear SVM and Random Forest.")
    ap.add_argument("--features-root", type=Path, default=Path("hog_hsv_features_v1"))
    ap.add_argument("--manifests-root", type=Path, default=Path("classifier_manifests_v1"))
    ap.add_argument("--outdir", type=Path, default=Path("svm_rf_valid_selection_v1"))
    ap.add_argument("--overwrite-output", action="store_true")
    ap.add_argument("--svm-c", type=float, default=1.0)
    ap.add_argument("--rf-trees", type=int, default=500)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if read_hash(args.manifests_root / "classifier_manifest_package_fingerprint.sha256") != EXPECTED_MANIFEST_PACKAGE_FP:
        raise RuntimeError("Classifier manifest package fingerprint mismatch")

    # This script intentionally has no TEST manifest argument and never opens test.csv.
    X, feature_index, feature_config = load_feature_cache(args.features_root)
    valid_manifest = read_manifest(args.manifests_root / "shared" / "valid_natural.csv")
    valid_rows, y_valid = rows_for_manifest(valid_manifest, feature_index)
    X_valid = X[valid_rows]

    ensure_clean(args.outdir, args.overwrite_output)

    result_rows = []
    fitted = {}
    profiles = ["bg1x", "bg2x"]

    for profile in profiles:
        train_manifest = read_manifest(args.manifests_root / profile / "train.csv")
        train_rows, y_train = rows_for_manifest(train_manifest, feature_index)
        X_train = X[train_rows]

        models = {
            "linear_svm": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LinearSVC(
                    C=args.svm_c,
                    class_weight=None,
                    dual=False,
                    tol=1e-4,
                    max_iter=20000,
                    random_state=args.seed,
                )),
            ]),
            "random_forest": RandomForestClassifier(
                n_estimators=args.rf_trees,
                criterion="gini",
                max_depth=args.rf_max_depth,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
                class_weight=None,
                random_state=args.seed,
                n_jobs=-1,
                verbose=0,
            ),
        }

        for model_name, model in models.items():
            print("=" * 88, flush=True)
            print(f"TRAIN {model_name} | profile={profile} | n_train={len(train_manifest)} | n_valid={len(valid_manifest)}", flush=True)
            t0 = time.time()
            model.fit(X_train, y_train)
            fit_s = time.time() - t0
            t1 = time.time()
            y_pred = model.predict(X_valid)
            pred_s = time.time() - t1

            summary, per_class, cm = evaluate(y_valid, y_pred)
            summary.update({
                "model": model_name,
                "profile": profile,
                "train_samples": len(train_manifest),
                "valid_samples": len(valid_manifest),
                "fit_seconds": fit_s,
                "predict_seconds": pred_s,
                "test_accessed": False,
            })
            result_rows.append(summary)

            mdir = args.outdir / model_name / profile
            save_eval(mdir, y_valid, y_pred, valid_manifest, summary, per_class, cm)
            model_path = mdir / "model.joblib"
            joblib.dump(model, model_path, compress=3)
            fitted[(model_name, profile)] = model_path

            print(
                f"macro_f1={summary['macro_f1']:.6f} | balanced_acc={summary['balanced_accuracy']:.6f} | "
                f"drone_f1={summary['drone_f1']:.6f} | fit={fit_s/60:.1f} min",
                flush=True,
            )

    results = pd.DataFrame(result_rows)
    results = results.sort_values(["model", "macro_f1", "drone_f1", "balanced_accuracy", "profile"], ascending=[True, False, False, False, True])
    results.to_csv(args.outdir / "valid_model_profile_results.csv", index=False, lineterminator="\n")

    selections = []
    for model_name in ["linear_svm", "random_forest"]:
        g = results[results["model"] == model_name].copy()
        best = g.iloc[0]
        selections.append({
            "model": model_name,
            "selected_profile": str(best["profile"]),
            "selection_metric": "macro_f1",
            "macro_f1": float(best["macro_f1"]),
            "drone_f1": float(best["drone_f1"]),
            "balanced_accuracy": float(best["balanced_accuracy"]),
            "test_accessed": False,
        })
    selection_df = pd.DataFrame(selections)
    selection_df.to_csv(args.outdir / "selected_profiles_valid_only.csv", index=False, lineterminator="\n")

    cfg = {
        "experiment": "SVM_RF_VALID_ONLY_V1",
        "test_accessed": False,
        "selection_metric_primary": "macro_f1",
        "tie_breakers": ["drone_f1", "balanced_accuracy", "profile_name_ascending"],
        "features": json.loads((args.features_root / "feature_config.json").read_text(encoding="utf-8")),
        "models": {
            "linear_svm": {
                "implementation": "sklearn.svm.LinearSVC",
                "C": args.svm_c,
                "class_weight": None,
                "dual": False,
                "tol": 1e-4,
                "max_iter": 20000,
                "random_state": args.seed,
                "preprocessing": "StandardScaler fitted on TRAIN only",
            },
            "random_forest": {
                "implementation": "sklearn.ensemble.RandomForestClassifier",
                "n_estimators": args.rf_trees,
                "criterion": "gini",
                "max_depth": args.rf_max_depth,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
                "class_weight": None,
                "random_state": args.seed,
                "n_jobs": -1,
            },
        },
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn_version,
            "joblib": joblib.__version__,
        },
    }
    (args.outdir / "experiment_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    hashes = {}
    for p in sorted(args.outdir.rglob("*")):
        if p.is_file() and p.name != "artifact_hashes.json":
            hashes[str(p.relative_to(args.outdir))] = sha256_file(p)
    (args.outdir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    exp_fp = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    (args.outdir / "valid_selection_fingerprint.sha256").write_text(exp_fp + "\n", encoding="utf-8")

    print("=" * 88)
    print("VALID-ONLY SVM/RF SELECTION COMPLETED")
    print("=" * 88)
    print(selection_df.to_string(index=False))
    print("TEST accessed: NO")
    print(f"selection fingerprint: {exp_fp}")
    print(f"output: {args.outdir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
