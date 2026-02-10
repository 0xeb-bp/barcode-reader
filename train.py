#!/usr/bin/env python3
"""
Train player fingerprint classifier and save model to disk.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
import joblib

from collections import Counter

from features import (
    DB_PATH, extract_player_samples_by_aurora, get_pro_identities,
    select_global_ngrams, apply_ngram_features, create_feature_matrix,
)

MODEL_PATH = DB_PATH.parent / "model.joblib"
CV_RESULTS_PATH = DB_PATH.parent / "cv_results.json"
MIN_DATE = "2025-01-01"  # Modern era only
MIN_GAMES = 20
MIN_OFFRACE = 20  # Keep offrace games only if player has >= this many


def main():
    parser = argparse.ArgumentParser(description="Train player fingerprint classifier")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Max games per player (default: unlimited). Use 25 for fast experiments, 100 for production.")
    parser.add_argument("--analyze", action="store_true",
                        help="Run outlier detection after CV (flag samples far from class centroid)")
    args = parser.parse_args()
    max_games = args.max_games

    cap_str = f", max {max_games}/player" if max_games else ""
    print("=" * 70)
    print(f"TRAINING PLAYER FINGERPRINT CLASSIFIER (modern era: >={MIN_DATE}{cap_str})")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    # Get all confirmed pros from player_identities (aurora_id-based)
    pros = get_pro_identities(conn, min_games=MIN_GAMES)
    print(f"\nConfirmed pros with {MIN_GAMES}+ modern games: {len(pros)}")
    sys.stdout.flush()

    # Build training dataset — modern only, aurora_id-based
    print(f"\nExtracting features from modern games (>= {MIN_DATE})...")
    sys.stdout.flush()
    all_samples = []

    for canonical, aurora_ids, total in pros:
        player_samples = extract_player_samples_by_aurora(
            conn, aurora_ids, label=canonical, min_date=MIN_DATE)

        # Per-race filtering: main race capped at max_games, each offrace kept only if >= MIN_OFFRACE
        race_counts = Counter(s["race"] for s in player_samples)
        main_race = race_counts.most_common(1)[0][0] if race_counts else None

        selected = []
        filtered_races = []
        kept_races = {}
        for race, count in race_counts.most_common():
            race_games = [s for s in player_samples if s["race"] == race]
            if race != main_race and count < MIN_OFFRACE:
                filtered_races.append(f"{count} {race}")
                continue
            if max_games and len(race_games) > max_games:
                race_games = race_games[:max_games]
            selected.extend(race_games)
            if race != main_race:
                kept_races[race] = count

        player_samples = selected
        race_note = ""
        if kept_races:
            race_note += f" [offrace kept: {kept_races}]"
        if filtered_races:
            race_note += f" [filtered: {', '.join(filtered_races)}]"
        if len(player_samples) < MIN_GAMES:
            print(f"  {canonical}: {len(player_samples)} valid games — SKIPPED (< {MIN_GAMES})")
            sys.stdout.flush()
            continue
        all_samples.extend(player_samples)
        aliases_used = set(s["alias"] for s in player_samples)
        print(f"  {canonical}: {len(player_samples)} valid games (aurora_ids: {aurora_ids}, names: {aliases_used}){race_note}")
        sys.stdout.flush()

    print(f"\nTotal training samples: {len(all_samples)}")

    # Two-pass n-gram selection
    print("\nTwo-pass n-gram selection...")
    global_ngrams = select_global_ngrams([s["raw_ngrams"] for s in all_samples])
    total_ngram_features = 0
    for prefix in sorted(global_ngrams.keys()):
        grams = global_ngrams[prefix]
        total_ngram_features += len(grams)
        print(f"  {prefix}: {len(grams)} global n-grams selected")
    print(f"  Total n-gram features: {total_ngram_features}")

    # Apply global n-grams to all training samples
    for s in all_samples:
        apply_ngram_features(s["features"], s["raw_ngrams"], global_ngrams)

    # Create feature matrix
    X, y, feature_names = create_feature_matrix(all_samples)
    print(f"Feature matrix shape: {X.shape}")

    # LOO cross-validation
    print("\n" + "=" * 70)
    print("LEAVE-ONE-OUT CROSS-VALIDATION")
    print("=" * 70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42,
                                    class_weight="balanced", n_jobs=-1)
    loo = LeaveOneOut()
    predictions = cross_val_predict(clf, X_scaled, y, cv=loo)

    correct = sum(1 for p, a in zip(predictions, y) if p == a)
    accuracy = correct / len(y)

    print(f"\nOverall accuracy: {correct}/{len(y)} = {accuracy:.1%}")

    # Per-player accuracy
    print("\nPer-player accuracy:")
    player_results = {}
    for sample, pred in zip(all_samples, predictions):
        actual = sample["label"]
        if actual not in player_results:
            player_results[actual] = {"correct": 0, "total": 0}
        player_results[actual]["total"] += 1
        if pred == actual:
            player_results[actual]["correct"] += 1

    for player in sorted(player_results.keys(),
                         key=lambda p: -player_results[p]["correct"]/player_results[p]["total"]):
        r = player_results[player]
        pct = r["correct"] / r["total"] * 100
        bar = "#" * int(pct / 5)
        print(f"  {player:25s}: {r['correct']:2d}/{r['total']:2d} = {pct:5.1f}% {bar}")

    # Save per-sample CV results
    cv_results = []
    for i, (true, pred) in enumerate(zip(y, predictions)):
        cv_results.append({
            "sample_idx": i,
            "true_label": true,
            "predicted": pred,
            "correct": true == pred,
            "alias": all_samples[i]["alias"],
            "file": all_samples[i]["file"],
            "replay_id": all_samples[i]["replay_id"],
            "race": all_samples[i]["race"],
        })

    with open(CV_RESULTS_PATH, "w") as f:
        json.dump(cv_results, f, indent=2)
    misclassified = [r for r in cv_results if not r["correct"]]
    print(f"\nCV results saved to: {CV_RESULTS_PATH}")
    print(f"  Misclassified: {len(misclassified)}/{len(cv_results)}")
    if misclassified:
        print("\n  Misclassifications:")
        for r in misclassified:
            print(f"    {r['alias']:20s} → predicted {r['predicted']:15s} (true: {r['true_label']}) [{r['file']}]")

    # Outlier detection
    if args.analyze:
        print("\n" + "=" * 70)
        print("OUTLIER DETECTION (Euclidean distance from class centroid)")
        print("=" * 70)

        classes = sorted(set(y))
        distances = np.zeros(len(y))
        class_stats = {}

        for cls in classes:
            mask = y == cls
            cls_features = X_scaled[mask]
            centroid = cls_features.mean(axis=0)
            dists = np.sqrt(((cls_features - centroid) ** 2).sum(axis=1))
            distances[mask] = dists
            class_stats[cls] = {"mean": dists.mean(), "std": dists.std()}

        # Flag samples > 2.5 std devs from their class centroid
        outlier_threshold = 2.5
        outlier_indices = []
        for i in range(len(y)):
            cls = y[i]
            stats = class_stats[cls]
            if stats["std"] > 0 and (distances[i] - stats["mean"]) / stats["std"] > outlier_threshold:
                outlier_indices.append(i)

        print(f"\nOutlier threshold: >{outlier_threshold} std devs from class centroid")
        print(f"Total outliers: {len(outlier_indices)}/{len(y)}")

        # Per-player outlier counts
        outlier_by_player = {}
        for i in outlier_indices:
            cls = y[i]
            outlier_by_player.setdefault(cls, []).append(i)

        if outlier_by_player:
            print("\nPer-player outlier counts:")
            for player in sorted(outlier_by_player.keys()):
                indices = outlier_by_player[player]
                total_player = sum(1 for label in y if label == player)
                print(f"  {player:25s}: {len(indices)}/{total_player} outliers")

            print("\nOutlier details:")
            for i in outlier_indices:
                s = all_samples[i]
                cls = y[i]
                stats = class_stats[cls]
                z_score = (distances[i] - stats["mean"]) / stats["std"] if stats["std"] > 0 else 0
                mis_flag = " ** MISCLASSIFIED **" if predictions[i] != y[i] else ""
                print(f"  {s['alias']:20s} dist={distances[i]:.1f} z={z_score:.1f} [{s['file']}]{mis_flag}")

        # Overlap: misclassified AND outlier
        outlier_set = set(outlier_indices)
        misclassified_indices = set(i for i, r in enumerate(cv_results) if not r["correct"])
        overlap = outlier_set & misclassified_indices
        if overlap:
            print(f"\nMisclassified + Outlier overlap: {len(overlap)} samples")
            for i in sorted(overlap):
                s = all_samples[i]
                print(f"  {s['alias']:20s} predicted={predictions[i]:15s} true={y[i]} [{s['file']}]")
        else:
            print("\nNo overlap between misclassified and outlier samples.")

    # Train final model on all data
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    clf.fit(X_scaled, y)

    # Feature importance
    importances = sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1])
    print("\nTop 15 most important features:")
    for name, imp in importances[:15]:
        bar = "#" * int(imp * 100)
        print(f"  {name:30s} {imp:.3f} {bar}")

    # Save model
    model_data = {
        "clf": clf,
        "scaler": scaler,
        "feature_names": feature_names,
        "global_ngrams": global_ngrams,
        "classes": list(clf.classes_),
        "accuracy": accuracy,
        "num_samples": len(all_samples),
        "num_players": len(pros),
        "trained_at": datetime.now().isoformat(),
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"  Classes: {model_data['classes']}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Accuracy: {accuracy:.1%}")

    conn.close()


if __name__ == "__main__":
    main()
