#!/usr/bin/env python3
"""
Train player fingerprint classifier and save model to disk.
"""

import sqlite3
import sys
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
import joblib

from features import (
    DB_PATH, extract_player_samples, get_pro_aliases,
    select_global_ngrams, apply_ngram_features, create_feature_matrix,
)

MODEL_PATH = DB_PATH.parent / "model.joblib"
MIN_DATE = "2025-01-01"  # Modern era only


def main():
    print("=" * 70)
    print(f"TRAINING PLAYER FINGERPRINT CLASSIFIER (modern era: >={MIN_DATE})")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    # Get all confirmed pros — we'll filter by date during extraction
    pros = get_pro_aliases(conn, min_games=10)
    print(f"\nConfirmed pros with 10+ games (all eras): {len(pros)}")
    sys.stdout.flush()

    # Build training dataset — modern only
    print(f"\nExtracting features from modern games (>= {MIN_DATE})...")
    sys.stdout.flush()
    all_samples = []

    for canonical, aliases, total in pros:
        player_samples = []
        for alias in aliases:
            samples = extract_player_samples(conn, alias, label=canonical,
                                             min_date=MIN_DATE, max_games=25)
            player_samples.extend(samples)
        # Cap at 25 per player to avoid class imbalance
        if len(player_samples) > 25:
            np.random.seed(42)
            indices = np.random.choice(len(player_samples), 25, replace=False)
            player_samples = [player_samples[i] for i in indices]
        if len(player_samples) < 10:
            print(f"  {canonical}: {len(player_samples)} valid games — SKIPPED (< 10)")
            sys.stdout.flush()
            continue
        all_samples.extend(player_samples)
        print(f"  {canonical}: {len(player_samples)} valid games (from {len(aliases)} aliases)")
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

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
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
