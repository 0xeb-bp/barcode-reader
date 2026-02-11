#!/usr/bin/env python3
"""
Blind validation: predict on held-out games that weren't used in training.

Players with more games than the training cap have unseen replays.
This script predicts those and reports accuracy — a true blind test.
"""

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import joblib

from features import (
    DB_PATH, extract_player_samples_by_aurora, get_pro_identities,
    apply_ngram_features,
)

MODEL_PATH = DB_PATH.parent / "model.joblib"
CV_RESULTS_PATH = DB_PATH.parent / "cv_results.json"
MIN_DATE = "2025-01-01"


def main():
    # Load model
    model = joblib.load(MODEL_PATH)
    print(f"Model: {model['trained_at']}, accuracy: {model['accuracy']:.1%}, "
          f"classes: {len(model['classes'])}")

    # Load training set replay IDs
    with open(CV_RESULTS_PATH) as f:
        cv_results = json.load(f)
    training_replay_ids = {r["replay_id"] for r in cv_results}
    print(f"Training set: {len(training_replay_ids)} replays")

    clf = model["clf"]
    scaler = model["scaler"]
    feature_names = model["feature_names"]
    global_ngrams = model["global_ngrams"]

    conn = sqlite3.connect(DB_PATH)
    pros = get_pro_identities(conn, min_games=20)

    total_correct = 0
    total_tested = 0
    player_results = []

    for canonical, aurora_ids, total in pros:
        # Extract ALL samples for this player
        all_samples = extract_player_samples_by_aurora(
            conn, aurora_ids, label=canonical, min_date=MIN_DATE)

        # Determine which races were trained on for this player.
        # train.py filters out offrace games if that race has < MIN_OFFRACE (20) games.
        # We must mirror that here: only validate on races the model actually saw
        # in training, otherwise we'd be testing on offrace games the model never
        # learned, which would show as 0% and pollute the accuracy numbers.
        trained_samples = [s for s in all_samples if s["replay_id"] in training_replay_ids]
        trained_races = {s["race"] for s in trained_samples}

        held_out = [s for s in all_samples
                    if s["replay_id"] not in training_replay_ids
                    and s["race"] in trained_races]

        if len(held_out) < 3:
            continue

        # Apply ngram features
        for s in held_out:
            apply_ngram_features(s["features"], s["raw_ngrams"], global_ngrams)

        # Build feature matrix and predict
        X = np.array([[s["features"].get(f, 0) for f in feature_names] for s in held_out])
        X_scaled = scaler.transform(X)
        preds = clf.predict(X_scaled)

        correct = sum(1 for p in preds if p == canonical)
        accuracy = correct / len(preds)
        total_correct += correct
        total_tested += len(preds)

        # What did wrong predictions say?
        wrong_preds = Counter(p for p in preds if p != canonical)

        player_results.append({
            "player": canonical,
            "held_out": len(held_out),
            "correct": correct,
            "accuracy": accuracy,
            "wrong_preds": dict(wrong_preds),
        })

    # Sort by accuracy
    player_results.sort(key=lambda x: (x["accuracy"], -x["held_out"]))

    # Print results
    print(f"\n{'=' * 80}")
    print(f"BLIND VALIDATION — held-out games not used in training")
    print(f"{'=' * 80}")
    print(f"{'Player':<20} {'Held-out':>8} {'Correct':>8} {'Accuracy':>8}   Wrong predictions")
    print(f"{'-' * 80}")

    for r in player_results:
        wrong_str = ""
        if r["wrong_preds"]:
            wrong_str = ", ".join(f"{v}x {k}" for k, v in
                                  sorted(r["wrong_preds"].items(), key=lambda x: -x[1]))
        bar = "#" * int(r["accuracy"] * 20)
        print(f"{r['player']:<20} {r['held_out']:>8} {r['correct']:>8} "
              f"{r['accuracy']:>7.1%}   {bar}  {wrong_str}")

    overall = total_correct / total_tested if total_tested else 0
    print(f"{'-' * 80}")
    print(f"{'TOTAL':<20} {total_tested:>8} {total_correct:>8} {overall:>7.1%}")

    players_tested = len(player_results)
    perfect = sum(1 for r in player_results if r["accuracy"] == 1.0)
    print(f"\nPlayers tested: {players_tested}, Perfect accuracy: {perfect}/{players_tested}")

    conn.close()


if __name__ == "__main__":
    main()
