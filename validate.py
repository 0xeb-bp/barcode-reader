#!/usr/bin/env python3
"""
Blind validation: predict on held-out games that weren't used in training.

Players with more games than the training cap have unseen replays.
This script predicts those and reports accuracy — a true blind test.

Modes:
  (default)          Per-game accuracy — each game predicted independently
  --ensemble vote    Majority vote — each game votes, most common wins
  --ensemble proba   Average probabilities — average predict_proba across games
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Blind validation on held-out games")
    parser.add_argument("--ensemble", choices=["vote", "proba"],
                        help="Ensemble mode: 'vote' for majority vote, 'proba' for averaged probabilities")
    parser.add_argument("--min-held-out", type=int, default=3,
                        help="Minimum held-out games to include a player (default 3)")
    args = parser.parse_args()

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

        if len(held_out) < args.min_held_out:
            continue

        # Apply ngram features
        for s in held_out:
            apply_ngram_features(s["features"], s["raw_ngrams"], global_ngrams)

        # Build feature matrix
        X = np.array([[s["features"].get(f, 0) for f in feature_names] for s in held_out])
        X_scaled = scaler.transform(X)

        if args.ensemble:
            # Ensemble: one prediction per player from all their held-out games
            if args.ensemble == "vote":
                preds = clf.predict(X_scaled)
                vote_counts = Counter(preds)
                winner = vote_counts.most_common(1)[0][0]
                winner_votes = vote_counts.most_common(1)[0][1]
                is_correct = winner == canonical
                # Runner-up info
                others = {k: v for k, v in vote_counts.items() if k != winner}
            else:  # proba
                probas = clf.predict_proba(X_scaled)
                avg_proba = probas.mean(axis=0)
                classes = clf.classes_
                top_idx = np.argmax(avg_proba)
                winner = classes[top_idx]
                winner_votes = avg_proba[top_idx]
                is_correct = winner == canonical
                # Runner-up info
                sorted_idx = np.argsort(avg_proba)[::-1]
                others = {classes[i]: avg_proba[i] for i in sorted_idx[1:4] if avg_proba[i] > 0.01}

            total_correct += int(is_correct)
            total_tested += 1

            player_results.append({
                "player": canonical,
                "held_out": len(held_out),
                "winner": winner,
                "winner_score": winner_votes,
                "correct": is_correct,
                "others": others,
            })
        else:
            # Per-game: each game predicted independently
            preds = clf.predict(X_scaled)
            correct = sum(1 for p in preds if p == canonical)
            accuracy = correct / len(preds)
            total_correct += correct
            total_tested += len(preds)

            wrong_preds = Counter(p for p in preds if p != canonical)

            player_results.append({
                "player": canonical,
                "held_out": len(held_out),
                "correct": correct,
                "accuracy": accuracy,
                "wrong_preds": dict(wrong_preds),
            })

    conn.close()

    # Print results
    if args.ensemble:
        mode_label = "majority vote" if args.ensemble == "vote" else "averaged probabilities"
        print(f"\n{'=' * 90}")
        print(f"BLIND VALIDATION — ensemble ({mode_label})")
        print(f"{'=' * 90}")

        player_results.sort(key=lambda x: (x["correct"], -x["held_out"]))

        if args.ensemble == "vote":
            print(f"{'Player':<15} {'Games':>5} {'Prediction':<15} {'Votes':>7} {'':>3}  Runner-up")
            print(f"{'-' * 90}")
            for r in player_results:
                mark = "OK" if r["correct"] else "MISS"
                others_str = ", ".join(f"{k}({v})" for k, v in
                                       sorted(r["others"].items(), key=lambda x: -x[1]))
                print(f"{r['player']:<15} {r['held_out']:>5} {r['winner']:<15} "
                      f"{r['winner_score']:>5}/{r['held_out']:<3} {mark:>4}  {others_str}")
        else:  # proba
            print(f"{'Player':<15} {'Games':>5} {'Prediction':<15} {'Conf':>7} {'':>3}  Runner-up")
            print(f"{'-' * 90}")
            for r in player_results:
                mark = "OK" if r["correct"] else "MISS"
                others_str = ", ".join(f"{k}({v:.1%})" for k, v in
                                       sorted(r["others"].items(), key=lambda x: -x[1]))
                print(f"{r['player']:<15} {r['held_out']:>5} {r['winner']:<15} "
                      f"{r['winner_score']:>6.1%} {mark:>4}  {others_str}")

        print(f"{'-' * 90}")
        overall = total_correct / total_tested if total_tested else 0
        print(f"TOTAL: {total_correct}/{total_tested} players correctly identified ({overall:.1%})")

    else:
        # Per-game output (original)
        player_results.sort(key=lambda x: (x["accuracy"], -x["held_out"]))

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


if __name__ == "__main__":
    main()
