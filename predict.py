#!/usr/bin/env python3
"""
Load trained model and predict on all unlabeled players.
"""

import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import joblib

from features import (
    DB_PATH, extract_player_samples, extract_player_samples_by_aurora,
    apply_ngram_features,
)

MODEL_PATH = DB_PATH.parent / "model.joblib"
PREDICTIONS_PATH = DB_PATH.parent / "predictions.json"
MIN_GAMES = 10
MIN_DATE = "2025-01-01"  # Modern era only, same as training


def get_unlabeled_players(conn, min_games=MIN_GAMES, min_date=MIN_DATE):
    """Get unlabeled players with min_games+ modern games.
    Returns two lists:
      - aurora_id-based: (aurora_id, [display_names], game_count, races)
      - name-based fallback: (player_name, game_count, races) for players without aurora_id
    """
    c = conn.cursor()

    # Players with aurora_id not in player_identities
    c.execute("""
        SELECT p.aurora_id,
               GROUP_CONCAT(DISTINCT p.player_name) as names,
               COUNT(DISTINCT p.replay_id) as cnt,
               GROUP_CONCAT(DISTINCT p.race) as races
        FROM players p
        LEFT JOIN player_identities pi ON p.aurora_id = pi.aurora_id
        JOIN replays r ON r.id = p.replay_id
        WHERE pi.id IS NULL
          AND p.aurora_id IS NOT NULL
          AND p.is_human = 1
          AND r.game_date >= ?
        GROUP BY p.aurora_id
        HAVING cnt >= ?
        ORDER BY cnt DESC
    """, (min_date, min_games))
    aurora_players = c.fetchall()

    # Fallback: players without aurora_id and not in player_identities
    c.execute("""
        SELECT p.player_name,
               COUNT(DISTINCT p.replay_id) as cnt,
               GROUP_CONCAT(DISTINCT p.race) as races
        FROM players p
        LEFT JOIN player_identities pi ON p.aurora_id = pi.aurora_id
        JOIN replays r ON r.id = p.replay_id
        WHERE pi.id IS NULL
          AND p.aurora_id IS NULL
          AND p.is_human = 1
          AND r.game_date >= ?
        GROUP BY p.player_name
        HAVING cnt >= ?
        ORDER BY cnt DESC
    """, (min_date, min_games))
    name_players = c.fetchall()

    return aurora_players, name_players


def main():
    # Load model
    if not MODEL_PATH.exists():
        print(f"No model found at {MODEL_PATH}. Run train.py first.")
        sys.exit(1)

    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    clf = model["clf"]
    scaler = model["scaler"]
    feature_names = model["feature_names"]
    global_ngrams = model["global_ngrams"]

    print(f"  Trained: {model['trained_at']}")
    print(f"  Accuracy: {model['accuracy']:.1%}")
    print(f"  Classes: {model['classes']}")
    print(f"  Features: {len(feature_names)}")

    conn = sqlite3.connect(DB_PATH)

    # Get unlabeled players
    aurora_players, name_players = get_unlabeled_players(conn, MIN_GAMES)
    total_players = len(aurora_players) + len(name_players)
    print(f"\nUnlabeled players with {MIN_GAMES}+ games: {total_players}")
    print(f"  Aurora_id-based: {len(aurora_players)}, Name-based fallback: {len(name_players)}")

    all_predictions = []
    idx = 0

    # Aurora_id-based predictions
    for aurora_id, names, game_count, races in aurora_players:
        if idx % 20 == 0:
            print(f"  Processing {idx+1}/{total_players}...")
            sys.stdout.flush()
        idx += 1

        samples = extract_player_samples_by_aurora(
            conn, [aurora_id], label=None, min_date=MIN_DATE)

        for s in samples:
            apply_ngram_features(s["features"], s["raw_ngrams"], global_ngrams)

        if len(samples) < 3:
            continue

        X = np.array([[s["features"].get(f, 0) for f in feature_names] for s in samples])
        X_scaled = scaler.transform(X)

        preds = clf.predict(X_scaled)
        probs = clf.predict_proba(X_scaled)

        pred_counts = Counter(preds)
        top_pred, top_count = pred_counts.most_common(1)[0]
        confidence = top_count / len(preds)

        class_idx = list(clf.classes_).index(top_pred)
        avg_prob = float(np.mean([p[class_idx] for p in probs]))

        display_name = names.split(",")[0] if names else f"aurora:{aurora_id}"
        all_predictions.append({
            "player": display_name,
            "aurora_id": aurora_id,
            "display_names": names.split(",") if names else [],
            "races": races,
            "total_games": game_count,
            "games_analyzed": len(samples),
            "prediction": top_pred,
            "confidence": confidence,
            "avg_prob": avg_prob,
            "all_preds": dict(pred_counts),
        })

    # Name-based fallback predictions
    for player_name, game_count, races in name_players:
        if idx % 20 == 0:
            print(f"  Processing {idx+1}/{total_players}...")
            sys.stdout.flush()
        idx += 1

        samples = extract_player_samples(conn, player_name, min_date=MIN_DATE, max_games=20)

        for s in samples:
            apply_ngram_features(s["features"], s["raw_ngrams"], global_ngrams)

        if len(samples) < 3:
            continue

        X = np.array([[s["features"].get(f, 0) for f in feature_names] for s in samples])
        X_scaled = scaler.transform(X)

        preds = clf.predict(X_scaled)
        probs = clf.predict_proba(X_scaled)

        pred_counts = Counter(preds)
        top_pred, top_count = pred_counts.most_common(1)[0]
        confidence = top_count / len(preds)

        class_idx = list(clf.classes_).index(top_pred)
        avg_prob = float(np.mean([p[class_idx] for p in probs]))

        all_predictions.append({
            "player": player_name,
            "aurora_id": None,
            "display_names": [player_name],
            "races": races,
            "total_games": game_count,
            "games_analyzed": len(samples),
            "prediction": top_pred,
            "confidence": confidence,
            "avg_prob": avg_prob,
            "all_preds": dict(pred_counts),
        })

    # Sort by confidence
    all_predictions.sort(key=lambda x: (-x["confidence"], -x["avg_prob"]))

    # Print results
    print("\n" + "=" * 100)
    print("PREDICTIONS (sorted by confidence)")
    print("=" * 100)
    print(f"{'Player':<25} {'Race':<12} {'Games':>5} {'Analyzed':>8} {'Prediction':<15} {'Conf':>6} {'Prob':>6}")
    print("-" * 100)

    for p in all_predictions:
        print(f"{p['player'][:24]:<25} {p['races'][:11]:<12} {p['total_games']:>5} "
              f"{p['games_analyzed']:>8} {p['prediction']:<15} "
              f"{p['confidence']:>5.0%} {p['avg_prob']:>5.0%}")

    # High confidence matches
    high_conf = [p for p in all_predictions if p["confidence"] >= 0.6]
    print(f"\n{'=' * 100}")
    print(f"HIGH CONFIDENCE MATCHES (>= 60%): {len(high_conf)}")
    print("=" * 100)

    for p in high_conf:
        names_str = ", ".join(p.get("display_names", [p["player"]]))
        aurora_str = f" [aurora:{p['aurora_id']}]" if p.get("aurora_id") else ""
        print(f"\n  {names_str}{aurora_str} ({p['races']}) -> {p['prediction']}")
        print(f"    Confidence: {p['confidence']:.0%} ({p['games_analyzed']} games analyzed)")
        print(f"    Avg probability: {p['avg_prob']:.0%}")
        print(f"    All predictions: {p['all_preds']}")

    # Save to JSON
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump({
            "model_trained_at": model["trained_at"],
            "model_accuracy": model["accuracy"],
            "predicted_at": datetime.now().isoformat(),
            "min_games": MIN_GAMES,
            "total_players": len(all_predictions),
            "predictions": all_predictions,
        }, f, indent=2)
    print(f"\nPredictions saved to: {PREDICTIONS_PATH}")

    conn.close()


if __name__ == "__main__":
    main()
