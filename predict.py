#!/usr/bin/env python3
"""
Load trained model and predict player identity from replays.

Usage:
  python predict.py                        # all unlabeled players
  python predict.py --aurora-id 19619537   # specific bnet account
  python predict.py --name qweqewqqe       # specific player name
"""

import argparse
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
MIN_GAMES = 20
MIN_DATE = "2025-01-01"  # Modern era only, same as training


def load_model():
    """Load trained model from disk."""
    if not MODEL_PATH.exists():
        print(f"No model found at {MODEL_PATH}. Run train.py first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    print("Loading model...")
    print(f"  Trained: {model['trained_at']}")
    print(f"  Accuracy: {model['accuracy']:.1%}")
    print(f"  Classes: {list(model['classes'])}")
    print(f"  Features: {len(model['feature_names'])}")
    return model


def predict_samples(samples, model):
    """Run the full prediction pipeline on extracted samples.

    Handles ngram projection, scaling, and classification.
    Returns None if not enough samples, otherwise a dict with:
      prediction, confidence, avg_prob, all_preds, games_analyzed, per_game
    """
    if len(samples) < 3:
        return None

    clf = model["clf"]
    scaler = model["scaler"]
    feature_names = model["feature_names"]
    global_ngrams = model["global_ngrams"]

    # Apply n-gram features (the step that must not be skipped)
    for s in samples:
        apply_ngram_features(s["features"], s["raw_ngrams"], global_ngrams)

    # Build feature matrix and scale
    X = np.array([[s["features"].get(f, 0) for f in feature_names] for s in samples])
    X_scaled = scaler.transform(X)

    # Predict
    preds = clf.predict(X_scaled)
    probs = clf.predict_proba(X_scaled)

    # Aggregate
    pred_counts = Counter(preds)
    top_pred, top_count = pred_counts.most_common(1)[0]
    confidence = top_count / len(preds)

    class_idx = list(clf.classes_).index(top_pred)
    avg_prob = float(np.mean([p[class_idx] for p in probs]))

    # Top classes by average probability
    avg_probs = {
        cls: float(np.mean([p[i] for p in probs]))
        for i, cls in enumerate(clf.classes_)
    }

    return {
        "prediction": top_pred,
        "confidence": confidence,
        "avg_prob": avg_prob,
        "all_preds": dict(pred_counts),
        "avg_probs": avg_probs,
        "games_analyzed": len(samples),
    }


# --- Single-player prediction ---

def predict_by_aurora_id(conn, model, aurora_id):
    """Predict identity for a specific aurora_id."""
    # Look up display names
    c = conn.cursor()
    c.execute("""
        SELECT GROUP_CONCAT(DISTINCT p.player_name),
               GROUP_CONCAT(DISTINCT p.race),
               COUNT(DISTINCT p.replay_id)
        FROM players p
        JOIN replays r ON r.id = p.replay_id
        WHERE p.aurora_id = ? AND p.is_human = 1 AND r.game_date >= ?
    """, (aurora_id, MIN_DATE))
    row = c.fetchone()
    if not row or not row[0]:
        print(f"No games found for aurora_id {aurora_id}")
        return

    names, races, game_count = row
    print(f"\nPlayer: {names} [aurora:{aurora_id}] ({races}, {game_count} games)")

    samples = extract_player_samples_by_aurora(
        conn, [aurora_id], label=None, min_date=MIN_DATE)
    result = predict_samples(samples, model)

    if not result:
        print(f"Not enough valid samples (need 3, got {len(samples)})")
        return

    print_single_result(result)


def predict_by_name(conn, model, name):
    """Predict identity for a specific player name."""
    # Check if this name has an aurora_id we should use instead
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT aurora_id FROM players
        WHERE player_name = ? AND aurora_id IS NOT NULL
    """, (name,))
    aurora_ids = [row[0] for row in c.fetchall()]

    if len(aurora_ids) == 1:
        print(f"(found aurora_id {aurora_ids[0]} for '{name}', using that)")
        predict_by_aurora_id(conn, model, aurora_ids[0])
        return

    if len(aurora_ids) > 1:
        print(f"Warning: '{name}' has multiple aurora_ids: {aurora_ids}")
        print(f"Use --aurora-id to specify which one.")
        return

    # No aurora_id, use name-based extraction
    c.execute("""
        SELECT GROUP_CONCAT(DISTINCT p.race), COUNT(DISTINCT p.replay_id)
        FROM players p
        JOIN replays r ON r.id = p.replay_id
        WHERE p.player_name = ? AND p.is_human = 1 AND r.game_date >= ?
    """, (name, MIN_DATE))
    row = c.fetchone()
    if not row or not row[1]:
        print(f"No games found for '{name}'")
        return

    races, game_count = row
    print(f"\nPlayer: {name} ({races}, {game_count} games)")

    samples = extract_player_samples(conn, name, min_date=MIN_DATE, max_games=200)
    result = predict_samples(samples, model)

    if not result:
        print(f"Not enough valid samples (need 3, got {len(samples)})")
        return

    print_single_result(result)


def print_single_result(result):
    """Print prediction result for a single player."""
    print(f"\n  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.0%} "
          f"({result['all_preds'][result['prediction']]}/{result['games_analyzed']} games)")
    print(f"  Avg probability: {result['avg_prob']:.1%}")

    if len(result['all_preds']) > 1:
        print(f"  All votes: {result['all_preds']}")

    # Top 5 by avg probability
    top5 = sorted(result['avg_probs'].items(), key=lambda x: -x[1])[:5]
    print(f"\n  Top 5 by avg probability:")
    for cls, prob in top5:
        print(f"    {cls:<15} {prob:.1%}")


# --- All-unlabeled prediction ---

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


def predict_all_unlabeled(conn, model):
    """Predict all unlabeled players and save results."""
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
        result = predict_samples(samples, model)
        if not result:
            continue

        display_name = names.split(",")[0] if names else f"aurora:{aurora_id}"
        all_predictions.append({
            "player": display_name,
            "aurora_id": aurora_id,
            "display_names": names.split(",") if names else [],
            "races": races,
            "total_games": game_count,
            **result,
        })

    # Name-based fallback predictions
    for player_name, game_count, races in name_players:
        if idx % 20 == 0:
            print(f"  Processing {idx+1}/{total_players}...")
            sys.stdout.flush()
        idx += 1

        samples = extract_player_samples(conn, player_name, min_date=MIN_DATE, max_games=20)
        result = predict_samples(samples, model)
        if not result:
            continue

        all_predictions.append({
            "player": player_name,
            "aurora_id": None,
            "display_names": [player_name],
            "races": races,
            "total_games": game_count,
            **result,
        })

    # Sort by confidence
    all_predictions.sort(key=lambda x: (-x["confidence"], -x["avg_prob"]))

    # Print results table
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

    # Save to JSON (strip avg_probs to keep file clean)
    save_preds = [{k: v for k, v in p.items() if k != "avg_probs"} for p in all_predictions]
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump({
            "model_trained_at": model["trained_at"],
            "model_accuracy": model["accuracy"],
            "predicted_at": datetime.now().isoformat(),
            "min_games": MIN_GAMES,
            "total_players": len(save_preds),
            "predictions": save_preds,
        }, f, indent=2)
    print(f"\nPredictions saved to: {PREDICTIONS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Predict player identity from replays")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--aurora-id", type=int, help="Predict for a specific aurora_id")
    group.add_argument("--name", type=str, help="Predict for a specific player name")
    args = parser.parse_args()

    model = load_model()
    conn = sqlite3.connect(DB_PATH)

    if args.aurora_id:
        predict_by_aurora_id(conn, model, args.aurora_id)
    elif args.name:
        predict_by_name(conn, model, args.name)
    else:
        predict_all_unlabeled(conn, model)

    conn.close()


if __name__ == "__main__":
    main()
