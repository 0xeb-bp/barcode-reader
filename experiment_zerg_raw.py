#!/usr/bin/env python3
"""
Experiment 16: Zerg-Only Raw N-gram Ablation Study

Compares abstracted vs raw command-type n-grams for Zerg-only classification.
Runs two LOO CV passes on the same data and prints a comparison table.

Usage:
  python experiment_zerg_raw.py              # Full run (all games, ~1-2 hours)
  python experiment_zerg_raw.py --max-games 30  # Fast iteration (~5 min)
"""

import argparse
import copy
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

from features import (
    DB_PATH, GLOBAL_NGRAM_TOP_N,
    parse_replay, trim_at_leave, extract_features,
    get_pro_identities, get_player_replays_by_aurora,
    select_global_ngrams, apply_ngram_features, create_feature_matrix,
)

MIN_DATE = "2025-01-01"

# Non-gameplay commands to exclude from raw n-grams
IGNORED_CMDS = {"Chat", "Leave Game", "Alliance", "Vision"}


def extract_raw_ngrams(commands, n):
    """Extract n-grams using raw command type names (no abstraction, no Prod collapse)."""
    types = [c["Type"]["Name"] for c in commands if c["Type"]["Name"] not in IGNORED_CMDS]
    ngrams = Counter()
    for i in range(len(types) - n + 1):
        gram = "_".join(types[i:i + n])
        ngrams[gram] += 1
    return ngrams


def extract_samples_dual(conn, aurora_ids, label, min_date=None):
    """Extract samples with both abstracted and raw n-grams stored per sample."""
    replays = get_player_replays_by_aurora(conn, aurora_ids, min_date=min_date)
    samples = []

    for file_path, replay_id, player_name in replays:
        path = Path(file_path)
        if not path.exists():
            continue

        try:
            data = parse_replay(path)
            all_cmds = data.get("Commands", {}).get("Cmds", [])
            game_frames = data["Header"]["Frames"]

            for player in data["Header"]["Players"]:
                if player["Type"]["Name"] != "Human":
                    continue
                if player["Name"] != player_name:
                    continue

                player_cmds = [c for c in all_cmds if c["PlayerID"] == player["ID"]]
                player_cmds, effective_frames = trim_at_leave(player_cmds, all_cmds, game_frames)

                # Standard extraction (includes abstracted n-grams in raw_ngrams)
                features, raw_ngrams = extract_features(player_cmds, effective_frames)

                if features:
                    # Also extract raw n-grams and store under rng2/rng3/rng4
                    for n in [2, 3, 4]:
                        ngrams = extract_raw_ngrams(player_cmds, n)
                        total_ng = sum(ngrams.values())
                        if total_ng > 0:
                            raw_ngrams[f"rng{n}"] = (ngrams, total_ng)

                    samples.append({
                        "features": features,
                        "raw_ngrams": raw_ngrams,
                        "label": label,
                        "alias": player_name,
                        "race": player["Race"]["Name"],
                        "replay_id": replay_id,
                        "file": path.name,
                    })
        except Exception:
            pass

    return samples


def run_loo_cv(samples, ngram_prefixes, label):
    """Run LOO CV on samples using the specified n-gram prefix keys. Returns results dict."""
    samples = copy.deepcopy(samples)

    # Filter raw_ngrams to only the requested prefixes (+ always include hotkey n-grams)
    hotkey_prefixes = {"hkg2", "hkg3", "ehkg2", "ehkg3"}
    active_prefixes = set(ngram_prefixes) | hotkey_prefixes

    filtered_ngrams = []
    for s in samples:
        filtered = {k: v for k, v in s["raw_ngrams"].items() if k in active_prefixes}
        filtered_ngrams.append(filtered)

    # Two-pass global n-gram selection
    global_ngrams = select_global_ngrams(filtered_ngrams)

    # Apply n-gram features
    for s, fn in zip(samples, filtered_ngrams):
        apply_ngram_features(s["features"], fn, global_ngrams)

    # Build feature matrix
    X, y, feature_names = create_feature_matrix(samples)

    ngram_count = sum(len(v) for v in global_ngrams.values())
    print(f"\n{'=' * 70}")
    print(f"RUN {label}")
    print(f"{'=' * 70}")
    print(f"  N-gram features: {ngram_count} (from {sorted(global_ngrams.keys())})")
    print(f"  Total features: {X.shape[1]}")
    sys.stdout.flush()

    # LOO CV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42,
                                 class_weight="balanced", n_jobs=-1)
    loo = LeaveOneOut()
    predictions = cross_val_predict(clf, X_scaled, y, cv=loo)

    correct = sum(1 for p, a in zip(predictions, y) if p == a)
    accuracy = correct / len(y)

    print(f"  LOO CV accuracy: {correct}/{len(y)} = {accuracy:.1%}")

    # Per-player accuracy
    player_results = {}
    for s, pred in zip(samples, predictions):
        name = s["label"]
        if name not in player_results:
            player_results[name] = {"correct": 0, "total": 0}
        player_results[name]["total"] += 1
        if pred == name:
            player_results[name]["correct"] += 1

    print(f"\n  Per-player accuracy:")
    for name in sorted(player_results, key=lambda n: player_results[n]["correct"] / player_results[n]["total"]):
        r = player_results[name]
        acc = r["correct"] / r["total"]
        bar = "#" * int(acc * 20)
        print(f"    {name:<15} {r['correct']:>4}/{r['total']:<4} = {acc:>6.1%} {bar}")

    # Misclassifications
    misclassified = [(s, pred) for s, pred in zip(samples, predictions) if pred != s["label"]]
    if misclassified:
        print(f"\n  Misclassifications ({len(misclassified)}):")
        for s, pred in misclassified:
            print(f"    {s['alias']:<25} → {pred:<15} (true: {s['label']}) [{s['file']}]")

    # Train final model for feature importances
    clf.fit(X_scaled, y)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print(f"\n  Top 15 features:")
    for i in top_idx:
        print(f"    {feature_names[i]:<35} {importances[i]:.3f}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(y),
        "features": X.shape[1],
        "ngram_count": ngram_count,
        "player_results": player_results,
        "misclassified_count": len(misclassified),
    }


def main():
    parser = argparse.ArgumentParser(description="Zerg-only raw n-gram ablation study")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Max Zerg games per player (default: unlimited)")
    parser.add_argument("--min-games", type=int, default=100,
                        help="Minimum Zerg games to include a player (default: 100)")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 16: ZERG-ONLY RAW N-GRAM ABLATION STUDY")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    pros = get_pro_identities(conn, min_games=20)

    # Extract samples with dual n-grams
    print(f"\nExtracting features (dual n-grams: abstracted + raw)...")
    sys.stdout.flush()
    all_samples = []

    for canonical, aurora_ids, total in pros:
        player_samples = extract_samples_dual(
            conn, aurora_ids, label=canonical, min_date=MIN_DATE)

        # Keep only Zerg games
        zerg_games = [s for s in player_samples if s["race"] == "Zerg"]

        if len(zerg_games) < args.min_games:
            continue

        # Cap if requested
        if args.max_games and len(zerg_games) > args.max_games:
            zerg_games = zerg_games[:args.max_games]

        all_samples.extend(zerg_games)
        aliases = set(s["alias"] for s in zerg_games)
        print(f"  {canonical}: {len(zerg_games)} Zerg games (handles: {aliases})")
        sys.stdout.flush()

    conn.close()

    print(f"\nTotal Zerg samples: {len(all_samples)}")
    players = sorted(set(s["label"] for s in all_samples))
    print(f"Players: {len(players)} ({', '.join(players)})")

    # Run A: Abstracted n-grams (baseline)
    result_a = run_loo_cv(all_samples, ["ng2", "ng3", "ng4"], "A: ABSTRACTED N-GRAMS (baseline)")

    # Run B: Raw n-grams (same budget)
    result_b = run_loo_cv(all_samples, ["rng2", "rng3", "rng4"], "B: RAW N-GRAMS (experimental)")

    # Comparison table
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'':25} {'Abstracted':>12} {'Raw':>12} {'Delta':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'Overall accuracy':25} {result_a['accuracy']:>11.1%} {result_b['accuracy']:>11.1%} {result_b['accuracy'] - result_a['accuracy']:>+9.1%}")
    print(f"  {'Total features':25} {result_a['features']:>12} {result_b['features']:>12} {result_b['features'] - result_a['features']:>+10}")
    print(f"  {'N-gram features':25} {result_a['ngram_count']:>12} {result_b['ngram_count']:>12} {result_b['ngram_count'] - result_a['ngram_count']:>+10}")
    print(f"  {'Misclassifications':25} {result_a['misclassified_count']:>12} {result_b['misclassified_count']:>12} {result_b['misclassified_count'] - result_a['misclassified_count']:>+10}")

    print(f"\n  Per-player:")
    print(f"  {'Player':<15} {'Abstracted':>12} {'Raw':>12} {'Delta':>10}")
    print(f"  {'-' * 50}")
    for name in players:
        ra = result_a["player_results"][name]
        rb = result_b["player_results"][name]
        acc_a = ra["correct"] / ra["total"]
        acc_b = rb["correct"] / rb["total"]
        delta = acc_b - acc_a
        marker = " ***" if abs(delta) > 0.01 else ""
        print(f"  {name:<15} {acc_a:>11.1%} {acc_b:>11.1%} {delta:>+9.1%}{marker}")


if __name__ == "__main__":
    main()
