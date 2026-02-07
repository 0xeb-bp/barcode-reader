#!/usr/bin/env python3
"""
Simple classifier to identify SC:BW players from replay features.
"""

import json
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
import statistics

# We'll use scikit-learn for the classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

SCREP_PATH = Path.home() / "go/bin/screp"
FRAME_MS = 42
FRAMES_PER_MINUTE = (1000 / FRAME_MS) * 60


def parse_replay(replay_path: Path) -> dict:
    result = subprocess.run(
        [str(SCREP_PATH), "-cmds", str(replay_path)],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)


def get_command_type(cmd: dict) -> str:
    """Get simple command type."""
    return cmd["Type"]["Name"]


def get_detailed_command(cmd: dict) -> str:
    """Get detailed command signature."""
    cmd_type = cmd["Type"]["Name"]
    if cmd_type == "Hotkey":
        hotkey_type = cmd.get("HotkeyType", {}).get("Name", "?")
        group = cmd.get("Group", "?")
        return f"HK_{hotkey_type}_{group}"
    return cmd_type


def extract_ngrams(commands: list, n: int) -> Counter:
    """Extract n-grams from command sequence."""
    types = [get_command_type(c) for c in commands]
    ngrams = Counter()
    for i in range(len(types) - n + 1):
        gram = "_".join(types[i:i+n])
        ngrams[gram] += 1
    return ngrams


def extract_all_features(commands: list, game_frames: int) -> dict:
    """Extract all features for a player's commands in a game."""

    if len(commands) < 100:
        return None

    game_minutes = (game_frames * FRAME_MS) / 1000 / 60

    # === Basic stats ===
    features = {
        "apm": len(commands) / game_minutes,
    }

    # === Command type distribution ===
    cmd_types = Counter(get_command_type(c) for c in commands)
    total = sum(cmd_types.values())

    for cmd in ["Hotkey", "Right Click", "Select", "Targeted Order", "Train", "Unit Morph"]:
        features[f"pct_{cmd.replace(' ', '_')}"] = cmd_types.get(cmd, 0) / total

    # === Timing features ===
    frames = [c["Frame"] for c in commands]
    gaps = [frames[i] - frames[i-1] for i in range(1, len(frames))]

    if gaps:
        features["gap_mean_ms"] = statistics.mean(gaps) * FRAME_MS
        features["gap_std_ms"] = statistics.stdev(gaps) * FRAME_MS if len(gaps) > 1 else 0
        features["rapid_ratio"] = sum(1 for g in gaps if g * FRAME_MS < 100) / len(gaps)
        features["slow_ratio"] = sum(1 for g in gaps if g * FRAME_MS > 500) / len(gaps)

    # === Hotkey patterns ===
    hotkey_cmds = [c for c in commands if c["Type"]["Name"] == "Hotkey"]
    if hotkey_cmds:
        # Which groups used
        groups_used = Counter(c.get("Group", 0) for c in hotkey_cmds)
        for g in range(1, 6):  # Groups 1-5
            features[f"hotkey_group_{g}_pct"] = groups_used.get(g, 0) / len(hotkey_cmds)

        # Assign vs Select ratio
        assigns = sum(1 for c in hotkey_cmds if c.get("HotkeyType", {}).get("Name") == "Assign")
        features["hotkey_assign_ratio"] = assigns / len(hotkey_cmds)

    # === Early game features (first 2 minutes) ===
    early_cutoff = int(2 * FRAMES_PER_MINUTE)
    early_cmds = [c for c in commands if c["Frame"] < early_cutoff]

    if len(early_cmds) > 20:
        features["early_apm"] = len(early_cmds) / 2.0

        early_gaps = [early_cmds[i]["Frame"] - early_cmds[i-1]["Frame"]
                      for i in range(1, len(early_cmds))]
        if early_gaps:
            features["early_rapid_ratio"] = sum(1 for g in early_gaps if g * FRAME_MS < 100) / len(early_gaps)
            features["early_gap_mean"] = statistics.mean(early_gaps) * FRAME_MS

        # First hotkey assigns
        early_hotkeys = [c for c in early_cmds if c["Type"]["Name"] == "Hotkey"]
        assigns = [c.get("Group", 0) for c in early_hotkeys
                   if c.get("HotkeyType", {}).get("Name") == "Assign"]

        # Encode first 3 assigns as features
        for i, g in enumerate(assigns[:3]):
            features[f"first_assign_{i}"] = g

    # === N-gram features ===
    # Use top n-grams as features
    for n in [2, 3, 4]:
        ngrams = extract_ngrams(commands, n)
        total_ngrams = sum(ngrams.values())

        # Get top 10 most common n-grams
        for gram, count in ngrams.most_common(20):
            features[f"ngram{n}_{gram}"] = count / total_ngrams

    # === Click position features ===
    clicks = [(c["Pos"]["X"], c["Pos"]["Y"]) for c in commands if "Pos" in c]
    if len(clicks) > 10:
        # Average click distance between consecutive clicks
        dists = []
        for i in range(1, len(clicks)):
            dx = clicks[i][0] - clicks[i-1][0]
            dy = clicks[i][1] - clicks[i-1][1]
            dists.append((dx**2 + dy**2) ** 0.5)

        features["click_dist_mean"] = statistics.mean(dists)
        features["click_dist_std"] = statistics.stdev(dists) if len(dists) > 1 else 0

    # === Selection patterns ===
    selections = [c for c in commands if c["Type"]["Name"] == "Select"]
    if selections:
        sizes = [len(c.get("UnitTags", [])) for c in selections if "UnitTags" in c]
        if sizes:
            features["select_size_mean"] = statistics.mean(sizes)

    # === Queued actions ===
    queued = sum(1 for c in commands if c.get("Queued", False))
    features["queued_ratio"] = queued / len(commands)

    return features


def build_dataset(replay_dir: Path) -> tuple:
    """Build feature matrix and labels from replays."""

    # Map aliases to real names
    alias_map = {
        "na1st": "Bisu", "fa1con": "Bisu",
        "FlaSh": "Flash",
        "HwaSeungOZ Jaedong": "Jaedong",
        "lllIlIlIIIl11": "Soma", "wwvvwwvvwvwww": "Soma",
        "weiguginsagi": "Dewalt", "weigugindagsin": "Dewalt",
        "TT1": "TT1",
        "HwaSeungOZ HiyA": "Hiya",
        "Sacsri": "Sacsri",
        "By.SnOw1": "Snow",
        "LC_Tyson": "Tyson",
        "thecute": "Pretty",
        "Carariyo_jum2": "Shuttle",
    }

    samples = []

    for rep_file in replay_dir.glob("*.rep"):
        try:
            data = parse_replay(rep_file)
            commands = data.get("Commands", {}).get("Cmds", [])
            game_frames = data["Header"]["Frames"]

            for player in data["Header"]["Players"]:
                if player["Type"]["Name"] != "Human":
                    continue

                name = player["Name"]
                real_name = alias_map.get(name)

                if not real_name:
                    continue

                player_cmds = [c for c in commands if c["PlayerID"] == player["ID"]]
                features = extract_all_features(player_cmds, game_frames)

                if features:
                    samples.append({
                        "features": features,
                        "label": real_name,
                        "alias": name,
                        "replay": rep_file.name,
                        "race": player["Race"]["Name"],
                    })

        except Exception as e:
            print(f"Error with {rep_file}: {e}")

    return samples


def create_feature_matrix(samples: list) -> tuple:
    """Convert samples to numpy arrays for sklearn."""

    # Get all feature names from all samples
    all_features = set()
    for s in samples:
        all_features.update(s["features"].keys())

    feature_names = sorted(all_features)

    # Build matrix
    X = []
    y = []

    for s in samples:
        row = [s["features"].get(f, 0) for f in feature_names]
        X.append(row)
        y.append(s["label"])

    return np.array(X), np.array(y), feature_names


def main():
    replay_dir = Path("/home/campesino/workspace/gosu-unveiled/data/poc_2008_2023/replays")

    print("Building dataset...")
    samples = build_dataset(replay_dir)

    print(f"\nFound {len(samples)} player-games:")
    label_counts = Counter(s["label"] for s in samples)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} games")

    # Only use players with 2+ games for meaningful evaluation
    valid_labels = {label for label, count in label_counts.items() if count >= 2}
    samples = [s for s in samples if s["label"] in valid_labels]

    print(f"\nUsing {len(samples)} samples from players with 2+ games")

    X, y, feature_names = create_feature_matrix(samples)
    print(f"Feature matrix shape: {X.shape}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use Leave-One-Out cross-validation (since we have small data)
    # This trains on all-but-one sample, predicts the left-out one, repeats
    print("\n" + "="*60)
    print("LEAVE-ONE-OUT CROSS-VALIDATION")
    print("(Train on all but one game, predict the left-out game)")
    print("="*60)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)

    loo = LeaveOneOut()
    predictions = cross_val_predict(clf, X_scaled, y, cv=loo)

    # Show results
    print("\nPredictions:")
    print("-" * 60)

    correct = 0
    for i, (sample, pred) in enumerate(zip(samples, predictions)):
        actual = sample["label"]
        match = "✓" if pred == actual else "✗"
        if pred == actual:
            correct += 1
        print(f"  {sample['alias'][:20]:20s} → Predicted: {pred:10s} Actual: {actual:10s} {match}")

    accuracy = correct / len(samples)
    print(f"\nOverall accuracy: {correct}/{len(samples)} = {accuracy:.1%}")

    # Show confusion matrix
    print("\nConfusion Matrix:")
    labels = sorted(set(y))
    cm = confusion_matrix(y, predictions, labels=labels)

    print("              ", "  ".join(f"{l[:6]:>6s}" for l in labels))
    for i, label in enumerate(labels):
        row = "  ".join(f"{cm[i,j]:6d}" for j in range(len(labels)))
        print(f"  {label:10s}  {row}")

    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (what the model found useful)")
    print("="*60)

    # Train on full data to get feature importances
    clf.fit(X_scaled, y)
    importances = list(zip(feature_names, clf.feature_importances_))
    importances.sort(key=lambda x: -x[1])

    print("\nTop 20 most important features:")
    for name, importance in importances[:20]:
        bar = "█" * int(importance * 100)
        print(f"  {name:40s} {importance:.3f} {bar}")


if __name__ == "__main__":
    main()
