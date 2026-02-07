#!/usr/bin/env python3
"""
Classifier using race-neutral features only.
"""

import json
import subprocess
from pathlib import Path
from collections import Counter
import statistics
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

SCREP_PATH = Path.home() / "go/bin/screp"
FRAME_MS = 42
FRAMES_PER_MINUTE = (1000 / FRAME_MS) * 60


def parse_replay(replay_path: Path) -> dict:
    result = subprocess.run(
        [str(SCREP_PATH), "-cmds", str(replay_path)],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)


def extract_race_neutral_features(commands: list, game_frames: int) -> dict:
    """Extract ONLY race-neutral features."""
    if len(commands) < 100:
        return None

    game_minutes = (game_frames * FRAME_MS) / 1000 / 60
    features = {}

    # === TIMING PATTERNS ===
    frames = [c["Frame"] for c in commands]
    gaps = [frames[i] - frames[i-1] for i in range(1, len(frames))]

    if gaps:
        gaps_ms = [g * FRAME_MS for g in gaps]
        features["gap_mean"] = statistics.mean(gaps_ms)
        features["gap_std"] = statistics.stdev(gaps_ms) if len(gaps_ms) > 1 else 0
        features["gap_median"] = statistics.median(gaps_ms)
        features["rapid_ratio"] = sum(1 for g in gaps_ms if g < 100) / len(gaps_ms)
        features["moderate_ratio"] = sum(1 for g in gaps_ms if 100 <= g < 300) / len(gaps_ms)
        features["slow_ratio"] = sum(1 for g in gaps_ms if g >= 500) / len(gaps_ms)
        features["burstiness"] = features["gap_std"] / features["gap_mean"] if features["gap_mean"] > 0 else 0

    # === HOTKEY PATTERNS ===
    hotkey_cmds = [c for c in commands if c["Type"]["Name"] == "Hotkey"]
    if hotkey_cmds:
        groups = [c.get("Group", 0) for c in hotkey_cmds]
        group_counts = Counter(groups)
        features["hotkey_diversity"] = len(group_counts)
        top_2 = sum(c for _, c in group_counts.most_common(2))
        features["hotkey_concentration"] = top_2 / len(hotkey_cmds)
        assigns = sum(1 for c in hotkey_cmds if c.get("HotkeyType", {}).get("Name") == "Assign")
        features["hotkey_assign_ratio"] = assigns / len(hotkey_cmds)
        features["hotkey_action_ratio"] = len(hotkey_cmds) / len(commands)
        most_common_group = group_counts.most_common(1)[0][0] if group_counts else 0
        features["primary_hotkey_group"] = most_common_group

    # === CLICK PATTERNS ===
    clicks = [(c["Pos"]["X"], c["Pos"]["Y"]) for c in commands if "Pos" in c]
    if len(clicks) > 10:
        distances = []
        for i in range(1, len(clicks)):
            dx = clicks[i][0] - clicks[i-1][0]
            dy = clicks[i][1] - clicks[i-1][1]
            distances.append((dx**2 + dy**2) ** 0.5)
        features["click_dist_mean"] = statistics.mean(distances)
        features["click_dist_std"] = statistics.stdev(distances) if len(distances) > 1 else 0
        features["click_dist_median"] = statistics.median(distances)
        features["small_move_ratio"] = sum(1 for d in distances if d < 100) / len(distances)
        features["big_jump_ratio"] = sum(1 for d in distances if d > 1000) / len(distances)

    # === EARLY GAME ===
    early_cutoff = int(2 * FRAMES_PER_MINUTE)
    early_cmds = [c for c in commands if c["Frame"] < early_cutoff]
    if len(early_cmds) > 20:
        features["early_apm"] = len(early_cmds) / 2.0
        early_gaps = [(early_cmds[i]["Frame"] - early_cmds[i-1]["Frame"]) * FRAME_MS
                      for i in range(1, len(early_cmds))]
        if early_gaps:
            features["early_gap_mean"] = statistics.mean(early_gaps)
            features["early_rapid_ratio"] = sum(1 for g in early_gaps if g < 100) / len(early_gaps)

        early_hotkeys = [c for c in early_cmds if c["Type"]["Name"] == "Hotkey"]
        early_assigns = [c.get("Group", 0) for c in early_hotkeys
                        if c.get("HotkeyType", {}).get("Name") == "Assign"]
        for i in range(3):
            features[f"first_assign_{i}"] = early_assigns[i] if i < len(early_assigns) else -1

    # === OTHER RACE-NEUTRAL ===
    queued = sum(1 for c in commands if c.get("Queued", False))
    features["queued_ratio"] = queued / len(commands)

    selections = [c for c in commands if c["Type"]["Name"] == "Select"]
    if selections:
        sizes = [len(c.get("UnitTags", [])) for c in selections if "UnitTags" in c]
        if sizes:
            features["select_size_mean"] = statistics.mean(sizes)
            features["select_size_std"] = statistics.stdev(sizes) if len(sizes) > 1 else 0
        features["selection_action_ratio"] = len(selections) / len(commands)

    features["apm"] = len(commands) / game_minutes

    frames_per_min = int(FRAMES_PER_MINUTE)
    apm_buckets = Counter(c["Frame"] // frames_per_min for c in commands)
    apm_curve = [apm_buckets.get(i, 0) for i in range(10)]
    if len(apm_curve) >= 5:
        early_apm_avg = statistics.mean(apm_curve[:3]) if apm_curve[:3] else 0
        late_apm_avg = statistics.mean(apm_curve[5:8]) if len(apm_curve) > 5 else 0
        features["apm_decay"] = (early_apm_avg - late_apm_avg) / early_apm_avg if early_apm_avg > 0 else 0
        features["apm_variance"] = statistics.stdev(apm_curve[:8]) if len(apm_curve) >= 8 else 0

    cmd_types = Counter(c["Type"]["Name"] for c in commands)
    total = sum(cmd_types.values())
    features["pct_hotkey"] = cmd_types.get("Hotkey", 0) / total
    features["pct_right_click"] = cmd_types.get("Right Click", 0) / total
    features["pct_select"] = cmd_types.get("Select", 0) / total
    features["pct_targeted_order"] = cmd_types.get("Targeted Order", 0) / total

    thinking = cmd_types.get("Hotkey", 0) + cmd_types.get("Select", 0)
    doing = cmd_types.get("Right Click", 0) + cmd_types.get("Targeted Order", 0)
    features["think_do_ratio"] = thinking / doing if doing > 0 else 0

    return features


def build_dataset(replay_dir: Path):
    alias_map = {
        # Bisu
        "na1st": "Bisu", "fa1con": "Bisu", "Bisu": "Bisu", "Bisu[Shield]": "Bisu",
        # Flash
        "FlaSh": "Flash", "KTF FlaSh": "Flash",
        # Jaedong
        "HwaSeungOZ Jaedong": "Jaedong",
        # Soma
        "lllIlIlIIIl11": "Soma", "wwvvwwvvwvwww": "Soma",
        # Dewalt
        "weiguginsagi": "Dewalt", "weigugindagsin": "Dewalt",
        # Hiya
        "HwaSeungOZ HiyA": "Hiya",
        # Shuttle
        "Carariyo_jum2": "Shuttle",
        # Effort
        "effort": "Effort",
        # Hydra
        "by.hydra": "Hydra",
        # Movie
        "BY.MOVIE": "Movie", "mOvie": "Movie",
        # Zero
        "Zero": "Zero",
        # Sea
        "Sea[Shield]": "Sea",
        # Light
        "Light[aLive]": "Light", "mbclight": "Light",
        # Violet
        "Violet[Name]": "Violet",
        # Sonic
        "Sonic": "Sonic",
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
                features = extract_race_neutral_features(player_cmds, game_frames)

                if features:
                    samples.append({
                        "features": features,
                        "label": real_name,
                        "alias": name,
                        "race": player["Race"]["Name"],
                    })
        except:
            pass

    return samples


def create_feature_matrix(samples):
    all_features = set()
    for s in samples:
        all_features.update(s["features"].keys())
    feature_names = sorted(all_features)

    X = np.array([[s["features"].get(f, 0) for f in feature_names] for s in samples])
    y = np.array([s["label"] for s in samples])
    return X, y, feature_names


def main():
    replay_dir = Path("/home/campesino/workspace/gosu-unveiled/data/poc_2008_2023/replays")

    print("Building dataset with RACE-NEUTRAL features...")
    samples = build_dataset(replay_dir)

    print(f"\nFound {len(samples)} player-games:")
    label_counts = Counter(s["label"] for s in samples)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} games")

    valid_labels = {label for label, count in label_counts.items() if count >= 2}
    samples = [s for s in samples if s["label"] in valid_labels]
    print(f"\nUsing {len(samples)} samples from players with 2+ games")

    X, y, feature_names = create_feature_matrix(samples)
    print(f"Feature matrix shape: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n" + "="*60)
    print("LEAVE-ONE-OUT CROSS-VALIDATION (Race-Neutral Features)")
    print("="*60)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    loo = LeaveOneOut()
    predictions = cross_val_predict(clf, X_scaled, y, cv=loo)

    print("\nPredictions:")
    print("-" * 60)

    correct = 0
    for sample, pred in zip(samples, predictions):
        actual = sample["label"]
        match = "✓" if pred == actual else "✗"
        if pred == actual:
            correct += 1
        print(f"  {sample['alias'][:20]:20s} ({sample['race'][0]}) → Pred: {pred:10s} Actual: {actual:10s} {match}")

    accuracy = correct / len(samples)
    print(f"\nOverall accuracy: {correct}/{len(samples)} = {accuracy:.1%}")

    # Compare with original
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"  Original features (with race): 75.0% (15/20)")
    print(f"  Race-neutral features:         {accuracy:.1%} ({correct}/{len(samples)})")

    # Confusion matrix
    print("\nConfusion Matrix:")
    labels = sorted(set(y))
    cm = confusion_matrix(y, predictions, labels=labels)
    print("              ", "  ".join(f"{l[:6]:>6s}" for l in labels))
    for i, label in enumerate(labels):
        row = "  ".join(f"{cm[i,j]:6d}" for j in range(len(labels)))
        print(f"  {label:10s}  {row}")

    # Feature importance
    clf.fit(X_scaled, y)
    importances = sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1])

    print("\nTop 15 most important features:")
    for name, imp in importances[:15]:
        bar = "█" * int(imp * 100)
        print(f"  {name:30s} {imp:.3f} {bar}")


if __name__ == "__main__":
    main()
