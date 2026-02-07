#!/usr/bin/env python3
"""
Race-neutral clustering - focus on features that don't depend on race.
"""

import json
import subprocess
from pathlib import Path
from collections import Counter
import statistics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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
    """
    Extract ONLY features that are race-independent.
    Focus on: timing, hotkeys, click patterns, rhythm.
    """
    if len(commands) < 100:
        return None

    game_minutes = (game_frames * FRAME_MS) / 1000 / 60
    features = {}

    # === 1. TIMING PATTERNS (race-neutral) ===
    frames = [c["Frame"] for c in commands]
    gaps = [frames[i] - frames[i-1] for i in range(1, len(frames))]

    if gaps:
        gaps_ms = [g * FRAME_MS for g in gaps]
        features["gap_mean"] = statistics.mean(gaps_ms)
        features["gap_std"] = statistics.stdev(gaps_ms) if len(gaps_ms) > 1 else 0
        features["gap_median"] = statistics.median(gaps_ms)

        # Rhythm patterns
        features["rapid_ratio"] = sum(1 for g in gaps_ms if g < 100) / len(gaps_ms)      # Very fast
        features["moderate_ratio"] = sum(1 for g in gaps_ms if 100 <= g < 300) / len(gaps_ms)  # Normal
        features["slow_ratio"] = sum(1 for g in gaps_ms if g >= 500) / len(gaps_ms)      # Pauses

        # Burstiness: ratio of std to mean (higher = more bursty)
        features["burstiness"] = features["gap_std"] / features["gap_mean"] if features["gap_mean"] > 0 else 0

    # === 2. HOTKEY PATTERNS (race-neutral) ===
    hotkey_cmds = [c for c in commands if c["Type"]["Name"] == "Hotkey"]

    if hotkey_cmds:
        # Which groups they use (0-9)
        groups = [c.get("Group", 0) for c in hotkey_cmds]
        group_counts = Counter(groups)

        # Hotkey diversity: how many different groups do they use?
        features["hotkey_diversity"] = len(group_counts)

        # Concentration: what % of hotkeys are their top 2 groups?
        top_2 = sum(c for _, c in group_counts.most_common(2))
        features["hotkey_concentration"] = top_2 / len(hotkey_cmds)

        # Assign vs Select ratio
        assigns = sum(1 for c in hotkey_cmds if c.get("HotkeyType", {}).get("Name") == "Assign")
        features["hotkey_assign_ratio"] = assigns / len(hotkey_cmds)

        # Hotkey as % of total actions
        features["hotkey_action_ratio"] = len(hotkey_cmds) / len(commands)

        # Which group they use most (encoded as a feature)
        most_common_group = group_counts.most_common(1)[0][0] if group_counts else 0
        features["primary_hotkey_group"] = most_common_group

    # === 3. CLICK/MOVEMENT PATTERNS (race-neutral) ===
    clicks = [(c["Pos"]["X"], c["Pos"]["Y"]) for c in commands if "Pos" in c]

    if len(clicks) > 10:
        # Distance between consecutive clicks
        distances = []
        for i in range(1, len(clicks)):
            dx = clicks[i][0] - clicks[i-1][0]
            dy = clicks[i][1] - clicks[i-1][1]
            distances.append((dx**2 + dy**2) ** 0.5)

        features["click_dist_mean"] = statistics.mean(distances)
        features["click_dist_std"] = statistics.stdev(distances) if len(distances) > 1 else 0
        features["click_dist_median"] = statistics.median(distances)

        # Small movements vs big jumps
        features["small_move_ratio"] = sum(1 for d in distances if d < 100) / len(distances)
        features["big_jump_ratio"] = sum(1 for d in distances if d > 1000) / len(distances)

    # === 4. EARLY GAME (first 2 min - before race matters much) ===
    early_cutoff = int(2 * FRAMES_PER_MINUTE)
    early_cmds = [c for c in commands if c["Frame"] < early_cutoff]

    if len(early_cmds) > 20:
        features["early_apm"] = len(early_cmds) / 2.0

        early_gaps = [(early_cmds[i]["Frame"] - early_cmds[i-1]["Frame"]) * FRAME_MS
                      for i in range(1, len(early_cmds))]
        if early_gaps:
            features["early_gap_mean"] = statistics.mean(early_gaps)
            features["early_rapid_ratio"] = sum(1 for g in early_gaps if g < 100) / len(early_gaps)

        # First hotkey setup sequence
        early_hotkeys = [c for c in early_cmds if c["Type"]["Name"] == "Hotkey"]
        early_assigns = [c.get("Group", 0) for c in early_hotkeys
                        if c.get("HotkeyType", {}).get("Name") == "Assign"]

        # Encode first 3 hotkey assigns as features
        for i in range(3):
            features[f"first_assign_{i}"] = early_assigns[i] if i < len(early_assigns) else -1

    # === 5. QUEUED ACTIONS (race-neutral) ===
    queued = sum(1 for c in commands if c.get("Queued", False))
    features["queued_ratio"] = queued / len(commands)

    # === 6. SELECTION PATTERNS (somewhat race-neutral) ===
    selections = [c for c in commands if c["Type"]["Name"] == "Select"]
    if selections:
        sizes = [len(c.get("UnitTags", [])) for c in selections if "UnitTags" in c]
        if sizes:
            features["select_size_mean"] = statistics.mean(sizes)
            features["select_size_std"] = statistics.stdev(sizes) if len(sizes) > 1 else 0

        # Reselection ratio: selecting same units repeatedly
        features["selection_action_ratio"] = len(selections) / len(commands)

    # === 7. APM PATTERNS (race-neutral) ===
    features["apm"] = len(commands) / game_minutes

    # APM over time - capture the shape
    frames_per_min = int(FRAMES_PER_MINUTE)
    apm_buckets = Counter(c["Frame"] // frames_per_min for c in commands)
    apm_curve = [apm_buckets.get(i, 0) for i in range(10)]  # First 10 minutes

    if len(apm_curve) >= 5:
        # Early vs late game APM
        early_apm_avg = statistics.mean(apm_curve[:3]) if apm_curve[:3] else 0
        late_apm_avg = statistics.mean(apm_curve[5:8]) if len(apm_curve) > 5 else 0
        features["apm_decay"] = (early_apm_avg - late_apm_avg) / early_apm_avg if early_apm_avg > 0 else 0

        # APM variance (consistency)
        features["apm_variance"] = statistics.stdev(apm_curve[:8]) if len(apm_curve) >= 8 else 0

    # === 8. COMMAND TYPE RATIOS (race-neutral ones only) ===
    cmd_types = Counter(c["Type"]["Name"] for c in commands)
    total = sum(cmd_types.values())

    # These are race-neutral: everyone uses hotkeys, right-clicks, selects
    features["pct_hotkey"] = cmd_types.get("Hotkey", 0) / total
    features["pct_right_click"] = cmd_types.get("Right Click", 0) / total
    features["pct_select"] = cmd_types.get("Select", 0) / total
    features["pct_targeted_order"] = cmd_types.get("Targeted Order", 0) / total

    # Ratio of "thinking" (select/hotkey) vs "doing" (right click/orders)
    thinking = cmd_types.get("Hotkey", 0) + cmd_types.get("Select", 0)
    doing = cmd_types.get("Right Click", 0) + cmd_types.get("Targeted Order", 0)
    features["think_do_ratio"] = thinking / doing if doing > 0 else 0

    return features


def load_all_replays(replay_dir: Path):
    """Load all replays and extract features."""
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

    for rep_file in sorted(replay_dir.glob("*.rep")):
        try:
            data = parse_replay(rep_file)
            commands = data.get("Commands", {}).get("Cmds", [])
            game_frames = data["Header"]["Frames"]

            for player in data["Header"]["Players"]:
                if player["Type"]["Name"] != "Human":
                    continue

                name = player["Name"]
                player_cmds = [c for c in commands if c["PlayerID"] == player["ID"]]
                features = extract_race_neutral_features(player_cmds, game_frames)

                if features and features["apm"] > 100:
                    known_name = alias_map.get(name)
                    samples.append({
                        "features": features,
                        "name": name,
                        "known": known_name,
                        "race": player["Race"]["Name"],
                        "replay": rep_file.name,
                    })
        except Exception as e:
            pass

    return samples


def create_matrix(samples):
    all_features = set()
    for s in samples:
        all_features.update(s["features"].keys())
    feature_names = sorted(all_features)
    X = np.array([[s["features"].get(f, 0) for f in feature_names] for s in samples])
    return X, feature_names


def main():
    replay_dir = Path("/home/campesino/workspace/gosu-unveiled/data/poc_2008_2023/replays")

    print("Loading replays with RACE-NEUTRAL features...")
    samples = load_all_replays(replay_dir)
    print(f"Loaded {len(samples)} player-games")

    X, feature_names = create_matrix(samples)
    print(f"Feature matrix: {X.shape}")
    print(f"Features used: {len(feature_names)}")
    print(f"  {feature_names[:10]}...")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color by known player
    known_players = list(set(s["known"] for s in samples if s["known"]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(known_players)))
    player_colors = {p: colors[i] for i, p in enumerate(known_players)}

    # Left plot: colored by PLAYER (what we want)
    ax = axes[0]
    for i, sample in enumerate(samples):
        x, y = X_2d[i]
        known = sample["known"]
        if known:
            color = player_colors[known]
            ax.scatter(x, y, c=[color], s=150, alpha=0.8, edgecolors='black', linewidth=1)
            ax.annotate(known, (x, y), fontsize=7)
        else:
            ax.scatter(x, y, c='lightgray', marker='x', s=30, alpha=0.3)

    for player, color in player_colors.items():
        ax.scatter([], [], c=[color], label=player, s=100)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title("Race-Neutral Features\nColored by PLAYER")
    ax.grid(True, alpha=0.3)

    # Right plot: colored by RACE (should be mixed if race-neutral works)
    ax = axes[1]
    race_colors = {'Zerg': 'purple', 'Terran': 'blue', 'Protoss': 'gold'}
    for i, sample in enumerate(samples):
        x, y = X_2d[i]
        color = race_colors.get(sample["race"], 'gray')
        ax.scatter(x, y, c=color, s=50, alpha=0.5)

    for race, color in race_colors.items():
        ax.scatter([], [], c=color, label=race, s=100)
    ax.legend(loc='upper right')
    ax.set_title("Same Data\nColored by RACE (should be mixed)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/campesino/workspace/gosu-unveiled/clustering_race_neutral.png", dpi=150)
    print("Saved clustering_race_neutral.png")

    # Self-consistency check
    print("\n" + "="*60)
    print("SELF-CONSISTENCY (race-neutral features)")
    print("="*60)

    known_players_set = set(s["known"] for s in samples if s["known"])
    for player in sorted(known_players_set):
        indices = [i for i, s in enumerate(samples) if s["known"] == player]
        if len(indices) < 2:
            continue

        player_X = X_scaled[indices]
        intra_distances = []
        for i in range(len(player_X)):
            for j in range(i+1, len(player_X)):
                intra_distances.append(np.linalg.norm(player_X[i] - player_X[j]))

        avg_intra = np.mean(intra_distances)

        other_indices = [i for i, s in enumerate(samples) if s["known"] and s["known"] != player]
        if other_indices:
            player_center = player_X.mean(axis=0)
            inter_distances = [np.linalg.norm(X_scaled[idx] - player_center) for idx in other_indices]
            avg_inter = np.mean(inter_distances)

            ratio = avg_inter / avg_intra if avg_intra > 0 else 0
            races = set(samples[i]["race"] for i in indices)
            print(f"{player:12s} ({','.join(r[0] for r in races)}): intra={avg_intra:.2f}, inter={avg_inter:.2f}, ratio={ratio:.2f} {'✓' if ratio > 1.5 else ''}")

    # Compare: do same-race players still cluster?
    print("\n" + "="*60)
    print("RACE SEPARATION CHECK (should be LOW if race-neutral works)")
    print("="*60)

    for race in ["Zerg", "Terran", "Protoss"]:
        race_indices = [i for i, s in enumerate(samples) if s["race"] == race]
        other_indices = [i for i, s in enumerate(samples) if s["race"] != race]

        if race_indices and other_indices:
            race_center = X_scaled[race_indices].mean(axis=0)

            intra = [np.linalg.norm(X_scaled[i] - race_center) for i in race_indices]
            inter = [np.linalg.norm(X_scaled[i] - race_center) for i in other_indices]

            print(f"{race:8s}: intra={np.mean(intra):.2f}, inter={np.mean(inter):.2f}, ratio={np.mean(inter)/np.mean(intra):.2f}")


if __name__ == "__main__":
    main()
