#!/usr/bin/env python3
"""
Cluster SC:BW players by their gameplay fingerprint.
No labels needed - see if players naturally group together.
"""

import json
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
import statistics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
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


def extract_ngrams(commands: list, n: int) -> Counter:
    types = [c["Type"]["Name"] for c in commands]
    ngrams = Counter()
    for i in range(len(types) - n + 1):
        gram = "_".join(types[i:i+n])
        ngrams[gram] += 1
    return ngrams


def extract_features(commands: list, game_frames: int) -> dict:
    """Extract features for clustering."""
    if len(commands) < 100:
        return None

    game_minutes = (game_frames * FRAME_MS) / 1000 / 60
    features = {"apm": len(commands) / game_minutes}

    # Command distribution
    cmd_types = Counter(c["Type"]["Name"] for c in commands)
    total = sum(cmd_types.values())
    for cmd in ["Hotkey", "Right Click", "Select", "Targeted Order", "Train", "Unit Morph"]:
        features[f"pct_{cmd.replace(' ', '_')}"] = cmd_types.get(cmd, 0) / total

    # Timing
    frames = [c["Frame"] for c in commands]
    gaps = [frames[i] - frames[i-1] for i in range(1, len(frames))]
    if gaps:
        features["gap_mean_ms"] = statistics.mean(gaps) * FRAME_MS
        features["gap_std_ms"] = statistics.stdev(gaps) * FRAME_MS if len(gaps) > 1 else 0
        features["rapid_ratio"] = sum(1 for g in gaps if g * FRAME_MS < 100) / len(gaps)

    # Hotkey patterns
    hotkey_cmds = [c for c in commands if c["Type"]["Name"] == "Hotkey"]
    if hotkey_cmds:
        groups = Counter(c.get("Group", 0) for c in hotkey_cmds)
        for g in range(1, 6):
            features[f"hotkey_group_{g}_pct"] = groups.get(g, 0) / len(hotkey_cmds)

    # Early game
    early_cutoff = int(2 * FRAMES_PER_MINUTE)
    early_cmds = [c for c in commands if c["Frame"] < early_cutoff]
    if len(early_cmds) > 20:
        features["early_apm"] = len(early_cmds) / 2.0
        early_gaps = [early_cmds[i]["Frame"] - early_cmds[i-1]["Frame"] for i in range(1, len(early_cmds))]
        if early_gaps:
            features["early_rapid_ratio"] = sum(1 for g in early_gaps if g * FRAME_MS < 100) / len(early_gaps)

    # N-grams
    for n in [2, 3]:
        ngrams = extract_ngrams(commands, n)
        total_ng = sum(ngrams.values())
        for gram, count in ngrams.most_common(15):
            features[f"ng{n}_{gram}"] = count / total_ng

    # Queued ratio
    queued = sum(1 for c in commands if c.get("Queued", False))
    features["queued_ratio"] = queued / len(commands)

    return features


def load_all_replays(replay_dir: Path):
    """Load all replays and extract features."""
    # Known aliases for labeling (for visualization only)
    alias_map = {
        "na1st": "Bisu", "fa1con": "Bisu",
        "FlaSh": "Flash",
        "HwaSeungOZ Jaedong": "Jaedong",
        "lllIlIlIIIl11": "Soma", "wwvvwwvvwvwww": "Soma",
        "weiguginsagi": "Dewalt", "weigugindagsin": "Dewalt",
        "HwaSeungOZ HiyA": "Hiya",
        "Carariyo_jum2": "Shuttle",
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
                features = extract_features(player_cmds, game_frames)

                if features and features["apm"] > 100:  # Filter observers
                    known_name = alias_map.get(name)
                    samples.append({
                        "features": features,
                        "name": name,
                        "known": known_name,  # None if unknown
                        "race": player["Race"]["Name"],
                        "replay": rep_file.name,
                    })
        except Exception as e:
            pass  # Skip problematic replays

    return samples


def create_matrix(samples):
    """Convert to numpy matrix."""
    all_features = set()
    for s in samples:
        all_features.update(s["features"].keys())
    feature_names = sorted(all_features)

    X = np.array([[s["features"].get(f, 0) for f in feature_names] for s in samples])
    return X, feature_names


def plot_clusters(X_2d, samples, title, filename):
    """Plot 2D projection with player labels."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color by known player (if known) or gray if unknown
    known_players = list(set(s["known"] for s in samples if s["known"]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(known_players)))
    player_colors = {p: colors[i] for i, p in enumerate(known_players)}

    for i, sample in enumerate(samples):
        x, y = X_2d[i]
        known = sample["known"]
        race = sample["race"][0]  # T, Z, or P

        if known:
            color = player_colors[known]
            marker = {'T': 's', 'Z': '^', 'P': 'o'}.get(race, 'o')
            ax.scatter(x, y, c=[color], marker=marker, s=150, alpha=0.8, edgecolors='black', linewidth=1)
            ax.annotate(known, (x, y), fontsize=8, ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')
        else:
            ax.scatter(x, y, c='lightgray', marker='x', s=50, alpha=0.5)

    # Legend for known players
    for player, color in player_colors.items():
        ax.scatter([], [], c=[color], label=player, s=100)

    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"/home/campesino/workspace/gosu-unveiled/{filename}", dpi=150)
    print(f"Saved {filename}")


def analyze_clusters(X, samples, n_clusters=8):
    """Run K-means and analyze results."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    print(f"\n{'='*60}")
    print(f"K-MEANS CLUSTERING (k={n_clusters})")
    print('='*60)

    # For each cluster, show who's in it
    for cluster_id in range(n_clusters):
        members = [samples[i] for i in range(len(samples)) if labels[i] == cluster_id]
        if not members:
            continue

        print(f"\nCluster {cluster_id} ({len(members)} members):")

        # Count known players in this cluster
        known_counts = Counter(m["known"] for m in members if m["known"])
        unknown_count = sum(1 for m in members if not m["known"])

        if known_counts:
            for player, count in known_counts.most_common():
                print(f"  {player}: {count}")
        if unknown_count:
            print(f"  (unknown): {unknown_count}")

        # Show some unknown names
        unknowns = [m["name"] for m in members if not m["known"]][:5]
        if unknowns:
            print(f"  Unknown names: {', '.join(unknowns)}")

    return labels


def find_similar_players(X, samples, target_name):
    """Find players most similar to a target."""
    # Find target indices
    target_indices = [i for i, s in enumerate(samples) if s["known"] == target_name or s["name"] == target_name]

    if not target_indices:
        print(f"Player {target_name} not found")
        return

    # Average target embedding
    target_embedding = X[target_indices].mean(axis=0)

    # Calculate distances to all other players
    distances = []
    for i, sample in enumerate(samples):
        if i in target_indices:
            continue
        dist = np.linalg.norm(X[i] - target_embedding)
        distances.append((dist, sample))

    distances.sort(key=lambda x: x[0])

    print(f"\nPlayers most similar to {target_name}:")
    print("-" * 40)
    for dist, sample in distances[:10]:
        known = sample["known"] or "?"
        print(f"  {sample['name'][:25]:25s} ({known:10s}) - distance: {dist:.2f}")


def main():
    replay_dir = Path("/home/campesino/workspace/gosu-unveiled/data/poc_2008_2023/replays")

    print("Loading replays...")
    samples = load_all_replays(replay_dir)
    print(f"Loaded {len(samples)} player-games")

    known_count = sum(1 for s in samples if s["known"])
    print(f"  Known players: {known_count}")
    print(f"  Unknown players: {len(samples) - known_count}")

    X, feature_names = create_matrix(samples)
    print(f"Feature matrix: {X.shape}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    # Plot
    plot_clusters(X_2d, samples, "Player Clustering (PCA projection)\nShapes: ○=Protoss, □=Terran, △=Zerg", "clustering_pca.png")

    # K-means clustering
    labels = analyze_clusters(X_scaled, samples, n_clusters=10)

    # Find similar players to known pros
    print("\n" + "="*60)
    print("SIMILARITY ANALYSIS")
    print("="*60)

    for player in ["Flash", "Bisu", "Jaedong", "Soma"]:
        find_similar_players(X_scaled, samples, player)

    # Self-consistency check
    print("\n" + "="*60)
    print("SELF-CONSISTENCY (do same players cluster together?)")
    print("="*60)

    known_players = set(s["known"] for s in samples if s["known"])
    for player in sorted(known_players):
        indices = [i for i, s in enumerate(samples) if s["known"] == player]
        if len(indices) < 2:
            continue

        # Calculate pairwise distances within this player
        player_X = X_scaled[indices]
        intra_distances = []
        for i in range(len(player_X)):
            for j in range(i+1, len(player_X)):
                intra_distances.append(np.linalg.norm(player_X[i] - player_X[j]))

        avg_intra = np.mean(intra_distances)

        # Calculate distance to other players
        other_indices = [i for i, s in enumerate(samples) if s["known"] and s["known"] != player]
        if other_indices:
            inter_distances = []
            player_center = player_X.mean(axis=0)
            for idx in other_indices:
                inter_distances.append(np.linalg.norm(X_scaled[idx] - player_center))
            avg_inter = np.mean(inter_distances)

            ratio = avg_inter / avg_intra if avg_intra > 0 else 0
            print(f"{player:12s}: intra={avg_intra:.2f}, inter={avg_inter:.2f}, ratio={ratio:.2f} {'✓ distinct' if ratio > 1.5 else ''}")


if __name__ == "__main__":
    main()
