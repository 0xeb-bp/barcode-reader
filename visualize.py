#!/usr/bin/env python3
"""
Visualize player fingerprint features from SC:BW replays.
"""

import json
import subprocess
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

SCREP_PATH = Path.home() / "go/bin/screp"
FRAME_MS = 42


def parse_replay(replay_path: Path) -> dict:
    result = subprocess.run(
        [str(SCREP_PATH), "-cmds", str(replay_path)],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)


def extract_features(replay_data: dict, player_id: int) -> dict:
    commands = replay_data.get("Commands", {}).get("Cmds", [])
    total_frames = replay_data["Header"]["Frames"]
    game_minutes = (total_frames * FRAME_MS) / 1000 / 60

    player_cmds = [c for c in commands if c.get("PlayerID") == player_id]
    if not player_cmds:
        return None

    cmd_types = defaultdict(int)
    for cmd in player_cmds:
        cmd_types[cmd["Type"]["Name"]] += 1

    # APM curve (first 15 minutes)
    frames_per_minute = int(60 * 1000 / FRAME_MS)
    apm_buckets = defaultdict(int)
    for cmd in player_cmds:
        bucket = cmd["Frame"] // frames_per_minute
        apm_buckets[bucket] += 1

    # Click positions
    clicks = [(c["Pos"]["X"], c["Pos"]["Y"]) for c in player_cmds if "Pos" in c]

    # Inter-action timing
    frames = [c["Frame"] for c in player_cmds]
    gaps = [frames[i] - frames[i-1] for i in range(1, len(frames))]

    return {
        "apm": len(player_cmds) / game_minutes,
        "cmd_dist": {k: v / len(player_cmds) for k, v in cmd_types.items()},
        "apm_curve": [apm_buckets.get(i, 0) for i in range(15)],
        "clicks": clicks,
        "gaps": gaps,
        "game_minutes": game_minutes,
    }


def analyze_all_replays(replay_dir: Path) -> dict:
    """Returns {player_name: [features_list]}"""
    players = defaultdict(list)

    # Manual mapping of aliases to real names (from filenames)
    alias_map = {
        "na1st": "Bisu",
        "fa1con": "Bisu",
        "FlaSh": "Flash",
        "HwaSeungOZ Jaedong": "Jaedong",
        "lllIlIlIIIl11": "Soma",
        "wwvvwwvvwvwww": "Soma",
        "weiguginsagi": "Dewalt",
        "weigugindagsin": "Dewalt",
        "TT1": "TT1",
        "HwaSeungOZ HiyA": "Hiya",
        "Sacsri": "Sacsri",
        "By.SnOw1": "Snow",
        "LC_Tyson": "Tyson",
        "august`rush": "Piano",
    }

    for rep_file in replay_dir.glob("*.rep"):
        try:
            data = parse_replay(rep_file)
            for player in data["Header"]["Players"]:
                if player["Type"]["Name"] != "Human":
                    continue
                name = player["Name"]
                # Skip observers (very low APM)
                features = extract_features(data, player["ID"])
                if features and features["apm"] > 100:  # Filter out observers
                    real_name = alias_map.get(name, name)
                    features["alias"] = name
                    features["race"] = player["Race"]["Name"]
                    features["replay"] = rep_file.name
                    players[real_name].append(features)
        except Exception as e:
            print(f"Error parsing {rep_file}: {e}")

    return players


def plot_apm_curves(players: dict):
    """Plot APM curves for players with multiple games."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Find players with 2+ games
    multi_game_players = {k: v for k, v in players.items() if len(v) >= 2}
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (name, games) in enumerate(list(multi_game_players.items())[:4]):
        ax = axes[idx]
        for i, game in enumerate(games):
            minutes = min(10, int(game["game_minutes"]))
            curve = game["apm_curve"][:minutes]
            ax.plot(range(len(curve)), curve,
                   label=f"{game['alias'][:15]} ({game['race'][0]})",
                   marker='o', markersize=4, alpha=0.7)

        ax.set_title(f"{name} - APM Curves Across Games")
        ax.set_xlabel("Game Minute")
        ax.set_ylabel("Actions per Minute")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/campesino/workspace/gosu-unveiled/apm_curves.png", dpi=150)
    print("Saved apm_curves.png")


def plot_command_distribution(players: dict):
    """Compare command distributions across players."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Key command types to compare
    cmd_types = ["Hotkey", "Right Click", "Select", "Targeted Order", "Train"]

    # Get players with 2+ games
    multi_game = {k: v for k, v in players.items() if len(v) >= 2}

    player_names = list(multi_game.keys())[:6]
    x = np.arange(len(cmd_types))
    width = 0.12

    for i, name in enumerate(player_names):
        # Average across games
        games = multi_game[name]
        avg_dist = {}
        for cmd in cmd_types:
            vals = [g["cmd_dist"].get(cmd, 0) for g in games]
            avg_dist[cmd] = np.mean(vals)

        values = [avg_dist.get(cmd, 0) * 100 for cmd in cmd_types]
        ax.bar(x + i * width, values, width, label=f"{name} ({len(games)} games)")

    ax.set_xlabel("Command Type")
    ax.set_ylabel("Percentage of Actions")
    ax.set_title("Command Distribution by Player (Averaged)")
    ax.set_xticks(x + width * len(player_names) / 2)
    ax.set_xticklabels(cmd_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("/home/campesino/workspace/gosu-unveiled/cmd_distribution.png", dpi=150)
    print("Saved cmd_distribution.png")


def plot_feature_consistency(players: dict):
    """Show how consistent features are within vs across players."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    multi_game = {k: v for k, v in players.items() if len(v) >= 2}

    # Plot 1: APM consistency
    ax1 = axes[0]
    for name, games in multi_game.items():
        apms = [g["apm"] for g in games]
        ax1.scatter([name] * len(apms), apms, s=100, alpha=0.7, label=name)
        ax1.plot([name, name], [min(apms), max(apms)], 'k-', alpha=0.3)

    ax1.set_ylabel("APM")
    ax1.set_title("APM Consistency Within Players")
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Hotkey % consistency
    ax2 = axes[1]
    for name, games in multi_game.items():
        hotkey_pcts = [g["cmd_dist"].get("Hotkey", 0) * 100 for g in games]
        ax2.scatter([name] * len(hotkey_pcts), hotkey_pcts, s=100, alpha=0.7)
        ax2.plot([name, name], [min(hotkey_pcts), max(hotkey_pcts)], 'k-', alpha=0.3)

    ax2.set_ylabel("Hotkey %")
    ax2.set_title("Hotkey Usage Consistency Within Players")
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("/home/campesino/workspace/gosu-unveiled/consistency.png", dpi=150)
    print("Saved consistency.png")


def print_summary(players: dict):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("PLAYER SUMMARY")
    print("="*60)

    multi_game = {k: v for k, v in players.items() if len(v) >= 2}

    for name, games in sorted(multi_game.items()):
        print(f"\n{name} ({len(games)} games)")
        print("-" * 40)

        apms = [g["apm"] for g in games]
        hotkeys = [g["cmd_dist"].get("Hotkey", 0) * 100 for g in games]
        right_clicks = [g["cmd_dist"].get("Right Click", 0) * 100 for g in games]

        print(f"  APM: {np.mean(apms):.0f} ± {np.std(apms):.1f}")
        print(f"  Hotkey %: {np.mean(hotkeys):.1f} ± {np.std(hotkeys):.1f}")
        print(f"  Right Click %: {np.mean(right_clicks):.1f} ± {np.std(right_clicks):.1f}")

        for g in games:
            print(f"    - {g['alias'][:20]} ({g['race'][0]}) APM:{g['apm']:.0f}")


def main():
    replay_dir = Path("/home/campesino/workspace/gosu-unveiled/data/poc_2008_2023/replays")
    players = analyze_all_replays(replay_dir)

    print(f"Found {len(players)} unique players")
    for name, games in players.items():
        print(f"  {name}: {len(games)} game(s)")

    print_summary(players)
    plot_apm_curves(players)
    plot_command_distribution(players)
    plot_feature_consistency(players)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
