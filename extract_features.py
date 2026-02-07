#!/usr/bin/env python3
"""
Extract player fingerprint features from StarCraft: Brood War replays.
"""

import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import statistics

SCREP_PATH = Path.home() / "go/bin/screp"
FRAME_MS = 42  # 1 frame = 42 milliseconds


def parse_replay(replay_path: Path) -> dict:
    """Run screp and return parsed JSON."""
    result = subprocess.run(
        [str(SCREP_PATH), "-cmds", str(replay_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"screp failed: {result.stderr}")
    return json.loads(result.stdout)


def extract_player_features(replay_data: dict, player_id: int) -> dict:
    """Extract fingerprint features for a specific player."""

    commands = replay_data.get("Commands", {}).get("Cmds", [])
    total_frames = replay_data["Header"]["Frames"]
    game_minutes = (total_frames * FRAME_MS) / 1000 / 60

    # Filter to this player's commands
    player_cmds = [c for c in commands if c.get("PlayerID") == player_id]

    if not player_cmds:
        return None

    # Basic counts
    cmd_types = defaultdict(int)
    for cmd in player_cmds:
        cmd_types[cmd["Type"]["Name"]] += 1

    # Click positions (for heatmap)
    click_positions = []
    for cmd in player_cmds:
        if "Pos" in cmd:
            click_positions.append((cmd["Pos"]["X"], cmd["Pos"]["Y"]))

    # Selection sizes
    selection_sizes = []
    for cmd in player_cmds:
        if cmd["Type"]["Name"] == "Select" and "UnitTags" in cmd:
            selection_sizes.append(len(cmd["UnitTags"]))

    # Inter-action timing (frames between consecutive actions)
    frames = [c["Frame"] for c in player_cmds]
    inter_action_frames = []
    for i in range(1, len(frames)):
        gap = frames[i] - frames[i-1]
        inter_action_frames.append(gap)

    # APM over time (split game into 1-minute buckets)
    frames_per_minute = int(60 * 1000 / FRAME_MS)
    apm_buckets = defaultdict(int)
    for cmd in player_cmds:
        bucket = cmd["Frame"] // frames_per_minute
        apm_buckets[bucket] += 1

    # Hotkey usage
    hotkey_sets = 0
    hotkey_gets = 0
    for cmd in player_cmds:
        if cmd["Type"]["Name"] == "Hotkey":
            if cmd.get("Set"):
                hotkey_sets += 1
            else:
                hotkey_gets += 1

    # Queued actions ratio
    queued_count = sum(1 for c in player_cmds if c.get("Queued", False))

    features = {
        "total_actions": len(player_cmds),
        "game_minutes": round(game_minutes, 2),
        "apm": round(len(player_cmds) / game_minutes, 1),

        # Command distribution (normalized)
        "cmd_distribution": {k: round(v / len(player_cmds), 4) for k, v in cmd_types.items()},

        # Selection behavior
        "avg_selection_size": round(statistics.mean(selection_sizes), 2) if selection_sizes else 0,
        "selection_size_std": round(statistics.stdev(selection_sizes), 2) if len(selection_sizes) > 1 else 0,

        # Timing patterns
        "avg_inter_action_frames": round(statistics.mean(inter_action_frames), 2) if inter_action_frames else 0,
        "inter_action_std": round(statistics.stdev(inter_action_frames), 2) if len(inter_action_frames) > 1 else 0,

        # APM curve (first 10 minutes)
        "apm_curve": [apm_buckets.get(i, 0) for i in range(min(10, int(game_minutes) + 1))],

        # Hotkey patterns
        "hotkey_sets": hotkey_sets,
        "hotkey_gets": hotkey_gets,
        "hotkey_ratio": round(hotkey_gets / hotkey_sets, 2) if hotkey_sets > 0 else 0,

        # Queuing behavior
        "queued_ratio": round(queued_count / len(player_cmds), 4),

        # Click positions (for heatmap analysis)
        "click_count": len(click_positions),
        "click_positions": click_positions[:100],  # Sample for visualization
    }

    return features


def analyze_replay(replay_path: Path) -> dict:
    """Analyze a replay and return features for all players."""
    data = parse_replay(replay_path)

    header = data["Header"]
    players = header["Players"]

    result = {
        "replay": replay_path.name,
        "map": header.get("Map", "Unknown"),
        "duration_minutes": round((header["Frames"] * FRAME_MS) / 1000 / 60, 2),
        "players": []
    }

    for player in players:
        if player["Type"]["Name"] != "Human":
            continue

        player_id = player["ID"]
        features = extract_player_features(data, player_id)

        if features:
            result["players"].append({
                "name": player["Name"],
                "race": player["Race"]["Name"],
                "id": player_id,
                "features": features
            })

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_features.py <replay.rep> [replay2.rep ...]")
        sys.exit(1)

    for replay_file in sys.argv[1:]:
        path = Path(replay_file)
        if not path.exists():
            print(f"File not found: {path}")
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing: {path.name}")
        print('='*60)

        try:
            result = analyze_replay(path)

            print(f"Map: {result['map']}")
            print(f"Duration: {result['duration_minutes']} minutes")
            print()

            for player in result["players"]:
                print(f"Player: {player['name']} ({player['race']})")
                f = player["features"]
                print(f"  APM: {f['apm']}")
                print(f"  Total actions: {f['total_actions']}")
                print(f"  Avg selection size: {f['avg_selection_size']}")
                print(f"  Hotkey gets/sets: {f['hotkey_gets']}/{f['hotkey_sets']} (ratio: {f['hotkey_ratio']})")
                print(f"  Queued action ratio: {f['queued_ratio']}")
                print(f"  Avg inter-action gap: {f['avg_inter_action_frames']} frames ({round(f['avg_inter_action_frames'] * FRAME_MS)}ms)")
                print(f"  APM curve (per minute): {f['apm_curve']}")
                print()

                # Top command types
                print("  Command distribution:")
                sorted_cmds = sorted(f["cmd_distribution"].items(), key=lambda x: -x[1])[:5]
                for cmd, ratio in sorted_cmds:
                    print(f"    {cmd}: {ratio*100:.1f}%")
                print()

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
