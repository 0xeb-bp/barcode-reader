#!/usr/bin/env python3
"""
Extract n-gram patterns and early game signatures from SC:BW replays.
"""

import json
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
import statistics

SCREP_PATH = Path.home() / "go/bin/screp"
FRAME_MS = 42
FRAMES_PER_SECOND = 1000 / FRAME_MS  # ~23.8 fps
FRAMES_PER_MINUTE = FRAMES_PER_SECOND * 60  # ~1428 frames


def parse_replay(replay_path: Path) -> dict:
    result = subprocess.run(
        [str(SCREP_PATH), "-cmds", str(replay_path)],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)


def get_command_signature(cmd: dict) -> str:
    """Convert a command to a string signature for n-gram analysis."""
    cmd_type = cmd["Type"]["Name"]

    # Add specificity for certain commands
    if cmd_type == "Hotkey":
        hotkey_type = cmd.get("HotkeyType", {}).get("Name", "Unknown")
        group = cmd.get("Group", "?")
        return f"Hotkey_{hotkey_type}_{group}"

    if cmd_type == "Targeted Order" and "Order" in cmd:
        return f"Order_{cmd['Order']['Name']}"

    if cmd_type == "Train" and "Unit" in cmd:
        return f"Train_{cmd['Unit']['Name']}"

    if cmd_type == "Unit Morph" and "Unit" in cmd:
        return f"Morph_{cmd['Unit']['Name']}"

    return cmd_type


def get_simple_signature(cmd: dict) -> str:
    """Simpler signature - just the command type."""
    return cmd["Type"]["Name"]


def extract_ngrams(commands: list, n: int = 3, use_detailed: bool = False) -> Counter:
    """Extract n-grams from command sequence."""
    sig_func = get_command_signature if use_detailed else get_simple_signature
    signatures = [sig_func(cmd) for cmd in commands]

    ngrams = Counter()
    for i in range(len(signatures) - n + 1):
        gram = " → ".join(signatures[i:i+n])
        ngrams[gram] += 1

    return ngrams


def extract_early_game_features(commands: list, player_id: int, minutes: float = 2.0) -> dict:
    """Extract features from the first N minutes of the game."""

    cutoff_frame = int(minutes * FRAMES_PER_MINUTE)
    early_cmds = [c for c in commands if c["PlayerID"] == player_id and c["Frame"] < cutoff_frame]

    if len(early_cmds) < 10:
        return None

    # === Timing patterns ===
    frames = [c["Frame"] for c in early_cmds]
    gaps = [frames[i] - frames[i-1] for i in range(1, len(frames))]

    # Gap distribution (how "bursty" vs "steady" are they?)
    gap_stats = {
        "mean_gap_ms": statistics.mean(gaps) * FRAME_MS if gaps else 0,
        "std_gap_ms": statistics.stdev(gaps) * FRAME_MS if len(gaps) > 1 else 0,
        "min_gap_ms": min(gaps) * FRAME_MS if gaps else 0,
        "max_gap_ms": max(gaps) * FRAME_MS if gaps else 0,
    }

    # Burst detection: count rapid sequences (gap < 100ms)
    rapid_actions = sum(1 for g in gaps if g * FRAME_MS < 100)
    gap_stats["rapid_action_ratio"] = rapid_actions / len(gaps) if gaps else 0

    # === Initial hotkey setup ===
    hotkey_cmds = [c for c in early_cmds if c["Type"]["Name"] == "Hotkey"]

    # What groups do they assign first?
    assigns = [(c["Frame"], c.get("Group"))
               for c in hotkey_cmds
               if c.get("HotkeyType", {}).get("Name") == "Assign"]

    # First 5 hotkey assignments (in order)
    first_assigns = [g for _, g in sorted(assigns)[:5]]

    # Which groups do they use most in early game?
    group_usage = Counter(c.get("Group") for c in hotkey_cmds)

    # === Command rhythm ===
    # How do they start? First 20 commands
    first_20_types = [c["Type"]["Name"] for c in early_cmds[:20]]

    # === Spam patterns ===
    # Look for repeated identical commands in sequence
    spam_sequences = []
    current_spam = 1
    for i in range(1, len(early_cmds)):
        if get_simple_signature(early_cmds[i]) == get_simple_signature(early_cmds[i-1]):
            current_spam += 1
        else:
            if current_spam >= 3:
                spam_sequences.append((get_simple_signature(early_cmds[i-1]), current_spam))
            current_spam = 1

    # === N-grams for early game ===
    early_ngrams = extract_ngrams(early_cmds, n=3, use_detailed=False)

    # === Click position patterns (early game) ===
    early_clicks = [(c["Pos"]["X"], c["Pos"]["Y"])
                    for c in early_cmds if "Pos" in c]

    # Calculate "jitter" - how much do consecutive clicks move?
    click_distances = []
    for i in range(1, len(early_clicks)):
        dx = early_clicks[i][0] - early_clicks[i-1][0]
        dy = early_clicks[i][1] - early_clicks[i-1][1]
        dist = (dx**2 + dy**2) ** 0.5
        click_distances.append(dist)

    return {
        "early_game_minutes": minutes,
        "early_action_count": len(early_cmds),
        "early_apm": len(early_cmds) / minutes,

        # Timing
        **gap_stats,

        # Hotkey setup
        "first_hotkey_assigns": first_assigns,
        "hotkey_group_usage": dict(group_usage.most_common(5)),

        # Opening sequence
        "first_20_commands": first_20_types,

        # Spam patterns
        "spam_sequences": spam_sequences[:5],

        # Top n-grams
        "top_ngrams": dict(early_ngrams.most_common(10)),

        # Click patterns
        "avg_click_distance": statistics.mean(click_distances) if click_distances else 0,
        "click_distance_std": statistics.stdev(click_distances) if len(click_distances) > 1 else 0,
    }


def analyze_player(replay_path: Path, player_name_filter: str = None):
    """Analyze a replay and show n-grams + early game patterns."""

    data = parse_replay(replay_path)
    header = data["Header"]
    commands = data.get("Commands", {}).get("Cmds", [])

    print(f"\n{'='*70}")
    print(f"Replay: {replay_path.name}")
    print(f"Map: {header.get('Map', 'Unknown')}")
    print('='*70)

    for player in header["Players"]:
        if player["Type"]["Name"] != "Human":
            continue

        name = player["Name"]
        player_id = player["ID"]

        # Skip low-APM players (observers)
        player_cmds = [c for c in commands if c["PlayerID"] == player_id]
        if len(player_cmds) < 500:
            continue

        if player_name_filter and player_name_filter.lower() not in name.lower():
            continue

        print(f"\n--- {name} ({player['Race']['Name']}) ---")

        # Full game n-grams
        print("\n[FULL GAME N-GRAMS - Top 15]")
        ngrams = extract_ngrams(player_cmds, n=3, use_detailed=False)
        for gram, count in ngrams.most_common(15):
            pct = count / len(player_cmds) * 100
            print(f"  {gram}: {count} ({pct:.1f}%)")

        # Detailed n-grams (with hotkey groups, specific orders)
        print("\n[DETAILED N-GRAMS - Top 10]")
        detailed_ngrams = extract_ngrams(player_cmds, n=3, use_detailed=True)
        for gram, count in detailed_ngrams.most_common(10):
            print(f"  {gram}: {count}")

        # Early game analysis
        print("\n[EARLY GAME - First 2 Minutes]")
        early = extract_early_game_features(commands, player_id, minutes=2.0)

        if early:
            print(f"  Actions in first 2 min: {early['early_action_count']}")
            print(f"  Early APM: {early['early_apm']:.0f}")
            print(f"  Mean gap between actions: {early['mean_gap_ms']:.0f}ms")
            print(f"  Gap std dev: {early['std_gap_ms']:.0f}ms")
            print(f"  Rapid action ratio (<100ms): {early['rapid_action_ratio']*100:.1f}%")

            print(f"\n  First hotkey assigns (groups): {early['first_hotkey_assigns']}")
            print(f"  Hotkey group usage: {early['hotkey_group_usage']}")

            print(f"\n  First 20 commands:")
            # Print in rows of 5
            cmds = early['first_20_commands']
            for i in range(0, len(cmds), 5):
                print(f"    {' → '.join(cmds[i:i+5])}")

            if early['spam_sequences']:
                print(f"\n  Spam patterns detected: {early['spam_sequences']}")

            print(f"\n  Avg click movement: {early['avg_click_distance']:.0f} pixels")
            print(f"  Click movement std: {early['click_distance_std']:.0f} pixels")

            print(f"\n  Top early-game n-grams:")
            for gram, count in list(early['top_ngrams'].items())[:7]:
                print(f"    {gram}: {count}")


def compare_players_early_game(replay_dir: Path):
    """Compare early game signatures across known players."""

    # Map aliases to real names
    alias_map = {
        "na1st": "Bisu", "fa1con": "Bisu",
        "FlaSh": "Flash",
        "HwaSeungOZ Jaedong": "Jaedong",
        "lllIlIlIIIl11": "Soma", "wwvvwwvvwvwww": "Soma",
        "weiguginsagi": "Dewalt", "weigugindagsin": "Dewalt",
    }

    player_early_games = defaultdict(list)

    for rep_file in replay_dir.glob("*.rep"):
        try:
            data = parse_replay(rep_file)
            commands = data.get("Commands", {}).get("Cmds", [])

            for player in data["Header"]["Players"]:
                if player["Type"]["Name"] != "Human":
                    continue

                name = player["Name"]
                real_name = alias_map.get(name)
                if not real_name:
                    continue

                early = extract_early_game_features(commands, player["ID"], minutes=2.0)
                if early:
                    early["alias"] = name
                    early["replay"] = rep_file.name
                    player_early_games[real_name].append(early)

        except Exception as e:
            print(f"Error with {rep_file}: {e}")

    print("\n" + "="*70)
    print("EARLY GAME COMPARISON (First 2 Minutes)")
    print("="*70)

    for player, games in sorted(player_early_games.items()):
        if len(games) < 2:
            continue

        print(f"\n{player} ({len(games)} games)")
        print("-" * 50)

        # Compare metrics across games
        apms = [g["early_apm"] for g in games]
        gaps = [g["mean_gap_ms"] for g in games]
        rapid = [g["rapid_action_ratio"] * 100 for g in games]

        print(f"  Early APM: {statistics.mean(apms):.0f} ± {statistics.stdev(apms):.1f}")
        print(f"  Mean action gap: {statistics.mean(gaps):.0f}ms ± {statistics.stdev(gaps):.1f}ms")
        print(f"  Rapid action %: {statistics.mean(rapid):.1f}% ± {statistics.stdev(rapid):.1f}%")

        # First hotkey patterns
        print(f"\n  First hotkey assigns per game:")
        for g in games:
            print(f"    {g['alias'][:15]}: {g['first_hotkey_assigns']}")

        # Common opening n-grams
        print(f"\n  Top early n-grams (consistent across games):")
        all_ngrams = Counter()
        for g in games:
            for gram, count in g["top_ngrams"].items():
                all_ngrams[gram] += count
        for gram, count in all_ngrams.most_common(5):
            print(f"    {gram}: {count}")


def main():
    import sys

    replay_dir = Path("/home/campesino/workspace/gosu-unveiled/data/poc_2008_2023/replays")

    if len(sys.argv) > 1:
        # Analyze specific replay
        for rep in sys.argv[1:]:
            analyze_player(Path(rep))
    else:
        # Compare known players
        compare_players_early_game(replay_dir)

        # Also show detailed analysis for a couple replays
        print("\n\n" + "="*70)
        print("DETAILED ANALYSIS - Sample Replays")
        print("="*70)

        samples = [
            "Jaedong vs Hiya.rep",
            "Bisu vs Soulkey.rep",
        ]

        for sample in samples:
            path = replay_dir / sample
            if path.exists():
                analyze_player(path)


if __name__ == "__main__":
    main()
