#!/usr/bin/env python3
"""
Shared feature extraction for StarCraft: Brood War player fingerprinting.
"""

import json
import subprocess
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict
import statistics
import numpy as np

SCREP_PATH = Path.home() / "go/bin/screp"
DB_PATH = Path(__file__).parent / "data" / "replays.db"
FRAME_MS = 42
FRAMES_PER_MINUTE = (1000 / FRAME_MS) * 60
MIN_GAME_MINUTES = 4
MIN_COMMANDS = 100

# Action category mapping - abstracts race-specific actions
ACTION_CATEGORIES = {
    # Race-neutral actions - KEEP DISTINCT (these are the fingerprint)
    "Hotkey": "H", "Select": "S", "Select Add": "S+", "Select Remove": "S-",
    "Right Click": "R", "Hold Position": "HP", "Stop": "ST", "Return Cargo": "RC",
    "Targeted Order": "TO",

    # Race-specific abilities - ALL collapse to "Abl" (no race leakage)
    "Stim": "Abl", "Burrow": "Abl", "Unburrow": "Abl",
    "Siege": "Abl", "Unsiege": "Abl",
    "Cloack": "Abl", "Decloack": "Abl",
    "Merge Archon": "Abl", "Merge Dark Archon": "Abl",
    "Lift Off": "Abl", "Land": "Abl",

    # Production - all races produce, abstract the method
    "Train": "Prod", "Unit Morph": "Prod", "Build": "Bld",
    "Building Morph": "Prod", "Train Fighter": "Prod",

    # Upgrades
    "Upgrade": "Upg", "Tech": "Tech",

    # Transport
    "Unload": "Unld", "Unload All": "UnldA",

    # Cancel
    "Cancel Train": "Can", "Cancel Build": "Can", "Cancel Morph": "Can",
    "Cancel Upgrade": "Can", "Cancel Tech": "Can", "Cancel Addon": "Can",
}

# Global top-N n-grams to keep per type (two-pass selection)
GLOBAL_NGRAM_TOP_N = {
    "ng2": 25, "ng3": 20, "ng4": 15,
    "hkg2": 15, "hkg3": 10,
    "ehkg2": 10, "ehkg3": 8,
}


def parse_replay(replay_path: Path) -> dict:
    """Parse replay with screp."""
    result = subprocess.run(
        [str(SCREP_PATH), "-cmds", str(replay_path)],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(f"screp failed: {result.stderr}")
    return json.loads(result.stdout)


def trim_at_leave(player_cmds: list, all_cmds: list, game_frames: int) -> tuple:
    """Trim player commands at the first Leave Game by any player.
    Returns (trimmed_cmds, effective_game_frames)."""
    leave_frame = None
    for c in all_cmds:
        if c["Type"]["Name"] == "Leave Game":
            leave_frame = c["Frame"]
            break

    if leave_frame is not None:
        player_cmds = [c for c in player_cmds if c["Frame"] <= leave_frame]
        game_frames = min(game_frames, leave_frame)

    return player_cmds, game_frames


def collapse_consecutive_prod(categories: list) -> list:
    """Collapse consecutive Prod tokens (race-mechanical queuing, not playstyle)."""
    if not categories:
        return []
    collapsed = [categories[0]]
    for cat in categories[1:]:
        if cat == "Prod" and collapsed[-1] == "Prod":
            continue
        collapsed.append(cat)
    return collapsed


def extract_abstracted_ngrams(commands: list, n: int) -> Counter:
    """Extract n-grams using abstracted action categories with consecutive dedup."""
    categories = [ACTION_CATEGORIES.get(c["Type"]["Name"], "O") for c in commands]
    categories = collapse_consecutive_prod(categories)
    ngrams = Counter()
    for i in range(len(categories) - n + 1):
        gram = "_".join(categories[i:i+n])
        ngrams[gram] += 1
    return ngrams


def extract_features(commands: list, game_frames: int) -> tuple:
    """Extract base features + raw n-gram counters (for two-pass selection)."""
    if len(commands) < MIN_COMMANDS:
        return None, None

    game_minutes = (game_frames * FRAME_MS) / 1000 / 60
    if game_minutes < MIN_GAME_MINUTES:
        return None, None

    features = {}
    raw_ngrams = {}

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

        # === HOTKEY GROUP TRANSITION MATRIX ===
        if len(groups) > 5:
            used_groups = sorted(group_counts.keys())
            transitions = Counter()
            for i in range(1, len(groups)):
                transitions[(groups[i-1], groups[i])] += 1
            for g_from in used_groups[:5]:
                from_total = sum(transitions[(g_from, g_to)] for g_to in used_groups)
                if from_total > 0:
                    for g_to in used_groups[:5]:
                        prob = transitions[(g_from, g_to)] / from_total
                        if prob > 0:
                            features[f"hk_tr_{g_from}_{g_to}"] = prob

        # === HOTKEY GROUP N-GRAMS (raw counters for two-pass) ===
        if len(groups) > 10:
            group_strs = [str(g) for g in groups]
            for n in [2, 3]:
                grp_ngrams = Counter()
                for i in range(len(group_strs) - n + 1):
                    gram = "_".join(group_strs[i:i+n])
                    grp_ngrams[gram] += 1
                total_gng = sum(grp_ngrams.values())
                if total_gng > 0:
                    raw_ngrams[f"hkg{n}"] = (grp_ngrams, total_gng)

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

        # Early hotkey group n-grams (raw counters for two-pass)
        early_groups = [c.get("Group", 0) for c in early_hotkeys]
        if len(early_groups) > 5:
            early_grp_strs = [str(g) for g in early_groups]
            for n in [2, 3]:
                early_gng = Counter()
                for i in range(len(early_grp_strs) - n + 1):
                    gram = "_".join(early_grp_strs[i:i+n])
                    early_gng[gram] += 1
                total_egng = sum(early_gng.values())
                if total_egng > 0:
                    raw_ngrams[f"ehkg{n}"] = (early_gng, total_egng)

    # === OTHER RACE-NEUTRAL ===
    queued = sum(1 for c in commands if c.get("Queued", False))
    features["queued_ratio"] = queued / len(commands)

    # === SELECTION PATTERNS ===
    selections = [c for c in commands if c["Type"]["Name"] == "Select"]
    select_adds = [c for c in commands if c["Type"]["Name"] == "Select Add"]
    if selections:
        sizes = [len(c.get("UnitTags", [])) for c in selections if "UnitTags" in c]
        if sizes:
            features["select_size_mean"] = statistics.mean(sizes)
            features["select_size_std"] = statistics.stdev(sizes) if len(sizes) > 1 else 0
        features["selection_action_ratio"] = len(selections) / len(commands)

        # Select Add ratio — shift-clicker vs drag-boxer vs pure hotkey
        features["select_add_ratio"] = len(select_adds) / len(selections) if selections else 0

        # Selection tempo — time gaps between consecutive select-type commands
        select_cmds = [c for c in commands if c["Type"]["Name"] in ("Select", "Select Add", "Select Remove")]
        if len(select_cmds) > 5:
            sel_gaps_ms = [(select_cmds[i]["Frame"] - select_cmds[i-1]["Frame"]) * FRAME_MS
                          for i in range(1, len(select_cmds))]
            features["select_gap_mean"] = statistics.mean(sel_gaps_ms)
            features["select_gap_median"] = statistics.median(sel_gaps_ms)
            features["reselect_burst_ratio"] = sum(1 for g in sel_gaps_ms if g < 200) / len(sel_gaps_ms)

    # TODO: Selection size distribution shape (binned) — disabled, may correlate with race/game-state
    # pct_select_1    = count(size == 1)  / total    # single click
    # pct_select_2_4  = count(2 <= s <= 4)  / total  # small drag or double-click
    # pct_select_5_8  = count(5 <= s <= 8)  / total  # medium drag box
    # pct_select_9_12 = count(9 <= s <= 12) / total  # fat drag box (BW max 12)

    # === SELECT→ACTION LATENCY (race-invariant motor pattern) ===
    sa_gaps = []
    for i in range(len(commands) - 1):
        if commands[i]["Type"]["Name"] in ("Select", "Hotkey"):
            if commands[i+1]["Type"]["Name"] not in ("Select", "Select Add", "Select Remove", "Hotkey"):
                gap = (commands[i+1]["Frame"] - commands[i]["Frame"]) * FRAME_MS
                if gap < 2000:
                    sa_gaps.append(gap)
    if sa_gaps:
        features["sa_latency_mean"] = statistics.mean(sa_gaps)
        features["sa_latency_median"] = statistics.median(sa_gaps)

    # === BURST STRUCTURE (race-invariant rhythm) ===
    if gaps:
        burst_threshold_ms = 150
        bursts = []
        current_burst = 1
        for g in gaps_ms:
            if g < burst_threshold_ms:
                current_burst += 1
            else:
                if current_burst > 1:
                    bursts.append(current_burst)
                current_burst = 1
        if current_burst > 1:
            bursts.append(current_burst)

        if bursts:
            features["burst_count_per_min"] = len(bursts) / game_minutes
            features["burst_size_mean"] = statistics.mean(bursts)

        inter_burst_gaps = [g for g in gaps_ms if g >= burst_threshold_ms]
        if inter_burst_gaps:
            features["inter_burst_gap_mean"] = statistics.mean(inter_burst_gaps)

    # === RHYTHM AUTOCORRELATION (race-invariant timing signature) ===
    if len(gaps) > 20:
        g_arr = np.array(gaps_ms[:200])
        g_centered = g_arr - g_arr.mean()
        var = np.sum(g_centered ** 2)
        if var > 0:
            features["autocorr_lag1"] = float(np.sum(g_centered[:-1] * g_centered[1:]) / var)

    # === MAP JUMPS (multitask switching — race-invariant) ===
    if len(clicks) > 10:
        map_jumps = 0
        click_frames = [(c["Pos"]["X"], c["Pos"]["Y"], c["Frame"]) for c in commands if "Pos" in c]
        for i in range(1, len(click_frames)):
            dx = click_frames[i][0] - click_frames[i-1][0]
            dy = click_frames[i][1] - click_frames[i-1][1]
            dist = (dx**2 + dy**2) ** 0.5
            time_gap = (click_frames[i][2] - click_frames[i-1][2]) * FRAME_MS
            if dist > 2000 and time_gap < 500:
                map_jumps += 1
        features["map_jumps_per_min"] = map_jumps / game_minutes

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

    # === ABSTRACTED N-GRAMS (raw counters for two-pass) ===
    for n in [2, 3, 4]:
        ngrams = extract_abstracted_ngrams(commands, n)
        total_ng = sum(ngrams.values())
        if total_ng > 0:
            raw_ngrams[f"ng{n}"] = (ngrams, total_ng)

    return features, raw_ngrams


def select_global_ngrams(all_raw_ngrams):
    """Two-pass: aggregate n-gram counts across all samples and pick global top-N."""
    global_counts = {}

    for sample_ngrams in all_raw_ngrams:
        for prefix, (counter, total) in sample_ngrams.items():
            if prefix not in global_counts:
                global_counts[prefix] = Counter()
            global_counts[prefix] += counter

    selected = {}
    for prefix, counter in global_counts.items():
        top_n = GLOBAL_NGRAM_TOP_N.get(prefix, 10)
        selected[prefix] = [gram for gram, _ in counter.most_common(top_n)]

    return selected


def apply_ngram_features(features, raw_ngrams, global_ngrams):
    """Apply globally-selected n-gram set to a single sample's features."""
    for prefix, selected_grams in global_ngrams.items():
        counter, total = raw_ngrams.get(prefix, (Counter(), 0))
        for gram in selected_grams:
            features[f"{prefix}_{gram}"] = counter.get(gram, 0) / total if total > 0 else 0


def get_player_replays(conn, player_name: str, year: str = None, min_date: str = None):
    """Legacy: Get replay paths for a player by display name."""
    c = conn.cursor()
    if year:
        c.execute("""
            SELECT r.file_path, r.id
            FROM replays r
            JOIN players p ON r.id = p.replay_id
            WHERE p.player_name = ?
              AND p.is_human = 1
              AND substr(r.game_date, 1, 4) = ?
        """, (player_name, year))
    elif min_date:
        c.execute("""
            SELECT r.file_path, r.id
            FROM replays r
            JOIN players p ON r.id = p.replay_id
            WHERE p.player_name = ?
              AND p.is_human = 1
              AND r.game_date >= ?
        """, (player_name, min_date))
    else:
        c.execute("""
            SELECT r.file_path, r.id
            FROM replays r
            JOIN players p ON r.id = p.replay_id
            WHERE p.player_name = ?
              AND p.is_human = 1
        """, (player_name,))
    return c.fetchall()


def extract_player_samples(conn, player_name: str, label: str = None,
                           year: str = None, min_date: str = None,
                           max_games: int = 30):
    """Legacy: Extract feature samples for a player by display name."""
    replays = get_player_replays(conn, player_name, year=year, min_date=min_date)
    if label is None:
        label = player_name
    samples = []

    for file_path, replay_id in replays[:max_games]:
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
                features, raw_ngrams = extract_features(player_cmds, effective_frames)

                if features:
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



def get_pro_identities(conn, min_games=10):
    """Get all confirmed pros from player_identities, filtered by modern-era game count.
    Returns list of (canonical_name, [aurora_ids], game_count)."""
    c = conn.cursor()
    c.execute("SELECT canonical_name, aurora_id FROM player_identities")

    pros = defaultdict(list)
    for canonical, aurora_id in c.fetchall():
        pros[canonical].append(aurora_id)

    result = []
    for canonical, aurora_ids in pros.items():
        placeholders = ",".join(["?"] * len(aurora_ids))
        c.execute(f"""
            SELECT COUNT(DISTINCT p.replay_id)
            FROM players p
            JOIN replays r ON r.id = p.replay_id
            WHERE p.aurora_id IN ({placeholders})
              AND p.is_human = 1
              AND r.game_date >= '2025-01-01'
        """, aurora_ids)
        total = c.fetchone()[0]
        if total >= min_games:
            result.append((canonical, aurora_ids, total))

    result.sort(key=lambda x: -x[2])
    return result


def get_player_replays_by_aurora(conn, aurora_ids, min_date=None):
    """Get replay paths for a player by aurora_id(s).
    Returns (file_path, replay_id, player_name) — player_name needed to find
    the right player when parsing the .rep file."""
    c = conn.cursor()
    placeholders = ",".join(["?"] * len(aurora_ids))
    if min_date:
        c.execute(f"""
            SELECT r.file_path, r.id, p.player_name
            FROM replays r
            JOIN players p ON r.id = p.replay_id
            WHERE p.aurora_id IN ({placeholders})
              AND p.is_human = 1
              AND r.game_date >= ?
        """, aurora_ids + [min_date])
    else:
        c.execute(f"""
            SELECT r.file_path, r.id, p.player_name
            FROM replays r
            JOIN players p ON r.id = p.replay_id
            WHERE p.aurora_id IN ({placeholders})
              AND p.is_human = 1
        """, aurora_ids)
    return c.fetchall()


def extract_player_samples_by_aurora(conn, aurora_ids, label, min_date=None):
    """Extract feature samples for a player by aurora_id(s)."""
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
                features, raw_ngrams = extract_features(player_cmds, effective_frames)

                if features:
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


def create_feature_matrix(samples, feature_names=None):
    """Convert samples to numpy arrays."""
    if feature_names is None:
        all_features = set()
        for s in samples:
            all_features.update(s["features"].keys())
        feature_names = sorted(all_features)

    X = np.array([[s["features"].get(f, 0) for f in feature_names] for s in samples])
    y = np.array([s["label"] for s in samples])
    return X, y, feature_names
