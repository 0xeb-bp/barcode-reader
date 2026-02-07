#!/usr/bin/env python3
"""
Download replays from Team Liquid replay database.
"""

import subprocess
import re
from pathlib import Path
from urllib.parse import unquote
import time

# Change this to target different datasets
REPLAY_DIR = Path("/home/campesino/workspace/gosu-unveiled/data/new_downloads")


def get_replay_filename(replay_id: int) -> str | None:
    """Get the filename for a replay ID by checking the redirect."""
    result = subprocess.run(
        ["curl", "-sI", f"https://tl.net/replay/download.php?replay={replay_id}"],
        capture_output=True, text=True, timeout=10
    )

    for line in result.stdout.split('\n'):
        if line.lower().startswith('location:'):
            location = line.split(':', 1)[1].strip()
            if location and 'upload/' in location:
                filename = location.split('/')[-1]
                # URL decode
                filename = unquote(filename)
                return filename
    return None


def download_replay(replay_id: int, filename: str) -> bool:
    """Download a replay file."""
    filepath = REPLAY_DIR / filename

    if filepath.exists():
        print(f"  Skipping {replay_id}: {filename} (already exists)")
        return False

    result = subprocess.run(
        ["curl", "-sL", f"https://tl.net/replay/download.php?replay={replay_id}",
         "-o", str(filepath)],
        capture_output=True, timeout=30
    )

    # Verify it's a valid file (not empty or error page)
    if filepath.exists() and filepath.stat().st_size > 1000:
        return True
    else:
        filepath.unlink(missing_ok=True)
        return False


def extract_players_from_filename(filename: str) -> list[str]:
    """Try to extract player names from replay filename."""
    # Remove .rep extension
    name = filename.replace('.rep', '')

    # Common patterns: "Player1 vs Player2", "Player1vsPlayer2", "Player1_vs_Player2"
    patterns = [
        r'(.+?)\s*vs\.?\s*(.+)',
        r'(.+?)_vs_(.+)',
        r'(.+?)VS(.+)',
    ]

    for pattern in patterns:
        match = re.match(pattern, name, re.IGNORECASE)
        if match:
            return [match.group(1).strip(), match.group(2).strip()]

    return []


def main():
    REPLAY_DIR.mkdir(exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    # Scan a range of IDs
    # Start from higher IDs (more recent) and work down
    start_id = 2108
    end_id = 1  # Go all the way back
    target_count = 150  # How many new replays we want

    print(f"Scanning replay IDs {start_id} down to {end_id}...")
    print(f"Target: {target_count} new replays")
    print(f"Saving to: {REPLAY_DIR}")
    print("-" * 60)

    for replay_id in range(start_id, end_id, -1):
        if downloaded >= target_count:
            break

        filename = get_replay_filename(replay_id)

        if not filename:
            continue

        # Clean up filename (remove double .rep.rep if present)
        if filename.endswith('.rep.rep'):
            filename = filename[:-4]

        players = extract_players_from_filename(filename)
        player_str = f" ({' vs '.join(players)})" if players else ""

        print(f"[{replay_id}] {filename}{player_str}")

        if download_replay(replay_id, filename):
            downloaded += 1
            print(f"  ✓ Downloaded ({downloaded}/{target_count})")
        else:
            skipped += 1

        # Small delay to be nice to the server
        time.sleep(0.2)

    print("-" * 60)
    print(f"Done! Downloaded: {downloaded}, Skipped: {skipped}")
    print(f"Total replays in folder: {len(list(REPLAY_DIR.glob('*.rep')))}")


if __name__ == "__main__":
    main()
