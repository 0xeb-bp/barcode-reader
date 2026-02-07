#!/usr/bin/env python3
"""
Scrape high-MMR replays from cwal.gg via their Supabase API.
"""

import os
import json
import time
import requests
from pathlib import Path
from urllib.parse import urlencode

# cwal.gg Supabase config (public anon key)
SUPABASE_URL = "https://xmploueumzkrdvapbyfs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhtcGxvdWV1bXprcmR2YXBieWZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NzI4ODY5MTQsImV4cCI6MTk4ODQ2MjkxNH0.p8Jkm2fnFzzy7YYdCs0NVjBdqLmUzvBFJjdf3V0bHuo"

# Korean gateway (30) has the pros
GATEWAY_KOREA = 30

# Output directory
OUTPUT_DIR = Path("/home/campesino/workspace/gosu-unveiled/data/to_ingest")


def get_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept-Profile": "public",
        "Content-Type": "application/json",
    }


def fetch_top_players(limit=50):
    """Fetch top players from rankings."""
    url = f"{SUPABASE_URL}/rest/v1/rankings_view"
    params = {
        "select": "standing,rating,wins,losses,disconnects,avatar,race,rank,alias,gateway,gateway_name",
        "standing": f"lte.{limit}",
        "order": "standing.asc",
    }
    resp = requests.get(url, headers=get_headers(), params=params)
    resp.raise_for_status()
    return resp.json()


def fetch_player_matches(gateway, alias, limit=50, offset=0):
    """Fetch match history for a player."""
    url = f"{SUPABASE_URL}/rest/v1/player_matches"
    params = {
        "select": "*",
        "gateway": f"eq.{gateway}",
        "alias": f"eq.{alias}",
        "order": "timestamp.desc",
        "limit": limit,
        "offset": offset,
    }
    resp = requests.get(url, headers=get_headers(), params=params)
    resp.raise_for_status()
    return resp.json()


def download_replay(replay_url, output_path):
    """Download a replay file."""
    if output_path.exists():
        print(f"  Already exists: {output_path.name}")
        return False

    resp = requests.get(replay_url)
    if resp.status_code == 200:
        output_path.write_bytes(resp.content)
        print(f"  Downloaded: {output_path.name}")
        return True
    else:
        print(f"  Failed ({resp.status_code}): {replay_url}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Also save metadata
    metadata_file = OUTPUT_DIR / "metadata.jsonl"

    print("Fetching top ladder players...")
    top_players = fetch_top_players(limit=200)  # Get top 200 players

    print(f"\nFound {len(top_players)} players:")
    for p in top_players[:10]:
        print(f"  #{p['standing']:3d} {p['alias']:20s} ({p['race'][0].upper()}) Rating: {p['rating']}")

    total_downloaded = 0

    for player in top_players:
        alias = player['alias']
        gateway = player['gateway']
        rating = player['rating']
        race = player['race']

        print(f"\n{'='*60}")
        print(f"Fetching matches for {alias} (Rating: {rating}, {race})")
        print('='*60)

        try:
            matches = fetch_player_matches(gateway, alias, limit=100)  # API caps at 100
            print(f"Found {len(matches)} matches")

            for match in matches:
                replay_url = match.get('replay_url')
                if not replay_url:
                    continue

                # Create filename with metadata
                match_id = replay_url.split('/')[-1].replace('.rep', '')
                opponent = match.get('opponent_alias', 'unknown')
                matchup = match.get('matchup', 'unknown')
                result = match.get('result', 'unknown')
                map_name = match.get('map_file_name', 'unknown').replace('/', '_')

                filename = f"{alias}_vs_{opponent}_{matchup}_{result}_{match_id}.rep"
                output_path = OUTPUT_DIR / filename

                if download_replay(replay_url, output_path):
                    total_downloaded += 1

                    # Save metadata
                    with open(metadata_file, 'a') as f:
                        meta = {
                            'filename': filename,
                            'player': alias,
                            'player_race': race,
                            'opponent': opponent,
                            'opponent_race': match.get('opponent_race'),
                            'matchup': matchup,
                            'result': result,
                            'rating': rating,
                            'mmr': match.get('mmr'),
                            'map': map_name,
                            'timestamp': match.get('timestamp'),
                            'replay_url': replay_url,
                        }
                        f.write(json.dumps(meta) + '\n')

                # Be nice to the server
                time.sleep(0.2)

            time.sleep(0.5)

        except Exception as e:
            print(f"Error fetching {alias}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Done! Downloaded {total_downloaded} replays to {OUTPUT_DIR}")
    print(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    main()
