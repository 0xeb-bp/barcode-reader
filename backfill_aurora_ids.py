#!/usr/bin/env python3
"""
Migration script: Backfill aurora_ids and seed player_identities table.

Tier 1 (no API calls):
  - Populate match_id on all replays from filenames
  - Seed player_identities from player_aliases aurora_ids
  - UPDATE players.aurora_id for all known aliases by player_name

Tier 2 (API calls, no replay downloads):
  - For each labeled player, fetch match history from cwal.gg
  - Match by match_id to local replays
  - Update opponent aurora_id on players table

Usage:
  python backfill_aurora_ids.py          # Tier 1 only (safe, offline)
  python backfill_aurora_ids.py --api    # Tier 1 + Tier 2 (makes API calls)
"""

import argparse
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "replays.db"


def extract_match_id(file_name: str) -> str:
    """Extract MM-XXXXXXXX match_id from replay filename."""
    m = re.search(r'(MM-[0-9A-Fa-f-]+)', file_name)
    return m.group(1) if m else None


def tier1_backfill_match_ids(conn):
    """Populate match_id on all replays from filenames."""
    c = conn.cursor()
    c.execute("SELECT id, file_name FROM replays WHERE match_id IS NULL")
    rows = c.fetchall()
    updated = 0
    for replay_id, file_name in rows:
        match_id = extract_match_id(file_name)
        if match_id:
            c.execute("UPDATE replays SET match_id = ? WHERE id = ?", (match_id, replay_id))
            updated += 1
    conn.commit()
    print(f"  match_id backfilled on {updated}/{len(rows)} replays")
    return updated


def tier1_seed_identities(conn):
    """Seed player_identities from player_aliases that have aurora_ids."""
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT canonical_name, aurora_id
        FROM player_aliases
        WHERE aurora_id IS NOT NULL
    """)
    alias_rows = c.fetchall()

    inserted = 0
    for canonical, aurora_id in alias_rows:
        c.execute("SELECT id FROM player_identities WHERE aurora_id = ?", (aurora_id,))
        if c.fetchone():
            continue
        c.execute("""
            INSERT INTO player_identities (canonical_name, aurora_id, source, created_at)
            VALUES (?, ?, 'backfill_from_aliases', ?)
        """, (canonical, aurora_id, datetime.now().isoformat()))
        inserted += 1

    conn.commit()
    print(f"  Seeded {inserted} player_identities ({len(alias_rows)} alias aurora_ids found)")
    return inserted


def tier1_backfill_players_by_name(conn):
    """Update players.aurora_id for all known aliases by player_name match."""
    c = conn.cursor()
    # Get all aliases that have aurora_ids
    c.execute("""
        SELECT alias, aurora_id
        FROM player_aliases
        WHERE aurora_id IS NOT NULL
    """)
    alias_map = c.fetchall()

    total_updated = 0
    for alias, aurora_id in alias_map:
        c.execute("""
            UPDATE players SET aurora_id = ?
            WHERE player_name = ? AND aurora_id IS NULL
        """, (aurora_id, alias))
        total_updated += c.rowcount

    conn.commit()
    print(f"  Updated {total_updated} player rows by name match ({len(alias_map)} aliases)")
    return total_updated


def tier2_api_backfill(conn):
    """Fetch match history for labeled players, match to local replays, update opponent aurora_ids."""
    # Import cwal API functions
    sys.path.insert(0, str(Path(__file__).parent))
    from cwal import api_matches_all, GATEWAY_NAMES

    c = conn.cursor()

    # Get all identities and their known aliases
    c.execute("""
        SELECT pi.canonical_name, pi.aurora_id, pa.alias
        FROM player_identities pi
        JOIN player_aliases pa ON pa.aurora_id = pi.aurora_id
    """)
    identity_aliases = {}
    for canonical, aurora_id, alias in c.fetchall():
        if canonical not in identity_aliases:
            identity_aliases[canonical] = {"aurora_id": aurora_id, "aliases": []}
        identity_aliases[canonical]["aliases"].append(alias)

    print(f"\n  Fetching match history for {len(identity_aliases)} players...")
    total_updated = 0
    total_api_calls = 0

    for canonical, info in sorted(identity_aliases.items()):
        # Use the first alias for the API query
        alias = info["aliases"][0]

        # Try all gateways
        for gateway in [30, 10, 20, 45, 11]:
            try:
                matches = api_matches_all(alias, gateway=gateway, limit=999)
                total_api_calls += 1
            except Exception as e:
                print(f"    {canonical} ({alias}, gw={gateway}): API error: {e}")
                continue

            if not matches:
                continue

            player_updated = 0
            for m in matches:
                # Extract match_id from replay_url
                replay_url = m.get("replay_url", "")
                if not replay_url:
                    continue
                match_id = replay_url.split("/")[-1].replace(".rep", "")
                if not match_id:
                    continue

                # Find this match in our DB
                c.execute("SELECT id FROM replays WHERE match_id = ?", (match_id,))
                replay_row = c.fetchone()
                if not replay_row:
                    continue
                replay_id = replay_row[0]

                # Update opponent aurora_id
                opponent_alias = m.get("opponent_alias")
                opponent_aurora_id = m.get("opponent_aurora_id")
                if opponent_alias and opponent_aurora_id:
                    c.execute("""
                        UPDATE players SET aurora_id = ?
                        WHERE replay_id = ? AND player_name = ? AND aurora_id IS NULL
                    """, (opponent_aurora_id, replay_id, opponent_alias))
                    player_updated += c.rowcount

                # Also update our own aurora_id if missing
                player_aurora_id = m.get("aurora_id")
                player_alias = m.get("alias")
                if player_alias and player_aurora_id:
                    c.execute("""
                        UPDATE players SET aurora_id = ?
                        WHERE replay_id = ? AND player_name = ? AND aurora_id IS NULL
                    """, (player_aurora_id, replay_id, player_alias))
                    player_updated += c.rowcount

            if player_updated > 0:
                print(f"    {canonical} ({alias}, gw={gateway}): {player_updated} rows updated from {len(matches)} matches")
                total_updated += player_updated
            break  # Found matches on this gateway, no need to try others

            time.sleep(0.5)

        conn.commit()

    print(f"\n  Tier 2: {total_updated} player rows updated via {total_api_calls} API calls")
    return total_updated


def tier3_direct_lookup(conn):
    """Look up aurora_id for every player name that's still missing one.

    Uses batch queries against the cwal.gg players table (Supabase 'in' operator)
    instead of per-name API calls, reducing ~3000 individual requests to ~16 batches.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    import requests
    from cwal import SUPABASE_URL, get_headers

    c = conn.cursor()

    # Get all distinct player names missing aurora_id (modern era)
    c.execute("""
        SELECT DISTINCT p.player_name
        FROM players p
        JOIN replays r ON r.id = p.replay_id
        WHERE p.aurora_id IS NULL
          AND p.is_human = 1
          AND r.game_date >= '2025-01-01'
        ORDER BY p.player_name
    """)
    missing_names = [row[0] for row in c.fetchall()]
    print(f"\n  {len(missing_names)} unique player names missing aurora_id")

    # Filter names with chars that break Supabase 'in' operator
    safe_names = [n for n in missing_names if ',' not in n and '(' not in n and ')' not in n]
    skipped = len(missing_names) - len(safe_names)
    if skipped:
        print(f"  Skipping {skipped} names with special characters")

    # Batch lookup against players table (all gateways, non-null battlenet_account)
    BATCH_SIZE = 200
    url = f"{SUPABASE_URL}/rest/v1/players"
    resolved = {}  # alias -> aurora_id

    for batch_start in range(0, len(safe_names), BATCH_SIZE):
        batch = safe_names[batch_start:batch_start + BATCH_SIZE]
        names_str = ",".join(batch)
        params = {
            "select": "alias,battlenet_account",
            "battlenet_account": "not.is.null",
            "alias": f"in.({names_str})",
            "limit": 5000,
        }
        try:
            resp = requests.get(url, headers=get_headers(), params=params, timeout=30)
            resp.raise_for_status()
            for row in resp.json():
                alias = row["alias"]
                aid = row["battlenet_account"]
                if alias not in resolved and aid is not None:
                    resolved[alias] = aid
        except Exception as e:
            print(f"  Batch {batch_start}-{batch_start+len(batch)} error: {e}")

        print(f"  Batch {batch_start//BATCH_SIZE + 1}/{(len(safe_names) + BATCH_SIZE - 1)//BATCH_SIZE}: "
              f"queried {min(batch_start + BATCH_SIZE, len(safe_names))}/{len(safe_names)}, "
              f"{len(resolved)} found so far")
        sys.stdout.flush()
        time.sleep(0.3)

    # Apply resolved aurora_ids to players table
    total_updated = 0
    for alias, aurora_id in resolved.items():
        c.execute("""
            UPDATE players SET aurora_id = ?
            WHERE player_name = ? AND aurora_id IS NULL
        """, (aurora_id, alias))
        total_updated += c.rowcount

    conn.commit()
    print(f"\n  Tier 3: {len(resolved)} names resolved, {total_updated} rows updated "
          f"({len(safe_names)} queried in {(len(safe_names) + BATCH_SIZE - 1)//BATCH_SIZE} batches)")
    return total_updated


def tier4_match_lookup(conn):
    """Look up aurora_ids via match_id for replays where players are still missing IDs.

    Queries the cwal.gg player_matches view by replay_url (which contains the match_id).
    Each match row returns aurora_id for both players, so even opponents not on
    the ladder can be resolved if they appear in any recorded match.
    """
    import requests
    from cwal import SUPABASE_URL, get_headers

    c = conn.cursor()

    # Get match_ids for replays that still have missing aurora_ids
    c.execute("""
        SELECT DISTINCT r.match_id
        FROM replays r
        JOIN players p ON p.replay_id = r.id
        WHERE r.match_id IS NOT NULL
          AND p.aurora_id IS NULL
          AND p.is_human = 1
          AND r.game_date >= '2025-01-01'
    """)
    match_ids = [row[0] for row in c.fetchall()]
    print(f"\n  {len(match_ids)} replays with missing player aurora_ids")

    if not match_ids:
        return 0

    # Build a map of match_id -> replay_id for fast local lookup
    c.execute("SELECT match_id, id FROM replays WHERE match_id IS NOT NULL")
    match_to_replay = dict(c.fetchall())

    url = f"{SUPABASE_URL}/rest/v1/player_matches"
    BATCH_SIZE = 50
    total_updated = 0
    api_hits = 0

    for batch_start in range(0, len(match_ids), BATCH_SIZE):
        batch = match_ids[batch_start:batch_start + BATCH_SIZE]
        replay_urls = [f"https://replays.cwal.gg/{mid}.rep" for mid in batch]
        urls_str = ",".join(replay_urls)

        params = {
            "select": "alias,aurora_id,opponent_alias,opponent_aurora_id,replay_url",
            "replay_url": f"in.({urls_str})",
            "limit": 5000,
        }
        try:
            resp = requests.get(url, headers=get_headers(), params=params, timeout=60)
            resp.raise_for_status()
            rows = resp.json()
            api_hits += len(rows)
        except Exception as e:
            print(f"  Batch error at {batch_start}: {e}")
            time.sleep(1)
            continue

        # Apply both player and opponent aurora_ids
        for row in rows:
            # Extract match_id from replay_url
            replay_url = row.get("replay_url", "")
            mid = replay_url.split("/")[-1].replace(".rep", "")
            replay_id = match_to_replay.get(mid)
            if not replay_id:
                continue

            # Update player
            if row.get("aurora_id"):
                c.execute("""
                    UPDATE players SET aurora_id = ?
                    WHERE replay_id = ? AND player_name = ? AND aurora_id IS NULL
                """, (row["aurora_id"], replay_id, row["alias"]))
                total_updated += c.rowcount

            # Update opponent
            if row.get("opponent_aurora_id"):
                c.execute("""
                    UPDATE players SET aurora_id = ?
                    WHERE replay_id = ? AND player_name = ? AND aurora_id IS NULL
                """, (row["opponent_aurora_id"], replay_id, row["opponent_alias"]))
                total_updated += c.rowcount

        num_batches = (len(match_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_start//BATCH_SIZE + 1}/{num_batches}: "
              f"queried {min(batch_start + BATCH_SIZE, len(match_ids))}/{len(match_ids)}, "
              f"{total_updated} rows updated so far")
        sys.stdout.flush()

        if batch_start % (BATCH_SIZE * 5) == 0 and batch_start > 0:
            conn.commit()
        time.sleep(0.5)

    conn.commit()
    print(f"\n  Tier 4: {total_updated} rows updated from {api_hits} API match rows "
          f"({len(match_ids)} replays queried in {(len(match_ids) + BATCH_SIZE - 1)//BATCH_SIZE} batches)")
    return total_updated


def print_coverage_report(conn):
    """Print aurora_id coverage stats."""
    c = conn.cursor()
    print("\n" + "=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)

    # Overall
    c.execute("SELECT COUNT(*) FROM players WHERE aurora_id IS NOT NULL")
    with_aid = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM players WHERE is_human = 1")
    total_human = c.fetchone()[0]
    print(f"\nPlayers with aurora_id: {with_aid}/{total_human} ({with_aid/total_human:.1%})")

    # Modern era
    c.execute("""
        SELECT COUNT(*) FROM players p
        JOIN replays r ON r.id = p.replay_id
        WHERE p.aurora_id IS NOT NULL AND r.game_date >= '2025-01-01' AND p.is_human = 1
    """)
    modern_aid = c.fetchone()[0]
    c.execute("""
        SELECT COUNT(*) FROM players p
        JOIN replays r ON r.id = p.replay_id
        WHERE r.game_date >= '2025-01-01' AND p.is_human = 1
    """)
    modern_total = c.fetchone()[0]
    print(f"Modern-era with aurora_id: {modern_aid}/{modern_total} ({modern_aid/modern_total:.1%})")

    # player_identities
    c.execute("SELECT COUNT(*) FROM player_identities")
    print(f"Player identities: {c.fetchone()[0]}")

    # match_id coverage
    c.execute("SELECT COUNT(*) FROM replays WHERE match_id IS NOT NULL")
    with_mid = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM replays")
    total_replays = c.fetchone()[0]
    print(f"Replays with match_id: {with_mid}/{total_replays} ({with_mid/total_replays:.1%})")

    # Per-player aurora_id coverage for labeled players
    print("\nLabeled player aurora_id coverage (modern era):")
    c.execute("""
        SELECT pi.canonical_name,
               COUNT(DISTINCT p.replay_id) as total_games,
               COUNT(DISTINCT CASE WHEN p.aurora_id IS NOT NULL THEN p.replay_id END) as games_with_aid
        FROM player_identities pi
        JOIN players p ON p.aurora_id = pi.aurora_id
        JOIN replays r ON r.id = p.replay_id
        WHERE r.game_date >= '2025-01-01' AND p.is_human = 1
        GROUP BY pi.canonical_name
        ORDER BY total_games DESC
    """)
    for canonical, total, with_aid in c.fetchall():
        pct = with_aid / total * 100 if total > 0 else 0
        print(f"  {canonical:20s} {with_aid:4d}/{total:4d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Backfill aurora_ids and seed player_identities")
    parser.add_argument("--api", action="store_true", help="Run Tier 2 (match history backfill)")
    parser.add_argument("--direct", action="store_true", help="Run Tier 3 (direct aurora_id lookup for all missing players)")
    parser.add_argument("--matches", action="store_true", help="Run Tier 4 (match-based aurora_id lookup by replay match_id)")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)

    # Ensure schema is up to date
    from ingest_replays import init_db as _init
    _init_conn = sqlite3.connect(DB_PATH)
    c = _init_conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS player_identities (
            id INTEGER PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            aurora_id INTEGER NOT NULL UNIQUE,
            source TEXT,
            notes TEXT,
            created_at TEXT
        )
    ''')
    try:
        c.execute("ALTER TABLE replays ADD COLUMN match_id TEXT")
    except sqlite3.OperationalError:
        pass
    c.execute('CREATE INDEX IF NOT EXISTS idx_player_aurora_id ON players(aurora_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_replay_match_id ON replays(match_id)')
    _init_conn.commit()
    _init_conn.close()

    # Re-open connection to get updated schema
    conn = sqlite3.connect(DB_PATH)

    print("=" * 60)
    print("TIER 1: Offline backfill (no API calls)")
    print("=" * 60)

    print("\n1. Backfilling match_id from filenames...")
    tier1_backfill_match_ids(conn)

    print("\n2. Seeding player_identities from player_aliases...")
    tier1_seed_identities(conn)

    print("\n3. Backfilling players.aurora_id by name match...")
    tier1_backfill_players_by_name(conn)

    if args.api:
        print("\n" + "=" * 60)
        print("TIER 2: API-based backfill")
        print("=" * 60)
        tier2_api_backfill(conn)

    if args.direct:
        print("\n" + "=" * 60)
        print("TIER 3: Direct aurora_id lookup for missing players")
        print("=" * 60)
        tier3_direct_lookup(conn)

    if args.matches:
        print("\n" + "=" * 60)
        print("TIER 4: Match-based aurora_id lookup by replay match_id")
        print("=" * 60)
        tier4_match_lookup(conn)

    print_coverage_report(conn)
    conn.close()


if __name__ == "__main__":
    main()
