#!/usr/bin/env python3
"""
Data ingestion pipeline for StarCraft: Brood War replays.
Scans directories, extracts metadata, and stores in SQLite for analysis.
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

SCREP_PATH = Path.home() / "go/bin/screp"
DB_PATH = Path(__file__).parent / "data" / "replays.db"
FRAME_MS = 42  # 1 frame = 42 milliseconds


def init_db():
    """Initialize SQLite database with schema."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS replays (
            id INTEGER PRIMARY KEY,
            file_hash TEXT UNIQUE,
            file_path TEXT,
            file_name TEXT,
            source_dir TEXT,
            map_name TEXT,
            game_date TEXT,
            duration_seconds INTEGER,
            frames INTEGER,
            version TEXT,
            created_at TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY,
            replay_id INTEGER,
            slot_id INTEGER,
            player_name TEXT,
            race TEXT,
            is_human INTEGER,
            FOREIGN KEY (replay_id) REFERENCES replays(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS player_aliases (
            id INTEGER PRIMARY KEY,
            canonical_name TEXT,
            alias TEXT UNIQUE,
            confidence REAL,
            source TEXT
        )
    ''')

    # Index for fast lookups
    c.execute('CREATE INDEX IF NOT EXISTS idx_player_name ON players(player_name)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_game_date ON replays(game_date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_alias ON player_aliases(alias)')

    conn.commit()
    return conn


def file_hash(path: Path) -> str:
    """Get MD5 hash of file for deduplication."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_replay_metadata(replay_path: Path) -> dict:
    """Extract metadata using screp (no commands, just header)."""
    result = subprocess.run(
        [str(SCREP_PATH), "-map", str(replay_path)],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"screp failed: {result.stderr}")
    return json.loads(result.stdout)


def process_replay(replay_path: Path) -> dict:
    """Process a single replay file."""
    try:
        fhash = file_hash(replay_path)
        data = parse_replay_metadata(replay_path)
        header = data.get("Header", {})

        # Extract game date
        start_time = header.get("StartTime", "")
        if start_time:
            # Parse ISO format date
            game_date = start_time.split("T")[0]
        else:
            game_date = None

        # Duration
        frames = header.get("Frames", 0)
        duration_seconds = int(frames * FRAME_MS / 1000)

        # Computed data (winner, positions)
        computed = data.get("Computed", {})
        winner_team = computed.get("WinnerTeam")
        player_descs = computed.get("PlayerDescs", [])

        # Players
        players = []
        for i, p in enumerate(header.get("Players", [])):
            pd = player_descs[i] if i < len(player_descs) else {}
            loc = pd.get("StartLocation", {})
            players.append({
                "slot_id": p.get("SlotID"),
                "name": p.get("Name", "Unknown"),
                "race": p.get("Race", {}).get("Name", "Unknown"),
                "is_human": p.get("Type", {}).get("Name") == "Human",
                "start_x": loc.get("X"),
                "start_y": loc.get("Y"),
                "start_direction": pd.get("StartDirection"),
            })

        return {
            "file_hash": fhash,
            "file_path": str(replay_path.absolute()),
            "file_name": replay_path.name,
            "source_dir": replay_path.parent.name,
            "map_name": header.get("Map", "Unknown"),
            "game_date": game_date,
            "duration_seconds": duration_seconds,
            "frames": frames,
            "version": header.get("Version", "Unknown"),
            "winner_team": winner_team,
            "players": players,
            "error": None
        }
    except Exception as e:
        return {
            "file_path": str(replay_path),
            "error": str(e)
        }


def insert_replay(c, result):
    """Insert a processed replay and its players into the database."""
    c.execute('''
        INSERT OR IGNORE INTO replays
        (file_hash, file_path, file_name, source_dir, map_name,
         game_date, duration_seconds, frames, version, winner_team, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result["file_hash"],
        result["file_path"],
        result["file_name"],
        result["source_dir"],
        result["map_name"],
        result["game_date"],
        result["duration_seconds"],
        result["frames"],
        result["version"],
        result.get("winner_team"),
        datetime.now().isoformat()
    ))

    replay_id = c.lastrowid

    for p in result["players"]:
        c.execute('''
            INSERT INTO players (replay_id, slot_id, player_name, race, is_human,
                                 start_x, start_y, start_direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (replay_id, p["slot_id"], p["name"], p["race"], p["is_human"],
              p.get("start_x"), p.get("start_y"), p.get("start_direction")))

    return replay_id


def ingest_directory(conn, directory: Path, max_workers: int = 4):
    """Ingest all replays from a directory."""
    c = conn.cursor()

    replays = list(directory.rglob("*.rep"))
    print(f"Found {len(replays)} replays in {directory}")

    c.execute("SELECT file_hash FROM replays")
    existing_hashes = {row[0] for row in c.fetchall()}

    processed = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_replay, r): r for r in replays}

        for future in as_completed(futures):
            result = future.result()

            if result.get("error"):
                errors += 1
                if errors <= 5:
                    print(f"  Error: {result['file_path']}: {result['error']}")
                continue

            if result["file_hash"] in existing_hashes:
                skipped += 1
                continue

            insert_replay(c, result)
            processed += 1
            existing_hashes.add(result["file_hash"])

            if processed % 100 == 0:
                conn.commit()
                print(f"  Processed {processed} replays...")

    conn.commit()
    print(f"\nCompleted: {processed} new, {skipped} duplicates, {errors} errors")


def add_known_aliases(conn):
    """Add known player aliases from filenames and external sources."""
    c = conn.cursor()

    # Known pro player aliases (lowercase canonical -> aliases)
    known_aliases = {
        "flash": ["flash", "Flash", "FlaSh", "FLasH", "By.FLasH", "By.Flash", "KT_Flash"],
        "bisu": ["bisu", "Bisu", "BiSu", "STX_Bisu", "sKT_Bisu"],
        "jaedong": ["jaedong", "Jaedong", "JaeDong", "Hwaseung_Jaedong"],
        "stork": ["stork", "Stork", "STX_Stork"],
        "fantasy": ["fantasy", "Fantasy", "FanTaSy", "By.FanTaSy"],
        "rain": ["rain", "Rain", "RaiN"],
        "effort": ["effort", "Effort", "EffOrt"],
        "best": ["best", "Best", "BeSt"],
        "savior": ["savior", "sAviOr", "Savior", "sAvIOr"],
        "july": ["july", "July"],
        "yellow": ["yellow", "Yellow", "YellOw"],
        "nada": ["nada", "NaDa", "Nada"],
        "boxer": ["boxer", "Boxer", "BoxeR", "SlayerS_BoxeR"],
        "scan": ["scan", "Scan", "YB_Scan"],
        "larva": ["larva", "Larva", "JSA_Larva"],
        "soulkey": ["soulkey", "Soulkey", "SoulKey", "Neo.G_Soulkey"],
        "hero": ["hero", "herO", "By.herO", "CJ_herO"],
        "mind": ["mind", "Mind", "InteR.Mind"],
        "jangbi": ["jangbi", "Jangbi", "JangBi"],
        "zero": ["zero", "Zero", "ZerO"],
        "soma": ["soma", "Soma", "SoMa"],
        "light": ["light", "Light"],
        "action": ["action", "Action"],
        "mini": ["mini", "Mini"],
        "shuttle": ["shuttle", "Shuttle"],
    }

    for canonical, aliases in known_aliases.items():
        for alias in aliases:
            c.execute('''
                INSERT OR IGNORE INTO player_aliases (canonical_name, alias, confidence, source)
                VALUES (?, ?, ?, ?)
            ''', (canonical, alias, 1.0, "known_pros"))

    conn.commit()
    print(f"Added {sum(len(a) for a in known_aliases.values())} known aliases")


def stats(conn):
    """Print database statistics."""
    c = conn.cursor()

    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)

    c.execute("SELECT COUNT(*) FROM replays")
    print(f"Total replays: {c.fetchone()[0]}")

    c.execute("SELECT COUNT(DISTINCT player_name) FROM players WHERE is_human=1")
    print(f"Unique player names: {c.fetchone()[0]}")

    c.execute("SELECT MIN(game_date), MAX(game_date) FROM replays WHERE game_date IS NOT NULL")
    row = c.fetchone()
    print(f"Date range: {row[0]} to {row[1]}")

    c.execute("""
        SELECT game_date, COUNT(*) as cnt
        FROM replays
        WHERE game_date IS NOT NULL
        GROUP BY substr(game_date, 1, 4)
        ORDER BY game_date
    """)
    print("\nGames by year:")
    for row in c.fetchall():
        print(f"  {row[0][:4]}: {row[1]}")

    c.execute("""
        SELECT player_name, COUNT(*) as cnt
        FROM players
        WHERE is_human=1
        GROUP BY player_name
        ORDER BY cnt DESC
        LIMIT 20
    """)
    print("\nTop 20 players by game count:")
    for row in c.fetchall():
        print(f"  {row[0]}: {row[1]}")

    c.execute("SELECT COUNT(*) FROM player_aliases")
    print(f"\nKnown aliases: {c.fetchone()[0]}")


def ingest_new(conn, to_ingest_dir: Path, dest_dir: Path, max_workers: int = 4):
    """Ingest replays from to_ingest dir, then move them to dest dir."""
    replays = list(to_ingest_dir.rglob("*.rep"))
    if not replays:
        print(f"No new replays in {to_ingest_dir}")
        return

    print(f"Found {len(replays)} new replays to ingest")
    dest_dir.mkdir(parents=True, exist_ok=True)

    c = conn.cursor()
    c.execute("SELECT file_hash FROM replays")
    existing_hashes = {row[0] for row in c.fetchall()}

    processed = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_replay, r): r for r in replays}

        for future in as_completed(futures):
            replay_path = futures[future]
            result = future.result()
            dest = dest_dir / replay_path.name

            if result.get("error"):
                errors += 1
                if errors <= 5:
                    print(f"  Error: {result['file_path']}: {result['error']}")
            elif result["file_hash"] in existing_hashes:
                skipped += 1
            else:
                # Update file_path to destination before inserting
                result["file_path"] = str(dest.absolute())
                result["source_dir"] = dest_dir.name
                insert_replay(c, result)
                processed += 1
                existing_hashes.add(result["file_hash"])

                if processed % 100 == 0:
                    conn.commit()
                    print(f"  Processed {processed} replays...")

            # Move file to dest (or delete if duplicate name)
            if not dest.exists():
                replay_path.rename(dest)
            else:
                replay_path.unlink()

    conn.commit()
    print(f"Completed: {processed} new, {skipped} duplicates, {errors} errors")


def main():
    print("Initializing database...")
    conn = init_db()

    # Add known aliases
    add_known_aliases(conn)

    data_dir = Path(__file__).parent / "data"
    to_ingest_dir = data_dir / "to_ingest"

    if to_ingest_dir.exists() and any(to_ingest_dir.rglob("*.rep")):
        # Fast path: only process new replays
        print(f"\n{'='*60}")
        print(f"Ingesting new replays from: {to_ingest_dir}")
        print("="*60)
        ingest_new(conn, to_ingest_dir, data_dir / "replays")
    else:
        # Full re-ingest from consolidated replays dir
        replays_dir = data_dir / "replays"
        if replays_dir.exists():
            print(f"\n{'='*60}")
            print(f"Ingesting: {replays_dir}")
            print("="*60)
            ingest_directory(conn, replays_dir)

    stats(conn)
    conn.close()
    print(f"\nDatabase saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
