#!/usr/bin/env python3
"""cwal.gg ladder API tool for Barcode Reader."""

import argparse
import json
import re
import sqlite3
import time
from datetime import datetime, timedelta

import requests
from pathlib import Path

# cwal.gg Supabase config (public anon key)
SUPABASE_URL = "https://xmploueumzkrdvapbyfs.supabase.co"
SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhtcGxvdWV1bXprcmR2YXBieWZzIiwi"
    "cm9sZSI6ImFub24iLCJpYXQiOjE2NzI4ODY5MTQsImV4cCI6MTk4ODQ2MjkxNH0."
    "p8Jkm2fnFzzy7YYdCs0NVjBdqLmUzvBFJjdf3V0bHuo"
)

DEFAULT_GATEWAY = 30  # Korea
DEFAULT_LIMIT = 50
PAGE_SIZE = 50  # Supabase page size
API_PAGE_DELAY = 0.5  # seconds between API pages
DOWNLOAD_DELAY = 0.15  # seconds between replay downloads

GATEWAY_NAMES = {10: "US West", 11: "US West", 20: "US East", 30: "Korea", 45: "Europe"}

OUTPUT_DIR = Path(__file__).parent / "data" / "to_ingest"
DB_PATH = Path(__file__).parent / "data" / "replays.db"
LEDGER_PATH = Path(__file__).parent / "docs" / "scrape_ledger.md"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept-Profile": "public",
        "Content-Type": "application/json",
    }


def parse_duration(s):
    """Parse 'MM:SS' or 'HH:MM:SS' into seconds."""
    parts = s.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise argparse.ArgumentTypeError(f"Invalid duration: {s!r} (use MM:SS or HH:MM:SS)")


def format_duration(seconds):
    """Format duration (seconds) as M:SS."""
    if seconds is None:
        return "—"
    seconds = int(float(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"


def clean_map_name(raw):
    """Strip BW color codes (control chars) from map names."""
    if not raw:
        return None
    return re.sub(r'[\x00-\x1f]', '', raw)


def get_map_display(match):
    """Get clean map name from match data."""
    # Prefer map_file_name (e.g. "(4)Pole Star 1.1.scx"), strip path prefix and .scx
    mfn = match.get("map_file_name") or ""
    if mfn:
        # Strip .scx/.scm extension
        for ext in (".scx", ".scm"):
            if mfn.lower().endswith(ext):
                mfn = mfn[:-len(ext)]
        # Strip leading (N) player count
        mfn = re.sub(r'^\(\d+\)', '', mfn).strip()
        if mfn:
            return mfn
    # Fallback to cleaned map_name
    return clean_map_name(match.get("map_name")) or "—"


def format_date(ts):
    """Format ISO timestamp as YYYY-MM-DD."""
    if not ts:
        return "—"
    return ts[:10]


# ---------------------------------------------------------------------------
# API functions (return raw JSON, reusable from other scripts)
# ---------------------------------------------------------------------------

def api_match_count(alias, gateway=DEFAULT_GATEWAY):
    """Get total match count for a player (HEAD request with count=exact)."""
    url = f"{SUPABASE_URL}/rest/v1/player_matches"
    params = {
        "alias": f"eq.{alias}",
        "gateway": f"eq.{gateway}",
        "limit": 1,
    }
    headers = get_headers()
    headers["Prefer"] = "count=exact"
    resp = requests.head(url, headers=headers, params=params)
    resp.raise_for_status()
    cr = resp.headers.get("content-range", "")
    # content-range: 0-0/22 or */0
    if "/" in cr:
        return int(cr.split("/")[1])
    return 0


def api_matches(alias, gateway=DEFAULT_GATEWAY, limit=PAGE_SIZE, offset=0):
    """Fetch match history page for a player."""
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


def api_matches_all(alias, gateway=DEFAULT_GATEWAY, limit=DEFAULT_LIMIT):
    """Fetch up to `limit` matches, auto-paginating."""
    results = []
    offset = 0
    while len(results) < limit:
        page_size = min(PAGE_SIZE, limit - len(results))
        page = api_matches(alias, gateway, limit=page_size, offset=offset)
        if not page:
            break
        results.extend(page)
        if len(page) < page_size:
            break
        offset += len(page)
        time.sleep(API_PAGE_DELAY)
    return results[:limit]


def api_rankings(gateway=DEFAULT_GATEWAY, limit=DEFAULT_LIMIT):
    """Fetch ladder rankings."""
    url = f"{SUPABASE_URL}/rest/v1/rankings_view"
    params = {
        "select": "standing,rating,wins,losses,disconnects,avatar,race,rank,alias,gateway,gateway_name",
        "gateway": f"eq.{gateway}",
        "standing": f"lte.{limit}",
        "order": "standing.asc",
    }
    resp = requests.get(url, headers=get_headers(), params=params)
    resp.raise_for_status()
    return resp.json()


def api_search(query, gateway=DEFAULT_GATEWAY):
    """Search players by alias (ilike)."""
    url = f"{SUPABASE_URL}/rest/v1/rankings_view"
    params = {
        "select": "standing,rating,wins,losses,race,rank,alias,gateway,gateway_name",
        "alias": f"ilike.*{query}*",
        "gateway": f"eq.{gateway}",
        "order": "standing.asc",
        "limit": 50,
    }
    resp = requests.get(url, headers=get_headers(), params=params)
    resp.raise_for_status()
    return resp.json()


def api_aurora_id(alias, gateway=DEFAULT_GATEWAY):
    """Look up aurora_id for a player alias from their match history."""
    url = f"{SUPABASE_URL}/rest/v1/player_matches"
    params = {
        "select": "aurora_id",
        "alias": f"eq.{alias}",
        "gateway": f"eq.{gateway}",
        "limit": 1,
    }
    resp = requests.get(url, headers=get_headers(), params=params)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return None
    return rows[0]["aurora_id"]


def api_handles(battlenet_account):
    """Fetch all aliases linked to a battlenet_account ID."""
    url = f"{SUPABASE_URL}/rest/v1/players"
    params = {
        "select": "standing,rating,wins,losses,race,rank,alias,gateway,battlenet_account",
        "battlenet_account": f"eq.{battlenet_account}",
        "order": "standing.asc",
    }
    resp = requests.get(url, headers=get_headers(), params=params)
    resp.raise_for_status()
    return resp.json()


def api_matches_since(alias, gateway=DEFAULT_GATEWAY, since=None, until=None):
    """Fetch all matches for an alias since a given date. Auto-paginates."""
    results = []
    offset = 0
    while True:
        # Use list of tuples to allow duplicate 'timestamp' params for range queries
        params = [
            ("select", "*"),
            ("gateway", f"eq.{gateway}"),
            ("alias", f"eq.{alias}"),
            ("order", "timestamp.desc"),
            ("limit", PAGE_SIZE),
            ("offset", offset),
        ]
        if since:
            params.append(("timestamp", f"gte.{since}"))
        if until:
            params.append(("timestamp", f"lte.{until}"))
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/player_matches",
            headers=get_headers(), params=params)
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        results.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += len(page)
        time.sleep(API_PAGE_DELAY)
    return results


# ---------------------------------------------------------------------------
# Shared download helper
# ---------------------------------------------------------------------------

def extract_replay_match_id(replay_url):
    """Extract MM-XXXXXXXX match_id from a replay URL."""
    return replay_url.split("/")[-1].replace(".rep", "")


def download_matches(matches, output_dir, existing_match_ids=None, dry_run=False):
    """Download replay files from a list of API match dicts.

    Args:
        matches: list of match dicts from the cwal API
        output_dir: Path to download directory
        existing_match_ids: set of MM-... match_ids already in DB (skip these)
        dry_run: if True, only print what would be downloaded

    Returns:
        (downloaded, skipped) counts
    """
    downloadable = [m for m in matches if m.get("replay_url")]
    if not downloadable:
        return 0, 0

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "_metadata.jsonl"
    downloaded = 0
    skipped = 0

    for m in downloadable:
        replay_url = m["replay_url"]
        match_id = extract_replay_match_id(replay_url)

        # DB dedup
        if existing_match_ids and match_id in existing_match_ids:
            skipped += 1
            continue

        alias = m.get("alias", "unknown")
        opponent = m.get("opponent_alias", "unknown")
        matchup = m.get("matchup", "unknown")
        result = m.get("result", "unknown")
        filename = f"{alias}_vs_{opponent}_{matchup}_{result}_{match_id}.rep"
        output_path = output_dir / filename

        if dry_run:
            if output_path.exists():
                skipped += 1
            else:
                downloaded += 1
                print(f"  [GET] {filename}")
            continue

        # Filesystem dedup
        if output_path.exists():
            skipped += 1
            continue

        resp = requests.get(replay_url)
        if resp.status_code == 200:
            output_path.write_bytes(resp.content)
            downloaded += 1
            print(f"  Downloaded: {filename}")

            meta = {
                "match_id": m.get("id"),
                "file_name": filename,
                "alias": alias,
                "aurora_id": m.get("aurora_id"),
                "opponent_alias": m.get("opponent_alias"),
                "opponent_aurora_id": m.get("opponent_aurora_id"),
                "gateway": m.get("gateway"),
            }
            with open(metadata_path, "a") as f:
                f.write(json.dumps(meta) + "\n")
        else:
            print(f"  Failed ({resp.status_code}): {filename}")

        time.sleep(DOWNLOAD_DELAY)

    return downloaded, skipped


def load_existing_match_ids():
    """Load all match_ids from the DB for dedup."""
    conn = sqlite3.connect(DB_PATH)
    ids = {row[0] for row in conn.execute(
        "SELECT match_id FROM replays WHERE match_id IS NOT NULL")}
    conn.close()
    return ids


def append_ledger(command, players, new, skipped, notes=""):
    """Append a row to the scrape ledger."""
    today = datetime.now().strftime("%Y-%m-%d")
    if not LEDGER_PATH.exists():
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEDGER_PATH.write_text(
            "# Scrape Ledger\n\n"
            "| Date | Command | Players | New | Skipped | Notes |\n"
            "|------|---------|---------|-----|---------|-------|\n"
        )
    with open(LEDGER_PATH, "a") as f:
        f.write(f"| {today} | {command} | {players} | {new} | {skipped} | {notes} |\n")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_matches(args):
    """Search a player's match history with optional filters."""
    print(f"Fetching matches for {args.alias} (gateway={args.gateway})...")
    matches = api_matches_all(args.alias, gateway=args.gateway, limit=args.limit)

    if not matches:
        print("No matches found.")
        return

    # Client-side filters
    filtered = matches
    if args.map:
        map_lower = args.map.lower()
        filtered = [m for m in filtered if map_lower in get_map_display(m).lower()]
    if args.matchup:
        mu = args.matchup.lower()
        filtered = [m for m in filtered if (m.get("matchup") or "").lower() == mu]
    if args.opponent:
        opp_lower = args.opponent.lower()
        filtered = [m for m in filtered if opp_lower in (m.get("opponent_alias") or "").lower()]
    if args.min_duration:
        min_sec = parse_duration(args.min_duration)
        filtered = [m for m in filtered if (m.get("duration") or 0) >= min_sec]
    if args.max_duration:
        max_sec = parse_duration(args.max_duration)
        filtered = [m for m in filtered if (m.get("duration") or 9999999) <= max_sec]

    if not filtered:
        print(f"No matches after filtering ({len(matches)} total fetched).")
        return

    # Print table
    print(f"\n{len(filtered)} matches (of {len(matches)} fetched):\n")
    header = f"{'Date':<12} {'Map':<20} {'Duration':>8} {'Matchup':<6} {'Opponent':<20} {'Result':<7} {'MMR':>5}"
    print(header)
    print("—" * len(header))
    for m in filtered:
        date = format_date(m.get("timestamp"))
        map_name = get_map_display(m)[:20]
        dur = format_duration(m.get("duration"))
        matchup = m.get("matchup") or "—"
        opponent = (m.get("opponent_alias") or "—")[:20]
        result = m.get("result") or "—"
        mmr = m.get("mmr")
        mmr_str = str(mmr) if mmr is not None else "—"
        print(f"{date:<12} {map_name:<20} {dur:>8} {matchup:<6} {opponent:<20} {result:<7} {mmr_str:>5}")


def cmd_scrape(args):
    """Download replays for a player."""
    print(f"Fetching matches for {args.alias} (gateway={args.gateway})...")
    matches = api_matches_all(args.alias, gateway=args.gateway, limit=args.limit)

    if not matches:
        print("No matches found.")
        return

    downloadable = [m for m in matches if m.get("replay_url")]
    print(f"Found {len(downloadable)} matches with replays (of {len(matches)} total).")

    if not downloadable:
        return

    output_dir = Path(args.output)
    downloaded, skipped = download_matches(
        matches, output_dir, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\nDry run — would download to {output_dir}/")
    else:
        print(f"\nDone: {downloaded} downloaded, {skipped} already existed.")


def cmd_rankings(args):
    """Show ladder standings."""
    print(f"Fetching top {args.limit} (gateway={args.gateway})...")
    players = api_rankings(gateway=args.gateway, limit=args.limit)

    if args.race:
        race_lower = args.race.lower()
        players = [p for p in players if (p.get("race") or "").lower().startswith(race_lower)]

    if not players:
        print("No players found.")
        return

    print()
    header = f"{'#':>4} {'Alias':<22} {'Race':<10} {'Rating':>6} {'W-L':<10} {'Gateway'}"
    print(header)
    print("—" * len(header))
    for p in players:
        standing = p.get("standing", "—")
        alias = (p.get("alias") or "—")[:22]
        race = p.get("race") or "—"
        rating = p.get("rating") or "—"
        wins = p.get("wins", 0)
        losses = p.get("losses", 0)
        wl = f"{wins}-{losses}"
        gw_name = p.get("gateway_name") or str(p.get("gateway", "—"))
        print(f"{standing:>4} {alias:<22} {race:<10} {rating:>6} {wl:<10} {gw_name}")


def cmd_search(args):
    """Search for a player by name."""
    print(f"Searching for '{args.query}' (gateway={args.gateway})...")
    players = api_search(args.query, gateway=args.gateway)

    if not players:
        print("No players found.")
        return

    print(f"\n{len(players)} result(s):\n")
    header = f"{'#':>4} {'Alias':<22} {'Race':<10} {'Rating':>6} {'W-L':<10} {'Gateway'}"
    print(header)
    print("—" * len(header))
    for p in players:
        standing = p.get("standing", "—")
        alias = (p.get("alias") or "—")[:22]
        race = p.get("race") or "—"
        rating = p.get("rating") or "—"
        wins = p.get("wins", 0)
        losses = p.get("losses", 0)
        wl = f"{wins}-{losses}"
        gw_name = p.get("gateway_name") or str(p.get("gateway", "—"))
        print(f"{standing:>4} {alias:<22} {race:<10} {rating:>6} {wl:<10} {gw_name}")


def cmd_handles(args):
    """Look up all handles for a player (by alias or battlenet_account ID)."""
    query = args.player
    # If numeric, treat as battlenet_account directly
    if query.isdigit():
        battlenet_account = int(query)
        print(f"Looking up battlenet_account={battlenet_account}...")
    else:
        # Look up aurora_id from alias, trying all gateways
        gateways = [args.gateway] + [g for g in [30, 10, 20, 45, 11] if g != args.gateway]
        battlenet_account = None
        for gw in gateways:
            print(f"Looking up aurora_id for '{query}' (gateway={gw})...")
            battlenet_account = api_aurora_id(query, gateway=gw)
            if battlenet_account is not None:
                print(f"Found battlenet_account={battlenet_account} (gateway {gw})")
                break
        if battlenet_account is None:
            print(f"No matches found for '{query}' on any gateway.")
            return
    handles = api_handles(battlenet_account)

    if not handles:
        print("No handles found for that account.")
        return

    print(f"\n{len(handles)} handle(s):\n")
    header = f"{'#':>4} {'Alias':<22} {'Race':<10} {'Rating':>6} {'W-L':<10} {'Gateway':>10}"
    print(header)
    print("—" * len(header))
    for p in handles:
        standing = p.get("standing", "—")
        alias = (p.get("alias") or "—")[:22]
        race = p.get("race") or "—"
        rating = p.get("rating") or "—"
        wins = p.get("wins", 0)
        losses = p.get("losses", 0)
        wl = f"{wins}-{losses}"
        gw = p.get("gateway", 0)
        gw_name = GATEWAY_NAMES.get(gw, str(gw))
        print(f"{standing:>4} {alias:<22} {race:<10} {rating:>6} {wl:<10} {gw_name:>10}")


def cmd_count(args):
    """Get total match count for a player on cwal.gg."""
    alias = args.player
    gateways = [args.gateway] + [g for g in [30, 10, 20, 45, 11] if g != args.gateway]
    for gw in gateways:
        count = api_match_count(alias, gateway=gw)
        if count > 0:
            print(f"{alias}: {count} games on {GATEWAY_NAMES.get(gw, f'gw {gw}')}")
            return
    print(f"{alias}: 0 games (checked all gateways)")


def cmd_refresh(args):
    """Scrape new games for all labeled players."""
    since = args.since or (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"Refreshing labeled players (games since {since})...")

    existing = load_existing_match_ids()
    print(f"DB has {len(existing)} existing replays")

    conn = sqlite3.connect(DB_PATH)
    identities = conn.execute(
        "SELECT canonical_name, aurora_id FROM player_identities ORDER BY canonical_name"
    ).fetchall()
    conn.close()
    print(f"Found {len(identities)} labeled players\n")

    output_dir = Path(args.output)
    total_new = 0
    total_skipped = 0

    for canonical, aurora_id in identities:
        handles = api_handles(aurora_id)
        if not handles:
            print(f"  {canonical}: no handles found (aurora_id={aurora_id})")
            continue

        player_new = 0
        handle_strs = []

        for h in handles:
            alias = h.get("alias")
            gw = h.get("gateway", DEFAULT_GATEWAY)
            if not alias:
                continue

            try:
                matches = api_matches_since(alias, gateway=gw, since=since)
            except requests.exceptions.HTTPError as e:
                print(f"  {canonical}: API error for {alias}@{GATEWAY_NAMES.get(gw, str(gw))}: {e}")
                continue

            if not matches:
                continue

            downloaded, skipped = download_matches(
                matches, output_dir, existing_match_ids=existing,
                dry_run=args.dry_run)

            if downloaded > 0:
                handle_strs.append(
                    f"{downloaded} from {alias}@{GATEWAY_NAMES.get(gw, str(gw))}")
            player_new += downloaded
            total_skipped += skipped

            # Track downloaded match_ids to prevent re-downloading via other handles
            for m in matches:
                url = m.get("replay_url")
                if url:
                    existing.add(extract_replay_match_id(url))

        if player_new > 0:
            print(f"  {canonical}: {player_new} new ({', '.join(handle_strs)})")
        total_new += player_new

    print(f"\nRefresh complete: {total_new} new, {total_skipped} skipped")

    if not args.dry_run and total_new > 0:
        append_ledger(
            f"refresh --since {since}",
            f"{len(identities)} labeled",
            total_new, total_skipped)


def cmd_scrape_date(args):
    """Scrape recent ladder replays from top ranked players."""
    since = args.since
    until = getattr(args, 'until', None)
    top = args.top
    date_desc = f"since {since}" + (f", until {until}" if until else "")
    print(f"Scraping replays from top {top} ranked players ({date_desc})...")

    existing = load_existing_match_ids()
    print(f"DB has {len(existing)} existing replays")

    # Fetch ranked players across all major gateways
    all_players = []
    seen_aliases = set()
    for gw in [30, 10, 20, 45]:
        try:
            ranked = api_rankings(gateway=gw, limit=top)
        except requests.exceptions.HTTPError:
            continue
        for p in ranked:
            alias = p.get("alias")
            key = (alias, gw)
            if key not in seen_aliases:
                seen_aliases.add(key)
                all_players.append((alias, gw))

    print(f"Found {len(all_players)} ranked players across gateways\n")

    output_dir = Path(args.output)
    total_new = 0
    total_skipped = 0
    seen_urls = set()  # Deduplicate across players (same match appears for both)

    for i, (alias, gw) in enumerate(all_players):
        try:
            matches = api_matches_since(alias, gateway=gw, since=since, until=until)
        except requests.exceptions.HTTPError as e:
            # Barcode names often 500 — skip silently
            continue

        if not matches:
            continue

        # Deduplicate by replay_url across players
        unique_matches = []
        for m in matches:
            url = m.get("replay_url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_matches.append(m)

        if not unique_matches:
            continue

        downloaded, skipped = download_matches(
            unique_matches, output_dir, existing_match_ids=existing,
            dry_run=args.dry_run)

        if downloaded > 0:
            gw_name = GATEWAY_NAMES.get(gw, str(gw))
            print(f"  [{i+1}/{len(all_players)}] {alias}@{gw_name}: {downloaded} new")
        total_new += downloaded
        total_skipped += skipped

        # Update existing set
        for m in unique_matches:
            url = m.get("replay_url")
            if url:
                existing.add(extract_replay_match_id(url))

    date_range = since + (f" to {until}" if until else "+")
    print(f"\nScrape-date complete: {total_new} new, {total_skipped} skipped ({date_range})")

    if not args.dry_run and total_new > 0:
        cmd_str = f"scrape-date --since {since}"
        if until:
            cmd_str += f" --until {until}"
        append_ledger(cmd_str, f"{len(all_players)} ranked", total_new, total_skipped)


# ---------------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="cwal",
        description="cwal.gg ladder API tool for Barcode Reader",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- matches --
    p_matches = sub.add_parser("matches", help="Search a player's match history")
    p_matches.add_argument("alias", help="Player alias (exact match)")
    p_matches.add_argument("--map", help="Filter by map name (substring)")
    p_matches.add_argument("--matchup", help="Filter by matchup (e.g. pvt, zvz)")
    p_matches.add_argument("--opponent", help="Filter by opponent name (substring)")
    p_matches.add_argument("--min-duration", metavar="MM:SS", help="Minimum game duration")
    p_matches.add_argument("--max-duration", metavar="MM:SS", help="Maximum game duration")
    p_matches.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Max matches to fetch (default {DEFAULT_LIMIT})")
    p_matches.add_argument("--gateway", type=int, default=DEFAULT_GATEWAY, help=f"Gateway ID (default {DEFAULT_GATEWAY}=Korea)")
    p_matches.set_defaults(func=cmd_matches)

    # -- scrape --
    p_scrape = sub.add_parser("scrape", help="Download replays for a player")
    p_scrape.add_argument("alias", help="Player alias (exact match)")
    p_scrape.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Max replays to download (default {DEFAULT_LIMIT})")
    p_scrape.add_argument("--output", default=str(OUTPUT_DIR), help=f"Output directory (default {OUTPUT_DIR})")
    p_scrape.add_argument("--dry-run", action="store_true", help="Show what would download without downloading")
    p_scrape.add_argument("--gateway", type=int, default=DEFAULT_GATEWAY, help=f"Gateway ID (default {DEFAULT_GATEWAY}=Korea)")
    p_scrape.set_defaults(func=cmd_scrape)

    # -- rankings --
    p_rankings = sub.add_parser("rankings", help="Show ladder standings")
    p_rankings.add_argument("--limit", type=int, default=20, help="Number of players to show (default 20)")
    p_rankings.add_argument("--race", help="Filter by race (protoss, terran, zerg)")
    p_rankings.add_argument("--gateway", type=int, default=DEFAULT_GATEWAY, help=f"Gateway ID (default {DEFAULT_GATEWAY}=Korea)")
    p_rankings.set_defaults(func=cmd_rankings)

    # -- search --
    p_search = sub.add_parser("search", help="Search for a player by name")
    p_search.add_argument("query", help="Search string (substring match)")
    p_search.add_argument("--gateway", type=int, default=DEFAULT_GATEWAY, help=f"Gateway ID (default {DEFAULT_GATEWAY}=Korea)")
    p_search.set_defaults(func=cmd_search)

    # -- handles --
    p_handles = sub.add_parser("handles", help="Look up all handles for a player (alias or account ID)")
    p_handles.add_argument("player", help="Player alias or battlenet_account ID (numeric)")
    p_handles.add_argument("--gateway", type=int, default=DEFAULT_GATEWAY, help=f"Gateway ID (default {DEFAULT_GATEWAY}=Korea)")
    p_handles.set_defaults(func=cmd_handles)

    p_count = sub.add_parser("count", help="Get total match count for a player on cwal.gg")
    p_count.add_argument("player", help="Player alias")
    p_count.add_argument("--gateway", type=int, default=DEFAULT_GATEWAY, help=f"Gateway ID (default {DEFAULT_GATEWAY}=Korea)")
    p_count.set_defaults(func=cmd_count)

    # -- refresh --
    p_refresh = sub.add_parser("refresh", help="Scrape new games for all labeled players")
    p_refresh.add_argument("--since", help="Only fetch games after this date (YYYY-MM-DD, default 7 days ago)")
    p_refresh.add_argument("--output", default=str(OUTPUT_DIR), help=f"Output directory (default {OUTPUT_DIR})")
    p_refresh.add_argument("--dry-run", action="store_true", help="Show what would download without downloading")
    p_refresh.set_defaults(func=cmd_refresh)

    # -- scrape-date --
    p_scrape_date = sub.add_parser("scrape-date", help="Scrape recent ladder replays from top ranked players")
    p_scrape_date.add_argument("--since", required=True, help="Fetch games after this date (YYYY-MM-DD)")
    p_scrape_date.add_argument("--until", help="Fetch games before this date (YYYY-MM-DD)")
    p_scrape_date.add_argument("--top", type=int, default=200, help="Number of top ranked players to scrape per gateway (default 200)")
    p_scrape_date.add_argument("--output", default=str(OUTPUT_DIR), help=f"Output directory (default {OUTPUT_DIR})")
    p_scrape_date.add_argument("--dry-run", action="store_true", help="Show what would download without downloading")
    p_scrape_date.set_defaults(func=cmd_scrape_date)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
