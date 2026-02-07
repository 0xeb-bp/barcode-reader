#!/usr/bin/env python3
"""cwal.gg ladder API tool for Gosu Unveiled."""

import argparse
import re
import time

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

OUTPUT_DIR = Path(__file__).parent / "data" / "to_ingest"


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

    # Filter to matches with replay URLs
    downloadable = [m for m in matches if m.get("replay_url")]
    print(f"Found {len(downloadable)} matches with replays (of {len(matches)} total).")

    if not downloadable:
        return

    output_dir = Path(args.output)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0

    for m in downloadable:
        replay_url = m["replay_url"]
        match_id = replay_url.split("/")[-1].replace(".rep", "")
        opponent = m.get("opponent_alias", "unknown")
        matchup = m.get("matchup", "unknown")
        result = m.get("result", "unknown")
        filename = f"{args.alias}_vs_{opponent}_{matchup}_{result}_{match_id}.rep"
        output_path = output_dir / filename

        if args.dry_run:
            tag = "SKIP" if output_path.exists() else "GET"
            print(f"  [{tag}] {filename}")
            continue

        if output_path.exists():
            skipped += 1
            continue

        resp = requests.get(replay_url)
        if resp.status_code == 200:
            output_path.write_bytes(resp.content)
            downloaded += 1
            print(f"  Downloaded: {filename}")
        else:
            print(f"  Failed ({resp.status_code}): {filename}")

        time.sleep(DOWNLOAD_DELAY)

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
    """Look up all handles for a battlenet_account ID."""
    print(f"Looking up battlenet_account={args.battlenet_account}...")
    handles = api_handles(args.battlenet_account)

    if not handles:
        print("No handles found for that account.")
        return

    print(f"\n{len(handles)} handle(s):\n")
    header = f"{'#':>4} {'Alias':<22} {'Race':<10} {'Rating':>6} {'W-L':<10} {'Gateway':>7}"
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
        gw = p.get("gateway", "—")
        print(f"{standing:>4} {alias:<22} {race:<10} {rating:>6} {wl:<10} {gw:>7}")


# ---------------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="cwal",
        description="cwal.gg ladder API tool for Gosu Unveiled",
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
    p_handles = sub.add_parser("handles", help="Look up all handles for a battlenet_account ID")
    p_handles.add_argument("battlenet_account", help="battlenet_account ID (numeric)")
    p_handles.set_defaults(func=cmd_handles)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
