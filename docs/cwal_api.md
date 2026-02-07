# cwal.gg API Reference

cwal.gg uses a Supabase backend with a public anon key. No auth needed beyond the API key.

## Base Config

```
URL:  https://xmploueumzkrdvapbyfs.supabase.co/rest/v1
Key:  (see scrape_cwal.py)
Headers:
  apikey: <key>
  Authorization: Bearer <key>
  Accept-Profile: public
```

## Endpoints

### Rankings: `GET /rankings_view`

Current leaderboard snapshot. Not historical - only shows today's standings.

**Params:**
- `standing=lte.100` - filter by rank (top 100)
- `standing=gte.101` - paginate deeper (works up to at least 400+)
- `order=standing.asc`
- `limit=100` - API caps at 100 per request

**Returns per player:**
| Field | Example | Notes |
|-------|---------|-------|
| `standing` | 1 | Current rank |
| `rating` | 2703 | MMR rating |
| `wins` / `losses` | 150 / 80 | Season record |
| `alias` | `JSA_Larva` | Player name |
| `race` | `zerg` | Lowercase |
| `gateway` | 30 | 30 = Korea |
| `rank` | `S` | League tier |
| `disconnects` | 2 | DC count |
| `avatar` | | Profile icon |

**Pagination**: Use `standing=gte.101`, `standing=gte.201`, etc. to go beyond top 100.

### Player Matches: `GET /player_matches`

Match history for a specific player.

**Params:**
- `gateway=eq.30` - Korea server
- `alias=eq.JSA_Larva` - player name (exact match)
- `opponent_alias=eq.C9_NeedMoney.` - search by opponent too
- `order=timestamp.desc` - newest first
- `limit=100` - API caps at 100 matches regardless
- `offset=0` - pagination (but 100 is the hard cap)

**Returns per match:**
| Field | Example | Notes |
|-------|---------|-------|
| `id` | 12345 | Match ID |
| `aurora_id` | | Internal ID |
| `timestamp` | `2026-02-05T09:33:53+00:00` | Match time |
| `result` | `win` / `loss` | From this player's perspective |
| `map_name` | `Polypoid 1.75` | Map display name |
| `map_file_name` | `Polypoid 1.75` | Map filename |
| `alias` | `JSA_Larva` | This player |
| `race` | `zerg` | This player's race |
| `opponent_alias` | `llIIll1ll1lI` | Opponent name |
| `opponent_race` | `terran` | Opponent's race |
| `matchup` | `zvt` | Matchup string |
| `mmr` | 2467 | Player's MMR at time of match |
| `mmr_delta` | +12 | MMR change from this game |
| `winner_race` | `zerg` | Who won |
| `duration` | 637 | Game duration (seconds) |
| `replay_url` | `https://.../.rep` | Direct download link (permanent) |
| `replay_data` | | Binary replay (don't use, use URL) |

## Limits

- **100 players** per rankings request (paginate with standing ranges)
- **100 matches** per player (hard cap, goes back ~2-3 months)
- Rankings are a **live snapshot** - no historical data
- Replay URLs don't expire

## Our Scraper: `scrape_cwal.py`

Fetches top 200 players (2 requests), then 100 matches each, downloads .rep files.
Skips already-downloaded replays. Rate limited (0.2s between downloads, 0.5s between players).

Run periodically to catch players rotating in/out of top 200.

## Useful Queries

```python
# Search by opponent (find all games a barcode played)
fetch_player_matches(30, opponent_alias='eq.llIIll1ll1lI')

# Get deeper rankings
fetch_top_players(standing='gte.201', limit=100)
```
