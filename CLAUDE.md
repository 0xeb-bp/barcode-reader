# Gosu Unveiled

## ML Terminology Coaching
When the user uses ML terminology (correctly or incorrectly), briefly correct or affirm their usage BEFORE continuing with the primary response. This helps them learn to speak more precisely about ML concepts. Keep corrections short and constructive.

## Project Structure
- `features.py` — Shared feature extraction module (n-grams, timing, hotkeys, clicks)
- `train.py` — Train model, LOO cross-validation, save to `data/model.joblib`
- `predict.py` — Load saved model, predict on all unlabeled players (10+ games)
- `ingest_replays.py` — Replay ingestion pipeline (to_ingest → cwal_replays)
- `cwal.py` — CLI tool for cwal.gg API (matches, scrape, rankings, search, handles)
- `scrape_cwal.py` — Legacy bulk scraper (prefer `cwal.py scrape` instead)
- `classifier_2026.py` — Legacy single-script classifier (reference only)
- `experiments.md` — **MUST UPDATE** after every training iteration

## Conventions
- Python venv at `.venv/` — use `.venv/bin/python` to run scripts
- Replay parser: `~/go/bin/screp` (Go binary, use with `-cmds` for commands, `-map` for positions)
- Database: `data/replays.db` (SQLite)
- New replays go to `data/to_ingest/`, get moved to `data/cwal_replays/` after ingestion

## Data Labeling Rules
- This is data for training a classification model. It must be **HIGH FIDELITY**. Never alias or label without 100% confidence that the mapping is correct.
- **NEVER add player_aliases without explicit user confirmation.** Only add the exact aliases the user tells you.
- Don't guess or infer additional aliases from similar names in the database.

## Scraping Rules
- When scraping a player from cwal.gg, download **50-100 replays max** per player (use `limit=50`).
- Don't over-scrape. Quality > quantity for labeled training data.

## Database Schema (data/replays.db)
- **replays** — one row per replay file: `id, file_hash, file_path, file_name, source_dir, map_name, game_date, duration_seconds, frames, version, created_at, winner_team`
- **players** — one row per player per replay: `id, replay_id, slot_id, player_name, race, is_human, start_x, start_y, start_direction`
- **player_aliases** — maps in-game names to canonical pro names: `id, canonical_name, alias, confidence, source`

## Current Model
- **Random Forest**: depth=10, trees=200, StandardScaler, LOO cross-validation
- **Modern era only**: Trains on replays with game_date >= 2025-01-01
- **99.1% accuracy** on 16 modern pros (last training run)
- Model saved to `data/model.joblib`, predictions to `data/predictions.json`
- Race-neutral features: abstract race-specific actions into generic categories (Train/Unit Morph → Prod)
- Consecutive Prod collapse to reduce race-mechanical signal

## Training Roster (modern era, currently trained on)
Larva(107), Ample(107), BishOp(105), Stork(102), Flash(101), Scan(100), Rush(100), Best(99), SoulKey(97), Sharp(96), soO(60), EffOrt(59), Tyson(59), Rain(50), Fantasy(29), Sky(27)

## Labeled But Not In Training Set
These players are labeled and have modern era data, but were not in the last training run:
- Speed(58), Air(58), yOOn(54), Artosis(52), Rich(10)

## Legacy-Only Players (no modern era data)
sAviOr, Jaedong, NaDa, Bisu, Nal_rA, Effort (lowercase, separate from EffOrt), Zero, July, Mind, YellOw, Jangbi, Boxer, herO, Shuttle, action

## Workflow: Adding a New Player
1. Find their cwal.gg alias: `cwal.py search <name>`
2. Scrape replays: `cwal.py scrape <alias> --limit 50`
3. Add alias to DB (only with user confirmation): INSERT INTO player_aliases
4. Ingest: `python ingest_replays.py`
5. Retrain: `python train.py`
6. Update experiments.md with results

## cwal.gg API Notes
- Duration in `player_matches` is in **seconds** (decimal, e.g. 417.396 = 6:57)
- `map_name` field contains BW color codes (control chars) — use `map_file_name` instead
- `battlenet_account` field is on the `players` table, NOT on `rankings_view`
- Default gateway 30 = Korea, 10 = US West, 20 = US East, 45 = Europe
