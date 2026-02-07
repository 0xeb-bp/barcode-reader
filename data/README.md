# Data Directory

## Summary

| Dataset | Replays | Date Range | Labeled? | Status |
|---------|---------|------------|----------|--------|
| poc_2008_2023 | 167 | 2008-2023 | 59 games | POC complete |
| cwal_replays | 1291+ | Feb 2026 | By player | Scraping... |
| bisu_archive | 10 | ~2007-2009 | All Bisu | Extracted |
| flash_archive | 44 | ~2007-2009 | All Flash | Extracted |
| 680 Progamer reps | 680 | ~2006-2010 | All labeled | Ready |
| replays_to_review/* | 4270 | Mixed | Partial | Needs processing |

**Total**: ~6,000+ replays

---

## Datasets

### poc_2008_2023/
**Status**: Proof of concept (complete)
**Games**: 167 replays
**Date range**: 2008-12-15 to 2023-10-25 (15 years!)
**Source**: Team Liquid replay database
**Known players**: 15 (59 labeled games)

**Issues**:
- Huge time span - player styles evolve
- Effort has 7/9 games from same day (2010-01-22)
- Some players only have games from 1-2 days
- Mixed eras: old KeSPA pros, modern ASL players

**Results**: 72.9% accuracy (best with depth=10, trees=200)

---

### cwal_replays/
**Status**: Actively scraping
**Games**: 1291+ replays (growing)
**Date range**: Feb 2026 (all recent ladder games)
**Source**: cwal.gg Supabase API
**MMR range**: 2400-2700 (top ladder)

**Known players**:
- YB_Scan (Scan, ASL pro Terran)
- wico\`ddd (Zerg)
- ImSky (Protoss)
- IadderKing (Zerg)
- Multiple barcodes (unknown identity)

**Why this is valuable**:
- All post-2018 (modern era, stable meta)
- High skill level (top ladder)
- Includes barcodes to test identification
- Metadata in metadata.jsonl

---

### bisu_archive/
**Status**: Extracted
**Games**: 10 replays
**Source**: Personal archive (bisu.rar)
**Players**: All Bisu games

Files:
- bisuvsbest.rep, bisuvsbest(1).rep
- bisuvshuky.rep
- bisuvssoo.rep
- bisuvsssak.rep
- bisuvsthezerg.rep
- bisuvszerg[kal].rep
- By.FlashVSBisu[Shield]1-3.rep

---

### flash_archive/
**Status**: Extracted
**Games**: 44 replays
**Source**: Personal archive (flash.rar)
**Players**: All Flash games

Notable opponents: Bisu, Jangbi, Fantasy, Best, herO, Lomo, Mind, Lucifer

---

### 680 Progamer reps/
**Status**: Ready to process
**Games**: 680 replays
**Date range**: ~2006-2010 (patch 1.15 and 1.16)
**Source**: Korean pro scene archive
**Labeling**: Filename format "Player1 vs Player2.rep"

**Organization**:
- Split by patch version (1.15, 1.16)
- Split by matchup (ZvP, ZvT, ZvZ, PvT, PvP, TvT)

**Unique players**: ~137 pros including:
- Flash, Bisu, Jaedong, Stork
- sAviOr, July, Yellow, Nal_rA
- Fantasy, Best, Effort, Calm
- And 120+ more

**High value**: Clean labeled data for training

---

## replays_to_review/ (Raw Archives)

Total: 4,270 replays across multiple collections:

| Folder | Replays | Notes |
|--------|---------|-------|
| Star replays from Korea | 1,928 | Large Korean archive |
| 680 Progamer reps | 680 | See above |
| ygosu | 588 | Organized by matchup |
| Pro-Semipro reps | ~200 | Mixed quality |
| replay heaven 3 | ~150 | Community collection |
| IEF2008/2009 | ~100 | Tournament replays |
| China/Europe/USA vs Korea | ~100 | International matches |
| STX Cup Masters 2010 | ~50 | Tournament replays |

---

## Data Pipeline TODO

1. **Extract player names from replay files** - Use screp parser
2. **Build player alias database** - Same player, different names
3. **Date extraction** - Filter for post-2018 games
4. **Deduplication** - Same replay in multiple folders
5. **Quality filtering** - Remove corrupted/short games
6. **Feature extraction** - Generate training features

---

## Notes

- **2018+ is ideal**: Last balance patch, modern meta
- **Pro identification priority**: Flash, Bisu, Jaedong, Stork, Rain, sOs, herO, Best
- **Barcode detection**: Use cwal ladder data with known MMR ranges
