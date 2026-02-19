# Experiment Tracking

## Metrics
- **Accuracy**: % correct predictions (Leave-One-Out CV)
- **Self-consistency ratio**: inter-player distance / intra-player distance (>1.5 = distinct)
- **Race separation**: should be LOW (~1.0) if features are race-neutral

---

## Experiment 1: Original Features
**Date**: 2024
**Features**: APM, command distributions (including Train, Unit Morph), timing, hotkeys, n-grams
**Feature count**: ~80 (varies with n-grams)

| Metric | Value |
|--------|-------|
| Samples | 14 (7 players) |
| Accuracy | 85.7% (12/14) |
| Notes | Small sample, n-grams included |

---

## Experiment 2: Expanded Data, Original Features
**Date**: 2024
**Features**: Same as Exp 1
**Feature count**: ~80

| Metric | Value |
|--------|-------|
| Samples | 20 (7 players) |
| Accuracy | 75.0% (15/20) |
| Notes | More data but accuracy dropped - possible overfitting to race |

---

## Experiment 3: Race-Neutral Features (no n-grams)
**Date**: 2024
**Features**: Timing patterns, hotkey patterns, click patterns, early game, APM curves
**Removed**: pct_Train, pct_Unit_Morph, all n-grams
**Feature count**: 35

| Metric | Value |
|--------|-------|
| Samples | 59 (15 players) |
| Accuracy | 61.0% (36/59) |
| Self-consistency | 11/14 players with ratio > 1.5 |
| Race separation | Zerg 1.16, Terran 1.12, Protoss 1.04 (good - low) |
| Notes | Lower accuracy but more players, better race neutrality |

**Per-player accuracy**:
- 100%: Bisu, Dewalt, Flash, Sonic
- Weak: Effort (often mispredicted), Zerg players generally

---

## Experiment 4: Race-Neutral + Abstracted N-grams
**Date**: 2024
**Features**: Exp 3 + n-grams using abstracted categories (C=control, M=move, A=ability, P=produce, U=upgrade, T=transport, X=cancel)
**N-gram sizes**: 2, 3, 4 (top 10 each)
**Feature count**: 116 (35 base + 81 n-gram)

| Metric | Value |
|--------|-------|
| Samples | 59 (15 players) |
| Accuracy | **64.4% (38/59)** |
| Notes | +3.4% over Exp 3. Jaedong improved to 100%, Flash dropped to 50% |

**Per-player accuracy:**
- 100%: Bisu (6/6), Jaedong (4/4), Sonic (4/4)
- 88.9%: Effort (8/9)
- 75%: Dewalt (3/4)
- 66.7%: Light (4/6), Movie (2/3), Soma (2/3)
- 50%: Flash (2/4), Violet (2/4)
- 33%: Sea (1/3)
- 0%: Hiya (0/2), Hydra (0/3), Shuttle (0/2), Zero (0/2)

**Top n-gram features:**
- `ng4_C_C_C_C` - Hotkey spam pattern (importance: 0.026)
- `ng2_A_C` - Ability then control (0.024)
- `ng3_C_M_C` - Control-move-control rhythm (0.021)
- `ng2_C_P` - Select then produce (0.019)

---

## Experiment 4b: Smarter Abstraction (keep race-neutral actions distinct)
**Date**: 2024
**Features**: Only abstract race-specific actions (Train/Unit Morph → Prod), keep Hotkey, Select, RightClick distinct
**Feature count**: 171 (35 base + 136 n-gram)

| Metric | Value |
|--------|-------|
| Samples | 59 (15 players) |
| Accuracy | **66.1% (39/59)** |
| Notes | +5% over Exp 3, +1.7% over Exp 4a. Flash back to 100% |

**Per-player accuracy:**
- 100%: Bisu (6/6), Flash (4/4), Jaedong (4/4), Sonic (4/4)
- 88.9%: Effort (8/9)
- 75%: Violet (3/4)
- 66.7%: Light (4/6), Movie (2/3), Soma (2/3)
- 50%: Dewalt (2/4)
- 0%: Hiya (0/2), Hydra (0/3), Sea (0/3), Shuttle (0/2), Zero (0/2)

**Top n-gram features (now meaningful!):**
- `ng3_H_H_R` - Hotkey Hotkey RightClick (0.027)
- `ng2_R_R` - RightClick RightClick movement spam (0.023)
- `ng3_R_R_H` - RightClick RightClick Hotkey (0.022)
- `ng3_TO_H_H` - Targeted Order then check groups (0.020)

---

## Feature Categories (for abstracted n-grams)

```python
ACTION_CATEGORIES = {
    # Control
    "Hotkey": "control",
    "Select": "control",
    "Select Add": "control",
    "Select Remove": "control",

    # Movement
    "Right Click": "move",
    "Hold Position": "move",
    "Stop": "move",
    "Return Cargo": "move",

    # Abilities (spells, special actions)
    "Targeted Order": "ability",
    "Stim": "ability",
    "Burrow": "ability",
    "Unburrow": "ability",
    "Siege": "ability",
    "Unsiege": "ability",
    "Merge Archon": "ability",
    "Merge Dark Archon": "ability",
    "Cloack": "ability",
    "Decloack": "ability",

    # Production
    "Train": "produce",
    "Unit Morph": "produce",
    "Build": "produce",
    "Building Morph": "produce",
    "Train Fighter": "produce",

    # Upgrades
    "Upgrade": "upgrade",
    "Tech": "upgrade",

    # Transport
    "Unload": "transport",
    "Unload All": "transport",
    "Lift Off": "transport",
    "Land": "transport",

    # Cancel (might indicate mistakes/adaptation)
    "Cancel Train": "cancel",
    "Cancel Build": "cancel",
    "Cancel Morph": "cancel",
    "Cancel Upgrade": "cancel",
    "Cancel Tech": "cancel",
    "Cancel Addon": "cancel",
}
```

---

---

## Experiment 5: Optimized Hyperparameters
**Date**: 2024
**Features**: Same as Exp 4b (smart abstracted n-grams)
**Hyperparameters**: max_depth=10, n_estimators=200 (was depth=5, trees=100)
**Feature count**: 171

| Metric | Value |
|--------|-------|
| Samples | 59 (15 players) |
| Accuracy | **72.9% (43/59)** |
| Notes | +11.9% over Exp 3 baseline. 7 players at 100%. |

**Per-player accuracy:**
- 100%: Bisu (6/6), Flash (4/4), Jaedong (4/4), Movie (3/3), Sonic (4/4), Zero (2/2)
- 88.9%: Effort (8/9)
- 75%: Dewalt (3/4), Violet (3/4)
- 66.7%: Light (4/6), Soma (2/3)
- 0%: Hiya (0/2), Hydra (0/3), Sea (0/3), Shuttle (0/2)

**Key insight**: Deeper trees (10 vs 5) dramatically helped. Zero went from 0% to 100%.

**Confusion patterns**:
- Sea always predicted as Light (both Terran, similar style?)
- Hydra confused with multiple players (inconsistent fingerprint or too few games?)

---

---

## Data Collection (Feb 2026)

### Replay Database Built
- **Total replays**: 3,788
- **Unique player names**: 4,577
- **Date range**: 1999-2026

### Sources Ingested:
| Source | Replays | Era |
|--------|---------|-----|
| cwal.gg ladder | 1,364 | Feb 2026 |
| Star replays from Korea | 1,127 | 2007-2010 |
| 680 Progamer reps | 610 | 2006-2010 |
| ygosu archives | 471 | Mixed |
| poc_2008_2023 | 166 | 2008-2023 |
| Flash archive | 40 | 2007-2009 |
| Bisu archive | 10 | 2009 |

### Known Pro Games in Database:
- YB_Scan (modern): 50 games (2026)
- sAviOr: 46 games (2007-2009)
- Flash: 42 games (2006-2010)
- Bisu: 33 games (2003-2011)
- Stork: 30 games (2007-2010)
- Best: 26 games (2006-2011)
- Effort: 24 games (2008-2010)
- Fantasy: 22 games (2006-2010)
- Jaedong: 17 games (2007-2010)

### Barcodes in 2026 Ladder:
- **182 unique barcodes** (names like `llllllIIllIllll`)
- **814 games** to potentially unmask
- Breakdown: 68 Protoss, 64 Zerg, 50 Terran

### Next Steps:
1. ~~Train classifier on known pros from 2026 ladder (YB_Scan, etc)~~
2. ~~Predict on barcodes to find hidden pros~~
3. Validate by cross-referencing with match timing/MMR

---

## Experiment 6: 2026 Ladder Data (Modern Era)
**Date**: Feb 2026
**Features**: Same as Exp 5 (race-neutral + abstracted n-grams)
**Hyperparameters**: max_depth=10, n_estimators=200
**Feature count**: 156

| Metric | Value |
|--------|-------|
| Samples | 148 (12 players) |
| Accuracy | **87.2%** |

**Per-player accuracy:**
- 100%: BlackProbe\`net, BBiDDaDDuE, JSA_Larva, NCS_Gun, C9_Jaesung, YB_Scan
- 92.9%: !lii!!lili!l!ii
- 91.7%: YB_Ample
- 90.0%: ImSky.
- 57.1%: awdfzxvczvccv12
- 50.0%: agzxcvbvzbxcbvc, C9_NeedMoney.

**Top features:**
1. `moderate_ratio` (0.033) - timing between actions
2. `primary_hotkey_group` (0.031) - most-used hotkey group
3. `first_assign_0` (0.027) - first hotkey assigned (muscle memory)
4. `queued_ratio` (0.026) - shift-queue usage
5. `ng3_R_R_R` (0.022) - triple right-click pattern

**Barcode Predictions (High Confidence):**
| Barcode | Race | Predicted | Conf |
|---------|------|-----------|------|
| lillljilililili | Z | JSA_Larva | 100% |
| IlIIlIIIIIIIl | Z | JSA_Larva | 100% |
| lllllllIIIIlIlI | Z | JSA_Larva | 90% |
| IIlIlIIIlIIIll | T | C9_Jaesung | 80% |
| lllllli112l | P | YB_Scan | 78% |
| lllllIIIllIllIl | T | YB_Scan | 67% |

**Insights:**
- JSA_Larva appears to have multiple barcode accounts
- YB_Scan (ASL pro) possibly smurfing on barcodes
- Early hotkey assignments are highly discriminative

---

## Experiment 7: All Known Pros (All Eras)
**Date**: Feb 2026
**Features**: Same as Exp 6 (race-neutral + abstracted n-grams)
**Hyperparameters**: max_depth=10, n_estimators=200
**Feature count**: 230
**Training data**: ALL confirmed pro games across all eras (2003-2026)

| Metric | Value |
|--------|-------|
| Samples | 302 (14 players) |
| Accuracy | **87.1%** |

**Per-player accuracy:**
- 100.0%: Scan (19/19)
- 95.5%: Fantasy (21/22)
- 94.4%: SoulKey (17/18)
- 94.1%: Jaedong (16/17)
- 93.8%: Zero (15/16)
- 92.9%: Larva (13/14)
- 92.0%: Stork (23/25)
- 91.7%: Nal_rA (22/24)
- 90.9%: Effort (20/22)
- 88.0%: Flash (22/25)
- 84.0%: NaDa (21/25)
- 80.0%: sAviOr (20/25)
- 72.0%: Bisu (18/25)
- 64.0%: Best (16/25)

**Top features:**
1. `first_assign_0` (0.049) - first hotkey assignment
2. `moderate_ratio` (0.027) - timing between actions
3. `early_rapid_ratio` (0.026) - early game speed
4. `hotkey_assign_ratio` (0.025) - assign vs recall ratio
5. `ng2_H_S` (0.024) - Hotkey→Select pattern

**Barcode Predictions (High Confidence):**
| Barcode | Race | Predicted | Conf | Prob |
|---------|------|-----------|------|------|
| IlIIlIIIIIIIl | Z | Larva | 100% | 49% |
| lillljilililili | Z | Larva | 100% | 43% |
| IIlIlIllllIIIl1 | Z | Larva | 100% | 39% |
| IllllIlIIIlIlll | P | NaDa | 100% | 38% |
| IlIllIlIIIIllII | Z | Larva | 100% | 35% |
| l\|I\|l\|I\|l\|I | Z | Larva | 100% | 32% |
| llIIllIIlIIlIlI | P | NaDa | 100% | 24% |
| IllIIIIlllIIIll | P | NaDa | 86% | 24% |
| IIIIlIIIllIIIII | Z | Effort | 80% | 32% |
| IIlIlIIIlIIIll | T | NaDa | 80% | 24% |
| llIIll1ll1lI | Z | Jaedong | 62% | 22% |

**Key Insights:**
- Larva confirmed as most prolific barcode user (5+ barcode accounts, all 100% match)
- NaDa's fingerprint matches 4 barcodes - NaDa playing on 2026 ladder?
- Jaedong-style barcode detected (llIIll1ll1lI)
- Effort-style barcode detected (IIIIlIIIllIIIII)
- `first_assign_0` is by far the top feature (0.049) - your first hotkey is your fingerprint

---

## Experiment 9: Two-Pass Global N-gram Selection
**Date**: Feb 2026
**Features**: Same as Exp 8, but n-grams selected globally instead of per-game
**Hyperparameters**: max_depth=10, n_estimators=200
**Feature count**: 195 (92 base + 103 n-gram)

**Change**: Instead of taking top-N n-grams per game (which misses consistent-but-not-top patterns and includes noisy one-off patterns), we now:
1. **Pass 1**: Collect raw n-gram counts from all training games
2. **Pass 2**: Select globally most common n-grams, score every game on that fixed set

**Global n-gram counts**: ng2=25, ng3=20, ng4=15, hkg2=15, hkg3=10, ehkg2=10, ehkg3=8

| Metric | Value |
|--------|-------|
| Samples | 302 (14 players) |
| Accuracy | **89.4%** (+1.0% over Exp 8, fewer features) |

**Per-player accuracy:**
- 100.0%: Larva (14/14), Scan (19/19), Fantasy (22/22), Zero (16/16)
- 96.0%: Stork (24/25)
- 94.4%: SoulKey (17/18)
- 94.1%: Jaedong (16/17)
- 92.0%: Flash (23/25)
- 91.7%: Nal_rA (22/24)
- 90.9%: Effort (20/22)
- 88.0%: NaDa (22/25)
- 84.0%: sAviOr (21/25)
- 72.0%: Bisu (18/25)
- 64.0%: Best (16/25)

**Top features:**
1. `first_assign_0` (0.024) - first hotkey assignment
2. `ng3_H_S_Prod` (0.023) - Hotkey→Select→Produce pattern
3. `ehkg2_3_3` (0.020) - early game group 3 double-tap
4. `ehkg3_4_4_4` (0.019) - early game group 4 triple-tap
5. `ehkg3_3_3_3` (0.018) - early game group 3 triple-tap

**Key insight**: Fewer features (195 vs 230), better accuracy. The global n-gram set captures universally relevant patterns rather than per-game noise. Early game hotkey group n-grams dominate the top features - muscle memory is the strongest fingerprint.

**Barcode Predictions (High Confidence):**
| Barcode | Race | Predicted | Conf | Prob |
|---------|------|-----------|------|------|
| IlIIlIIIIIIIl | Z | Larva | 100% | 47% |
| lllllIIIllIllIl | T | NaDa | 100% | 41% |
| llIIll1ll1lI | Z | Jaedong | 100% | 38% |
| IIlIlIIIlIIIll | T | NaDa | 100% | 37% |
| lilil...lililil | T | Best | 100% | 33% |
| IIIIlIIIllIIIII | Z | Effort | 100% | 25% |

**Jaedong barcode confirmed**: `llIIll1ll1lI` went from 62% confidence (Exp 7) to **100%** (8/8 games). The global n-gram approach makes this a clean match.

---

## Experiment 10: Modern Era, Prod Collapse, 16 Players
**Date**: Feb 2026
**Features**: Race-neutral + abstracted n-grams + consecutive Prod collapse
**Hyperparameters**: max_depth=10, n_estimators=200, StandardScaler
**Feature count**: 189
**Training data**: Modern era only (>=2025-01-01), 25 samples/player max

| Metric | Value |
|--------|-------|
| Samples | 322 (16 players) |
| Accuracy | **99.1%** |

**Per-player accuracy:**
- 100%: Ample, Best, Fantasy, Flash, Larva, Rain, Rush, Scan, Sharp, Sky, SoulKey, soO, Stork, Tyson (14/16)
- 95.5%: BishOp (21/22)
- 90.0%: EffOrt (18/20)

**Notes**: Consecutive Prod collapse removed race-mechanical queuing signal from n-grams. Modern era only gave much cleaner data than all-eras training.

---

## Experiment 11: Modern Era, 20 Players (Added Speed, Air, yOOn, Artosis)
**Date**: Feb 2026
**Features**: Same as Exp 10
**Hyperparameters**: max_depth=10, n_estimators=200, StandardScaler
**Feature count**: 189
**Training data**: Modern era only (>=2025-01-01), 25 samples/player max

| Metric | Value |
|--------|-------|
| Samples | 407 (20 players) |
| Accuracy | **99.0%** |

**Per-player accuracy:**
- 100%: Air (21/21), Ample (22/22), Artosis (18/18), Best (23/23), Fantasy (23/23), Flash (14/14), Larva (16/16), Rain (20/20), Rush (17/17), Scan (23/23), Sharp (24/24), Sky (23/23), SoulKey (11/11), soO (22/22), Speed (20/20), Stork (24/24), Tyson (24/24)
- 95.5%: BishOp (21/22)
- 95.0%: yOOn (19/20)
- 90.0%: EffOrt (18/20)

**Top features:**
1. `ehkg2_3_3` (0.022) - early hotkey group 3 double-tap
2. `ehkg2_4_4` (0.021) - early hotkey group 4 double-tap
3. `ehkg2_2_2` (0.021) - early hotkey group 2 double-tap
4. `ehkg3_2_2_2` (0.020) - early hotkey group 2 triple-tap
5. `early_rapid_ratio` (0.019) - early game speed

**Notes**: Added 4 new labeled players (Speed, Air, yOOn, Artosis). Rich skipped (only 7 valid modern games). All 4 new players at 100% except yOOn (95%). Accuracy held steady at 99.0% despite 25% more players. Early hotkey group patterns dominate feature importance.

---

## Experiment 12: Aurora ID Migration, 25 Players (max 50/player)
**Date**: Feb 2026
**Features**: Same as Exp 11 (race-neutral + abstracted n-grams + Prod collapse)
**Hyperparameters**: max_depth=10, n_estimators=200, StandardScaler
**Feature count**: 199
**Training data**: Modern era only (>=2025-01-01), 50 samples/player max, aurora_id-based identity

**Key change**: Migrated from alias-based (`player_aliases`) to aurora_id-based (`player_identities`) player identity. Training now joins on `players.aurora_id → player_identities.aurora_id`, automatically merging alt accounts. Added `--max-games` CLI flag to `train.py`.

| Metric | Value |
|--------|-------|
| Samples | 1,114 (25 players, max 50 each) |
| Accuracy | **98.6%** (1098/1114) |

**Per-player accuracy:**
- 100%: soO (50), Speed (50), Artosis (50), Ample (50), Stork (50), Scan (50), Rush (50), Jaedong (50), Best (50), SoulKey (50), gypsy (44), Tyson (50), Air (50), Rain (40), Fantasy (27), Sky (24), Shuttle (18)
- 98.0%: sSak (49/50), Flash (49/50)
- 96.2%: snOw (25/26)
- 96.0%: BishOp (48/50), yOOn (48/50)
- 94.0%: Larva (47/50)
- 93.3%: EffOrt (42/45)
- 92.5%: ProMise (37/40)

**New players (vs Exp 11):** +9 (sSak, snOw, soO, gypsy, Tyson, ProMise, HyuK*, Rich*, Jaedong)
- HyuK and Rich skipped (< 10 modern games)
- sSak auto-merged JSA_sSak1 + wkelkqwlewqe (151 total games via aurora_id)
- soO auto-merged fdsafasfsdafsda + lililllilillill (368 total games)
- snOw auto-merged IllIIlIlIl + IIIlIllIIllllIl (26 games)
- Stork auto-merged Stork + SSU_Stork (82 games)

**Top features:**
1. `ehkg2_1_1` (0.022) - early hotkey group 1 double-tap
2. `first_assign_0` (0.021) - first hotkey assignment
3. `ehkg2_4_2` (0.021) - early hotkey group 4→2 transition
4. `ehkg2_3_3` (0.019) - early hotkey group 3 double-tap
5. `ehkg3_1_1_1` (0.018) - early hotkey group 1 triple-tap

**Blind validation:**
- `llIIll1ll1lI` → Jaedong: **100% confidence** (17/17 games), 74% avg prob. Aurora_id 13968871 matches jd2321232. Confirmed.
- `wkelkqwlewqe` no longer appears as unlabeled — auto-merged with sSak via aurora_id 18372656.

**Barcode predictions (high confidence, >=80% prob):**
| Barcode | Race | Predicted | Conf | Prob |
|---------|------|-----------|------|------|
| aewinruvop | T | Rush | 100% | 87% |
| llIIIIllIIlIlI | P | Stork | 100% | 84% |
| dlqwkdckdl | P | Air | 100% | 83% |
| baldaction | Z | soO | 100% | 81% |
| dustmqgkwk!!! | T | Rush | 100% | 81% |
| zerosugarmarine | T | Scan | 100% | 80% |
| lllllllIIIIlIlI | Z | soO | 100% | 80% |

**Notes:**
- Accuracy dropped slightly from 99.1% (Exp 11, 16 players) to 98.6% (25 players) — expected with 56% more classes
- 17/25 players at 100%, lowest is ProMise at 92.5%
- 215 unlabeled players predicted, 139 at >=60% confidence
- Sharp not included (no aurora_id yet — needs lookup to join new training path)

---

## Experiment 13: Data Quality — Leave Trimming, Deep Features, Offrace Filtering
**Date**: Feb 2026
**Features**: Exp 12 features + 7 new race-invariant "deep features" + post-leave command trimming
**Hyperparameters**: max_depth=10, n_estimators=200, StandardScaler
**Feature count**: 206 (199 base/n-gram + 7 deep)
**Training data**: Modern era only (>=2025-01-01), 25 samples/player max, aurora_id-based, min 20 games

**Three changes from Exp 12:**

1. **Post-leave trimming** — Trim all commands at the first "Leave Game" so winners' post-leave actions (moving units around empty map) don't pollute features. Also caps game_frames for APM calculation.

2. **7 deep features** (race-invariant motor patterns):
   - `sa_latency_mean`, `sa_latency_median` — time between Select/Hotkey and next action command
   - `burst_count_per_min`, `burst_size_mean`, `inter_burst_gap_mean` — burst rhythm (consecutive commands < 150ms)
   - `autocorr_lag1` — lag-1 autocorrelation of command gaps (rhythmic vs random)
   - `map_jumps_per_min` — large cursor jumps within 500ms (multitask switching)

3. **Offrace filtering** — If a player has < 20 offrace games, only main-race games used. Players with 20+ offrace (Best, sSak, EffOrt, BishOp, Shine) keep all. Min games raised from 10 to 20.

**Investigation**: Analyzed offrace impact on 9 players. Found two categories:
- **Race-stable** (Best z≤1.1, sSak z≤1.5): features barely change across races
- **Race-divergent** (Larva z=18, BishOp z=14.8): hotkey groups completely flip, APM drops
- Deep features (burst structure, autocorrelation) stayed stable for all players tested

| Metric | Value |
|--------|-------|
| Samples | 798 (32 players, max 25 each) |
| Accuracy | **99.2%** (792/798) |

**Per-player accuracy:**
- 100%: 27/32 players
- 96.0%: Shine (24/25), BishOp (24/25), Dewalt (24/25), Larva (24/25)
- 92.0%: EffOrt (23/25)

**Misclassifications: 6** (down from 16 in comparable Exp 12 run)
- EffOrt → Best (×2, both Terran games kept via 20+ offrace threshold)
- BishOp → SoulKey (ZvZ game)
- Dewalt → Artosis
- Larva → soma (ZvZ)
- Shine → EffOrt

**Outlier analysis** (--analyze flag, 2.5σ threshold):
- 22 outliers / 798 samples
- 3 in misclassified+outlier overlap (EffOrt, BishOp, Dewalt — persistent hard cases)

**Top features:**
1. `ehkg2_3_3` (0.023) - early hotkey group 3 double-tap
2. `ehkg2_4_4` (0.018)
3. `ng2_H_Prod` (0.018)
4. `first_assign_0` (0.017)
5. `burst_size_mean` (0.014) ← NEW deep feature in top 15

**Notes:**
- Shuttle dropped (18 games < new 20 min threshold)
- Not a clean A/B test: dataset smaller (798 vs 818) due to offrace filtering, so some "improvement" is from removing hard cases rather than model improvement
- `burst_size_mean` entering top 15 features confirms deep features contribute signal
- Key offrace players filtered: Larva (6 games), Flash (19), SoulKey (6), Rain (17)
- Key offrace players kept: Best (49), sSak (51), EffOrt (45), BishOp (34), Shine (21)

---

## Experiment 14: Per-Race Filtering, --max-games 100, 33 Players
**Date**: Feb 2026
**Features**: Same as Exp 13 (race-neutral + abstracted n-grams + Prod collapse + deep features + leave trimming)
**Hyperparameters**: max_depth=10, n_estimators=200, StandardScaler
**Feature count**: ~206
**Training data**: Modern era only (>=2025-01-01), 100 samples/player max (per race), aurora_id-based, min 20 games

**Key changes from Exp 13:**

1. **Per-race --max-games cap** — Each race capped independently at --max-games. Previously the cap was global across all races. Now a player with 200Z + 30T gets up to 100Z + 30T (not 100 total).

2. **Most-recent-first ordering** — `ORDER BY game_date DESC` in feature extraction, so the --max-games cap takes the most recent games.

3. **Blind validation script** (`validate.py`) — Games beyond the training cap serve as natural held-out data. Predicts on unseen replays per player, filtered to trained races only.

4. **New players**: Light (aurora_id 14926205, 3 handles: kimsabuho, YB_Lightt, ZergPraticeGGUL)

| Metric | Value |
|--------|-------|
| Samples | 2,860 (33 players, max 100/race) |
| Accuracy | **99.9%** (2857/2860) |

**Per-player accuracy:**
- 100%: 30/33 players (soO, Shine, Air, Rush, Free, Sky, soma, EffOrt, Speed, yOOn, Artosis, RoyaL, Light, Jaedong, BishOp, sSak, Ample, Tyson, Scan, Stork, Flash, SoulKey, Best, Rich, Rain, gypsy, n00b, HyuK, Fantasy, snOw)
- 99.0%: Dewalt (99/100), ProMise (99/100)
- 98.8%: Larva (81/82)

**Misclassifications: 3**
- Dewalt → Artosis (IIII-------IIII, ZvP game)
- ProMise → Artosis (NCS_GuRaMise, TvP game)
- Larva → soma (JSA_Larva, ZvZ game)

**Shuttle**: Skipped (< 20 modern games)

### Blind Validation (held-out games not used in training)
Trained on 2,682 replays. 1,510 held-out games tested (same-race only, offrace excluded as noise).

| Player | Held-out | Correct | Accuracy |
|--------|----------|---------|----------|
| Shine | 193 | 193 | 100.0% |
| Air | 176 | 176 | 100.0% |
| Rush | 116 | 116 | 100.0% |
| Free | 113 | 113 | 100.0% |
| Sky | 113 | 113 | 100.0% |
| soma | 68 | 68 | 100.0% |
| Artosis | 54 | 54 | 100.0% |
| Speed | 49 | 49 | 100.0% |
| EffOrt | 38 | 38 | 100.0% |
| Jaedong | 35 | 35 | 100.0% |
| RoyaL | 33 | 33 | 100.0% |
| ProMise | 5 | 5 | 100.0% |
| Dewalt | 3 | 3 | 100.0% |
| Scan | 3 | 3 | 100.0% |
| Light | 51 | 50 | 98.0% |
| BishOp | 26 | 25 | 96.2% |
| soO | 305 | 290 | 95.1% |
| yOOn | 49 | 43 | 87.8% |
| Best | 33 | 19 | 57.6% |
| Shuttle | 17 | 0 | 0.0% |
| Rain | 13 | 0 | 0.0% |
| sSak | 11 | 0 | 0.0% |
| Larva | 3 | 0 | 0.0% |
| SoulKey | 3 | 0 | 0.0% |
| **TOTAL** | **1,510** | **1,426** | **94.4%** |

Players tested: 24, Perfect accuracy: 14/24

**Blind validation notes:**
- 0% players (Shuttle, Rain, sSak, Larva, SoulKey) all had very few held-out games (3-17), likely all offrace that slipped through or edge cases
- soO's 15 misclassifications all predicted as soma
- yOOn's 6 misclassifications all predicted as Sky
- Best's 10 misclassifications predicted as soma

**Data scrape following analysis:**
Scraped ~1,400 new replays for weak players to improve future training:
- soO: +974 (lililllilillill 380, fdsafasfsdafsda 272, lllllllIIIIlIlI 206, INO_soO 116)
- Larva: +253 (JSA_Larva)
- Rain: +199 (Mstz_Dex 168, MiniMaxii 31)
- sSak: +181 (JSA_sSak1 162, HM_sSak 19 — new handle discovered)
- snOw: +44 (IIIlIllIIllllIl, US West gateway)
- yOOn: +24 (C9_y00n)
- Flash: +2 (C9_FlaSh)
- Total DB: 11,493 replays (up from ~10,069)

---

## Ideas to Try
- [x] Abstracted n-grams (Experiment 4)
- [x] Hyperparameter tuning (Experiment 5)
- [x] Build replay database with SQLite
- [x] Scrape cwal.gg ladder replays
- [x] Train on 2026 known players (Scan, Larva, etc)
- [x] Predict on barcodes
- [x] Two-pass global n-gram selection (Experiment 9)
- [ ] Class balancing (undersample Effort)
- [ ] Different models (SVM, XGBoost)
- [ ] Per-race models (train separate model for each race)
- [ ] Time-windowed features (early/mid/late game separately)
- [ ] Investigate Sea vs Light confusion
- [x] Smurf detection: per-player outlier detection (distance from centroid or Isolation Forest), then classify outliers to identify who's actually playing. Could also clean training data by removing smurf-contaminated games. (Experiment 13 — outlier detection via Euclidean distance from class centroid, --analyze flag)
- [x] Leverage cwal.gg aurora_id for account linking (Experiment 12 — player_identities table)
- [x] Blind validation using held-out games beyond --max-games cap (Experiment 14 — validate.py)
- [ ] Run Tier 2 API backfill for opponent aurora_ids (improves unlabeled player grouping)
- [x] Add Sharp aurora_id to enable training

---

## Experiment 15: 43 Players, No Cap, class_weight="balanced"
**Date**: Feb 2026
**Features**: Same as Exp 14 (race-neutral + abstracted n-grams + Prod collapse + deep features + leave trimming)
**Hyperparameters**: max_depth=10, n_estimators=200, StandardScaler, class_weight="balanced"
**Feature count**: 218
**Training data**: Modern era only (>=2025-01-01), NO per-player cap, aurora_id-based, min 20 games

**Key changes from Exp 14:**
1. **No per-player cap** — uses all available games with `class_weight="balanced"` to handle imbalance
2. **18 new players** added since Exp 14: NaDa, herO, Noel, Mong, Sharp, Movie, Motive, Quickly, MadYellow, ZeloT (+ others from prior sessions)
3. Shuttle skipped again (18 games < 20 min threshold)

| Metric | Value |
|--------|-------|
| Samples | 6,960 (43 players) |
| Accuracy | **99.6%** (6933/6960) |

**Per-player accuracy:**
- 100%: 30/43 players (Shine, Sky, Air, Rush, Free, soma, RoyaL, Speed, Rain, Light, Noel, BishOp, Artosis, sSak, Jaedong, SoulKey, Motive, Scan, Tyson, Rich, Stork, Sharp, ZeloT, HyuK, MadYellow, gypsy, n00b, snOw, NaDa, Fantasy)
- 99.8%: soO (809/811)
- 99.5%: EffOrt (183/184)
- 99.4%: Best (166/167)
- 99.3%: Dewalt (134/135)
- 99.2%: Ample (123/124)
- 98.9%: Larva (187/189)
- 98.8%: Movie (82/83), Quickly (160/162), Flash (80/81)
- 98.4%: ProMise (127/129)
- 97.9%: yOOn (506/517)
- 97.1%: Mong (34/35)
- 95.8%: herO (23/24)

**Misclassifications: 27**
Notable confusion patterns:
- yOOn → Sky (×10) — persistent, both Protoss with similar style
- soO → soma (×2) — both top Zerg
- Larva → soma/soO (×2) — Zerg confusion
- ProMise → Quickly/Artosis (×2)
- Quickly → Noel/Sky (×2)

**Top features:**
1. `ehkg2_4_4` (0.026) - early hotkey group 4 double-tap
2. `first_assign_0` (0.026) - first hotkey assignment
3. `ehkg2_3_3` (0.025) - early hotkey group 3 double-tap
4. `ng2_H_Prod` (0.024) - Hotkey→Produce pattern
5. `ehkg2_1_2` (0.019) - early hotkey group 1→2 transition

**Notes:**
- Accuracy held at 99.6% despite nearly tripling player count (from 33 to 43) and 2.4× more samples
- All 18 new players integrated cleanly — 13 at 100%, lowest is herO (95.8%, only 24 games)
- BishOp improved from previous sessions: 153/153 = 100% despite K_JyJ4 handle (61 games) being half his data
- yOOn↔Sky confusion remains the hardest problem (10 of 27 misclassifications)

---

## Experiment 16: Zerg-Only Raw N-gram Ablation Study
**Date**: Feb 2026
**Question**: Does using raw command types (Unit Morph, Burrow, etc.) instead of abstracted categories (Prod, Abl) improve Zerg classification?
**Hypothesis**: Since all players are Zerg, race-neutral abstraction is unnecessary and may discard Zerg-specific signal (larva inject rhythms, burrow habits, morph patterns).

**Setup**: Two LOO CV runs on identical Zerg-only data:
- **Run A (baseline)**: Abstracted n-grams (current production approach)
- **Run B (experimental)**: Raw command-type n-grams (no abstraction, no consecutive Prod collapse)
- Non-n-gram features (timing, hotkeys, APM, deep features) identical in both runs
- Same n-gram budget: top 25/20/15 for 2/3/4-grams
- Hotkey group n-grams (hkg, ehkg) included identically in both runs
- class_weight="balanced", max_depth=10, n_estimators=200

**Players** (Zerg mains with 100+ Zerg games, 7 total):
| Player | Zerg Games |
|--------|-----------|
| soO | 811 |
| yOOn | 471 |
| Shine | 347 |
| Larva | 189 |
| soma | 170 |
| EffOrt | 141 |
| Jaedong | 139 |

| Metric | Abstracted | Raw | Delta |
|--------|-----------|-----|-------|
| Samples | 2,268 | 2,268 | — |
| Accuracy | **100.0%** | **100.0%** | 0 |
| Features | 201 | 171 | -30 |
| N-gram features | 103 | 73 | -30 |
| Misclassifications | 0 | 0 | 0 |

Per-player: All 7 players at 100.0% in both runs.

**Top features (both runs)**: Entirely hotkey patterns — no action n-grams in top 15:
| Rank | Abstracted | Raw |
|------|-----------|-----|
| 1 | ehkg2_3_3 (0.037) | ehkg2_3_3 (0.038) |
| 2 | ehkg2_5_5 (0.032) | ehkg2_4_4 (0.035) |
| 3 | ehkg2_5_3 (0.032) | first_assign_0 (0.033) |
| 4 | ehkg3_3_5_3 (0.031) | early_rapid_ratio (0.032) |
| 5 | ehkg2_4_4 (0.031) | ehkg2_3_5 (0.029) |

**Sanity check** (--max-games 30, 210 samples): Both runs 99.5% (209/210), same single misclassification (soO→soma).

**Conclusions:**
1. **Abstraction does NOT hurt Zerg classification.** Raw Zerg-specific commands (Unit Morph, Burrow, etc.) add zero discriminative signal beyond what hotkey patterns already provide.
2. **Hotkey muscle memory dominates.** The top 15 features in both runs are exclusively early-game hotkey group patterns and timing features. Action n-grams are dead weight for this 7-player Zerg subset.
3. **soO↔soma confusion disappears** at full data. The 2 soO→soma misclassifications in the 43-player model (Exp 15) are likely caused by cross-race/cross-class noise, not missing Zerg-specific features.
4. **Fewer features, same accuracy.** Raw mode actually produces 30 fewer features (larger vocabulary → top-N spreads thinner) with no accuracy penalty — confirming the action n-grams are irrelevant here.
5. **Per-race models are not needed for Zerg.** The current abstracted approach already achieves perfect Zerg separation. The "Per-race models" idea from the Ideas list can be deprioritized.

---

## Experiment 17: Enhanced Hotkey Features (yOOn↔Sky Investigation)
**Date**: Feb 2026
**Goal**: Reduce yOOn↔Sky confusion (10 of 27 misclassifications in Exp 15)

**Investigation**: Extracted detailed hotkey data from yOOn's 10 misclassified Protoss games, his correctly classified Protoss games, his Zerg games, and Sky's Protoss games. Found:
- **All 10 misclassified games are Protoss** — yOOn's group 1/3 usage ratio flips in these games (group 3 dominates at 41% vs normally 35%), making his distribution look like Sky's (32%/32%)
- **Assign order stays distinct**: yOOn always [1, 3, 2], Sky always [1, 2, 3] — but we only captured `first_assign_0` (both = 1)
- **Timing stays distinct**: yOOn double-taps group 3 in ~151ms (Sky ~262ms), yOOn switches groups in ~160ms (Sky ~223ms) — not captured at all

**New features added to features.py (11 total):**
1. `dt_median_g0` through `dt_median_g5` — per-group double-tap timing (median ms between consecutive same-group presses, groups 0-5)
2. `group_switch_median`, `group_switch_mean` — time to switch between different control groups
3. `assign_order_1`, `assign_order_2`, `assign_order_3` — 2nd/3rd/4th control groups assigned (extending existing `first_assign_0`)

| Metric | Exp 15 | Exp 17 | Delta |
|--------|--------|--------|-------|
| Samples | 6,960 | 6,960 | — |
| Features | 218 | 229 | +11 |
| Accuracy | **99.6%** (6933) | **99.6%** (6933) | 0 |
| Misclassifications | 27 | 27 | 0 |
| yOOn→Sky errors | 10 | 3 | **-7** |

**Per-player changes (vs Exp 15):**
- yOOn: 506/517 → 510/517 (97.9% → 98.6%, +0.7%) — **7 fewer Sky confusions**, but picked up 4 new errors (→ProMise, →Free, →BishOp)
- Flash: 80/81 → 81/81 (98.8% → 100%) — fixed
- soO: 809/811 → 808/811 — 1 new miss
- Dewalt: 134/135 → 133/135 — 1 new miss
- Larva: 187/189 → 186/189 — 1 new miss
- sSak: 159/159 → 158/159 — 1 new miss
- snOw: 36/36 → 35/36 — 1 new miss

**Top features:**
1. `ehkg2_4_4` (0.030)
2. `first_assign_0` (0.025)
3. `ehkg2_3_3` (0.022)
4. `ng2_H_Prod` (0.021)
5. `ehkg3_3_3_3` (0.017)
...
9. **`assign_order_1` (0.016)** ← NEW, confirms 2nd assign group is discriminative

**Conclusions:**
- The yOOn→Sky systematic confusion (10 games) is largely resolved — down to 3 scattered errors
- `assign_order_1` proved itself as a real feature (top 15 importance)
- Total accuracy unchanged at 99.6% — the "error budget" redistributed rather than shrunk
- The model is likely at its noise floor for LOO CV with Random Forest at this feature set
