# Feature Reference

All features extracted from a single replay for one player. Each replay = one row in the training data.

## Base Features (~40)

### Timing (7 features)

Computed from the time gaps between every consecutive action.

| Feature | Formula | What it means |
|---------|---------|---------------|
| `gap_mean` | sum(gaps) / count | Average time between actions (ms) |
| `gap_std` | standard deviation of gaps | How much timing varies |
| `gap_median` | middle value when sorted | "Typical" gap (ignores outliers) |
| `rapid_ratio` | count(gaps < 100ms) / total | % of spam-speed actions |
| `moderate_ratio` | count(100 <= gaps < 300) / total | % of deliberate-speed actions |
| `slow_ratio` | count(gaps >= 500ms) / total | % of long pauses |
| `burstiness` | gap_std / gap_mean | Bursty (<1 steady, >1 spiky) |

**Example**: gaps = [50, 80, 60, 200, 50, 70, 300, 50, 60, 80]
- mean = 100ms, median = 65ms, std = 79ms
- rapid_ratio = 7/10 = 0.70 (most actions are fast)
- burstiness = 79/100 = 0.79

**Why it's a fingerprint**: mean vs median gap reveals if a player is steady (mean ≈ median) or bursty (mean >> median). Some pros are metronomes, others spam then pause.

---

### Hotkey Patterns (5 features)

From all hotkey presses (Ctrl+number = assign, just number = recall).

| Feature | Formula | What it means |
|---------|---------|---------------|
| `hotkey_diversity` | count of unique groups used | How many control groups (1-10) |
| `hotkey_concentration` | top_2_groups / total_presses | Reliance on top 2 groups |
| `hotkey_assign_ratio` | assigns / total_presses | How often they reassign groups |
| `hotkey_action_ratio` | hotkey_presses / all_actions | What fraction of actions are hotkeys |
| `primary_hotkey_group` | most-pressed group number | Their "main" group (0-9) |

**Example**: 20 hotkey presses using groups {1, 3, 5}, group 1 pressed 10 times
- diversity = 3, concentration = 17/20 = 0.85
- primary_hotkey_group = 1

---

### Hotkey Transition Matrix (variable, up to 25 features)

After pressing group X, what group do they press next? Conditional probability.

| Feature | Formula | What it means |
|---------|---------|---------------|
| `hk_tr_X_Y` | count(X→Y) / count(all transitions from X) | Probability of going from group X to group Y |

**Example**: From the sequence [1, 1, 3, 1, 3, 5, 1]:
```
After 1: → 1 (1x), → 3 (2x), → 5 (0x)  →  hk_tr_1_1=0.33, hk_tr_1_3=0.67
After 3: → 1 (1x), → 5 (1x)              →  hk_tr_3_1=0.50, hk_tr_3_5=0.50
After 5: → 1 (1x)                         →  hk_tr_5_1=1.00
```

**Why it's a fingerprint**: Each player has a unique "circuit" they cycle through. Flash might always go 1→3→1→3 (army↔production). Jaedong might go 1→2→3→1→2→3 (cycle all bases).

---

### Hotkey Group N-grams (variable, ~16 per scope)

Sliding window over the sequence of group numbers pressed.

| Feature | Formula | What it means |
|---------|---------|---------------|
| `hkg2_X_Y` | count(X,Y pair) / total_bigrams | Full game: how often they press group X then Y |
| `hkg3_X_Y_Z` | count(X,Y,Z triple) / total_trigrams | Full game: three-press sequence |
| `ehkg2_X_Y` | same but first 2 minutes only | Opening routine bigram |
| `ehkg3_X_Y_Z` | same but first 2 minutes only | Opening routine trigram |

Top 8 per n-gram size kept per game. The `e` prefix = early game (first 2 min, pure muscle memory).

---

### Click Movement (5 features)

Euclidean distance between consecutive click positions on the map.

| Feature | Formula | What it means |
|---------|---------|---------------|
| `click_dist_mean` | avg of all consecutive click distances | Average mouse jump (pixels) |
| `click_dist_std` | std of distances | How varied the jumps are |
| `click_dist_median` | median distance | Typical jump size |
| `small_move_ratio` | count(dist < 100px) / total | % of tiny adjustments |
| `big_jump_ratio` | count(dist > 1000px) / total | % of cross-map jumps |

**Distance formula** (Pythagorean theorem):
```
distance = sqrt((x2-x1)^2 + (y2-y1)^2)
```

**Example**: clicks at (100,200), (120,210), (900,400)
- distance 1→2: sqrt(20^2 + 10^2) = sqrt(500) ≈ 22px (small adjustment)
- distance 2→3: sqrt(780^2 + 190^2) = sqrt(644500) ≈ 803px (map jump)

**Note**: These are map coordinates, not screen coordinates. A hotkey camera jump followed by a local click shows as a big distance even though the mouse barely moved on screen.

---

### Early Game (6 features)

First 2 minutes only. Pure opening routine / muscle memory.

| Feature | Formula | What it means |
|---------|---------|---------------|
| `early_apm` | early_actions / 2.0 | Opening speed |
| `early_gap_mean` | mean of early gaps | Opening rhythm |
| `early_rapid_ratio` | early rapid actions / total early | Opening spam speed |
| `first_assign_0` | first Ctrl+group pressed (0-9, or -1) | First control group assigned |
| `first_assign_1` | second Ctrl+group | Second group assigned |
| `first_assign_2` | third Ctrl+group | Third group assigned |

**Why it's a fingerprint**: The opening 2 minutes is autopilot. You always Ctrl+1 the same thing first, always set up groups the same way. `first_assign_0` is consistently the #1 most important feature in our models.

---

### General Behavior (11 features)

| Feature | Formula | What it means |
|---------|---------|---------------|
| `apm` | total_actions / game_minutes | Overall actions per minute |
| `apm_decay` | (early_avg - late_avg) / early_avg | How much speed drops over the game |
| `apm_variance` | std of per-minute APM buckets | How consistent speed is |
| `queued_ratio` | shift_queued_actions / total | How much they use shift-queue |
| `select_size_mean` | avg units per box-select | Typical selection size |
| `select_size_std` | std of selection sizes | How varied selections are |
| `selection_action_ratio` | selections / total_actions | What fraction of actions are selections |
| `pct_hotkey` | hotkey_actions / total | % of actions that are hotkey presses |
| `pct_right_click` | right_clicks / total | % that are right clicks |
| `pct_select` | selections / total | % that are box selections |
| `pct_targeted_order` | targeted_orders / total | % that are ability/spell casts |
| `think_do_ratio` | (hotkey+select) / (rightclick+targeted_order) | Thinking vs executing ratio |

---

## N-gram Features (~variable, typically 150-200)

### Action N-grams

Sliding window over the abstracted action sequence.

| Feature | Example | What it means |
|---------|---------|---------------|
| `ng2_H_S` | Hotkey then Select | Checking group then box-selecting |
| `ng3_R_R_R` | Triple right-click | Move-spam pattern |
| `ng4_H_H_H_H` | Quad hotkey | Obsessive group cycling |

**Abstraction mapping** (to prevent race leakage):
```
Race-neutral (kept distinct):
  Hotkey → H, Select → S, Right Click → R, Targeted Order → TO
  Hold Position → HP, Stop → ST, Return Cargo → RC
  Select Add → S+, Select Remove → S-

Race-specific (collapsed to generic):
  Train, Unit Morph, Building Morph, Train Fighter → Prod
  Build → Bld
  Stim, Burrow, Unburrow, Siege, Unsiege, Cloack, Decloack,
  Merge Archon, Merge Dark Archon, Lift Off, Land → Abl
  Upgrade → Upg, Tech → Tech
  Unload, Unload All → Unld / UnldA
  All Cancels → Can
```

Without abstraction, a Zerg "Unit Morph → Hotkey" and Terran "Train → Hotkey" would be different features even though the player did the same thing (produce unit, check group). The abstraction makes them both "Prod → H" so the model compares rhythm, not race.

Top 10 per game per n-gram size (2, 3, 4). Values are normalized to proportions.

---

## How These Become a Training Matrix

Each replay produces one row. All rows are stacked:

```
              gap_mean  apm  hk_tr_1_3  ng2_H_S  ng3_R_R_R  ...  (570 columns)
Flash game1:    95     312    0.44       0.15      0.08
Flash game2:    88     298    0.41       0.16      0.09
Bisu game1:    140     220    0.00       0.22      0.03
Jaedong game1:  70     350    0.55       0.11      0.12
...
```

Fixed-name features (gap_mean, apm, etc.) always have the same columns.
Variable-name features (n-grams, transitions) get unioned - if a feature doesn't exist for a game, it gets 0.

## How the Random Forest Uses Them

At each node in a decision tree:
1. Try every feature and every possible threshold between sorted values
2. For each candidate split, measure Gini impurity (how mixed the groups are)
3. Pick the split with lowest impurity (cleanest separation)
4. Recurse on each side until max_depth or pure leaves

**Gini impurity**: `gini = 1 - (frac_A^2 + frac_B^2 + ... + frac_N^2)`
- 0.0 = perfectly pure (all one player)
- 0.5 = maximally mixed (two players 50/50)

The Random Forest trains 200 trees, each on a random subset of rows and features, then takes a majority vote. This makes it robust to noise and overfitting.

## Current Results

**Experiment 8** (Feb 2026): 302 samples, 14 pros, 570 features → **88.4% accuracy**

Top features by importance:
1. `first_assign_0` (0.018) - first hotkey group assigned
2. `ehkg2_4_4` (0.017) - early game group 4 double-tap
3. `hk_tr_4_3` (0.015) - transition group 4 → group 3
4. `ehkg2_3_3` (0.014) - early game group 3 double-tap
5. `hotkey_assign_ratio` (0.011) - assign vs recall frequency
