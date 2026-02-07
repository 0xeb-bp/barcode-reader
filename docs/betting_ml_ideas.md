# ML Ideas for Sportsbook Odds Analysis

## Data: time series of odds across N sportsbooks for each market

## 1. Sportsbook Sharpness Ranking (no ML needed)
- Compare each book's **closing line** to actual outcome
- Score accuracy over thousands of markets (Brier score or log-loss)
- Rank books by accuracy — sharp books have best closing lines
- Gives you a trust weight per book

## 2. Lead-Lag Analysis (time series / correlation)
- **Cross-correlation**: shift Book B's movements in time, find when they best correlate with Book A. Tells you "Book B follows Book A by ~3 minutes"
- **Granger causality**: statistically tests if Book A's movement *predicts* Book B's future movement
- Output: directed graph / pecking order (e.g. FD -> DK -> BetMGM)

## 3. Stale Line Detection (anomaly detection)
- Sharp book moves, soft book hasn't yet = stale line = +EV window
- Track spread between sharp consensus and each soft book
- Flag when gap exceeds threshold
- ML enhancement: "given this pattern of sharp movement, probability soft book adjusts within X minutes?"

## 4. Movement Prediction (ML)
- **Features**: current odds across all books, velocity of recent changes, spread between books, time to event, how many books have moved, direction of movement
- **Target**: classification ("will this move +/- in next 10 min?") or regression ("what will closing line be?")
- **Models**: XGBoost, LSTM/RNN for sequential data
- Essentially market microstructure analysis (same as quant finance on order books)

## Comparison to StarCraft Project
| | StarCraft | Betting |
|---|---|---|
| Data | Action sequences | Price time series |
| Problem | Classification (who is this?) | Forecasting (where is this going?) |
| Features | N-grams, timing patterns | Spreads, velocity, book agreement |
| Model | Random Forest | XGBoost, time series models |
| Label | Player name | Actual outcome / closing line |
