# ML Concepts Reference

## Feature Space
Each sample (a single game's extracted features) is a point in N-dimensional space, where N = number of features. For us, N=199. Each axis is one feature (e.g., APM, gap_mean, hotkey n-grams). You can't visualize 199 dimensions, but the math works identically to 2D/3D.

A player's games form a **cluster** of points in this space. Players with similar playstyles have overlapping clusters; players with distinct styles have well-separated clusters.

## Centroid
The **centroid** is the average position of all points in a cluster — the "center of mass." Computed by averaging each feature across all samples in the class.

For a class with samples at positions [2, 4, 6] on one axis, the centroid on that axis = (2+4+6)/3 = 4. This extends to all dimensions.

Used in: k-means clustering (assigns points to nearest centroid), outlier detection, novelty detection.

## Standard Deviation (std)
Measures how spread out values are from the mean.

**Steps to calculate:**
1. Compute the mean: `mean = sum(values) / n`
2. Compute each value's squared difference from the mean: `(x - mean)^2`
3. Average those squared differences: `variance = sum((x - mean)^2) / n`
   - Note: for *sample* std dev (what sklearn/statistics use by default), divide by `n-1` instead of `n`. This corrects for bias when estimating from a sample rather than a full population.
4. Take the square root: `std = sqrt(variance)`

**Example:** values = [2, 4, 4, 4, 5, 5, 7, 9]
- mean = 40/8 = 5
- squared diffs: [9, 1, 1, 1, 0, 0, 4, 16] = sum 32
- variance = 32/7 = 4.57 (sample variance, n-1)
- std = sqrt(4.57) = 2.14

**Intuition:** std is roughly "the typical distance a value sits from the mean." Low std = tight cluster. High std = spread out.

## Z-Score
**"How many standard deviations is this value from the mean?"**

`z = (x - mean) / std`

- z = 0 means exactly at the mean
- z = 1 means 1 std dev above the mean
- z = -1 means 1 std dev below the mean
- z > 2.5 is unusual; z > 3 is very unusual

This is a general statistics concept — works on any values, not just distances. Used for: outlier detection, comparing values across different scales, hypothesis testing.

### How we use z-scores for outlier detection
1. Scale all features (StandardScaler: zero mean, unit variance per feature)
2. For each player class, compute the centroid (mean of all that player's feature vectors)
3. Compute each sample's Euclidean distance to its class centroid
4. Those distances form their own distribution with a mean and std
5. Z-score = `(sample_distance - mean_distance) / std_distance`
6. High z-score = the sample is unusually far from its class center compared to typical samples

A z of 8.0 doesn't mean "8 std devs from the centroid" — it means the sample's distance from the centroid is 8 std devs above the *average distance* that class's samples sit from the centroid.

## Euclidean Distance
Straight-line distance between two points in feature space. In 2D: `sqrt((x2-x1)^2 + (y2-y1)^2)`. Extends to N dimensions by adding more squared terms.

## Mahalanobis Distance
A more robust alternative to Euclidean distance that accounts for the **shape and correlation** of the cluster. Euclidean treats all directions equally; Mahalanobis stretches/compresses based on how the data is actually distributed. More expensive to compute, but better when features are correlated.

## StandardScaler
Transforms each feature to have mean=0 and std=1. Important because features on different scales (APM ~200 vs. hotkey_ratio ~0.3) would otherwise dominate distance calculations. After scaling, all features contribute equally to distances.
