# Unsupervised Learning & K-Means — Complete Beginner Guide
### Tied to your Customer Segmentation Notebook

---

## Part 1: What is Unsupervised Learning?

### Supervised vs Unsupervised — The Core Difference

In **supervised learning**, you give the algorithm both the questions AND the answers, and it learns the pattern.

```
Supervised:
  Input: [house size, bedrooms, age]  →  Label: $400,000
  Input: [email text]                 →  Label: "spam"
  The algorithm learns: question → answer
```

In **unsupervised learning**, you give the algorithm ONLY the data — no answers, no labels. It must find hidden structure on its own.

```
Unsupervised:
  Input: [income, spending_score]  →  No label given
  The algorithm figures out: "these customers seem to form 5 natural groups"
```

### Java Analogy

```
Supervised   = a student with a textbook that has answers at the back
Unsupervised = a student given 1000 documents with no answers — must
               figure out which ones are similar and group them
```

---

### When Do You Use Unsupervised Learning?

| Situation | Example |
|---|---|
| No labels exist | Customer data with no pre-assigned segments |
| Too expensive to label | Millions of images — labeling each costs money |
| Exploring new data | You don't know what patterns exist yet |
| Compression / simplification | Reduce 100 features to 3 important ones |

---

### Types of Unsupervised Learning

```
Unsupervised Learning
├── Clustering          → group similar things together
│   ├── K-Means         ← what your notebook uses
│   ├── DBSCAN
│   └── Hierarchical Clustering
│
├── Dimensionality Reduction → compress many features into fewer
│   ├── PCA (Principal Component Analysis)
│   └── t-SNE
│
└── Association Rules   → find "if X then Y" patterns
    └── Apriori (e.g. "customers who buy bread also buy butter")
```

---

## Part 2: K-Means Clustering

### What K-Means Does in Plain English

K-Means looks at data points and automatically groups them into **K clusters** — where each cluster contains points most similar to each other and most different from other clusters.

In your notebook:
- Each data point = one customer (described by `Annual_Income` and `Spending_Score`)
- K-Means finds 5 natural customer groups without being told what those groups are
- No one said "these are high spenders" — the algorithm discovered it purely from numbers

### The Shopping Mall Analogy

Imagine a shopping mall manager wants to group 200 customers by behaviour. She has no categories in mind — she just has income and spending data.

She asks K-Means: *"Find me 5 natural groups in this data."*

K-Means returns:
```
Group 1 → Low income, Low spending    → Budget shoppers
Group 2 → Low income, High spending   → Impulsive shoppers
Group 3 → Medium income, Medium spend → Average shoppers
Group 4 → High income, Low spending   → Wealthy but careful
Group 5 → High income, High spending  → Premium customers
```

No one told it these groups existed. It found them from raw numbers. That is the power of unsupervised learning.

---

## Part 3: How K-Means Works Step by Step

### The Core Idea (No Math Yet)

Imagine dropping 5 pins randomly on a map of customer dots:

```
Step 1 — Place pins randomly:
  · · · · × · ·
  × · · · · · ·
  · · · × · · ·
  · · · · · × ·
  · × · · · · ·
  (× = pin/centroid, · = customer)

Step 2 — Each customer moves to their nearest pin:
  Every dot picks the closest pin and joins that group

Step 3 — Each pin moves to the centre of its group:
  The pin repositions itself at the average location of all its members

Step 4 — Repeat Steps 2 and 3 until nothing changes
```

When no customer switches groups between iterations — the algorithm has converged and stops.

---

### Formula 1: Euclidean Distance (Assigning Customers to Clusters)

K-Means has one job at its core: measure how far each customer is from each cluster centre.

```
d(A, B) = √( (x₂ - x₁)² + (y₂ - y₁)² )
```

Where:
- `A` = the customer point        → (Annual_Income, Spending_Score)
- `B` = the cluster centre        → (centroid_income, centroid_spending)
- `d` = straight-line distance between them

**Concrete example from your data:**

```
Customer A:   Annual_Income = 60,  Spending_Score = 80
Centroid 1:   Annual_Income = 55,  Spending_Score = 78
Centroid 2:   Annual_Income = 90,  Spending_Score = 20

Distance to Centroid 1 = √( (60-55)² + (80-78)² )
                       = √( 25 + 4 )
                       = √29
                       ≈ 5.4  ← closer

Distance to Centroid 2 = √( (60-90)² + (80-20)² )
                       = √( 900 + 3600 )
                       = √4500
                       ≈ 67.1

→ Customer A is assigned to Centroid 1  (5.4 < 67.1)
```

Every customer is assigned this way — always goes to the nearest centroid.

---

### Formula 2: Updating the Centroid (Moving the Pin)

After all customers are assigned, each centroid moves to the **mean position** of its members.

```
new centroid = ( mean of all x values in cluster,  mean of all y values in cluster )
```

**Example:**

```
Cluster 1 members after assignment:
  Customer A: (60, 80)
  Customer B: (65, 72)
  Customer C: (58, 85)

New centroid x = (60 + 65 + 58) / 3 = 183 / 3 = 61.0
New centroid y = (80 + 72 + 85) / 3 = 237 / 3 = 79.0

New centroid = (61.0, 79.0)
```

The centroid repositions itself to the true centre of its members. Then customers are reassigned again. This repeats.

![KMeans flow](kmeans_customer_segmentation_flowchart.svg)

---

### Formula 3: Inertia / WCSS (How the Algorithm Knows When to Stop)

K-Means tracks a score called **inertia** (also called WCSS — Within Cluster Sum of Squares):

```
Inertia = sum of ( distance from each customer to its centroid )²
          across all customers
```

When inertia stops decreasing between iterations → the algorithm has converged → it stops.

**Lower inertia = tighter, more compact clusters = better fit.**

---

## Part 4: The Elbow Method — Choosing K

The biggest question in K-Means: **how many clusters should I use?**

You cannot let K-Means decide — you must tell it. The **Elbow Method** helps you choose the right K.

### How It Works

Run K-Means with K = 1, 2, 3, 4 ... 10. Plot the inertia for each:

```
Inertia
  |
  |*                         ← K=1: one big cluster, very high inertia
  |  *
  |    *
  |      *  ← ELBOW here     ← K=5: adding more clusters gives little gain
  |         *
  |           * * * * *      ← K=8+: clusters getting too small, diminishing returns
  |
  +─────────────────────── K
    1  2  3  4  5  6  7  8
```

Look for the "elbow" — the point where the curve bends and flattens. That is your optimal K.

In your notebook, the elbow is at **K=5**, so `n_clusters=5` was chosen → 5 customer segments.

---

## Part 5: The 5 Steps K-Means Runs in Your Notebook

```python
model = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = model.fit_predict(X_scaled)
```

That single `fit_predict` runs all 5 steps internally:

| Step | What Happens | Formula Used |
|---|---|---|
| 1. Initialise | Place 5 centroids randomly in the data space | `random_state=42` fixes starting positions |
| 2. Assign | Every customer assigned to nearest centroid | Euclidean distance d = √(Δx² + Δy²) |
| 3. Update | Each centroid moves to mean of its members | centroid = (mean x, mean y) |
| 4. Repeat | Steps 2 and 3 repeat until nothing changes | Inertia stops decreasing |
| 5. Output | Each customer gets a cluster label 0–4 | Stored in `df['Cluster']` |

---

## Part 6: What Your Notebook Produces

From `df.groupby('Cluster').mean()`, the five segments typically resolve to:

| Cluster | Annual Income | Spending Score | Customer Type | Business Action |
|---|---|---|---|---|
| 0 | Low | Low | Budget-conscious | Discounts, value deals |
| 1 | Low | High | Impulsive spenders | Promotions, flash sales |
| 2 | Medium | Medium | Average balanced | Loyalty programs |
| 3 | High | Low | Wealthy but careful | Premium quality messaging |
| 4 | High | High | Premium customers | VIP programs, exclusivity |

K-Means discovered these groups purely from numbers — no one told it these categories existed. That is unsupervised learning in action.

---

## Part 7: Important Things to Remember

### K Must Be Chosen by You
The algorithm does not figure out how many clusters to use. You pass `n_clusters=5`. If you pass `n_clusters=3` you get 3 segments. Use the Elbow Method to find the right number.

### Cluster Numbers Mean Nothing
Cluster 0 is not "better" or "first" in any meaningful sense. The numbers 0–4 are just arbitrary labels. Cluster 3 could be your most valuable customers — the number has no ranking.

### Scale Matters — Always Scale Before K-Means
If one feature is in thousands (`Annual_Income`: 15,000–130,000) and another is 1–100 (`Spending_Score`), the distance formula is dominated by income. Spending Score barely influences the result.

```
Without scaling:
  Income difference: 82,000 - 80,000 = 2,000   ← dominates
  Spending difference:       60 - 20 =    40    ← ignored

With StandardScaler applied first:
  Income difference:   0.9 - 0.8 = 0.1          ← equal weight
  Spending difference: 0.8 - 0.2 = 0.6          ← equal weight
```

Your notebook applies `StandardScaler` before K-Means — this is the correct approach.

### random_state=42 Makes Results Reproducible
K-Means starts with random centroid positions, so without fixing the seed you could get slightly different clusters each run. Setting `random_state=42` locks the starting positions so results are identical every run.

---

## Part 8: K-Means vs Other Clustering Algorithms

| Algorithm | How It Works | Use When |
|---|---|---|
| **K-Means** | Groups by distance to centroid | Clusters are round/spherical, K is known |
| **DBSCAN** | Groups by density of points | Clusters are irregular shapes, K is unknown |
| **Hierarchical** | Builds a tree of clusters | You want to see all possible groupings |

K-Means is the most popular starting point because it is fast, simple, and works well on clean data.

---

## Part 9: K-Means Limitations

| Limitation | What It Means | Workaround |
|---|---|---|
| You must choose K | No automatic detection of cluster count | Use Elbow Method or Silhouette Score |
| Sensitive to outliers | One extreme value can drag a centroid | Remove outliers before clustering |
| Assumes round clusters | Struggles with irregular-shaped groups | Use DBSCAN instead |
| Random initialisation | Different runs can give different results | Set `random_state` for reproducibility |

---

## Part 10: Quick Reference

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Select features
X = df[['Annual_Income', 'Spending_Score']].copy()

# 2. Scale (always do this before K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Find best K using Elbow Method
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# 4. Train with chosen K
model = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = model.fit_predict(X_scaled)

# 5. Analyse results
print(df.groupby('Cluster').mean())
```

---

## Summary

```
Unsupervised Learning
  └── No labels, algorithm finds hidden structure

K-Means
  └── Finds K groups by minimising distance between points and centroids

Key formulas:
  └── Distance:  d = √( (x₂-x₁)² + (y₂-y₁)² )
  └── Centroid:  new_centre = (mean x, mean y) of all members
  └── Inertia:   sum of squared distances (stops when this converges)

Key decisions:
  └── Choose K using the Elbow Method
  └── Always scale features before running K-Means
  └── Set random_state for reproducible results
```

*Reference: K-Means interactive visualisation → http://alekseynp.com/viz/k-means.html*
*Your notebook uses `sklearn.cluster.KMeans` with `Annual_Income` and `Spending_Score` as features and `n_clusters=5`.*
