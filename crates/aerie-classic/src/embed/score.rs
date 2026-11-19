//! Scoring embeddings (Mission 2): what can a listener read from Z?
//!
//! The plan's three tests. None of them looks at how Z was made, so DeepWalk, Node2Vec and
//! HOPE are all scored the same way:
//! - **Roles**: logistic regression on the vectors, 5-fold, macro-F1 (`role_f1`).
//! - **Links**: hide 10 % of the links *before* training (`hide_links`), embed what is
//!   left, then check that hidden pairs score higher than pairs that never talked
//!   (`link_auc`).
//! - **Squads**: k-means on the vectors, compared with the true squads by NMI (`kmeans`,
//!   `nmi`).
//!
//! Plain loops, no ML library.

use std::collections::HashSet;

use aerie_graph::EdgeList;
use rand::RngExt;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use super::skipgram::dot;

/// Each vector scaled to length 1 (a zero vector stays zero). Skip-gram's similarity is a
/// direction, so k-means runs on these.
pub fn normalized(z: &[Vec<f64>]) -> Vec<Vec<f64>> {
    z.iter()
        .map(|row| {
            let len = dot(row, row).sqrt();
            if len > 0.0 {
                row.iter().map(|x| x / len).collect()
            } else {
                row.clone()
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------------------
// Roles: logistic regression, cross-validated, macro-F1
// ---------------------------------------------------------------------------------------

/// Macro-F1: the F1 score of each class, averaged with equal weight, so a rare class
/// (compute drones) counts as much as a common one (recon).
/// F1 = 2·TP / (2·TP + FP + FN). Classes that appear in neither `truth` nor `pred` are
/// left out of the average.
pub fn macro_f1(truth: &[usize], pred: &[usize], classes: usize) -> f64 {
    assert_eq!(truth.len(), pred.len());
    let mut tp = vec![0usize; classes];
    let mut fp = vec![0usize; classes];
    let mut fn_ = vec![0usize; classes];
    for (&t, &p) in truth.iter().zip(pred) {
        if t == p {
            tp[t] += 1;
        } else {
            fp[p] += 1;
            fn_[t] += 1;
        }
    }
    let f1: Vec<f64> = (0..classes)
        .filter(|&c| tp[c] + fp[c] + fn_[c] > 0)
        .map(|c| 2.0 * tp[c] as f64 / (2 * tp[c] + fp[c] + fn_[c]) as f64)
        .collect();
    f1.iter().sum::<f64>() / f1.len() as f64
}

/// L2 penalty on the weights (not the bias), per drone: the loss is the mean
/// cross-entropy + `L2 / 2 · ‖W‖²`.
const L2: f64 = 1e-3;
/// Gradient steps. The step size comes from the data (see `fit`), so this is enough to
/// settle on every feature set in this project.
const STEPS: usize = 500;

/// Multinomial logistic regression: a score `w_c · x + b_c` per class, softmax, the class
/// with the highest score wins. Trained by full-batch gradient descent. Features are
/// standardised with the training drones' mean and spread, so their units don't matter.
pub struct LogisticRegression {
    mean: Vec<f64>,
    spread: Vec<f64>,
    /// One row per class: the weights, then the bias.
    weights: Vec<Vec<f64>>,
}

impl LogisticRegression {
    /// Fits on `x` (one feature vector per drone) and `y` (class of each, below `classes`).
    ///
    /// The step is 1 / (½·λ_max + L2), λ_max the largest eigenvalue of XᵀX/n: the softmax
    /// loss never curves more than that, so gradient descent can't overshoot.
    pub fn fit(x: &[Vec<f64>], y: &[usize], classes: usize) -> Self {
        let n = x.len() as f64;
        let d = x[0].len();
        let mut mean = vec![0.0; d];
        for row in x {
            for (m, v) in mean.iter_mut().zip(row) {
                *m += v / n;
            }
        }
        let mut spread = vec![0.0; d];
        for row in x {
            for ((s, v), m) in spread.iter_mut().zip(row).zip(&mean) {
                *s += (v - m) * (v - m) / n;
            }
        }
        for s in &mut spread {
            *s = if *s > 1e-24 { s.sqrt() } else { 1.0 };
        }
        let mut model = LogisticRegression {
            mean,
            spread,
            weights: vec![vec![0.0; d + 1]; classes],
        };

        let xs: Vec<Vec<f64>> = x.iter().map(|row| model.standardise(row)).collect();
        let step = 1.0 / (0.5 * top_eigenvalue(&xs) + L2);
        let mut p = vec![0.0; classes];
        for _ in 0..STEPS {
            let mut grad = vec![vec![0.0; d + 1]; classes];
            for (row, &label) in xs.iter().zip(y) {
                model.softmax(row, &mut p);
                for (c, g) in grad.iter_mut().enumerate() {
                    let err = p[c] - if c == label { 1.0 } else { 0.0 };
                    for (gj, v) in g.iter_mut().zip(row) {
                        *gj += err * v / n;
                    }
                }
            }
            for (w, g) in model.weights.iter_mut().zip(&grad) {
                for j in 0..=d {
                    let penalty = if j < d { L2 * w[j] } else { 0.0 };
                    w[j] -= step * (g[j] + penalty);
                }
            }
        }
        model
    }

    /// The class with the highest score.
    pub fn predict(&self, x: &[f64]) -> usize {
        let row = self.standardise(x);
        let scores: Vec<f64> = self.weights.iter().map(|w| dot(w, &row)).collect();
        argmax(&scores)
    }

    /// (x − mean) / spread, then a 1 for the bias.
    fn standardise(&self, x: &[f64]) -> Vec<f64> {
        let mut row: Vec<f64> = x
            .iter()
            .zip(&self.mean)
            .zip(&self.spread)
            .map(|((v, m), s)| (v - m) / s)
            .collect();
        row.push(1.0);
        row
    }

    /// Class probabilities for a standardised row, into `p`.
    fn softmax(&self, row: &[f64], p: &mut [f64]) {
        for (pc, w) in p.iter_mut().zip(&self.weights) {
            *pc = dot(w, row);
        }
        let top = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        for pc in p.iter_mut() {
            *pc = (*pc - top).exp(); // minus the top score: no overflow, same ratios
            total += *pc;
        }
        for pc in p.iter_mut() {
            *pc /= total;
        }
    }
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Largest eigenvalue of XᵀX/n by power iteration (a little above, to be safe).
fn top_eigenvalue(xs: &[Vec<f64>]) -> f64 {
    let n = xs.len() as f64;
    let d = xs[0].len();
    let mut v = vec![1.0 / (d as f64).sqrt(); d];
    let mut lambda = 0.0;
    for _ in 0..100 {
        let mut next = vec![0.0; d];
        for row in xs {
            let proj = dot(row, &v);
            for (nj, r) in next.iter_mut().zip(row) {
                *nj += proj * r / n;
            }
        }
        lambda = dot(&next, &next).sqrt();
        if lambda == 0.0 {
            break;
        }
        for (vj, nj) in v.iter_mut().zip(&next) {
            *vj = nj / lambda;
        }
    }
    lambda * 1.01
}

/// Splits drones into `k` folds, every class spread evenly (stratified): each class's
/// drones are shuffled and dealt round the folds, carrying on where the previous class
/// stopped. Returns the fold of each drone.
pub fn stratified_folds(y: &[usize], classes: usize, k: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let mut fold = vec![0; y.len()];
    let mut next = 0;
    for c in 0..classes {
        let mut members: Vec<usize> = (0..y.len()).filter(|&i| y[i] == c).collect();
        members.shuffle(rng);
        for i in members {
            fold[i] = next % k;
            next += 1;
        }
    }
    fold
}

/// Each drone's class as predicted by a model that never saw it: for each of the `k`
/// folds, train on the other folds and predict this one.
pub fn cross_val_predict(
    x: &[Vec<f64>],
    y: &[usize],
    classes: usize,
    k: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    let fold = stratified_folds(y, classes, k, rng);
    let mut pred = vec![0; y.len()];
    for f in 0..k {
        let train: Vec<usize> = (0..y.len()).filter(|&i| fold[i] != f).collect();
        let xt: Vec<Vec<f64>> = train.iter().map(|&i| x[i].clone()).collect();
        let yt: Vec<usize> = train.iter().map(|&i| y[i]).collect();
        let model = LogisticRegression::fit(&xt, &yt, classes);
        for i in (0..y.len()).filter(|&i| fold[i] == f) {
            pred[i] = model.predict(&x[i]);
        }
    }
    pred
}

/// The plan's role test: 5-fold logistic regression on `x`, macro-F1 over all drones'
/// out-of-fold predictions.
pub fn role_f1(x: &[Vec<f64>], roles: &[usize], classes: usize, rng: &mut ChaCha8Rng) -> f64 {
    let pred = cross_val_predict(x, roles, classes, 5, rng);
    macro_f1(roles, &pred, classes)
}

// ---------------------------------------------------------------------------------------
// Links: hide before training, AUC after
// ---------------------------------------------------------------------------------------

/// A graph with some links hidden, and the pairs to test on.
pub struct LinkSplit {
    /// What the embedding may see: every link but the hidden ones (same order as before).
    pub train: EdgeList,
    /// The hidden links (smaller id first): the positives.
    pub hidden: Vec<(usize, usize)>,
    /// As many pairs with no link in the full graph: the negatives.
    pub non_links: Vec<(usize, usize)>,
}

/// Hides `share` of the links (rounded), picked uniformly, and draws as many pairs that
/// never talked, uniformly among all such pairs. Hidden links may leave a drone with no
/// link at all; it then keeps its random start vector, which is what a listener who
/// missed all its traffic would get.
///
/// Panics if the graph has fewer non-links than links to hide.
pub fn hide_links(graph: &EdgeList, share: f64, rng: &mut ChaCha8Rng) -> LinkSplit {
    let n = graph.n as usize;
    let m = graph.edges.len();
    let k = (share * m as f64).round() as usize;
    assert!(n * (n - 1) / 2 - m >= k, "not enough pairs without a link");

    let mut order: Vec<usize> = (0..m).collect();
    order.shuffle(rng);
    let mut is_hidden = vec![false; m];
    for &e in &order[..k] {
        is_hidden[e] = true;
    }

    let mut train = Vec::with_capacity(m - k);
    let mut hidden = Vec::with_capacity(k);
    for (edge, &h) in graph.edges.iter().zip(&is_hidden) {
        if h {
            let (a, b) = (edge.src as usize, edge.dst as usize);
            hidden.push((a.min(b), a.max(b)));
        } else {
            train.push(edge.clone());
        }
    }

    let linked: HashSet<(usize, usize)> = graph
        .edges
        .iter()
        .map(|e| {
            let (a, b) = (e.src as usize, e.dst as usize);
            (a.min(b), a.max(b))
        })
        .collect();
    let mut picked = HashSet::new();
    let mut non_links = Vec::with_capacity(k);
    while non_links.len() < k {
        let a = rng.random_range(0..n);
        let b = rng.random_range(0..n);
        let pair = (a.min(b), a.max(b));
        if a != b && !linked.contains(&pair) && picked.insert(pair) {
            non_links.push(pair);
        }
    }

    LinkSplit {
        train: EdgeList {
            n: graph.n,
            edges: train,
        },
        hidden,
        non_links,
    }
}

/// Area under the ROC curve: the chance that a random positive scores above a random
/// negative, a tie counting half. 1 = perfect, 0.5 = guessing, 0 = exactly backwards.
/// Computed from ranks (Mann–Whitney): sort once instead of comparing every pair.
pub fn auc(pos: &[f64], neg: &[f64]) -> f64 {
    let mut all: Vec<(f64, bool)> = pos
        .iter()
        .map(|&s| (s, true))
        .chain(neg.iter().map(|&s| (s, false)))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j < all.len() && all[j].0 == all[i].0 {
            j += 1;
        }
        // Positions i..j share one score: each gets the mean of ranks i+1 ..= j.
        let mean_rank = (i + 1 + j) as f64 / 2.0;
        rank_sum += mean_rank * all[i..j].iter().filter(|e| e.1).count() as f64;
        i = j;
    }
    let (p, q) = (pos.len() as f64, neg.len() as f64);
    (rank_sum - p * (p + 1.0) / 2.0) / (p * q)
}

/// The plan's link test: score each pair by z_a · z_b (DeepWalk's decoder), hidden links
/// against non-links.
pub fn link_auc(z: &[Vec<f64>], split: &LinkSplit) -> f64 {
    let score = |&(a, b): &(usize, usize)| dot(&z[a], &z[b]);
    let pos: Vec<f64> = split.hidden.iter().map(score).collect();
    let neg: Vec<f64> = split.non_links.iter().map(score).collect();
    auc(&pos, &neg)
}

// ---------------------------------------------------------------------------------------
// Squads: k-means, NMI
// ---------------------------------------------------------------------------------------

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// k-means (Lloyd's algorithm) started by k-means++, run `restarts` times; keeps the run
/// with the smallest total squared distance to the centres. Returns each point's
/// cluster, 0..k.
pub fn kmeans(x: &[Vec<f64>], k: usize, restarts: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    assert!(k >= 1 && k <= x.len());
    let mut best: Option<(Vec<usize>, f64)> = None;
    for _ in 0..restarts.max(1) {
        let run = kmeans_once(x, k, rng);
        if best.as_ref().is_none_or(|b| run.1 < b.1) {
            best = Some(run);
        }
    }
    best.unwrap().0
}

/// One k-means run: (cluster of each point, total squared distance).
fn kmeans_once(x: &[Vec<f64>], k: usize, rng: &mut ChaCha8Rng) -> (Vec<usize>, f64) {
    let n = x.len();
    let d = x[0].len();

    // k-means++: the first centre uniform, each next one drawn in proportion to the squared
    // distance to the nearest centre so far, so the centres start spread out.
    let mut centres = vec![x[rng.random_range(0..n)].clone()];
    let mut nearest: Vec<f64> = x.iter().map(|p| squared_distance(p, &centres[0])).collect();
    while centres.len() < k {
        let total: f64 = nearest.iter().sum();
        let pick = if total > 0.0 {
            let mut r = rng.random_range(0.0..total);
            let mut pick = n - 1;
            for (i, &w) in nearest.iter().enumerate() {
                r -= w;
                if r < 0.0 {
                    pick = i;
                    break;
                }
            }
            pick
        } else {
            rng.random_range(0..n) // every point sits on a centre already
        };
        centres.push(x[pick].clone());
        for (near, p) in nearest.iter_mut().zip(x) {
            *near = near.min(squared_distance(p, &centres[centres.len() - 1]));
        }
    }

    let mut cluster = vec![usize::MAX; n];
    for _ in 0..300 {
        // Assign each point to its nearest centre.
        let mut changed = false;
        for (i, p) in x.iter().enumerate() {
            let c = argmax(
                &centres
                    .iter()
                    .map(|c| -squared_distance(p, c))
                    .collect::<Vec<_>>(),
            );
            if c != cluster[i] {
                cluster[i] = c;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Move each centre to the mean of its points. An empty cluster takes the point
        // farthest from its own centre, so k clusters stay in use.
        let mut sums = vec![vec![0.0; d]; k];
        let mut counts = vec![0usize; k];
        for (p, &c) in x.iter().zip(&cluster) {
            counts[c] += 1;
            for (s, v) in sums[c].iter_mut().zip(p) {
                *s += v;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                centres[c] = sums[c].iter().map(|s| s / counts[c] as f64).collect();
            } else {
                let far = argmax(
                    &x.iter()
                        .zip(&cluster)
                        .map(|(p, &cp)| squared_distance(p, &centres[cp]))
                        .collect::<Vec<_>>(),
                );
                centres[c] = x[far].clone();
            }
        }
    }
    let inertia = x
        .iter()
        .zip(&cluster)
        .map(|(p, &c)| squared_distance(p, &centres[c]))
        .sum();
    (cluster, inertia)
}

/// Normalised mutual information between two labelings of the same drones: 1 = the same
/// partition (whatever the numbers), 0 = independent. Mutual information divided by the
/// mean of the two entropies (scikit-learn's default). Two single-cluster labelings
/// count as the same partition (1).
///
/// Not 0 for random clusters: with 30 clusters of ~10 drones, chance alone gives a
/// clearly positive NMI. Compare against a random baseline, not against 0.
pub fn nmi(a: &[usize], b: &[usize]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let ka = a.iter().max().map_or(0, |m| m + 1);
    let kb = b.iter().max().map_or(0, |m| m + 1);
    let mut joint = vec![vec![0.0; kb]; ka];
    for (&i, &j) in a.iter().zip(b) {
        joint[i][j] += 1.0;
    }
    let row: Vec<f64> = joint.iter().map(|r| r.iter().sum()).collect();
    let col: Vec<f64> = (0..kb).map(|j| joint.iter().map(|r| r[j]).sum()).collect();
    let entropy = |counts: &[f64]| -> f64 {
        counts
            .iter()
            .filter(|&&c| c > 0.0)
            .map(|&c| -(c / n) * (c / n).ln())
            .sum()
    };
    let mut mi = 0.0;
    for (i, r) in joint.iter().enumerate() {
        for (j, &c) in r.iter().enumerate() {
            if c > 0.0 {
                mi += c / n * (n * c / (row[i] * col[j])).ln();
            }
        }
    }
    let mean_h = (entropy(&row) + entropy(&col)) / 2.0;
    if mean_h == 0.0 { 1.0 } else { mi / mean_h }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerie_graph::Edge;
    use rand::SeedableRng;

    fn rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    #[test]
    fn macro_f1_by_hand() {
        // Class 0: TP 2, FP 1, FN 0 -> 4/5. Class 1: TP 1, FP 0, FN 1 -> 2/3.
        let f = macro_f1(&[0, 0, 1, 1], &[0, 0, 1, 0], 2);
        assert!((f - (0.8 + 2.0 / 3.0) / 2.0).abs() < 1e-12);
        assert_eq!(macro_f1(&[0, 1, 2], &[0, 1, 2], 3), 1.0);
    }

    #[test]
    fn macro_f1_punishes_ignoring_the_rare_class() {
        // 9 of class 0, 1 of class 1, always guess 0: 90 % accuracy, but class 1 scores 0.
        let truth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let f = macro_f1(&truth, &[0; 10], 2);
        assert!((f - (18.0 / 19.0) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn logistic_regression_learns_separable_classes() {
        // Three clouds far apart in 2-D, with units 1000× apart to check the scaling.
        let mut r = rng(1);
        let centres = [[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]];
        let mut x = Vec::new();
        let mut y = Vec::new();
        for (c, centre) in centres.iter().enumerate() {
            for _ in 0..30 {
                x.push(vec![
                    centre[0] + r.random_range(-1.0..1.0),
                    1000.0 * (centre[1] + r.random_range(-1.0..1.0)),
                ]);
                y.push(c);
            }
        }
        let model = LogisticRegression::fit(&x, &y, 3);
        let right = x
            .iter()
            .zip(&y)
            .filter(|(xi, yi)| model.predict(xi) == **yi)
            .count();
        assert_eq!(right, 90);
    }

    #[test]
    fn folds_are_stratified_and_cover_every_drone_once() {
        let y: Vec<usize> = (0..100)
            .map(|i| {
                if i < 50 {
                    0
                } else if i < 80 {
                    1
                } else {
                    2
                }
            })
            .collect();
        let fold = stratified_folds(&y, 3, 5, &mut rng(2));
        for f in 0..5 {
            let count = |c| (0..100).filter(|&i| fold[i] == f && y[i] == c).count();
            assert_eq!((count(0), count(1), count(2)), (10, 6, 4), "fold {f}");
        }
    }

    #[test]
    fn noise_features_score_near_chance() {
        // Random features say nothing about the class: macro-F1 well below 1.
        let mut r = rng(3);
        let y: Vec<usize> = (0..200).map(|i| i % 4).collect();
        let x: Vec<Vec<f64>> = (0..200)
            .map(|_| (0..8).map(|_| r.random_range(-1.0..1.0)).collect())
            .collect();
        let f = role_f1(&x, &y, 4, &mut rng(4));
        assert!(f < 0.45, "{f}");
    }

    #[test]
    fn auc_perfect_backwards_and_ties() {
        assert_eq!(auc(&[3.0, 4.0], &[1.0, 2.0]), 1.0);
        assert_eq!(auc(&[1.0, 2.0], &[3.0, 4.0]), 0.0);
        assert_eq!(auc(&[1.0, 1.0], &[1.0, 1.0]), 0.5);
        // Pairs: (2 vs 1) win, (2 vs 3) lose, (4 vs 1) win, (4 vs 3) win -> 3/4.
        assert_eq!(auc(&[2.0, 4.0], &[1.0, 3.0]), 0.75);
        // One tie out of 4 pairs counts half: (2 vs 2) ½, (2 vs 1) 1, (3 vs 2) 1, (3 vs 1) 1.
        assert_eq!(auc(&[2.0, 3.0], &[2.0, 1.0]), 3.5 / 4.0);
    }

    fn ring(n: u32) -> EdgeList {
        EdgeList {
            n,
            edges: (0..n)
                .map(|i| Edge {
                    src: i.min((i + 1) % n),
                    dst: i.max((i + 1) % n),
                    weight: 1.0,
                })
                .collect(),
        }
    }

    #[test]
    fn hiding_links_splits_the_graph_and_draws_true_non_links() {
        let g = ring(40);
        let split = hide_links(&g, 0.1, &mut rng(5));
        assert_eq!(split.hidden.len(), 4);
        assert_eq!(split.non_links.len(), 4);
        assert_eq!(split.train.edges.len(), 36);
        let train: HashSet<(usize, usize)> = split
            .train
            .edges
            .iter()
            .map(|e| (e.src as usize, e.dst as usize))
            .collect();
        for &(a, b) in &split.hidden {
            assert!(
                !train.contains(&(a, b)),
                "hidden link {a}-{b} still in train"
            );
            assert!(b == a + 1 || (a == 0 && b == 39));
        }
        for &(a, b) in &split.non_links {
            assert!(a < b);
            assert!(b != a + 1 && !(a == 0 && b == 39), "{a}-{b} is a link");
        }
        let distinct: HashSet<_> = split.non_links.iter().collect();
        assert_eq!(distinct.len(), 4);
    }

    #[test]
    fn link_auc_is_one_when_linked_drones_share_a_direction() {
        // Drones 0-1 linked and pointing the same way; 0-2 not linked, opposite ways.
        let z = vec![vec![1.0, 0.0], vec![1.0, 0.1], vec![-1.0, 0.0]];
        let split = LinkSplit {
            train: EdgeList {
                n: 3,
                edges: vec![],
            },
            hidden: vec![(0, 1)],
            non_links: vec![(0, 2)],
        };
        assert_eq!(link_auc(&z, &split), 1.0);
    }

    #[test]
    fn nmi_ignores_cluster_numbers() {
        assert!((nmi(&[0, 0, 1, 1, 2, 2], &[2, 2, 0, 0, 1, 1]) - 1.0).abs() < 1e-12);
        assert_eq!(nmi(&[0, 0, 0], &[0, 0, 0]), 1.0);
        // Independent: each cluster of a holds one of each of b's.
        assert!(nmi(&[0, 0, 1, 1], &[0, 1, 0, 1]).abs() < 1e-12);
        // Everything in one cluster tells nothing about a real split.
        assert_eq!(nmi(&[0, 0, 0, 0], &[0, 0, 1, 1]), 0.0);
    }

    #[test]
    fn kmeans_finds_well_separated_blobs() {
        let mut r = rng(6);
        let mut x = Vec::new();
        let mut truth = Vec::new();
        for c in 0..4 {
            for _ in 0..25 {
                let angle = c as f64 * std::f64::consts::FRAC_PI_2;
                x.push(vec![
                    10.0 * angle.cos() + r.random_range(-1.0..1.0),
                    10.0 * angle.sin() + r.random_range(-1.0..1.0),
                ]);
                truth.push(c);
            }
        }
        let found = kmeans(&x, 4, 5, &mut rng(7));
        assert!((nmi(&found, &truth) - 1.0).abs() < 1e-12);
        assert_eq!(
            kmeans(&x, 4, 5, &mut rng(7)),
            found,
            "same rng, same clusters"
        );
    }

    #[test]
    fn normalized_rows_have_length_one() {
        let z = normalized(&[vec![3.0, 4.0], vec![0.0, 0.0]]);
        assert_eq!(z[0], vec![0.6, 0.8]);
        assert_eq!(z[1], vec![0.0, 0.0]);
    }
}
