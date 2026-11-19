//! Skip-gram playground (Mission 2): watch DeepWalk learn, one training pair at a time.
//!
//! Run with `cargo run --release -p aerie-viz --bin skipgram_playground`.
//!
//! - Left: the graph (presets, or click two drones to add / remove a link), walk and
//!   skip-gram settings, and the training buttons. With "Explain each pair in 9 steps" on,
//!   Space walks through one pair step by step: pick u, pick v, draw the negatives, dot
//!   products, σ, errors, loss, gradients, SGD step.
//! - Middle left: the graph with the current walk; u (gold), v (green) and the negatives
//!   (red) of the pair being trained.
//! - Middle right: the embedding space. At dim 2 these are the real vectors z (discs) and
//!   c (rings); above 2 it is the plane of Z's two main directions (PCA). The arrows are the
//!   SGD step, magnified (planned at step 8, done at step 9).
//! - Bottom: the walk, the pair's table (x = z·c, σ(x), g, loss) and the loss curve.
//! - Right: the Math tab (every formula of the pair with its numbers), how similar the
//!   learned vectors are (cosine), what the walks put together (co-occurrence counts), and
//!   how negatives are drawn (count^0.75).
//!
//! The trainer here runs the exact loop of `aerie_classic::embed::skipgram::train_skipgram`
//! and calls its `pair_loss_and_grads`, step by step. At the end it re-runs
//! `train_skipgram` and shows whether the two give the same Z, number for number.

use std::cell::Cell;

use aerie_classic::embed::skipgram::{
    SkipGramConfig, dot, pair_loss_and_grads, sigmoid, train_skipgram,
};
use aerie_classic::embed::walks::random_walks;
use aerie_graph::{Edge, EdgeList};
use aerie_viz::grid::{fmt, mix};
use aerie_viz::style::{draw_drone, role_color, role_name, squad_color};
use eframe::egui::{
    self, Align2, Color32, FontId, Key, Pos2, Rect, RichText, Sense, Shape, Stroke, StrokeKind,
    pos2, vec2,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::drone::Role;
use swarmsim::snapshot::Snapshot;
use swarmsim::traffic::TrafficConfig;

/// Ring around u, the drone whose embedding z_u is trained.
const U_RING: Color32 = Color32::from_rgb(255, 205, 60);
/// Ring around v, the drone seen near u in the walk (its context c_v is pulled in).
const V_RING: Color32 = Color32::from_rgb(70, 215, 120);
/// Ring around the negatives (their contexts c_n are pushed away).
const NEG_RING: Color32 = Color32::from_rgb(240, 80, 80);
/// The current walk on the graph.
const WALK_COLOR: Color32 = Color32::from_rgb(120, 170, 255);
/// Drone marker radius, pixels.
const NODE_PX: f32 = 8.0;
/// Loss curve: about this many points, whatever the number of pairs.
const LOSS_POINTS: usize = 600;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE skip-gram playground")
            .with_inner_size([1700.0, 1000.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE skip-gram playground",
        options,
        Box::new(|_cc| Ok(Box::new(Playground::default()))),
    )
}

// ------------------------------------------------------------------------------- graphs

/// The graphs to train on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Preset {
    TwoGroups,
    ThreeGroups,
    Ring,
    Path,
    Grid,
    StarAndChain,
    Swarm,
}

impl Preset {
    const ALL: [Preset; 7] = [
        Preset::TwoGroups,
        Preset::ThreeGroups,
        Preset::Ring,
        Preset::Path,
        Preset::Grid,
        Preset::StarAndChain,
        Preset::Swarm,
    ];

    fn name(self) -> &'static str {
        match self {
            Preset::TwoGroups => "Two groups + bridge",
            Preset::ThreeGroups => "Three groups",
            Preset::Ring => "Ring of 16",
            Preset::Path => "Path of 12",
            Preset::Grid => "Grid 5 × 5",
            Preset::StarAndChain => "Star + chain",
            Preset::Swarm => "Swarm traffic (100 drones)",
        }
    }

    fn about(self) -> &'static str {
        match self {
            Preset::TwoGroups => {
                "Two 6-drone cliques joined by one link (the unit test's graph). Walks rarely \
                 cross the bridge, so each group should end up pointing its own way."
            }
            Preset::ThreeGroups => {
                "Three 5-drone cliques in a triangle. At dim 2 the three groups have to share \
                 the plane: watch them split it into three directions."
            }
            Preset::Ring => {
                "16 drones in a circle. Neighbours meet in walks, the far side never does: at \
                 dim 2 the vectors often come out as a ring too."
            }
            Preset::Path => {
                "12 drones in a line. The two ends are the most different; the middle meets \
                 everyone."
            }
            Preset::Grid => {
                "A 5 × 5 lattice, coloured by column. Walks only know who is next to whom, \
                 yet the layout can come back."
            }
            Preset::StarAndChain => {
                "A hub with 6 leaves and a chain of 6 hanging off it. Leaves never touch each \
                 other but share the hub: skip-gram puts them together anyway (same \
                 neighbours, same context)."
            }
            Preset::Swarm => {
                "The Mission 2 traffic graph of a 100-drone swarm (who exchanged messages, \
                 weight = messages). Colour by squad or by role: which one do the vectors \
                 find?"
            }
        }
    }
}

/// One drone of the playground graph.
#[derive(Debug, Clone)]
struct Node {
    /// Where it is drawn (any units; the canvas fits them).
    pos: [f32; 2],
    /// Its group (squad, clique, column…): the colour, and the truth for the checks.
    group: usize,
    /// Swarm preset only: its role.
    role: Option<Role>,
}

#[derive(Debug, Clone)]
struct Graph {
    nodes: Vec<Node>,
    edges: EdgeList,
}

impl Graph {
    fn n(&self) -> usize {
        self.nodes.len()
    }

    fn build(preset: Preset, swarm_seed: u64) -> Graph {
        match preset {
            Preset::TwoGroups => {
                let mut g = Graph::empty();
                // Drone 5 on the right of group A, drone 6 on the left of group B.
                g.clique(6, [-1.6, 0.0], 1.0, 0.0, -5, 0);
                g.clique(6, [1.6, 0.0], 1.0, std::f32::consts::PI, 0, 1);
                g.link(5, 6, 1.0);
                g
            }
            Preset::ThreeGroups => {
                let mut g = Graph::empty();
                let corners = [[0.0, -1.9], [-1.9, 1.3], [1.9, 1.3]];
                for (k, &centre) in corners.iter().enumerate() {
                    g.clique(5, centre, 0.9, 0.0, 0, k);
                }
                g.link(4, 5, 1.0);
                g.link(9, 10, 1.0);
                g.link(0, 14, 1.0);
                g
            }
            Preset::Ring => {
                let mut g = Graph::empty();
                let n = 16;
                for i in 0..n {
                    let a = std::f32::consts::TAU * i as f32 / n as f32;
                    g.add([2.0 * a.cos(), 2.0 * a.sin()], i * 4 / n, None);
                }
                for i in 0..n {
                    g.link(i, (i + 1) % n, 1.0);
                }
                g
            }
            Preset::Path => {
                let mut g = Graph::empty();
                for i in 0..12 {
                    g.add([i as f32, 0.0], i * 3 / 12, None);
                }
                for i in 0..11 {
                    g.link(i, i + 1, 1.0);
                }
                g
            }
            Preset::Grid => {
                let mut g = Graph::empty();
                for r in 0..5 {
                    for c in 0..5 {
                        g.add([c as f32, r as f32], c, None);
                    }
                }
                for r in 0..5 {
                    for c in 0..5 {
                        let i = r * 5 + c;
                        if c < 4 {
                            g.link(i, i + 1, 1.0);
                        }
                        if r < 4 {
                            g.link(i, i + 5, 1.0);
                        }
                    }
                }
                g
            }
            Preset::StarAndChain => {
                let mut g = Graph::empty();
                g.add([0.0, 0.0], 0, None); // hub
                for k in 0..6 {
                    let a = std::f32::consts::PI * (0.5 + k as f32 / 5.0);
                    g.add([1.3 * a.cos(), 1.3 * a.sin()], 1, None);
                    g.link(0, k + 1, 1.0);
                }
                for k in 0..6 {
                    g.add([1.0 + k as f32, 0.0], 2, None);
                    g.link(if k == 0 { 0 } else { 6 + k }, 7 + k, 1.0);
                }
                g
            }
            Preset::Swarm => {
                let config = SimConfig::default();
                let snap = Snapshot::build(swarm_seed, &config);
                let edges = snap.traffic(&TrafficConfig::default()).undirected();
                let nodes = snap
                    .swarm
                    .iter()
                    .map(|d| Node {
                        pos: [d.position.x as f32, -d.position.y as f32],
                        group: d.squad.0 as usize,
                        role: Some(d.role),
                    })
                    .collect();
                Graph { nodes, edges }
            }
        }
    }

    fn empty() -> Graph {
        Graph {
            nodes: Vec::new(),
            edges: EdgeList {
                n: 0,
                edges: Vec::new(),
            },
        }
    }

    fn add(&mut self, pos: [f32; 2], group: usize, role: Option<Role>) {
        self.nodes.push(Node { pos, group, role });
        self.edges.n = self.nodes.len() as u32;
    }

    /// `k` drones on a circle, all linked to each other. Drone `first + i` sits at angle
    /// `start + 2π (i + shift) / k`.
    fn clique(
        &mut self,
        k: usize,
        centre: [f32; 2],
        radius: f32,
        start: f32,
        shift: i32,
        group: usize,
    ) {
        let first = self.n();
        for i in 0..k {
            let a = start + std::f32::consts::TAU * (i as i32 + shift) as f32 / k as f32;
            self.add(
                [centre[0] + radius * a.cos(), centre[1] + radius * a.sin()],
                group,
                None,
            );
        }
        for a in first..first + k {
            for b in a + 1..first + k {
                self.link(a, b, 1.0);
            }
        }
    }

    fn link(&mut self, a: usize, b: usize, weight: f64) {
        let (src, dst) = (a.min(b) as u32, a.max(b) as u32);
        self.edges.edges.push(Edge { src, dst, weight });
    }

    /// Adds the link a–b (weight 1) if it is missing, removes it if it is there.
    fn toggle(&mut self, a: usize, b: usize) {
        let (src, dst) = (a.min(b) as u32, a.max(b) as u32);
        let before = self.edges.edges.len();
        self.edges.edges.retain(|e| (e.src, e.dst) != (src, dst));
        if self.edges.edges.len() == before {
            self.link(a, b, 1.0);
        }
    }
}

// ------------------------------------------------------------------------------ trainer

/// Every setting of the walks and of skip-gram.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Settings {
    walks_per_node: usize,
    walk_length: usize,
    walk_seed: u64,
    dim: usize,
    window: usize,
    negatives: usize,
    epochs: usize,
    lr: f64,
    train_seed: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            walks_per_node: 10,
            walk_length: 20,
            walk_seed: 1,
            dim: 2,
            window: 3,
            negatives: 5,
            epochs: 3,
            lr: 0.025,
            train_seed: 2,
        }
    }
}

impl Settings {
    fn config(&self) -> SkipGramConfig {
        SkipGramConfig {
            dim: self.dim,
            window: self.window,
            negatives: self.negatives,
            epochs: self.epochs,
            lr: self.lr,
        }
    }

    fn walks_key(&self) -> (usize, usize, u64) {
        (self.walks_per_node, self.walk_length, self.walk_seed)
    }
}

/// The 9 steps of one training pair, as the Math tab and the 9-step mode show them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    Position,
    Context,
    Negatives,
    Dots,
    Sigmoid,
    Errors,
    Loss,
    Gradients,
    Update,
}

impl Phase {
    const ALL: [Phase; 9] = [
        Phase::Position,
        Phase::Context,
        Phase::Negatives,
        Phase::Dots,
        Phase::Sigmoid,
        Phase::Errors,
        Phase::Loss,
        Phase::Gradients,
        Phase::Update,
    ];

    fn number(self) -> usize {
        self as usize + 1
    }

    fn title(self) -> &'static str {
        match self {
            Phase::Position => "Pick u, the centre drone",
            Phase::Context => "Pick v, a drone from u's context",
            Phase::Negatives => "Draw the negatives",
            Phase::Dots => "Dot products x = z_u · c",
            Phase::Sigmoid => "Probabilities σ(x)",
            Phase::Errors => "Errors g = σ(x) − want",
            Phase::Loss => "Loss",
            Phase::Gradients => "Gradients",
            Phase::Update => "SGD step: move the vectors",
        }
    }

    fn next(self) -> Option<Phase> {
        Phase::ALL.get(self as usize + 1).copied()
    }
}

/// The vectors after the SGD step.
#[derive(Debug, Clone)]
struct After {
    z: Vec<f64>,
    cv: Vec<f64>,
    cn: Vec<Vec<f64>>,
}

/// One training pair: who, every number of the maths, and the vectors before (and, once
/// the step is applied, after).
#[derive(Debug, Clone)]
struct Example {
    walk: usize,
    t: usize,
    s: usize,
    u: usize,
    v: usize,
    /// Every draw for the negatives: (r, drone drawn, kept). A draw equal to v is thrown away.
    draws: Vec<(f64, usize, bool)>,
    negs: Vec<usize>,
    /// x = z_u · c_v, and x = z_u · c_n for each negative.
    x_pos: f64,
    x_negs: Vec<f64>,
    loss: f64,
    grad_z: Vec<f64>,
    grad_zv: Vec<f64>,
    grad_neg: Vec<Vec<f64>>,
    z_before: Vec<f64>,
    cv_before: Vec<f64>,
    cn_before: Vec<Vec<f64>>,
    /// `None` while the pair is prepared but not applied yet.
    after: Option<After>,
}

/// `train_skipgram`, one pair at a time: the same set-up, the same order of pairs, the same
/// random draws, the same updates. Running it to the end gives the same Z.
struct Trainer {
    walks: Vec<Vec<usize>>,
    settings: Settings,
    /// (walk, t, s) for every positive pair of one epoch, in the loop's order.
    pairs: Vec<(u32, u32, u32)>,
    z: Vec<Vec<f64>>,
    c: Vec<Vec<f64>>,
    /// How often each drone appears in the walks.
    counts: Vec<f64>,
    cum: Vec<f64>,
    total: f64,
    rng: ChaCha8Rng,
    /// Pairs trained so far, all epochs together.
    done: usize,
    /// Loss averaged over buckets of `bucket` pairs.
    loss_curve: Vec<f32>,
    bucket: usize,
    bucket_sum: f64,
    bucket_n: usize,
    last: Option<Example>,
}

impl Trainer {
    fn new(walks: Vec<Vec<usize>>, n: usize, settings: Settings) -> Trainer {
        let dim = settings.dim;
        let mut rng = ChaCha8Rng::seed_from_u64(settings.train_seed);

        // Same set-up as train_skipgram, in the same order (so the same random numbers).
        let half = 0.5 / dim as f64;
        let mut z: Vec<Vec<f64>> = Vec::new();
        for _ in 0..n {
            let mut row = Vec::new();
            for _ in 0..dim {
                row.push(rng.random_range(-half..half));
            }
            z.push(row);
        }
        let c = vec![vec![0.0; dim]; n];
        let mut counts = vec![0.0_f64; n];
        for walk in &walks {
            for &d in walk {
                counts[d] += 1.0;
            }
        }
        let mut cum = Vec::with_capacity(n);
        let mut total = 0.0;
        for &count in &counts {
            total += count.powf(0.75);
            cum.push(total);
        }

        let mut pairs = Vec::new();
        for (w, walk) in walks.iter().enumerate() {
            for t in 0..walk.len() {
                let (lo, hi) = window_of(walk.len(), t, settings.window);
                for s in lo..=hi {
                    if s != t {
                        pairs.push((w as u32, t as u32, s as u32));
                    }
                }
            }
        }
        let bucket = (pairs.len() * settings.epochs / LOSS_POINTS).max(1);

        Trainer {
            walks,
            settings,
            pairs,
            z,
            c,
            counts,
            cum,
            total,
            rng,
            done: 0,
            loss_curve: Vec::new(),
            bucket,
            bucket_sum: 0.0,
            bucket_n: 0,
            last: None,
        }
    }

    fn n(&self) -> usize {
        self.z.len()
    }

    fn total_pairs(&self) -> usize {
        self.pairs.len() * self.settings.epochs
    }

    fn finished(&self) -> bool {
        self.done >= self.total_pairs()
    }

    /// (epoch, walk) of the next pair to train.
    fn next_position(&self) -> Option<(usize, usize)> {
        if self.finished() {
            return None;
        }
        let len = self.pairs.len();
        Some((self.done / len, self.pairs[self.done % len].0 as usize))
    }

    /// (epoch, walk) of the pair after the next one.
    fn next_position_after(&self) -> Option<(usize, usize)> {
        let k = self.done + 1;
        if k >= self.total_pairs() {
            return None;
        }
        let len = self.pairs.len();
        Some((k / len, self.pairs[k % len].0 as usize))
    }

    /// Steps 1–8 of the next pair: picks u and v, draws the negatives (this uses the random
    /// numbers), computes the loss and the gradients. Changes no vector.
    fn prepare(&mut self) -> Option<Example> {
        if self.finished() {
            return None;
        }
        let (w, t, s) = self.pairs[self.done % self.pairs.len()];
        let (walk, t, s) = (w as usize, t as usize, s as usize);
        let u = self.walks[walk][t];
        let v = self.walks[walk][s];

        // Negatives, drawn exactly as in train_skipgram.
        let mut negs: Vec<usize> = Vec::new();
        let mut draws = Vec::new();
        while negs.len() < self.settings.negatives {
            let r = self.rng.random_range(0.0..self.total);
            let k = self.cum.partition_point(|&x| x <= r);
            draws.push((r, k, k != v && k != u));
            if k != v && k != u {
                negs.push(k);
            }
        }

        let c_neg: Vec<&[f64]> = negs.iter().map(|&k| self.c[k].as_slice()).collect();
        let (loss, grad_z, grad_zv, grad_neg) = pair_loss_and_grads(&self.z[u], &self.c[v], &c_neg);
        Some(Example {
            walk,
            t,
            s,
            u,
            v,
            draws,
            x_pos: dot(&self.z[u], &self.c[v]),
            x_negs: negs.iter().map(|&k| dot(&self.z[u], &self.c[k])).collect(),
            cn_before: negs.iter().map(|&k| self.c[k].clone()).collect(),
            negs,
            loss,
            grad_z,
            grad_zv,
            grad_neg,
            z_before: self.z[u].clone(),
            cv_before: self.c[v].clone(),
            after: None,
        })
    }

    /// Step 9: the SGD step of a prepared pair, in train_skipgram's order. With `record`,
    /// keeps the pair (with its after-vectors) in `last`.
    fn apply(&mut self, mut ex: Example, record: bool) {
        let lr = self.settings.lr;
        let (u, v) = (ex.u, ex.v);
        for i in 0..self.settings.dim {
            self.z[u][i] -= lr * ex.grad_z[i];
            self.c[v][i] -= lr * ex.grad_zv[i];
        }
        for (&k, g) in ex.negs.iter().zip(&ex.grad_neg) {
            for (ci, gi) in self.c[k].iter_mut().zip(g) {
                *ci -= lr * gi;
            }
        }

        self.done += 1;
        self.bucket_sum += ex.loss;
        self.bucket_n += 1;
        if self.bucket_n == self.bucket {
            self.loss_curve
                .push((self.bucket_sum / self.bucket_n as f64) as f32);
            self.bucket_sum = 0.0;
            self.bucket_n = 0;
        }
        if record {
            ex.after = Some(After {
                z: self.z[u].clone(),
                cv: self.c[v].clone(),
                cn: ex.negs.iter().map(|&k| self.c[k].clone()).collect(),
            });
            self.last = Some(ex);
        }
    }

    /// Trains the next pair (all 9 steps). With `record`, keeps it in `last`.
    fn step(&mut self, record: bool) {
        if let Some(ex) = self.prepare() {
            self.apply(ex, record);
        }
    }

    /// Trains `k` pairs; only the last one is recorded.
    fn run(&mut self, k: usize) {
        for i in 0..k {
            self.step(i + 1 == k);
        }
    }

    /// Trains until the current walk is done.
    fn finish_walk(&mut self) {
        let Some(start) = self.next_position() else {
            return;
        };
        while self.next_position() == Some(start) {
            let last = self.next_position_after() != Some(start);
            self.step(last);
        }
    }

    /// Trains until the current epoch is done.
    fn finish_epoch(&mut self) {
        let Some((epoch, _)) = self.next_position() else {
            return;
        };
        while self.next_position().is_some_and(|(e, _)| e == epoch) {
            let last = self.next_position_after().is_none_or(|(e, _)| e != epoch);
            self.step(last);
        }
    }

    fn finish(&mut self) {
        let left = self.total_pairs() - self.done;
        self.run(left);
    }

    /// Mean loss of the pairs in the bucket being filled (not yet in the curve).
    fn partial_loss(&self) -> Option<f32> {
        (self.bucket_n > 0).then(|| (self.bucket_sum / self.bucket_n as f64) as f32)
    }
}

/// The window around position `t` of a walk of `len` drones: positions `lo..=hi`.
fn window_of(len: usize, t: usize, window: usize) -> (usize, usize) {
    (t.saturating_sub(window), (t + window).min(len - 1))
}

/// How many times each pair (u, v) is a positive pair in one epoch: what skip-gram is fitted
/// to. Symmetric.
fn cooccurrence(walks: &[Vec<usize>], n: usize, window: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for walk in walks {
        for t in 0..walk.len() {
            let (lo, hi) = window_of(walk.len(), t, window);
            for s in lo..=hi {
                if s != t {
                    m[walk[t]][walk[s]] += 1.0;
                }
            }
        }
    }
    m
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let norm = (dot(a, a) * dot(b, b)).sqrt();
    if norm == 0.0 { 0.0 } else { dot(a, b) / norm }
}

/// The 2-D coordinates to draw: the vectors themselves at dim 2; above, a projection on two
/// directions. `axes` = the two directions (dim ≥ 3) and the point that lands on the origin.
struct Projection {
    axes: Option<([Vec<f64>; 2], Vec<f64>)>,
    /// Joint projection only: the share of the vectors' squared length the plane keeps.
    kept: Option<f64>,
}

impl Projection {
    /// Z alone, on its two main directions around its mean (PCA): the best picture of how
    /// the embeddings differ from each other. Dot products are not kept.
    fn of(z: &[Vec<f64>]) -> Projection {
        let dim = z.first().map_or(2, |r| r.len());
        if dim <= 2 || z.len() < 2 {
            return Projection {
                axes: None,
                kept: None,
            };
        }
        let n = z.len() as f64;
        let mean: Vec<f64> = (0..dim)
            .map(|i| z.iter().map(|r| r[i]).sum::<f64>() / n)
            .collect();
        let mut cov = vec![vec![0.0; dim]; dim];
        for r in z {
            for i in 0..dim {
                for j in 0..dim {
                    cov[i][j] += (r[i] - mean[i]) * (r[j] - mean[j]) / n;
                }
            }
        }
        let first = top_direction(&cov, None, None);
        let second = top_direction(&cov, Some(&first), None);
        Projection {
            axes: Some(([first, second], mean)),
            kept: None,
        }
    }

    /// Z and C together, on the two directions that keep the most of their length, with the
    /// origin left where it is (an uncentred PCA, i.e. the top of an SVD of [Z; C]). The dot
    /// product of two projected vectors is their real dot product minus what the plane
    /// drops: close when `kept` is high. `previous` (last frame's axes) keeps the picture
    /// from swapping or flipping its axes between frames.
    fn joint(z: &[Vec<f64>], c: &[Vec<f64>], previous: Option<&[Vec<f64>; 2]>) -> Projection {
        let dim = z.first().map_or(2, |r| r.len());
        if dim <= 2 {
            return Projection {
                axes: None,
                kept: Some(1.0),
            };
        }
        let mut m = vec![vec![0.0; dim]; dim];
        for r in z.iter().chain(c) {
            for i in 0..dim {
                for j in 0..dim {
                    m[i][j] += r[i] * r[j];
                }
            }
        }
        let trace: f64 = (0..dim).map(|i| m[i][i]).sum();
        let mut first = top_direction(&m, None, previous.map(|p| p[0].as_slice()));
        let mut second = top_direction(&m, Some(&first), previous.map(|p| p[1].as_slice()));
        if let Some(p) = previous {
            // Same axis order and the same signs as last frame, when that fits better.
            if dot(&first, &p[1]).abs() > dot(&first, &p[0]).abs() {
                std::mem::swap(&mut first, &mut second);
            }
            for (axis, old) in [&mut first, &mut second].into_iter().zip(p) {
                if dot(axis, old) < 0.0 {
                    axis.iter_mut().for_each(|x| *x = -*x);
                }
            }
        }
        let along = |v: &[f64]| -> f64 {
            let mv: Vec<f64> = (0..dim).map(|i| dot(&m[i], v)).collect();
            dot(v, &mv)
        };
        let kept = if trace > 0.0 {
            ((along(&first) + along(&second)) / trace).clamp(0.0, 1.0)
        } else {
            1.0
        };
        Projection {
            axes: Some(([first, second], vec![0.0; dim])),
            kept: Some(kept),
        }
    }

    fn point(&self, v: &[f64]) -> [f64; 2] {
        match &self.axes {
            None => [
                v.first().copied().unwrap_or(0.0),
                v.get(1).copied().unwrap_or(0.0),
            ],
            Some((axes, mean)) => {
                let centred: Vec<f64> = v.iter().zip(mean).map(|(a, m)| a - m).collect();
                [dot(&centred, &axes[0]), dot(&centred, &axes[1])]
            }
        }
    }
}

/// Main direction of a symmetric matrix by power iteration, kept orthogonal to `not`,
/// starting from `start` if given. Sign fixed (largest component positive) so the picture
/// doesn't flip between frames.
fn top_direction(m: &[Vec<f64>], not: Option<&[f64]>, start: Option<&[f64]>) -> Vec<f64> {
    let d = m.len();
    let mut v: Vec<f64> = match start {
        Some(s) if s.len() == d && dot(s, s) > 0.0 => s.to_vec(),
        _ => (0..d).map(|i| 1.0 + 0.1 * i as f64).collect(),
    };
    if let Some(o) = not {
        // A start (nearly) parallel to `not` would vanish once `not` is taken out: fall back
        // to generic starts (no zero entries, so never stuck where M has nothing).
        let leftover = |v: &[f64]| {
            let p = dot(v, o);
            let rest: f64 = v.iter().zip(o).map(|(x, oi)| (x - p * oi).powi(2)).sum();
            rest / dot(v, v).max(1e-300)
        };
        if leftover(&v) < 1e-6 {
            let generic: Vec<f64> = (0..d).map(|i| 1.0 + 0.1 * i as f64).collect();
            let alternating: Vec<f64> = (0..d)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 } * (1.0 + 0.1 * i as f64))
                .collect();
            v = [generic, alternating]
                .into_iter()
                .max_by(|a, b| leftover(a).total_cmp(&leftover(b)))
                .unwrap_or(v);
        }
    }
    for _ in 0..200 {
        if let Some(o) = not {
            let p = dot(&v, o);
            v.iter_mut().zip(o).for_each(|(x, oi)| *x -= p * oi);
        }
        let mut next: Vec<f64> = (0..d).map(|i| dot(&m[i], &v)).collect();
        let norm = dot(&next, &next).sqrt();
        if norm < 1e-300 {
            break;
        }
        next.iter_mut().for_each(|x| *x /= norm);
        v = next;
    }
    if let Some(o) = not {
        let p = dot(&v, o);
        v.iter_mut().zip(o).for_each(|(x, oi)| *x -= p * oi);
        let norm = dot(&v, &v).sqrt().max(1e-300);
        v.iter_mut().for_each(|x| *x /= norm);
    }
    let big = v
        .iter()
        .copied()
        .max_by(|a, b| a.abs().total_cmp(&b.abs()))
        .unwrap_or(1.0);
    if big < 0.0 {
        v.iter_mut().for_each(|x| *x = -*x);
    }
    v
}

/// Group checks on Z: how many drones have a same-group drone as most similar vector, and
/// the mean cosine inside groups and across them.
fn group_checks(z: &[Vec<f64>], nodes: &[Node]) -> (usize, f64, f64) {
    let n = z.len();
    let mut right = 0;
    let (mut inside, mut n_in, mut across, mut n_across) = (0.0, 0.0, 0.0, 0.0);
    for a in 0..n {
        let best = (0..n)
            .filter(|&b| b != a)
            .max_by(|&x, &y| cosine(&z[a], &z[x]).total_cmp(&cosine(&z[a], &z[y])));
        if best.is_some_and(|b| nodes[b].group == nodes[a].group) {
            right += 1;
        }
        for b in a + 1..n {
            let cs = cosine(&z[a], &z[b]);
            if nodes[a].group == nodes[b].group {
                inside += cs;
                n_in += 1.0;
            } else {
                across += cs;
                n_across += 1.0;
            }
        }
    }
    (
        right,
        inside / f64::max(n_in, 1.0),
        across / f64::max(n_across, 1.0),
    )
}

// ---------------------------------------------------------------------------------- app

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Math,
    Similarity,
    Cooccurrence,
    Negatives,
}

struct Playground {
    preset: Preset,
    swarm_seed: u64,
    graph: Graph,
    /// Bumped on every link edit, so the walks are rebuilt.
    graph_version: u64,
    settings: Settings,
    trainer: Trainer,
    /// What the trainer was built from: (graph version, settings).
    built: (u64, Settings),
    cooc: Vec<Vec<f64>>,
    /// At the end of training: does `train_skipgram` give the very same Z?
    matches: Option<bool>,

    /// 9-step mode: Space goes through one pair step by step.
    explain: bool,
    /// The pair being explained (prepared, not applied yet), and the step shown.
    pending: Option<Example>,
    phase: Phase,
    /// Scroll the Math tab to the current step on the next frame.
    scroll_math: Cell<bool>,

    playing: bool,
    /// Pairs per second while playing.
    speed: f64,
    clock: f64,

    color_by_role: bool,
    show_c: bool,
    show_walk: bool,
    magnify: f32,
    tab: Tab,

    hover: Option<usize>,
    hover_next: Option<usize>,
    selected: Option<usize>,
    dragging: Option<usize>,
    /// Half-width of the embedding view (smoothed so it doesn't jump).
    view_r: f32,
    /// Last frame's projection axes (dim ≥ 3), so the picture doesn't swap or flip them.
    prev_axes: Option<[Vec<f64>; 2]>,
}

impl Default for Playground {
    fn default() -> Self {
        let preset = Preset::TwoGroups;
        let graph = Graph::build(preset, 0);
        let settings = Settings::default();
        let walks = make_walks(&graph, &settings);
        let cooc = cooccurrence(&walks, graph.n(), settings.window);
        let trainer = Trainer::new(walks, graph.n(), settings);
        Playground {
            preset,
            swarm_seed: 0,
            graph,
            graph_version: 0,
            settings,
            trainer,
            built: (0, settings),
            cooc,
            matches: None,
            explain: true,
            pending: None,
            phase: Phase::Update,
            scroll_math: Cell::new(false),
            playing: false,
            speed: 5.0,
            clock: 0.0,
            color_by_role: false,
            show_c: true,
            show_walk: true,
            magnify: 20.0,
            tab: Tab::Math,
            hover: None,
            hover_next: None,
            selected: None,
            dragging: None,
            view_r: 0.1,
            prev_axes: None,
        }
    }
}

fn make_walks(graph: &Graph, settings: &Settings) -> Vec<Vec<usize>> {
    random_walks(
        &graph.edges,
        settings.walks_per_node,
        settings.walk_length,
        &mut ChaCha8Rng::seed_from_u64(settings.walk_seed),
    )
}

impl eframe::App for Playground {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.hover = self.hover_next.take();

        // Keyboard: Space = next step (9-step mode) or next pair, P = play / pause.
        if !ui.ctx().egui_wants_keyboard_input() {
            if ui.input(|i| i.key_pressed(Key::Space)) {
                self.playing = false;
                self.next();
            }
            if ui.input(|i| i.key_pressed(Key::P)) {
                self.toggle_play();
            }
        }

        if self.playing {
            self.clock += ui.input(|i| i.stable_dt) as f64 * self.speed;
            let k = self.clock.floor() as usize;
            self.clock -= k as f64;
            self.trainer.run(k);
            if self.trainer.finished() {
                self.playing = false;
            }
            ui.ctx().request_repaint();
        }
        if self.trainer.finished() && self.pending.is_none() && self.matches.is_none() {
            let mut rng = ChaCha8Rng::seed_from_u64(self.settings.train_seed);
            let z = train_skipgram(
                &self.trainer.walks,
                self.trainer.n(),
                &self.settings.config(),
                &mut rng,
            );
            self.matches = Some(z == self.trainer.z);
        }

        egui::Panel::left("controls")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });
        egui::Panel::right("matrices")
            .resizable(true)
            .default_size(460.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.right(ui));
            });
        egui::Panel::bottom("pair")
            .resizable(true)
            .default_size(280.0)
            .show(ui, |ui| self.bottom(ui));
        egui::CentralPanel::default().show(ui, |ui| {
            ui.columns(2, |cols| {
                self.graph_canvas(&mut cols[0]);
                self.embed_canvas(&mut cols[1]);
            });
        });

        self.rebuild_if_needed();
    }
}

impl Playground {
    /// Rebuilds the walks (graph or walk settings changed) and the trainer (any setting).
    fn rebuild_if_needed(&mut self) {
        let (version, old) = self.built;
        if version == self.graph_version && old == self.settings {
            return;
        }
        let walks = if version != self.graph_version || old.walks_key() != self.settings.walks_key()
        {
            make_walks(&self.graph, &self.settings)
        } else {
            std::mem::take(&mut self.trainer.walks)
        };
        self.cooc = cooccurrence(&walks, self.graph.n(), self.settings.window);
        self.trainer = Trainer::new(walks, self.graph.n(), self.settings);
        self.built = (self.graph_version, self.settings);
        self.matches = None;
        self.playing = false;
        self.pending = None;
        self.phase = Phase::Update;
    }

    fn load(&mut self, preset: Preset) {
        self.preset = preset;
        self.graph = Graph::build(preset, self.swarm_seed);
        self.graph_version += 1;
        self.selected = None;
        self.color_by_role = false;
        self.view_r = 0.1;
        // A 100-drone graph needs more room than 2 numbers per drone.
        if preset == Preset::Swarm && self.settings.dim == 2 {
            self.settings.dim = 16;
        }
    }

    fn reset(&mut self) {
        self.built.0 = u64::MAX; // forces a rebuild with the same walks and settings
        self.view_r = 0.1;
    }

    /// Space / Next: one step of the pair in 9-step mode, otherwise one whole pair.
    fn next(&mut self) {
        if !self.explain {
            self.flush();
            self.trainer.step(true);
            return;
        }
        match self.pending.take() {
            Some(ex) => match self.phase.next() {
                Some(Phase::Update) | None => {
                    self.trainer.apply(ex, true);
                    self.phase = Phase::Update;
                }
                Some(p) => {
                    self.phase = p;
                    self.pending = Some(ex);
                }
            },
            None => {
                if let Some(ex) = self.trainer.prepare() {
                    self.pending = Some(ex);
                    self.phase = Phase::Position;
                }
            }
        }
        self.scroll_math.set(true);
    }

    /// Finishes the pair being explained (applies its SGD step), so training can go on in
    /// the right order.
    fn flush(&mut self) {
        if let Some(ex) = self.pending.take() {
            self.trainer.apply(ex, true);
        }
        self.phase = Phase::Update;
    }

    fn toggle_play(&mut self) {
        self.flush();
        self.playing = !self.playing;
    }

    /// The pair to show and the step reached: the one being explained, otherwise the last
    /// one trained (all 9 steps done).
    fn shown(&self) -> Option<(&Example, Phase)> {
        match &self.pending {
            Some(ex) => Some((ex, self.phase)),
            None => self.trainer.last.as_ref().map(|ex| (ex, Phase::Update)),
        }
    }

    fn color(&self, i: usize) -> Color32 {
        let node = &self.graph.nodes[i];
        match node.role {
            Some(role) if self.color_by_role => role_color(role),
            _ => squad_color(node.group),
        }
    }

    // ------------------------------------------------------------------------ controls

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Graph");
        egui::ComboBox::from_id_salt("preset")
            .selected_text(self.preset.name())
            .width(250.0)
            .show_ui(ui, |ui| {
                for p in Preset::ALL {
                    if ui.selectable_label(self.preset == p, p.name()).clicked() {
                        self.load(p);
                    }
                }
            });
        ui.label(RichText::new(self.preset.about()).small());
        if self.preset == Preset::Swarm {
            ui.horizontal(|ui| {
                let changed = ui
                    .add(egui::DragValue::new(&mut self.swarm_seed).range(0..=99))
                    .changed();
                ui.label("swarm seed");
                if changed {
                    self.load(Preset::Swarm);
                }
            });
            ui.horizontal(|ui| {
                ui.label("colour by");
                ui.selectable_value(&mut self.color_by_role, false, "squad");
                ui.selectable_value(&mut self.color_by_role, true, "role");
            });
        }
        ui.label(
            RichText::new(format!(
                "{} drones, {} links. Click a drone, then another, to add or remove the link \
                 between them; drag to move.",
                self.graph.n(),
                self.graph.edges.edges.len()
            ))
            .small()
            .weak(),
        );

        ui.separator();
        ui.heading("Walks");
        let s = &mut self.settings;
        ui.add(egui::Slider::new(&mut s.walks_per_node, 1..=40).text("walks per drone"));
        ui.add(egui::Slider::new(&mut s.walk_length, 2..=80).text("walk length"));
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut s.walk_seed));
            ui.label("walk seed");
        });
        ui.label(
            RichText::new(format!(
                "{} walks, {} positive pairs per epoch",
                self.trainer.walks.len(),
                self.trainer.pairs.len()
            ))
            .small()
            .weak(),
        );

        ui.separator();
        ui.heading("Skip-gram");
        let s = &mut self.settings;
        ui.add(egui::Slider::new(&mut s.dim, 2..=32).text("dim (numbers per vector)"));
        ui.add(egui::Slider::new(&mut s.window, 1..=10).text("window"));
        ui.add(egui::Slider::new(&mut s.negatives, 1..=10).text("negatives per pair"));
        ui.add(egui::Slider::new(&mut s.epochs, 1..=20).text("epochs"));
        ui.add(
            egui::Slider::new(&mut s.lr, 0.001..=0.5)
                .logarithmic(true)
                .text("learning rate"),
        );
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut s.train_seed));
            ui.label("training seed (Z's start, negatives)");
        });

        ui.separator();
        ui.heading("Train");
        if ui
            .checkbox(&mut self.explain, "Explain each pair in 9 steps")
            .on_hover_text(
                "Space goes through one pair step by step; the Math tab shows each step.",
            )
            .changed()
            && !self.explain
        {
            self.flush();
        }
        let tr = &self.trainer;
        let (epoch, walk) = tr
            .next_position()
            .unwrap_or((self.settings.epochs, tr.walks.len().saturating_sub(1)));
        ui.label(format!(
            "pair {} of {} · epoch {} of {} · walk {}",
            tr.done + usize::from(self.pending.is_some()),
            tr.total_pairs(),
            (epoch + 1).min(self.settings.epochs),
            self.settings.epochs,
            walk
        ));
        if self.pending.is_some() {
            ui.label(
                RichText::new(format!(
                    "step {} of 9: {}",
                    self.phase.number(),
                    self.phase.title()
                ))
                .color(U_RING),
            );
        }
        let progress = tr.done as f32 / tr.total_pairs().max(1) as f32;
        ui.add(egui::ProgressBar::new(progress).desired_height(8.0));
        ui.horizontal_wrapped(|ui| {
            let next = if self.explain {
                "Next step"
            } else {
                "Next pair"
            };
            if ui.button(next).on_hover_text("Space").clicked() {
                self.playing = false;
                self.next();
            }
            if self.explain && ui.button("Whole pair").clicked() {
                self.playing = false;
                self.flush();
                self.trainer.step(true);
            }
            if ui.button("Finish walk").clicked() {
                self.playing = false;
                self.flush();
                self.trainer.finish_walk();
            }
            if ui.button("Finish epoch").clicked() {
                self.playing = false;
                self.flush();
                self.trainer.finish_epoch();
            }
            if ui.button("Train to the end").clicked() {
                self.playing = false;
                self.flush();
                self.trainer.finish();
            }
        });
        ui.horizontal(|ui| {
            let label = if self.playing {
                "⏸ Pause"
            } else {
                "▶ Play"
            };
            if ui.button(label).on_hover_text("P").clicked() {
                self.toggle_play();
            }
            if ui.button("⟲ Reset").clicked() {
                self.reset();
            }
        });
        ui.add(
            egui::Slider::new(&mut self.speed, 1.0..=200_000.0)
                .logarithmic(true)
                .text("pairs / second"),
        );
        match self.matches {
            Some(true) => {
                ui.label(
                    RichText::new("✔ Same Z as your train_skipgram, number for number.")
                        .color(Color32::from_rgb(89, 161, 79)),
                );
            }
            Some(false) => {
                ui.label(
                    RichText::new("✘ train_skipgram gives a different Z than this trainer.")
                        .color(Color32::from_rgb(225, 87, 89)),
                );
            }
            None => {}
        }

        ui.separator();
        ui.heading("View");
        ui.checkbox(&mut self.show_walk, "the current walk on the graph");
        ui.checkbox(
            &mut self.show_c,
            if self.settings.dim == 2 {
                "context vectors c"
            } else {
                "context vectors c (approximate above dim 2)"
            },
        )
        .on_hover_text(
            "Above dim 2, Z and C are drawn together on the 2 directions that keep the most of \
             their length, origin unmoved. The view shows how much the plane keeps, and how \
             close the on-screen dot product is to the real one. Untick it to see Z alone \
             (PCA around Z's mean: better for comparing embeddings, but no dot products).",
        );
        ui.add(
            egui::Slider::new(&mut self.magnify, 1.0..=500.0)
                .logarithmic(true)
                .text("arrow magnification"),
        );

        ui.separator();
        ui.label(RichText::new("Legend").strong());
        legend_ring(ui, U_RING, "u: the drone at position t; z_u is trained");
        legend_ring(
            ui,
            V_RING,
            "v: a drone within the window; c_v pulled toward z_u",
        );
        legend_ring(
            ui,
            NEG_RING,
            "negatives: random drones; c_n pushed away from z_u",
        );
        ui.label(
            RichText::new(
                "Embedding view: discs = z (the result), rings = c (training only). A large \
                 z_u · c_v means u and v were seen together.",
            )
            .small()
            .weak(),
        );
    }

    // -------------------------------------------------------------------------- bottom

    fn bottom(&mut self, ui: &mut egui::Ui) {
        ui.columns(2, |cols| {
            self.pair_details(&mut cols[0]);
            self.loss_plot(&mut cols[1]);
        });
    }

    fn pair_details(&self, ui: &mut egui::Ui) {
        let Some((ex, phase)) = self.shown() else {
            ui.heading("The pair being trained");
            ui.label(
                "Nothing trained yet. Press Space: the trainer takes the first walk, puts \
                 u = its first drone, and trains with every drone up to `window` steps away, \
                 one pair at a time. The Math tab (right) shows every step with its numbers.",
            );
            ui.label(format!(
                "Untrained, every dot product is 0 (C starts at zero), so every pair costs \
                 (1 + negatives)·ln 2 = {:.3}.",
                (1 + self.settings.negatives) as f64 * std::f64::consts::LN_2
            ));
            return;
        };
        egui::ScrollArea::vertical()
            .id_salt("pair_scroll")
            .show(ui, |ui| {
                ui.heading(if phase >= Phase::Context {
                    format!("The pair: u = {}, v = {}", ex.u, ex.v)
                } else {
                    format!("The pair: u = {}", ex.u)
                });

                // The walk, with the window around t.
                let walk = &self.trainer.walks[ex.walk];
                let w = self.settings.window;
                ui.label(format!(
                    "Walk {} (starts at drone {}), position t = {}, window ±{}:",
                    ex.walk, walk[0], ex.t, w
                ));
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    for (p, &d) in walk.iter().enumerate() {
                        let in_window = p + w >= ex.t && p <= ex.t + w;
                        let mut text = RichText::new(format!("{d}")).monospace();
                        if p == ex.t {
                            text = text.strong().color(Color32::BLACK).background_color(U_RING);
                        } else if p == ex.s && phase >= Phase::Context {
                            text = text.strong().color(Color32::BLACK).background_color(V_RING);
                        } else if in_window {
                            text = text.background_color(WALK_COLOR.gamma_multiply(0.35));
                        } else {
                            text = text.weak();
                        }
                        ui.label(text);
                    }
                });
                ui.add_space(4.0);

                // The table fills in as the steps are reached.
                let show = |reached: bool, s: String| {
                    if reached { s } else { "·".to_string() }
                };
                egui::Grid::new("pair_table")
                    .striped(true)
                    .num_columns(7)
                    .show(ui, |ui| {
                        for h in ["", "drone", "x = z_u·c", "σ(x)", "want", "g", "loss term"] {
                            ui.label(RichText::new(h).strong());
                        }
                        ui.end_row();
                        if phase < Phase::Context {
                            return;
                        }
                        let s = sigmoid(ex.x_pos);
                        ui.label(RichText::new("positive").color(V_RING));
                        ui.label(format!("{}", ex.v));
                        ui.label(show(phase >= Phase::Dots, format!("{:+.4}", ex.x_pos)));
                        ui.label(show(phase >= Phase::Sigmoid, format!("{s:.4}")));
                        ui.label("1");
                        ui.label(show(phase >= Phase::Errors, format!("{:+.4}", s - 1.0)));
                        ui.label(show(phase >= Phase::Loss, format!("{:.4}", -s.ln())));
                        ui.end_row();
                        if phase < Phase::Negatives {
                            return;
                        }
                        for (k, (&n, &x)) in ex.negs.iter().zip(&ex.x_negs).enumerate() {
                            let s = sigmoid(x);
                            ui.label(RichText::new(format!("negative {}", k + 1)).color(NEG_RING));
                            ui.label(format!("{n}"));
                            ui.label(show(phase >= Phase::Dots, format!("{x:+.4}")));
                            ui.label(show(phase >= Phase::Sigmoid, format!("{s:.4}")));
                            ui.label("0");
                            ui.label(show(phase >= Phase::Errors, format!("{s:+.4}")));
                            ui.label(show(
                                phase >= Phase::Loss,
                                format!("{:.4}", -sigmoid(-x).ln()),
                            ));
                            ui.end_row();
                        }
                    });
                if phase >= Phase::Loss {
                    ui.label(format!(
                        "Loss of this pair: {:.4}   (untrained: {:.4})",
                        ex.loss,
                        (1 + ex.negs.len()) as f64 * std::f64::consts::LN_2
                    ));
                }
                if let Some(after) = &ex.after {
                    ui.label(
                        RichText::new(format!(
                            "After the step: z_u·c_v = {:+.4} (was {:+.4})",
                            dot(&after.z, &after.cv),
                            ex.x_pos
                        ))
                        .small(),
                    );
                }
            });
    }

    fn loss_plot(&self, ui: &mut egui::Ui) {
        let tr = &self.trainer;
        ui.heading("Loss");
        let start = (1 + self.settings.negatives) as f32 * std::f32::consts::LN_2;
        let latest = tr.partial_loss().or(tr.loss_curve.last().copied());
        ui.label(
            RichText::new(format!(
                "Mean loss per pair, averaged over {} pair{}. Dashed: the untrained {:.3}.{}",
                tr.bucket,
                if tr.bucket == 1 { "" } else { "s" },
                start,
                latest.map_or(String::new(), |l| format!(" Now: {l:.3}."))
            ))
            .small(),
        );
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        let painter = ui.painter_at(rect);
        let visuals = ui.visuals();
        painter.rect_filled(rect, 2.0, visuals.extreme_bg_color);
        let plot = rect.shrink2(vec2(40.0, 12.0));

        let mut values = tr.loss_curve.clone();
        if let Some(p) = tr.partial_loss() {
            values.push(p);
        }
        let y_max = values.iter().copied().fold(start, f32::max) * 1.05;
        let x_max = (tr.total_pairs() as f32 / tr.bucket as f32).max(1.0);
        let to_screen = |i: f32, v: f32| {
            pos2(
                plot.left() + plot.width() * i / x_max,
                plot.bottom() - plot.height() * v / y_max,
            )
        };

        let weak = visuals.weak_text_color();
        painter.line_segment(
            [plot.left_bottom(), plot.right_bottom()],
            Stroke::new(1.0, weak),
        );
        painter.line_segment(
            [plot.left_bottom(), plot.left_top()],
            Stroke::new(1.0, weak),
        );
        for frac in [0.0, 0.5, 1.0] {
            let v = y_max * frac;
            painter.text(
                pos2(plot.left() - 4.0, to_screen(0.0, v).y),
                Align2::RIGHT_CENTER,
                format!("{v:.1}"),
                FontId::monospace(10.0),
                weak,
            );
        }
        painter.extend(Shape::dashed_line(
            &[to_screen(0.0, start), to_screen(x_max, start)],
            Stroke::new(1.0, weak),
            6.0,
            4.0,
        ));
        // Epoch boundaries.
        let per_epoch = tr.pairs.len() as f32 / tr.bucket as f32;
        for e in 1..self.settings.epochs {
            let x = to_screen(e as f32 * per_epoch, 0.0).x;
            painter.vline(
                x,
                plot.y_range(),
                Stroke::new(1.0, weak.gamma_multiply(0.3)),
            );
            painter.text(
                pos2(x + 3.0, plot.top()),
                Align2::LEFT_TOP,
                format!("epoch {}", e + 1),
                FontId::proportional(10.0),
                weak,
            );
        }
        if values.len() >= 2 {
            let points: Vec<Pos2> = values
                .iter()
                .enumerate()
                .map(|(i, &v)| to_screen(i as f32 + 1.0, v))
                .collect();
            painter.add(Shape::line(
                points,
                Stroke::new(1.5, Color32::from_rgb(242, 142, 43)),
            ));
        } else if let Some(&v) = values.first() {
            painter.circle_filled(to_screen(1.0, v), 3.0, Color32::from_rgb(242, 142, 43));
        }
        painter.text(
            plot.right_bottom() + vec2(0.0, 2.0),
            Align2::RIGHT_TOP,
            format!("{} pairs", tr.total_pairs()),
            FontId::proportional(10.0),
            weak,
        );
    }

    // ---------------------------------------------------------------------- graph view

    fn graph_canvas(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Graph and walk").strong());
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;
        let visuals = ui.visuals().clone();
        painter.rect_filled(rect, 2.0, visuals.extreme_bg_color);
        if self.graph.n() == 0 {
            return;
        }
        let n = self.graph.n();
        {
            // Input: drag moves a drone (drawing only), click-click toggles a link.
            let fit = Fit::new(&self.graph.nodes, rect.shrink(28.0));
            let at = |p: Pos2, nodes: &[Node]| {
                (0..nodes.len())
                    .map(|i| (i, fit.screen(nodes[i].pos).distance(p)))
                    .filter(|&(_, d)| d < NODE_PX + 5.0)
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(i, _)| i)
            };
            if let Some(p) = response.hover_pos() {
                self.hover_next = self.hover_next.or(at(p, &self.graph.nodes));
            }
            if response.drag_started() {
                let origin = ui
                    .input(|i| i.pointer.press_origin())
                    .or(response.interact_pointer_pos());
                self.dragging = origin.and_then(|p| at(p, &self.graph.nodes));
            }
            if let (Some(i), Some(p)) = (self.dragging, response.interact_pointer_pos())
                && response.dragged()
            {
                self.graph.nodes[i].pos = fit.world(p);
            }
            if response.drag_stopped() {
                self.dragging = None;
            }
            if response.clicked()
                && let Some(p) = response.interact_pointer_pos()
            {
                match (self.selected, at(p, &self.graph.nodes)) {
                    (Some(a), Some(b)) if a != b => {
                        self.graph.toggle(a, b);
                        self.graph_version += 1;
                        self.selected = None;
                    }
                    (Some(_), Some(_)) | (_, None) => self.selected = None,
                    (None, Some(b)) => self.selected = Some(b),
                }
            }
        }
        let fit = Fit::new(&self.graph.nodes, rect.shrink(28.0));
        let at = |i: usize| fit.screen(self.graph.nodes[i].pos);

        // Links.
        let max_w = self
            .graph
            .edges
            .edges
            .iter()
            .map(|e| e.weight)
            .fold(0.0, f64::max)
            .max(1e-12);
        let link_color = visuals.text_color().gamma_multiply(0.25);
        for e in &self.graph.edges.edges {
            let width = 0.6 + 2.0 * (e.weight / max_w).sqrt() as f32;
            painter.line_segment(
                [at(e.src as usize), at(e.dst as usize)],
                Stroke::new(width, link_color),
            );
        }

        // The current walk: faint outside the window, bright inside.
        let shown = self.shown();
        if self.show_walk
            && let Some((ex, _)) = shown
        {
            let walk = &self.trainer.walks[ex.walk];
            let w = self.settings.window;
            for (p, step) in walk.windows(2).enumerate() {
                let inside = p + w >= ex.t && p < ex.t + w;
                let color = WALK_COLOR.gamma_multiply(if inside { 0.95 } else { 0.25 });
                painter.line_segment(
                    [at(step[0]), at(step[1])],
                    Stroke::new(if inside { 3.0 } else { 1.5 }, color),
                );
            }
            // Arrowhead on the first step, to show which way the walk goes.
            if walk.len() >= 2 {
                let (a, b) = (at(walk[0]), at(walk[1]));
                painter.arrow(
                    a,
                    (b - a) * 0.5,
                    Stroke::new(2.0, WALK_COLOR.gamma_multiply(0.6)),
                );
            }
        }

        // Drones.
        let text = visuals.text_color();
        let weak = visuals.weak_text_color();
        let r = if n > 50 { NODE_PX * 0.7 } else { NODE_PX };
        for i in 0..n {
            let node = &self.graph.nodes[i];
            let c = at(i);
            draw_drone(
                &painter,
                c,
                r,
                node.role.unwrap_or(Role::Recon),
                self.color(i),
                Stroke::new(1.0, Color32::from_black_alpha(160)),
            );
            if n <= 50 {
                painter.text(
                    c + vec2(r + 2.0, -r),
                    Align2::LEFT_BOTTOM,
                    format!("{i}"),
                    FontId::proportional(11.0),
                    weak,
                );
            }
        }
        if let Some((ex, phase)) = shown {
            if phase >= Phase::Negatives {
                for &k in &ex.negs {
                    ring(&painter, at(k), NODE_PX + 5.0, NEG_RING);
                }
            }
            if phase >= Phase::Context {
                ring(&painter, at(ex.v), NODE_PX + 8.0, V_RING);
            }
            ring(&painter, at(ex.u), NODE_PX + 11.0, U_RING);
        }
        if let Some(i) = self.selected {
            painter.circle_stroke(at(i), NODE_PX + 14.0, Stroke::new(1.5, text));
            painter.text(
                rect.left_bottom() + vec2(8.0, -6.0),
                Align2::LEFT_BOTTOM,
                format!("Drone {i} selected: click another drone to add / remove a link."),
                FontId::proportional(12.0),
                text,
            );
        }
        if let Some(i) = self.hover {
            painter.circle_stroke(at(i), NODE_PX + 3.0, Stroke::new(2.0, text));
        }
        if let Some(i) = self.hover
            && response.hovered()
        {
            self.node_tip(response, i);
        }
    }

    fn node_tip(&self, response: egui::Response, i: usize) {
        let z = &self.trainer.z;
        let node = &self.graph.nodes[i];
        let mut similar: Vec<(usize, f64)> = (0..z.len())
            .filter(|&j| j != i)
            .map(|j| (j, cosine(&z[i], &z[j])))
            .collect();
        similar.sort_by(|a, b| b.1.total_cmp(&a.1));
        response.on_hover_ui_at_pointer(|ui| {
            ui.strong(format!(
                "Drone {i} · group {}{}",
                node.group,
                node.role
                    .map_or(String::new(), |r| format!(" · {}", role_name(r)))
            ));
            ui.label(format!("z = {}", vec_str(&z[i])));
            ui.label(format!("c = {}", vec_str(&self.trainer.c[i])));
            ui.label(format!(
                "In the walks {} times; negative weight count^0.75 = {:.1}",
                self.trainer.counts[i],
                self.trainer.counts[i].powf(0.75)
            ));
            ui.label("Most similar vectors (cosine):");
            for &(j, cs) in similar.iter().take(4) {
                let same = self.graph.nodes[j].group == node.group;
                ui.label(format!(
                    "  {j}: {cs:+.3}{}",
                    if same { "  (same group)" } else { "" }
                ));
            }
        });
    }

    // ------------------------------------------------------------------ embedding view

    fn embed_canvas(&mut self, ui: &mut egui::Ui) {
        let dim = self.settings.dim;
        let show_c = self.show_c;
        ui.label(
            RichText::new(if dim == 2 {
                "Embedding space: the vectors themselves (dim 2)".to_string()
            } else if show_c {
                format!(
                    "Embedding space: Z and C on their 2 main directions ({dim}-number vectors, \
                     origin kept)"
                )
            } else {
                format!("Embedding space: Z's two main directions (PCA of {dim}-number vectors)")
            })
            .strong(),
        );
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::hover());
        let rect = response.rect;
        let visuals = ui.visuals().clone();
        painter.rect_filled(rect, 2.0, visuals.extreme_bg_color);
        if self.trainer.n() == 0 {
            return;
        }
        let proj = if show_c {
            let previous = self.prev_axes.as_ref().filter(|p| p[0].len() == dim);
            Projection::joint(&self.trainer.z, &self.trainer.c, previous)
        } else {
            Projection::of(&self.trainer.z)
        };
        self.prev_axes = proj.axes.as_ref().map(|(axes, _)| axes.clone());
        // The origin is a real point (dot products make sense) unless Z alone is centred.
        let origin_kept = dim == 2 || show_c;
        let zs: Vec<[f64; 2]> = self.trainer.z.iter().map(|v| proj.point(v)).collect();
        let cs: Vec<[f64; 2]> = if show_c {
            self.trainer.c.iter().map(|v| proj.point(v)).collect()
        } else {
            Vec::new()
        };

        // Fit: centred on 0, grows at once, shrinks slowly.
        let needed = zs
            .iter()
            .chain(&cs)
            .map(|p| p[0].abs().max(p[1].abs()) as f32)
            .fold(1e-3, f32::max)
            * 1.15;
        if needed > self.view_r {
            self.view_r = needed;
        } else if needed < 0.8 * self.view_r {
            self.view_r += (needed - self.view_r) * 0.05;
            ui.ctx().request_repaint();
        }
        let scale = (rect.width().min(rect.height()) / 2.0 - 16.0) / self.view_r;
        let centre = rect.center();
        let screen = |p: [f64; 2]| centre + vec2(p[0] as f32 * scale, -(p[1] as f32) * scale);

        // Axes and a scale mark.
        let weak = visuals.weak_text_color();
        let axis = Stroke::new(1.0, weak.gamma_multiply(0.35));
        painter.hline(rect.x_range(), centre.y, axis);
        painter.vline(centre.x, rect.y_range(), axis);
        let tick = nice_tick(self.view_r);
        painter.circle_stroke(
            centre,
            tick * scale,
            Stroke::new(1.0, weak.gamma_multiply(0.2)),
        );
        painter.text(
            centre + vec2(tick * scale + 3.0, -3.0),
            Align2::LEFT_BOTTOM,
            fmt(tick as f64),
            FontId::monospace(10.0),
            weak,
        );

        // The pair: vectors from the origin, then the step (planned or done).
        let shown = self.shown();
        let mut pair_dots: Option<(f64, f64)> = None;
        if let Some((ex, phase)) = shown {
            let (z_now, cv_now) = match &ex.after {
                Some(a) => (&a.z, &a.cv),
                None => (&ex.z_before, &ex.cv_before),
            };
            if show_c && phase >= Phase::Context {
                let (a, b) = (proj.point(z_now), proj.point(cv_now));
                pair_dots = Some((dot(z_now, cv_now), a[0] * b[0] + a[1] * b[1]));
            }
            if origin_kept {
                let zu = screen(proj.point(z_now));
                painter.line_segment([centre, zu], Stroke::new(1.5, U_RING.gamma_multiply(0.7)));
                if show_c && phase >= Phase::Context {
                    let cv = screen(proj.point(cv_now));
                    painter.extend(Shape::dashed_line(
                        &[centre, cv],
                        Stroke::new(1.5, V_RING.gamma_multiply(0.7)),
                        5.0,
                        4.0,
                    ));
                }
                if show_c && (Phase::Dots..Phase::Update).contains(&phase) {
                    for cn in &ex.cn_before {
                        painter.extend(Shape::dashed_line(
                            &[centre, screen(proj.point(cn))],
                            Stroke::new(1.0, NEG_RING.gamma_multiply(0.45)),
                            3.0,
                            4.0,
                        ));
                    }
                }
            }
            let m = self.magnify;
            let lr = self.settings.lr;
            let arrow = |from: &[f64], to: &[f64], color: Color32| {
                let (a, b) = (screen(proj.point(from)), screen(proj.point(to)));
                let d = (b - a) * m;
                if d.length() > 1.0 {
                    painter.arrow(a, d, Stroke::new(2.0, color));
                }
            };
            let step_of = |v: &[f64], g: &[f64]| -> Vec<f64> {
                v.iter().zip(g).map(|(x, gi)| x - lr * gi).collect()
            };
            match &ex.after {
                // Done: from where the vectors were to where they are.
                Some(a) => {
                    arrow(&ex.z_before, &a.z, U_RING);
                    if show_c {
                        arrow(&ex.cv_before, &a.cv, V_RING);
                        for (b, after) in ex.cn_before.iter().zip(&a.cn) {
                            arrow(b, after, NEG_RING);
                        }
                    }
                }
                // Planned (step 8): where −lr × gradient would take them.
                None if phase >= Phase::Gradients => {
                    arrow(&ex.z_before, &step_of(&ex.z_before, &ex.grad_z), U_RING);
                    if show_c {
                        arrow(&ex.cv_before, &step_of(&ex.cv_before, &ex.grad_zv), V_RING);
                        for (b, g) in ex.cn_before.iter().zip(&ex.grad_neg) {
                            arrow(b, &step_of(b, g), NEG_RING);
                        }
                    }
                }
                None => {}
            }
        }

        // Points: rings for c, discs for z.
        let small = self.trainer.n() > 50;
        let r = if small { 4.0 } else { 7.0 };
        for (i, p) in cs.iter().enumerate() {
            painter.circle_stroke(screen(*p), r - 1.0, Stroke::new(1.5, self.color(i)));
        }
        for (i, p) in zs.iter().enumerate() {
            let c = screen(*p);
            painter.circle(
                c,
                r,
                self.color(i),
                Stroke::new(1.0, Color32::from_black_alpha(160)),
            );
            if !small {
                painter.text(
                    c + vec2(r + 1.0, -r),
                    Align2::LEFT_BOTTOM,
                    format!("{i}"),
                    FontId::proportional(11.0),
                    visuals.text_color(),
                );
            }
        }
        if let Some((ex, phase)) = shown {
            ring(&painter, screen(zs[ex.u]), r + 5.0, U_RING);
            if show_c && phase >= Phase::Context {
                ring(&painter, screen(cs[ex.v]), r + 4.0, V_RING);
            }
            if show_c && phase >= Phase::Negatives {
                for &k in &ex.negs {
                    ring(&painter, screen(cs[k]), r + 3.0, NEG_RING);
                }
            }
        }

        // Hover (shared with the graph view).
        if let Some(p) = response.hover_pos() {
            let near = (0..zs.len())
                .map(|i| (i, screen(zs[i]).distance(p)))
                .filter(|&(_, d)| d < r + 5.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i);
            self.hover_next = self.hover_next.or(near);
        }
        if let Some(i) = self.hover {
            painter.circle_stroke(
                screen(zs[i]),
                r + 3.0,
                Stroke::new(2.0, visuals.text_color()),
            );
            if show_c {
                painter.circle_stroke(
                    screen(cs[i]),
                    r + 2.0,
                    Stroke::new(1.0, visuals.text_color()),
                );
            }
        }
        painter.text(
            rect.left_bottom() + vec2(8.0, -6.0),
            Align2::LEFT_BOTTOM,
            if show_c {
                "discs: z (the embedding) · rings: c (contexts) · arrows: the step × magnification"
            } else {
                "discs: z (the embedding) · arrow: the step of z_u × magnification"
            },
            FontId::proportional(11.0),
            weak,
        );
        // How far to trust the picture's angles (projection of Z and C together).
        if dim > 2
            && show_c
            && let Some(kept) = proj.kept
        {
            let mut lines = vec![format!(
                "the plane keeps {:.0} % of the vectors' length² (the rest is in the other {} \
                 directions)",
                100.0 * kept,
                dim - 2
            )];
            if let Some((real, seen)) = pair_dots {
                lines.push(format!(
                    "z_u · c_v: real {real:+.4}, on screen {seen:+.4} (what the plane drops: \
                     {:+.4})",
                    real - seen
                ));
            }
            let color = if kept > 0.8 {
                visuals.text_color()
            } else {
                Color32::from_rgb(237, 160, 60)
            };
            painter.text(
                rect.left_top() + vec2(8.0, 6.0),
                Align2::LEFT_TOP,
                lines.join("\n"),
                FontId::proportional(12.0),
                color,
            );
        }
        if let Some(i) = self.hover
            && response.hovered()
        {
            self.node_tip(response, i);
        }
    }

    // --------------------------------------------------------------------- right panel

    fn right(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Math, "Math");
            ui.selectable_value(&mut self.tab, Tab::Similarity, "Similarity of Z");
            ui.selectable_value(&mut self.tab, Tab::Cooccurrence, "Co-occurrence");
            ui.selectable_value(&mut self.tab, Tab::Negatives, "Negatives");
        });
        ui.separator();
        if self.tab == Tab::Math {
            self.math_tab(ui);
            return;
        }

        let (right, inside, across) = group_checks(&self.trainer.z, &self.graph.nodes);
        let n = self.graph.n();
        ui.label(RichText::new("What the vectors learned").strong());
        ui.label(format!(
            "Most similar vector in the same {}: {right} of {n} drones",
            if self.preset == Preset::Swarm {
                "squad"
            } else {
                "group"
            }
        ));
        ui.label(format!(
            "Mean cosine: inside groups {inside:+.3}, across groups {across:+.3}"
        ));
        if self.preset == Preset::Swarm {
            let by_role: Vec<Node> = self
                .graph
                .nodes
                .iter()
                .map(|nd| Node {
                    group: nd.role.map_or(0, |r| r as usize),
                    ..nd.clone()
                })
                .collect();
            let (right_role, inside_role, across_role) = group_checks(&self.trainer.z, &by_role);
            ui.label(format!(
                "Same role as its most similar vector: {right_role} of {n}; mean cosine \
                 same role {inside_role:+.3}, different {across_role:+.3}"
            ));
        }
        ui.separator();

        // Drones sorted by group, so groups show up as blocks.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (self.group_key(i), i));
        match self.tab {
            Tab::Math => {}
            Tab::Similarity => {
                ui.label(
                    RichText::new(
                        "cos(z_i, z_j): orange = same direction, blue = opposite. Drones \
                         sorted by group: training should grow orange blocks on the diagonal.",
                    )
                    .small(),
                );
                let z = &self.trainer.z;
                let hit = heatmap(
                    ui,
                    &order,
                    |i, j| cosine(&z[i], &z[j]),
                    -1.0,
                    1.0,
                    |i| self.color(i),
                );
                if let Some((i, j)) = hit {
                    self.hover_next = Some(i);
                    ui.label(format!("cos(z_{i}, z_{j}) = {:+.3}", cosine(&z[i], &z[j])));
                }
            }
            Tab::Cooccurrence => {
                ui.label(
                    RichText::new(format!(
                        "How often j is within {} steps of i in the walks (one epoch): the \
                         positive pairs skip-gram is trained on. Similar vectors come from \
                         similar rows here, not from links.",
                        self.settings.window
                    ))
                    .small(),
                );
                let m = &self.cooc;
                let max = m.iter().flatten().copied().fold(1.0, f64::max);
                let hit = heatmap(ui, &order, |i, j| m[i][j], 0.0, max, |i| self.color(i));
                if let Some((i, j)) = hit {
                    self.hover_next = Some(i);
                    ui.label(format!("({i}, {j}) seen together {} times", m[i][j]));
                }
            }
            Tab::Negatives => self.negatives_tab(ui, &order),
        }
    }

    fn group_key(&self, i: usize) -> usize {
        match self.graph.nodes[i].role {
            Some(r) if self.color_by_role => r as usize,
            _ => self.graph.nodes[i].group,
        }
    }

    /// Every step of the shown pair: the formula, then the formula with this pair's numbers.
    fn math_tab(&self, ui: &mut egui::Ui) {
        let Some((ex, phase)) = self.shown() else {
            ui.label(RichText::new("The maths of one training pair").strong());
            ui.label(
                "Press Space (with \"Explain each pair in 9 steps\" on) to start the first \
                 pair. Each press shows one more step here, with this pair's real numbers.",
            );
            for p in Phase::ALL {
                ui.label(RichText::new(format!("{}. {}", p.number(), p.title())).weak());
            }
            return;
        };
        let explaining = self.pending.is_some();
        ui.label(
            RichText::new(if explaining {
                format!(
                    "Pair {} — step {} of 9. Space: next step.",
                    self.trainer.done + 1,
                    phase.number()
                )
            } else {
                format!(
                    "Pair {} — all 9 steps done. Space: {}.",
                    self.trainer.done,
                    if self.explain {
                        "start the next pair"
                    } else {
                        "next pair"
                    }
                )
            })
            .strong(),
        );
        ui.add_space(4.0);
        for p in Phase::ALL {
            let current = explaining && p == phase;
            if p > phase {
                ui.label(RichText::new(format!("{}. {}", p.number(), p.title())).weak());
                continue;
            }
            let frame = egui::Frame::group(ui.style()).stroke(if current {
                Stroke::new(2.0, U_RING)
            } else {
                ui.visuals().widgets.noninteractive.bg_stroke
            });
            let response = frame
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    let title = RichText::new(format!("{}. {}", p.number(), p.title())).strong();
                    ui.label(if current { title.color(U_RING) } else { title });
                    self.math_step(ui, ex, p);
                })
                .response;
            if current && self.scroll_math.replace(false) {
                response.scroll_to_me(Some(egui::Align::Center));
            }
        }
    }

    fn math_step(&self, ui: &mut egui::Ui, ex: &Example, p: Phase) {
        let tr = &self.trainer;
        let formula = |ui: &mut egui::Ui, s: &str| {
            ui.label(RichText::new(s).monospace().color(WALK_COLOR));
        };
        let mono = |ui: &mut egui::Ui, s: String| {
            ui.label(RichText::new(s).monospace());
        };
        let note = |ui: &mut egui::Ui, s: String| {
            ui.label(RichText::new(s).small().weak());
        };
        let c_is_zero = ex.cv_before.iter().all(|&x| x == 0.0)
            && ex.cn_before.iter().flatten().all(|&x| x == 0.0);
        match p {
            Phase::Position => {
                let walk = &tr.walks[ex.walk];
                let (lo, hi) = window_of(walk.len(), ex.t, self.settings.window);
                formula(ui, "u = walk[t]");
                mono(ui, format!("u = walk[{}] = {}", ex.t, ex.u));
                mono(ui, format!("z_u = {}", vec_str(&ex.z_before)));
                note(
                    ui,
                    format!(
                        "Walk {} (starts at drone {}). The window is positions {lo} to {hi} \
                         (t ± {}), so u gets {} positive pairs from this position, one after \
                         the other. z_u is u's embedding: the vector this pair trains.",
                        ex.walk,
                        walk[0],
                        self.settings.window,
                        hi - lo
                    ),
                );
            }
            Phase::Context => {
                formula(ui, "v = walk[s],   0 < |s − t| ≤ window");
                let d = ex.t.abs_diff(ex.s);
                mono(
                    ui,
                    format!(
                        "v = walk[{}] = {}   ({d} step{} from u)",
                        ex.s,
                        ex.v,
                        if d == 1 { "" } else { "s" }
                    ),
                );
                mono(ui, format!("c_v = {}", vec_str(&ex.cv_before)));
                note(
                    ui,
                    "To learn: u and v go together, so we want σ(z_u · c_v) close to 1. \
                     c_v is v's context vector (its ring), not its z."
                        .to_string(),
                );
            }
            Phase::Negatives => {
                formula(ui, "P(k) = count_k^0.75 / Σ count^0.75");
                note(
                    ui,
                    format!(
                        "Each draw: r uniform in [0, {:.1}), k = the first drone whose running \
                         total of count^0.75 passes r (binary search). A draw equal to u or \
                         v is thrown away and drawn again.",
                        tr.total
                    ),
                );
                for &(r, k, kept) in &ex.draws {
                    let chance = tr.counts[k].powf(0.75) / tr.total;
                    mono(
                        ui,
                        format!(
                            "r = {r:7.2} → k = {k:<3} (count {:>4}, P {:.3})  {}",
                            tr.counts[k],
                            chance,
                            if kept {
                                "✔"
                            } else {
                                "✘ = u or v, draw again"
                            }
                        ),
                    );
                }
                for (j, &k) in ex.negs.iter().enumerate() {
                    mono(ui, format!("c_{k} = {}", vec_str(&ex.cn_before[j])));
                }
                note(
                    ui,
                    "To learn: these random drones do not go with u, so we want σ(z_u · c_k) \
                     close to 0."
                        .to_string(),
                );
            }
            Phase::Dots => {
                formula(ui, "x = z_u · c = z_u[0]·c[0] + z_u[1]·c[1] + …");
                mono(
                    ui,
                    format!(
                        "x_v = z_u · c_{} = {}",
                        ex.v,
                        expand(&ex.z_before, &ex.cv_before)
                    ),
                );
                for (j, &k) in ex.negs.iter().enumerate() {
                    mono(
                        ui,
                        format!(
                            "x_{k} = z_u · c_{k} = {}",
                            expand(&ex.z_before, &ex.cn_before[j])
                        ),
                    );
                }
                note(
                    ui,
                    "x is how much the model thinks u and that drone go together right now: \
                     vectors pointing the same way give a big x."
                        .to_string(),
                );
                if c_is_zero {
                    note(ui, "C starts at zero, so every x is 0 for now.".to_string());
                }
            }
            Phase::Sigmoid => {
                formula(ui, "σ(x) = 1 / (1 + e^(−x))");
                let line = |name: String, x: f64| {
                    format!("σ(x_{name}) = 1 / (1 + e^({:+.4})) = {:.4}", -x, sigmoid(x))
                };
                mono(ui, line("v".into(), ex.x_pos));
                for (&k, &x) in ex.negs.iter().zip(&ex.x_negs) {
                    mono(ui, line(k.to_string(), x));
                }
                note(
                    ui,
                    "σ turns x into a probability between 0 and 1: the model's belief that \
                     the two drones go together. x = 0 gives 0.5: no idea."
                        .to_string(),
                );
            }
            Phase::Errors => {
                formula(ui, "g = σ(x) − want      (want: 1 for v, 0 for a negative)");
                let s = sigmoid(ex.x_pos);
                mono(ui, format!("g_v = {s:.4} − 1 = {:+.4}", s - 1.0));
                for (&k, &x) in ex.negs.iter().zip(&ex.x_negs) {
                    let s = sigmoid(x);
                    mono(ui, format!("g_{k} = {s:.4} − 0 = {s:+.4}"));
                }
                note(
                    ui,
                    "The sign says the direction (negative: pull together, positive: push \
                     apart), the size says how hard. A right answer has g ≈ 0: nothing to fix."
                        .to_string(),
                );
            }
            Phase::Loss => {
                formula(ui, "L = −ln σ(x_v) − Σₖ ln σ(−x_k)");
                let mut terms = vec![-sigmoid(ex.x_pos).ln()];
                terms.extend(ex.x_negs.iter().map(|&x| -sigmoid(-x).ln()));
                let parts: Vec<String> = terms.iter().map(|t| format!("{t:.4}")).collect();
                mono(ui, format!("L = {} = {:.4}", parts.join(" + "), ex.loss));
                note(
                    ui,
                    format!(
                        "One term per row: small when σ is near what we want. Untrained (every \
                         x = 0): (1 + {}) × ln 2 = {:.4}. Perfect: 0.",
                        ex.negs.len(),
                        (1 + ex.negs.len()) as f64 * std::f64::consts::LN_2
                    ),
                );
            }
            Phase::Gradients => {
                formula(ui, "∂L/∂c_v = g_v · z_u");
                formula(ui, "∂L/∂c_k = g_k · z_u");
                formula(ui, "∂L/∂z_u = g_v · c_v + Σₖ g_k · c_k");
                let g_v = sigmoid(ex.x_pos) - 1.0;
                mono(
                    ui,
                    format!(
                        "∂L/∂c_v = {g_v:+.4} × {} = {}",
                        vec_str(&ex.z_before),
                        vec_str(&ex.grad_zv)
                    ),
                );
                for (j, &k) in ex.negs.iter().enumerate() {
                    mono(
                        ui,
                        format!(
                            "∂L/∂c_{k} = {:+.4} × z_u = {}",
                            sigmoid(ex.x_negs[j]),
                            vec_str(&ex.grad_neg[j])
                        ),
                    );
                }
                mono(ui, format!("∂L/∂z_u = {}", vec_str(&ex.grad_z)));
                note(
                    ui,
                    "Each gradient is the error g times the other vector of its dot product. \
                     The arrows on the embedding view show −lr × gradient (magnified)."
                        .to_string(),
                );
                if c_is_zero {
                    note(
                        ui,
                        "Every c is still zero, so ∂L/∂z_u = 0: z_u can't move until the \
                         contexts do."
                            .to_string(),
                    );
                }
            }
            Phase::Update => {
                let lr = self.settings.lr;
                formula(ui, &format!("vector ← vector − lr × gradient,   lr = {lr}"));
                let Some(a) = &ex.after else {
                    return;
                };
                mono(
                    ui,
                    format!("c_v: {} → {}", vec_str(&ex.cv_before), vec_str(&a.cv)),
                );
                for (j, &k) in ex.negs.iter().enumerate() {
                    mono(
                        ui,
                        format!(
                            "c_{k}: {} → {}",
                            vec_str(&ex.cn_before[j]),
                            vec_str(&a.cn[j])
                        ),
                    );
                }
                mono(
                    ui,
                    format!("z_u: {} → {}", vec_str(&ex.z_before), vec_str(&a.z)),
                );
                let x_after = dot(&a.z, &a.cv);
                mono(
                    ui,
                    format!(
                        "check: z_u · c_v  {:+.5} → {:+.5}  {}",
                        ex.x_pos,
                        x_after,
                        arrow_word(x_after - ex.x_pos, true)
                    ),
                );
                for (j, &k) in ex.negs.iter().enumerate() {
                    let x_after = dot(&a.z, &a.cn[j]);
                    mono(
                        ui,
                        format!(
                            "       z_u · c_{k}  {:+.5} → {:+.5}  {}",
                            ex.x_negs[j],
                            x_after,
                            arrow_word(x_after - ex.x_negs[j], false)
                        ),
                    );
                }
                note(
                    ui,
                    "The positive dot product went up, the negatives' went down: one small \
                     step toward the right answers. Repeat for every pair of every walk."
                        .to_string(),
                );
            }
        }
    }

    fn negatives_tab(&self, ui: &mut egui::Ui, order: &[usize]) {
        let tr = &self.trainer;
        let raw: f64 = tr.counts.iter().sum();
        let n = tr.n();
        ui.label(
            RichText::new(
                "Chance of being drawn as a negative. Grey: ∝ count (busy drones dominate). \
                 Orange: ∝ count^0.75, what train_skipgram uses. The line: uniform, 1/n. The \
                 0.75 power flattens the grey toward the line.",
            )
            .small(),
        );
        let row_h = if n > 40 { 9.0 } else { 16.0 };
        let width = ui.available_width().max(200.0);
        let (rect, _) = ui.allocate_exact_size(vec2(width, row_h * n as f32 + 4.0), Sense::hover());
        let painter = ui.painter_at(rect);
        let weak = ui.visuals().weak_text_color();
        let bar_left = rect.left() + 36.0;
        let max_p = (0..n)
            .map(|i| (tr.counts[i] / raw).max(tr.counts[i].powf(0.75) / tr.total))
            .fold(1.0 / n as f64, f64::max);
        let x_of = |p: f64| bar_left + (rect.right() - 8.0 - bar_left) * (p / max_p) as f32;
        for (row, &i) in order.iter().enumerate() {
            let y = rect.top() + row as f32 * row_h;
            if row_h >= 12.0 {
                painter.text(
                    pos2(rect.left() + 30.0, y + row_h / 2.0),
                    Align2::RIGHT_CENTER,
                    format!("{i}"),
                    FontId::monospace(10.0),
                    self.color(i),
                );
            } else {
                painter.rect_filled(
                    Rect::from_min_size(pos2(rect.left() + 24.0, y + 1.0), vec2(6.0, row_h - 2.0)),
                    1.0,
                    self.color(i),
                );
            }
            let p_raw = tr.counts[i] / raw.max(1e-12);
            let p_pow = tr.counts[i].powf(0.75) / tr.total.max(1e-12);
            let h = (row_h - 2.0) / 2.0;
            painter.rect_filled(
                Rect::from_min_max(pos2(bar_left, y + 1.0), pos2(x_of(p_raw), y + 1.0 + h)),
                1.0,
                Color32::from_gray(140),
            );
            painter.rect_filled(
                Rect::from_min_max(
                    pos2(bar_left, y + 1.0 + h),
                    pos2(x_of(p_pow), y + 1.0 + 2.0 * h),
                ),
                1.0,
                Color32::from_rgb(242, 142, 43),
            );
        }
        painter.vline(x_of(1.0 / n as f64), rect.y_range(), Stroke::new(1.0, weak));
    }
}

/// A vector as `[+0.012, -0.340]`; long vectors show their first 6 numbers.
fn vec_str(v: &[f64]) -> String {
    let shown: Vec<String> = v.iter().take(6).map(|x| format!("{x:+.3}")).collect();
    if v.len() > 6 {
        format!("[{}, … ({} numbers)]", shown.join(", "), v.len())
    } else {
        format!("[{}]", shown.join(", "))
    }
}

/// a · b written out term by term (up to 3 terms), then its value.
fn expand(a: &[f64], b: &[f64]) -> String {
    let value = dot(a, b);
    if a.len() <= 3 {
        let terms: Vec<String> = a
            .iter()
            .zip(b)
            .map(|(x, y)| format!("({x:+.3})({y:+.3})"))
            .collect();
        format!("{} = {value:+.4}", terms.join(" + "))
    } else {
        format!("sum of {} products = {value:+.4}", a.len())
    }
}

/// "↑ pulled together" / "↓ pushed apart" (or the surprise, if it went the other way).
fn arrow_word(change: f64, positive: bool) -> &'static str {
    match (change >= 0.0, positive) {
        (true, true) => "↑ pulled together",
        (false, false) => "↓ pushed apart",
        (false, true) => "↓ (went down: another update in this step outweighed it)",
        (true, false) => "↑ (went up: another update in this step outweighed it)",
    }
}

/// Fits drawing positions into a rectangle (same scale on both axes).
struct Fit {
    min: [f32; 2],
    scale: f32,
    offset: egui::Vec2,
}

impl Fit {
    fn new(nodes: &[Node], rect: Rect) -> Fit {
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for nd in nodes {
            for k in 0..2 {
                lo[k] = lo[k].min(nd.pos[k]);
                hi[k] = hi[k].max(nd.pos[k]);
            }
        }
        let span = [(hi[0] - lo[0]).max(1e-3), (hi[1] - lo[1]).max(1e-3)];
        let scale = (rect.width() / span[0]).min(rect.height() / span[1]);
        let used = vec2(span[0] * scale, span[1] * scale);
        Fit {
            min: lo,
            scale,
            offset: rect.min.to_vec2() + (rect.size() - used) / 2.0,
        }
    }

    fn screen(&self, p: [f32; 2]) -> Pos2 {
        pos2(
            (p[0] - self.min[0]) * self.scale,
            (p[1] - self.min[1]) * self.scale,
        ) + self.offset
    }

    fn world(&self, p: Pos2) -> [f32; 2] {
        let q = p - self.offset;
        [
            q.x / self.scale + self.min[0],
            q.y / self.scale + self.min[1],
        ]
    }
}

/// A square heatmap of `value(i, j)` over the drones in `order`; colour bars on the edges
/// show each drone's group. Returns the hovered (i, j).
fn heatmap(
    ui: &mut egui::Ui,
    order: &[usize],
    value: impl Fn(usize, usize) -> f64,
    lo: f64,
    hi: f64,
    color: impl Fn(usize) -> Color32,
) -> Option<(usize, usize)> {
    let n = order.len();
    if n == 0 {
        return None;
    }
    let edge = 8.0;
    let cell = ((ui.available_width() - edge - 4.0) / n as f32).clamp(2.0, 24.0);
    let size = vec2(edge + cell * n as f32, edge + cell * n as f32);
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter_at(rect);
    let bg = ui.visuals().extreme_bg_color;
    let origin = rect.min + vec2(edge, edge);
    let warm = Color32::from_rgb(242, 142, 43);
    let cold = Color32::from_rgb(78, 121, 167);
    for (a, &i) in order.iter().enumerate() {
        painter.rect_filled(
            Rect::from_min_size(
                pos2(rect.min.x, origin.y + a as f32 * cell),
                vec2(edge - 2.0, cell),
            ),
            0.0,
            color(i),
        );
        painter.rect_filled(
            Rect::from_min_size(
                pos2(origin.x + a as f32 * cell, rect.min.y),
                vec2(cell, edge - 2.0),
            ),
            0.0,
            color(i),
        );
        for (b, &j) in order.iter().enumerate() {
            let v = value(i, j);
            let fill = if lo < 0.0 {
                // Diverging: blue below 0, orange above.
                if v >= 0.0 {
                    mix(bg, warm, (v / hi).clamp(0.0, 1.0) as f32)
                } else {
                    mix(bg, cold, (v / lo).clamp(0.0, 1.0) as f32)
                }
            } else {
                mix(
                    bg,
                    warm,
                    ((v - lo) / (hi - lo)).clamp(0.0, 1.0).sqrt() as f32,
                )
            };
            painter.rect_filled(
                Rect::from_min_size(
                    origin + vec2(b as f32 * cell, a as f32 * cell),
                    vec2(cell, cell),
                ),
                0.0,
                fill,
            );
        }
    }
    let hit = response.hover_pos().and_then(|p| {
        let q = p - origin;
        if q.x < 0.0 || q.y < 0.0 {
            return None;
        }
        let (b, a) = ((q.x / cell) as usize, (q.y / cell) as usize);
        (a < n && b < n).then(|| (order[a], order[b]))
    });
    if let Some((i, j)) = hit {
        let a = order.iter().position(|&x| x == i).unwrap_or(0);
        let b = order.iter().position(|&x| x == j).unwrap_or(0);
        painter.rect_stroke(
            Rect::from_min_size(
                origin + vec2(b as f32 * cell, a as f32 * cell),
                vec2(cell, cell),
            ),
            0.0,
            Stroke::new(1.5, ui.visuals().text_color()),
            StrokeKind::Inside,
        );
    }
    hit
}

fn ring(painter: &egui::Painter, c: Pos2, r: f32, color: Color32) {
    painter.circle_stroke(c, r, Stroke::new(2.5, color));
}

fn legend_ring(ui: &mut egui::Ui, color: Color32, text: &str) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(vec2(18.0, 18.0), Sense::hover());
        ui.painter()
            .circle_stroke(rect.center(), 6.0, Stroke::new(2.5, color));
        ui.label(RichText::new(text).small());
    });
}

/// A round number (1, 2 or 5 × 10^k) a bit below `r`, for the scale circle.
fn nice_tick(r: f32) -> f32 {
    let target = r * 0.6;
    let base = 10f32.powf(target.log10().floor());
    [5.0, 2.0, 1.0]
        .into_iter()
        .map(|m| m * base)
        .find(|&t| t <= target)
        .unwrap_or(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained(preset: Preset, settings: Settings) -> (Graph, Trainer) {
        let graph = Graph::build(preset, 0);
        let walks = make_walks(&graph, &settings);
        let mut tr = Trainer::new(walks, graph.n(), settings);
        tr.finish();
        (graph, tr)
    }

    fn same_as_train_skipgram(tr: &Trainer, settings: &Settings) -> bool {
        let mut rng = ChaCha8Rng::seed_from_u64(settings.train_seed);
        train_skipgram(&tr.walks, tr.n(), &settings.config(), &mut rng) == tr.z
    }

    #[test]
    fn step_by_step_gives_the_same_z_as_train_skipgram() {
        // The playground's trainer must be the user's algorithm, not a look-alike.
        for (preset, dim) in [
            (Preset::TwoGroups, 2),
            (Preset::Ring, 5),
            (Preset::StarAndChain, 3),
        ] {
            let settings = Settings {
                dim,
                epochs: 2,
                ..Settings::default()
            };
            let (_, tr) = trained(preset, settings);
            assert!(same_as_train_skipgram(&tr, &settings), "{}", preset.name());
        }
    }

    #[test]
    fn nine_steps_train_exactly_one_pair_and_mixing_modes_keeps_z_exact() {
        let mut p = Playground::default();
        // 9 presses = one pair: prepared at press 1, applied at press 9.
        for press in 1..=9 {
            p.next();
            if press < 9 {
                assert_eq!(p.trainer.done, 0, "press {press}");
                assert_eq!(p.phase.number(), press);
                assert!(p.pending.is_some());
            }
        }
        assert_eq!(p.trainer.done, 1);
        assert!(p.pending.is_none());
        assert!(p.trainer.last.as_ref().unwrap().after.is_some());

        // Half-way through a pair, then bulk training: the pair is finished first, so the
        // random draws stay in train_skipgram's order and the final Z is the same.
        for _ in 0..4 {
            p.next();
        }
        p.flush();
        p.trainer.run(50);
        for _ in 0..6 {
            p.next();
        }
        p.flush();
        p.trainer.finish();
        assert!(same_as_train_skipgram(&p.trainer, &p.settings));
    }

    #[test]
    fn the_negative_draws_are_recorded() {
        let graph = Graph::build(Preset::TwoGroups, 0);
        let settings = Settings::default();
        let mut tr = Trainer::new(make_walks(&graph, &settings), graph.n(), settings);
        let mut rejected = 0;
        for _ in 0..2000 {
            let ex = tr.prepare().unwrap();
            let kept: Vec<usize> = ex.draws.iter().filter(|d| d.2).map(|d| d.1).collect();
            assert_eq!(kept, ex.negs);
            for &(r, k, ok) in &ex.draws {
                assert!(r >= 0.0 && r < tr.total);
                assert_eq!(ok, k != ex.v && k != ex.u);
                rejected += usize::from(!ok);
            }
            assert!(!ex.negs.contains(&ex.u));
            tr.apply(ex, false);
        }
        assert!(
            rejected > 0,
            "on 12 drones, u or v must get drawn now and then"
        );
    }

    #[test]
    fn presets_are_valid_graphs() {
        for preset in Preset::ALL {
            let g = Graph::build(preset, 0);
            assert_eq!(g.edges.n as usize, g.n(), "{}", preset.name());
            assert!(g.n() > 0);
            for e in &g.edges.edges {
                assert!(
                    e.src < e.dst && (e.dst as usize) < g.n(),
                    "{}",
                    preset.name()
                );
                assert!(e.weight > 0.0);
            }
        }
        // The two groups: 2 × 15 clique links + the bridge 5–6.
        let g = Graph::build(Preset::TwoGroups, 0);
        assert_eq!(g.edges.edges.len(), 31);
        assert!(g.edges.edges.iter().any(|e| (e.src, e.dst) == (5, 6)));
    }

    #[test]
    fn toggling_a_link_adds_then_removes_it() {
        let mut g = Graph::build(Preset::Path, 0);
        let before = g.edges.edges.len();
        g.toggle(7, 2);
        assert_eq!(g.edges.edges.len(), before + 1);
        assert!(g.edges.edges.iter().any(|e| (e.src, e.dst) == (2, 7)));
        g.toggle(2, 7);
        assert_eq!(g.edges.edges.len(), before);
    }

    #[test]
    fn finish_walk_and_epoch_stop_at_the_boundary() {
        let graph = Graph::build(Preset::TwoGroups, 0);
        let settings = Settings::default();
        let mut tr = Trainer::new(make_walks(&graph, &settings), graph.n(), settings);
        tr.step(true);
        tr.finish_walk();
        assert_eq!(tr.next_position(), Some((0, 1)));
        assert_eq!(
            tr.last.as_ref().unwrap().walk,
            0,
            "last pair recorded is walk 0's"
        );
        tr.finish_epoch();
        assert_eq!(tr.done, tr.pairs.len());
        assert_eq!(tr.next_position(), Some((1, 0)));
        tr.finish();
        assert!(tr.finished());
        assert_eq!(tr.next_position(), None);
    }

    #[test]
    fn the_recorded_pair_is_what_the_step_did() {
        let graph = Graph::build(Preset::TwoGroups, 0);
        let settings = Settings::default();
        let mut tr = Trainer::new(make_walks(&graph, &settings), graph.n(), settings);
        tr.run(500);
        let ex = tr.last.clone().unwrap();
        let after = ex.after.clone().unwrap();
        assert_eq!(ex.u, tr.walks[ex.walk][ex.t]);
        assert_eq!(ex.v, tr.walks[ex.walk][ex.s]);
        assert!(ex.t.abs_diff(ex.s) <= settings.window && ex.t != ex.s);
        assert!(!ex.negs.contains(&ex.v));
        assert_eq!(after.z, tr.z[ex.u]);
        assert!((ex.x_pos - dot(&ex.z_before, &ex.cv_before)).abs() < 1e-15);
        // The loss and gradients in the record are the formula's.
        let negs: Vec<&[f64]> = ex.cn_before.iter().map(|c| c.as_slice()).collect();
        let (loss, grad_z, grad_zv, grad_neg) =
            pair_loss_and_grads(&ex.z_before, &ex.cv_before, &negs);
        assert_eq!(loss, ex.loss);
        assert_eq!(grad_z, ex.grad_z);
        assert_eq!(grad_zv, ex.grad_zv);
        assert_eq!(grad_neg, ex.grad_neg);
        // And the step is vector − lr × gradient.
        for i in 0..settings.dim {
            assert!((after.z[i] - (ex.z_before[i] - settings.lr * ex.grad_z[i])).abs() < 1e-15);
        }
    }

    #[test]
    fn cooccurrence_counts_every_positive_pair_once_and_is_symmetric() {
        let graph = Graph::build(Preset::Ring, 0);
        let settings = Settings::default();
        let walks = make_walks(&graph, &settings);
        let m = cooccurrence(&walks, graph.n(), settings.window);
        let tr = Trainer::new(walks, graph.n(), settings);
        let total: f64 = m.iter().flatten().sum();
        assert_eq!(total as usize, tr.pairs.len());
        for (i, row) in m.iter().enumerate() {
            for (j, &count) in row.iter().enumerate() {
                assert_eq!(count, m[j][i]);
            }
        }
    }

    #[test]
    fn the_two_groups_come_apart() {
        let (graph, tr) = trained(Preset::TwoGroups, Settings::default());
        let (right, inside, across) = group_checks(&tr.z, &graph.nodes);
        assert!(right >= 11, "{right}/12");
        assert!(
            inside > across + 0.5,
            "inside {inside:.3}, across {across:.3}"
        );
    }

    #[test]
    fn the_swarm_preset_finds_squads() {
        // The preset as loaded (dim 16): most drones' most similar vector is a squad-mate.
        let settings = Settings {
            dim: 16,
            ..Settings::default()
        };
        let (graph, tr) = trained(Preset::Swarm, settings);
        assert_eq!(graph.n(), 100);
        let (right, inside, across) = group_checks(&tr.z, &graph.nodes);
        println!("squads: {right}/100 nearest, cosine inside {inside:.3} across {across:.3}");
        let by_role: Vec<Node> = graph
            .nodes
            .iter()
            .map(|nd| Node {
                group: nd.role.unwrap() as usize,
                ..nd.clone()
            })
            .collect();
        let (right_role, inside_role, across_role) = group_checks(&tr.z, &by_role);
        println!(
            "roles: {right_role}/100 nearest, cosine same {inside_role:.3} different {across_role:.3}"
        );
        assert!(right >= 60, "{right}/100");
        assert!(tr.z.iter().flatten().all(|x| x.is_finite()));
    }

    #[test]
    fn pca_keeps_points_that_lie_on_a_plane() {
        // Points on a tilted plane in 4-D: the projection must keep their distances.
        let a = [0.5, 0.5, 0.5, 0.5];
        let b = [0.5, -0.5, 0.5, -0.5];
        let flat: Vec<[f64; 2]> = vec![[0.0, 0.0], [3.0, 0.0], [0.0, 1.0], [2.0, 2.0], [-1.0, 0.5]];
        let z: Vec<Vec<f64>> = flat
            .iter()
            .map(|p| (0..4).map(|i| 1.0 + p[0] * a[i] + p[1] * b[i]).collect())
            .collect();
        let proj = Projection::of(&z);
        let pts: Vec<[f64; 2]> = z.iter().map(|v| proj.point(v)).collect();
        let dist =
            |p: [f64; 2], q: [f64; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
        for i in 0..flat.len() {
            for j in 0..flat.len() {
                assert!((dist(pts[i], pts[j]) - dist(flat[i], flat[j])).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn joint_projection_keeps_dot_products_of_vectors_on_a_plane() {
        // Z and C in a tilted plane through the origin of 5-D space: the plane keeps 100 % and
        // every z · c on screen equals the real one.
        let a = [0.6, 0.0, 0.8, 0.0, 0.0];
        let b = [0.0, 0.6, 0.0, 0.0, 0.8];
        let at = |p: [f64; 2]| -> Vec<f64> { (0..5).map(|i| p[0] * a[i] + p[1] * b[i]).collect() };
        let z: Vec<Vec<f64>> = [[1.0, 0.2], [-0.5, 0.7], [0.3, -1.1]]
            .into_iter()
            .map(at)
            .collect();
        let c: Vec<Vec<f64>> = [[0.4, 0.4], [2.0, -0.3]].into_iter().map(at).collect();
        let proj = Projection::joint(&z, &c, None);
        assert!((proj.kept.unwrap() - 1.0).abs() < 1e-9);
        for zi in &z {
            for cj in &c {
                let (p, q) = (proj.point(zi), proj.point(cj));
                assert!((p[0] * q[0] + p[1] * q[1] - dot(zi, cj)).abs() < 1e-9);
            }
        }
        // With last frame's axes, the axes come back in the same order and with the same signs.
        let axes = proj.axes.clone().unwrap().0;
        let flipped = [
            axes[1].iter().map(|x| -x).collect::<Vec<_>>(),
            axes[0].clone(),
        ];
        let again = Projection::joint(&z, &c, Some(&flipped));
        let new_axes = again.axes.unwrap().0;
        assert!(
            dot(&new_axes[0], &flipped[0]) > 0.99,
            "old {axes:?}\nprevious {flipped:?}\nnew {new_axes:?}"
        );
        assert!(dot(&new_axes[1], &flipped[1]) > 0.99);
    }

    #[test]
    fn share_kept_by_the_plane_after_training() {
        // Not a rule, a reading: how faithful the "c above dim 2" picture is on two graphs.
        for (preset, dim) in [(Preset::TwoGroups, 16), (Preset::Swarm, 16)] {
            let settings = Settings {
                dim,
                ..Settings::default()
            };
            let (_, tr) = trained(preset, settings);
            let kept = Projection::joint(&tr.z, &tr.c, None).kept.unwrap();
            println!(
                "{} at dim {dim}: plane keeps {:.0} %",
                preset.name(),
                100.0 * kept
            );
            assert!(kept > 2.0 / dim as f64 && kept <= 1.0);
        }
    }

    #[test]
    fn joint_projection_says_how_much_it_drops() {
        // Four perpendicular directions of equal length: any plane keeps exactly half.
        let z: Vec<Vec<f64>> = (0..4)
            .map(|k| (0..4).map(|i| if i == k { 1.0 } else { 0.0 }).collect())
            .collect();
        let c = vec![vec![0.0; 4]; 4];
        let kept = Projection::joint(&z, &c, None).kept.unwrap();
        assert!((kept - 0.5).abs() < 1e-9, "{kept}");
    }

    #[test]
    fn helpers_format_vectors_and_dot_products() {
        assert_eq!(vec_str(&[0.5, -0.25]), "[+0.500, -0.250]");
        assert!(vec_str(&[0.0; 8]).ends_with("… (8 numbers)]"));
        assert_eq!(
            expand(&[1.0, 2.0], &[0.5, -1.0]),
            "(+1.000)(+0.500) + (+2.000)(-1.000) = -1.5000"
        );
        assert_eq!(window_of(10, 1, 3), (0, 4));
        assert_eq!(window_of(10, 8, 3), (5, 9));
    }

    #[test]
    fn nice_tick_is_round_and_below_the_view() {
        for r in [0.013, 0.2, 1.0, 7.5, 40.0] {
            let t = nice_tick(r);
            assert!(t <= r * 0.6 + 1e-6 && t > r * 0.1, "r {r}: tick {t}");
        }
        assert_eq!(nice_tick(1.0), 0.5);
    }
}
