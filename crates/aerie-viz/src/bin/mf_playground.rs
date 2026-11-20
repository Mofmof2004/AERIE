//! Matrix factorisation playground (Mission 2): watch Z Zᵀ learn to rebuild A.
//!
//! Run with `cargo run --release -p aerie-viz --bin mf_playground`.
//!
//! - Left: the graph (presets, or build your own: add drones, link them, paint labels,
//!   delete, and ✂ hide links for the link test), how link weights become A, the settings
//!   (dim, learning rate, l2, seed, epochs) and the training buttons. With "Explain each
//!   epoch in 7 small steps" on, Space follows one drone u through one epoch: its row of A,
//!   its dot products, its errors, each other drone's pull or push, the gradient, the step.
//! - Centre left: the graph, lit up small step by small step (u's links with their A
//!   values, then a line to every drone coloured by its error: green = pulled closer,
//!   red = pushed away). Right-click a drone to follow it.
//! - Centre right: the vectors (dim 2: themselves; above: their two main directions), the
//!   origin, and u's pull / push arrows and step (magnified).
//! - Right: *Math* (each small step's formula with this epoch's numbers), *Matrices* (A,
//!   Z Zᵀ and the errors E as heatmaps, drones sorted by label), *Hidden links* (their
//!   z_u · z_v against the pairs that never talked, and the AUC).
//! - Bottom: the loss curve.
//!
//! The trainer runs exactly `aerie_classic::embed::mf::matrix_factorization` (same start,
//! `mf_gradient`, `mf_loss`), epoch by epoch; when it reaches the last epoch it re-runs
//! `matrix_factorization` and shows ✔ if both give the same Z, number for number.

use std::cell::Cell;
use std::collections::HashSet;

use aerie_classic::embed::mf::{MfConfig, matrix_factorization, mf_gradient, mf_loss};
use aerie_classic::embed::score::auc;
use aerie_classic::embed::skipgram::dot;
use aerie_graph::{Edge, EdgeList};
use aerie_viz::style::{draw_drone, role_name, squad_color};
use eframe::egui::{
    self, Align2, Color32, FontId, Key, Pos2, Rect, RichText, Sense, Shape, Stroke, pos2, vec2,
};
use faer::Mat;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::drone::Role;
use swarmsim::snapshot::Snapshot;
use swarmsim::traffic::TrafficConfig;

/// Ring around the drone being followed (u).
const FOCUS_RING: Color32 = Color32::from_rgb(255, 205, 60);
/// e < 0: z_u · z_v too small, the step pulls z_u toward z_v.
const PULL_COLOR: Color32 = Color32::from_rgb(70, 200, 120);
/// e > 0: z_u · z_v too big, the step pushes z_u away from z_v.
const PUSH_COLOR: Color32 = Color32::from_rgb(235, 90, 80);
/// Hidden links (left out of A, kept for the link test).
const HIDDEN_COLOR: Color32 = Color32::from_rgb(190, 120, 255);
/// Drone marker radius, pixels.
const NODE_PX: f32 = 9.0;
/// The graph canvas always shows at least ±BOX.
const BOX: f32 = 3.0;
/// Rows per small step in the Math tab (largest errors first).
const MATH_ROWS: usize = 10;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE matrix factorisation playground")
            .with_inner_size([1700.0, 1000.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE matrix factorisation playground",
        options,
        Box::new(|_cc| Ok(Box::new(Playground::default()))),
    )
}

// ------------------------------------------------------------------------------- graphs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Preset {
    TwoGroups,
    Barbell,
    WheelsByGroup,
    WheelsByRole,
    Ring,
    Grid,
    SwarmBySquad,
    SwarmByRole,
    Empty,
}

impl Preset {
    const ALL: [Preset; 9] = [
        Preset::TwoGroups,
        Preset::Barbell,
        Preset::WheelsByGroup,
        Preset::WheelsByRole,
        Preset::Ring,
        Preset::Grid,
        Preset::SwarmBySquad,
        Preset::SwarmByRole,
        Preset::Empty,
    ];

    fn name(self) -> &'static str {
        match self {
            Preset::TwoGroups => "Two groups + bridge",
            Preset::Barbell => "Barbell (two cliques, a path)",
            Preset::WheelsByGroup => "Three wheels, labels = group",
            Preset::WheelsByRole => "Three wheels, labels = role",
            Preset::Ring => "Ring of 16",
            Preset::Grid => "Grid 6 × 6",
            Preset::SwarmBySquad => "Swarm traffic, labels = squad",
            Preset::SwarmByRole => "Swarm traffic, labels = role",
            Preset::Empty => "Empty: build your own",
        }
    }

    fn about(self) -> &'static str {
        match self {
            Preset::TwoGroups => {
                "Two 6-drone cliques joined by one link. With dim 2, each clique should get its \
                 own direction; the bridge is the one entry the vectors can't fit."
            }
            Preset::Barbell => {
                "Two 5-cliques joined by a path of 3. The path drones have few links: watch \
                 how short their vectors stay."
            }
            Preset::WheelsByGroup | Preset::WheelsByRole => {
                "Three wheels (a hub linked to a rim of 6), hubs joined through bridge drones. \
                 Same graph, two labelings: by group or by role. Hubs have many links: their \
                 vectors come out long."
            }
            Preset::Ring => {
                "16 drones in a circle. A has a band along the diagonal; at dim 2 the vectors \
                 often come out as a ring too."
            }
            Preset::Grid => {
                "A 6 × 6 lattice, labelled by quarter. Neighbours share links, so nearby \
                 drones end up with similar vectors."
            }
            Preset::SwarmBySquad | Preset::SwarmByRole => {
                "Mission 2's traffic graph of a 100-drone swarm, A = ln(1 + messages). Same \
                 graph, labelled by squad or by role."
            }
            Preset::Empty => {
                "Nothing yet: pick the Add tool and click to place drones, then Link two at \
                 a time, and Label to paint them."
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Node {
    pos: [f32; 2],
    label: usize,
    /// Swarm presets only: the drone's role (its marker shape).
    role: Option<Role>,
}

#[derive(Debug, Clone)]
struct Graph {
    nodes: Vec<Node>,
    edges: EdgeList,
    /// Names of the labels (missing ones are A, B, C…).
    label_names: Vec<String>,
}

impl Graph {
    fn empty() -> Graph {
        Graph {
            nodes: Vec::new(),
            edges: EdgeList {
                n: 0,
                edges: Vec::new(),
            },
            label_names: Vec::new(),
        }
    }

    fn n(&self) -> usize {
        self.nodes.len()
    }

    fn add(&mut self, pos: [f32; 2], label: usize) -> usize {
        self.nodes.push(Node {
            pos,
            label,
            role: None,
        });
        self.edges.n = self.nodes.len() as u32;
        self.nodes.len() - 1
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

    /// Deletes drone `i` and its links; drones after it move down by one.
    fn remove(&mut self, i: usize) {
        self.nodes.remove(i);
        let i = i as u32;
        self.edges.edges.retain(|e| e.src != i && e.dst != i);
        for e in &mut self.edges.edges {
            if e.src > i {
                e.src -= 1;
            }
            if e.dst > i {
                e.dst -= 1;
            }
        }
        self.edges.n = self.nodes.len() as u32;
    }

    fn label_name(&self, k: usize) -> String {
        self.label_names
            .get(k)
            .cloned()
            .unwrap_or_else(|| ((b'A' + (k % 26) as u8) as char).to_string())
    }

    /// Scales and centres the drawing to fit in ±2.6, keeping its shape.
    fn fit_box(&mut self) {
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for nd in &self.nodes {
            for k in 0..2 {
                lo[k] = lo[k].min(nd.pos[k]);
                hi[k] = hi[k].max(nd.pos[k]);
            }
        }
        let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-3);
        let mid = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
        for nd in &mut self.nodes {
            for (x, m) in nd.pos.iter_mut().zip(mid) {
                *x = (*x - m) * 5.2 / span;
            }
        }
    }

    /// `k` drones on a circle, all linked together. Returns their ids.
    fn clique(&mut self, k: usize, centre: [f32; 2], radius: f32, label: usize) -> Vec<usize> {
        let ids: Vec<usize> = (0..k)
            .map(|i| {
                let a = std::f32::consts::TAU * i as f32 / k as f32;
                self.add(
                    [centre[0] + radius * a.cos(), centre[1] + radius * a.sin()],
                    label,
                )
            })
            .collect();
        for (x, &a) in ids.iter().enumerate() {
            for &b in &ids[x + 1..] {
                self.link(a, b, 1.0);
            }
        }
        ids
    }

    fn build(preset: Preset) -> Graph {
        let mut g = Graph::empty();
        match preset {
            Preset::TwoGroups => {
                let a = g.clique(6, [-1.6, 0.0], 1.0, 0);
                let b = g.clique(6, [1.6, 0.0], 1.0, 1);
                g.link(a[0], b[3], 1.0);
            }
            Preset::Barbell => {
                let a = g.clique(5, [-2.4, 0.0], 0.9, 0);
                let b = g.clique(5, [2.4, 0.0], 0.9, 1);
                let path: Vec<usize> = (0..3)
                    .map(|i| g.add([-0.8 + 0.8 * i as f32, 0.0], 2))
                    .collect();
                g.link(a[0], path[0], 1.0);
                g.link(path[0], path[1], 1.0);
                g.link(path[1], path[2], 1.0);
                g.link(path[2], b[3], 1.0);
                g.label_names = vec!["left".into(), "right".into(), "path".into()];
            }
            Preset::WheelsByGroup | Preset::WheelsByRole => {
                let by_role = preset == Preset::WheelsByRole;
                let mut hubs = Vec::new();
                for w in 0..3 {
                    let a = std::f32::consts::TAU * w as f32 / 3.0 - std::f32::consts::FRAC_PI_2;
                    let centre = [2.0 * a.cos(), 2.0 * a.sin()];
                    let hub = g.add(centre, if by_role { 0 } else { w });
                    let rim: Vec<usize> = (0..6)
                        .map(|i| {
                            let b = std::f32::consts::TAU * i as f32 / 6.0;
                            g.add(
                                [centre[0] + 0.8 * b.cos(), centre[1] + 0.8 * b.sin()],
                                if by_role { 1 } else { w },
                            )
                        })
                        .collect();
                    for i in 0..6 {
                        g.link(hub, rim[i], 1.0);
                        g.link(rim[i], rim[(i + 1) % 6], 1.0);
                    }
                    hubs.push((hub, centre));
                }
                for w in 0..3 {
                    let (h1, c1) = hubs[w];
                    let (h2, c2) = hubs[(w + 1) % 3];
                    let mid = [(c1[0] + c2[0]) * 0.5, (c1[1] + c2[1]) * 0.5];
                    let bridge = g.add(mid, if by_role { 2 } else { 3 });
                    g.link(h1, bridge, 1.0);
                    g.link(bridge, h2, 1.0);
                }
                g.label_names = if by_role {
                    vec!["hub".into(), "rim".into(), "bridge".into()]
                } else {
                    vec![
                        "wheel A".into(),
                        "wheel B".into(),
                        "wheel C".into(),
                        "bridge".into(),
                    ]
                };
            }
            Preset::Ring => {
                for i in 0..16 {
                    let a = std::f32::consts::TAU * i as f32 / 16.0;
                    g.add([2.4 * a.cos(), 2.4 * a.sin()], i * 4 / 16);
                }
                for i in 0..16 {
                    g.link(i, (i + 1) % 16, 1.0);
                }
            }
            Preset::Grid => {
                for r in 0..6 {
                    for c in 0..6 {
                        g.add([c as f32, r as f32], (r / 3) * 2 + c / 3);
                    }
                }
                for r in 0..6 {
                    for c in 0..6 {
                        let i = r * 6 + c;
                        if c < 5 {
                            g.link(i, i + 1, 1.0);
                        }
                        if r < 5 {
                            g.link(i, i + 6, 1.0);
                        }
                    }
                }
            }
            Preset::SwarmBySquad | Preset::SwarmByRole => {
                let by_role = preset == Preset::SwarmByRole;
                let snap = Snapshot::build(0, &SimConfig::default());
                g.edges = snap.traffic(&TrafficConfig::default()).undirected();
                g.nodes = snap
                    .swarm
                    .iter()
                    .map(|d| Node {
                        pos: [d.position.x as f32, -d.position.y as f32],
                        label: if by_role {
                            d.role as usize
                        } else {
                            d.squad.0 as usize
                        },
                        role: Some(d.role),
                    })
                    .collect();
                g.label_names = if by_role {
                    Role::ALL
                        .iter()
                        .map(|&r| role_name(r).to_string())
                        .collect()
                } else {
                    (0..10).map(|s| format!("squad {s}")).collect()
                };
            }
            Preset::Empty => {}
        }
        if g.n() > 0 {
            g.fit_box();
        }
        g
    }
}

// ------------------------------------------------------------------------- A from the graph

/// How a link's weight becomes its entry of A.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Weights {
    One,
    Log,
    Raw,
}

impl Weights {
    const ALL: [Weights; 3] = [Weights::One, Weights::Log, Weights::Raw];

    fn name(self) -> &'static str {
        match self {
            Weights::One => "1 per link",
            Weights::Log => "ln(1 + weight)",
            Weights::Raw => "the weight as it is",
        }
    }

    fn apply(self, w: f64) -> f64 {
        match self {
            Weights::One => 1.0,
            Weights::Log => w.ln_1p(),
            Weights::Raw => w,
        }
    }
}

/// A (n × n): each link's weight at (i, j) and (j, i), hidden links left at 0.
fn build_a(graph: &Graph, hidden: &HashSet<(u32, u32)>, weights: Weights) -> Mat<f64> {
    let n = graph.n();
    let mut a = Mat::zeros(n, n);
    for e in &graph.edges.edges {
        if hidden.contains(&(e.src, e.dst)) {
            continue;
        }
        let w = weights.apply(e.weight);
        a[(e.src as usize, e.dst as usize)] = w;
        a[(e.dst as usize, e.src as usize)] = w;
    }
    a
}

// --------------------------------------------------------------------------------- trainer

/// Everything that decides Z: change one and training starts over.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Settings {
    config: MfConfig,
    seed: u64,
    weights: Weights,
}

impl Settings {
    /// Each preset's starting point: weights 1 and lr 0.01 on the small graphs; on the
    /// swarm, ln(1 + messages) and a smaller step (bigger A, more drones in each sum).
    fn for_preset(preset: Preset) -> Settings {
        let swarm = matches!(preset, Preset::SwarmBySquad | Preset::SwarmByRole);
        Settings {
            config: MfConfig {
                dim: 2,
                epochs: 300,
                lr: if swarm { 0.002 } else { 0.01 },
                l2: 0.01,
            },
            seed: 1,
            weights: if swarm { Weights::Log } else { Weights::One },
        }
    }
}

/// `matrix_factorization`, one epoch at a time.
struct Trainer {
    a: Mat<f64>,
    config: MfConfig,
    seed: u64,
    z: Vec<Vec<f64>>,
    epoch: usize,
    /// The loss before any step, then after each epoch.
    losses: Vec<f64>,
    /// The loss blew up (the learning rate is too big for this A).
    diverged: bool,
    /// At the last epoch: does `matrix_factorization` give the same Z?
    same: Option<bool>,
}

impl Trainer {
    /// Z's start exactly as `matrix_factorization` draws it (seed, stream 2, ±0.5/dim).
    fn new(a: Mat<f64>, config: MfConfig, seed: u64) -> Trainer {
        let n = a.nrows();
        let half = 0.5 / config.dim as f64;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(2);
        let z: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                (0..config.dim)
                    .map(|_| rng.random_range(-half..half))
                    .collect()
            })
            .collect();
        let loss = if n > 0 {
            mf_loss(&a, &z, config.l2)
        } else {
            0.0
        };
        Trainer {
            a,
            config,
            seed,
            z,
            epoch: 0,
            losses: vec![loss],
            diverged: false,
            same: None,
        }
    }

    fn n(&self) -> usize {
        self.z.len()
    }

    fn finished(&self) -> bool {
        self.epoch >= self.config.epochs || self.diverged || self.n() == 0
    }

    /// One full-batch step, exactly as `matrix_factorization` takes it.
    fn step(&mut self) {
        if self.finished() {
            return;
        }
        let grad = mf_gradient(&self.a, &self.z, self.config.l2);
        for (zu, gu) in self.z.iter_mut().zip(&grad) {
            for (x, g) in zu.iter_mut().zip(gu) {
                *x -= self.config.lr * g;
            }
        }
        self.epoch += 1;
        let loss = mf_loss(&self.a, &self.z, self.config.l2);
        if !loss.is_finite() || loss > 1e12 {
            self.diverged = true;
        }
        self.losses.push(loss);
        if self.epoch == self.config.epochs && !self.diverged {
            self.same = Some(matrix_factorization(&self.a, &self.config, self.seed) == self.z);
        }
    }

    fn run(&mut self, k: usize) {
        for _ in 0..k {
            if self.finished() {
                break;
            }
            self.step();
        }
    }
}

// ------------------------------------------------------------------------- explain mode

/// The 7 small steps of one epoch, seen from one drone u.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    /// The drone followed and its vector.
    Focus,
    /// Its row of A: what the vectors should rebuild.
    Row,
    /// z_u · z_v for every other drone: what the vectors say now.
    Dots,
    /// e_uv = z_u · z_v − A_uv.
    Errors,
    /// 4 e_uv z_v: each other drone's pull or push.
    Terms,
    /// Their sum plus 2 l2 z_u: the gradient.
    Gradient,
    /// z ← z − lr · gradient, for every drone at once.
    Step,
}

impl Phase {
    const ALL: [Phase; 7] = [
        Phase::Focus,
        Phase::Row,
        Phase::Dots,
        Phase::Errors,
        Phase::Terms,
        Phase::Gradient,
        Phase::Step,
    ];

    fn number(self) -> usize {
        self as usize + 1
    }

    fn title(self) -> &'static str {
        match self {
            Phase::Focus => "Follow one drone",
            Phase::Row => "What A says about u",
            Phase::Dots => "What the vectors say now",
            Phase::Errors => "The errors",
            Phase::Terms => "Each drone pulls or pushes",
            Phase::Gradient => "Add it all up: the gradient",
            Phase::Step => "Every drone moves at once",
        }
    }

    fn next(self) -> Option<Phase> {
        Phase::ALL.get(self as usize + 1).copied()
    }
}

/// One epoch seen from drone u, with every number of its gradient.
#[derive(Debug, Clone)]
struct Explained {
    phase: Phase,
    u: usize,
    z_u: Vec<f64>,
    /// Every drone's z at the start of the epoch.
    z_all: Vec<Vec<f64>>,
    /// Row u of A.
    a_row: Vec<f64>,
    /// z_u · z_v for every v.
    dots: Vec<f64>,
    /// e_uv (0 at v = u, which is left out).
    errors: Vec<f64>,
    /// 4 e_uv z_v for every v.
    terms: Vec<Vec<f64>>,
    /// 2 l2 z_u.
    penalty: Vec<f64>,
    /// ∂L/∂z_u, summed in the same order as `mf_gradient` (so it is the same number).
    grad: Vec<f64>,
    lr: f64,
    loss_before: f64,
    /// From `Phase::Step` on: z_u after the step, and the new loss.
    z_after: Option<Vec<f64>>,
    loss_after: Option<f64>,
}

impl Explained {
    fn new(t: &Trainer, u: usize, phase: Phase) -> Explained {
        let n = t.n();
        let z = &t.z;
        let a_row: Vec<f64> = (0..n).map(|v| t.a[(u, v)]).collect();
        let dots: Vec<f64> = (0..n).map(|v| dot(&z[u], &z[v])).collect();
        let errors: Vec<f64> = (0..n)
            .map(|v| if v == u { 0.0 } else { dots[v] - a_row[v] })
            .collect();
        let terms: Vec<Vec<f64>> = (0..n)
            .map(|v| z[v].iter().map(|x| 4.0 * errors[v] * x).collect())
            .collect();
        let penalty: Vec<f64> = z[u].iter().map(|x| 2.0 * t.config.l2 * x).collect();
        let mut grad = vec![0.0; z[u].len()];
        for (v, term) in terms.iter().enumerate() {
            if v != u {
                for (g, x) in grad.iter_mut().zip(term) {
                    *g += x;
                }
            }
        }
        for (g, x) in grad.iter_mut().zip(&penalty) {
            *g += x;
        }
        Explained {
            phase,
            u,
            z_u: z[u].clone(),
            z_all: z.to_vec(),
            a_row,
            dots,
            errors,
            terms,
            penalty,
            grad,
            lr: t.config.lr,
            loss_before: *t.losses.last().unwrap(),
            z_after: None,
            loss_after: None,
        }
    }

    /// The other drones, largest error first.
    fn order(&self) -> Vec<usize> {
        let mut others: Vec<usize> = (0..self.dots.len()).filter(|&v| v != self.u).collect();
        others.sort_by(|&a, &b| self.errors[b].abs().total_cmp(&self.errors[a].abs()));
        others
    }
}

// ----------------------------------------------------------------------------- projection

/// The vectors on a plane: themselves at dim 2, their two main directions (PCA) above.
struct Projection {
    axes: Option<([Vec<f64>; 2], Vec<f64>)>,
}

impl Projection {
    fn of(z: &[Vec<f64>]) -> Projection {
        let dim = z.first().map_or(0, |r| r.len());
        if dim <= 2 || z.len() < 2 {
            return Projection { axes: None };
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
        let first = top_direction(&cov, None);
        let second = top_direction(&cov, Some(&first));
        Projection {
            axes: Some(([first, second], mean)),
        }
    }

    fn point(&self, v: &[f64]) -> [f64; 2] {
        match &self.axes {
            None => [
                v.first().copied().unwrap_or(0.0),
                v.get(1).copied().unwrap_or(0.0),
            ],
            Some((axes, mean)) => {
                let c: Vec<f64> = v.iter().zip(mean).map(|(a, m)| a - m).collect();
                [dot(&c, &axes[0]), dot(&c, &axes[1])]
            }
        }
    }
}

// ------------------------------------------------------------------------------------ app

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Link,
    Add,
    Label,
    Delete,
    Hide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Math,
    Matrices,
    Hidden,
}

struct Playground {
    preset: Preset,
    graph: Graph,
    /// Bumped when A changes (links, drones, hidden links): training starts over.
    version: u64,
    tool: Tool,
    brush: usize,
    link_from: Option<usize>,
    dragging: Option<usize>,
    hover: Option<usize>,
    /// Links left out of A (stored as (src, dst), src < dst).
    hidden: HashSet<(u32, u32)>,
    settings: Settings,
    trainer: Trainer,
    trained_for: (u64, Settings),
    explain: bool,
    step: Option<Explained>,
    focus: usize,
    scroll_math: Cell<bool>,
    playing: bool,
    last_tick: f64,
    speed: f64,
    tab: Tab,
    message: Option<String>,
}

impl Default for Playground {
    fn default() -> Self {
        let preset = Preset::TwoGroups;
        let graph = Graph::build(preset);
        let settings = Settings::for_preset(preset);
        let hidden = HashSet::new();
        let trainer = Trainer::new(
            build_a(&graph, &hidden, settings.weights),
            settings.config,
            settings.seed,
        );
        Playground {
            preset,
            graph,
            version: 0,
            tool: Tool::Link,
            brush: 0,
            link_from: None,
            dragging: None,
            hover: None,
            hidden,
            settings,
            trainer,
            trained_for: (0, settings),
            explain: true,
            step: None,
            focus: 0,
            scroll_math: Cell::new(false),
            playing: false,
            last_tick: 0.0,
            speed: 10.0,
            tab: Tab::Math,
            message: None,
        }
    }
}

impl eframe::App for Playground {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.ensure_trainer();
        if ui.input(|i| i.key_pressed(Key::Space)) {
            self.next_input();
        }
        if self.playing {
            let now = ui.input(|i| i.time);
            if now - self.last_tick >= 1.0 / self.speed {
                self.last_tick = now;
                if self.trainer.finished() && self.step_done() {
                    self.playing = false;
                } else {
                    self.next_input();
                }
            }
            ui.ctx().request_repaint();
        }

        egui::Panel::left("controls")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });
        egui::Panel::right("details")
            .resizable(true)
            .default_size(470.0)
            .show(ui, |ui| self.right(ui));
        egui::Panel::bottom("loss")
            .resizable(true)
            .default_size(170.0)
            .show(ui, |ui| self.loss_plot(ui));
        egui::CentralPanel::default().show(ui, |ui| {
            ui.columns(2, |cols| {
                self.graph_canvas(&mut cols[0]);
                self.embed_canvas(&mut cols[1]);
            });
        });
    }
}

impl Playground {
    /// A new trainer if A or a setting changed.
    fn ensure_trainer(&mut self) {
        let key = (self.version, self.settings);
        if self.trained_for != key {
            self.trainer = Trainer::new(
                build_a(&self.graph, &self.hidden, self.settings.weights),
                self.settings.config,
                self.settings.seed,
            );
            self.trained_for = key;
            self.step = None;
        }
        self.focus = self.focus.min(self.graph.n().saturating_sub(1));
    }

    /// No small step under way (none, or the last one has moved).
    fn step_done(&self) -> bool {
        self.step.as_ref().is_none_or(|s| s.phase == Phase::Step)
    }

    /// Space: the next small step (explain mode) or a whole epoch.
    fn next_input(&mut self) {
        if self.explain {
            self.advance();
        } else {
            self.trainer.step();
            self.step = None;
        }
        self.scroll_math.set(true);
    }

    fn advance(&mut self) {
        if self.graph.n() == 0 {
            return;
        }
        match self.step.as_ref().map(|s| s.phase.next()) {
            None | Some(None) => {
                self.step = (!self.trainer.finished())
                    .then(|| Explained::new(&self.trainer, self.focus, Phase::Focus));
            }
            Some(Some(phase)) => {
                let s = self.step.as_mut().unwrap();
                s.phase = phase;
                if phase == Phase::Step {
                    self.trainer.step();
                    s.z_after = Some(self.trainer.z[s.u].clone());
                    s.loss_after = self.trainer.losses.last().copied();
                }
            }
        }
    }

    /// The rest of the epoch under way, or a whole new one.
    fn whole_epoch(&mut self) {
        if self.step_done() {
            self.trainer.step();
            self.step = None;
        } else {
            while !self.step_done() {
                self.advance();
            }
        }
        self.scroll_math.set(true);
    }

    fn set_focus(&mut self, u: usize) {
        self.focus = u;
        if let Some(s) = &self.step
            && s.phase < Phase::Step
        {
            self.step = Some(Explained::new(&self.trainer, u, s.phase));
        } else {
            self.step = None;
        }
    }

    fn load(&mut self, preset: Preset) {
        self.preset = preset;
        self.graph = Graph::build(preset);
        self.settings = Settings::for_preset(preset);
        self.hidden.clear();
        self.brush = 0;
        self.focus = 0;
        self.changed();
    }

    /// A changed: training starts over.
    fn changed(&mut self) {
        self.version += 1;
        self.link_from = None;
        self.step = None;
    }

    fn label_color(&self, k: usize) -> Color32 {
        squad_color(k)
    }

    fn labels(&self) -> Vec<usize> {
        self.graph.nodes.iter().map(|d| d.label).collect()
    }

    // ---------------------------------------------------------------------- left panel

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Matrix factorisation playground");
        ui.weak("Find one vector per drone so that z_u · z_v ≈ A_uv for every pair.");
        ui.separator();

        ui.label(RichText::new("Graph").strong());
        let mut chosen = self.preset;
        egui::ComboBox::from_id_salt("preset")
            .selected_text(self.preset.name())
            .width(260.0)
            .show_ui(ui, |ui| {
                for p in Preset::ALL {
                    ui.selectable_value(&mut chosen, p, p.name());
                }
            });
        if chosen != self.preset {
            self.load(chosen);
        }
        ui.weak(self.preset.about());
        if ui.small_button("Reload preset").clicked() {
            self.load(self.preset);
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Tool:");
            ui.selectable_value(&mut self.tool, Tool::Link, "🔗 Link");
            ui.selectable_value(&mut self.tool, Tool::Add, "➕ Add");
            ui.selectable_value(&mut self.tool, Tool::Label, "🖌 Label");
            ui.selectable_value(&mut self.tool, Tool::Delete, "✖ Delete");
            ui.selectable_value(&mut self.tool, Tool::Hide, "✂ Hide");
        });
        ui.weak(match self.tool {
            Tool::Link => "Click a drone, then another: adds the link, or removes it.",
            Tool::Add => "Click empty space: a new drone with the brush label.",
            Tool::Label => "Click a drone: it takes the brush label.",
            Tool::Delete => "Click a drone: deletes it and its links.",
            Tool::Hide => {
                "Click two linked drones: the link is hidden from A (A_uv = 0 in training) \
                 but kept for the link test (Hidden links tab). Again to bring it back."
            }
        });
        ui.weak("Any tool: drag to move a drone, right-click to follow it in the Math tab.");
        ui.horizontal_wrapped(|ui| {
            ui.label("Brush:");
            let used = self
                .graph
                .nodes
                .iter()
                .map(|d| d.label + 1)
                .max()
                .unwrap_or(0);
            for k in 0..(used + 1).clamp(4, 12) {
                let text = RichText::new(format!(" {} ", self.graph.label_name(k)))
                    .background_color(self.label_color(k))
                    .color(Color32::BLACK);
                if ui.selectable_label(self.brush == k, text).clicked() {
                    self.brush = k;
                }
            }
        });
        ui.label(format!(
            "{} drones, {} links, {} hidden",
            self.graph.n(),
            self.graph.edges.edges.len(),
            self.hidden.len()
        ));
        if let Some(m) = &self.message {
            ui.colored_label(PUSH_COLOR, m);
        }
        ui.separator();

        ui.label(RichText::new("What A holds").strong());
        egui::ComboBox::from_id_salt("weights")
            .selected_text(self.settings.weights.name())
            .width(200.0)
            .show_ui(ui, |ui| {
                for w in Weights::ALL {
                    ui.selectable_value(&mut self.settings.weights, w, w.name());
                }
            });
        ui.weak(
            "A_uv for each link (0 for no link, and for hidden links). On the swarm the \
             weight is the number of messages: the log keeps one busy relay link from \
             swamping the rest.",
        );
        ui.separator();

        ui.label(RichText::new("Settings").strong());
        let c = &mut self.settings.config;
        egui::Grid::new("settings").num_columns(2).show(ui, |ui| {
            ui.label("dimension");
            ui.add(egui::DragValue::new(&mut c.dim).range(1..=32));
            ui.end_row();
            ui.label("learning rate");
            ui.add(
                egui::Slider::new(&mut c.lr, 0.0001..=0.2)
                    .logarithmic(true)
                    .max_decimals(4),
            );
            ui.end_row();
            ui.label("l2");
            ui.add(
                egui::DragValue::new(&mut c.l2)
                    .range(0.0..=1.0)
                    .speed(0.001),
            );
            ui.end_row();
            ui.label("epochs");
            ui.add(egui::DragValue::new(&mut c.epochs).range(1..=5000));
            ui.end_row();
            ui.label("seed");
            ui.add(egui::DragValue::new(&mut self.settings.seed));
            ui.end_row();
        });
        ui.weak("Any change here starts training over (same start as matrix_factorization).");
        ui.separator();

        ui.label(RichText::new("Training").strong());
        ui.checkbox(&mut self.explain, "Explain each epoch in 7 small steps");
        let t = &self.trainer;
        ui.label(format!(
            "Epoch {} / {} · loss {}",
            t.epoch,
            t.config.epochs,
            num(*t.losses.last().unwrap_or(&0.0))
        ));
        if t.diverged {
            ui.colored_label(
                PUSH_COLOR,
                "The loss blew up: the learning rate is too big for this A. Each step \
                 overshoots, and the next overshoots more. Lower it.",
            );
        }
        if let Some(same) = t.same {
            if same {
                ui.colored_label(
                    PULL_COLOR,
                    "✔ Same Z as matrix_factorization, number for number.",
                );
            } else {
                ui.colored_label(PUSH_COLOR, "✘ Not the same Z as matrix_factorization.");
            }
        }
        if let Some(s) = &self.step
            && s.phase < Phase::Step
        {
            ui.label(
                RichText::new(format!(
                    "Small step {}/7: {}",
                    s.phase.number(),
                    s.phase.title()
                ))
                .color(FOCUS_RING),
            );
        }
        let finished = self.trainer.finished() && self.step_done();
        ui.horizontal_wrapped(|ui| {
            let label = if self.explain {
                "Next small step (Space)"
            } else {
                "Next epoch (Space)"
            };
            if ui
                .add_enabled(!finished, egui::Button::new(label))
                .clicked()
            {
                self.next_input();
            }
            if self.explain
                && ui
                    .add_enabled(!finished, egui::Button::new("Whole epoch"))
                    .clicked()
            {
                self.whole_epoch();
            }
            for k in [10, 100] {
                if ui
                    .add_enabled(!finished, egui::Button::new(format!("+{k} epochs")))
                    .clicked()
                {
                    self.trainer.run(k);
                    self.step = None;
                }
            }
            if ui
                .add_enabled(!finished, egui::Button::new("Train to the end"))
                .clicked()
            {
                self.trainer.run(usize::MAX);
                self.step = None;
            }
            let play = if self.playing {
                "⏸ Pause"
            } else {
                "▶ Play"
            };
            if ui.button(play).clicked() {
                self.playing = !self.playing;
            }
            if ui.button("Reset").clicked() {
                self.trained_for.0 = u64::MAX; // forces a new trainer
                self.playing = false;
            }
        });
        let unit = if self.explain {
            "small steps / s"
        } else {
            "epochs / s"
        };
        ui.add(
            egui::Slider::new(&mut self.speed, 0.5..=60.0)
                .logarithmic(true)
                .text(unit),
        );
    }

    // -------------------------------------------------------------------- graph canvas

    fn graph_canvas(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Graph").strong());
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;
        let visuals = ui.visuals().clone();
        let text = visuals.text_color();
        painter.rect_filled(rect, 2.0, visuals.extreme_bg_color);
        let fit = Fit::new(&self.graph.nodes, rect.shrink(30.0));

        // Input.
        let near = |p: Pos2, nodes: &[Node]| {
            (0..nodes.len())
                .map(|i| (i, fit.screen(nodes[i].pos).distance(p)))
                .filter(|&(_, d)| d < NODE_PX + 6.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
        };
        self.hover = response
            .hover_pos()
            .and_then(|p| near(p, &self.graph.nodes));
        if response.drag_started() {
            let origin = ui
                .input(|i| i.pointer.press_origin())
                .or(response.interact_pointer_pos());
            self.dragging = origin.and_then(|p| near(p, &self.graph.nodes));
        }
        if let (Some(i), Some(p)) = (self.dragging, response.interact_pointer_pos())
            && response.dragged()
            && i < self.graph.n()
        {
            self.graph.nodes[i].pos = fit.world(p);
        }
        if response.drag_stopped() {
            self.dragging = None;
        }
        if response.secondary_clicked()
            && let Some(i) = response
                .interact_pointer_pos()
                .and_then(|p| near(p, &self.graph.nodes))
        {
            self.set_focus(i);
        }
        if response.clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            self.message = None;
            let hit = near(p, &self.graph.nodes);
            match (self.tool, hit) {
                (Tool::Link | Tool::Hide, Some(b)) => match self.link_from {
                    Some(a) if a != b => {
                        let key = (a.min(b) as u32, a.max(b) as u32);
                        let linked = self.graph.edges.edges.iter().any(|e| (e.src, e.dst) == key);
                        if self.tool == Tool::Link {
                            self.graph.toggle(a, b);
                            self.hidden.remove(&key);
                        } else if linked {
                            if !self.hidden.remove(&key) {
                                self.hidden.insert(key);
                            }
                        } else {
                            self.message = Some(format!("{a} and {b} have no link to hide."));
                        }
                        self.changed();
                    }
                    Some(_) => self.link_from = None,
                    None => self.link_from = Some(b),
                },
                (Tool::Add, None) => {
                    self.graph.add(fit.world(p), self.brush);
                    self.changed();
                }
                (Tool::Label, Some(i)) => self.graph.nodes[i].label = self.brush,
                (Tool::Delete, Some(i)) => {
                    self.graph.remove(i);
                    self.hidden.clear(); // drone numbers moved
                    self.focus = self.focus.min(self.graph.n().saturating_sub(1));
                    self.changed();
                }
                _ => self.link_from = None,
            }
        }
        self.ensure_trainer();
        let n = self.graph.n();
        if n == 0 {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Empty graph: pick ➕ Add and click here to place drones.",
                FontId::proportional(15.0),
                text,
            );
            return;
        }
        let fit = Fit::new(&self.graph.nodes, rect.shrink(30.0));
        let at = |i: usize| fit.screen(self.graph.nodes[i].pos);
        let r = if n > 50 { NODE_PX * 0.7 } else { NODE_PX };

        // Links (hidden ones dashed, with their score once training has started).
        let max_w = self
            .graph
            .edges
            .edges
            .iter()
            .map(|e| e.weight)
            .fold(1e-12, f64::max);
        for e in &self.graph.edges.edges {
            let (a, b) = (at(e.src as usize), at(e.dst as usize));
            if self.hidden.contains(&(e.src, e.dst)) {
                painter.extend(Shape::dashed_line(
                    &[a, b],
                    Stroke::new(2.0, HIDDEN_COLOR),
                    6.0,
                    4.0,
                ));
                if self.trainer.epoch > 0 {
                    let s = dot(
                        &self.trainer.z[e.src as usize],
                        &self.trainer.z[e.dst as usize],
                    );
                    painter.text(
                        a + (b - a) * 0.5 + vec2(0.0, -10.0),
                        Align2::CENTER_CENTER,
                        format!("z·z = {}", num(s)),
                        FontId::proportional(11.0),
                        HIDDEN_COLOR,
                    );
                }
            } else {
                let w = 0.8 + 2.2 * (e.weight / max_w).sqrt() as f32;
                painter.line_segment([a, b], Stroke::new(w, text.gamma_multiply(0.22)));
            }
        }

        // The epoch being explained: u's row of A, then its errors with every drone.
        if let Some(s) = self
            .step
            .as_ref()
            .filter(|s| s.phase >= Phase::Row && s.u < n)
        {
            let from = at(s.u);
            if s.phase == Phase::Row {
                for v in (0..n).filter(|&v| s.a_row[v] != 0.0) {
                    painter.line_segment(
                        [from, at(v)],
                        Stroke::new(3.0, FOCUS_RING.gamma_multiply(0.8)),
                    );
                    painter.text(
                        at(v) + vec2(0.0, r + 14.0),
                        Align2::CENTER_CENTER,
                        format!("A = {}", num(s.a_row[v])),
                        FontId::proportional(11.5),
                        FOCUS_RING,
                    );
                }
            } else {
                let top = s.errors.iter().map(|e| e.abs()).fold(1e-12, f64::max);
                let order = s.order();
                for (k, &v) in order.iter().enumerate() {
                    let (color, width) = if s.phase == Phase::Dots {
                        (text.gamma_multiply(0.5), 1.0)
                    } else {
                        let c = if s.errors[v] > 0.0 {
                            PUSH_COLOR
                        } else {
                            PULL_COLOR
                        };
                        (c, 0.5 + 5.0 * (s.errors[v].abs() / top) as f32)
                    };
                    painter.line_segment(
                        [from, at(v)],
                        Stroke::new(width, color.gamma_multiply(0.75)),
                    );
                    if k < 6 || n <= 16 {
                        let label = if s.phase == Phase::Dots {
                            format!("z·z = {}", num(s.dots[v]))
                        } else {
                            format!("e = {}", num(s.errors[v]))
                        };
                        painter.text(
                            at(v) + vec2(0.0, r + 14.0),
                            Align2::CENTER_CENTER,
                            label,
                            FontId::proportional(11.5),
                            color,
                        );
                    }
                }
            }
        }

        // Drones.
        for i in 0..n {
            let node = &self.graph.nodes[i];
            draw_drone(
                &painter,
                at(i),
                r,
                node.role.unwrap_or(Role::Recon),
                self.label_color(node.label),
                Stroke::new(1.0, Color32::from_black_alpha(160)),
            );
            if n <= 50 {
                painter.text(
                    at(i) + vec2(r + 2.0, -r),
                    Align2::LEFT_BOTTOM,
                    format!("{i}"),
                    FontId::proportional(11.0),
                    visuals.weak_text_color(),
                );
            }
        }
        painter.circle_stroke(at(self.focus), r + 6.0, Stroke::new(3.0, FOCUS_RING));
        if let Some(a) = self.link_from {
            painter.circle_stroke(at(a), r + 11.0, Stroke::new(1.5, text));
            painter.text(
                rect.left_bottom() + vec2(8.0, -8.0),
                Align2::LEFT_BOTTOM,
                format!("Drone {a} picked: click another drone."),
                FontId::proportional(12.0),
                text,
            );
        }
        if let Some(i) = self.hover {
            painter.circle_stroke(at(i), r + 2.0, Stroke::new(2.0, text));
            let z = &self.trainer.z[i];
            response.on_hover_text_at_pointer(format!(
                "Drone {i} · label {} · z = {}",
                self.graph.label_name(self.graph.nodes[i].label),
                vec_str(z)
            ));
        }
    }

    // ---------------------------------------------------------------- embedding canvas

    fn embed_canvas(&self, ui: &mut egui::Ui) {
        let dim = self.settings.config.dim;
        ui.label(
            RichText::new(if dim <= 2 {
                "The vectors z (dim ≤ 2: themselves)".to_string()
            } else {
                format!("The vectors z (dim {dim}: their two main directions)")
            })
            .strong(),
        );
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::hover());
        let rect = response.rect;
        let visuals = ui.visuals();
        let text = visuals.text_color();
        painter.rect_filled(rect, 2.0, visuals.extreme_bg_color);
        let z = &self.trainer.z;
        if z.is_empty() || self.trainer.diverged {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                if self.trainer.diverged {
                    "The vectors blew up: lower the learning rate."
                } else {
                    ""
                },
                FontId::proportional(15.0),
                PUSH_COLOR,
            );
            return;
        }
        let proj = Projection::of(z);
        let points: Vec<[f64; 2]> = z.iter().map(|v| proj.point(v)).collect();
        // Fit the points and the origin (dot products are measured from it).
        let (mut lo, mut hi) = ([0.0_f64; 2], [0.0_f64; 2]);
        for p in &points {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let inner = rect.shrink(30.0);
        let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-9);
        let mid = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
        let s = inner.width().min(inner.height()) as f64 / span;
        let at = |p: [f64; 2]| {
            inner.center() + vec2(((p[0] - mid[0]) * s) as f32, (-(p[1] - mid[1]) * s) as f32)
        };
        let origin = at(proj.point(&vec![0.0; dim]));
        let faint = Stroke::new(1.0, text.gamma_multiply(0.15));
        painter.line_segment(
            [pos2(rect.left(), origin.y), pos2(rect.right(), origin.y)],
            faint,
        );
        painter.line_segment(
            [pos2(origin.x, rect.top()), pos2(origin.x, rect.bottom())],
            faint,
        );

        // The followed drone's pulls and pushes, magnified so the step is visible.
        if let Some(e) = self
            .step
            .as_ref()
            .filter(|e| e.phase >= Phase::Terms && e.u < z.len())
        {
            let start = if e.phase == Phase::Step {
                e.z_u.clone()
            } else {
                z[e.u].clone()
            };
            let shifted = |delta: &[f64], k: f64| -> Vec<f64> {
                start.iter().zip(delta).map(|(a, d)| a + k * d).collect()
            };
            let step: Vec<f64> = e.grad.iter().map(|g| -e.lr * g).collect();
            let step_px = (at(proj.point(&shifted(&step, 1.0))) - at(proj.point(&start))).length();
            let mag = if step_px > 0.0 {
                (80.0 / step_px as f64).max(1.0)
            } else {
                1.0
            };
            let from = at(proj.point(&start));
            if e.phase < Phase::Step {
                let order = e.order();
                for &v in order.iter().take(if z.len() <= 30 { z.len() } else { 12 }) {
                    let delta: Vec<f64> = e.terms[v].iter().map(|t| -e.lr * t).collect();
                    let to = at(proj.point(&shifted(&delta, mag)));
                    let c = if e.errors[v] > 0.0 {
                        PUSH_COLOR
                    } else {
                        PULL_COLOR
                    };
                    painter.arrow(from, to - from, Stroke::new(1.5, c.gamma_multiply(0.8)));
                }
            }
            if e.phase >= Phase::Gradient {
                let to = at(proj.point(&shifted(&step, mag)));
                painter.arrow(from, to - from, Stroke::new(3.5, FOCUS_RING));
                painter.text(
                    rect.left_top() + vec2(8.0, 8.0),
                    Align2::LEFT_TOP,
                    format!("arrows magnified × {}", num(mag.round())),
                    FontId::proportional(12.0),
                    text,
                );
            }
            if e.phase == Phase::Step {
                painter.circle_stroke(
                    from,
                    NODE_PX - 2.0,
                    Stroke::new(1.5, FOCUS_RING.gamma_multiply(0.6)),
                );
            }
        }

        let labels = self.labels();
        for (i, p) in points.iter().enumerate() {
            let c = at(*p);
            painter.circle_filled(c, 6.0, self.label_color(labels[i]));
            painter.circle_stroke(c, 6.0, Stroke::new(1.0, Color32::from_black_alpha(160)));
            if points.len() <= 40 {
                painter.text(
                    c + vec2(7.0, -7.0),
                    Align2::LEFT_BOTTOM,
                    i.to_string(),
                    FontId::proportional(10.0),
                    visuals.weak_text_color(),
                );
            }
        }
        if self.focus < points.len() {
            painter.circle_stroke(at(points[self.focus]), 10.0, Stroke::new(2.5, FOCUS_RING));
        }
        if let Some(i) = self.hover.filter(|&i| i < points.len()) {
            painter.circle_stroke(at(points[i]), 12.0, Stroke::new(2.0, text));
        }
        painter.text(
            rect.left_bottom() + vec2(8.0, -8.0),
            Align2::LEFT_BOTTOM,
            "+ is the origin: z_u · z_v is large when two arrows from it point the same way and are long",
            FontId::proportional(11.0),
            visuals.weak_text_color(),
        );
    }

    // ------------------------------------------------------------------- right panel

    fn right(&mut self, ui: &mut egui::Ui) {
        let n = self.graph.n();
        if n > 0 {
            let same = nearest_same_label(&self.trainer.z, &self.labels());
            ui.label(format!(
                "Epoch {} · loss {} · {same}/{n} drones: most similar vector has the same label",
                self.trainer.epoch,
                num(*self.trainer.losses.last().unwrap_or(&0.0))
            ));
        }
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Math, "Math");
            ui.selectable_value(&mut self.tab, Tab::Matrices, "Matrices");
            ui.selectable_value(&mut self.tab, Tab::Hidden, "Hidden links");
        });
        ui.separator();
        if n == 0 {
            ui.weak("Add drones and links first.");
            return;
        }
        egui::ScrollArea::vertical().show(ui, |ui| match self.tab {
            Tab::Math => self.math_tab(ui),
            Tab::Matrices => self.matrices_tab(ui),
            Tab::Hidden => self.hidden_tab(ui),
        });
    }

    fn math_tab(&self, ui: &mut egui::Ui) {
        let Some(s) = &self.step else {
            ui.label(RichText::new("Press Space to follow one epoch.").strong());
            ui.weak(
                "With \"Explain each epoch in 7 small steps\" on, each Space adds one small \
                 step here, with drone u's real numbers (right-click a drone to follow \
                 another), and lights it up on the graph and on the vectors.",
            );
            mono(ui, "e_uv = z_u · z_v − A_uv".into());
            mono(ui, "L    = Σ_u Σ_{v≠u} e_uv²  +  l2 · Σ_u |z_u|²".into());
            mono(ui, "∂L/∂z_u = 4 Σ_{v≠u} e_uv z_v  +  2 l2 z_u".into());
            return;
        };
        ui.label(
            RichText::new(format!(
                "Epoch {}, seen from drone u = {}",
                self.trainer.epoch + usize::from(s.phase < Phase::Step),
                s.u
            ))
            .strong(),
        );
        let text = ui.visuals().text_color();
        let faint = ui.visuals().weak_text_color().gamma_multiply(0.5);
        for phase in Phase::ALL.into_iter().filter(|&p| p <= s.phase) {
            let current = phase == s.phase;
            let frame = egui::Frame::group(ui.style()).stroke(Stroke::new(
                if current { 2.0 } else { 1.0 },
                if current { FOCUS_RING } else { faint },
            ));
            let response = frame
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new(format!("{}. {}", phase.number(), phase.title()))
                            .strong()
                            .color(if current { FOCUS_RING } else { text }),
                    );
                    self.math_step(ui, s, phase);
                })
                .response;
            if current && self.scroll_math.replace(false) {
                response.scroll_to_me(Some(egui::Align::Center));
            }
        }
    }

    fn math_step(&self, ui: &mut egui::Ui, s: &Explained, phase: Phase) {
        let u = s.u;
        let n = s.dots.len();
        let order = s.order();
        let shown: Vec<usize> = order.iter().copied().take(MATH_ROWS).collect();
        let more = |ui: &mut egui::Ui| {
            if n - 1 > shown.len() {
                ui.weak(format!(
                    "… and {} more drones (largest errors shown first).",
                    n - 1 - shown.len()
                ));
            }
        };
        let label = |v: usize| self.graph.label_name(self.graph.nodes[v].label);
        match phase {
            Phase::Focus => {
                mono(ui, format!("u = {u} (label {})", label(u)));
                mono(ui, format!("z_u = {}", vec_str(&s.z_u)));
                ui.weak(
                    "One epoch moves every drone at once, all from the same Z. Here we \
                     follow u: the other drones' moves are computed the same way.",
                );
            }
            Phase::Row => {
                let linked: Vec<usize> = (0..n).filter(|&v| s.a_row[v] != 0.0).collect();
                for &v in &linked {
                    mono(ui, format!("A_{u},{v} = {}", num(s.a_row[v])));
                }
                let hidden: Vec<usize> = (0..n)
                    .filter(|&v| self.hidden.contains(&(u.min(v) as u32, u.max(v) as u32)))
                    .collect();
                for v in hidden {
                    colored(
                        ui,
                        HIDDEN_COLOR,
                        format!("A_{u},{v} = 0   (hidden: a link, but training sees 0)"),
                    );
                }
                mono(
                    ui,
                    format!("A_{u},v = 0 for the other {} drones", n - 1 - linked.len()),
                );
                ui.weak("This row is what z_u · z_v should rebuild, for every v ≠ u.");
            }
            Phase::Dots => {
                mono(ui, "z_u · z_v  (what the vectors say now)".into());
                for &v in &shown {
                    mono(
                        ui,
                        format!(
                            "z_{u} · z_{v} = {} · {} = {}",
                            vec_str(&s.z_u),
                            vec_str(&s.z_all[v]),
                            num(s.dots[v])
                        ),
                    );
                }
                more(ui);
            }
            Phase::Errors => {
                mono(ui, "e_uv = z_u · z_v − A_uv".into());
                for &v in &shown {
                    let e = s.errors[v];
                    let (c, word) = if e > 0.0 {
                        (PUSH_COLOR, "too similar → push apart")
                    } else {
                        (PULL_COLOR, "not similar enough → pull together")
                    };
                    colored(
                        ui,
                        c,
                        format!(
                            "e_{u},{v} = {} − {} = {}   {word}",
                            num(s.dots[v]),
                            num(s.a_row[v]),
                            num(e)
                        ),
                    );
                }
                more(ui);
                let sq: f64 = (0..n)
                    .filter(|&v| v != u)
                    .map(|v| s.errors[v].powi(2))
                    .sum();
                mono(
                    ui,
                    format!(
                        "Σ_v e_uv² = {}   (u's share of the loss, counted twice in L)",
                        num(sq)
                    ),
                );
            }
            Phase::Terms => {
                mono(ui, "each v adds 4 e_uv z_v to the gradient".into());
                for &v in &shown {
                    let c = if s.errors[v] > 0.0 {
                        PUSH_COLOR
                    } else {
                        PULL_COLOR
                    };
                    colored(
                        ui,
                        c,
                        format!(
                            "4 × {} × {} = {}",
                            num(s.errors[v]),
                            vec_str(&s.z_all[v]),
                            vec_str(&s.terms[v])
                        ),
                    );
                }
                more(ui);
                ui.weak(
                    "The step goes against the gradient: −4 e z_v points toward z_v when e < 0 \
                     (pull) and away from it when e > 0 (push). The 4: z_u is in e_uv and in \
                     e_vu, and each e² gives 2e.",
                );
            }
            Phase::Gradient => {
                let mut sum = vec![0.0; s.grad.len()];
                for v in (0..n).filter(|&v| v != u) {
                    for (a, t) in sum.iter_mut().zip(&s.terms[v]) {
                        *a += t;
                    }
                }
                mono(ui, "∂L/∂z_u = Σ_{v≠u} 4 e_uv z_v  +  2 l2 z_u".into());
                mono(
                    ui,
                    format!(
                        "        = {}  +  2 × {} × {}",
                        vec_str(&sum),
                        num(self.trainer.config.l2),
                        vec_str(&s.z_u)
                    ),
                );
                mono(
                    ui,
                    format!("        = {}  +  {}", vec_str(&sum), vec_str(&s.penalty)),
                );
                mono(ui, format!("        = {}", vec_str(&s.grad)));
            }
            Phase::Step => {
                let step: Vec<f64> = s.grad.iter().map(|g| -s.lr * g).collect();
                mono(ui, "z_u ← z_u − lr · ∂L/∂z_u".into());
                mono(
                    ui,
                    format!(
                        "    = {} − {} × {}",
                        vec_str(&s.z_u),
                        num(s.lr),
                        vec_str(&s.grad)
                    ),
                );
                mono(
                    ui,
                    format!("    = {} + {}", vec_str(&s.z_u), vec_str(&step)),
                );
                if let Some(after) = &s.z_after {
                    mono(ui, format!("    = {}", vec_str(after)));
                }
                if let Some(l) = s.loss_after {
                    let c = if l < s.loss_before {
                        PULL_COLOR
                    } else {
                        PUSH_COLOR
                    };
                    colored(ui, c, format!("loss {} → {}", num(s.loss_before), num(l)));
                }
                ui.weak(
                    "Every drone took its own step from the same old Z (full batch): that is \
                     one epoch. Space starts the next one.",
                );
            }
        }
    }

    fn matrices_tab(&self, ui: &mut egui::Ui) {
        let n = self.graph.n();
        let z = &self.trainer.z;
        let a = &self.trainer.a;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (self.graph.nodes[i].label, i));
        let focus = self.focus;
        let colors: Vec<Color32> = order
            .iter()
            .map(|&i| self.label_color(self.graph.nodes[i].label))
            .collect();
        let a_max = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| a[(i, j)])
            .fold(1e-12, f64::max);
        ui.weak(
            "Drones sorted by label, so groups show up as blocks. Gold marks: the drone followed.",
        );
        ui.label(RichText::new("A: what to rebuild").strong());
        heatmap(
            ui,
            "a",
            &order,
            &colors,
            focus,
            &|i, j| a[(i, j)],
            Scale::Sequential(a_max),
            true,
        );
        ui.label(RichText::new("Z Zᵀ: what the vectors say (z_u · z_v)").strong());
        heatmap(
            ui,
            "zz",
            &order,
            &colors,
            focus,
            &|i, j| dot(&z[i], &z[j]),
            Scale::Sequential(a_max),
            false,
        );
        let e_max = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .filter(|(i, j)| i != j)
            .map(|(i, j)| (dot(&z[i], &z[j]) - a[(i, j)]).abs())
            .fold(1e-12, f64::max);
        ui.label(RichText::new("E = Z Zᵀ − A: the errors (diagonal left out)").strong());
        heatmap(
            ui,
            "e",
            &order,
            &colors,
            focus,
            &|i, j| dot(&z[i], &z[j]) - a[(i, j)],
            Scale::Diverging(e_max),
            false,
        );
        ui.horizontal(|ui| {
            ui.colored_label(PULL_COLOR, "■ e < 0: too small (pull)");
            ui.colored_label(PUSH_COLOR, "■ e > 0: too big (push)");
        });
        ui.weak(format!(
            "Z Zᵀ has rank at most {} (the dimension): it can only copy A's big patterns, \
             like the blocks, not every entry.",
            self.settings.config.dim
        ));
    }

    fn hidden_tab(&self, ui: &mut egui::Ui) {
        if self.hidden.is_empty() {
            ui.label(RichText::new("No hidden link yet.").strong());
            ui.weak(
                "Pick ✂ Hide and click two linked drones. Training then sees A_uv = 0 for \
                 that pair, exactly like Mission 2's link test. Does z_u · z_v still come out \
                 high?",
            );
            return;
        }
        let z = &self.trainer.z;
        let n = self.graph.n();
        let linked: HashSet<(u32, u32)> = self
            .graph
            .edges
            .edges
            .iter()
            .map(|e| (e.src, e.dst))
            .collect();
        let non_links: Vec<f64> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .filter(|&(i, j)| !linked.contains(&(i as u32, j as u32)))
            .map(|(i, j)| dot(&z[i], &z[j]))
            .collect();
        let mut hidden: Vec<(u32, u32)> = self.hidden.iter().copied().collect();
        hidden.sort_unstable();
        let scores: Vec<f64> = hidden
            .iter()
            .map(|&(a, b)| dot(&z[a as usize], &z[b as usize]))
            .collect();
        egui::Grid::new("hidden").striped(true).show(ui, |ui| {
            for h in ["link", "z_u · z_v", "beats this share of non-links"] {
                ui.label(RichText::new(h).strong());
            }
            ui.end_row();
            for (&(a, b), &s) in hidden.iter().zip(&scores) {
                let beaten = non_links.iter().filter(|&&x| x < s).count() as f64
                    / non_links.len().max(1) as f64;
                ui.colored_label(HIDDEN_COLOR, format!("{a} – {b}"));
                ui.label(num(s));
                ui.label(format!("{:.0} %", 100.0 * beaten));
                ui.end_row();
            }
        });
        if !non_links.is_empty() {
            mono(
                ui,
                format!(
                    "AUC (hidden links vs {} pairs that never talked) = {:.3}",
                    non_links.len(),
                    auc(&scores, &non_links)
                ),
            );
        }
        ui.weak(
            "Training pushes a hidden pair toward 0 like any non-link. But with few numbers \
             per drone the vectors can't follow every entry: if u and v share most of their \
             neighbours, their vectors point the same way anyway, and the hidden link still \
             scores high. That is why MF did well on Mission 2's link test (AUC 0.976).",
        );
    }

    // ------------------------------------------------------------------ bottom panel

    fn loss_plot(&self, ui: &mut egui::Ui) {
        let losses = &self.trainer.losses;
        ui.label(RichText::new("Loss (log scale) per epoch").strong());
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::hover());
        let rect = response.rect;
        let visuals = ui.visuals();
        let text = visuals.text_color();
        let plot = Rect::from_min_max(rect.min + vec2(50.0, 6.0), rect.max - vec2(10.0, 18.0));
        painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
        let finite: Vec<f64> = losses
            .iter()
            .copied()
            .filter(|l| l.is_finite() && *l > 0.0)
            .collect();
        if finite.is_empty() {
            return;
        }
        let lo = finite
            .iter()
            .copied()
            .fold(f64::MAX, f64::min)
            .log10()
            .floor();
        let hi = finite
            .iter()
            .copied()
            .fold(0.0, f64::max)
            .log10()
            .ceil()
            .max(lo + 1.0);
        let epochs = self.trainer.config.epochs.max(1) as f32;
        let at = |k: usize, l: f64| {
            pos2(
                plot.left() + plot.width() * (k as f32 / epochs).min(1.0),
                plot.bottom() - plot.height() * ((l.max(1e-300).log10() - lo) / (hi - lo)) as f32,
            )
        };
        for d in (lo as i32)..=(hi as i32) {
            let y = at(0, 10f64.powi(d)).y;
            painter.line_segment(
                [pos2(plot.left(), y), pos2(plot.right(), y)],
                Stroke::new(1.0, text.gamma_multiply(0.1)),
            );
            painter.text(
                pos2(plot.left() - 4.0, y),
                Align2::RIGHT_CENTER,
                format!("1e{d}"),
                FontId::proportional(10.0),
                text,
            );
        }
        let pts: Vec<Pos2> = losses
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_finite() && **l > 0.0)
            .map(|(k, &l)| at(k, l.min(10f64.powf(hi))))
            .collect();
        painter.add(Shape::line(pts, Stroke::new(2.0, FOCUS_RING)));
        painter.text(
            pos2(plot.right(), plot.bottom() + 2.0),
            Align2::RIGHT_TOP,
            format!(
                "epoch {} / {}",
                self.trainer.epoch, self.trainer.config.epochs
            ),
            FontId::proportional(10.0),
            text,
        );
    }
}

// ---------------------------------------------------------------------------- drawing

fn mono(ui: &mut egui::Ui, t: String) {
    ui.label(RichText::new(t).monospace());
}

fn colored(ui: &mut egui::Ui, c: Color32, t: String) {
    ui.label(RichText::new(t).monospace().color(c));
}

/// A vector as (a, b, c), only its first 4 numbers if it is longer.
fn vec_str(v: &[f64]) -> String {
    let shown: Vec<String> = v.iter().take(4).map(|&x| num(x)).collect();
    format!(
        "({}{})",
        shown.join(", "),
        if v.len() > 4 { ", …" } else { "" }
    )
}

/// How a heatmap maps values to colours.
#[derive(Debug, Clone, Copy)]
enum Scale {
    /// 0 → background, max → blue.
    Sequential(f64),
    /// −max → green (pull), 0 → background, +max → red (push).
    Diverging(f64),
}

/// An n × n heatmap of `value(i, j)` over the drones in `order`, with label colour bars on
/// the edges, the followed drone's row and column marked, and a tooltip.
#[allow(clippy::too_many_arguments)]
fn heatmap(
    ui: &mut egui::Ui,
    id: &str,
    order: &[usize],
    colors: &[Color32],
    focus: usize,
    value: &dyn Fn(usize, usize) -> f64,
    scale: Scale,
    with_diagonal: bool,
) {
    let n = order.len();
    let side = ui.available_width().min(330.0);
    let (response, painter) = ui.allocate_painter(vec2(side, side), Sense::hover());
    let rect = response.rect;
    let bar = 6.0;
    let cell = (side - bar) / n.max(1) as f32;
    let grid = Rect::from_min_size(
        rect.min + vec2(bar, bar),
        vec2(cell * n as f32, cell * n as f32),
    );
    let bg = ui.visuals().extreme_bg_color;
    let mix = |c: Color32, t: f32| {
        let t = t.clamp(0.0, 1.0);
        let m = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t) as u8;
        Color32::from_rgb(m(bg.r(), c.r()), m(bg.g(), c.g()), m(bg.b(), c.b()))
    };
    for (r, &i) in order.iter().enumerate() {
        painter.rect_filled(
            Rect::from_min_size(
                pos2(rect.left(), grid.top() + r as f32 * cell),
                vec2(bar - 1.0, cell),
            ),
            0.0,
            colors[r],
        );
        painter.rect_filled(
            Rect::from_min_size(
                pos2(grid.left() + r as f32 * cell, rect.top()),
                vec2(cell, bar - 1.0),
            ),
            0.0,
            colors[r],
        );
        for (c, &j) in order.iter().enumerate() {
            let x = if i == j && !with_diagonal {
                None
            } else {
                Some(value(i, j))
            };
            let fill = match (x, scale) {
                (None, _) => ui.visuals().faint_bg_color,
                (Some(x), Scale::Sequential(max)) => {
                    mix(Color32::from_rgb(78, 121, 167), (x / max) as f32)
                }
                (Some(x), Scale::Diverging(max)) if x >= 0.0 => mix(PUSH_COLOR, (x / max) as f32),
                (Some(x), Scale::Diverging(max)) => mix(PULL_COLOR, (-x / max) as f32),
            };
            painter.rect_filled(
                Rect::from_min_size(
                    pos2(grid.left() + c as f32 * cell, grid.top() + r as f32 * cell),
                    vec2(cell, cell),
                ),
                0.0,
                fill,
            );
        }
    }
    if let Some(r) = order.iter().position(|&i| i == focus) {
        let stroke = Stroke::new(1.5, FOCUS_RING);
        painter.rect_stroke(
            Rect::from_min_size(
                pos2(grid.left(), grid.top() + r as f32 * cell),
                vec2(grid.width(), cell),
            ),
            0.0,
            stroke,
            egui::StrokeKind::Outside,
        );
    }
    if let Some(p) = response.hover_pos()
        && grid.contains(p)
    {
        let r = (((p.y - grid.top()) / cell) as usize).min(n - 1);
        let c = (((p.x - grid.left()) / cell) as usize).min(n - 1);
        let (i, j) = (order[r], order[c]);
        let tip = if i == j && !with_diagonal {
            format!("{i}, {j}: left out (a drone isn't linked to itself)")
        } else {
            format!("{i}, {j}: {}", num(value(i, j)))
        };
        response.on_hover_text_at_pointer(tip);
    }
    let _ = id;
}

// ---------------------------------------------------------- shared with the Node2Vec playground

/// World ↔ screen for the graph canvas: fits the drones and at least ±BOX.
struct Fit {
    min: [f32; 2],
    scale: f32,
    offset: egui::Vec2,
}

impl Fit {
    fn new(nodes: &[Node], rect: Rect) -> Fit {
        let (mut lo, mut hi) = ([-BOX; 2], [BOX; 2]);
        for nd in nodes {
            for k in 0..2 {
                lo[k] = lo[k].min(nd.pos[k]);
                hi[k] = hi[k].max(nd.pos[k]);
            }
        }
        let span = [hi[0] - lo[0], hi[1] - lo[1]];
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

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let d = (dot(a, a) * dot(b, b)).sqrt();
    if d > 0.0 { dot(a, b) / d } else { 0.0 }
}

fn nearest_same_label(z: &[Vec<f64>], labels: &[usize]) -> usize {
    (0..z.len())
        .filter(|&i| {
            (0..z.len())
                .filter(|&j| j != i)
                .max_by(|&a, &b| cosine(&z[i], &z[a]).total_cmp(&cosine(&z[i], &z[b])))
                .is_some_and(|j| labels[j] == labels[i])
        })
        .count()
}

/// Main direction of a symmetric matrix (power iteration), orthogonal to `not`.
fn top_direction(m: &[Vec<f64>], not: Option<&[f64]>) -> Vec<f64> {
    let d = m.len();
    let remove = |v: &mut Vec<f64>| {
        if let Some(u) = not {
            let k = dot(v, u);
            for (x, y) in v.iter_mut().zip(u) {
                *x -= k * y;
            }
        }
    };
    let normalise = |v: &mut Vec<f64>| {
        let len = dot(v, v).sqrt();
        if len > 0.0 {
            v.iter_mut().for_each(|x| *x /= len);
        }
        len
    };
    let mut v: Vec<f64> = (0..d).map(|i| ((i as f64 + 1.0) * 1.618).sin()).collect();
    remove(&mut v);
    normalise(&mut v);
    for _ in 0..200 {
        let mut next: Vec<f64> = (0..d).map(|i| dot(&m[i], &v)).collect();
        remove(&mut next);
        if normalise(&mut next) == 0.0 {
            break;
        }
        v = next;
    }
    v
}

/// A number for the Math tab: whole numbers as they are, others with 3 decimals.
fn num(x: f64) -> String {
    if x == x.round() && x.abs() < 1e9 {
        format!("{x:.0}")
    } else {
        format!("{x:.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trainer_for(preset: Preset) -> (Graph, Trainer) {
        let g = Graph::build(preset);
        let s = Settings::for_preset(preset);
        let t = Trainer::new(build_a(&g, &HashSet::new(), s.weights), s.config, s.seed);
        (g, t)
    }

    #[test]
    fn epoch_by_epoch_is_matrix_factorization_exactly() {
        let (_, mut t) = trainer_for(Preset::TwoGroups);
        t.run(40);
        let config = MfConfig {
            epochs: 40,
            ..t.config
        };
        assert_eq!(t.z, matrix_factorization(&t.a, &config, t.seed));
        t.run(usize::MAX);
        assert_eq!(t.epoch, t.config.epochs);
        assert_eq!(t.same, Some(true));
    }

    #[test]
    fn the_explained_gradient_is_mf_gradient_and_the_step_is_the_trainers() {
        let (_, mut t) = trainer_for(Preset::Barbell);
        t.run(5);
        let full = mf_gradient(&t.a, &t.z, t.config.l2);
        for u in [0, 6, 11] {
            let e = Explained::new(&t, u, Phase::Focus);
            assert_eq!(e.grad, full[u], "drone {u}");
            assert_eq!(e.errors[u], 0.0);
            let order = e.order();
            assert_eq!(order.len(), t.n() - 1);
            assert!(
                order
                    .windows(2)
                    .all(|w| e.errors[w[0]].abs() >= e.errors[w[1]].abs())
            );
        }
    }

    #[test]
    fn seven_small_steps_make_one_epoch() {
        let mut app = Playground::default();
        app.ensure_trainer();
        app.focus = 3;
        for k in 0..7 {
            app.advance();
            let s = app.step.as_ref().unwrap();
            assert_eq!(s.phase, Phase::ALL[k]);
            assert_eq!(app.trainer.epoch, usize::from(k == 6));
        }
        let s = app.step.as_ref().unwrap();
        let moved: Vec<f64> = s
            .z_u
            .iter()
            .zip(&s.grad)
            .map(|(z, g)| z - s.lr * g)
            .collect();
        assert_eq!(s.z_after.as_ref().unwrap(), &moved);
        assert_eq!(&app.trainer.z[3], &moved);
        // Whole epoch from the middle of one finishes it; from its end, takes a new one.
        app.advance();
        app.advance();
        app.whole_epoch();
        assert_eq!(app.trainer.epoch, 2);
        app.whole_epoch();
        assert_eq!(app.trainer.epoch, 3);
    }

    #[test]
    fn every_preset_trains_with_its_own_settings() {
        for preset in Preset::ALL {
            let (g, mut t) = trainer_for(preset);
            if g.n() == 0 {
                continue;
            }
            let before = t.losses[0];
            t.run(usize::MAX);
            assert!(!t.diverged, "{}", preset.name());
            // Down, and settled: the last 50 epochs change it by under 1 %. (Not always far
            // down: at dim 2, Z Zᵀ can't copy a ring's A; the ring stops at 70 % of its start.)
            let last = *t.losses.last().unwrap();
            let earlier = t.losses[t.losses.len() - 51];
            assert!(last < 0.8 * before, "{}: {before} -> {last}", preset.name());
            assert!(
                earlier - last < 0.01 * last,
                "{}: still falling",
                preset.name()
            );
            assert_eq!(t.same, Some(true), "{}", preset.name());
        }
    }

    #[test]
    fn too_big_a_learning_rate_is_caught() {
        let g = Graph::build(Preset::SwarmBySquad);
        let config = MfConfig {
            dim: 2,
            epochs: 300,
            lr: 0.2,
            l2: 0.01,
        };
        let mut t = Trainer::new(build_a(&g, &HashSet::new(), Weights::Raw), config, 1);
        t.run(usize::MAX);
        assert!(t.diverged);
        assert!(t.finished());
    }

    #[test]
    fn a_hidden_link_is_zero_in_a_and_still_scores_high_in_two_groups() {
        let g = Graph::build(Preset::TwoGroups);
        let hidden: HashSet<(u32, u32)> = [(0, 1)].into_iter().collect();
        let a = build_a(&g, &hidden, Weights::One);
        assert_eq!((a[(0, 1)], a[(1, 0)], a[(0, 2)]), (0.0, 0.0, 1.0));
        let config = MfConfig {
            dim: 2,
            epochs: 400,
            lr: 0.01,
            l2: 0.01,
        };
        let mut t = Trainer::new(a, config, 1);
        t.run(usize::MAX);
        let hidden_score = dot(&t.z[0], &t.z[1]);
        let across = dot(&t.z[0], &t.z[8]);
        assert!(
            hidden_score > across + 0.3,
            "hidden {hidden_score}, across {across}"
        );
    }

    #[test]
    fn weights_modes() {
        assert_eq!(Weights::One.apply(57.0), 1.0);
        assert_eq!(Weights::Raw.apply(57.0), 57.0);
        assert!((Weights::Log.apply(std::f64::consts::E - 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn presets_are_valid_graphs() {
        for preset in Preset::ALL {
            let g = Graph::build(preset);
            assert_eq!(g.edges.n as usize, g.n(), "{}", preset.name());
            let mut seen = HashSet::new();
            for e in &g.edges.edges {
                assert!(
                    e.src < e.dst && (e.dst as usize) < g.n(),
                    "{}",
                    preset.name()
                );
                assert!(
                    seen.insert((e.src, e.dst)),
                    "duplicate link in {}",
                    preset.name()
                );
            }
        }
    }

    #[test]
    fn every_small_step_and_tab_draws_without_panicking() {
        for preset in [Preset::TwoGroups, Preset::SwarmByRole] {
            let mut app = Playground::default();
            app.load(preset);
            app.ensure_trainer();
            app.hidden.insert((0, 1));
            app.changed();
            app.ensure_trainer();
            for _ in 0..9 {
                app.advance();
                for tab in [Tab::Math, Tab::Matrices, Tab::Hidden] {
                    app.tab = tab;
                    let mut out =
                        egui::Context::default().run_ui(egui::RawInput::default(), |ui| {
                            app.right(ui);
                            app.graph_canvas(ui);
                            app.embed_canvas(ui);
                            app.loss_plot(ui);
                        });
                    out.textures_delta.clear();
                }
            }
            assert_eq!(app.trainer.epoch, 1, "{}", preset.name());
        }
    }
}
