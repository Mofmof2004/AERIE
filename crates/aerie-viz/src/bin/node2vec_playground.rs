//! Node2Vec playground (Mission 2): see how p and q bend the walks.
//!
//! Run with `cargo run --release -p aerie-viz --bin node2vec_playground`.
//!
//! - Left: the graph (presets, or build your own: add drones, link them, paint labels,
//!   delete), p and q, the walk builder, the many-walks settings, the embedding settings.
//!   With "Explain each step in 10 small steps" on, Space goes through one walk step in 10:
//!   where the walk is (t, v), v's neighbours, where each lies from t, its bias, weight ×
//!   bias = π, the total Z, the chances π / Z, the random number u, where u falls, the move.
//! - Centre: the graph, lit up small step by small step: current drone (gold), previous
//!   drone t (violet), every neighbour ringed and labelled (weight, kind, bias, π, chance),
//!   arrows coloured by bias (back to t ×1/p red, next to t ×1 amber, away from t ×1/q blue)
//!   and as wide as their chance. Right-click a drone to start the walk there.
//! - Right: *Math* (every small step reached so far as a formula with this walk's numbers),
//!   *This step* (the table: link weight × bias = chance, against DeepWalk's chance, and
//!   where the random number fell), *Many walks* (thousands of walks: how often they go back
//!   / near / away, how often they change label, how far they get, and a q sweep),
//!   *Embedding* (skip-gram on those walks, Node2Vec next to DeepWalk, coloured by label).
//! - Bottom: the walk being built, and walks from the start drone under Node2Vec and
//!   DeepWalk side by side.
//!
//! The walks follow the rule of `aerie_classic::embed::node2vec::biased_walks` step for
//! step (same neighbour order, one random number per step), so with p = q = 1 they are
//! exactly `random_walks`' walks, and the embeddings are exactly `train_skipgram` on them.

use std::cell::Cell;
use std::collections::{HashSet, VecDeque};

use aerie_classic::embed::skipgram::{SkipGramConfig, dot, train_skipgram};
use aerie_graph::{Edge, EdgeList};
use aerie_viz::style::{draw_drone, role_name, squad_color};
use eframe::egui::{
    self, Align2, Color32, FontId, Key, Pos2, Rect, RichText, Sense, Stroke, pos2, vec2,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::drone::Role;
use swarmsim::snapshot::Snapshot;
use swarmsim::traffic::TrafficConfig;

/// Ring around the drone the walk is on.
const CURRENT_RING: Color32 = Color32::from_rgb(255, 205, 60);
/// Ring around the drone the walk came from (t).
const PREVIOUS_RING: Color32 = Color32::from_rgb(190, 120, 255);
/// A step back to t (×1/p).
const BACK_COLOR: Color32 = Color32::from_rgb(235, 90, 80);
/// A step to a neighbour of t (×1).
const NEAR_COLOR: Color32 = Color32::from_rgb(240, 180, 60);
/// A step away from t (×1/q).
const FAR_COLOR: Color32 = Color32::from_rgb(80, 160, 255);
/// The first step (no t yet, ×1).
const FIRST_COLOR: Color32 = Color32::from_rgb(160, 160, 160);
/// The walk already built.
const WALK_COLOR: Color32 = Color32::from_rgb(110, 200, 160);
/// Node2Vec's series in the plots.
const N2V_COLOR: Color32 = Color32::from_rgb(80, 160, 255);
/// DeepWalk's series in the plots.
const DW_COLOR: Color32 = Color32::from_rgb(150, 150, 150);
/// Drone marker radius, pixels.
const NODE_PX: f32 = 9.0;
/// Presets fit in ±2.6; the canvas always shows at least ±BOX, so adding a drone near the
/// edge doesn't rescale everything.
const BOX: f32 = 3.0;
/// The q values of the sweep.
const Q_SWEEP: [f64; 7] = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0];
/// Walks shown per method in the bottom panel.
const SAMPLE_WALKS: usize = 4;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE Node2Vec playground")
            .with_inner_size([1650.0, 980.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE Node2Vec playground",
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
                "Two 6-drone cliques joined by one link. How often does a walk cross the \
                 bridge, and does q change that?"
            }
            Preset::Barbell => {
                "Two 5-cliques joined by a path of 3. Inside a clique almost every neighbour \
                 is also next to t (×1); on the path every step forward is away from t (×1/q)."
            }
            Preset::WheelsByGroup | Preset::WheelsByRole => {
                "Three wheels (a hub linked to a rim of 6), hubs joined through bridge drones. \
                 Same graph, two labelings: by group (community) or by role (hub, rim, \
                 bridge: structure). Which q makes the vectors find which labels?"
            }
            Preset::Ring => {
                "From any drone there are only two moves: back to t (×1/p) or on, away from \
                 t (×1/q). The cleanest place to see p and q."
            }
            Preset::Grid => {
                "A 6 × 6 lattice, labelled by quarter. No triangles: a neighbour of the \
                 current drone is never next to t, so every forward step is ×1/q."
            }
            Preset::SwarmBySquad | Preset::SwarmByRole => {
                "Mission 2's traffic graph of a 100-drone swarm (weight = messages). Same \
                 graph, labelled by squad or by role: experiment 1 in miniature."
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

// ----------------------------------------------------------------------------- the rule

/// Where a neighbour x of the current drone v lies, seen from t (the drone before v).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// The first step of a walk: no t yet, plain DeepWalk.
    First,
    /// x is t itself: going back.
    Back,
    /// x is also a neighbour of t: staying near t.
    Near,
    /// x is not next to t: moving away from t.
    Far,
}

impl Kind {
    fn bias(self, p: f64, q: f64) -> f64 {
        match self {
            Kind::First | Kind::Near => 1.0,
            Kind::Back => 1.0 / p,
            Kind::Far => 1.0 / q,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::First => "first step",
            Kind::Back => "back to t",
            Kind::Near => "next to t",
            Kind::Far => "away from t",
        }
    }

    fn rule(self) -> &'static str {
        match self {
            Kind::First | Kind::Near => "× 1",
            Kind::Back => "× 1/p",
            Kind::Far => "× 1/q",
        }
    }

    fn color(self) -> Color32 {
        match self {
            Kind::First => FIRST_COLOR,
            Kind::Back => BACK_COLOR,
            Kind::Near => NEAR_COLOR,
            Kind::Far => FAR_COLOR,
        }
    }
}

/// One possible next step.
#[derive(Debug, Clone, Copy)]
struct Choice {
    to: usize,
    /// Link weight.
    weight: f64,
    kind: Kind,
    /// weight × bias: the unnormalised chance.
    biased: f64,
}

/// Neighbour lists in the order `random_walks` builds them, and a link set for "is x next
/// to t?".
struct Adjacency {
    neighbours: Vec<Vec<(usize, f64)>>,
    linked: HashSet<(usize, usize)>,
}

impl Adjacency {
    fn new(graph: &EdgeList) -> Adjacency {
        let n = graph.n as usize;
        let mut neighbours: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut linked = HashSet::new();
        for e in &graph.edges {
            let (a, b) = (e.src as usize, e.dst as usize);
            neighbours[a].push((b, e.weight));
            neighbours[b].push((a, e.weight));
            linked.insert((a, b));
            linked.insert((b, a));
        }
        Adjacency { neighbours, linked }
    }

    fn kind(&self, previous: Option<usize>, x: usize) -> Kind {
        match previous {
            None => Kind::First,
            Some(t) if x == t => Kind::Back,
            Some(t) if self.linked.contains(&(t, x)) => Kind::Near,
            Some(_) => Kind::Far,
        }
    }

    /// Every neighbour of `current`, with its bias seen from `previous`.
    fn options(&self, previous: Option<usize>, current: usize, p: f64, q: f64) -> Vec<Choice> {
        self.neighbours[current]
            .iter()
            .map(|&(to, weight)| {
                let kind = self.kind(previous, to);
                Choice {
                    to,
                    weight,
                    kind,
                    biased: weight * kind.bias(p, q),
                }
            })
            .collect()
    }
}

/// One random number u in 0..total, as `random_walks` and `biased_walks` draw it.
fn draw_u(options: &[Choice], rng: &mut ChaCha8Rng) -> f64 {
    let total: f64 = options.iter().map(|c| c.biased).sum();
    rng.random_range(0.0..total)
}

/// Where u falls: subtract the chances in order until it goes below 0. Returns the index
/// of the choice (the last one if rounding leaves u >= 0 at the end).
fn slice(options: &[Choice], u: f64) -> usize {
    let mut u = u;
    for (i, c) in options.iter().enumerate() {
        u -= c.biased;
        if u < 0.0 {
            return i;
        }
    }
    options.len() - 1
}

/// The pick of `random_walks` and `biased_walks`. Returns (index of the choice, u).
fn pick(options: &[Choice], rng: &mut ChaCha8Rng) -> (usize, f64) {
    let u = draw_u(options, rng);
    (slice(options, u), u)
}

/// `walks_per_node` rounds of one walk from every drone, as `biased_walks` makes them.
fn biased_walks(
    adj: &Adjacency,
    walks_per_node: usize,
    length: usize,
    p: f64,
    q: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec<usize>> {
    let n = adj.neighbours.len();
    let mut walks = Vec::with_capacity(walks_per_node * n);
    for _round in 0..walks_per_node {
        for start in 0..n {
            let mut walk = vec![start];
            let mut current = start;
            let mut previous = None;
            for _step in 1..length {
                let options = adj.options(previous, current, p, q);
                if options.is_empty() {
                    break;
                }
                let next = options[pick(&options, rng).0].to;
                walk.push(next);
                previous = Some(current);
                current = next;
            }
            walks.push(walk);
        }
    }
    walks
}

/// The walk RNG: the seed on stream 1, as `deepwalk` and `node2vec` use it.
fn walk_rng(seed: u64) -> ChaCha8Rng {
    let mut r = ChaCha8Rng::seed_from_u64(seed);
    r.set_stream(1);
    r
}

/// Hops between every pair of drones (BFS from each), `None` if unreachable.
fn hops(adj: &Adjacency) -> Vec<Vec<Option<usize>>> {
    let n = adj.neighbours.len();
    (0..n)
        .map(|s| {
            let mut d = vec![None; n];
            d[s] = Some(0);
            let mut queue = VecDeque::from([s]);
            while let Some(v) = queue.pop_front() {
                for &(x, _) in &adj.neighbours[v] {
                    if d[x].is_none() {
                        d[x] = Some(d[v].unwrap() + 1);
                        queue.push_back(x);
                    }
                }
            }
            d
        })
        .collect()
}

// ---------------------------------------------------------------------- walk builder

/// The 10 small steps of one walk step, as the explain mode shows them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    /// Where the walk is (v) and where it came from (t).
    Where,
    /// v's neighbours and their link weights.
    Neighbours,
    /// Where each neighbour lies from t: back, next to t, away.
    Kinds,
    /// The bias α of each: 1/p, 1 or 1/q.
    Biases,
    /// π = link weight × bias.
    Weighted,
    /// Z = the sum of the π.
    Total,
    /// The chances π / Z (and DeepWalk's, for comparison).
    Chances,
    /// One random number u in [0, Z).
    Draw,
    /// Where u falls: the next drone.
    Slice,
    /// The walk moves: t ← v, v ← the next drone.
    Move,
}

impl Phase {
    const ALL: [Phase; 10] = [
        Phase::Where,
        Phase::Neighbours,
        Phase::Kinds,
        Phase::Biases,
        Phase::Weighted,
        Phase::Total,
        Phase::Chances,
        Phase::Draw,
        Phase::Slice,
        Phase::Move,
    ];

    fn number(self) -> usize {
        self as usize + 1
    }

    fn title(self) -> &'static str {
        match self {
            Phase::Where => "Where the walk is",
            Phase::Neighbours => "The neighbours of v",
            Phase::Kinds => "Where each neighbour lies from t",
            Phase::Biases => "The bias of each neighbour",
            Phase::Weighted => "Link weight × bias",
            Phase::Total => "The total",
            Phase::Chances => "The chances",
            Phase::Draw => "One random number",
            Phase::Slice => "Where the number falls",
            Phase::Move => "The walk moves",
        }
    }

    fn next(self) -> Option<Phase> {
        Phase::ALL.get(self as usize + 1).copied()
    }
}

/// One walk step, small step by small step.
#[derive(Debug, Clone)]
struct Explained {
    phase: Phase,
    previous: Option<usize>,
    current: usize,
    /// Every neighbour of `current` with its bias (recomputed if p or q move before the
    /// draw, frozen after).
    options: Vec<Choice>,
    /// The random number, from `Phase::Draw` on.
    u: Option<f64>,
    /// Index into `options` of the next drone, from `Phase::Slice` on.
    chosen: Option<usize>,
}

impl Explained {
    fn total(&self) -> f64 {
        self.options.iter().map(|c| c.biased).sum()
    }
}

/// The step just taken, to show where the random number fell.
#[derive(Debug, Clone)]
struct LastStep {
    previous: Option<usize>,
    from: usize,
    options: Vec<Choice>,
    chosen: usize,
    u: f64,
}

/// One walk, built step by step (or small step by small step). Its random numbers come
/// from its own stream, drawn exactly as `biased_walks` draws them: one per step.
struct Builder {
    start: usize,
    seed: u64,
    rng: ChaCha8Rng,
    walk: Vec<usize>,
    /// The step being explained, if one is under way (or just finished, at `Move`).
    step: Option<Explained>,
    last: Option<LastStep>,
}

impl Builder {
    fn new(start: usize, seed: u64) -> Builder {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(7);
        Builder {
            start,
            seed,
            rng,
            walk: vec![start],
            step: None,
            last: None,
        }
    }

    fn current(&self) -> usize {
        *self.walk.last().unwrap()
    }

    fn previous(&self) -> Option<usize> {
        (self.walk.len() >= 2).then(|| self.walk[self.walk.len() - 2])
    }

    /// (t, v) as the canvas should show them: the explained step's until it moves.
    fn shown(&self) -> (Option<usize>, usize) {
        match &self.step {
            Some(s) if s.phase < Phase::Move => (s.previous, s.current),
            _ => (self.previous(), self.current()),
        }
    }

    fn done(&self, adj: &Adjacency, length: usize) -> bool {
        self.walk.len() >= length || adj.neighbours[self.current()].is_empty()
    }

    /// Before the draw, the options follow p and q as they move.
    fn refresh(&mut self, adj: &Adjacency, p: f64, q: f64) {
        if let Some(s) = &mut self.step
            && s.u.is_none()
        {
            s.options = adj.options(s.previous, s.current, p, q);
        }
    }

    fn begin(&self, adj: &Adjacency, p: f64, q: f64) -> Explained {
        Explained {
            phase: Phase::Where,
            previous: self.previous(),
            current: self.current(),
            options: adj.options(self.previous(), self.current(), p, q),
            u: None,
            chosen: None,
        }
    }

    /// The next small step. After `Move`, the next step starts at `Where`.
    fn advance(&mut self, adj: &Adjacency, p: f64, q: f64, length: usize) {
        let next = self.step.as_ref().map(|s| s.phase.next());
        match next {
            // Nothing under way, or the last step has moved: start the next one.
            None | Some(None) => {
                self.step = None;
                if !self.done(adj, length) {
                    self.step = Some(self.begin(adj, p, q));
                }
            }
            Some(Some(phase)) => {
                let mut s = self.step.take().unwrap();
                s.phase = phase;
                match phase {
                    Phase::Draw => s.u = Some(draw_u(&s.options, &mut self.rng)),
                    Phase::Slice => s.chosen = Some(slice(&s.options, s.u.unwrap())),
                    Phase::Move => {
                        let chosen = s.chosen.unwrap();
                        self.walk.push(s.options[chosen].to);
                        self.last = Some(LastStep {
                            previous: s.previous,
                            from: s.current,
                            options: s.options.clone(),
                            chosen,
                            u: s.u.unwrap(),
                        });
                    }
                    _ => {}
                }
                self.step = Some(s);
            }
        }
    }

    /// The whole step at once: the rest of the one under way, or a new one.
    fn whole_step(&mut self, adj: &Adjacency, p: f64, q: f64, length: usize) {
        let fresh = self.step.as_ref().is_none_or(|s| s.phase == Phase::Move);
        if fresh {
            if self.done(adj, length) {
                return;
            }
            self.step = Some(self.begin(adj, p, q));
        }
        while self.step.as_ref().is_some_and(|s| s.phase != Phase::Move) {
            self.advance(adj, p, q, length);
        }
    }

    fn finish(&mut self, adj: &Adjacency, p: f64, q: f64, length: usize) {
        while !self.done(adj, length) {
            self.whole_step(adj, p, q, length);
        }
    }
}

// ----------------------------------------------------------------------- statistics

#[derive(Debug, Clone, Copy, PartialEq)]
struct RunSettings {
    walks_per_node: usize,
    length: usize,
    seed: u64,
    window: usize,
}

impl Default for RunSettings {
    fn default() -> Self {
        RunSettings {
            walks_per_node: 10,
            length: 20,
            seed: 1,
            window: 5,
        }
    }
}

/// What many walks with one (p, q) do.
#[derive(Debug, Clone)]
struct Stats {
    walks: Vec<Vec<usize>>,
    /// Steps of each kind: first, back, near, far.
    kinds: [usize; 4],
    /// Share of steps that change label.
    label_changes: f64,
    /// Share of training pairs (u, v within the window) with the same label.
    same_label_pairs: f64,
    /// Mean hops from the start after k steps.
    distance: Vec<f64>,
    /// Mean number of different drones in a walk.
    distinct: f64,
}

impl Stats {
    fn share(&self, kind: Kind) -> f64 {
        let moves = (self.kinds[1] + self.kinds[2] + self.kinds[3]).max(1);
        let k = match kind {
            Kind::First => return 0.0,
            Kind::Back => 1,
            Kind::Near => 2,
            Kind::Far => 3,
        };
        self.kinds[k] as f64 / moves as f64
    }

    fn end_distance(&self) -> f64 {
        self.distance.last().copied().unwrap_or(0.0)
    }
}

fn stats(
    graph: &Graph,
    adj: &Adjacency,
    hops: &[Vec<Option<usize>>],
    run: &RunSettings,
    p: f64,
    q: f64,
) -> Stats {
    let walks = biased_walks(
        adj,
        run.walks_per_node,
        run.length,
        p,
        q,
        &mut walk_rng(run.seed),
    );
    let label = |i: usize| graph.nodes[i].label;
    let mut kinds = [0usize; 4];
    let (mut steps, mut changes) = (0usize, 0usize);
    let (mut pairs, mut same) = (0usize, 0usize);
    let mut dist_sum = vec![0.0; run.length];
    let mut dist_count = vec![0usize; run.length];
    let mut distinct = 0.0;
    for w in &walks {
        for t in 1..w.len() {
            let previous = (t >= 2).then(|| w[t - 2]);
            kinds[adj.kind(previous, w[t]) as usize] += 1;
            steps += 1;
            changes += usize::from(label(w[t]) != label(w[t - 1]));
        }
        for t in 0..w.len() {
            let lo = t.saturating_sub(run.window);
            let hi = (t + run.window).min(w.len() - 1);
            for s in lo..=hi {
                if s != t {
                    pairs += 1;
                    same += usize::from(label(w[s]) == label(w[t]));
                }
            }
            if let Some(d) = hops[w[0]][w[t]] {
                dist_sum[t] += d as f64;
                dist_count[t] += 1;
            }
        }
        let mut seen: Vec<usize> = w.clone();
        seen.sort_unstable();
        seen.dedup();
        distinct += seen.len() as f64;
    }
    let distance: Vec<f64> = dist_sum
        .iter()
        .zip(&dist_count)
        .take_while(|&(_, &c)| c > 0)
        .map(|(s, &c)| s / c as f64)
        .collect();
    Stats {
        kinds,
        label_changes: changes as f64 / steps.max(1) as f64,
        same_label_pairs: same as f64 / pairs.max(1) as f64,
        distance,
        distinct: distinct / walks.len().max(1) as f64,
        walks,
    }
}

// ------------------------------------------------------------------------ embedding

#[derive(Debug, Clone, Copy, PartialEq)]
struct EmbedSettings {
    dim: usize,
    epochs: usize,
    negatives: usize,
    lr: f64,
}

impl Default for EmbedSettings {
    fn default() -> Self {
        EmbedSettings {
            dim: 8,
            epochs: 3,
            negatives: 5,
            lr: 0.025,
        }
    }
}

struct Embedding {
    /// Node2Vec (current p, q) and DeepWalk, trained on the walks of `Stats`.
    z: [Vec<Vec<f64>>; 2],
    /// Their 2-D pictures (raw at dim 2, PCA above).
    points: [Vec<[f64; 2]>; 2],
    /// Drones whose most similar other vector (cosine) has the same label.
    same_label_nearest: [usize; 2],
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

/// skip-gram on `walks` with the training stream of `deepwalk` / `node2vec` (stream 2).
fn train(walks: &[Vec<usize>], n: usize, run: &RunSettings, e: &EmbedSettings) -> Vec<Vec<f64>> {
    let mut rng = ChaCha8Rng::seed_from_u64(run.seed);
    rng.set_stream(2);
    let config = SkipGramConfig {
        dim: e.dim,
        window: run.window,
        negatives: e.negatives,
        epochs: e.epochs,
        lr: e.lr,
    };
    train_skipgram(walks, n, &config, &mut rng)
}

/// The vectors on a plane: themselves at dim 2, their two main directions (PCA) above.
fn project(z: &[Vec<f64>]) -> Vec<[f64; 2]> {
    let dim = z.first().map_or(0, |r| r.len());
    if dim <= 2 {
        return z
            .iter()
            .map(|r| {
                [
                    r.first().copied().unwrap_or(0.0),
                    r.get(1).copied().unwrap_or(0.0),
                ]
            })
            .collect();
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
    z.iter()
        .map(|r| {
            let c: Vec<f64> = r.iter().zip(&mean).map(|(a, m)| a - m).collect();
            [dot(&c, &first), dot(&c, &second)]
        })
        .collect()
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

// ------------------------------------------------------------------------------ app

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Link,
    Add,
    Label,
    Delete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Math,
    Step,
    Walks,
    Embedding,
}

/// What the q sweep was computed for: graph version, walk settings, p.
type SweepKey = (u64, RunSettings, f64);
/// One point of the q sweep: (q, same-label share of pairs, hops at the end).
type SweepPoint = (f64, f64, f64);
/// What the embedding was trained for: graph version, walk and embedding settings, p, q.
type EmbedKey = (u64, RunSettings, EmbedSettings, f64, f64);

struct Playground {
    preset: Preset,
    graph: Graph,
    /// Bumped at every change of links, drones or labels: everything is recomputed.
    version: u64,
    tool: Tool,
    brush: usize,
    link_from: Option<usize>,
    dragging: Option<usize>,
    hover: Option<usize>,
    p: f64,
    q: f64,
    builder: Builder,
    /// Space goes through each walk step in 10 small steps (else: one whole step).
    explain: bool,
    /// The Math tab scrolls to the current small step once after it changes.
    scroll_math: Cell<bool>,
    playing: bool,
    last_tick: f64,
    speed: f64,
    run: RunSettings,
    embed: EmbedSettings,
    auto_train: bool,
    show_heat: bool,
    tab: Tab,
    adj: Adjacency,
    hops: Vec<Vec<Option<usize>>>,
    built_for: Option<u64>,
    stats: Option<((u64, RunSettings, f64, f64), Stats)>,
    dw_stats: Option<((u64, RunSettings), Stats)>,
    sweep: Option<(SweepKey, Vec<SweepPoint>)>,
    embedding: Option<(EmbedKey, Embedding)>,
}

impl Default for Playground {
    fn default() -> Self {
        let preset = Preset::Barbell;
        let graph = Graph::build(preset);
        let adj = Adjacency::new(&graph.edges);
        Playground {
            preset,
            graph,
            version: 0,
            tool: Tool::Link,
            brush: 0,
            link_from: None,
            dragging: None,
            hover: None,
            p: 1.0,
            q: 1.0,
            builder: Builder::new(0, 1),
            explain: true,
            scroll_math: Cell::new(false),
            playing: false,
            last_tick: 0.0,
            speed: 3.0,
            run: RunSettings::default(),
            embed: EmbedSettings::default(),
            auto_train: true,
            show_heat: false,
            tab: Tab::Math,
            hops: hops(&adj),
            adj,
            built_for: Some(0),
            stats: None,
            dw_stats: None,
            sweep: None,
            embedding: None,
        }
    }
}

impl eframe::App for Playground {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.recompute();
        if ui.input(|i| i.key_pressed(Key::Space)) && self.graph.n() > 0 {
            self.next_input();
        }
        if self.playing && self.graph.n() > 0 {
            let now = ui.input(|i| i.time);
            if now - self.last_tick >= 1.0 / self.speed {
                self.last_tick = now;
                if self.walk_finished() {
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
        egui::Panel::bottom("walks")
            .resizable(true)
            .default_size(190.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.bottom(ui));
            });
        egui::CentralPanel::default().show(ui, |ui| self.canvas(ui));
    }
}

impl Playground {
    /// Space: the next small step (explain mode) or the next whole step.
    fn next_input(&mut self) {
        let (adj, p, q, length) = (&self.adj, self.p, self.q, self.run.length);
        if self.explain {
            self.builder.advance(adj, p, q, length);
        } else {
            self.builder.whole_step(adj, p, q, length);
        }
        self.scroll_math.set(true);
    }

    /// The walk is as long as it gets and no step is under way.
    fn walk_finished(&self) -> bool {
        self.builder.done(&self.adj, self.run.length)
            && self
                .builder
                .step
                .as_ref()
                .is_none_or(|s| s.phase == Phase::Move)
    }

    fn load(&mut self, preset: Preset) {
        self.preset = preset;
        self.graph = Graph::build(preset);
        self.changed();
        self.brush = 0;
    }

    /// The graph changed: new walk from a valid start, everything recomputed.
    fn changed(&mut self) {
        self.version += 1;
        self.link_from = None;
        let start = if self.builder.start < self.graph.n() {
            self.builder.start
        } else {
            0
        };
        self.builder = Builder::new(start, self.builder.seed);
    }

    /// Rebuilds what depends on the graph and settings, if they changed.
    fn recompute(&mut self) {
        if self.built_for != Some(self.version) {
            self.adj = Adjacency::new(&self.graph.edges);
            self.hops = hops(&self.adj);
            self.built_for = Some(self.version);
        }
        self.builder.refresh(&self.adj, self.p, self.q);
        if self.graph.n() == 0 {
            self.stats = None;
            self.dw_stats = None;
            self.sweep = None;
            self.embedding = None;
            return;
        }
        let key = (self.version, self.run, self.p, self.q);
        if self.stats.as_ref().is_none_or(|(k, _)| *k != key) {
            let s = stats(
                &self.graph,
                &self.adj,
                &self.hops,
                &self.run,
                self.p,
                self.q,
            );
            self.stats = Some((key, s));
        }
        let dw_key = (self.version, self.run);
        if self.dw_stats.as_ref().is_none_or(|(k, _)| *k != dw_key) {
            let s = stats(&self.graph, &self.adj, &self.hops, &self.run, 1.0, 1.0);
            self.dw_stats = Some((dw_key, s));
        }
        let sweep_key = (self.version, self.run, self.p);
        if self.tab == Tab::Walks && self.sweep.as_ref().is_none_or(|(k, _)| *k != sweep_key) {
            let points = Q_SWEEP
                .iter()
                .map(|&q| {
                    let s = stats(&self.graph, &self.adj, &self.hops, &self.run, self.p, q);
                    (q, s.same_label_pairs, s.end_distance())
                })
                .collect();
            self.sweep = Some((sweep_key, points));
        }
        let embed_key = (self.version, self.run, self.embed, self.p, self.q);
        let stale = self.embedding.as_ref().is_none_or(|(k, _)| *k != embed_key);
        if stale && self.auto_train {
            self.train_now();
        }
    }

    fn train_now(&mut self) {
        let (Some((_, n2v)), Some((_, dw))) = (&self.stats, &self.dw_stats) else {
            return;
        };
        let n = self.graph.n();
        let labels: Vec<usize> = self.graph.nodes.iter().map(|d| d.label).collect();
        let z = [
            train(&n2v.walks, n, &self.run, &self.embed),
            train(&dw.walks, n, &self.run, &self.embed),
        ];
        let points = [project(&z[0]), project(&z[1])];
        let same_label_nearest = [
            nearest_same_label(&z[0], &labels),
            nearest_same_label(&z[1], &labels),
        ];
        let key = (self.version, self.run, self.embed, self.p, self.q);
        self.embedding = Some((
            key,
            Embedding {
                z,
                points,
                same_label_nearest,
            },
        ));
    }

    fn label_color(&self, k: usize) -> Color32 {
        squad_color(k)
    }

    // ------------------------------------------------------------------ left panel

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Node2Vec playground");
        ui.weak("Biased random walks: how p and q change where a walk goes next.");
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
        });
        ui.weak(match self.tool {
            Tool::Link => "Click a drone, then another: adds the link, or removes it.",
            Tool::Add => "Click empty space: a new drone with the brush label.",
            Tool::Label => "Click a drone: it takes the brush label.",
            Tool::Delete => "Click a drone: deletes it and its links.",
        });
        ui.weak("Any tool: drag to move a drone, right-click to start the walk there.");
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
            "{} drones, {} links",
            self.graph.n(),
            self.graph.edges.edges.len()
        ));
        ui.separator();

        ui.label(RichText::new("Bias").strong());
        ui.add(
            egui::Slider::new(&mut self.p, 0.1..=10.0)
                .logarithmic(true)
                .text("p (return)"),
        );
        ui.add(
            egui::Slider::new(&mut self.q, 0.1..=10.0)
                .logarithmic(true)
                .text("q (in-out)"),
        );
        ui.horizontal_wrapped(|ui| {
            for (name, p, q) in [
                ("p = q = 1 (DeepWalk)", 1.0, 1.0),
                ("q = 0.25", self.p, 0.25),
                ("q = 4", self.p, 4.0),
                ("p = 0.25", 0.25, self.q),
                ("p = 4", 4.0, self.q),
            ] {
                if ui.small_button(name).clicked() {
                    self.p = p;
                    self.q = q;
                }
            }
        });
        self.sketch(ui);
        ui.separator();

        ui.label(RichText::new("Build one walk").strong());
        if self.graph.n() > 0 {
            ui.checkbox(&mut self.explain, "Explain each step in 10 small steps");
            ui.label(format!(
                "From drone {} · {} of {} drones",
                self.builder.start,
                self.builder.walk.len(),
                self.run.length
            ));
            if let Some(s) = &self.builder.step
                && s.phase < Phase::Move
            {
                ui.label(
                    RichText::new(format!(
                        "Small step {}/10: {}",
                        s.phase.number(),
                        s.phase.title()
                    ))
                    .color(CURRENT_RING),
                );
            }
            let finished = self.walk_finished();
            ui.horizontal_wrapped(|ui| {
                let label = if self.explain {
                    "Next small step (Space)"
                } else {
                    "Next step (Space)"
                };
                if ui
                    .add_enabled(!finished, egui::Button::new(label))
                    .clicked()
                {
                    self.next_input();
                }
                if self.explain
                    && ui
                        .add_enabled(!finished, egui::Button::new("Whole step"))
                        .clicked()
                {
                    self.builder
                        .whole_step(&self.adj, self.p, self.q, self.run.length);
                    self.scroll_math.set(true);
                }
                if ui
                    .add_enabled(!finished, egui::Button::new("Walk to the end"))
                    .clicked()
                {
                    self.builder
                        .finish(&self.adj, self.p, self.q, self.run.length);
                }
                if ui.button("New walk").clicked() {
                    self.builder = Builder::new(self.builder.start, self.builder.seed + 1);
                }
                let play = if self.playing {
                    "⏸ Pause"
                } else {
                    "▶ Play"
                };
                if ui.button(play).clicked() {
                    self.playing = !self.playing;
                }
            });
            let unit = if self.explain {
                "small steps / s"
            } else {
                "steps / s"
            };
            ui.add(
                egui::Slider::new(&mut self.speed, 0.5..=20.0)
                    .logarithmic(true)
                    .text(unit),
            );
            ui.weak("Right-click a drone on the graph to start the walk there.");
        } else {
            ui.weak("Add drones first.");
        }
        ui.separator();

        ui.label(RichText::new("Many walks").strong());
        egui::Grid::new("run").num_columns(2).show(ui, |ui| {
            ui.label("walks per drone");
            ui.add(egui::DragValue::new(&mut self.run.walks_per_node).range(1..=200));
            ui.end_row();
            ui.label("walk length");
            ui.add(egui::DragValue::new(&mut self.run.length).range(2..=80));
            ui.end_row();
            ui.label("window (pairs)");
            ui.add(egui::DragValue::new(&mut self.run.window).range(1..=10));
            ui.end_row();
            ui.label("seed");
            ui.add(egui::DragValue::new(&mut self.run.seed));
            ui.end_row();
        });
        ui.checkbox(
            &mut self.show_heat,
            "Map: where walks from the start drone go",
        );
        ui.separator();

        ui.label(RichText::new("Embedding").strong());
        egui::Grid::new("embed").num_columns(2).show(ui, |ui| {
            ui.label("dimension");
            ui.add(egui::DragValue::new(&mut self.embed.dim).range(2..=64));
            ui.end_row();
            ui.label("epochs");
            ui.add(egui::DragValue::new(&mut self.embed.epochs).range(1..=20));
            ui.end_row();
            ui.label("negatives");
            ui.add(egui::DragValue::new(&mut self.embed.negatives).range(1..=20));
            ui.end_row();
            ui.label("learning rate");
            ui.add(
                egui::DragValue::new(&mut self.embed.lr)
                    .range(0.001..=0.5)
                    .speed(0.001),
            );
            ui.end_row();
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.auto_train, "Train on every change");
            if !self.auto_train && ui.button("Train").clicked() {
                self.train_now();
            }
        });
    }

    /// The 3-drone sketch of the rule, with this p and q.
    fn sketch(&self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(vec2(270.0, 132.0), Sense::hover());
        let r = response.rect;
        let text = ui.visuals().text_color();
        let at = |x: f32, y: f32| pos2(r.left() + x, r.top() + y);
        let (t, v, x1, x2) = (
            at(40.0, 92.0),
            at(140.0, 92.0),
            at(90.0, 30.0),
            at(240.0, 92.0),
        );
        let link = Stroke::new(1.5, text.gamma_multiply(0.35));
        for (a, b) in [(t, v), (t, x1), (v, x1), (v, x2)] {
            painter.line_segment([a, b], link);
        }
        let total = 1.0 / self.p + 1.0 + 1.0 / self.q;
        for (to, kind) in [(t, Kind::Back), (x1, Kind::Near), (x2, Kind::Far)] {
            let share = kind.bias(self.p, self.q) / total;
            let dir = (to - v) * 0.62;
            painter.arrow(v, dir, Stroke::new(1.0 + 7.0 * share as f32, kind.color()));
            painter.text(
                v + dir * 0.55 + vec2(0.0, -12.0),
                Align2::CENTER_CENTER,
                format!("{} {:.0} %", kind.rule(), 100.0 * share),
                FontId::proportional(11.0),
                kind.color(),
            );
        }
        for (c, name, ring) in [
            (t, "t", Some(PREVIOUS_RING)),
            (v, "v", Some(CURRENT_RING)),
            (x1, "x1", None),
            (x2, "x2", None),
        ] {
            painter.circle_filled(c, 9.0, text.gamma_multiply(0.5));
            if let Some(col) = ring {
                painter.circle_stroke(c, 12.0, Stroke::new(2.0, col));
            }
            painter.text(
                c + vec2(0.0, 22.0),
                Align2::CENTER_CENTER,
                name,
                FontId::proportional(12.0),
                text,
            );
        }
        ui.weak(
            "The walk came from t and is on v. Each neighbour's link weight is multiplied: \
             back to t by 1/p, a drone also next to t (x1) by 1, a drone away from t (x2) by \
             1/q. Percentages here: all weights 1.",
        );
    }

    // ------------------------------------------------------------------------ canvas

    fn canvas(&mut self, ui: &mut egui::Ui) {
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
            self.builder = Builder::new(i, self.builder.seed);
        }
        if response.clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            let hit = near(p, &self.graph.nodes);
            match (self.tool, hit) {
                (Tool::Link, Some(b)) => match self.link_from {
                    Some(a) if a != b => {
                        self.graph.toggle(a, b);
                        self.changed();
                    }
                    Some(_) => self.link_from = None,
                    None => self.link_from = Some(b),
                },
                (Tool::Add, None) => {
                    self.graph.add(fit.world(p), self.brush);
                    self.changed();
                }
                (Tool::Label, Some(i)) => {
                    self.graph.nodes[i].label = self.brush;
                    self.changed();
                }
                (Tool::Delete, Some(i)) => {
                    self.graph.remove(i);
                    if self.builder.start >= i && self.builder.start > 0 {
                        self.builder.start -= 1;
                    }
                    self.changed();
                }
                _ => self.link_from = None,
            }
        }
        if self.version != self.built_for.unwrap_or(u64::MAX) {
            self.recompute();
        }
        let n = self.graph.n();
        if n == 0 {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Empty graph: pick the ➕ Add tool and click here to place drones.",
                FontId::proportional(15.0),
                text,
            );
            return;
        }
        let fit = Fit::new(&self.graph.nodes, rect.shrink(30.0));
        let at = |i: usize| fit.screen(self.graph.nodes[i].pos);

        // Links, width by weight.
        let max_w = self
            .graph
            .edges
            .edges
            .iter()
            .map(|e| e.weight)
            .fold(1e-12, f64::max);
        for e in &self.graph.edges.edges {
            let w = 0.8 + 2.2 * (e.weight / max_w).sqrt() as f32;
            painter.line_segment(
                [at(e.src as usize), at(e.dst as usize)],
                Stroke::new(w, text.gamma_multiply(0.22)),
            );
        }

        // Where walks from the start drone go (Node2Vec), as halos.
        if self.show_heat
            && let Some((_, s)) = &self.stats
        {
            let mut visits = vec![0.0; n];
            let mut total = 0.0;
            for w in s.walks.iter().filter(|w| w[0] == self.builder.start) {
                for &x in &w[1..] {
                    visits[x] += 1.0;
                    total += 1.0;
                }
            }
            let top = visits.iter().copied().fold(0.0, f64::max);
            if top > 0.0 {
                for (i, &v) in visits.iter().enumerate() {
                    if v > 0.0 {
                        let r = NODE_PX + 22.0 * (v / top).sqrt() as f32;
                        painter.circle_filled(at(i), r, CURRENT_RING.gamma_multiply(0.18));
                    }
                }
                painter.text(
                    rect.left_top() + vec2(8.0, 8.0),
                    Align2::LEFT_TOP,
                    format!(
                        "Halo: share of visits by walks from drone {} ({} steps in all)",
                        self.builder.start, total
                    ),
                    FontId::proportional(12.0),
                    text,
                );
            }
        }

        // The walk built so far, older steps fainter, with step numbers.
        let walk = &self.builder.walk;
        let len = walk.len();
        for (k, step) in walk.windows(2).enumerate() {
            let age = (len - 1 - k) as f32;
            let alpha = (1.0 - age / (len as f32 + 2.0)).max(0.25);
            let (a, b) = (at(step[0]), at(step[1]));
            painter.line_segment([a, b], Stroke::new(4.0, WALK_COLOR.gamma_multiply(alpha)));
            painter.text(
                a + (b - a) * 0.5 + vec2(0.0, -9.0),
                Align2::CENTER_CENTER,
                format!("{}", k + 1),
                FontId::proportional(11.0),
                WALK_COLOR,
            );
        }

        // The step being explained: its neighbours light up small step by small step.
        let r = if n > 50 { NODE_PX * 0.7 } else { NODE_PX };
        if let Some(s) = self
            .builder
            .step
            .as_ref()
            .filter(|s| s.phase >= Phase::Neighbours && s.phase < Phase::Move)
        {
            let total = s.total();
            let from = at(s.current);
            for (k, c) in s.options.iter().enumerate() {
                let to = at(c.to);
                let fade = if s.chosen.is_none_or(|x| x == k) {
                    1.0
                } else {
                    0.25
                };
                let color = if s.phase >= Phase::Kinds {
                    c.kind.color()
                } else {
                    text
                }
                .gamma_multiply(fade);
                if s.phase >= Phase::Weighted {
                    let share = (c.biased / total) as f32;
                    painter.arrow(
                        from,
                        (to - from) * 0.78,
                        Stroke::new(1.5 + 9.0 * share, color.gamma_multiply(0.9)),
                    );
                } else {
                    painter.line_segment([from, to], Stroke::new(2.5, color.gamma_multiply(0.7)));
                }
                painter.circle_stroke(to, r + 4.0, Stroke::new(2.0, color));
                let label = match s.phase {
                    Phase::Neighbours => format!("w = {}", num(c.weight)),
                    Phase::Kinds => c.kind.name().to_string(),
                    Phase::Biases => format!("{} = {}", c.kind.rule(), num(c.biased / c.weight)),
                    Phase::Weighted | Phase::Total => format!("π = {}", num(c.biased)),
                    _ => format!("{:.0} %", 100.0 * c.biased / total),
                };
                painter.text(
                    to + vec2(0.0, r + 24.0),
                    Align2::CENTER_CENTER,
                    label,
                    FontId::proportional(12.5),
                    color,
                );
            }
        }

        // Drones.
        for i in 0..n {
            let node = &self.graph.nodes[i];
            let c = at(i);
            draw_drone(
                &painter,
                c,
                r,
                node.role.unwrap_or(Role::Recon),
                self.label_color(node.label),
                Stroke::new(1.0, Color32::from_black_alpha(160)),
            );
            if n <= 50 {
                painter.text(
                    c + vec2(r + 2.0, -r),
                    Align2::LEFT_BOTTOM,
                    format!("{i}"),
                    FontId::proportional(11.0),
                    visuals.weak_text_color(),
                );
            }
        }
        let (shown_t, shown_v) = self.builder.shown();
        if let Some(t) = shown_t {
            painter.circle_stroke(at(t), r + 5.0, Stroke::new(2.5, PREVIOUS_RING));
            painter.text(
                at(t) + vec2(0.0, r + 11.0),
                Align2::CENTER_CENTER,
                "t",
                FontId::proportional(13.0),
                PREVIOUS_RING,
            );
        }
        painter.circle_stroke(at(shown_v), r + 8.0, Stroke::new(3.0, CURRENT_RING));
        painter.circle_stroke(
            at(self.builder.start),
            r + 3.0,
            Stroke::new(1.5, WALK_COLOR),
        );
        if let Some(a) = self.link_from {
            painter.circle_stroke(at(a), r + 12.0, Stroke::new(1.5, text));
            painter.text(
                rect.left_bottom() + vec2(8.0, -8.0),
                Align2::LEFT_BOTTOM,
                format!("Drone {a} picked: click another drone to add or remove the link."),
                FontId::proportional(12.0),
                text,
            );
        }
        if let Some(i) = self.hover {
            painter.circle_stroke(at(i), r + 2.0, Stroke::new(2.0, text));
            let node = &self.graph.nodes[i];
            let degree = self.adj.neighbours.get(i).map_or(0, Vec::len);
            response.on_hover_text_at_pointer(format!(
                "Drone {i} · label {} · {degree} links",
                self.graph.label_name(node.label)
            ));
        }
    }

    // ------------------------------------------------------------------ right panel

    fn right(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Math, "Math");
            ui.selectable_value(&mut self.tab, Tab::Step, "This step");
            ui.selectable_value(&mut self.tab, Tab::Walks, "Many walks");
            ui.selectable_value(&mut self.tab, Tab::Embedding, "Embedding");
        });
        ui.separator();
        if self.graph.n() == 0 {
            ui.weak("Add drones and links first.");
            return;
        }
        egui::ScrollArea::vertical().show(ui, |ui| match self.tab {
            Tab::Math => self.math_tab(ui),
            Tab::Step => self.step_tab(ui),
            Tab::Walks => self.walks_tab(ui),
            Tab::Embedding => self.embedding_tab(ui),
        });
    }

    fn step_tab(&self, ui: &mut egui::Ui) {
        let b = &self.builder;
        match &b.step {
            Some(s) if s.phase < Phase::Move => {
                ui.label(
                    RichText::new(format!(
                        "The step under way (small step {}/10)",
                        s.phase.number()
                    ))
                    .strong(),
                );
                self.options_table(ui, "now", s.previous, s.current, &s.options, s.u, s.chosen);
            }
            _ if b.done(&self.adj, self.run.length) => {
                ui.label(RichText::new("The walk is finished.").strong());
                ui.weak("New walk, or right-click a drone to start from it.");
            }
            _ => {
                ui.label(RichText::new("Press Space for the next step.").strong());
            }
        }
        if let Some(last) = &b.last {
            ui.add_space(10.0);
            ui.label(RichText::new("The step just taken").strong());
            self.options_table(
                ui,
                "last",
                last.previous,
                last.from,
                &last.options,
                Some(last.u),
                Some(last.chosen),
            );
        }
    }

    /// The table of one step, the chances as one bar, and where u fell (once drawn).
    #[allow(clippy::too_many_arguments)]
    fn options_table(
        &self,
        ui: &mut egui::Ui,
        id: &str,
        previous: Option<usize>,
        current: usize,
        options: &[Choice],
        u: Option<f64>,
        chosen: Option<usize>,
    ) {
        ui.label(match previous {
            None => {
                format!("On drone {current}, first step: no t, every bias is 1 (plain DeepWalk).")
            }
            Some(t) => format!(
                "On drone v = {current}, came from t = {t}. p = {:.2}, q = {:.2}.",
                self.p, self.q
            ),
        });
        let total: f64 = options.iter().map(|c| c.biased).sum();
        let plain: f64 = options.iter().map(|c| c.weight).sum();
        egui::Grid::new(("options", id))
            .striped(true)
            .show(ui, |ui| {
                for h in [
                    "to",
                    "weight",
                    "where",
                    "bias",
                    "w × bias",
                    "Node2Vec",
                    "DeepWalk",
                ] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for (k, c) in options.iter().enumerate() {
                    let picked = chosen == Some(k);
                    let to = format!("{}{}", c.to, if picked { "  ◀" } else { "" });
                    ui.label(if picked {
                        RichText::new(to).strong()
                    } else {
                        RichText::new(to)
                    });
                    ui.label(num(c.weight));
                    ui.colored_label(c.kind.color(), c.kind.name());
                    ui.label(c.kind.rule());
                    ui.label(num(c.biased));
                    ui.label(format!("{:.1} %", 100.0 * c.biased / total));
                    ui.label(format!("{:.1} %", 100.0 * c.weight / plain));
                    ui.end_row();
                }
            });
        chance_bar(ui, options, u, chosen);
        if let (Some(u), Some(k)) = (u, chosen) {
            ui.weak(format!(
                "Random number u = {} out of Z = {}: it falls in drone {}'s slice.",
                num(u),
                num(total),
                options[k].to
            ));
        }
    }

    /// Every small step reached so far, as a formula with this walk's numbers.
    fn math_tab(&self, ui: &mut egui::Ui) {
        let Some(s) = &self.builder.step else {
            if self.walk_finished() {
                ui.label(RichText::new("The walk is finished.").strong());
                ui.weak("New walk, or right-click a drone to start from it.");
            } else {
                ui.label(RichText::new("Press Space to start the next step.").strong());
                ui.weak(
                    "With \"Explain each step in 10 small steps\" on, each Space adds one small \
                     step here, with this walk's numbers, and lights it up on the graph.",
                );
            }
            return;
        };
        ui.label(
            RichText::new(format!(
                "Step {} of the walk, from drone {}",
                self.builder.walk.len() - usize::from(s.phase == Phase::Move),
                s.current
            ))
            .strong(),
        );
        let text = ui.visuals().text_color();
        let faint = ui.visuals().weak_text_color().gamma_multiply(0.5);
        for phase in Phase::ALL.into_iter().filter(|&ph| ph <= s.phase) {
            let current = phase == s.phase;
            let frame = egui::Frame::group(ui.style()).stroke(Stroke::new(
                if current { 2.0 } else { 1.0 },
                if current { CURRENT_RING } else { faint },
            ));
            let response = frame
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new(format!("{}. {}", phase.number(), phase.title()))
                            .strong()
                            .color(if current { CURRENT_RING } else { text }),
                    );
                    self.math_step(ui, s, phase);
                })
                .response;
            if current && self.scroll_math.replace(false) {
                response.scroll_to_me(Some(egui::Align::Center));
            }
        }
    }

    /// One small step's formula and numbers.
    fn math_step(&self, ui: &mut egui::Ui, s: &Explained, phase: Phase) {
        let mono = |ui: &mut egui::Ui, t: String| {
            ui.label(RichText::new(t).monospace());
        };
        let colored = |ui: &mut egui::Ui, c: Color32, t: String| {
            ui.label(RichText::new(t).monospace().color(c));
        };
        let v = s.current;
        let total = s.total();
        match phase {
            Phase::Where => {
                let walk = &self.builder.walk;
                let so_far = &walk[..walk.len() - usize::from(s.phase == Phase::Move)];
                match s.previous {
                    None => {
                        mono(ui, format!("v = {v}    t = none (first step)"));
                        ui.weak(
                            "The walk has just started at v. With no t there is nothing to \
                             compare with: every bias is 1, this step is plain DeepWalk.",
                        );
                    }
                    Some(t) => {
                        mono(ui, format!("t = {t}  →  v = {v}"));
                        ui.weak(
                            "The walk is on v and came from t. Node2Vec remembers one step \
                             back: every neighbour of v is judged by where it lies from t.",
                        );
                    }
                }
                mono(ui, format!("walk so far: {}", walk_str(so_far)));
            }
            Phase::Neighbours => {
                let ids: Vec<String> = s.options.iter().map(|c| c.to.to_string()).collect();
                mono(ui, format!("N(v) = {{{}}}", ids.join(", ")));
                for c in &s.options {
                    mono(ui, format!("w(v, {}) = {}", c.to, num(c.weight)));
                }
                ui.weak(
                    "w is the link weight (messages on the swarm graphs, 1 elsewhere). The order \
                     is the order the links were added: it only decides where each drone sits \
                     on the bar of small step 9.",
                );
            }
            Phase::Kinds => {
                mono(
                    ui,
                    "d(t, x) = 0 if x = t,  1 if x is next to t,  2 otherwise".into(),
                );
                for c in &s.options {
                    let x = c.to;
                    let line = match (c.kind, s.previous) {
                        (Kind::First, _) | (_, None) => format!("x = {x}: no t → first step"),
                        (Kind::Back, _) => format!("x = {x}: x = t → d = 0, back to t"),
                        (Kind::Near, Some(t)) => {
                            format!("x = {x}: x ≠ t, link {t}–{x} exists → d = 1, next to t")
                        }
                        (Kind::Far, Some(t)) => {
                            format!("x = {x}: x ≠ t, no link {t}–{x} → d = 2, away from t")
                        }
                    };
                    colored(ui, c.kind.color(), line);
                }
            }
            Phase::Biases => {
                mono(
                    ui,
                    "α(t, x) = 1/p if d = 0,  1 if d = 1,  1/q if d = 2".into(),
                );
                mono(
                    ui,
                    format!(
                        "p = {}, q = {}  →  1/p = {}, 1/q = {}",
                        num(self.p),
                        num(self.q),
                        num(1.0 / self.p),
                        num(1.0 / self.q)
                    ),
                );
                for c in &s.options {
                    let alpha = c.biased / c.weight;
                    let line = match c.kind {
                        Kind::First => format!("α({}) = 1 (first step)", c.to),
                        Kind::Near => format!("α({}) = 1", c.to),
                        Kind::Back => format!("α({}) = 1/p = {}", c.to, num(alpha)),
                        Kind::Far => format!("α({}) = 1/q = {}", c.to, num(alpha)),
                    };
                    colored(ui, c.kind.color(), line);
                }
                if s.u.is_none() {
                    ui.weak("Move the p and q sliders now: the numbers follow until the draw.");
                }
            }
            Phase::Weighted => {
                mono(ui, "π(x) = w(v, x) · α(t, x)".into());
                for c in &s.options {
                    colored(
                        ui,
                        c.kind.color(),
                        format!(
                            "π({}) = {} × {} = {}",
                            c.to,
                            num(c.weight),
                            num(c.biased / c.weight),
                            num(c.biased)
                        ),
                    );
                }
                ui.weak("On the graph, each arrow is now as wide as its π.");
            }
            Phase::Total => {
                let terms: Vec<String> = s.options.iter().take(8).map(|c| num(c.biased)).collect();
                let more = if s.options.len() > 8 { " + …" } else { "" };
                mono(
                    ui,
                    format!("Z = Σ π(x) = {}{more} = {}", terms.join(" + "), num(total)),
                );
                ui.weak(
                    "The π don't need to add up to 1: dividing by Z (next) turns them into \
                     chances, and the draw (small step 8) only ever needs Z.",
                );
            }
            Phase::Chances => {
                let plain: f64 = s.options.iter().map(|c| c.weight).sum();
                mono(
                    ui,
                    "P(next = x) = π(x) / Z        DeepWalk: w(v, x) / Σ w".into(),
                );
                for c in &s.options {
                    colored(
                        ui,
                        c.kind.color(),
                        format!(
                            "P({}) = {} / {} = {:5.1} %     DeepWalk {:5.1} %",
                            c.to,
                            num(c.biased),
                            num(total),
                            100.0 * c.biased / total,
                            100.0 * c.weight / plain
                        ),
                    );
                }
                chance_bar(ui, &s.options, None, None);
            }
            Phase::Draw => {
                let u = s.u.unwrap_or(0.0);
                mono(ui, format!("u = rng.random_range(0.0..Z) = {}", num(u)));
                ui.weak(format!(
                    "One random number per step, uniform between 0 and Z = {}. The same call as \
                     random_walks: with p = q = 1 every π is the link weight, so the same u \
                     picks the same drone as DeepWalk.",
                    num(total)
                ));
                chance_bar(ui, &s.options, s.u, None);
            }
            Phase::Slice => {
                let (Some(u), Some(chosen)) = (s.u, s.chosen) else {
                    return;
                };
                mono(ui, format!("u = {}", num(u)));
                let mut left = u;
                for (k, c) in s.options.iter().enumerate() {
                    let after = left - c.biased;
                    if k == chosen {
                        colored(
                            ui,
                            c.kind.color(),
                            format!(
                                "{} − π({}) = {} − {} = {} < 0  →  next = {}",
                                num(left),
                                c.to,
                                num(left),
                                num(c.biased),
                                num(after),
                                c.to
                            ),
                        );
                        if after >= 0.0 {
                            ui.weak("(rounding left u just above 0: the last drone takes it)");
                        }
                        break;
                    }
                    mono(
                        ui,
                        format!(
                            "{} − π({}) = {} − {} = {} ≥ 0  →  not {}, go on",
                            num(left),
                            c.to,
                            num(left),
                            num(c.biased),
                            num(after),
                            c.to
                        ),
                    );
                    left = after;
                }
                ui.weak(
                    "Subtracting until u goes below 0 = finding which slice of the bar u is in.",
                );
                chance_bar(ui, &s.options, Some(u), Some(chosen));
            }
            Phase::Move => {
                let Some(chosen) = s.chosen else {
                    return;
                };
                let x = s.options[chosen].to;
                mono(
                    ui,
                    format!("walk.push({x});  previous = Some({v});  current = {x};"),
                );
                mono(ui, format!("walk = {}", walk_str(&self.builder.walk)));
                ui.weak(if self.builder.done(&self.adj, self.run.length) {
                    "The walk has all its drones: finished.".to_string()
                } else {
                    format!("Next step: t = {v}, v = {x}. Space starts it.")
                });
            }
        }
    }

    fn walks_tab(&self, ui: &mut egui::Ui) {
        let (Some((_, s)), Some((_, d))) = (&self.stats, &self.dw_stats) else {
            return;
        };
        ui.label(
            RichText::new(format!(
                "{} walks of up to {} drones, Node2Vec (p {:.2}, q {:.2}) against DeepWalk",
                s.walks.len(),
                self.run.length,
                self.p,
                self.q
            ))
            .strong(),
        );
        egui::Grid::new("compare").striped(true).show(ui, |ui| {
            ui.label("");
            ui.colored_label(N2V_COLOR, "Node2Vec");
            ui.colored_label(DW_COLOR, "DeepWalk");
            ui.end_row();
            let pct = |x: f64| format!("{:.1} %", 100.0 * x);
            for kind in [Kind::Back, Kind::Near, Kind::Far] {
                ui.colored_label(kind.color(), format!("steps {}", kind.name()));
                ui.label(pct(s.share(kind)));
                ui.label(pct(d.share(kind)));
                ui.end_row();
            }
            ui.label("steps that change label");
            ui.label(pct(s.label_changes));
            ui.label(pct(d.label_changes));
            ui.end_row();
            ui.label("training pairs with the same label");
            ui.label(pct(s.same_label_pairs));
            ui.label(pct(d.same_label_pairs));
            ui.end_row();
            ui.label("different drones per walk");
            ui.label(format!("{:.1}", s.distinct));
            ui.label(format!("{:.1}", d.distinct));
            ui.end_row();
            ui.label("hops from the start at the end");
            ui.label(format!("{:.2}", s.end_distance()));
            ui.label(format!("{:.2}", d.end_distance()));
            ui.end_row();
        });
        ui.weak(
            "Training pairs with the same label: of all (u, v) pairs skip-gram trains on \
             (v within the window of u), the share where u and v share a label. High = the \
             vectors will group by label.",
        );
        ui.add_space(8.0);
        ui.label(RichText::new("How far the walks get").strong());
        let xs: Vec<f64> = (0..s.distance.len().max(d.distance.len()))
            .map(|k| k as f64)
            .collect();
        line_plot(
            ui,
            "mean hops from the start drone, after k steps",
            &xs,
            &[
                ("Node2Vec", N2V_COLOR, &s.distance),
                ("DeepWalk", DW_COLOR, &d.distance),
            ],
            false,
            None,
        );
        if let Some((_, points)) = &self.sweep {
            ui.add_space(8.0);
            ui.label(RichText::new(format!("q sweep (p = {:.2})", self.p)).strong());
            let qs: Vec<f64> = points.iter().map(|p| p.0).collect();
            let same: Vec<f64> = points.iter().map(|p| p.1).collect();
            let far: Vec<f64> = points.iter().map(|p| p.2).collect();
            line_plot(
                ui,
                "training pairs with the same label",
                &qs,
                &[("", N2V_COLOR, &same)],
                true,
                Some(self.q),
            );
            line_plot(
                ui,
                "hops from the start at the end",
                &qs,
                &[("", N2V_COLOR, &far)],
                true,
                Some(self.q),
            );
        }
    }

    fn embedding_tab(&mut self, ui: &mut egui::Ui) {
        if !self.auto_train && ui.button("Train").clicked() {
            self.train_now();
        }
        let Some((key, e)) = &self.embedding else {
            ui.weak("Not trained yet.");
            return;
        };
        let current = (self.version, self.run, self.embed, self.p, self.q);
        if *key != current {
            ui.colored_label(BACK_COLOR, "Out of date: press Train.");
        }
        let n = self.graph.n();
        ui.label(format!(
            "skip-gram on the walks of the other tab (dim {}, window {}, {} epochs). Each \
             drone's most similar other vector (cosine) has the same label:",
            self.embed.dim, self.run.window, self.embed.epochs
        ));
        ui.horizontal(|ui| {
            ui.colored_label(
                N2V_COLOR,
                format!("Node2Vec {}/{n}", e.same_label_nearest[0]),
            );
            ui.colored_label(
                DW_COLOR,
                format!("DeepWalk {}/{n}", e.same_label_nearest[1]),
            );
        });
        ui.weak(if self.embed.dim > 2 {
            "Pictures: the two main directions of the vectors (PCA)."
        } else {
            "Pictures: the vectors themselves (dim 2)."
        });
        let labels: Vec<usize> = self.graph.nodes.iter().map(|d| d.label).collect();
        let w = ui.available_width();
        for (k, name) in [
            (0, format!("Node2Vec (p {:.2}, q {:.2})", self.p, self.q)),
            (1, "DeepWalk".to_string()),
        ] {
            ui.label(RichText::new(name).strong());
            scatter(
                ui,
                &e.points[k],
                &labels,
                vec2(w.min(440.0), 230.0),
                |l| self.label_color(l),
                self.hover,
            );
        }
        let _ = &e.z;
    }

    // ----------------------------------------------------------------- bottom panel

    fn bottom(&self, ui: &mut egui::Ui) {
        if self.graph.n() == 0 {
            return;
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new("The walk being built").strong());
            ui.weak("chips coloured by label; the arrow before each says what kind of step it was");
        });
        self.chips(ui, &self.builder.walk);
        let (Some((_, s)), Some((_, d))) = (&self.stats, &self.dw_stats) else {
            return;
        };
        let start = self.builder.start;
        let n = self.graph.n();
        for (name, color, st) in [("Node2Vec", N2V_COLOR, s), ("DeepWalk", DW_COLOR, d)] {
            ui.add_space(4.0);
            ui.colored_label(
                color,
                format!("{name}: walks from drone {start} (Many walks settings)"),
            );
            for r in 0..SAMPLE_WALKS.min(self.run.walks_per_node) {
                if let Some(w) = st.walks.get(r * n + start) {
                    self.chips(ui, w);
                }
            }
        }
    }

    /// A walk as a row of label-coloured chips, each step's kind as its arrow.
    fn chips(&self, ui: &mut egui::Ui, walk: &[usize]) {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (t, &x) in walk.iter().enumerate() {
                if t >= 1 {
                    let kind = self.adj.kind((t >= 2).then(|| walk[t - 2]), x);
                    let arrow = if kind == Kind::Back { "↩" } else { "→" };
                    ui.colored_label(kind.color(), arrow);
                }
                ui.label(
                    RichText::new(format!(" {x} "))
                        .background_color(self.label_color(self.graph.nodes[x].label))
                        .color(Color32::BLACK)
                        .monospace(),
                );
            }
        });
    }
}

// ------------------------------------------------------------------------- drawing

/// A number for the Math tab: whole numbers as they are, others with 3 decimals.
fn num(x: f64) -> String {
    if x == x.round() && x.abs() < 1e9 {
        format!("{x:.0}")
    } else {
        format!("{x:.3}")
    }
}

/// A walk as [a, b, c], only its last 14 drones if it is longer.
fn walk_str(walk: &[usize]) -> String {
    let skip = walk.len().saturating_sub(14);
    let ids: Vec<String> = walk[skip..].iter().map(|x| x.to_string()).collect();
    format!("[{}{}]", if skip > 0 { "…, " } else { "" }, ids.join(", "))
}

/// The chances as slices of one bar (in the order of the options, as the draw sees them);
/// `u` marked once drawn, the `chosen` slice outlined.
fn chance_bar(ui: &mut egui::Ui, options: &[Choice], u: Option<f64>, chosen: Option<usize>) {
    let total: f64 = options.iter().map(|c| c.biased).sum();
    let (response, painter) =
        ui.allocate_painter(vec2(ui.available_width().min(440.0), 38.0), Sense::hover());
    let r = response.rect.shrink2(vec2(2.0, 8.0));
    let text = ui.visuals().text_color();
    if total <= 0.0 {
        return;
    }
    let mut x = r.left();
    for (k, c) in options.iter().enumerate() {
        let w = r.width() * (c.biased / total) as f32;
        let slice = Rect::from_min_size(pos2(x, r.top()), vec2(w, r.height()));
        let fade = if chosen.is_none_or(|ch| ch == k) {
            0.85
        } else {
            0.3
        };
        painter.rect_filled(slice.shrink(0.5), 1.0, c.kind.color().gamma_multiply(fade));
        if chosen == Some(k) {
            painter.rect_stroke(
                slice.expand(1.5),
                1.0,
                Stroke::new(2.0, text),
                egui::StrokeKind::Outside,
            );
        }
        if w > 16.0 {
            painter.text(
                slice.center(),
                Align2::CENTER_CENTER,
                c.to.to_string(),
                FontId::proportional(11.0),
                Color32::BLACK,
            );
        }
        x += w;
    }
    painter.text(
        pos2(r.left(), r.bottom() + 1.0),
        Align2::LEFT_TOP,
        "0",
        FontId::proportional(9.0),
        text,
    );
    painter.text(
        pos2(r.right(), r.bottom() + 1.0),
        Align2::RIGHT_TOP,
        format!("Z = {}", num(total)),
        FontId::proportional(9.0),
        text,
    );
    if let Some(u) = u {
        let ux = r.left() + r.width() * (u / total) as f32;
        painter.line_segment(
            [pos2(ux, r.top() - 6.0), pos2(ux, r.bottom() + 6.0)],
            Stroke::new(2.5, text),
        );
        painter.text(
            pos2(ux, r.top() - 7.0),
            Align2::CENTER_BOTTOM,
            "u",
            FontId::proportional(10.0),
            text,
        );
    }
}

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

/// Points coloured by label; `hover` (a drone hovered on the graph) is ringed.
fn scatter(
    ui: &mut egui::Ui,
    points: &[[f64; 2]],
    labels: &[usize],
    size: egui::Vec2,
    color: impl Fn(usize) -> Color32,
    hover: Option<usize>,
) {
    let (response, painter) = ui.allocate_painter(size, Sense::hover());
    let rect = response.rect;
    let visuals = ui.visuals();
    painter.rect_filled(rect, 2.0, visuals.extreme_bg_color);
    if points.is_empty() {
        return;
    }
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for p in points {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let inner = rect.shrink(14.0);
    let span = ((hi[0] - lo[0]).max(hi[1] - lo[1])).max(1e-9);
    let mid = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
    let s = (inner.width().min(inner.height()) as f64) / span;
    let at = |p: &[f64; 2]| {
        inner.center() + vec2(((p[0] - mid[0]) * s) as f32, (-(p[1] - mid[1]) * s) as f32)
    };
    let mut hovered = None;
    for (i, p) in points.iter().enumerate() {
        let c = at(p);
        painter.circle_filled(c, 5.0, color(labels[i]));
        painter.circle_stroke(c, 5.0, Stroke::new(1.0, Color32::from_black_alpha(160)));
        if points.len() <= 40 {
            painter.text(
                c + vec2(6.0, -6.0),
                Align2::LEFT_BOTTOM,
                i.to_string(),
                FontId::proportional(10.0),
                visuals.weak_text_color(),
            );
        }
        if response.hover_pos().is_some_and(|h| h.distance(c) < 7.0) {
            hovered = Some(i);
        }
    }
    if let Some(i) = hover.filter(|&i| i < points.len()) {
        painter.circle_stroke(at(&points[i]), 9.0, Stroke::new(2.0, visuals.text_color()));
    }
    if let Some(i) = hovered {
        response.on_hover_text(format!("Drone {i}"));
    }
}

/// A small line plot: x values, one or more series; optional log x and a marker line.
fn line_plot(
    ui: &mut egui::Ui,
    title: &str,
    xs: &[f64],
    series: &[(&str, Color32, &Vec<f64>)],
    log_x: bool,
    marker: Option<f64>,
) {
    let size = vec2(ui.available_width().min(440.0), 150.0);
    let (response, painter) = ui.allocate_painter(size, Sense::hover());
    let rect = response.rect;
    let visuals = ui.visuals();
    let text = visuals.text_color();
    let plot = Rect::from_min_max(rect.min + vec2(38.0, 18.0), rect.max - vec2(8.0, 20.0));
    painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
    painter.text(
        rect.left_top() + vec2(38.0, 2.0),
        Align2::LEFT_TOP,
        title,
        FontId::proportional(11.0),
        text,
    );
    if xs.len() < 2 {
        return;
    }
    let fx = |x: f64| if log_x { x.log2() } else { x };
    let (x0, x1) = (fx(xs[0]), fx(xs[xs.len() - 1]));
    let y_max = series
        .iter()
        .flat_map(|s| s.2.iter().copied())
        .fold(0.0_f64, f64::max)
        .max(1e-9)
        * 1.08;
    let to = |x: f64, y: f64| {
        pos2(
            plot.left() + plot.width() * ((fx(x) - x0) / (x1 - x0).max(1e-12)) as f32,
            plot.bottom() - plot.height() * (y / y_max) as f32,
        )
    };
    for k in 0..=4 {
        let y = y_max * k as f64 / 4.0;
        let p = to(xs[0], y);
        painter.line_segment(
            [pos2(plot.left(), p.y), pos2(plot.right(), p.y)],
            Stroke::new(1.0, text.gamma_multiply(0.1)),
        );
        painter.text(
            pos2(plot.left() - 4.0, p.y),
            Align2::RIGHT_CENTER,
            format!("{y:.2}"),
            FontId::proportional(10.0),
            text,
        );
    }
    for &x in xs {
        let p = to(x, 0.0);
        let label = if log_x && x < 1.0 {
            format!("1/{}", (1.0 / x).round())
        } else {
            format!("{x}")
        };
        if log_x || xs.len() <= 12 || (x as usize).is_multiple_of(5) {
            painter.text(
                pos2(p.x, plot.bottom() + 3.0),
                Align2::CENTER_TOP,
                label,
                FontId::proportional(10.0),
                text,
            );
        }
    }
    if let Some(m) = marker {
        let p = to(m.clamp(xs[0], xs[xs.len() - 1]), 0.0);
        painter.line_segment(
            [pos2(p.x, plot.top()), pos2(p.x, plot.bottom())],
            Stroke::new(1.5, CURRENT_RING),
        );
    }
    let mut legend_x = plot.right();
    for (name, color, ys) in series.iter().rev() {
        let pts: Vec<Pos2> = xs.iter().zip(ys.iter()).map(|(&x, &y)| to(x, y)).collect();
        painter.add(egui::Shape::line(pts.clone(), Stroke::new(2.0, *color)));
        for p in pts {
            painter.circle_filled(p, 2.5, *color);
        }
        if !name.is_empty() {
            let r = painter.text(
                pos2(legend_x, rect.top() + 2.0),
                Align2::RIGHT_TOP,
                *name,
                FontId::proportional(11.0),
                *color,
            );
            legend_x = r.left() - 10.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerie_classic::embed::walks::random_walks;

    /// The 3-drone sketch: t–v, t–x1, v–x1, v–x2 (x1 = 2, x2 = 3), weights 1, 2, 3, 4.
    fn sketch() -> Graph {
        let mut g = Graph::empty();
        for k in 0..4 {
            g.add([k as f32, 0.0], 0);
        }
        g.link(0, 1, 1.0); // t–v
        g.link(0, 2, 2.0); // t–x1
        g.link(1, 2, 3.0); // v–x1
        g.link(1, 3, 4.0); // v–x2
        g
    }

    #[test]
    fn the_sketch_gets_back_near_and_far() {
        let g = sketch();
        let adj = Adjacency::new(&g.edges);
        let options = adj.options(Some(0), 1, 2.0, 0.5);
        let by_to = |x: usize| *options.iter().find(|c| c.to == x).unwrap();
        assert_eq!(by_to(0).kind, Kind::Back);
        assert_eq!(by_to(2).kind, Kind::Near);
        assert_eq!(by_to(3).kind, Kind::Far);
        assert_eq!(by_to(0).biased, 1.0 / 2.0); // weight 1, × 1/p
        assert_eq!(by_to(2).biased, 3.0); // weight 3, × 1
        assert_eq!(by_to(3).biased, 4.0 / 0.5); // weight 4, × 1/q
        assert!(
            adj.options(None, 1, 2.0, 0.5)
                .iter()
                .all(|c| c.kind == Kind::First && c.biased == c.weight)
        );
    }

    #[test]
    fn with_p_and_q_at_1_the_walks_are_random_walks_exactly() {
        for preset in Preset::ALL {
            let g = Graph::build(preset);
            let adj = Adjacency::new(&g.edges);
            let ours = biased_walks(&adj, 3, 12, 1.0, 1.0, &mut walk_rng(4));
            let theirs = random_walks(&g.edges, 3, 12, &mut walk_rng(4));
            assert_eq!(ours, theirs, "{}", preset.name());
        }
    }

    #[test]
    fn the_walks_are_aerie_classics_biased_walks_exactly() {
        // The playground's walker against the real one, for several p and q.
        for preset in Preset::ALL {
            let g = Graph::build(preset);
            let adj = Adjacency::new(&g.edges);
            for (p, q) in [(1.0, 1.0), (0.5, 3.0), (4.0, 0.25)] {
                let ours = biased_walks(&adj, 2, 12, p, q, &mut walk_rng(8));
                let real = aerie_classic::embed::node2vec::biased_walks(
                    &g.edges,
                    2,
                    12,
                    p,
                    q,
                    &mut walk_rng(8),
                );
                assert_eq!(ours, real, "{} p {p} q {q}", preset.name());
            }
        }
    }

    #[test]
    fn a_huge_p_never_steps_back_and_a_small_q_goes_further_on_a_ring() {
        let g = Graph::build(Preset::Ring);
        let adj = Adjacency::new(&g.edges);
        let h = hops(&adj);
        let run = RunSettings::default();
        let never_back = stats(&g, &adj, &h, &run, 1e12, 1.0);
        assert_eq!(never_back.kinds[Kind::Back as usize], 0);
        // On a ring, a walk that never turns back moves one hop per step (up to 8).
        assert_eq!(never_back.distance[5], 5.0);
        let out = stats(&g, &adj, &h, &run, 1.0, 0.25);
        let home = stats(&g, &adj, &h, &run, 1.0, 4.0);
        assert!(
            out.end_distance() > home.end_distance() + 1.0,
            "{} vs {}",
            out.end_distance(),
            home.end_distance()
        );
        assert!(out.share(Kind::Far) > 0.75 && home.share(Kind::Back) > 0.75);
    }

    #[test]
    fn ten_small_steps_take_exactly_one_step() {
        let g = Graph::build(Preset::Barbell);
        let adj = Adjacency::new(&g.edges);
        let mut b = Builder::new(0, 3);
        let phase = |b: &Builder| b.step.as_ref().unwrap().phase;
        b.advance(&adj, 1.0, 1.0, 5);
        assert_eq!(phase(&b), Phase::Where);
        for _ in 0..8 {
            b.advance(&adj, 1.0, 1.0, 5);
        }
        assert_eq!(phase(&b), Phase::Slice);
        assert_eq!(b.walk.len(), 1, "no move before small step 10");
        b.advance(&adj, 1.0, 1.0, 5);
        assert_eq!(phase(&b), Phase::Move);
        assert_eq!(b.walk.len(), 2);
        let s = b.step.as_ref().unwrap();
        assert_eq!(b.walk[1], s.options[s.chosen.unwrap()].to);
        assert!(adj.linked.contains(&(b.walk[0], b.walk[1])));
        // The next Space starts the next step from where the walk now is.
        b.advance(&adj, 1.0, 1.0, 5);
        let s = b.step.as_ref().unwrap();
        assert_eq!(s.phase, Phase::Where);
        assert_eq!((s.previous, s.current), (Some(0), b.walk[1]));
    }

    #[test]
    fn small_steps_whole_steps_and_biased_walks_build_the_same_walk() {
        let g = Graph::build(Preset::WheelsByRole);
        let adj = Adjacency::new(&g.edges);
        let (p, q, length) = (0.5, 3.0, 15);
        // biased_walks with 1 walk per drone: its first walk starts at drone 0 and uses the
        // RNG first.
        let reference = biased_walks(&adj, 1, length, p, q, &mut walk_rng(9))[0].clone();
        let mut small = Builder::new(0, 0);
        small.rng = walk_rng(9);
        while !(small.done(&adj, length)
            && small.step.as_ref().is_none_or(|s| s.phase == Phase::Move))
        {
            small.advance(&adj, p, q, length);
        }
        let mut whole = Builder::new(0, 0);
        whole.rng = walk_rng(9);
        whole.finish(&adj, p, q, length);
        assert_eq!(small.walk, reference);
        assert_eq!(whole.walk, reference);
    }

    #[test]
    fn p_and_q_move_the_options_until_the_draw_only() {
        let g = sketch();
        let adj = Adjacency::new(&g.edges);
        let mut b = Builder::new(0, 1);
        b.walk = vec![0, 1]; // came from t = 0, on v = 1
        b.advance(&adj, 1.0, 1.0, 10);
        let back = |b: &Builder| {
            b.step
                .as_ref()
                .unwrap()
                .options
                .iter()
                .find(|c| c.to == 0)
                .unwrap()
                .biased
        };
        assert_eq!(back(&b), 1.0);
        b.refresh(&adj, 4.0, 1.0);
        assert_eq!(back(&b), 0.25, "before the draw the bias follows p");
        for _ in 0..7 {
            b.advance(&adj, 4.0, 1.0, 10);
        }
        assert_eq!(b.step.as_ref().unwrap().phase, Phase::Draw);
        b.refresh(&adj, 0.1, 1.0);
        assert_eq!(back(&b), 0.25, "after the draw the step is frozen");
    }

    #[test]
    fn every_small_step_draws_without_panicking() {
        // The Math tab, the This step tab and the canvas, headless, at each of the 10 small
        // steps of two walk steps (the first has no t), on a small graph and on the swarm.
        for preset in [Preset::Barbell, Preset::SwarmByRole] {
            let mut app = Playground::default();
            app.load(preset);
            app.recompute();
            for _ in 0..20 {
                app.next_input();
                let ctx = egui::Context::default();
                let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
                    app.math_tab(ui);
                    app.step_tab(ui);
                    app.canvas(ui);
                });
                out.textures_delta.clear();
            }
            assert_eq!(app.builder.walk.len(), 3, "{}", preset.name());
        }
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
            for d in &g.nodes {
                assert!(d.pos.iter().all(|x| x.abs() <= 2.61));
            }
        }
    }

    #[test]
    fn deleting_a_drone_renumbers_the_links() {
        let mut g = sketch();
        g.remove(1); // v goes; t–x1 stays, now 0–1
        assert_eq!(g.n(), 3);
        assert_eq!(g.edges.edges.len(), 1);
        assert_eq!((g.edges.edges[0].src, g.edges.edges[0].dst), (0, 1));
        g.toggle(1, 0);
        assert!(g.edges.edges.is_empty());
        g.toggle(2, 0);
        assert_eq!((g.edges.edges[0].src, g.edges.edges[0].dst), (0, 2));
    }

    #[test]
    fn stats_share_and_pairs_are_consistent() {
        let g = Graph::build(Preset::TwoGroups);
        let adj = Adjacency::new(&g.edges);
        let h = hops(&adj);
        let s = stats(&g, &adj, &h, &RunSettings::default(), 1.0, 1.0);
        let total: f64 = [Kind::Back, Kind::Near, Kind::Far]
            .iter()
            .map(|&k| s.share(k))
            .sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(s.same_label_pairs > 0.5 && s.same_label_pairs <= 1.0);
        assert_eq!(s.distance[0], 0.0);
        assert_eq!(s.walks.len(), 10 * 12);
    }

    #[test]
    fn the_embedding_is_skipgram_on_the_walks_and_groups_two_cliques() {
        let mut app = Playground::default();
        app.load(Preset::TwoGroups);
        app.recompute();
        let (_, e) = app.embedding.as_ref().expect("trained on load");
        let (_, s) = app.stats.as_ref().unwrap();
        assert_eq!(e.z[0], train(&s.walks, 12, &app.run, &app.embed));
        assert!(e.same_label_nearest[0] >= 11, "{}", e.same_label_nearest[0]);
    }

    #[test]
    fn projection_keeps_points_on_a_plane() {
        // Points in 3-D on the plane z = x: the picture keeps their spread.
        let z: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let (a, b) = ((i % 5) as f64, (i / 5) as f64);
                vec![a, b, a]
            })
            .collect();
        let p = project(&z);
        let d =
            |i: usize, j: usize| ((p[i][0] - p[j][0]).powi(2) + (p[i][1] - p[j][1]).powi(2)).sqrt();
        let real = |i: usize, j: usize| {
            z[i].iter()
                .zip(&z[j])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        for (i, j) in [(0, 19), (3, 7), (1, 12)] {
            assert!((d(i, j) - real(i, j)).abs() < 1e-6);
        }
    }
}
