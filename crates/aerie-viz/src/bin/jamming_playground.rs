//! Jamming playground: place jammers and a ridge, see the jamming power everywhere.
//!
//! Run with `cargo run --release -p aerie-viz --bin jamming_playground`.
//!
//! The heatmap calls the real `swarmsim::jamming::jam_power` at every pixel.
//! Tools: Move (drag jammers, ridge ends or the whole ridge; drag empty space to pan),
//! Add jammer (click), Draw ridge (drag from one end to the other).
//! Right-click a jammer or the ridge to delete it. Scroll to zoom.
//! Optional swarm layer: which drones are jammed and which links fail (previews 2c / 2d).
//! "Place from seed" runs `place_jamming` on the chosen seed's swarm, exactly as the
//! dataset will: spawn the swarm, then place jammer and ridge with the same RNG.
//! Readings (2c): each drone's true power comes from your `true_power`, its sensor
//! reading from your `readings`, its label from your `jammed_labels`. Drones can be
//! coloured by label against reading, and a scatter plot shows truth against reading.
//! Link model (2e): links come from your `live_graph` (signal against noise plus jamming,
//! D03) or, for comparison, from `fail_links` (either end jammed, D02). Live links are
//! coloured by margin, one-way links are arrows, and a plot shows how far each role is
//! heard at each distance from the jammer.
//! Two tabs: "Seed swarm" (a spawned swarm, Place from seed) and "Build your own" (place
//! drones by hand, pick role and squad, drag them, read both margins on every link).
//! Hover a link in either tab for its distance, weight, signal and margin both ways.

use aerie_graph::EdgeList;
use eframe::egui::{
    self, Align2, Color32, ColorImage, FontId, Key, Pos2, Rect, RichText, Sense, Shape, Stroke,
    TextureHandle, TextureOptions, Vec2 as EVec2, pos2, vec2,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::drone::{Drone, DroneId, Role, SquadId, Vec2};
use swarmsim::graphs::epsilon::epsilon_graph;
use swarmsim::graphs::stats::stats;
use swarmsim::jamming::{
    Jammer, Ridge, fail_links, jam_power, jammed_labels, place_jamming, readings, true_power,
};
use swarmsim::links::{LiveGraph, live_graph};
use swarmsim::spawn::spawn_swarm;

use aerie_viz::style::{draw_drone, role_color, role_name};

/// Heatmap resolution, km per pixel.
const RES: f64 = 0.2;
/// Grab distances, pixels.
const GRAB_JAMMER: f32 = 12.0;
const GRAB_END: f32 = 10.0;
const GRAB_RIDGE: f32 = 6.0;

const RIDGE_COLOR: Color32 = Color32::from_rgb(170, 120, 70);
const JAMMER_COLOR: Color32 = Color32::from_rgb(230, 60, 60);
const BLOCKED_COLOR: Color32 = Color32::from_rgb(120, 180, 255);
const ONE_WAY_COLOR: Color32 = Color32::from_rgb(60, 220, 200);
const DEAD_COLOR: Color32 = Color32::from_rgb(255, 70, 70);
/// One-way direction between drones that are not linked in the geometric graph.
const RANGE_ONE_WAY_COLOR: Color32 = Color32::from_rgb(190, 120, 255);

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE jamming playground")
            .with_inner_size([1500.0, 880.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE jamming playground",
        options,
        Box::new(|_cc| Ok(Box::new(JamApp::default()))),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Move,
    AddJammer,
    DrawRidge,
    /// "Build your own" tab only.
    AddDrone,
}

/// The two tabs: a spawned swarm, or drones placed by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scene {
    Seed,
    Hand,
}

/// How the power of several jammers adds up at one point. Not decided yet (D02).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Combine {
    Sum,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scale {
    Linear,
    Log,
}

/// Which rule decides the links under jamming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinkRule {
    /// D03: a direction works when signal ≥ noise + jamming at the receiver.
    SignalVsJamming,
    /// D02: a link fails when either end is in the jammed zone.
    EitherEnd,
}

/// What happened to one geometric link under jamming.
#[derive(Debug, Clone, Copy, PartialEq)]
enum LinkState {
    /// Works both ways. The margin is known under the D03 rule.
    Live(Option<f64>),
    /// Only `from` -> `to` works (D03 only).
    OneWay {
        from: usize,
        to: usize,
    },
    Dead,
}

/// How the swarm's drones are coloured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DroneColor {
    Role,
    Truth,
    Reading,
    Check,
}

/// A drone's label (from the truth) against what its reading says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Check {
    /// Jammed, and the reading is above the threshold too.
    Hit,
    /// Jammed, but the reading is below the threshold.
    Missed,
    /// Not jammed, but the reading is above the threshold.
    FalseAlarm,
    /// Not jammed, and the reading is below the threshold too.
    Clear,
}

impl Check {
    const ALL: [Check; 4] = [Check::Hit, Check::Missed, Check::FalseAlarm, Check::Clear];

    fn name(self) -> &'static str {
        match self {
            Check::Hit => "jammed, reading agrees",
            Check::Missed => "jammed, reading below threshold",
            Check::FalseAlarm => "not jammed, reading above threshold",
            Check::Clear => "not jammed, reading agrees",
        }
    }

    fn color(self) -> Color32 {
        match self {
            Check::Hit => Color32::from_rgb(255, 70, 70),
            Check::Missed => Color32::from_rgb(255, 170, 0),
            Check::FalseAlarm => Color32::from_rgb(220, 90, 255),
            Check::Clear => Color32::from_rgb(160, 160, 160),
        }
    }
}

/// Sorts a drone by its label (truth at or above the threshold) and its reading.
fn check(label: bool, reading: f64, threshold: f64) -> Check {
    match (label, reading >= threshold) {
        (true, true) => Check::Hit,
        (true, false) => Check::Missed,
        (false, true) => Check::FalseAlarm,
        (false, false) => Check::Clear,
    }
}

/// What the current drag is doing.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Drag {
    Jammer(usize),
    /// A hand-placed drone ("Build your own").
    Drone(usize),
    RidgeA,
    RidgeB,
    RidgeBody,
    NewRidge(Vec2),
    Pan,
}

/// Everything the heatmap depends on. The texture is rebuilt only when this changes.
#[derive(Debug, Clone, PartialEq)]
struct HeatKey {
    jammers: Vec<Jammer>,
    ridge: Option<Ridge>,
    d0: f64,
    factor: f64,
    threshold: f64,
    p0: f64,
    combine: Combine,
    scale: Scale,
    contours: bool,
}

struct JamApp {
    config: SimConfig,
    jammers: Vec<Jammer>,
    ridge: Option<Ridge>,
    tool: Tool,
    combine: Combine,
    scale: Scale,
    snap: bool,
    selected: Option<usize>,
    drag: Option<Drag>,
    /// Pointer position in km, from the last frame (the side panel shows it).
    probe: Option<Vec2>,

    // View.
    zoom: f32,
    pan: EVec2,

    // Display.
    contours: bool,
    d0_rings: bool,
    probe_lines: bool,

    // Swarm layer.
    show_swarm: bool,
    show_failed: bool,
    seed: u64,
    swarm: Vec<Drone>,
    graph: EdgeList,
    swarm_seed: Option<u64>,
    /// Re-run `place_jamming` whenever the seed changes.
    auto_place: bool,
    /// The seed the last `place_jamming` run was for (to warn when the seed moved on).
    placed_for: Option<u64>,
    /// Message from the last placement (success or failure).
    place_msg: Option<String>,

    // Readings (2c).
    /// The seed's RNG right after spawn_swarm (and after place_jamming when placed):
    /// where the dataset's readings draw their noise from.
    rng_after: ChaCha8Rng,
    /// 0 = the dataset's noise; 1, 2, ... = other noise draws for the same truth.
    noise_draw: u64,
    drone_color: DroneColor,
    show_heat: bool,

    // Link model (2e).
    link_rule: LinkRule,
    show_one_way: bool,
    margin_colors: bool,

    // "Build your own" tab.
    scene: Scene,
    /// Your drones. In this tab `swarm` is a copy and `graph` is rebuilt from them.
    hand: Vec<Drone>,
    new_role: Role,
    new_squad: u16,
    selected_drone: Option<usize>,
    /// Write both margins on every link (this tab only).
    show_margins: bool,
    /// Drone under the cursor in the truth / reading plot (highlighted on the map).
    plot_hover: Option<usize>,

    // Heatmap texture cache.
    heat: Option<TextureHandle>,
    heat_key: Option<HeatKey>,
}

impl Default for JamApp {
    fn default() -> Self {
        let mut app = JamApp {
            config: SimConfig::default(),
            jammers: Vec::new(),
            ridge: None,
            tool: Tool::Move,
            combine: Combine::Sum,
            scale: Scale::Linear,
            snap: true,
            selected: None,
            drag: None,
            probe: None,
            zoom: 1.0,
            pan: EVec2::ZERO,
            contours: true,
            d0_rings: true,
            probe_lines: true,
            show_swarm: false,
            show_failed: true,
            seed: 0,
            swarm: Vec::new(),
            graph: EdgeList {
                n: 0,
                edges: Vec::new(),
            },
            swarm_seed: None,
            auto_place: false,
            placed_for: None,
            place_msg: None,
            rng_after: ChaCha8Rng::seed_from_u64(0),
            noise_draw: 0,
            drone_color: DroneColor::Check,
            show_heat: true,
            link_rule: LinkRule::SignalVsJamming,
            show_one_way: true,
            margin_colors: true,
            scene: Scene::Seed,
            hand: Vec::new(),
            new_role: Role::Recon,
            new_squad: 0,
            selected_drone: None,
            show_margins: true,
            plot_hover: None,
            heat: None,
            heat_key: None,
        };
        app.load_test_setup();
        app
    }
}

impl eframe::App for JamApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::left("controls")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });

        match self.scene {
            Scene::Seed => self.update_swarm(),
            Scene::Hand => self.sync_hand(),
        }

        egui::Panel::right("details")
            .resizable(true)
            .default_size(380.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.details(ui));
            });

        egui::CentralPanel::default().show(ui, |ui| self.canvas(ui));
    }
}

impl JamApp {
    // ---------------------------------------------------------------- model

    /// The ridge to pass to `jam_power`. With no ridge, a zero-length one far away:
    /// `orient` is then always 0, so nothing is ever shadowed.
    fn ridge_or_none(&self) -> Ridge {
        self.ridge.unwrap_or(Ridge {
            a: Vec2 { x: -1e6, y: -1e6 },
            b: Vec2 { x: -1e6, y: -1e6 },
        })
    }

    /// Power of one jammer at `at`, from your `jam_power`.
    fn power_of(&self, j: &Jammer, at: Vec2) -> f64 {
        jam_power(j, &self.ridge_or_none(), at, &self.config)
    }

    /// Power of one jammer at `at` if there were no ridge (open ground).
    fn open_power(&self, j: &Jammer, at: Vec2) -> f64 {
        let r = j.position.dist(at) / self.config.jam_d0;
        j.power / (1.0 + r * r)
    }

    /// True when the ridge cuts the line from `j` to `at`.
    fn blocked(&self, j: &Jammer, at: Vec2) -> bool {
        self.ridge.is_some() && self.power_of(j, at) < self.open_power(j, at) * (1.0 - 1e-9)
    }

    /// Total jamming power at `at`, combining every jammer.
    fn total_power(&self, at: Vec2) -> f64 {
        let each = self.jammers.iter().map(|j| self.power_of(j, at));
        match self.combine {
            Combine::Sum => each.sum(),
            Combine::Max => each.fold(0.0, f64::max),
        }
    }

    fn clamp_to_arena(&self, p: Vec2) -> Vec2 {
        let snap = |v: f64| {
            if self.snap {
                (v * 2.0).round() / 2.0
            } else {
                v
            }
        };
        Vec2 {
            x: snap(p.x).clamp(0.0, self.config.arena_w),
            y: snap(p.y).clamp(0.0, self.config.arena_h),
        }
    }

    /// The jammed threshold in absolute power: `jam_threshold` is a fraction of P0,
    /// exactly as in `place_jamming`.
    fn threshold(&self) -> f64 {
        self.config.jam_threshold * self.config.jam_p0
    }

    /// Respawns the swarm and its epsilon graph when the seed changes.
    fn update_swarm(&mut self) {
        if self.swarm_seed == Some(self.seed) {
            return;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.swarm = spawn_swarm(&self.config, &mut rng);
        self.graph = epsilon_graph(&self.swarm, &self.config);
        self.swarm_seed = Some(self.seed);
        self.rng_after = rng;
    }

    /// True power per drone. With exactly one jammer this is your `true_power`;
    /// with several, the GUI combines them (sum or max), which `true_power` does not do.
    fn truth(&self) -> Vec<f64> {
        if let [jammer] = self.jammers.as_slice() {
            true_power(&self.swarm, jammer, &self.ridge_or_none(), &self.config)
        } else {
            self.swarm
                .iter()
                .map(|d| self.total_power(d.position))
                .collect()
        }
    }

    /// Sensor readings from your `readings`. Draw 0 continues the seed's RNG after
    /// spawn (and place_jamming), exactly as the dataset will; other draws use a fresh
    /// RNG. The noise depends only on the RNG, so moving the jammer changes the truth
    /// but not the noise on each drone.
    fn sensor_readings(&self, truth: &[f64]) -> Vec<f64> {
        let mut rng = if self.noise_draw == 0 {
            self.rng_after.clone()
        } else {
            ChaCha8Rng::seed_from_u64(
                self.seed
                    .wrapping_mul(1_000_003)
                    .wrapping_add(self.noise_draw),
            )
        };
        readings(truth, &self.config, &mut rng)
    }

    /// "Build your own": the swarm is your drones and the geometric graph is rebuilt
    /// from them with `epsilon_graph`. Readings use a fresh RNG from the seed.
    fn sync_hand(&mut self) {
        self.swarm = self.hand.clone();
        self.graph = epsilon_graph(&self.swarm, &self.config);
        self.swarm_seed = None;
        self.rng_after = ChaCha8Rng::seed_from_u64(self.seed);
    }

    /// Switches tabs. Jammers and the ridge are shared; the seed swarm is respawned
    /// when you come back.
    fn set_scene(&mut self, scene: Scene) {
        self.scene = scene;
        self.drag = None;
        self.selected_drone = None;
        match scene {
            Scene::Hand => {
                self.show_swarm = true;
                self.placed_for = None;
                self.sync_hand();
            }
            Scene::Seed => {
                if self.tool == Tool::AddDrone {
                    self.tool = Tool::Move;
                }
                self.swarm_seed = None;
                self.update_swarm();
            }
        }
    }

    fn remove_drone(&mut self, i: usize) {
        self.hand.remove(i);
        // IDs follow the list, like a freshly spawned swarm.
        for (k, d) in self.hand.iter_mut().enumerate() {
            d.id = DroneId(k as u32);
        }
        self.selected_drone = match self.selected_drone {
            Some(s) if s == i => None,
            Some(s) if s > i => Some(s - 1),
            other => other,
        };
        self.drag = None;
        self.sync_hand();
    }

    fn after_hand_preset(&mut self) {
        self.selected = Some(0);
        self.selected_drone = None;
        self.show_swarm = true;
        self.placed_for = None;
        self.sync_hand();
    }

    /// One jammer at (10, 10), no ridge, three drones on y = 10 to check margins on
    /// paper: recon at 14 and 15 km (4 and 5 km from the jammer), relay at 17 km.
    fn hand_preset_check(&mut self) {
        self.jammers = vec![Jammer {
            position: Vec2 { x: 10.0, y: 10.0 },
            power: self.config.jam_p0,
        }];
        self.ridge = None;
        self.hand = vec![
            hand_drone(0, 0, Role::Recon, 14.0, 10.0),
            hand_drone(1, 0, Role::Recon, 15.0, 10.0),
            hand_drone(2, 0, Role::Relay, 17.0, 10.0),
        ];
        self.after_hand_preset();
    }

    /// A squad at the edge of the jammed zone and a second one outside it.
    fn hand_preset_two_squads(&mut self) {
        self.jammers = vec![Jammer {
            position: Vec2 { x: 10.0, y: 10.0 },
            power: self.config.jam_p0,
        }];
        self.ridge = None;
        let layout = [
            (0, Role::Recon, 13.8, 10.6),
            (0, Role::Recon, 14.6, 11.2),
            (0, Role::Recon, 15.2, 10.2),
            (0, Role::Recon, 14.0, 9.2),
            (0, Role::Recon, 15.0, 9.4),
            (0, Role::RfSensing, 13.5, 9.9),
            (0, Role::Relay, 15.6, 10.8),
            (0, Role::Compute, 14.6, 10.0),
            (1, Role::Recon, 21.4, 10.8),
            (1, Role::Recon, 22.4, 11.0),
            (1, Role::Recon, 22.8, 9.6),
            (1, Role::Recon, 21.6, 9.2),
            (1, Role::RfSensing, 22.0, 10.2),
            (1, Role::Relay, 20.8, 10.0),
            (1, Role::Compute, 23.2, 10.4),
        ];
        self.hand = layout
            .iter()
            .enumerate()
            .map(|(i, &(squad, role, x, y))| hand_drone(i as u32, squad, role, x, y))
            .collect();
        self.after_hand_preset();
    }

    /// Relays every 6 km along the valley, a recon drone next to each, jammer in the middle.
    fn hand_preset_relay_chain(&mut self) {
        self.jammers = vec![Jammer {
            position: Vec2 { x: 20.0, y: 10.0 },
            power: self.config.jam_p0,
        }];
        self.ridge = None;
        let mut hand = Vec::new();
        for (k, x) in [8.0, 14.0, 20.0, 26.0, 32.0].into_iter().enumerate() {
            let squad = k as u16;
            hand.push(hand_drone(hand.len() as u32, squad, Role::Relay, x, 11.0));
            hand.push(hand_drone(
                hand.len() as u32,
                squad,
                Role::Recon,
                x + 0.8,
                10.0,
            ));
        }
        self.hand = hand;
        self.after_hand_preset();
    }

    /// The one-way directions of your `live_graph`: `cut_one_way` (teal, a geometric link
    /// cut by jamming) then `range_one_way` (violet, the ranges differ), with the margin
    /// of the working direction.
    fn one_way_arrows(&self, truth: &[f64]) -> Vec<OneWayArrow> {
        let live = self.live(truth);
        let cut = live.cut_one_way.iter().map(|&p| (p, true));
        let range = live.range_one_way.iter().map(|&p| (p, false));
        cut.chain(range)
            .map(|((from, to), geometric)| {
                let (from, to) = (from as usize, to as usize);
                let ld = link_detail(
                    &self.swarm[from],
                    &self.swarm[to],
                    truth[from],
                    truth[to],
                    &self.config,
                );
                OneWayArrow {
                    from,
                    to,
                    margin: ld.margin_ij,
                    geometric,
                }
            })
            .collect()
    }

    /// The live graph from your `live_graph` (D03), for this truth.
    fn live(&self, truth: &[f64]) -> LiveGraph {
        live_graph(&self.graph, &self.swarm, truth, &self.config)
    }

    /// The comm graph after jamming, under the chosen rule.
    fn after_graph(&self, truth: &[f64]) -> EdgeList {
        match self.link_rule {
            LinkRule::SignalVsJamming => self.live(truth).graph,
            LinkRule::EitherEnd => fail_links(&self.graph, &jammed_labels(truth, &self.config)),
        }
    }

    /// The state of every geometric link, in the order of `self.graph.edges`.
    fn link_states(&self, truth: &[f64]) -> Vec<LinkState> {
        match self.link_rule {
            LinkRule::SignalVsJamming => link_states_from(&self.graph, &self.live(truth)),
            LinkRule::EitherEnd => {
                let jammed = jammed_labels(truth, &self.config);
                self.graph
                    .edges
                    .iter()
                    .map(|e| {
                        if jammed[e.src as usize] || jammed[e.dst as usize] {
                            LinkState::Dead
                        } else {
                            LinkState::Live(None)
                        }
                    })
                    .collect()
            }
        }
    }

    /// Jammed flag per drone (the zone label), from your `jammed_labels`.
    fn jammed(&self) -> Vec<bool> {
        jammed_labels(&self.truth(), &self.config)
    }

    /// Same geometry as the unit tests, moved by (10, 10): jammer at (10, 10),
    /// ridge at x = 12 from y = 5 to 15.
    fn load_test_setup(&mut self) {
        self.jammers = vec![Jammer {
            position: Vec2 { x: 10.0, y: 10.0 },
            power: 1.0,
        }];
        self.ridge = Some(Ridge {
            a: Vec2 { x: 12.0, y: 5.0 },
            b: Vec2 { x: 12.0, y: 15.0 },
        });
        self.selected = Some(0);
    }

    /// Spawns the swarm for `self.seed` and places one jammer and one ridge with
    /// `place_jamming`, using the same RNG after the spawn (as the dataset will).
    /// `place_jamming` panics when no placement works (for example with the jammed
    /// threshold slider at 1.0); the GUI catches that and shows a message instead.
    fn place_from_seed(&mut self) {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let swarm = spawn_swarm(&self.config, &mut rng);
        let config = self.config;
        // Silence the panic message while trying: the GUI shows its own.
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            place_jamming(&swarm, &config, &mut rng)
        }));
        std::panic::set_hook(previous_hook);
        self.rng_after = rng;
        self.graph = epsilon_graph(&swarm, &self.config);
        self.swarm = swarm;
        self.swarm_seed = Some(self.seed);
        self.show_swarm = true;
        match result {
            Ok((jammer, ridge)) => {
                self.jammers = vec![jammer];
                self.ridge = Some(ridge);
                self.selected = Some(0);
                self.placed_for = Some(self.seed);
                let jammed = self.jammed().iter().filter(|&&j| j).count();
                self.place_msg = Some(format!(
                    "Seed {}: {} of {} drones jammed ({:.0} %).",
                    self.seed,
                    jammed,
                    self.swarm.len(),
                    100.0 * jammed as f64 / self.swarm.len().max(1) as f64
                ));
            }
            Err(_) => {
                self.placed_for = None;
                self.place_msg = Some(format!(
                    "Seed {}: place_jamming found no placement with {} to {} % jammed. \
                     Check the threshold and d0 sliders.",
                    self.seed,
                    100.0 * self.config.jam_frac_min,
                    100.0 * self.config.jam_frac_max
                ));
            }
        }
    }

    fn load_two_jammers(&mut self) {
        self.jammers = vec![
            Jammer {
                position: Vec2 { x: 13.0, y: 10.0 },
                power: 1.0,
            },
            Jammer {
                position: Vec2 { x: 21.0, y: 10.0 },
                power: 1.0,
            },
        ];
        self.ridge = Some(Ridge {
            a: Vec2 { x: 26.0, y: 3.0 },
            b: Vec2 { x: 29.0, y: 17.0 },
        });
        self.selected = None;
    }

    // ------------------------------------------------------------- heatmap

    fn heat_key_now(&self) -> HeatKey {
        HeatKey {
            jammers: self.jammers.clone(),
            ridge: self.ridge,
            d0: self.config.jam_d0,
            factor: self.config.ridge_factor,
            threshold: self.config.jam_threshold,
            p0: self.config.jam_p0,
            combine: self.combine,
            scale: self.scale,
            contours: self.contours,
        }
    }

    /// Maps a power to 0..1 for the colour ramp.
    fn scale_t(&self, power: f64) -> f32 {
        let t = match self.scale {
            Scale::Linear => power,
            // 0.01 P0 -> 0, 1 P0 -> 1.
            Scale::Log => (power.max(1e-9).log10() + 2.0) / 2.0,
        };
        t.clamp(0.0, 1.0) as f32
    }

    /// One pixel per RES km. Row 0 is the top of the arena (y = arena_h).
    fn heat_image(&self) -> ColorImage {
        let (aw, ah) = (self.config.arena_w, self.config.arena_h);
        let w = (aw / RES).round() as usize;
        let h = (ah / RES).round() as usize;
        let mut vals = vec![0.0; w * h];
        for r in 0..h {
            for c in 0..w {
                let at = Vec2 {
                    x: (c as f64 + 0.5) * RES,
                    y: ah - (r as f64 + 0.5) * RES,
                };
                vals[r * w + c] = self.total_power(at);
            }
        }
        let p0 = self.config.jam_p0.max(1e-9);
        let mut px: Vec<Color32> = vals
            .iter()
            .map(|&v| heat_color(self.scale_t(v / p0)))
            .collect();

        if self.contours {
            // A pixel is on a contour when it and its right or lower neighbour sit on
            // different sides of the level.
            let levels = [
                (self.threshold(), Color32::WHITE),
                (0.2 * p0, Color32::from_white_alpha(110)),
                (0.1 * p0, Color32::from_white_alpha(70)),
            ];
            for r in 0..h {
                for c in 0..w {
                    let v = vals[r * w + c];
                    let right = if c + 1 < w { vals[r * w + c + 1] } else { v };
                    let down = if r + 1 < h { vals[(r + 1) * w + c] } else { v };
                    for &(level, color) in levels.iter().rev() {
                        let above = v >= level;
                        if above != (right >= level) || above != (down >= level) {
                            px[r * w + c] = blend(px[r * w + c], color);
                        }
                    }
                }
            }
        }
        ColorImage::new([w, h], px)
    }

    // ---------------------------------------------------------- left panel

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Jamming playground");
        ui.horizontal(|ui| {
            let mut scene = self.scene;
            ui.selectable_value(
                &mut scene,
                Scene::Seed,
                RichText::new("Seed swarm").strong(),
            );
            ui.selectable_value(
                &mut scene,
                Scene::Hand,
                RichText::new("Build your own").strong(),
            );
            if scene != self.scene {
                self.set_scene(scene);
            }
        });
        ui.weak(match self.scene {
            Scene::Seed => "A spawned swarm. Every heatmap pixel calls your jam_power.",
            Scene::Hand => {
                "Your own drones: add, drag, pick role and squad. Links come from \
                 epsilon_graph and live_graph."
            }
        });
        ui.separator();

        ui.label(RichText::new("Tool").strong());
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tool, Tool::Move, "Move");
            ui.selectable_value(&mut self.tool, Tool::AddJammer, "Add jammer");
            ui.selectable_value(&mut self.tool, Tool::DrawRidge, "Draw ridge");
            if self.scene == Scene::Hand {
                ui.selectable_value(&mut self.tool, Tool::AddDrone, "Add drone");
            }
        });
        ui.weak(match self.tool {
            Tool::Move => {
                "Drag a jammer, a drone (your own tab), a ridge end or the ridge. \
                 Drag empty space to pan."
            }
            Tool::AddJammer => "Click to add a jammer. Drag a jammer to move it.",
            Tool::DrawRidge => "Drag from one end of the ridge to the other (one ridge).",
            Tool::AddDrone => "Click to add a drone (role and squad below). Drag to move it.",
        });
        ui.weak("Right-click a jammer, a drone or the ridge to delete it. Scroll: zoom.");
        ui.checkbox(&mut self.snap, "Snap to 0.5 km");
        ui.separator();

        // Jammers.
        ui.label(RichText::new("Jammers").strong());
        let mut remove = None;
        for (i, j) in self.jammers.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                let label = RichText::new(format!("J{i}")).color(JAMMER_COLOR).strong();
                if ui
                    .selectable_label(self.selected == Some(i), label)
                    .clicked()
                {
                    self.selected = Some(i);
                }
                ui.add(
                    egui::DragValue::new(&mut j.power)
                        .range(0.0..=5.0)
                        .speed(0.01)
                        .prefix("P0 "),
                );
                ui.add(
                    egui::DragValue::new(&mut j.position.x)
                        .range(0.0..=self.config.arena_w)
                        .speed(0.1)
                        .prefix("x "),
                );
                ui.add(
                    egui::DragValue::new(&mut j.position.y)
                        .range(0.0..=self.config.arena_h)
                        .speed(0.1)
                        .prefix("y "),
                );
                if ui.small_button("✖").on_hover_text("Delete").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.remove_jammer(i);
        }
        if self.jammers.is_empty() {
            ui.weak("No jammer. Pick \"Add jammer\" and click the map.");
        }
        ui.add_space(4.0);

        // Ridge.
        ui.label(RichText::new("Ridge").strong());
        match &mut self.ridge {
            Some(r) => {
                for (name, end) in [("a", &mut r.a), ("b", &mut r.b)] {
                    ui.horizontal(|ui| {
                        ui.label(name);
                        ui.add(
                            egui::DragValue::new(&mut end.x)
                                .range(0.0..=self.config.arena_w)
                                .speed(0.1)
                                .prefix("x "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut end.y)
                                .range(0.0..=self.config.arena_h)
                                .speed(0.1)
                                .prefix("y "),
                        );
                    });
                }
                let len = r.a.dist(r.b);
                ui.weak(format!("Length {len:.1} km"));
                if ui.button("Remove ridge").clicked() {
                    self.ridge = None;
                }
            }
            None => {
                ui.weak("No ridge. Pick \"Draw ridge\" and drag on the map.");
            }
        }
        ui.separator();

        // Physics. Moving any of these re-runs place_jamming when a placement is shown.
        ui.label(RichText::new("Physics (SimConfig)").strong());
        let physics_before = (
            self.config.jam_p0,
            self.config.jam_d0,
            self.config.ridge_factor,
            self.config.jam_threshold,
        );
        ui.add(egui::Slider::new(&mut self.config.jam_p0, 0.1..=3.0).text("P0  jammer power"));
        ui.add(
            egui::Slider::new(&mut self.config.jam_d0, 0.5..=10.0)
                .text("d0  half-power distance")
                .suffix(" km"),
        );
        ui.add(egui::Slider::new(&mut self.config.ridge_factor, 0.0..=1.0).text("ridge factor"));
        ui.add(
            egui::Slider::new(&mut self.config.jam_threshold, 0.05..=1.0).text("jammed threshold"),
        );
        ui.weak("Threshold, 0.2 and 0.1 contours and colours are fractions of P0.");
        let physics_after = (
            self.config.jam_p0,
            self.config.jam_d0,
            self.config.ridge_factor,
            self.config.jam_threshold,
        );
        if physics_after != physics_before {
            if self.placed_for.is_some() {
                // Live: a new placement under the new physics.
                self.place_from_seed();
            } else if physics_after.0 != physics_before.0 {
                // No placement shown: P0 applies to the jammers on the map.
                for j in &mut self.jammers {
                    j.power = self.config.jam_p0;
                }
            }
        }
        ui.horizontal(|ui| {
            ui.label("Several jammers:");
            ui.selectable_value(&mut self.combine, Combine::Sum, "sum");
            ui.selectable_value(&mut self.combine, Combine::Max, "max");
        });
        ui.weak("How jammers add up is not decided yet (D02). Noise powers usually add: sum.");
        if ui.button("Reset physics to defaults").clicked() {
            let d = SimConfig::default();
            self.config.jam_p0 = d.jam_p0;
            self.config.jam_d0 = d.jam_d0;
            self.config.ridge_factor = d.ridge_factor;
            self.config.jam_threshold = d.jam_threshold;
            self.config.jam_frac_min = d.jam_frac_min;
            self.config.jam_frac_max = d.jam_frac_max;
            self.config.noise_sd = d.noise_sd;
        }
        ui.separator();

        // Presets.
        ui.label(RichText::new("Presets").strong());
        ui.horizontal(|ui| {
            if ui
                .button("Test setup")
                .on_hover_text(
                    "The unit-test geometry moved by (10, 10): jammer at (10, 10),\n\
                     ridge at x = 12. Hover (10, 14): 0.5, (14, 10): 0.05, (12, 10): 0.8.",
                )
                .clicked()
            {
                self.load_test_setup();
            }
            if ui.button("Two jammers").clicked() {
                self.load_two_jammers();
            }
            if ui.button("Clear").clicked() {
                self.jammers.clear();
                self.ridge = None;
                self.selected = None;
            }
        });
        ui.separator();

        // Display.
        ui.label(RichText::new("Display").strong());
        ui.horizontal(|ui| {
            ui.label("Colour scale:");
            ui.selectable_value(&mut self.scale, Scale::Linear, "linear");
            ui.selectable_value(&mut self.scale, Scale::Log, "log");
        });
        ui.checkbox(
            &mut self.contours,
            "Contours (threshold white, 0.2 and 0.1 grey)",
        );
        ui.checkbox(&mut self.d0_rings, "d0 ring around each jammer");
        ui.checkbox(&mut self.probe_lines, "Lines from jammers to the cursor");
        ui.horizontal(|ui| {
            if ui.button("Reset view").clicked() {
                self.zoom = 1.0;
                self.pan = EVec2::ZERO;
            }
            ui.weak(format!("zoom ×{:.1}", self.zoom));
        });
        ui.separator();

        match self.scene {
            Scene::Seed => self.seed_controls(ui),
            Scene::Hand => self.hand_controls(ui),
        }
        ui.weak("Jammed (red ring) = true power ≥ threshold: the zone label.");
        ui.separator();

        // Link model (2e).
        ui.label(RichText::new("Link rule").strong());
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.link_rule,
                LinkRule::SignalVsJamming,
                "signal vs jamming (D03)",
            );
            ui.selectable_value(
                &mut self.link_rule,
                LinkRule::EitherEnd,
                "either end jammed (D02)",
            );
        });
        ui.weak(match self.link_rule {
            LinkRule::SignalVsJamming => {
                "A direction works when the sender's signal ≥ N0 + jamming at the receiver. \
                 A link needs both directions. Your live_graph."
            }
            LinkRule::EitherEnd => {
                "The old rule: a link fails when either end is in the jammed zone. Your fail_links."
            }
        });
        ui.add_enabled_ui(self.link_rule == LinkRule::SignalVsJamming, |ui| {
            ui.add(
                egui::Slider::new(&mut self.config.noise_floor, 0.005..=0.5)
                    .logarithmic(true)
                    .text("N0  noise floor (× P0)"),
            )
            .on_hover_text(
                "Lower N0 = FEWER links under jamming. Transmit power is tied to N0 \
                 (tx = N0 (1 + (range / scale)²)), so a lower N0 means quieter drones. \
                 margin = geometry × N0 / (N0 + jamming): with no jammer N0 cancels out; \
                 with a jammer, raising N0 makes the drones louder against it.",
            );
            ui.weak("Raise N0 = drones louder against the jammer = more links survive.");
            ui.weak("Signal scale per role: smaller = louder transmitter = resists jamming.");
            let scales = [
                (Role::Recon, &mut self.config.signal_scale_recon),
                (Role::RfSensing, &mut self.config.signal_scale_rf),
                (Role::Relay, &mut self.config.signal_scale_relay),
                (Role::Compute, &mut self.config.signal_scale_compute),
            ];
            for (role, value) in scales {
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
                    ui.painter()
                        .circle_filled(rect.center(), 4.5, role_color(role));
                    ui.add(
                        egui::Slider::new(value, 0.05..=3.0)
                            .logarithmic(true)
                            .suffix(" km")
                            .text(role_name(role)),
                    );
                });
            }
            if ui.button("Reset link model to defaults").clicked() {
                let d = SimConfig::default();
                self.config.noise_floor = d.noise_floor;
                self.config.signal_scale_recon = d.signal_scale_recon;
                self.config.signal_scale_rf = d.signal_scale_rf;
                self.config.signal_scale_relay = d.signal_scale_relay;
                self.config.signal_scale_compute = d.signal_scale_compute;
            }
        });
        ui.add_enabled_ui(self.show_swarm, |ui| {
            ui.checkbox(&mut self.show_failed, "Dead links (red, dashed)");
            ui.add_enabled_ui(self.link_rule == LinkRule::SignalVsJamming, |ui| {
                ui.checkbox(
                    &mut self.show_one_way,
                    "One-way links (arrow: sender → receiver)",
                );
                ui.checkbox(
                    &mut self.margin_colors,
                    "Colour live links by margin (orange ≈ 1, about to break; green ≥ 10)",
                );
            });
        });
        ui.separator();

        // Readings (2c).
        ui.label(RichText::new("Readings (2c)").strong());
        ui.add(egui::Slider::new(&mut self.config.noise_sd, 0.0..=0.5).text("noise sd (× P0)"));
        ui.horizontal(|ui| {
            ui.label("Noise draw:");
            let name = if self.noise_draw == 0 {
                "dataset".to_string()
            } else {
                format!("#{}", self.noise_draw)
            };
            ui.label(RichText::new(name).monospace());
            if ui
                .button("New draw")
                .on_hover_text("Fresh noise on every drone, same truth.")
                .clicked()
            {
                self.noise_draw += 1;
            }
            if ui
                .add_enabled(self.noise_draw != 0, egui::Button::new("Dataset draw"))
                .on_hover_text(
                    "The noise the dataset gets: the seed's RNG, continued after \
                     spawn_swarm and place_jamming.",
                )
                .clicked()
            {
                self.noise_draw = 0;
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Colour drones by:");
            ui.selectable_value(&mut self.drone_color, DroneColor::Role, "role");
            ui.selectable_value(&mut self.drone_color, DroneColor::Truth, "true power");
            ui.selectable_value(&mut self.drone_color, DroneColor::Reading, "reading");
            ui.selectable_value(&mut self.drone_color, DroneColor::Check, "label vs reading");
        });
        if self.drone_color == DroneColor::Check {
            for c in Check::ALL {
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(vec2(12.0, 12.0), Sense::hover());
                    ui.painter().circle_filled(rect.center(), 5.0, c.color());
                    ui.label(c.name());
                });
            }
        }
        ui.checkbox(
            &mut self.show_heat,
            "Heatmap (turn off to see drone colours)",
        );
        ui.weak(
            "Reading = truth + Normal(0, noise sd). The label (red ring) comes from the \
             truth; the model will only see the readings.",
        );
        ui.separator();

        // Legend.
        ui.label(RichText::new("Legend").strong());
        ui.horizontal(|ui| {
            for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let (rect, _) = ui.allocate_exact_size(vec2(18.0, 12.0), Sense::hover());
                ui.painter().rect_filled(rect, 2.0, heat_color(t));
                let label = match self.scale {
                    Scale::Linear => format!("{t}"),
                    Scale::Log => format!("{}", 10f64.powf(t as f64 * 2.0 - 2.0)),
                };
                ui.label(label);
            }
        });
        ui.weak("Power as a fraction of P0 = 1.");
    }

    /// Left panel, "Seed swarm" tab: seed, Place from seed, placement bounds.
    fn seed_controls(&mut self, ui: &mut egui::Ui) {
        // Swarm.
        ui.label(RichText::new("Swarm").strong());
        ui.checkbox(&mut self.show_swarm, "Show a spawned swarm");
        let seed_before = self.seed;
        ui.horizontal(|ui| {
            ui.label("Seed");
            ui.add(egui::DragValue::new(&mut self.seed).range(0..=999_999));
            if ui.small_button("◀").clicked() {
                self.seed = self.seed.saturating_sub(1);
            }
            if ui.small_button("▶").clicked() {
                self.seed += 1;
            }
        });
        ui.horizontal(|ui| {
            if ui
                .button("Place from seed")
                .on_hover_text(
                    "Runs your place_jamming on this seed's swarm: one jammer and one \
                     ridge, redrawn until 5-30 % of the drones are jammed.\n\
                     Replaces the jammers and the ridge on the map.",
                )
                .clicked()
            {
                self.place_from_seed();
            }
            ui.checkbox(&mut self.auto_place, "on every seed change");
        });

        // Placement bounds (D02), live: moving a slider re-runs place_jamming.
        let bounds_before = (self.config.jam_frac_min, self.config.jam_frac_max);
        ui.add(
            egui::Slider::new(&mut self.config.jam_frac_min, 0.0..=1.0)
                .text("min jammed share")
                .custom_formatter(|v, _| format!("{:.0} %", v * 100.0)),
        );
        ui.add(
            egui::Slider::new(&mut self.config.jam_frac_max, 0.0..=1.0)
                .text("max jammed share")
                .custom_formatter(|v, _| format!("{:.0} %", v * 100.0)),
        );
        // Keep min <= max: whichever slider moved pushes the other one.
        if self.config.jam_frac_min > self.config.jam_frac_max {
            if self.config.jam_frac_min != bounds_before.0 {
                self.config.jam_frac_max = self.config.jam_frac_min;
            } else {
                self.config.jam_frac_min = self.config.jam_frac_max;
            }
        }
        let bounds_moved = (self.config.jam_frac_min, self.config.jam_frac_max) != bounds_before;
        if bounds_moved && self.placed_for.is_some() {
            self.place_from_seed();
        }
        ui.weak("place_jamming redraws until the jammed share is between these. Moving them re-places live.");
        if self.auto_place && self.seed != seed_before {
            self.place_from_seed();
        }
        if let Some(msg) = &self.place_msg {
            ui.weak(msg);
        }
        if self.placed_for.is_some() && self.placed_for != Some(self.seed) {
            ui.weak("The seed changed since the placement: press the button again.");
        }
    }

    /// Left panel, "Build your own" tab: the new drone's role and squad, your drones
    /// (role, squad, position), presets.
    fn hand_controls(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Your drones").strong());
        ui.horizontal_wrapped(|ui| {
            ui.label("New drone:");
            for role in Role::ALL {
                ui.selectable_value(
                    &mut self.new_role,
                    role,
                    RichText::new(role_name(role)).color(role_color(role)),
                );
            }
        });
        ui.horizontal(|ui| {
            ui.label("in squad");
            ui.add(
                egui::DragValue::new(&mut self.new_squad)
                    .range(0..=9)
                    .prefix("S"),
            );
            if ui.button("Add drone tool").clicked() {
                self.tool = Tool::AddDrone;
            }
        });
        let mut remove = None;
        let mut changed = false;
        egui::Grid::new("hand_drones").striped(true).show(ui, |ui| {
            for (i, d) in self.hand.iter_mut().enumerate() {
                let label = RichText::new(format!("#{i}"))
                    .color(role_color(d.role))
                    .strong();
                if ui
                    .selectable_label(self.selected_drone == Some(i), label)
                    .clicked()
                {
                    self.selected_drone = Some(i);
                }
                egui::ComboBox::from_id_salt(("hand_role", i))
                    .selected_text(role_name(d.role))
                    .width(90.0)
                    .show_ui(ui, |ui| {
                        for role in Role::ALL {
                            changed |= ui
                                .selectable_value(&mut d.role, role, role_name(role))
                                .changed();
                        }
                    });
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut d.squad.0)
                            .range(0..=9)
                            .prefix("S"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut d.position.x)
                            .range(0.0..=self.config.arena_w)
                            .speed(0.05)
                            .prefix("x "),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut d.position.y)
                            .range(0.0..=self.config.arena_h)
                            .speed(0.05)
                            .prefix("y "),
                    )
                    .changed();
                if ui.small_button("✖").on_hover_text("Delete").clicked() {
                    remove = Some(i);
                }
                ui.end_row();
            }
        });
        if let Some(i) = remove {
            self.remove_drone(i);
        } else if changed {
            self.sync_hand();
        }
        if self.hand.is_empty() {
            ui.weak("No drones yet. Pick \"Add drone\" and click the map, or load a preset.");
        }
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Paper check")
                .on_hover_text(
                    "Jammer at (10, 10), no ridge. Recon at 14 and 15 km, relay at 17 km \
                     on y = 10: check the margins by hand.",
                )
                .clicked()
            {
                self.hand_preset_check();
            }
            if ui
                .button("Squad at the zone edge")
                .on_hover_text("One squad 4-5 km from the jammer, another one 11-13 km away.")
                .clicked()
            {
                self.hand_preset_two_squads();
            }
            if ui
                .button("Relay chain")
                .on_hover_text("Relays every 6 km along the valley, the jammer in the middle.")
                .clicked()
            {
                self.hand_preset_relay_chain();
            }
            if ui.button("Clear drones").clicked() {
                self.hand.clear();
                self.selected_drone = None;
                self.sync_hand();
            }
        });
        ui.checkbox(
            &mut self.show_margins,
            "Write both margins on each link (a→b over b→a; ≥ 1 works)",
        );
        ui.checkbox(
            &mut self.show_one_way,
            "One-way directions as arrows (sender → receiver)",
        );
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(vec2(18.0, 10.0), Sense::hover());
            ui.painter().line_segment(
                [rect.left_center(), rect.right_center()],
                Stroke::new(2.0, ONE_WAY_COLOR),
            );
            ui.label("a geometric link cut one way by jamming");
        });
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(vec2(18.0, 10.0), Sense::hover());
            ui.painter().line_segment(
                [rect.left_center(), rect.right_center()],
                Stroke::new(2.0, RANGE_ONE_WAY_COLOR),
            );
            ui.label("not a link: the ranges differ (e.g. a relay heard by a far recon)");
        });
        ui.weak("Arrows need the D03 link rule. Geometric links use the no-jammer ranges (epsilon_graph).");
    }

    fn remove_jammer(&mut self, i: usize) {
        self.jammers.remove(i);
        self.selected = match self.selected {
            Some(s) if s == i => None,
            Some(s) if s > i => Some(s - 1),
            other => other,
        };
        self.drag = None;
    }

    // --------------------------------------------------------- right panel

    fn details(&mut self, ui: &mut egui::Ui) {
        self.plot_hover = None;
        ui.label(RichText::new("Power against distance (as a fraction of P0)").strong());
        ui.label(RichText::new("P(d) = P0 / (1 + (d / d0)²)").monospace());
        falloff_plot(ui, &self.config);
        let d0 = self.config.jam_d0;
        let f = self.config.ridge_factor;
        egui::Grid::new("falloff_values")
            .striped(true)
            .show(ui, |ui| {
                ui.label("d");
                ui.label("open ground");
                ui.label("behind ridge");
                ui.end_row();
                for k in [0.0, 1.0, 2.0, 3.0] {
                    let p = self.config.jam_p0 / (1.0 + k * k);
                    ui.label(format!("{k} d0 = {:.1} km", k * d0));
                    ui.label(format!("{p:.3}"));
                    ui.label(format!("{:.3}", p * f));
                    ui.end_row();
                }
            });
        let t = self.config.jam_threshold;
        if t < 1.0 {
            let r = d0 * (1.0 / t - 1.0).sqrt();
            ui.weak(format!(
                "Jammed zone in open ground (P0 = 1): radius d0·√(1/{t} − 1) = {r:.2} km."
            ));
        }
        ui.separator();

        // Probe.
        ui.label(RichText::new("At the cursor").strong());
        match self.probe {
            None => {
                ui.weak("Hover the map.");
            }
            Some(at) => {
                let total = self.total_power(at);
                let jammed = total >= self.threshold();
                ui.label(format!("({:.2}, {:.2}) km", at.x, at.y));
                ui.label(
                    RichText::new(format!(
                        "total {:.3}  →  {}",
                        total,
                        if jammed { "JAMMED" } else { "not jammed" }
                    ))
                    .strong()
                    .color(if jammed {
                        JAMMER_COLOR
                    } else {
                        ui.visuals().text_color()
                    }),
                );
                egui::Grid::new("probe").striped(true).show(ui, |ui| {
                    ui.label("");
                    ui.label("d (km)");
                    ui.label("open");
                    ui.label("ridge");
                    ui.label("power");
                    ui.end_row();
                    for (i, j) in self.jammers.iter().enumerate() {
                        let blocked = self.blocked(j, at);
                        ui.label(RichText::new(format!("J{i}")).color(JAMMER_COLOR));
                        ui.label(format!("{:.2}", j.position.dist(at)));
                        ui.label(format!("{:.3}", self.open_power(j, at)));
                        ui.label(if blocked {
                            RichText::new("blocked").color(BLOCKED_COLOR)
                        } else {
                            RichText::new("clear")
                        });
                        ui.label(format!("{:.3}", self.power_of(j, at)));
                        ui.end_row();
                    }
                });
            }
        }
        ui.separator();

        // Swarm effect.
        ui.label(RichText::new("Effect on the swarm").strong());
        if !self.show_swarm {
            ui.weak("Turn on \"Show a spawned swarm\" in the left panel.");
            return;
        }
        let truth = self.truth();
        let jammed = jammed_labels(&truth, &self.config);
        let n_jammed = jammed.iter().filter(|&&j| j).count();
        let before = stats(&self.graph, &self.swarm);
        let after_graph = self.after_graph(&truth);
        let after = stats(&after_graph, &self.swarm);
        let states = self.link_states(&truth);
        let n_one_way = states
            .iter()
            .filter(|s| matches!(s, LinkState::OneWay { .. }))
            .count();
        let n_fragile = states
            .iter()
            .filter(|s| matches!(s, LinkState::Live(Some(m)) if *m < 2.0))
            .count();
        egui::Grid::new("swarm_effect")
            .striped(true)
            .show(ui, |ui| {
                ui.label("");
                ui.label("before");
                ui.label("after jamming");
                ui.end_row();
                ui.label("Jammed drones");
                ui.label("");
                ui.label(format!(
                    "{n_jammed} ({:.0} %)",
                    100.0 * n_jammed as f64 / self.swarm.len().max(1) as f64
                ));
                ui.end_row();
                ui.label("Links");
                ui.label(before.n_edges.to_string());
                ui.label(after.n_edges.to_string());
                ui.end_row();
                ui.label("Mean degree");
                ui.label(format!("{:.2}", before.mean_degree));
                ui.label(format!("{:.2}", after.mean_degree));
                ui.end_row();
                ui.label("Components");
                ui.label(before.n_components.to_string());
                ui.label(after.n_components.to_string());
                ui.end_row();
                ui.label("Largest component");
                ui.label(before.largest_component.to_string());
                ui.label(after.largest_component.to_string());
                ui.end_row();
                ui.label("Isolated drones");
                ui.label(before.isolated.to_string());
                ui.label(after.isolated.to_string());
                ui.end_row();
                if self.link_rule == LinkRule::SignalVsJamming {
                    ui.label("One-way links");
                    ui.label("0");
                    ui.label(n_one_way.to_string());
                    ui.end_row();
                    ui.label("Live links with margin < 2");
                    ui.label("");
                    ui.label(n_fragile.to_string());
                    ui.end_row();
                }
            });
        ui.weak(match self.link_rule {
            LinkRule::SignalVsJamming => {
                "D03: short links and relays hold out; long links into the jammed area die \
                 first. A drone near the jammer can be heard but cannot hear (one-way)."
            }
            LinkRule::EitherEnd => {
                "D02: every jammed drone loses all its links, so it becomes isolated."
            }
        });
        ui.separator();

        // Range under jamming.
        ui.label(RichText::new("How far each role is heard (D03)").strong());
        ui.weak(
            "By the receiver's distance to one jammer in open ground. Solves \
             tx / (1 + (d / scale)²) = N0 + jamming for d.",
        );
        range_plot(ui, &self.config);
        egui::Grid::new("range_table").striped(true).show(ui, |ui| {
            ui.label("receiver at");
            for role in Role::ALL {
                ui.label(RichText::new(role_name(role)).color(role_color(role)));
            }
            ui.end_row();
            ui.label("tx power");
            for role in Role::ALL {
                ui.label(format!("{:.2}", role.tx_power(&self.config)));
            }
            ui.end_row();
            ui.label("no jammer");
            for role in Role::ALL {
                ui.label(range_text(heard_range(role, 0.0, &self.config)));
            }
            ui.end_row();
            for dist in [20.0, 12.0, 4.0, 0.0] {
                ui.label(format!("{dist} km"));
                let jam = open_jam(dist, &self.config);
                for role in Role::ALL {
                    ui.label(range_text(heard_range(role, jam, &self.config)));
                }
                ui.end_row();
            }
        });
        ui.separator();

        // Readings against truth.
        ui.label(RichText::new("Readings against truth (2c)").strong());
        let truth = self.truth();
        let reads = self.sensor_readings(&truth);
        let thr = self.threshold();
        let sd = self.config.noise_sd * self.config.jam_p0;
        ui.label(format!(
            "noise sd = noise_sd × P0 = {sd:.3}   ·   draw: {}",
            if self.noise_draw == 0 {
                "dataset".to_string()
            } else {
                format!("#{}", self.noise_draw)
            }
        ));
        let mut counts = [0usize; 4];
        for (&t, &r) in truth.iter().zip(&reads) {
            counts[check(t >= thr, r, thr) as usize] += 1;
        }
        egui::Grid::new("checks").striped(true).show(ui, |ui| {
            for c in Check::ALL {
                let (rect, _) = ui.allocate_exact_size(vec2(12.0, 12.0), Sense::hover());
                ui.painter().circle_filled(rect.center(), 5.0, c.color());
                ui.label(c.name());
                ui.label(counts[c as usize].to_string());
                ui.end_row();
            }
        });
        let n = truth.len().max(1) as f64;
        let noise: Vec<f64> = reads.iter().zip(&truth).map(|(r, t)| r - t).collect();
        let mean = noise.iter().sum::<f64>() / n;
        let spread = (noise.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n).sqrt();
        ui.label(format!(
            "Measured noise over {} drones: mean {mean:+.3}, sd {spread:.3}",
            truth.len()
        ));
        self.plot_hover = truth_reading_plot(ui, &truth, &reads, thr, self.config.jam_p0);
        ui.weak(
            "Each dot is a drone. On the diagonal: no noise. Right of the vertical line: \
             jammed (label, from the truth). Above the horizontal line: the reading says \
             jammed. Top left: false alarms. Bottom right: missed. Hover a dot to find \
             the drone on the map.",
        );
    }

    // --------------------------------------------------------------- canvas

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let (aw, ah) = (self.config.arena_w, self.config.arena_h);

        // Zoom around the cursor.
        if let Some(pointer) = response.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let before =
                    Transform::fit(response.rect, aw, ah, self.zoom, self.pan).to_km(pointer);
                self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(0.5, 25.0);
                let moved =
                    Transform::fit(response.rect, aw, ah, self.zoom, self.pan).to_screen(before);
                self.pan += pointer - moved;
            }
        }
        let t = Transform::fit(response.rect, aw, ah, self.zoom, self.pan);
        let visuals = ui.visuals().clone();
        let text = visuals.text_color();

        // --- Input.
        if response.drag_started() {
            let origin = ui
                .input(|i| i.pointer.press_origin())
                .or(response.interact_pointer_pos());
            self.drag = origin.map(|p| match self.hit(&t, p) {
                Some(Hit::Jammer(i)) => {
                    self.selected = Some(i);
                    Drag::Jammer(i)
                }
                Some(Hit::Drone(i)) => {
                    self.selected_drone = Some(i);
                    Drag::Drone(i)
                }
                Some(Hit::RidgeA) => Drag::RidgeA,
                Some(Hit::RidgeB) => Drag::RidgeB,
                Some(Hit::Ridge) if self.tool != Tool::DrawRidge => Drag::RidgeBody,
                _ if self.tool == Tool::DrawRidge => {
                    Drag::NewRidge(self.clamp_to_arena(t.to_km(p)))
                }
                _ => Drag::Pan,
            });
        }
        if response.dragged()
            && let Some(drag) = self.drag
        {
            let pointer = response.interact_pointer_pos();
            let km = pointer.map(|p| self.clamp_to_arena(t.to_km(p)));
            match (drag, km) {
                (Drag::Jammer(i), Some(km)) if i < self.jammers.len() => {
                    self.jammers[i].position = km
                }
                (Drag::Drone(i), Some(km)) if i < self.hand.len() => {
                    self.hand[i].position = km;
                    self.sync_hand();
                }
                (Drag::RidgeA, Some(km)) => {
                    if let Some(r) = &mut self.ridge {
                        r.a = km;
                    }
                }
                (Drag::RidgeB, Some(km)) => {
                    if let Some(r) = &mut self.ridge {
                        r.b = km;
                    }
                }
                (Drag::RidgeBody, _) => {
                    let d = response.drag_delta();
                    let (dx, dy) = ((d.x / t.scale) as f64, (-d.y / t.scale) as f64);
                    if let Some(r) = &mut self.ridge {
                        r.a.x += dx;
                        r.a.y += dy;
                        r.b.x += dx;
                        r.b.y += dy;
                    }
                }
                (Drag::Pan, _) => self.pan += response.drag_delta(),
                _ => {}
            }
        }
        if response.drag_stopped() {
            if let Some(Drag::NewRidge(start)) = self.drag
                && let Some(p) = response.interact_pointer_pos()
            {
                let end = self.clamp_to_arena(t.to_km(p));
                if start.dist(end) > 0.3 {
                    self.ridge = Some(Ridge { a: start, b: end });
                }
            }
            self.drag = None;
        }
        if response.clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            match self.hit(&t, p) {
                Some(Hit::Jammer(i)) => self.selected = Some(i),
                Some(Hit::Drone(i)) => self.selected_drone = Some(i),
                _ if self.tool == Tool::AddDrone && t.arena_rect().contains(p) => {
                    let position = self.clamp_to_arena(t.to_km(p));
                    let id = self.hand.len() as u32;
                    self.hand.push(Drone::new(
                        DroneId(id),
                        SquadId(self.new_squad),
                        self.new_role,
                        position,
                    ));
                    self.selected_drone = Some(self.hand.len() - 1);
                    self.sync_hand();
                }
                _ if self.tool == Tool::AddJammer && t.arena_rect().contains(p) => {
                    let position = self.clamp_to_arena(t.to_km(p));
                    self.jammers.push(Jammer {
                        position,
                        power: self.config.jam_p0,
                    });
                    self.selected = Some(self.jammers.len() - 1);
                }
                _ => {
                    self.selected = None;
                    self.selected_drone = None;
                }
            }
        }
        if response.secondary_clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            match self.hit(&t, p) {
                Some(Hit::Jammer(i)) => self.remove_jammer(i),
                Some(Hit::Drone(i)) => self.remove_drone(i),
                Some(Hit::Ridge | Hit::RidgeA | Hit::RidgeB) => self.ridge = None,
                None => {}
            }
        }
        if !ui.ctx().egui_wants_keyboard_input() {
            if let Some(i) = self.selected
                && ui.input(|inp| inp.key_pressed(Key::Delete))
            {
                self.remove_jammer(i);
            }
            if ui.input(|inp| inp.key_pressed(Key::Escape)) {
                self.selected = None;
                self.selected_drone = None;
            }
        }
        if self.scene == Scene::Hand {
            self.sync_hand();
        }
        self.probe = response
            .hover_pos()
            .filter(|p| t.arena_rect().contains(*p))
            .map(|p| t.to_km(p));

        // --- Heatmap (rebuilt only when something it depends on changed).
        let key = self.heat_key_now();
        if self.heat.is_none() || self.heat_key.as_ref() != Some(&key) {
            let img = self.heat_image();
            match &mut self.heat {
                Some(tex) => tex.set(img, TextureOptions::LINEAR),
                None => {
                    self.heat = Some(ui.ctx().load_texture(
                        "jam_heat",
                        img,
                        TextureOptions::LINEAR,
                    ));
                }
            }
            self.heat_key = Some(key);
        }
        let arena = t.arena_rect();
        painter.rect_filled(arena, 0.0, Color32::BLACK);
        if self.show_heat
            && let Some(tex) = &self.heat
        {
            painter.image(
                tex.id(),
                arena,
                Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        }

        // --- Grid every 5 km, with labels.
        let grid = Stroke::new(1.0, Color32::from_white_alpha(28));
        let label_color = text.gamma_multiply(0.7);
        let mut x = 0.0;
        while x <= aw + 1e-9 {
            let top = t.to_screen(Vec2 { x, y: ah });
            let bottom = t.to_screen(Vec2 { x, y: 0.0 });
            painter.line_segment([top, bottom], grid);
            painter.text(
                bottom + vec2(0.0, 4.0),
                Align2::CENTER_TOP,
                format!("{x}"),
                FontId::monospace(10.0),
                label_color,
            );
            x += 5.0;
        }
        let mut y = 0.0;
        while y <= ah + 1e-9 {
            let left = t.to_screen(Vec2 { x: 0.0, y });
            let right = t.to_screen(Vec2 { x: aw, y });
            painter.line_segment([left, right], grid);
            painter.text(
                left - vec2(4.0, 0.0),
                Align2::RIGHT_CENTER,
                format!("{y}"),
                FontId::monospace(10.0),
                label_color,
            );
            y += 5.0;
        }

        // --- Swarm layer.
        let truth = if self.show_swarm {
            self.truth()
        } else {
            Vec::new()
        };
        let reads = self.sensor_readings(&truth);
        let thr = self.threshold();
        let states = if self.show_swarm {
            self.link_states(&truth)
        } else {
            Vec::new()
        };
        // The link under the cursor (not when a drone is under it).
        let hovered_link: Option<usize> = if self.show_swarm && self.drag.is_none() {
            response
                .hover_pos()
                .filter(|p| self.drone_at(&t, *p).is_none())
                .and_then(|p| {
                    self.graph
                        .edges
                        .iter()
                        .enumerate()
                        .map(|(k, e)| {
                            let a = t.to_screen(self.swarm[e.src as usize].position);
                            let b = t.to_screen(self.swarm[e.dst as usize].position);
                            (k, point_segment_distance(p, a, b))
                        })
                        .filter(|&(_, d)| d <= 5.0)
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(k, _)| k)
                })
        } else {
            None
        };
        // "Build your own": every one-way direction, from your live_graph.
        let arrows: Vec<OneWayArrow> = if self.scene == Scene::Hand
            && self.show_one_way
            && self.link_rule == LinkRule::SignalVsJamming
        {
            self.one_way_arrows(&truth)
        } else {
            Vec::new()
        };
        // An arrow outside the geometric graph under the cursor (geometric ones are
        // covered by `hovered_link`).
        let hovered_arrow: Option<usize> = if hovered_link.is_none() && self.drag.is_none() {
            response
                .hover_pos()
                .filter(|p| self.drone_at(&t, *p).is_none())
                .and_then(|p| {
                    arrows
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| !a.geometric)
                        .map(|(k, a)| {
                            let from = t.to_screen(self.swarm[a.from].position);
                            let to = t.to_screen(self.swarm[a.to].position);
                            (k, point_segment_distance(p, from, to))
                        })
                        .filter(|&(_, d)| d <= 5.0)
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(k, _)| k)
                })
        } else {
            None
        };
        if self.show_swarm {
            let jammed = jammed_labels(&truth, &self.config);
            let at = |i: usize| t.to_screen(self.swarm[i].position);
            for (k, (e, state)) in self.graph.edges.iter().zip(&states).enumerate() {
                let (a, b) = (at(e.src as usize), at(e.dst as usize));
                if hovered_link == Some(k) {
                    painter.line_segment([a, b], Stroke::new(5.0, Color32::from_white_alpha(90)));
                }
                match *state {
                    LinkState::Live(Some(m)) if self.margin_colors => {
                        painter.line_segment(
                            [a, b],
                            Stroke::new(1.0 + 1.5 * e.weight as f32, margin_color(m)),
                        );
                    }
                    LinkState::Live(_) => {
                        let alpha = (60.0 + 160.0 * e.weight) as u8;
                        painter.line_segment(
                            [a, b],
                            Stroke::new(1.0, Color32::from_white_alpha(alpha)),
                        );
                    }
                    LinkState::OneWay { from, to }
                        if self.show_one_way && self.scene == Scene::Seed =>
                    {
                        draw_one_way(&painter, at(from), at(to));
                    }
                    LinkState::Dead if self.show_failed => {
                        painter.extend(Shape::dashed_line(
                            &[a, b],
                            Stroke::new(1.2, DEAD_COLOR),
                            4.0,
                            3.0,
                        ));
                    }
                    _ => {}
                }
            }
            let radius = (4.5 * self.zoom.sqrt()).clamp(3.5, 10.0);
            for (k, arrow) in arrows.iter().enumerate() {
                let color = if arrow.geometric {
                    ONE_WAY_COLOR
                } else {
                    RANGE_ONE_WAY_COLOR
                };
                let (from, to) = (at(arrow.from), at(arrow.to));
                if hovered_arrow == Some(k) {
                    painter
                        .line_segment([from, to], Stroke::new(6.0, Color32::from_white_alpha(90)));
                }
                draw_arrow(&painter, from, to, radius + 3.0, color);
                if self.show_margins {
                    let pos = from + (to - from) * 0.3;
                    let galley = painter.layout_no_wrap(
                        format!("{}→{} {:.2}", arrow.from, arrow.to, arrow.margin),
                        FontId::monospace(10.0),
                        color,
                    );
                    let pos = pos - galley.size() * 0.5;
                    painter.rect_filled(
                        Rect::from_min_size(pos, galley.size()).expand(2.0),
                        2.0,
                        Color32::from_black_alpha(170),
                    );
                    painter.galley(pos, galley, color);
                }
            }
            let p0 = self.config.jam_p0.max(1e-9);
            let outline = match self.drone_color {
                DroneColor::Role | DroneColor::Check => Stroke::new(1.0, Color32::BLACK),
                DroneColor::Truth | DroneColor::Reading => {
                    Stroke::new(1.0, Color32::from_white_alpha(180))
                }
            };
            for (i, d) in self.swarm.iter().enumerate() {
                let c = t.to_screen(d.position);
                let fill = match self.drone_color {
                    DroneColor::Role => role_color(d.role),
                    DroneColor::Truth => heat_color(self.scale_t(truth[i] / p0)),
                    DroneColor::Reading => heat_color(self.scale_t(reads[i] / p0)),
                    DroneColor::Check => check(jammed[i], reads[i], thr).color(),
                };
                draw_drone(&painter, c, radius, d.role, fill, outline);
                if jammed[i] {
                    painter.circle_stroke(
                        c,
                        radius + 3.0,
                        Stroke::new(2.0, Color32::from_rgb(255, 70, 70)),
                    );
                }
                if self.plot_hover == Some(i) {
                    painter.circle_stroke(c, radius + 8.0, Stroke::new(2.5, Color32::WHITE));
                }
                if self.scene == Scene::Hand {
                    if self.selected_drone == Some(i) {
                        painter.circle_stroke(c, radius + 6.0, Stroke::new(2.0, Color32::WHITE));
                    }
                    painter.text(
                        c + vec2(radius + 3.0, -radius - 2.0),
                        Align2::LEFT_BOTTOM,
                        format!("#{i} S{}", d.squad.0),
                        FontId::proportional(11.0),
                        Color32::WHITE,
                    );
                }
            }
            // Both margins on every link ("Build your own" tab).
            if self.scene == Scene::Hand && self.show_margins {
                for e in &self.graph.edges {
                    let (i, j) = (e.src as usize, e.dst as usize);
                    let ld = link_detail(
                        &self.swarm[i],
                        &self.swarm[j],
                        truth[i],
                        truth[j],
                        &self.config,
                    );
                    let mid = at(i) + (at(j) - at(i)) * 0.5;
                    let line = |m: f64, from: usize, to: usize| {
                        format!("{from}→{to} {m:.2}{}", if m >= 1.0 { "" } else { " ✘" })
                    };
                    let label = format!(
                        "{}\\n{}",
                        line(ld.margin_ij, i, j),
                        line(ld.margin_ji, j, i)
                    );
                    let galley =
                        painter.layout_no_wrap(label, FontId::monospace(10.0), Color32::WHITE);
                    let pos = mid - galley.size() * 0.5;
                    painter.rect_filled(
                        Rect::from_min_size(pos, galley.size()).expand(2.0),
                        2.0,
                        Color32::from_black_alpha(170),
                    );
                    painter.galley(pos, galley, Color32::WHITE);
                }
            }
        }

        // --- Ridge.
        if let Some(r) = self.ridge {
            let a = t.to_screen(r.a);
            let b = t.to_screen(r.b);
            painter.line_segment([a, b], Stroke::new(6.0, Color32::from_black_alpha(160)));
            painter.line_segment([a, b], Stroke::new(4.0, RIDGE_COLOR));
            for (end, name) in [(a, "a"), (b, "b")] {
                painter.circle(end, 5.0, RIDGE_COLOR, Stroke::new(1.5, Color32::WHITE));
                painter.text(
                    end + vec2(7.0, -7.0),
                    Align2::LEFT_BOTTOM,
                    name,
                    FontId::proportional(12.0),
                    Color32::WHITE,
                );
            }
        }
        if let Some(Drag::NewRidge(start)) = self.drag
            && let Some(p) = response.interact_pointer_pos()
        {
            let end = t.to_screen(self.clamp_to_arena(t.to_km(p)));
            painter.extend(Shape::dashed_line(
                &[t.to_screen(start), end],
                Stroke::new(3.0, RIDGE_COLOR),
                8.0,
                5.0,
            ));
        }

        // --- Lines from each jammer to the cursor: dashed blue when the ridge blocks them.
        if self.probe_lines
            && self.drag.is_none()
            && let Some(at) = self.probe
        {
            let cursor = t.to_screen(at);
            for j in &self.jammers {
                let from = t.to_screen(j.position);
                if self.blocked(j, at) {
                    painter.extend(Shape::dashed_line(
                        &[from, cursor],
                        Stroke::new(1.5, BLOCKED_COLOR),
                        6.0,
                        4.0,
                    ));
                } else {
                    painter.line_segment(
                        [from, cursor],
                        Stroke::new(1.0, Color32::from_white_alpha(140)),
                    );
                }
            }
        }

        // --- Jammers.
        for (i, j) in self.jammers.iter().enumerate() {
            let c = t.to_screen(j.position);
            if self.d0_rings {
                let ring = circle_points(c, t.len(self.config.jam_d0), 96);
                painter.extend(Shape::dashed_line(
                    &ring,
                    Stroke::new(1.0, Color32::from_white_alpha(150)),
                    5.0,
                    4.0,
                ));
            }
            draw_jammer(&painter, c, self.selected == Some(i));
            painter.text(
                c + vec2(10.0, -10.0),
                Align2::LEFT_BOTTOM,
                format!("J{i}  P0 {:.2}", j.power),
                FontId::proportional(12.0),
                Color32::WHITE,
            );
        }

        // --- Overlays.
        painter.text(
            response.rect.left_top() + vec2(8.0, 6.0),
            Align2::LEFT_TOP,
            match self.tool {
                Tool::Move => "Move: drag jammers / ridge · drag empty: pan · right-click: delete",
                Tool::AddJammer => "Add jammer: click the map",
                Tool::DrawRidge => "Draw ridge: drag from one end to the other",
                Tool::AddDrone => "Add drone: click the map · drag drones · right-click: delete",
            },
            FontId::proportional(12.0),
            text.gamma_multiply(0.8),
        );
        if let Some(at) = self.probe {
            painter.text(
                response.rect.right_top() + vec2(-8.0, 6.0),
                Align2::RIGHT_TOP,
                format!(
                    "({:.1}, {:.1}) km   power {:.3}",
                    at.x,
                    at.y,
                    self.total_power(at)
                ),
                FontId::monospace(12.0),
                text,
            );
        }
        if self.show_swarm
            && let Some(p) = response.hover_pos()
            && let Some(i) = self.drone_at(&t, p)
        {
            let d = &self.swarm[i];
            let (power, reading) = (truth[i], reads[i]);
            let label = power >= thr;
            let verdict = check(label, reading, thr);
            let (mut n_geo, mut n_live, mut n_out, mut n_in) = (0, 0, 0, 0);
            for (e, state) in self.graph.edges.iter().zip(&states) {
                if e.src as usize != i && e.dst as usize != i {
                    continue;
                }
                n_geo += 1;
                match *state {
                    LinkState::Live(_) => n_live += 1,
                    LinkState::OneWay { from, .. } if from == i => n_out += 1,
                    LinkState::OneWay { .. } => n_in += 1,
                    LinkState::Dead => {}
                }
            }
            let tx = d.role.tx_power(&self.config);
            let scale = d.role.signal_scale(&self.config);
            let rule = self.link_rule;
            response.on_hover_ui_at_pointer(|ui| {
                ui.strong(format!(
                    "Drone {} · squad S{} · {}",
                    d.id.0,
                    d.squad.0,
                    role_name(d.role)
                ));
                ui.label(RichText::new(format!("True power {power:.3}")).monospace());
                ui.label(
                    RichText::new(format!(
                        "Reading    {reading:.3}  (noise {:+.3})",
                        reading - power
                    ))
                    .monospace(),
                );
                ui.label(if label {
                    "Label: jammed (in the zone)"
                } else {
                    "Label: not jammed"
                });
                ui.label(RichText::new(verdict.name()).color(verdict.color()));
                ui.separator();
                ui.label(format!("Links: {n_live} live of {n_geo} geometric"));
                if rule == LinkRule::SignalVsJamming {
                    ui.label(format!(
                        "One-way: {n_out} out (heard, cannot hear), {n_in} in"
                    ));
                    ui.label(
                        RichText::new(format!("tx power {tx:.2} · scale {scale:.2} km"))
                            .monospace(),
                    );
                }
            });
        } else if let Some(k) = hovered_link {
            let e = &self.graph.edges[k];
            let (i, j) = (e.src as usize, e.dst as usize);
            let (a, b) = (&self.swarm[i], &self.swarm[j]);
            let ld = link_detail(a, b, truth[i], truth[j], &self.config);
            let n0 = self.config.noise_floor;
            let state = states[k];
            let rule = self.link_rule;
            let weight = e.weight;
            response.on_hover_ui_at_pointer(|ui| {
                ui.strong(format!(
                    "Link #{i} {} – #{j} {}",
                    role_name(a.role),
                    role_name(b.role)
                ));
                ui.label(
                    RichText::new(format!(
                        "distance {:.2} km · weight {weight:.3} (Gaussian)",
                        ld.dist
                    ))
                    .monospace(),
                );
                let (text, color) = match state {
                    LinkState::Live(Some(m)) => (format!("live, margin {m:.2}"), margin_color(m)),
                    LinkState::Live(None) => ("live".to_string(), Color32::WHITE),
                    LinkState::OneWay { from, to } => {
                        (format!("one-way: only #{from} → #{to}"), ONE_WAY_COLOR)
                    }
                    LinkState::Dead => ("dead".to_string(), DEAD_COLOR),
                };
                ui.label(RichText::new(text).strong().color(color));
                ui.separator();
                ui.label(
                    RichText::new("direction  signal  N0+jam  margin")
                        .monospace()
                        .weak(),
                );
                for (from, to, signal, jam, m) in [
                    (i, j, ld.signal_ij, truth[j], ld.margin_ij),
                    (j, i, ld.signal_ji, truth[i], ld.margin_ji),
                ] {
                    ui.label(
                        RichText::new(format!(
                            "#{from} → #{to}  {signal:>6.3}  {:>6.3}  {m:>6.2} {}",
                            n0 + jam,
                            if m >= 1.0 { "✔" } else { "✘" }
                        ))
                        .monospace(),
                    );
                }
                ui.weak("Each direction is jammed at its receiver. A link needs both.");
                if rule == LinkRule::EitherEnd {
                    ui.weak("The state above follows the D02 rule; the margins are D03.");
                }
            });
        } else if let Some(k) = hovered_arrow {
            let arrow = &arrows[k];
            let (i, j) = (arrow.from, arrow.to);
            let (a, b) = (&self.swarm[i], &self.swarm[j]);
            let ld = link_detail(a, b, truth[i], truth[j], &self.config);
            let n0 = self.config.noise_floor;
            let (range_a, range_b) = (
                a.role.radio_range(&self.config),
                b.role.radio_range(&self.config),
            );
            response.on_hover_ui_at_pointer(|ui| {
                ui.strong(format!(
                    "One-way #{i} {} → #{j} {}",
                    role_name(a.role),
                    role_name(b.role)
                ));
                ui.label(
                    RichText::new("not a link: outside the geometric graph")
                        .color(RANGE_ONE_WAY_COLOR),
                );
                ui.label(
                    RichText::new(format!(
                        "distance {:.2} km · ranges {range_a:.1} and {range_b:.1} km",
                        ld.dist
                    ))
                    .monospace(),
                );
                ui.separator();
                ui.label(
                    RichText::new("direction  signal  N0+jam  margin")
                        .monospace()
                        .weak(),
                );
                for (from, to, signal, jam, m) in [
                    (i, j, ld.signal_ij, truth[j], ld.margin_ij),
                    (j, i, ld.signal_ji, truth[i], ld.margin_ji),
                ] {
                    ui.label(
                        RichText::new(format!(
                            "#{from} → #{to}  {signal:>6.3}  {:>6.3}  {m:>6.2} {}",
                            n0 + jam,
                            if m >= 1.0 { "✔" } else { "✘" }
                        ))
                        .monospace(),
                    );
                }
                ui.weak(
                    "The sender's range reaches the receiver, the receiver's does not \
                     reach back. live_graph lists it in range_one_way (D03, option b).",
                );
            });
        }
    }

    /// What is under screen point `p`: a jammer first, then a ridge end, then the ridge.
    fn hit(&self, t: &Transform, p: Pos2) -> Option<Hit> {
        let nearest = self
            .jammers
            .iter()
            .enumerate()
            .map(|(i, j)| (i, t.to_screen(j.position).distance(p)))
            .filter(|&(_, d)| d <= GRAB_JAMMER)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, _)) = nearest {
            return Some(Hit::Jammer(i));
        }
        if self.scene == Scene::Hand
            && let Some(i) = self.drone_at(t, p)
        {
            return Some(Hit::Drone(i));
        }
        let r = self.ridge?;
        let (a, b) = (t.to_screen(r.a), t.to_screen(r.b));
        if a.distance(p) <= GRAB_END {
            Some(Hit::RidgeA)
        } else if b.distance(p) <= GRAB_END {
            Some(Hit::RidgeB)
        } else if point_segment_distance(p, a, b) <= GRAB_RIDGE {
            Some(Hit::Ridge)
        } else {
            None
        }
    }

    fn drone_at(&self, t: &Transform, p: Pos2) -> Option<usize> {
        self.swarm
            .iter()
            .enumerate()
            .map(|(i, d)| (i, t.to_screen(d.position).distance(p)))
            .filter(|&(_, d)| d <= 8.0)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Hit {
    Jammer(usize),
    Drone(usize),
    RidgeA,
    RidgeB,
    Ridge,
}

/// Power against distance, open ground (solid) and behind the ridge (dashed),
/// with the jammed threshold and d0 marked. Hover for a readout.
fn falloff_plot(ui: &mut egui::Ui, cfg: &SimConfig) {
    let d0 = cfg.jam_d0;
    let d_max = 4.0 * d0;
    let width = ui.available_width().max(200.0);
    let (response, painter) = ui.allocate_painter(vec2(width, 220.0), Sense::hover());
    let rect = response.rect;
    let plot = Rect::from_min_max(rect.min + vec2(36.0, 10.0), rect.max - vec2(10.0, 28.0));
    let x = |d: f64| plot.left() + (d / d_max) as f32 * plot.width();
    let y = |p: f64| plot.bottom() - p as f32 * plot.height();
    let visuals = ui.visuals();
    let text = visuals.text_color();
    let weak = visuals.weak_text_color();
    let small = FontId::proportional(11.0);

    painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
    for p in [0.0, 0.25, 0.5, 0.75, 1.0] {
        painter.hline(
            plot.x_range(),
            y(p),
            Stroke::new(1.0, weak.gamma_multiply(0.25)),
        );
        painter.text(
            pos2(plot.left() - 4.0, y(p)),
            Align2::RIGHT_CENTER,
            format!("{p}"),
            small.clone(),
            weak,
        );
    }
    for k in 0..=4 {
        let d = k as f64 * d0;
        painter.vline(
            x(d),
            plot.y_range(),
            Stroke::new(1.0, weak.gamma_multiply(0.15)),
        );
        painter.text(
            pos2(x(d), plot.bottom() + 3.0),
            Align2::CENTER_TOP,
            format!("{d:.0}"),
            small.clone(),
            weak,
        );
    }
    painter.text(
        pos2(plot.center().x, rect.bottom()),
        Align2::CENTER_BOTTOM,
        "distance d (km)",
        small.clone(),
        weak,
    );

    let open = |d: f64| 1.0 / (1.0 + (d / d0) * (d / d0));
    let n = 200;
    let pts = |f: &dyn Fn(f64) -> f64| -> Vec<Pos2> {
        (0..=n)
            .map(|k| d_max * k as f64 / n as f64)
            .map(|d| pos2(x(d), y(f(d))))
            .collect()
    };
    painter.line(pts(&open), Stroke::new(2.5, JAMMER_COLOR));
    let behind = pts(&|d| open(d) * cfg.ridge_factor);
    painter.extend(Shape::dashed_line(
        &behind,
        Stroke::new(1.5, BLOCKED_COLOR),
        5.0,
        4.0,
    ));

    // Threshold and d0.
    let thr = cfg.jam_threshold;
    painter.extend(Shape::dashed_line(
        &[pos2(plot.left(), y(thr)), pos2(plot.right(), y(thr))],
        Stroke::new(1.0, text.gamma_multiply(0.8)),
        4.0,
        3.0,
    ));
    painter.text(
        pos2(plot.right() - 2.0, y(thr) - 2.0),
        Align2::RIGHT_BOTTOM,
        "jammed above",
        small.clone(),
        text,
    );
    painter.circle_filled(pos2(x(d0), y(open(d0))), 3.5, JAMMER_COLOR);
    painter.text(
        pos2(x(d0) + 5.0, y(open(d0)) - 3.0),
        Align2::LEFT_BOTTOM,
        "d0",
        FontId::proportional(13.0),
        JAMMER_COLOR,
    );
    painter.text(
        plot.left_top() + vec2(6.0, 4.0),
        Align2::LEFT_TOP,
        "— open ground   - - behind ridge",
        small,
        text.gamma_multiply(0.8),
    );

    if let Some(p) = response.hover_pos()
        && plot.contains(p)
    {
        let d = ((p.x - plot.left()) / plot.width()) as f64 * d_max;
        painter.vline(
            p.x,
            plot.y_range(),
            Stroke::new(1.0, text.gamma_multiply(0.5)),
        );
        let label = format!(
            "d = {d:.2} km\nopen   {:.3}\nbehind {:.3}",
            open(d),
            open(d) * cfg.ridge_factor
        );
        let galley = painter.layout_no_wrap(label, FontId::monospace(11.0), text);
        let mut pos = p + vec2(8.0, 8.0);
        if pos.x + galley.size().x > rect.right() {
            pos.x = p.x - 8.0 - galley.size().x;
        }
        painter.rect_filled(
            Rect::from_min_size(pos, galley.size()).expand(3.0),
            3.0,
            visuals.window_fill,
        );
        painter.galley(pos, galley, text);
    }
}

/// Each drone as a dot: true power (x) against its reading (y), both as fractions of
/// P0, with the diagonal (no noise) and the threshold on both axes. Returns the drone
/// under the cursor.
fn truth_reading_plot(
    ui: &mut egui::Ui,
    truth: &[f64],
    reads: &[f64],
    threshold: f64,
    p0: f64,
) -> Option<usize> {
    let p0 = p0.max(1e-9);
    let thr = threshold / p0;
    let xs: Vec<f64> = truth.iter().map(|v| v / p0).collect();
    let ys: Vec<f64> = reads.iter().map(|v| v / p0).collect();
    let mut lo: f64 = 0.0;
    let mut hi: f64 = (thr * 1.6).max(0.2);
    for &v in xs.iter().chain(&ys) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let pad = 0.05 * (hi - lo);
    let (lo, hi) = (lo - pad, hi + pad);

    let width = ui.available_width().max(200.0);
    let side = (width - 50.0).min(300.0);
    let (response, painter) = ui.allocate_painter(vec2(width, side + 40.0), Sense::hover());
    let rect = response.rect;
    let plot = Rect::from_min_size(rect.min + vec2(40.0, 8.0), vec2(side, side));
    let sx = |v: f64| plot.left() + ((v - lo) / (hi - lo)) as f32 * plot.width();
    let sy = |v: f64| plot.bottom() - ((v - lo) / (hi - lo)) as f32 * plot.height();
    let visuals = ui.visuals();
    let text = visuals.text_color();
    let weak = visuals.weak_text_color();
    let small = FontId::proportional(11.0);

    painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
    let step = if hi - lo > 2.0 { 0.5 } else { 0.25 };
    let mut k = (lo / step).ceil() * step;
    while k <= hi {
        let grid = Stroke::new(1.0, weak.gamma_multiply(0.2));
        painter.hline(plot.x_range(), sy(k), grid);
        painter.vline(sx(k), plot.y_range(), grid);
        painter.text(
            pos2(plot.left() - 4.0, sy(k)),
            Align2::RIGHT_CENTER,
            format!("{k}"),
            small.clone(),
            weak,
        );
        painter.text(
            pos2(sx(k), plot.bottom() + 3.0),
            Align2::CENTER_TOP,
            format!("{k}"),
            small.clone(),
            weak,
        );
        k += step;
    }
    painter.text(
        pos2(plot.center().x, rect.bottom()),
        Align2::CENTER_BOTTOM,
        "x: true power   ·   y: reading   (× P0)",
        small.clone(),
        weak,
    );

    // Diagonal (reading = truth) and the threshold on both axes.
    painter.line_segment(
        [pos2(sx(lo), sy(lo)), pos2(sx(hi), sy(hi))],
        Stroke::new(1.0, text.gamma_multiply(0.35)),
    );
    let dash = Stroke::new(1.0, text.gamma_multiply(0.8));
    painter.extend(Shape::dashed_line(
        &[pos2(sx(thr), plot.top()), pos2(sx(thr), plot.bottom())],
        dash,
        4.0,
        3.0,
    ));
    painter.extend(Shape::dashed_line(
        &[pos2(plot.left(), sy(thr)), pos2(plot.right(), sy(thr))],
        dash,
        4.0,
        3.0,
    ));
    painter.text(
        plot.left_top() + vec2(4.0, 3.0),
        Align2::LEFT_TOP,
        "false alarm",
        small.clone(),
        Check::FalseAlarm.color(),
    );
    painter.text(
        plot.right_bottom() - vec2(4.0, 3.0),
        Align2::RIGHT_BOTTOM,
        "missed",
        small.clone(),
        Check::Missed.color(),
    );

    for (i, (&x, &y)) in xs.iter().zip(&ys).enumerate() {
        let c = check(truth[i] >= threshold, reads[i], threshold);
        painter.circle_filled(pos2(sx(x), sy(y)), 3.0, c.color());
    }

    let hovered = response.hover_pos().and_then(|p| {
        xs.iter()
            .zip(&ys)
            .enumerate()
            .map(|(i, (&x, &y))| (i, pos2(sx(x), sy(y)).distance(p)))
            .filter(|&(_, d)| d <= 6.0)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    });
    if let Some(i) = hovered {
        let c = pos2(sx(xs[i]), sy(ys[i]));
        painter.circle_stroke(c, 6.0, Stroke::new(1.5, text));
        let label = format!(
            "drone {i}\ntruth   {:.3}\nreading {:.3}",
            truth[i], reads[i]
        );
        let galley = painter.layout_no_wrap(label, FontId::monospace(11.0), text);
        let mut pos = c + vec2(8.0, 8.0);
        if pos.x + galley.size().x > rect.right() {
            pos.x = c.x - 8.0 - galley.size().x;
        }
        painter.rect_filled(
            Rect::from_min_size(pos, galley.size()).expand(3.0),
            3.0,
            visuals.window_fill,
        );
        painter.galley(pos, galley, text);
    }
    hovered
}

/// Matches the live graph back onto the geometric edges. `live_graph` walks the geometric
/// edges in order, so `graph.edges` and `cut_one_way` are in that order too: one pass
/// with two cursors classifies every link.
fn link_states_from(geometric: &EdgeList, live: &LiveGraph) -> Vec<LinkState> {
    let (mut l, mut o) = (0, 0);
    geometric
        .edges
        .iter()
        .map(|e| {
            let pair = (e.src, e.dst);
            if live
                .graph
                .edges
                .get(l)
                .is_some_and(|x| (x.src, x.dst) == pair)
            {
                l += 1;
                return LinkState::Live(Some(live.margins[l - 1]));
            }
            if let Some(&(from, to)) = live.cut_one_way.get(o)
                && (from.min(to), from.max(to)) == pair
            {
                o += 1;
                return LinkState::OneWay {
                    from: from as usize,
                    to: to as usize,
                };
            }
            LinkState::Dead
        })
        .collect()
}

fn hand_drone(id: u32, squad: u16, role: Role, x: f64, y: f64) -> Drone {
    Drone::new(DroneId(id), SquadId(squad), role, Vec2 { x, y })
}

/// Both directions of one link, for display.
struct LinkDetail {
    dist: f64,
    signal_ij: f64,
    signal_ji: f64,
    /// Direction i -> j, jammed at j.
    margin_ij: f64,
    /// Direction j -> i, jammed at i.
    margin_ji: f64,
}

/// The same formula as `swarmsim::links::margin` (private there; a test checks they
/// agree): signal = tx / (1 + (d / scale)²) with the sender's tx and scale,
/// margin = signal / (N0 + jamming at the receiver).
fn link_detail(a: &Drone, b: &Drone, jam_a: f64, jam_b: f64, cfg: &SimConfig) -> LinkDetail {
    let dist = a.position.dist(b.position);
    let signal =
        |s: &Drone| s.role.tx_power(cfg) / (1.0 + (dist / s.role.signal_scale(cfg)).powi(2));
    let (signal_ij, signal_ji) = (signal(a), signal(b));
    LinkDetail {
        dist,
        signal_ij,
        signal_ji,
        margin_ij: signal_ij / (cfg.noise_floor + jam_b),
        margin_ji: signal_ji / (cfg.noise_floor + jam_a),
    }
}

/// One direction that works while the reverse does not.
struct OneWayArrow {
    from: usize,
    to: usize,
    /// Margin of the working direction (≥ 1).
    margin: f64,
    /// True when the pair is a link of the geometric graph (cut one way by jamming);
    /// false when it is not (the two ranges differ).
    geometric: bool,
}

#[cfg(test)]
fn is_geometric_pair(g: &EdgeList, a: usize, b: usize) -> bool {
    let pair = (a.min(b) as u32, a.max(b) as u32);
    g.edges.iter().any(|e| (e.src, e.dst) == pair)
}

/// Every ordered pair where only one direction works, by brute force over all pairs.
/// Test oracle for `live_graph`'s one-way list (the GUI uses `live_graph` itself).
#[cfg(test)]
fn one_way_all(
    swarm: &[Drone],
    truth: &[f64],
    geometric: &EdgeList,
    cfg: &SimConfig,
) -> Vec<OneWayArrow> {
    let mut arrows = Vec::new();
    for i in 0..swarm.len() {
        for j in (i + 1)..swarm.len() {
            let ld = link_detail(&swarm[i], &swarm[j], truth[i], truth[j], cfg);
            let (fwd, back) = (ld.margin_ij >= 1.0, ld.margin_ji >= 1.0);
            if fwd == back {
                continue; // both work (a live link) or neither does
            }
            let in_graph = is_geometric_pair(geometric, i, j);
            let (from, to, margin) = if fwd {
                (i, j, ld.margin_ij)
            } else {
                (j, i, ld.margin_ji)
            };
            arrows.push(OneWayArrow {
                from,
                to,
                margin,
                geometric: in_graph,
            });
        }
    }
    arrows
}

/// A solid arrow from `from` to `to`, its tip stopping `gap` pixels short of `to`.
fn draw_arrow(painter: &egui::Painter, from: Pos2, to: Pos2, gap: f32, color: Color32) {
    let v = to - from;
    if v.length() < 2.0 * gap {
        return;
    }
    let dir = v.normalized();
    let normal = vec2(-dir.y, dir.x);
    let tip = to - dir * gap;
    let base = tip - dir * 9.0;
    painter.line_segment([from + dir * gap, base], Stroke::new(2.0, color));
    painter.add(Shape::convex_polygon(
        vec![tip, base + normal * 5.0, base - normal * 5.0],
        color,
        Stroke::NONE,
    ));
}

/// Jamming power at `dist` km from one jammer of power P0, in open ground.
fn open_jam(dist: f64, cfg: &SimConfig) -> f64 {
    cfg.jam_p0 / (1.0 + (dist / cfg.jam_d0).powi(2))
}

/// How far a sender of `role` is heard by a receiver with jamming `jam`: solves
/// tx / (1 + (d / scale)²) = N0 + jam for d. None when it is not heard even next to it.
fn heard_range(role: Role, jam: f64, cfg: &SimConfig) -> Option<f64> {
    let ratio = role.tx_power(cfg) / (cfg.noise_floor + jam);
    (ratio >= 1.0).then(|| role.signal_scale(cfg) * (ratio - 1.0).sqrt())
}

fn range_text(r: Option<f64>) -> String {
    match r {
        Some(r) => format!("{r:.1} km"),
        None => "not heard".to_string(),
    }
}

/// Orange at margin 1 (about to break) to green at margin 10 and above, on a log scale.
fn margin_color(m: f64) -> Color32 {
    let t = (m.max(1.0).log10()).clamp(0.0, 1.0) as f32;
    let mix = |a: f32, b: f32| (a + (b - a) * t) as u8;
    Color32::from_rgb(mix(255.0, 120.0), mix(140.0, 220.0), mix(40.0, 120.0))
}

/// A one-way link: dashed, with an arrowhead pointing at the receiver.
fn draw_one_way(painter: &egui::Painter, from: Pos2, to: Pos2) {
    painter.extend(Shape::dashed_line(
        &[from, to],
        Stroke::new(1.3, ONE_WAY_COLOR),
        5.0,
        3.0,
    ));
    let dir = (to - from).normalized();
    let normal = vec2(-dir.y, dir.x);
    let tip = from + (to - from) * 0.65;
    painter.add(Shape::convex_polygon(
        vec![
            tip + dir * 6.0,
            tip - dir * 4.0 + normal * 4.5,
            tip - dir * 4.0 - normal * 4.5,
        ],
        ONE_WAY_COLOR,
        Stroke::NONE,
    ));
}

/// Range at which each role is heard, against the receiver's distance to the jammer
/// (open ground, 0 to 20 km). Gaps where a role is not heard at all. Hover for values.
fn range_plot(ui: &mut egui::Ui, cfg: &SimConfig) {
    let d_max = 20.0;
    let r_max = Role::ALL
        .iter()
        .filter_map(|&r| heard_range(r, 0.0, cfg))
        .fold(1.0, f64::max)
        * 1.05;
    let width = ui.available_width().max(200.0);
    let (response, painter) = ui.allocate_painter(vec2(width, 200.0), Sense::hover());
    let rect = response.rect;
    let plot = Rect::from_min_max(rect.min + vec2(36.0, 8.0), rect.max - vec2(10.0, 28.0));
    let x = |d: f64| plot.left() + (d / d_max) as f32 * plot.width();
    let y = |r: f64| plot.bottom() - (r / r_max) as f32 * plot.height();
    let visuals = ui.visuals();
    let text = visuals.text_color();
    let weak = visuals.weak_text_color();
    let small = FontId::proportional(11.0);

    painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
    let step = if r_max > 10.0 { 4.0 } else { 1.0 };
    let mut r = 0.0;
    while r <= r_max {
        painter.hline(
            plot.x_range(),
            y(r),
            Stroke::new(1.0, weak.gamma_multiply(0.2)),
        );
        painter.text(
            pos2(plot.left() - 4.0, y(r)),
            Align2::RIGHT_CENTER,
            format!("{r}"),
            small.clone(),
            weak,
        );
        r += step;
    }
    for d in [0.0, 4.0, 8.0, 12.0, 16.0, 20.0] {
        painter.vline(
            x(d),
            plot.y_range(),
            Stroke::new(1.0, weak.gamma_multiply(0.15)),
        );
        painter.text(
            pos2(x(d), plot.bottom() + 3.0),
            Align2::CENTER_TOP,
            format!("{d}"),
            small.clone(),
            weak,
        );
    }
    painter.text(
        pos2(plot.center().x, rect.bottom()),
        Align2::CENTER_BOTTOM,
        "receiver's distance to the jammer (km)  ·  y: range heard (km)",
        small.clone(),
        weak,
    );

    let n = 200;
    for role in Role::ALL {
        let mut run: Vec<Pos2> = Vec::new();
        for k in 0..=n {
            let d = d_max * k as f64 / n as f64;
            match heard_range(role, open_jam(d, cfg), cfg) {
                Some(r) => run.push(pos2(x(d), y(r))),
                None => {
                    if run.len() > 1 {
                        painter.line(std::mem::take(&mut run), Stroke::new(2.0, role_color(role)));
                    }
                    run.clear();
                }
            }
        }
        if run.len() > 1 {
            painter.line(run, Stroke::new(2.0, role_color(role)));
        }
    }

    if let Some(p) = response.hover_pos()
        && plot.contains(p)
    {
        let d = ((p.x - plot.left()) / plot.width()) as f64 * d_max;
        painter.vline(
            p.x,
            plot.y_range(),
            Stroke::new(1.0, text.gamma_multiply(0.5)),
        );
        let jam = open_jam(d, cfg);
        let mut label = format!("receiver at {d:.1} km, jamming {jam:.3}");
        for role in Role::ALL {
            label.push_str(&format!(
                "\n{:<10} {}",
                role_name(role),
                range_text(heard_range(role, jam, cfg))
            ));
        }
        let galley = painter.layout_no_wrap(label, FontId::monospace(11.0), text);
        let mut pos = p + vec2(8.0, 8.0);
        if pos.x + galley.size().x > rect.right() {
            pos.x = p.x - 8.0 - galley.size().x;
        }
        painter.rect_filled(
            Rect::from_min_size(pos, galley.size()).expand(3.0),
            3.0,
            visuals.window_fill,
        );
        painter.galley(pos, galley, text);
    }
}

/// A jammer marker: a red disc with four short rays.
fn draw_jammer(painter: &egui::Painter, c: Pos2, selected: bool) {
    let stroke = Stroke::new(2.0, Color32::WHITE);
    for k in 0..4 {
        let a = std::f32::consts::FRAC_PI_4 + k as f32 * std::f32::consts::FRAC_PI_2;
        let dir = vec2(a.cos(), a.sin());
        painter.line_segment([c + dir * 9.0, c + dir * 14.0], stroke);
    }
    painter.circle(c, 7.0, JAMMER_COLOR, stroke);
    if selected {
        painter.circle_stroke(c, 17.0, Stroke::new(2.0, Color32::WHITE));
    }
}

/// Dark purple (0) to orange to pale yellow (1), close to the "inferno" map.
fn heat_color(t: f32) -> Color32 {
    const STOPS: [[f32; 3]; 5] = [
        [10.0, 7.0, 30.0],
        [87.0, 16.0, 110.0],
        [188.0, 55.0, 84.0],
        [249.0, 142.0, 9.0],
        [252.0, 255.0, 164.0],
    ];
    let t = t.clamp(0.0, 1.0) * (STOPS.len() - 1) as f32;
    let i = (t.floor() as usize).min(STOPS.len() - 2);
    let u = t - i as f32;
    let mix = |k: usize| (STOPS[i][k] + (STOPS[i + 1][k] - STOPS[i][k]) * u) as u8;
    Color32::from_rgb(mix(0), mix(1), mix(2))
}

/// Paints `top` (with its alpha) over the opaque `base`.
fn blend(base: Color32, top: Color32) -> Color32 {
    let a = top.a() as f32 / 255.0;
    let mix = |b: u8, t: u8| (b as f32 * (1.0 - a) + t as f32 * a) as u8;
    // `top` is premultiplied; un-premultiply its colour before mixing.
    let un = |c: u8| {
        if a > 0.0 {
            (c as f32 / a).min(255.0) as u8
        } else {
            0
        }
    };
    Color32::from_rgb(
        mix(base.r(), un(top.r())),
        mix(base.g(), un(top.g())),
        mix(base.b(), un(top.b())),
    )
}

/// Distance in pixels from `p` to the segment a-b.
fn point_segment_distance(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    if len2 == 0.0 {
        return p.distance(a);
    }
    let u = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    p.distance(a + ab * u)
}

/// Points on a circle, closed.
fn circle_points(c: Pos2, r: f32, n: usize) -> Vec<Pos2> {
    (0..=n)
        .map(|k| {
            let a = k as f32 / n as f32 * std::f32::consts::TAU;
            pos2(c.x + r * a.cos(), c.y + r * a.sin())
        })
        .collect()
}

/// km to screen and back: the arena fitted into the panel, then zoom and pan; y up.
struct Transform {
    left: f32,
    bottom: f32,
    scale: f32,
    w: f64,
    h: f64,
}

impl Transform {
    fn fit(rect: Rect, w: f64, h: f64, zoom: f32, pan: EVec2) -> Self {
        let margin = 30.0;
        let fit = ((rect.width() - 2.0 * margin) / w as f32)
            .min((rect.height() - 2.0 * margin) / h as f32)
            .max(1.0);
        let scale = fit * zoom;
        Transform {
            left: rect.center().x - w as f32 * scale / 2.0 + pan.x,
            bottom: rect.center().y + h as f32 * scale / 2.0 + pan.y,
            scale,
            w,
            h,
        }
    }

    fn to_screen(&self, p: Vec2) -> Pos2 {
        pos2(
            self.left + p.x as f32 * self.scale,
            self.bottom - p.y as f32 * self.scale,
        )
    }

    fn to_km(&self, p: Pos2) -> Vec2 {
        Vec2 {
            x: ((p.x - self.left) / self.scale) as f64,
            y: ((self.bottom - p.y) / self.scale) as f64,
        }
    }

    fn len(&self, km: f64) -> f32 {
        km as f32 * self.scale
    }

    fn arena_rect(&self) -> Rect {
        Rect::from_min_max(
            self.to_screen(Vec2 { x: 0.0, y: self.h }),
            self.to_screen(Vec2 { x: self.w, y: 0.0 }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: f64, y: f64) -> Vec2 {
        Vec2 { x, y }
    }

    #[test]
    fn test_setup_matches_the_unit_test_values() {
        let app = JamApp::default(); // loads the test setup
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        assert!(close(app.total_power(at(10.0, 14.0)), 0.5)); // d0, open
        assert!(close(app.total_power(at(14.0, 10.0)), 0.05)); // d0, behind the ridge
        assert!(close(app.total_power(at(12.0, 10.0)), 0.8)); // on the ridge
        assert!(app.blocked(&app.jammers[0], at(14.0, 10.0)));
        assert!(!app.blocked(&app.jammers[0], at(10.0, 14.0)));
    }

    #[test]
    fn no_ridge_means_nothing_is_blocked() {
        let app = JamApp {
            ridge: None,
            ..Default::default()
        };
        assert!((app.total_power(at(14.0, 10.0)) - 0.5).abs() < 1e-12);
        assert!(!app.blocked(&app.jammers[0], at(14.0, 10.0)));
    }

    #[test]
    fn sum_and_max_combine_two_jammers() {
        let mut app = JamApp {
            ridge: None,
            ..Default::default()
        };
        app.jammers = vec![
            Jammer {
                position: at(10.0, 10.0),
                power: 1.0,
            },
            Jammer {
                position: at(18.0, 10.0),
                power: 1.0,
            },
        ];
        // Midpoint: 4 km from both, 0.5 each.
        app.combine = Combine::Sum;
        assert!((app.total_power(at(14.0, 10.0)) - 1.0).abs() < 1e-12);
        app.combine = Combine::Max;
        assert!((app.total_power(at(14.0, 10.0)) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn heat_image_has_one_pixel_per_res_km() {
        let app = JamApp::default();
        let img = app.heat_image();
        let cfg = SimConfig::default();
        assert_eq!(
            img.size,
            [
                (cfg.arena_w / RES).round() as usize,
                (cfg.arena_h / RES).round() as usize
            ]
        );
    }

    #[test]
    fn jammed_drones_lose_all_their_links() {
        let mut app = JamApp {
            show_swarm: true,
            ..Default::default()
        };
        app.update_swarm();
        // Put the jammer on drone 0.
        app.jammers[0].position = app.swarm[0].position;
        let jammed = app.jammed();
        assert!(jammed[0]);
        let after = fail_links(&app.graph, &jammed);
        assert!(
            after
                .edges
                .iter()
                .all(|e| !jammed[e.src as usize] && !jammed[e.dst as usize])
        );
        assert!(after.edges.len() < app.graph.edges.len());
    }

    #[test]
    fn place_from_seed_matches_the_dataset_order() {
        let mut app = JamApp {
            seed: 4,
            ..Default::default()
        };
        app.place_from_seed();
        // Same calls, same RNG order as the dataset: spawn, then place.
        let cfg = SimConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let swarm = spawn_swarm(&cfg, &mut rng);
        let (jammer, ridge) = place_jamming(&swarm, &cfg, &mut rng);
        assert_eq!(app.swarm, swarm);
        assert_eq!(app.jammers, vec![jammer]);
        assert_eq!(app.ridge, Some(ridge));
        assert_eq!(app.placed_for, Some(4));
        // The placement respects the bounds.
        let f = app.jammed().iter().filter(|&&j| j).count() as f64 / app.swarm.len() as f64;
        assert!((cfg.jam_frac_min..=cfg.jam_frac_max).contains(&f));
    }

    #[test]
    fn p0_cancels_out_of_who_is_jammed() {
        // The threshold is a fraction of P0 and the placed jammer has power P0,
        // so doubling P0 doubles every power and the threshold: same placement.
        let mut a = JamApp {
            seed: 9,
            ..Default::default()
        };
        a.place_from_seed();
        let mut b = JamApp {
            seed: 9,
            ..Default::default()
        };
        b.config.jam_p0 = 2.0;
        b.place_from_seed();
        assert_eq!(a.jammed(), b.jammed());
        assert_eq!(a.ridge, b.ridge);
        assert_eq!(a.jammers[0].position, b.jammers[0].position);
    }

    #[test]
    fn readings_follow_the_dataset_order() {
        let mut app = JamApp {
            seed: 4,
            ..Default::default()
        };
        app.place_from_seed();
        // The dataset: spawn, place, truth, readings, all with one RNG.
        let cfg = SimConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let swarm = spawn_swarm(&cfg, &mut rng);
        let (jammer, ridge) = place_jamming(&swarm, &cfg, &mut rng);
        let truth = true_power(&swarm, &jammer, &ridge, &cfg);
        let reads = readings(&truth, &cfg, &mut rng);
        assert_eq!(app.truth(), truth);
        assert_eq!(app.sensor_readings(&app.truth()), reads);
    }

    /// Reading minus truth, per drone.
    fn noise(app: &JamApp) -> Vec<f64> {
        let truth = app.truth();
        app.sensor_readings(&truth)
            .iter()
            .zip(&truth)
            .map(|(r, t)| r - t)
            .collect()
    }

    #[test]
    fn moving_the_jammer_changes_the_truth_not_the_noise() {
        let mut app = JamApp {
            seed: 4,
            ..Default::default()
        };
        app.place_from_seed();
        let truth_before = app.truth();
        let noise_before = noise(&app);
        app.jammers[0].position.x += 3.0;
        assert_ne!(app.truth(), truth_before);
        for (a, b) in noise(&app).iter().zip(&noise_before) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn a_new_draw_changes_the_noise() {
        let mut app = JamApp {
            seed: 4,
            ..Default::default()
        };
        app.place_from_seed();
        let dataset = noise(&app);
        app.noise_draw = 1;
        assert_ne!(noise(&app), dataset);
        app.noise_draw = 0;
        assert_eq!(noise(&app), dataset);
    }

    #[test]
    fn check_sorts_label_against_reading() {
        assert_eq!(check(true, 0.6, 0.5), Check::Hit);
        assert_eq!(check(true, 0.4, 0.5), Check::Missed);
        assert_eq!(check(false, 0.6, 0.5), Check::FalseAlarm);
        assert_eq!(check(false, 0.4, 0.5), Check::Clear);
        assert_eq!(check(false, 0.5, 0.5), Check::FalseAlarm); // >= like the label
    }

    #[test]
    fn link_states_match_your_live_graph() {
        let mut app = JamApp {
            seed: 4,
            ..Default::default()
        };
        app.place_from_seed();
        let truth = app.truth();
        let live = live_graph(&app.graph, &app.swarm, &truth, &app.config);
        let states = app.link_states(&truth);
        assert_eq!(states.len(), app.graph.edges.len());
        let live_n = states
            .iter()
            .filter(|s| matches!(s, LinkState::Live(_)))
            .count();
        let one_way: Vec<(u32, u32)> = states
            .iter()
            .filter_map(|s| match *s {
                LinkState::OneWay { from, to } => Some((from as u32, to as u32)),
                _ => None,
            })
            .collect();
        assert_eq!(live_n, live.graph.edges.len());
        // The states only cover geometric links: live.cut_one_way.
        assert_eq!(one_way, live.cut_one_way);
        assert_eq!(app.after_graph(&truth), live.graph);
    }

    #[test]
    fn either_end_rule_is_your_fail_links() {
        let mut app = JamApp {
            seed: 4,
            link_rule: LinkRule::EitherEnd,
            ..Default::default()
        };
        app.place_from_seed();
        let truth = app.truth();
        let expected = fail_links(&app.graph, &jammed_labels(&truth, &app.config));
        assert_eq!(app.after_graph(&truth), expected);
        assert!(
            app.link_states(&truth)
                .iter()
                .all(|s| !matches!(s, LinkState::OneWay { .. }))
        );
    }

    #[test]
    fn with_no_jamming_each_role_is_heard_at_its_range() {
        let cfg = SimConfig::default();
        for role in Role::ALL {
            let r = heard_range(role, 0.0, &cfg).unwrap();
            assert!((r - role.radio_range(&cfg)).abs() < 1e-9, "{role:?}");
        }
        // Jamming so strong that even the loudest drone is drowned next to it.
        assert_eq!(heard_range(Role::Recon, 1e9, &cfg), None);
    }

    #[test]
    fn link_detail_agrees_with_your_live_graph() {
        let mut app = JamApp {
            seed: 4,
            ..Default::default()
        };
        app.place_from_seed();
        let truth = app.truth();
        let live = live_graph(&app.graph, &app.swarm, &truth, &app.config);
        let mut k = 0;
        for e in &app.graph.edges {
            let (i, j) = (e.src as usize, e.dst as usize);
            let ld = link_detail(
                &app.swarm[i],
                &app.swarm[j],
                truth[i],
                truth[j],
                &app.config,
            );
            if ld.margin_ij >= 1.0 && ld.margin_ji >= 1.0 {
                assert_eq!(live.graph.edges[k], *e);
                assert!((live.margins[k] - ld.margin_ij.min(ld.margin_ji)).abs() < 1e-12);
                k += 1;
            }
        }
        assert_eq!(k, live.graph.edges.len());
    }

    #[test]
    fn build_your_own_uses_your_drones() {
        let mut app = JamApp::default();
        app.set_scene(Scene::Hand);
        app.hand_preset_check();
        assert_eq!(app.swarm, app.hand);
        assert_eq!(app.graph, epsilon_graph(&app.hand, &app.config));
        // Recon 14-15 (1 km): linked. Recon 15 - relay 17 (2 km): linked. 14-17: 3 km > 2.5.
        let pairs: Vec<(u32, u32)> = app.graph.edges.iter().map(|e| (e.src, e.dst)).collect();
        assert_eq!(pairs, vec![(0, 1), (1, 2)]);
        // Deleting renumbers the IDs and rebuilds the graph.
        app.remove_drone(0);
        assert_eq!(app.hand.len(), 2);
        assert_eq!(app.hand[0].id, DroneId(0));
        assert_eq!(app.graph.edges.len(), 1);
        // Back to the seed tab: a spawned swarm again.
        app.set_scene(Scene::Seed);
        assert_eq!(app.swarm.len(), 100);
    }

    #[test]
    fn hand_presets_build_valid_graphs() {
        let mut app = JamApp::default();
        app.set_scene(Scene::Hand);
        for preset in [
            JamApp::hand_preset_check,
            JamApp::hand_preset_two_squads,
            JamApp::hand_preset_relay_chain,
        ] {
            preset(&mut app);
            let truth = app.truth();
            assert_eq!(truth.len(), app.hand.len());
            assert_eq!(app.link_states(&truth).len(), app.graph.edges.len());
            assert!(!app.graph.edges.is_empty());
        }
    }

    #[test]
    fn your_one_way_list_matches_brute_force() {
        // live_graph's one_way (geometric pass + larger-range pass) finds exactly the
        // directions a check of every pair finds, on a hand-built and on a seed swarm.
        let mut app = JamApp::default();
        app.set_scene(Scene::Hand);
        app.hand_preset_two_squads();
        for _ in 0..2 {
            let truth = app.truth();
            let key = |a: &OneWayArrow| (a.from, a.to, a.geometric);
            let mut ours: Vec<_> = app.one_way_arrows(&truth).iter().map(key).collect();
            let mut brute: Vec<_> = one_way_all(&app.swarm, &truth, &app.graph, &app.config)
                .iter()
                .map(key)
                .collect();
            ours.sort();
            brute.sort();
            assert_eq!(ours, brute);
            app.set_scene(Scene::Seed);
            app.seed = 4;
            app.place_from_seed();
        }
    }

    #[test]
    fn one_way_arrows_include_live_graph_one_way_and_more() {
        // Every one-way direction of live_graph is a teal (geometric) arrow.
        let mut app = JamApp {
            seed: 4,
            ..Default::default()
        };
        app.place_from_seed();
        let truth = app.truth();
        let live = live_graph(&app.graph, &app.swarm, &truth, &app.config);
        let arrows = one_way_all(&app.swarm, &truth, &app.graph, &app.config);
        let teal: Vec<(u32, u32)> = arrows
            .iter()
            .filter(|a| a.geometric)
            .map(|a| (a.from as u32, a.to as u32))
            .collect();
        let mut expected = live.cut_one_way.clone();
        expected.sort_by_key(|&(a, b)| (a.min(b), a.max(b)));
        assert_eq!(teal, expected);
        assert!(arrows.iter().all(|a| a.margin >= 1.0));
    }

    #[test]
    fn a_relay_is_heard_beyond_the_recon_range() {
        // No jammer. Relay and recon 5 km apart: not a geometric link (2.5 km),
        // but the relay's 16 km range reaches the recon. One violet arrow, relay -> recon.
        let cfg = SimConfig::default();
        let swarm = vec![
            hand_drone(0, 0, Role::Relay, 10.0, 10.0),
            hand_drone(1, 0, Role::Recon, 15.0, 10.0),
        ];
        let graph = epsilon_graph(&swarm, &cfg);
        assert!(graph.edges.is_empty());
        let arrows = one_way_all(&swarm, &[0.0, 0.0], &graph, &cfg);
        assert_eq!(arrows.len(), 1);
        assert_eq!((arrows[0].from, arrows[0].to), (0, 1));
        assert!(!arrows[0].geometric);
    }

    #[test]
    fn transform_round_trips() {
        let t = Transform::fit(
            Rect::from_min_size(pos2(0.0, 0.0), vec2(900.0, 500.0)),
            40.0,
            20.0,
            2.5,
            vec2(30.0, -10.0),
        );
        let p = at(12.3, 7.7);
        let back = t.to_km(t.to_screen(p));
        assert!((back.x - p.x).abs() < 1e-3 && (back.y - p.y).abs() < 1e-3);
    }
}
