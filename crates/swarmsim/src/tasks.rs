//! Tasks the swarm votes on (step 3): observe a sector, hold a track on a contact,
//! relay for a cut-off squad, fuse imagery. Each task has a kind and a position; the
//! kind decides which roles can take it (the strict mapping chosen in step 3a).

use crate::config::SimConfig;
use crate::drone::{Drone, Role, Vec2};
use aerie_graph::EdgeList;
use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Stable task identity. Its own type, like `DroneId`, so it cannot be mixed up with a
/// drone ID or used as an index by accident (D00).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TaskId(pub u32);

/// What a task asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TaskKind {
    /// Watch a sector with a camera or an RF sensor.
    Observe,
    /// Keep a camera on a moving contact.
    Track,
    /// Stand between a cut-off squad and the rest of the swarm, passing its traffic on.
    Relay,
    /// Combine imagery from recon drones into one picture.
    Fuse,
}

/// One task of a snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    /// Stable identity (the task's index when it is spawned).
    pub id: TaskId,
    /// What the task asks for, which decides the roles that can take it.
    pub kind: TaskKind,
    /// Where the task is, km.
    pub position: Vec2,
}

impl TaskKind {
    /// Every kind, in a fixed order (used to draw a random kind).
    pub const ALL: [TaskKind; TaskKind::NUM_KINDS] = [
        TaskKind::Observe,
        TaskKind::Track,
        TaskKind::Relay,
        TaskKind::Fuse,
    ];

    /// Number of task kinds.
    pub const NUM_KINDS: usize = 4;

    /// Whether a drone with this role can take the task (strict mapping, step 3a):
    /// observe = recon or RF sensing, track = recon, relay = relay, fuse = compute.
    /// Roles only: the relay rule (step 3b) and the cost (3c) come on top.
    pub fn can_do(self, role: Role) -> bool {
        if self == TaskKind::Observe {
            match role {
                Role::Recon => true,
                Role::RfSensing => true,
                Role::Relay => false,
                Role::Compute => false,
            }
        } else if self == TaskKind::Track {
            match role {
                Role::Recon => true,
                Role::RfSensing => false,
                Role::Relay => false,
                Role::Compute => false,
            }
        } else if self == TaskKind::Relay {
            match role {
                Role::Recon => false,
                Role::RfSensing => false,
                Role::Relay => true,
                Role::Compute => false,
            }
        } else if self == TaskKind::Fuse {
            match role {
                Role::Recon => false,
                Role::RfSensing => false,
                Role::Relay => false,
                Role::Compute => true,
            }
        } else {
            panic!("..");
        }
    }

    /// Whether the relay rule (step 3b) applies: observe and track data has to get out
    /// of the area, so the drone needs a relay within two hops on the live graph.
    pub fn needs_relay(self) -> bool {
        matches!(self, TaskKind::Observe | TaskKind::Track)
    }
}

/// The tasks of one snapshot: `n_tasks` tasks with a random kind. The first
/// `task_near_frac` share sit near a random drone (within `task_near_radius` km in x
/// and y, redrawn until inside the arena); the rest are uniform over the arena.
/// Call after spawn, jamming and readings, with the same RNG (dataset order).
pub fn spawn_tasks(swarm: &[Drone], config: &SimConfig, rng: &mut ChaCha8Rng) -> Vec<Task> {
    let n_near = (config.n_tasks as f64 * config.task_near_frac).round() as usize;
    let mut tasks: Vec<Task> = Vec::new();
    for id in 0..n_near {
        let task_kind = TaskKind::ALL[rng.random_range(0..TaskKind::NUM_KINDS)];
        let base_drone_position = swarm[rng.random_range(0..swarm.len())].position;

        let mut pos_set = false;
        let mut x: f64 = 0.0;
        let mut y: f64 = 0.0;

        while !pos_set {
            let offset_x = rng.random_range(-config.task_near_radius..config.task_near_radius);
            let offset_y = rng.random_range(-config.task_near_radius..config.task_near_radius);
            x = base_drone_position.x + offset_x;
            y = base_drone_position.y + offset_y;

            if x >= 0.0 && x <= config.arena_w && y >= 0.0 && y <= config.arena_h {
                pos_set = true;
            }
        }

        let position: Vec2 = Vec2 { x, y };
        let taskid: TaskId = TaskId(id as u32);
        let task: Task = Task {
            id: taskid,
            kind: task_kind,
            position,
        };
        tasks.push(task);
    }

    for id in n_near..(config.n_tasks) {
        let task_kind = TaskKind::ALL[rng.random_range(0..TaskKind::NUM_KINDS)];
        let x = rng.random_range(0.0..config.arena_w);
        let y = rng.random_range(0.0..config.arena_h);
        let position: Vec2 = Vec2 { x, y };
        let taskid: TaskId = TaskId(id as u32);
        let task: Task = Task {
            id: taskid,
            kind: task_kind,
            position,
        };
        tasks.push(task);
    }
    tasks
}
/// The relay rule (step 3b): for every drone, whether a relay is at most 2 hops away on the
/// live graph. A relay counts at 0 hops. Observe and Track need it (`needs_relay`): their
/// data must reach a relay to leave the area.
///
/// One small breadth-first search per drone, stopped at depth 2. The queue holds
/// (drone, hops from the start), so each entry knows its own depth. Two hops on purpose:
/// a GNN with K layers sees exactly K hops, so a 2-layer GNN can learn this rule.
pub fn relay_within_two_hops(live: &EdgeList, swarm: &[Drone]) -> Vec<bool> {
    let neighbours = live.neighbours();
    let mut result: Vec<bool> = Vec::new();

    for i in 0..swarm.len() {
        // A relay is 0 hops from itself.
        let mut found = swarm[i].role == Role::Relay;
        // Each entry is (drone, its distance from i).
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
        queue.push_back((i, 0));

        'search: while let Some((drone, hop)) = queue.pop_front() {
            if found {
                break;
            }
            if hop == 2 {
                continue; // don't look further than 2 hops
            }
            for &j in &neighbours[drone] {
                if swarm[j].role == Role::Relay {
                    found = true;
                    break 'search;
                }
                queue.push_back((j, hop + 1));
            }
        }
        result.push(found);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jamming::{place_jamming, readings, true_power};
    use crate::spawn::spawn_swarm;
    use rand::SeedableRng;

    /// The dataset order: spawn, jamming, readings, then tasks, all with one RNG.
    fn snapshot(seed: u64, cfg: &SimConfig) -> (Vec<Drone>, Vec<Task>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let swarm = spawn_swarm(cfg, &mut rng);
        let (jammer, ridge) = place_jamming(&swarm, cfg, &mut rng);
        let truth = true_power(&swarm, &jammer, &ridge, cfg);
        let _readings = readings(&truth, cfg, &mut rng);
        let tasks = spawn_tasks(&swarm, cfg, &mut rng);
        (swarm, tasks)
    }

    /// Within `r` km of `drone` in x and in y (the square the near tasks are drawn in).
    fn in_square(task: &Task, drone: &Drone, r: f64) -> bool {
        (task.position.x - drone.position.x).abs() <= r
            && (task.position.y - drone.position.y).abs() <= r
    }

    #[test]
    fn can_do_follows_the_strict_mapping() {
        use Role::*;
        let table = [
            (TaskKind::Observe, [true, true, false, false]),
            (TaskKind::Track, [true, false, false, false]),
            (TaskKind::Relay, [false, false, true, false]),
            (TaskKind::Fuse, [false, false, false, true]),
        ];
        for (kind, expected) in table {
            for (role, ok) in [Recon, RfSensing, Relay, Compute].into_iter().zip(expected) {
                assert_eq!(kind.can_do(role), ok, "{kind:?} by {role:?}");
            }
        }
    }

    #[test]
    fn relay_rule_applies_to_observe_and_track_only() {
        assert!(TaskKind::Observe.needs_relay());
        assert!(TaskKind::Track.needs_relay());
        assert!(!TaskKind::Relay.needs_relay());
        assert!(!TaskKind::Fuse.needs_relay());
    }

    #[test]
    fn prediction_2_qualified_drones_by_role() {
        // 50 recon, 15 RF, 20 relays, 15 compute: observe 65, track 50, relay 20, fuse 15.
        let cfg = SimConfig::default();
        let (swarm, _) = snapshot(0, &cfg);
        let qualified = |kind: TaskKind| swarm.iter().filter(|d| kind.can_do(d.role)).count();
        assert_eq!(qualified(TaskKind::Observe), 65);
        assert_eq!(qualified(TaskKind::Track), 50);
        assert_eq!(qualified(TaskKind::Relay), 20);
        assert_eq!(qualified(TaskKind::Fuse), 15);
    }

    #[test]
    fn fifty_tasks_with_ids_in_order() {
        let cfg = SimConfig::default();
        let (_, tasks) = snapshot(1, &cfg);
        assert_eq!(tasks.len(), 50);
        for (k, t) in tasks.iter().enumerate() {
            assert_eq!(t.id, TaskId(k as u32));
        }
    }

    #[test]
    fn prediction_1_first_25_are_near_a_drone() {
        // n_near = round(50 * 0.5) = 25, and each one lies in the 3 km square of a drone.
        let cfg = SimConfig::default();
        for seed in 0..20 {
            let (swarm, tasks) = snapshot(seed, &cfg);
            for t in &tasks[..25] {
                assert!(
                    swarm.iter().any(|d| in_square(t, d, cfg.task_near_radius)),
                    "seed {seed}, task {:?}",
                    t.id
                );
            }
        }
    }

    #[test]
    fn every_task_is_inside_the_arena() {
        let cfg = SimConfig::default();
        for seed in 0..20 {
            let (_, tasks) = snapshot(seed, &cfg);
            for t in &tasks {
                assert!((0.0..=cfg.arena_w).contains(&t.position.x), "{t:?}");
                assert!((0.0..=cfg.arena_h).contains(&t.position.y), "{t:?}");
            }
        }
    }

    #[test]
    fn positions_are_real_numbers_not_a_corner_or_a_grid() {
        // Guards the two bugs met while writing spawn_tasks: shadowed x / y left every near
        // task at (0, 0), and an integer range put uniform tasks on whole kilometres.
        let cfg = SimConfig::default();
        let (_, tasks) = snapshot(2, &cfg);
        assert!(tasks.iter().all(|t| t.position != Vec2 { x: 0.0, y: 0.0 }));
        let whole = tasks
            .iter()
            .filter(|t| t.position.x.fract() == 0.0 && t.position.y.fract() == 0.0)
            .count();
        assert_eq!(whole, 0);
    }

    #[test]
    fn prediction_1_kinds_are_drawn_evenly() {
        // 4 kinds, uniform: about 12.5 Track tasks out of 50, a quarter of 5,000 overall.
        let cfg = SimConfig::default();
        let mut counts = [0usize; TaskKind::NUM_KINDS];
        for seed in 0..100 {
            for t in snapshot(seed, &cfg).1 {
                counts[TaskKind::ALL.iter().position(|&k| k == t.kind).unwrap()] += 1;
            }
        }
        for (kind, &c) in TaskKind::ALL.iter().zip(&counts) {
            let share = c as f64 / 5_000.0;
            assert!((share - 0.25).abs() < 0.02, "{kind:?}: {share}");
        }
    }

    #[test]
    fn all_near_or_all_uniform() {
        // task_near_frac 1: every task near a drone. 0: none is forced near.
        let cfg = SimConfig {
            task_near_frac: 1.0,
            ..SimConfig::default()
        };
        let (swarm, tasks) = snapshot(3, &cfg);
        assert!(
            tasks
                .iter()
                .all(|t| swarm.iter().any(|d| in_square(t, d, cfg.task_near_radius)))
        );
        let cfg = SimConfig {
            task_near_frac: 0.0,
            ..SimConfig::default()
        };
        assert_eq!(snapshot(3, &cfg).1.len(), 50);
    }

    /// Drones in a row on the x axis, 1 km apart; `relays` lists the relay indices.
    fn row(n: u32, relays: &[u32]) -> Vec<Drone> {
        (0..n)
            .map(|i| {
                let role = if relays.contains(&i) {
                    Role::Relay
                } else {
                    Role::Recon
                };
                Drone::new(
                    crate::drone::DroneId(i),
                    crate::drone::SquadId(0),
                    role,
                    Vec2 {
                        x: i as f64,
                        y: 0.0,
                    },
                )
            })
            .collect()
    }

    /// A graph from (src, dst) pairs, weight 1.
    fn graph(n: u32, pairs: &[(u32, u32)]) -> EdgeList {
        let edges = pairs
            .iter()
            .map(|&(src, dst)| aerie_graph::Edge {
                src,
                dst,
                weight: 1.0,
            })
            .collect();
        EdgeList { n, edges }
    }

    /// The path 0 - 1 - 2 - 3 - 4.
    fn path5() -> EdgeList {
        graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4)])
    }

    /// Another way to get the same answer, to check against: one BFS started from all
    /// relays at once, giving every drone its distance to the nearest relay.
    fn hops_from_relays(live: &EdgeList, swarm: &[Drone]) -> Vec<Option<usize>> {
        let neighbours = live.neighbours();
        let mut hops = vec![None; swarm.len()];
        let mut queue = VecDeque::new();
        for (i, d) in swarm.iter().enumerate() {
            if d.role == Role::Relay {
                hops[i] = Some(0);
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            let h = hops[i].unwrap();
            for &j in &neighbours[i] {
                if hops[j].is_none() {
                    hops[j] = Some(h + 1);
                    queue.push_back(j);
                }
            }
        }
        hops
    }

    #[test]
    fn prediction_1_path_with_a_relay_at_one_end() {
        let ok = relay_within_two_hops(&path5(), &row(5, &[0]));
        assert_eq!(ok, [true, true, true, false, false]);
    }

    #[test]
    fn prediction_2_relays_at_both_ends_cover_the_path() {
        let ok = relay_within_two_hops(&path5(), &row(5, &[0, 4]));
        assert_eq!(ok, [true; 5]);
    }

    #[test]
    fn prediction_3_a_group_without_a_relay_is_cut_off() {
        // 0 - 1 with a relay at 0, and 2 - 3 with none: hops never cross groups.
        let ok = relay_within_two_hops(&graph(4, &[(0, 1), (2, 3)]), &row(4, &[0]));
        assert_eq!(ok, [true, true, false, false]);
    }

    #[test]
    fn a_relay_counts_at_zero_hops_even_alone() {
        // No links at all: only the relay itself passes.
        let ok = relay_within_two_hops(&graph(3, &[]), &row(3, &[1]));
        assert_eq!(ok, [false, true, false]);
        // No relay anywhere: nobody passes.
        assert_eq!(relay_within_two_hops(&path5(), &row(5, &[])), [false; 5]);
    }

    #[test]
    fn links_count_in_both_directions() {
        // Each link is stored once with src < dst. The relay is at the dst end here,
        // so a search that only follows src -> dst would miss it.
        let ok = relay_within_two_hops(&graph(3, &[(0, 1), (1, 2)]), &row(3, &[2]));
        assert_eq!(ok, [true, true, true]);
    }

    #[test]
    fn a_cycle_does_not_loop_forever() {
        // Triangle 0-1-2 plus a tail 2-3-4-5; relay at 5. Drone 3 is 2 hops, 2 is 3.
        let g = graph(6, &[(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5)]);
        let ok = relay_within_two_hops(&g, &row(6, &[5]));
        assert_eq!(ok, [false, false, false, true, true, true]);
    }

    #[test]
    fn matches_a_bfs_from_all_relays_on_real_seeds() {
        // Searching out from each drone or out from all relays must agree.
        let cfg = SimConfig::default();
        for seed in 0..20 {
            let (swarm, live) = live_snapshot(seed, &cfg);
            let ok = relay_within_two_hops(&live, &swarm);
            let hops = hops_from_relays(&live, &swarm);
            for i in 0..swarm.len() {
                assert_eq!(
                    ok[i],
                    hops[i].is_some_and(|h| h <= 2),
                    "seed {seed}, drone {i}"
                );
            }
        }
    }

    /// Swarm and live graph of a seed, in the dataset order.
    fn live_snapshot(seed: u64, cfg: &SimConfig) -> (Vec<Drone>, EdgeList) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let swarm = spawn_swarm(cfg, &mut rng);
        let (jammer, ridge) = place_jamming(&swarm, cfg, &mut rng);
        let truth = true_power(&swarm, &jammer, &ridge, cfg);
        let geometric = crate::graphs::epsilon::epsilon_graph(&swarm, cfg);
        let live = crate::links::live_graph(&geometric, &swarm, &truth, cfg).graph;
        (swarm, live)
    }

    #[test]
    #[ignore]
    fn print_relay_share() {
        let cfg = SimConfig::default();
        let mut shares = Vec::new();
        let mut hop_counts = [0usize; 4]; // 0, 1, 2, more or never
        for seed in 0..100 {
            let (swarm, live) = live_snapshot(seed, &cfg);
            let ok = relay_within_two_hops(&live, &swarm);
            shares.push(ok.iter().filter(|&&b| b).count() as f64 / swarm.len() as f64);
            for h in hops_from_relays(&live, &swarm) {
                hop_counts[h.unwrap_or(3).min(3)] += 1;
            }
        }
        shares.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = shares.iter().sum::<f64>() / shares.len() as f64;
        println!(
            "mean {mean:.3} min {:.3} median {:.3} max {:.3}",
            shares[0], shares[50], shares[99]
        );
        println!("hops 0/1/2/more-or-never: {hop_counts:?}");
    }

    #[test]
    fn deterministic() {
        let cfg = SimConfig::default();
        assert_eq!(snapshot(7, &cfg).1, snapshot(7, &cfg).1);
        assert_ne!(snapshot(7, &cfg).1, snapshot(8, &cfg).1);
    }
}
