//! One dataset snapshot: everything SwarmSim generates for one seed, built in the
//! dataset order with a single RNG (spawn, place_jamming, truth, readings, tasks).
//! Missions read their graphs and labels from here instead of re-running the steps.

use std::ops::Range;

use aerie_graph::EdgeList;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::config::SimConfig;
use crate::drone::Drone;
use crate::graphs::epsilon::epsilon_graph;
use crate::graphs::knn::knn_graph;
use crate::jamming::{Jammer, Ridge, jammed_labels, place_jamming, readings, true_power};
use crate::links::{LiveGraph, live_graph};
use crate::spawn::spawn_swarm;
use crate::tasks::{Task, spawn_tasks};

/// The seed split (plan, "split by scenario, never by node").
pub const TRAIN_SEEDS: Range<u64> = 0..80;
pub const VAL_SEEDS: Range<u64> = 80..90;
pub const TEST_SEEDS: Range<u64> = 90..100;

/// Everything generated for one seed. Drone `i` is index `i` in every field.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub seed: u64,
    pub swarm: Vec<Drone>,
    /// The geometric graph without jamming (`epsilon_graph`, D00).
    pub geometric: EdgeList,
    pub jammer: Jammer,
    pub ridge: Ridge,
    /// True jamming power at each drone.
    pub truth: Vec<f64>,
    /// What each drone's sensor reads.
    pub readings: Vec<f64>,
    /// Zone label: true power at or above the threshold.
    pub jammed: Vec<bool>,
    /// The comm graph under jamming (D03).
    pub live: LiveGraph,
    pub tasks: Vec<Task>,
}

impl Snapshot {
    /// Builds seed `seed`. Same seed and config, same snapshot, bit for bit.
    ///
    /// Panics if `place_jamming` finds no placement (never happens with the default
    /// config on seeds 0 to 99).
    pub fn build(seed: u64, config: &SimConfig) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let swarm = spawn_swarm(config, &mut rng);
        let geometric = epsilon_graph(&swarm, config);
        let (jammer, ridge) = place_jamming(&swarm, config, &mut rng);
        let truth = true_power(&swarm, &jammer, &ridge, config);
        let readings = readings(&truth, config, &mut rng);
        let jammed = jammed_labels(&truth, config);
        let live = live_graph(&geometric, &swarm, &truth, config);
        let tasks = spawn_tasks(&swarm, config, &mut rng);
        Snapshot {
            seed,
            swarm,
            geometric,
            jammer,
            ridge,
            truth,
            readings,
            jammed,
            live,
            tasks,
        }
    }

    /// The kNN graph of this swarm (union rule, D00). Built on demand: no RNG involved.
    pub fn knn(&self, config: &SimConfig) -> EdgeList {
        knn_graph(&self.swarm, config)
    }

    pub fn n(&self) -> usize {
        self.swarm.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_the_dataset_order() {
        let config = SimConfig::default();
        let s = Snapshot::build(3, &config);

        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let swarm = spawn_swarm(&config, &mut rng);
        let (jammer, ridge) = place_jamming(&swarm, &config, &mut rng);
        let truth = true_power(&swarm, &jammer, &ridge, &config);
        let reads = readings(&truth, &config, &mut rng);
        let tasks = spawn_tasks(&swarm, &config, &mut rng);

        assert_eq!(s.jammer, jammer);
        assert_eq!(s.truth, truth);
        assert_eq!(s.readings, reads);
        assert_eq!(s.tasks, tasks);
        assert_eq!(s.jammed, jammed_labels(&truth, &config));
        assert_eq!(s.live, live_graph(&s.geometric, &swarm, &truth, &config));
    }

    #[test]
    fn same_seed_same_snapshot() {
        let config = SimConfig::default();
        let (a, b) = (Snapshot::build(7, &config), Snapshot::build(7, &config));
        assert_eq!(a.readings, b.readings);
        assert_eq!(a.live, b.live);
        assert_ne!(a.readings, Snapshot::build(8, &config).readings);
    }

    #[test]
    fn the_seed_split_covers_0_to_99_once() {
        assert_eq!(TRAIN_SEEDS.end, VAL_SEEDS.start);
        assert_eq!(VAL_SEEDS.end, TEST_SEEDS.start);
        assert_eq!((TRAIN_SEEDS.start, TEST_SEEDS.end), (0, 100));
    }
}
