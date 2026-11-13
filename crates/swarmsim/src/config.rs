//! Simulation parameters. Every distance is in km.

use serde::{Deserialize, Serialize};

/// All parameters of one SwarmSim run.
///
/// `Default` holds the starting values from the plan ("Starting values" table).
/// An experiment changes only what it needs:
/// `SimConfig { squad_spread: 2.0, ..Default::default() }`.
/// Saved as JSON next to every dataset, so each run records its exact setup.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct SimConfig {
    /// Number of squads. Plan: 10.
    pub n_squads: usize,
    /// Drones per squad. Plan: 10, so N = 100 drones by default.
    pub squad_size: usize,
    /// Arena width (x), km. Plan: 40.
    pub arena_w: f64,
    /// Arena height (y), km. Plan: 20.
    pub arena_h: f64,
    /// Radio range of recon, RF sensing and compute drones, km. Plan: 5.
    pub range_short: f64,
    /// Radio range of relay drones, km. Plan: 9.
    pub range_long: f64,
    /// Standard deviation of a drone's offset from its squad centre, per axis, km.
    /// Main knob for mean degree and cross-squad links. Tuned in step 1.
    pub squad_spread: f64,
    /// k for the kNN graph. With the union rule each drone gets between k and 2k links.
    pub knn_k: usize,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            n_squads: 10,
            squad_size: 10,
            arena_w: 40.0,
            arena_h: 20.0,
            range_short: 5.0,
            range_long: 9.0,
            squad_spread: 1.0,
            knn_k: 5,
        }
    }
}

// Unit tests: compiled only by `cargo test`.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_plan() {
        let cfg = SimConfig::default();
        assert_eq!(cfg.n_squads * cfg.squad_size, 100);
        assert!(
            cfg.range_long > cfg.range_short,
            "relays must reach further"
        );
    }

    #[test]
    fn struct_update_changes_one_field() {
        let cfg = SimConfig {
            squad_spread: 2.0,
            ..Default::default()
        };
        assert_eq!(cfg.squad_spread, 2.0);
        assert_eq!(cfg.n_squads, SimConfig::default().n_squads);
    }

    #[test]
    fn json_round_trip() {
        // A config written to disk reads back identical.
        let cfg = SimConfig::default();
        let text = serde_json::to_string_pretty(&cfg).unwrap();
        let back: SimConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(cfg, back);
    }
}
