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
    /// Width of one squad cell, km. The arena is cut into a grid of cells and each
    /// squad is centred in one cell. Must divide `arena_w` exactly.
    pub cell_width: f64,
    /// Height of one squad cell, km. Must divide `arena_h` exactly.
    /// The grid needs at least `n_squads` cells: each squad gets a different random cell.
    pub cell_height: f64,
    /// Radio range of recon, RF sensing and compute drones, km. Plan: 5.
    pub range_short: f64,
    /// Radio range of relay drones, km. Plan: 9.
    pub range_long: f64,
    /// Half-width of the square around the squad centre where members are placed,
    /// uniformly, km. A member is at most `squad_spread` from the centre on each axis.
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
            cell_width: 5.0,
            cell_height: 5.0,
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

    /// True when `whole / part` is a whole number (within float tolerance).
    fn divides(part: f64, whole: f64) -> bool {
        let ratio = whole / part;
        (ratio - ratio.round()).abs() < 1e-9
    }

    #[test]
    fn cells_divide_the_arena() {
        let cfg = SimConfig::default();
        assert!(
            divides(cfg.cell_width, cfg.arena_w),
            "cell_width {} does not divide arena_w {}",
            cfg.cell_width,
            cfg.arena_w
        );
        assert!(
            divides(cfg.cell_height, cfg.arena_h),
            "cell_height {} does not divide arena_h {}",
            cfg.cell_height,
            cfg.arena_h
        );
    }

    #[test]
    fn enough_cells_for_every_squad() {
        // spawn_swarm gives each squad a different random cell. With fewer cells
        // than squads its search for a free cell would never end.
        let cfg = SimConfig::default();
        let cols = (cfg.arena_w / cfg.cell_width).round() as usize;
        let rows = (cfg.arena_h / cfg.cell_height).round() as usize;
        assert!(
            cols * rows >= cfg.n_squads,
            "grid is {cols} x {rows} = {} cells for {} squads",
            cols * rows,
            cfg.n_squads
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
