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
    /// k for the kNN graph. With the union rule every drone gets at least k links and
    /// the mean degree is between k and 2k; a hub can get more.
    pub knn_k: usize,
    /// Jamming falloff distance, km: power = P0 / (1 + (d / d0)^2), so half power at d0.
    pub jam_d0: f64,
    /// P0: power of a new jammer at its own position. Thresholds and noise are fractions of it.
    pub jam_p0: f64,
    /// Multiplier on jamming power behind the ridge (the ridge shadow).
    pub ridge_factor: f64,
    /// Jamming power (as a fraction of P0) at or above which a drone is in the jammed
    /// zone (the zone label, and the placement bounds). Links no longer fail on this
    /// threshold: the signal-against-jamming rule decides (D03).
    pub jam_threshold: f64,
    /// Standard deviation of the Gaussian sensor noise on jam readings, as a fraction of P0.
    pub noise_sd: f64,
    /// A jamming placement is kept only if at least this fraction of drones is jammed.
    pub jam_frac_min: f64,
    /// ...and at most this fraction (decision D02).
    pub jam_frac_max: f64,
    /// N0: background radio noise at every receiver, in units of P0 (decision D03).
    /// A link direction works when signal ≥ noise_floor + jamming.
    ///
    /// Counter-intuitive: a **lower** N0 gives **fewer** links under jamming. Transmit
    /// power is tied to it (`tx = N0 · (1 + (range / scale)²)`, so each drone still
    /// reaches its range), and the margin splits into
    /// `geometry × N0 / (N0 + jamming)`. With no jammer N0 cancels out and the graph is
    /// the same for any N0; with a jammer, a small N0 means quiet drones that the jammer
    /// drowns easily. N0 really sets how loud the drones are against the jammer.
    pub noise_floor: f64,
    /// Signal fade scale of a recon drone, km: signal = P / (1 + (d / scale)²).
    /// Smaller = holds up better under jamming. Does not change the no-jammer range (D03).
    pub signal_scale_recon: f64,
    /// Same for RF-sensing drones, km.
    pub signal_scale_rf: f64,
    /// Same for relays, km.
    pub signal_scale_relay: f64,
    /// Same for compute drones, km.
    pub signal_scale_compute: f64,
    /// Tasks per snapshot (step 3).
    pub n_tasks: usize,
    /// Share of tasks placed near a drone; the rest are uniform over the arena.
    pub task_near_frac: f64,
    /// A near task lies within this many km of a random drone, in x and in y.
    pub task_near_radius: f64,
    /// Cost of an empty battery, in km of travel (step 3c): the cost adds
    /// `cost_beta * (1 - battery)`. Batteries spawn in 0.5..1, so it adds 0 to 1.5 km.
    pub cost_beta: f64,
    /// Cost added when a drone cannot take a task (role does not fit, or no relay within
    /// 2 hops for Observe and Track). Larger than any real cost, but finite so the
    /// Hungarian solver (step 3d) still has a number in every cell.
    pub cost_penalty: f64,
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
            range_short: 2.5,
            range_long: 16.0,
            squad_spread: 2.0,
            knn_k: 5,
            jam_d0: 4.0,
            jam_p0: 1.0,
            ridge_factor: 0.1,
            jam_threshold: 0.5,
            jam_frac_min: 0.05,
            jam_frac_max: 0.30,
            noise_sd: 0.1,
            noise_floor: 0.5,
            signal_scale_recon: 0.5,
            signal_scale_rf: 0.7,
            signal_scale_relay: 0.3,
            signal_scale_compute: 1.0,
            n_tasks: 50,
            task_near_frac: 0.5,
            task_near_radius: 3.0,
            cost_beta: 3.0,
            cost_penalty: 1000.0,
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
