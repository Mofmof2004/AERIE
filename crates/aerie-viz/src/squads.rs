//! Per-squad summaries and grid-cell helpers, shared by every GUI.

use std::collections::BTreeMap;

use swarmsim::config::SimConfig;
use swarmsim::drone::{Drone, Role};

/// Summary of one squad, recomputed whenever the swarm changes.
pub struct SquadInfo {
    pub squad: usize,
    /// Indices into the swarm vector.
    pub members: Vec<usize>,
    /// Mean position of the members, km.
    pub centroid: (f64, f64),
    /// Distance from the centroid to the furthest member, km.
    pub footprint: f64,
    /// Distance to the closest other squad centroid, km.
    pub nearest_other: Option<f64>,
    /// Members per role, in the order of `Role::ALL`.
    pub role_counts: [usize; 4],
}

/// Number of cells in the grid for a cell size.
pub fn cell_count(cfg: &SimConfig, w: f64, h: f64) -> usize {
    ((cfg.arena_w / w).round() as usize) * ((cfg.arena_h / h).round() as usize)
}

/// Cell sizes that divide the arena exactly and leave room for one squad each.
pub fn cell_options(cfg: &SimConfig) -> Vec<(f64, f64)> {
    let divides = |part: f64, whole: f64| {
        let r = whole / part;
        (r - r.round()).abs() < 1e-9
    };
    let mut options = Vec::new();
    for w in [2.0, 4.0, 5.0, 8.0, 10.0, 20.0, 40.0] {
        for h in [2.0, 4.0, 5.0, 10.0, 20.0] {
            if divides(w, cfg.arena_w) && divides(h, cfg.arena_h) {
                options.push((w, h));
            }
        }
    }
    options
}

/// Groups the swarm by squad and computes each squad's summary.
pub fn squad_infos(swarm: &[Drone]) -> Vec<SquadInfo> {
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, d) in swarm.iter().enumerate() {
        groups.entry(d.squad.0 as usize).or_default().push(i);
    }

    let mut infos: Vec<SquadInfo> = groups
        .into_iter()
        .map(|(squad, members)| {
            let n = members.len() as f64;
            let cx = members.iter().map(|&i| swarm[i].position.x).sum::<f64>() / n;
            let cy = members.iter().map(|&i| swarm[i].position.y).sum::<f64>() / n;
            let footprint = members
                .iter()
                .map(|&i| (swarm[i].position.x - cx).hypot(swarm[i].position.y - cy))
                .fold(0.0, f64::max);
            let mut role_counts = [0; 4];
            for &i in &members {
                let k = Role::ALL
                    .iter()
                    .position(|&r| r == swarm[i].role)
                    .unwrap_or(0);
                role_counts[k] += 1;
            }
            SquadInfo {
                squad,
                members,
                centroid: (cx, cy),
                footprint,
                nearest_other: None,
                role_counts,
            }
        })
        .collect();

    let centroids: Vec<(f64, f64)> = infos.iter().map(|s| s.centroid).collect();
    for (k, s) in infos.iter_mut().enumerate() {
        s.nearest_other = centroids
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != k)
            .map(|(_, c)| (c.0 - s.centroid.0).hypot(c.1 - s.centroid.1))
            .min_by(f64::total_cmp);
    }
    infos
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use swarmsim::spawn::spawn_swarm;

    #[test]
    fn cell_options_all_divide_the_arena() {
        let cfg = SimConfig::default();
        let options = cell_options(&cfg);
        assert!(options.contains(&(cfg.cell_width, cfg.cell_height)));
        for (w, h) in options {
            assert_eq!((cfg.arena_w / w).fract(), 0.0);
            assert_eq!((cfg.arena_h / h).fract(), 0.0);
        }
    }

    #[test]
    fn squad_infos_cover_every_drone_once() {
        let cfg = SimConfig::default();
        let swarm = spawn_swarm(&cfg, &mut ChaCha8Rng::seed_from_u64(0));
        let infos = squad_infos(&swarm);
        assert_eq!(infos.len(), cfg.n_squads);
        let total: usize = infos.iter().map(|s| s.members.len()).sum();
        assert_eq!(total, swarm.len());
        for s in &infos {
            assert_eq!(s.role_counts.iter().sum::<usize>(), s.members.len());
            assert!(s.nearest_other.is_some());
        }
    }
}
