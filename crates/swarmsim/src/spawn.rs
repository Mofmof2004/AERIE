//! Creating the swarm: each squad in its own random cell, drones scattered around it.

use crate::config::SimConfig;
use crate::drone::{Drone, DroneId, Role, SquadId, Vec2};
use rand::RngExt;
use rand_chacha::ChaCha8Rng;

/// Creates `n_squads * squad_size` drones, on station (zero velocity).
pub fn spawn_swarm(config: &SimConfig, rng: &mut ChaCha8Rng) -> Vec<Drone> {
    let mut swarm: Vec<Drone> = Vec::with_capacity(config.n_squads * config.squad_size);
    let mut droneid_count: u32 = 0;

    let n_cols = (config.arena_w / config.cell_width) as usize;
    let n_rows = (config.arena_h / config.cell_height) as usize;

    // Roles by position in the squad. The last few are split between
    // RF sensing and compute, alternating between even and odd squads.
    let n_recon = config.squad_size / 2;
    let n_relay = config.squad_size / 5;
    let n_rest = config.squad_size - n_recon - n_relay;

    for squad in 0..config.n_squads {
        // Draw random cells until one is free. Member 0 of every earlier
        // squad sits exactly on its squad centre, so compare against those.
        let center_position = loop {
            let col = rng.random_range(0..n_cols);
            let row = rng.random_range(0..n_rows);
            let candidate = Vec2 {
                x: (col as f64 + 0.5) * config.cell_width,
                y: (row as f64 + 0.5) * config.cell_height,
            };

            let mut taken = false;
            for index in 0..squad {
                let leader = &swarm[index * config.squad_size];
                if leader.position == candidate {
                    taken = true;
                }
            }
            if !taken {
                break candidate;
            }
        };

        let n_rf_sensing = if squad % 2 == 0 {
            n_rest.div_ceil(2)
        } else {
            n_rest / 2
        };

        for member in 0..config.squad_size {
            let role = if member < n_recon {
                Role::Recon
            } else if member < n_recon + n_rf_sensing {
                Role::RfSensing
            } else if member < n_recon + n_rf_sensing + n_relay {
                Role::Relay
            } else {
                Role::Compute
            };

            let position = if member == 0 {
                center_position
            } else {
                let s = config.squad_spread;
                let x = rng.random_range(center_position.x - s..center_position.x + s);
                let y = rng.random_range(center_position.y - s..center_position.y + s);
                Vec2 {
                    x: x.clamp(0.0, config.arena_w),
                    y: y.clamp(0.0, config.arena_h),
                }
            };

            let mut drone = Drone::new(
                DroneId(droneid_count),
                SquadId(squad as u16),
                role,
                position,
            );
            drone.battery = rng.random_range(0.5..1.0);
            drone.sensor_quality = rng.random_range(0.5..1.0);
            drone.compute_capacity = match role {
                Role::Compute => rng.random_range(0.7..1.0),
                Role::Recon | Role::RfSensing | Role::Relay => rng.random_range(0.0..0.3),
            };

            swarm.push(drone);
            droneid_count += 1;
        }
    }

    swarm
}

// Unit tests: compiled only by `cargo test`.
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn swarm(seed: u64) -> Vec<Drone> {
        spawn_swarm(&SimConfig::default(), &mut ChaCha8Rng::seed_from_u64(seed))
    }

    #[test]
    fn hundred_drones_with_plan_role_mix() {
        let s = swarm(42);
        assert_eq!(s.len(), 100);
        let count = |r: Role| s.iter().filter(|d| d.role == r).count();
        assert_eq!(count(Role::Recon), 50);
        assert_eq!(count(Role::RfSensing), 15);
        assert_eq!(count(Role::Relay), 20);
        assert_eq!(count(Role::Compute), 15);
    }

    #[test]
    fn ids_are_0_to_99_in_order() {
        for (i, d) in swarm(42).iter().enumerate() {
            assert_eq!(d.id, DroneId(i as u32));
        }
    }

    #[test]
    fn drone_17_is_a_relay_in_squad_1() {
        // Squad 1 is odd: members 0-4 recon, 5 RF sensing, 6-7 relays, 8-9 compute.
        let d = &swarm(42)[17];
        assert_eq!(d.squad, SquadId(1));
        assert_eq!(d.role, Role::Relay);
    }

    #[test]
    fn same_seed_same_swarm_other_seed_other_swarm() {
        assert_eq!(swarm(42), swarm(42));
        assert_ne!(swarm(42), swarm(43));
    }

    #[test]
    fn every_drone_inside_the_arena() {
        let cfg = SimConfig::default();
        for d in swarm(42) {
            assert!((0.0..=cfg.arena_w).contains(&d.position.x), "{:?}", d.id);
            assert!((0.0..=cfg.arena_h).contains(&d.position.y), "{:?}", d.id);
        }
    }

    #[test]
    fn every_squad_has_its_own_cell() {
        let cfg = SimConfig::default();
        let s = swarm(42);
        let centres: Vec<Vec2> = (0..cfg.n_squads)
            .map(|k| s[k * cfg.squad_size].position)
            .collect();
        for i in 0..centres.len() {
            for j in i + 1..centres.len() {
                assert_ne!(centres[i], centres[j], "squads {i} and {j} share a cell");
            }
        }
    }

    #[test]
    fn members_stay_within_spread_of_their_centre() {
        let cfg = SimConfig::default();
        let s = swarm(42);
        for d in &s {
            let leader = &s[d.squad.0 as usize * cfg.squad_size];
            assert!((d.position.x - leader.position.x).abs() <= cfg.squad_spread);
            assert!((d.position.y - leader.position.y).abs() <= cfg.squad_spread);
        }
    }

    #[test]
    fn features_in_their_ranges() {
        for d in swarm(42) {
            assert!((0.5..1.0).contains(&d.battery));
            assert!((0.5..1.0).contains(&d.sensor_quality));
            let (lo, hi) = if d.role == Role::Compute {
                (0.7, 1.0)
            } else {
                (0.0, 0.3)
            };
            assert!((lo..hi).contains(&d.compute_capacity), "{:?}", d.role);
            assert_eq!(d.velocity, Vec2 { x: 0.0, y: 0.0 });
        }
    }
}
