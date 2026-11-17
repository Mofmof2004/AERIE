use crate::config::SimConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DroneId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SquadId(pub u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Role {
    Recon,
    RfSensing,
    Relay,
    Compute,
}

impl Role {
    /// Every role once. Update when adding a role.
    pub const ALL: [Role; 4] = [Role::Recon, Role::RfSensing, Role::Relay, Role::Compute];

    /// Role as a one-hot vector, in the order of `Role::ALL`.
    pub fn one_hot(self) -> [f64; 4] {
        match self {
            Role::Recon => [1.0, 0.0, 0.0, 0.0],
            Role::RfSensing => [0.0, 1.0, 0.0, 0.0],
            Role::Relay => [0.0, 0.0, 1.0, 0.0],
            Role::Compute => [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Number of roles (4).
    pub fn num_roles() -> usize {
        Role::ALL.len()
    }

    /// How far this drone's radio reaches, in km. Relays get the long range,
    /// every other role the short one. Read from the config, never stored on the drone.
    pub fn radio_range(self, config: &SimConfig) -> f64 {
        match self {
            Role::Recon | Role::RfSensing | Role::Compute => config.range_short,
            Role::Relay => config.range_long,
        }
    }

    /// How fast this role's signal fades with distance, km (decision D03).
    /// Smaller = louder transmitter = holds up better under jamming.
    pub fn signal_scale(self, config: &SimConfig) -> f64 {
        match self {
            Role::Recon => config.signal_scale_recon,
            Role::RfSensing => config.signal_scale_rf,
            Role::Compute => config.signal_scale_compute,
            Role::Relay => config.signal_scale_relay,
        }
    }

    /// Transmit power, in units of P0: N0 · (1 + (range / scale)²), chosen so that with
    /// no jammer the signal drops to the noise floor exactly at the radio range (D03).
    pub fn tx_power(self, config: &SimConfig) -> f64 {
        config.noise_floor
            * (1.0
                + ((self.radio_range(config) / self.signal_scale(config))
                    * (self.radio_range(config) / self.signal_scale(config))))
    }
}

/// A 2D point or vector in the arena, in km (velocities in km per step).
/// `Copy` because it is a small value, like a number.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec2 {
    /// Along the valley, 0 to `arena_w`.
    pub x: f64,
    /// Across the valley, 0 to `arena_h`.
    pub y: f64,
}

/// One drone: its identity and its state at one moment.
/// Not `Copy`: changing a copy by mistake would leave the swarm unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Drone {
    /// Stable identity, survives attrition. Not the index in the swarm vector.
    pub id: DroneId,
    /// The squad this drone flies with.
    pub squad: SquadId,
    /// What the drone does; also decides its radio range.
    pub role: Role,
    /// Position in the arena, km.
    pub position: Vec2,
    /// Velocity, km per step. Zero when the swarm is on station.
    pub velocity: Vec2,
    /// Battery level: 0 = empty, 1 = full.
    pub battery: f64,
    /// Sensor quality: 0 = useless, 1 = perfect.
    pub sensor_quality: f64,
    /// Compute capacity: 0 = none, 1 = the best on board.
    pub compute_capacity: f64,
}

impl Drone {
    /// A drone on station with the default settings for its role:
    /// zero velocity, full battery, perfect sensor, and compute capacity
    /// high for compute drones and low for every other role.
    ///
    /// `spawn_swarm` then overwrites the fields it randomizes.
    pub fn new(id: DroneId, squad: SquadId, role: Role, position: Vec2) -> Self {
        let compute_capacity = match role {
            Role::Compute => 1.0,
            Role::Recon | Role::RfSensing | Role::Relay => 0.2,
        };

        Drone {
            id,
            squad,
            role,
            position,
            velocity: Vec2 { x: 0.0, y: 0.0 },
            battery: 1.0,
            sensor_quality: 1.0,
            compute_capacity,
        }
    }
}

impl Vec2 {
    /// Euclidean distance to `other`, in km.
    pub fn dist(self, other: Vec2) -> f64 {
        let y_diff: f64 = self.y - other.y;
        let x_diff: f64 = self.x - other.x;

        let x_squared: f64 = x_diff * x_diff;
        let y_squared: f64 = y_diff * y_diff;

        (x_squared + y_squared).sqrt()
    }
}

// Unit tests: compiled only by `cargo test`.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_lists_every_role_once() {
        assert_eq!(Role::num_roles(), 4);
        for (i, a) in Role::ALL.iter().enumerate() {
            for b in &Role::ALL[i + 1..] {
                assert_ne!(a, b, "{a:?} listed twice in Role::ALL");
            }
        }
    }

    #[test]
    fn one_hot_has_a_single_one() {
        for role in Role::ALL {
            let v = role.one_hot();
            assert_eq!(v.iter().sum::<f64>(), 1.0, "{role:?} does not sum to 1");
            assert_eq!(
                v.iter().filter(|&&x| x == 1.0).count(),
                1,
                "{role:?} must have exactly one 1.0"
            );
        }
    }

    #[test]
    fn one_hots_are_all_different() {
        // Two roles sharing a one-hot would be indistinguishable to every model.
        for a in Role::ALL {
            for b in Role::ALL {
                if a != b {
                    assert_ne!(a.one_hot(), b.one_hot(), "{a:?} and {b:?} collide");
                }
            }
        }
    }

    #[test]
    fn ids_compare_by_value() {
        assert_eq!(DroneId(3), DroneId(3));
        assert_ne!(DroneId(3), DroneId(4));
        // Ord is used later to break ties in max-consensus (Mission 5).
        assert!(DroneId(3) < DroneId(4));
    }

    #[test]
    fn json_format() {
        // This is how the datasets will look on disk.
        assert_eq!(serde_json::to_string(&DroneId(17)).unwrap(), "17");
        assert_eq!(serde_json::to_string(&SquadId(2)).unwrap(), "2");
        assert_eq!(serde_json::to_string(&Role::Relay).unwrap(), "\"Relay\"");
        // And it reads back to the same value.
        let back: Role = serde_json::from_str("\"RfSensing\"").unwrap();
        assert_eq!(back, Role::RfSensing);
    }

    #[test]
    fn radio_range_by_role() {
        let cfg = SimConfig::default();
        assert_eq!(Role::Relay.radio_range(&cfg), cfg.range_long);
        for role in [Role::Recon, Role::RfSensing, Role::Compute] {
            assert_eq!(role.radio_range(&cfg), cfg.range_short, "{role:?}");
        }
        // The range follows the config, so experiments can change it.
        let long_relays = SimConfig {
            range_long: 12.0,
            ..Default::default()
        };
        assert_eq!(Role::Relay.radio_range(&long_relays), 12.0);
    }

    /// Every scale set to a different value, so a swapped match arm shows up.
    fn distinct_scales() -> SimConfig {
        SimConfig {
            signal_scale_recon: 0.1,
            signal_scale_rf: 0.2,
            signal_scale_relay: 0.3,
            signal_scale_compute: 0.4,
            ..Default::default()
        }
    }

    #[test]
    fn signal_scale_by_role() {
        let cfg = distinct_scales();
        assert_eq!(Role::Recon.signal_scale(&cfg), 0.1);
        assert_eq!(Role::RfSensing.signal_scale(&cfg), 0.2);
        assert_eq!(Role::Relay.signal_scale(&cfg), 0.3);
        assert_eq!(Role::Compute.signal_scale(&cfg), 0.4);
    }

    #[test]
    fn tx_power_hand_values() {
        // Scale 1 everywhere: N0 (1 + r²). Recon 0.05 * 7.25, relay 0.05 * 257.
        // N0 pinned at 0.05 so tuning the default cannot break the hand values.
        let cfg = SimConfig {
            noise_floor: 0.05,
            signal_scale_recon: 1.0,
            signal_scale_relay: 1.0,
            ..Default::default()
        };
        assert!((Role::Recon.tx_power(&cfg) - 0.3625).abs() < 1e-12);
        assert!((Role::Relay.tx_power(&cfg) - 12.85).abs() < 1e-12);
        // Halving the scale makes the transmitter louder: 0.05 * (1 + 25) = 1.3.
        let loud = SimConfig {
            signal_scale_recon: 0.5,
            ..cfg
        };
        assert!((Role::Recon.tx_power(&loud) - 1.3).abs() < 1e-12);
    }

    #[test]
    fn with_no_jammer_the_signal_reaches_the_noise_floor_exactly_at_the_range() {
        // signal(d) = P / (1 + (d / s)²) equals N0 at d = range, for every role and scale.
        for cfg in [SimConfig::default(), distinct_scales()] {
            for role in Role::ALL {
                let (r, s) = (role.radio_range(&cfg), role.signal_scale(&cfg));
                let signal_at_range = role.tx_power(&cfg) / (1.0 + (r / s).powi(2));
                assert!(
                    (signal_at_range - cfg.noise_floor).abs() < 1e-12,
                    "{role:?}: {signal_at_range}"
                );
            }
        }
    }

    #[test]
    fn smaller_scale_means_louder_transmitter() {
        let mut cfg = SimConfig::default();
        let mut last = 0.0;
        for s in [2.0, 1.0, 0.5, 0.25] {
            cfg.signal_scale_recon = s;
            let p = Role::Recon.tx_power(&cfg);
            assert!(p > last, "scale {s}: {p}");
            last = p;
        }
    }

    #[test]
    fn new_drone_has_role_defaults() {
        let pos = Vec2 { x: 4.0, y: 5.0 };
        for role in Role::ALL {
            let d = Drone::new(DroneId(7), SquadId(1), role, pos);
            assert_eq!(d.id, DroneId(7));
            assert_eq!(d.squad, SquadId(1));
            assert_eq!(d.role, role);
            assert_eq!(d.position, pos);
            assert_eq!(d.velocity, Vec2 { x: 0.0, y: 0.0 }, "on station");
            assert_eq!(d.battery, 1.0);
            let expected_compute = if role == Role::Compute { 1.0 } else { 0.2 };
            assert_eq!(d.compute_capacity, expected_compute, "{role:?}");
        }
    }

    #[test]
    fn dist_three_four_five() {
        let a = Vec2 { x: 0.0, y: 0.0 };
        let b = Vec2 { x: 3.0, y: 4.0 };
        assert!((a.dist(b) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn dist_is_symmetric_and_zero_to_self() {
        let a = Vec2 { x: 1.5, y: -2.0 };
        let b = Vec2 { x: 7.25, y: 3.0 };
        assert_eq!(a.dist(b), b.dist(a));
        assert_eq!(a.dist(a), 0.0);
    }
}
