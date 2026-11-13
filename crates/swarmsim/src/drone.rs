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
    pub fn one_hot(self) -> [f64; 4] {
        match self {
            Role::Recon => [1.0, 0.0, 0.0, 0.0],
            Role::RfSensing => [0.0, 1.0, 0.0, 0.0],
            Role::Relay => [0.0, 0.0, 1.0, 0.0],
            Role::Compute => [0.0, 0.0, 0.0, 1.0],
        }
    }
}

// Unit tests: compiled only by `cargo test`.
#[cfg(test)]
mod tests {
    use super::*;

    /// Every role once. If a fifth role is added, add it here too.
    const ALL_ROLES: [Role; 4] = [Role::Recon, Role::RfSensing, Role::Relay, Role::Compute];

    #[test]
    fn one_hot_has_a_single_one() {
        for role in ALL_ROLES {
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
        for a in ALL_ROLES {
            for b in ALL_ROLES {
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
}
