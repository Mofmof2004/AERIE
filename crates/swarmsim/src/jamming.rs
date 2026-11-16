//! Jamming: the jammer, the ridge, and the jamming power at any point.

use crate::config::SimConfig;
use crate::drone::{Drone, Vec2};
use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// An enemy emitter. Its power falls off with distance (see `jam_power`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Jammer {
    pub position: Vec2,
    /// P0, the power at the jammer itself. We use 1.0, so powers are fractions of P0.
    pub power: f64,
}

/// A straight ridge from `a` to `b`. It blocks the line of sight to the jammer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ridge {
    pub a: Vec2,
    pub b: Vec2,
}

/// Which side of the line a -> b the point c is on:
/// positive = left, negative = right, zero = on the line.
fn orient(a: Vec2, b: Vec2, c: Vec2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// True when segment p1-p2 crosses segment q1-q2 (each pair of ends on opposite
/// sides of the other segment). Touching at an end point does not count.
fn segments_cross(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2) -> bool {
    orient(p1, p2, q1) * orient(p1, p2, q2) < 0.0 && orient(q1, q2, p1) * orient(q1, q2, p2) < 0.0
}

/// Jamming power at `at`: P0 / (1 + (d / d0)^2), times `ridge_factor` when the
/// ridge lies between the jammer and `at`.
pub fn jam_power(jammer: &Jammer, ridge: &Ridge, at: Vec2, config: &SimConfig) -> f64 {
    let d = jammer.position.dist(at);
    let mut power = jammer.power / (1.0 + ((d / config.jam_d0) * (d / config.jam_d0)));

    if segments_cross(jammer.position, at, ridge.a, ridge.b) {
        power *= config.ridge_factor
    }

    power
}

/// Distance from the jammer to the middle of the ridge, km.
const RIDGE_DIST_MIN: f64 = 2.0;
const RIDGE_DIST_MAX: f64 = 3.0;
/// Ridge length, km.
const RIDGE_LEN_MIN: f64 = 10.0;
const RIDGE_LEN_MAX: f64 = 14.0;
/// Largest random tilt of the ridge away from facing the jammer, radians.
const RIDGE_TILT: f64 = 0.5;
/// Safety limit on redraws, so a bad config cannot loop forever.
const MAX_ATTEMPTS: usize = 10_000;

/// One jammer and one ridge for this swarm (decision D02, rejection sampling).
///
/// The jammer is drawn uniformly in the arena; the ridge's middle is 2 to 3 km from it,
/// roughly facing it, 10 to 14 km long. The pair is kept only if the fraction of
/// jammed drones (ridge included) is between `jam_frac_min` and `jam_frac_max`;
/// otherwise both are drawn again.
///
/// Panics if no placement works after `MAX_ATTEMPTS` tries.
pub fn place_jamming(swarm: &[Drone], config: &SimConfig, rng: &mut ChaCha8Rng) -> (Jammer, Ridge) {
    for _attempt in 0..MAX_ATTEMPTS {
        // 1. Jammer anywhere in the arena.
        let jammer = Jammer {
            position: Vec2 {
                x: rng.random_range(0.0..config.arena_w),
                y: rng.random_range(0.0..config.arena_h),
            },
            power: config.jam_p0,
        };

        // 2. Ridge middle: a random direction and distance from the jammer.
        let phi: f64 = rng.random_range(0.0..2.0 * PI);
        let r: f64 = rng.random_range(RIDGE_DIST_MIN..RIDGE_DIST_MAX);
        let middle = Vec2 {
            x: jammer.position.x + r * phi.cos(),
            y: jammer.position.y + r * phi.sin(),
        };

        // 3. Ridge direction: perpendicular to phi (facing the jammer), slightly tilted.
        let theta = phi + PI / 2.0 + rng.random_range(-RIDGE_TILT..RIDGE_TILT);

        // 4. Ridge ends, half the length on each side of the middle.
        let half = rng.random_range(RIDGE_LEN_MIN..RIDGE_LEN_MAX) / 2.0;
        let ridge = Ridge {
            a: Vec2 {
                x: middle.x - half * theta.cos(),
                y: middle.y - half * theta.sin(),
            },
            b: Vec2 {
                x: middle.x + half * theta.cos(),
                y: middle.y + half * theta.sin(),
            },
        };

        // 5. Fraction of drones jammed, ridge included.
        let threshold = config.jam_threshold * config.jam_p0;
        let mut jammed = 0;
        for drone in swarm {
            if jam_power(&jammer, &ridge, drone.position, config) >= threshold {
                jammed += 1;
            }
        }
        let fraction = jammed as f64 / swarm.len() as f64;

        // 6. Keep it if the jammed zone has a useful size.
        if fraction >= config.jam_frac_min && fraction <= config.jam_frac_max {
            return (jammer, ridge);
        }
    }
    panic!("no jamming placement after {MAX_ATTEMPTS} attempts");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default config with the jamming values these tests were written for
    /// (d0 = 4 km, ridge factor 0.1), so tuning the defaults cannot break them.
    fn config() -> SimConfig {
        SimConfig {
            jam_d0: 4.0,
            ridge_factor: 0.1,
            ..SimConfig::default()
        }
    }

    fn p(x: f64, y: f64) -> Vec2 {
        Vec2 { x, y }
    }

    /// Jammer at the origin with P0 = 1; vertical ridge at x = 2 from y = -5 to 5.
    fn setup() -> (Jammer, Ridge) {
        let jammer = Jammer {
            position: p(0.0, 0.0),
            power: 1.0,
        };
        let ridge = Ridge {
            a: p(2.0, -5.0),
            b: p(2.0, 5.0),
        };
        (jammer, ridge)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn full_power_at_the_jammer() {
        let (j, r) = setup();
        assert!(close(jam_power(&j, &r, p(0.0, 0.0), &config()), 1.0));
    }

    #[test]
    fn half_power_at_d0_in_open_ground() {
        let (j, r) = setup();
        assert!(close(jam_power(&j, &r, p(0.0, 4.0), &config()), 0.5));
    }

    #[test]
    fn a_fifth_at_twice_d0() {
        let (j, r) = setup();
        assert!(close(jam_power(&j, &r, p(0.0, 8.0), &config()), 0.2));
    }

    #[test]
    fn behind_the_ridge_is_cut_by_ridge_factor() {
        // Same 4 km as the open-ground case, but the wall is in between.
        let (j, r) = setup();
        assert!(close(jam_power(&j, &r, p(4.0, 0.0), &config()), 0.05));
    }

    #[test]
    fn standing_on_the_ridge_is_not_shadowed() {
        // orient(...) = 0 on the line, so the product is 0, not < 0: open ground at 2 km.
        let (j, r) = setup();
        assert!(close(jam_power(&j, &r, p(2.0, 0.0), &config()), 0.8));
    }

    #[test]
    fn slanted_line_through_the_ridge_is_shadowed() {
        // The line to (6, 10) meets x = 2 at y = 3.33, inside the ridge.
        let (j, r) = setup();
        let at = p(6.0, 10.0);
        let open = 1.0 / (1.0 + (136.0_f64.sqrt() / 4.0).powi(2));
        assert!(close(jam_power(&j, &r, at, &config()), 0.1 * open));
    }

    #[test]
    fn line_passing_beyond_the_ridge_end_is_not_shadowed() {
        // The line to (6, 20) meets x = 2 at y = 6.67, above the ridge end at y = 5.
        let (j, r) = setup();
        let at = p(6.0, 20.0);
        let open = 1.0 / (1.0 + (436.0_f64.sqrt() / 4.0).powi(2));
        assert!(close(jam_power(&j, &r, at, &config()), open));
    }

    #[test]
    fn power_follows_p0_d0_and_ridge_factor_from_inputs() {
        let (mut j, r) = setup();
        j.power = 2.0;
        let cfg = SimConfig {
            jam_d0: 2.0,
            ridge_factor: 0.5,
            ..SimConfig::default()
        };
        // d = d0 = 2 km in open ground: half of P0.
        assert!(close(jam_power(&j, &r, p(0.0, 2.0), &cfg), 1.0));
        // Behind the ridge at 4 km: 2 / (1 + 4) * 0.5.
        assert!(close(jam_power(&j, &r, p(4.0, 0.0), &cfg), 0.2));
    }

    #[test]
    fn segments_cross_basic_cases() {
        // An X crosses.
        assert!(segments_cross(
            p(0.0, 0.0),
            p(2.0, 2.0),
            p(0.0, 2.0),
            p(2.0, 0.0)
        ));
        // Swapping the segments or their ends gives the same answer.
        assert!(segments_cross(
            p(0.0, 2.0),
            p(2.0, 0.0),
            p(2.0, 2.0),
            p(0.0, 0.0)
        ));
        // Parallel segments never cross.
        assert!(!segments_cross(
            p(0.0, 0.0),
            p(2.0, 0.0),
            p(0.0, 1.0),
            p(2.0, 1.0)
        ));
        // A T that stops short of the other segment.
        assert!(!segments_cross(
            p(0.0, 0.0),
            p(2.0, 0.0),
            p(1.0, 0.5),
            p(1.0, 2.0)
        ));
    }

    #[test]
    fn orient_signs() {
        let (a, b) = (p(0.0, 0.0), p(1.0, 0.0));
        assert!(orient(a, b, p(0.5, 1.0)) > 0.0, "left of a -> b");
        assert!(orient(a, b, p(0.5, -1.0)) < 0.0, "right of a -> b");
        assert_eq!(orient(a, b, p(3.0, 0.0)), 0.0, "on the line");
    }
}
