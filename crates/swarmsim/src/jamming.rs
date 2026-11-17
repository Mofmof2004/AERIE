//! Jamming: the jammer, the ridge, and the jamming power at any point.

use crate::config::SimConfig;
use crate::drone::{Drone, Vec2};
use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
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

/// The true jamming power at each drone, in swarm order (element `i` is drone `i`).
///
/// This is what the simulator knows. The drones only see noisy `readings` of it.
/// Zone labels and link failure are computed from these values.
pub fn true_power(swarm: &[Drone], jammer: &Jammer, ridge: &Ridge, config: &SimConfig) -> Vec<f64> {
    let mut true_power: Vec<f64> = Vec::new();
    for drone in swarm {
        let drone_jam: f64 = jam_power(jammer, ridge, drone.position, config);
        true_power.push(drone_jam);
    }
    true_power
}

/// What each drone's sensor measures: its true power plus Gaussian noise with
/// mean 0 and standard deviation `noise_sd * jam_p0`, in the same order as `truth`.
///
/// Not clamped: a reading can be slightly negative. The model sees these values,
/// never the truth.
pub fn readings(truth: &[f64], config: &SimConfig, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let sd = config.noise_sd * config.jam_p0;
    let normal = Normal::new(0.0, sd).unwrap();
    let mut reading: Vec<f64> = Vec::new();
    for power in truth {
        let reading_with_noise = power + normal.sample(rng);
        reading.push(reading_with_noise);
    }
    reading
}

pub fn jammed_labels(truth: &[f64], config: &SimConfig) -> Vec<bool> {
    let mut zone_label = Vec::new();
    for &power in truth {
        if power >= config.jam_threshold * config.jam_p0 {
            zone_label.push(true);
        } else {
            zone_label.push(false);
        }
    }

    zone_label
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

    // ------------------------------------------------------------ place_jamming

    use crate::spawn::spawn_swarm;
    use rand::SeedableRng;

    /// Spawns the swarm for `seed`, then places jamming with the same RNG
    /// (the dataset order).
    fn placed(seed: u64, cfg: &SimConfig) -> (Vec<Drone>, Jammer, Ridge) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let swarm = spawn_swarm(cfg, &mut rng);
        let (jammer, ridge) = place_jamming(&swarm, cfg, &mut rng);
        (swarm, jammer, ridge)
    }

    fn jammed_share(swarm: &[Drone], j: &Jammer, r: &Ridge, cfg: &SimConfig) -> f64 {
        let threshold = cfg.jam_threshold * cfg.jam_p0;
        let n = swarm
            .iter()
            .filter(|d| jam_power(j, r, d.position, cfg) >= threshold)
            .count();
        n as f64 / swarm.len() as f64
    }

    #[test]
    fn placement_jams_a_share_within_the_bounds() {
        let cfg = SimConfig::default();
        for seed in 0..20 {
            let (swarm, j, r) = placed(seed, &cfg);
            let share = jammed_share(&swarm, &j, &r, &cfg);
            assert!(
                (cfg.jam_frac_min..=cfg.jam_frac_max).contains(&share),
                "seed {seed}: {share}"
            );
        }
    }

    #[test]
    fn placement_is_deterministic() {
        let cfg = SimConfig::default();
        let (_, j1, r1) = placed(42, &cfg);
        let (_, j2, r2) = placed(42, &cfg);
        assert_eq!((j1, r1), (j2, r2));
        let (_, j3, _) = placed(43, &cfg);
        assert_ne!(j1, j3, "another seed, another jammer");
    }

    #[test]
    fn jammer_is_in_the_arena_with_power_p0() {
        let cfg = SimConfig {
            jam_p0: 1.7,
            ..SimConfig::default()
        };
        for seed in 0..10 {
            let (_, j, _) = placed(seed, &cfg);
            assert!((0.0..cfg.arena_w).contains(&j.position.x));
            assert!((0.0..cfg.arena_h).contains(&j.position.y));
            assert_eq!(j.power, 1.7);
        }
    }

    #[test]
    fn ridge_geometry_follows_the_constants() {
        let cfg = SimConfig::default();
        for seed in 0..20 {
            let (_, j, r) = placed(seed, &cfg);
            let len = r.a.dist(r.b);
            assert!(
                (RIDGE_LEN_MIN..RIDGE_LEN_MAX).contains(&len),
                "length {len}"
            );

            let middle = p((r.a.x + r.b.x) / 2.0, (r.a.y + r.b.y) / 2.0);
            let dist = j.position.dist(middle);
            assert!(
                (RIDGE_DIST_MIN - 1e-9..RIDGE_DIST_MAX + 1e-9).contains(&dist),
                "distance {dist}"
            );

            // Roughly facing the jammer: the ridge is within RIDGE_TILT of perpendicular
            // to the jammer -> middle direction, so |cos(angle)| <= sin(RIDGE_TILT).
            let (ux, uy) = ((r.b.x - r.a.x) / len, (r.b.y - r.a.y) / len);
            let (vx, vy) = (
                (middle.x - j.position.x) / dist,
                (middle.y - j.position.y) / dist,
            );
            let cos = (ux * vx + uy * vy).abs();
            assert!(cos <= RIDGE_TILT.sin() + 1e-9, "cos {cos}");
        }
    }

    #[test]
    fn the_ridge_shadows_some_of_the_arena() {
        // Facing the jammer, the ridge must block the line to a point just behind
        // its middle.
        let cfg = SimConfig::default();
        let (_, j, r) = placed(5, &cfg);
        let middle = p((r.a.x + r.b.x) / 2.0, (r.a.y + r.b.y) / 2.0);
        let behind = p(
            middle.x + 0.5 * (middle.x - j.position.x),
            middle.y + 0.5 * (middle.y - j.position.y),
        );
        assert!(segments_cross(j.position, behind, r.a, r.b));
    }

    #[test]
    #[should_panic(expected = "no jamming placement")]
    fn impossible_bounds_panic_instead_of_looping_forever() {
        // Every drone jammed at once is impossible: the squads span the whole arena.
        let cfg = SimConfig {
            jam_frac_min: 1.0,
            jam_frac_max: 1.0,
            ..SimConfig::default()
        };
        placed(0, &cfg);
    }

    // ---------------------------------------------------- 2c: truth, readings, labels

    /// `n` readings of the same true power, from a fixed seed.
    fn many_readings(truth: f64, n: usize, cfg: &SimConfig, seed: u64) -> Vec<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        readings(&vec![truth; n], cfg, &mut rng)
    }

    fn mean_and_sd(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        (mean, var.sqrt())
    }

    #[test]
    fn true_power_is_jam_power_at_each_drone_in_order() {
        let cfg = SimConfig::default();
        let (swarm, j, r) = placed(3, &cfg);
        let truth = true_power(&swarm, &j, &r, &cfg);
        assert_eq!(truth.len(), swarm.len());
        for (i, d) in swarm.iter().enumerate() {
            assert_eq!(truth[i], jam_power(&j, &r, d.position, &cfg));
        }
    }

    #[test]
    fn true_power_matches_the_hand_values() {
        // Drones moved to the 2a test points: d0 in open ground and d0 behind the ridge.
        let cfg = config();
        let (j, r) = setup();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut swarm = spawn_swarm(&cfg, &mut rng);
        swarm.truncate(2);
        swarm[0].position = p(0.0, 4.0);
        swarm[1].position = p(4.0, 0.0);
        let truth = true_power(&swarm, &j, &r, &cfg);
        assert!(close(truth[0], 0.5));
        assert!(close(truth[1], 0.05));
    }

    #[test]
    fn empty_swarm_gives_empty_outputs() {
        let cfg = SimConfig::default();
        let (j, r) = setup();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        assert!(true_power(&[], &j, &r, &cfg).is_empty());
        assert!(readings(&[], &cfg, &mut rng).is_empty());
        assert!(jammed_labels(&[], &cfg).is_empty());
    }

    #[test]
    fn readings_keep_length_and_order() {
        let cfg = SimConfig::default();
        let truth = [0.0, 0.2, 0.9, 0.45, 0.6];
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let reads = readings(&truth, &cfg, &mut rng);
        assert_eq!(reads.len(), truth.len());
        // Each reading stays within 5 sd of its own truth: the order is kept.
        for (r, t) in reads.iter().zip(&truth) {
            assert!((r - t).abs() < 5.0 * cfg.noise_sd);
        }
    }

    #[test]
    fn zero_noise_gives_the_truth_back() {
        let cfg = SimConfig {
            noise_sd: 0.0,
            ..SimConfig::default()
        };
        let truth = [0.0, 0.3, 0.7];
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        assert_eq!(readings(&truth, &cfg, &mut rng), truth.to_vec());
    }

    #[test]
    fn prediction_2_noise_has_mean_zero_and_sd_noise_sd() {
        // 10,000 readings of truth 0.3: mean 0.3 (standard error 0.001), sd 0.1.
        let cfg = SimConfig::default();
        let (mean, sd) = mean_and_sd(&many_readings(0.3, 10_000, &cfg, 7));
        assert!((mean - 0.3).abs() < 0.005, "mean {mean}");
        assert!((sd - 0.1).abs() < 0.005, "sd {sd}");
    }

    #[test]
    fn noise_sd_scales_with_p0() {
        // Noise sd = noise_sd * P0: doubling P0 doubles the noise.
        let cfg = SimConfig {
            jam_p0: 2.0,
            ..SimConfig::default()
        };
        let (_, sd) = mean_and_sd(&many_readings(0.6, 10_000, &cfg, 8));
        assert!((sd - 0.2).abs() < 0.01, "sd {sd}");
    }

    #[test]
    fn prediction_1_truth_045_reads_above_05_about_31_percent() {
        // 0.5 is half an sd above 0.45: P(Z > 0.5) = 0.309.
        let cfg = SimConfig::default();
        let reads = many_readings(0.45, 10_000, &cfg, 9);
        let above = reads.iter().filter(|&&r| r >= 0.5).count() as f64 / 10_000.0;
        assert!((above - 0.309).abs() < 0.02, "share above {above}");
    }

    #[test]
    fn readings_are_not_clamped() {
        // Truth 0: about half the readings are negative, and they are kept.
        let cfg = SimConfig::default();
        let reads = many_readings(0.0, 1_000, &cfg, 10);
        let negative = reads.iter().filter(|&&r| r < 0.0).count();
        assert!((400..=600).contains(&negative), "{negative} negative");
    }

    #[test]
    fn readings_are_deterministic_per_seed() {
        let cfg = SimConfig::default();
        let truth = [0.1, 0.5, 0.9];
        assert_eq!(
            many_readings(0.4, 50, &cfg, 11),
            many_readings(0.4, 50, &cfg, 11)
        );
        let mut a = ChaCha8Rng::seed_from_u64(11);
        let mut b = ChaCha8Rng::seed_from_u64(12);
        assert_ne!(
            readings(&truth, &cfg, &mut a),
            readings(&truth, &cfg, &mut b)
        );
    }

    #[test]
    fn labels_use_the_threshold_with_equality_jammed() {
        let cfg = SimConfig::default(); // threshold 0.5 * P0 = 0.5
        let truth = [0.0, 0.49, 0.5, 0.51, 1.0];
        assert_eq!(
            jammed_labels(&truth, &cfg),
            vec![false, false, true, true, true]
        );
    }

    #[test]
    fn label_threshold_is_a_fraction_of_p0() {
        let cfg = SimConfig {
            jam_p0: 2.0,
            ..SimConfig::default()
        }; // threshold 0.5 * 2 = 1.0
        assert_eq!(
            jammed_labels(&[0.6, 0.99, 1.0], &cfg),
            vec![false, false, true]
        );
    }

    #[test]
    fn prediction_3_label_and_reading_can_disagree_both_ways() {
        let cfg = SimConfig::default();
        let threshold = cfg.jam_threshold * cfg.jam_p0;
        // Truth 0.55 is jammed, yet some readings fall below the threshold (missed).
        assert!(jammed_labels(&[0.55], &cfg)[0]);
        assert!(
            many_readings(0.55, 1_000, &cfg, 13)
                .iter()
                .any(|&r| r < threshold)
        );
        // Truth 0.45 is not jammed, yet some readings land above it (false alarm).
        assert!(!jammed_labels(&[0.45], &cfg)[0]);
        assert!(
            many_readings(0.45, 1_000, &cfg, 14)
                .iter()
                .any(|&r| r >= threshold)
        );
    }

    #[test]
    fn placed_share_matches_the_labels() {
        // place_jamming's own count and jammed_labels(true_power) agree on every seed.
        let cfg = SimConfig::default();
        for seed in 0..20 {
            let (swarm, j, r) = placed(seed, &cfg);
            let labels = jammed_labels(&true_power(&swarm, &j, &r, &cfg), &cfg);
            let share = labels.iter().filter(|&&l| l).count() as f64 / swarm.len() as f64;
            assert_eq!(share, jammed_share(&swarm, &j, &r, &cfg));
            assert!((cfg.jam_frac_min..=cfg.jam_frac_max).contains(&share));
        }
    }
}
