//! Classical graph methods: plain loops and faer, no neural network.
//!
//! - `lp`: label propagation, iterated and closed form (Mission 1).
//! - `rwr`: random walk with restart (Mission 1).
//! - `baselines`: majority class and nearest reporter (km or hops), what the methods must beat.
//! - `eval`: Mission 1 evaluation: reporter draws, methods, scoring.
//! - `m1`: the Mission 1 experiment (labels, graphs, the sweep, CSV and tables).

pub mod baselines;
pub mod eval;
pub mod lp;
pub mod m1;
pub mod rwr;
