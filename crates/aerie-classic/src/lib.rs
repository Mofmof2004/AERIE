//! Classical graph methods: plain loops and faer, no neural network.
//!
//! - `lp`: label propagation, iterated and closed form (Mission 1).
//! - `rwr`: random walk with restart (Mission 1).
//! - `baselines`: majority class and nearest reporter, what the methods must beat.

pub mod baselines;
pub mod lp;
pub mod rwr;
