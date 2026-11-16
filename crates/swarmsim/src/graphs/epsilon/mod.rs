//! Weighted epsilon-neighbourhood graph.
//!
//! Two drones are linked when their distance is at most the smaller of their two
//! radio ranges (decision D00). Link weight: Gaussian kernel exp(-d^2 / (2 sigma^2))
//! with sigma = half the link range.

pub mod builder;

// Once `epsilon_graph` exists in builder.rs, export it from here:
// pub use builder::epsilon_graph;

#[cfg(test)]
mod tests;
