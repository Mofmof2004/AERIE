//! Weighted epsilon-neighbourhood graph.
//!
//! Two drones are linked when their distance is at most the smaller of their two
//! radio ranges (decision D00). Link weight: Gaussian kernel exp(-d^2 / (2 sigma^2))
//! with sigma = half the link range.

pub mod builder;

// Re-exported so callers write `graphs::epsilon::epsilon_graph`.
pub use builder::{epsilon_graph, textbook_epsilon_graph};

#[cfg(test)]
mod tests;
