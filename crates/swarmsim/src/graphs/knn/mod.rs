//! k-nearest-neighbour graph.
//!
//! Each drone picks its k closest drones; the graph keeps the union, so i and j are
//! linked when either one picked the other (decision D00). Every drone has at least
//! k links and the mean degree is between k and 2k, but one drone can have many more:
//! nothing limits how many others pick it (a hub). Weight: the same Gaussian kernel
//! as the epsilon graph, so links beyond radio range get weights below exp(-2).

pub mod builder;

// Re-exported so callers write `graphs::knn::knn_graph`.
pub use builder::knn_graph;

#[cfg(test)]
mod tests;
