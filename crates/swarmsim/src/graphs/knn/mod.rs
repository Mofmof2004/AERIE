//! k-nearest-neighbour graph.
//!
//! Each drone picks its k closest drones; the graph keeps the union, so i and j are
//! linked when either one picked the other (decision D00). Each drone ends up with
//! between k and 2k links. Weight: the same Gaussian kernel as the epsilon graph.

pub mod builder;

// Once `knn_graph` exists in builder.rs, export it from here:
// pub use builder::knn_graph;

#[cfg(test)]
mod tests;
