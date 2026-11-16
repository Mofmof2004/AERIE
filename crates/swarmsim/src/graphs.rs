//! Builders of the comm graph from a swarm: the construction methods of lecture 1.
//!
//! One folder per recipe in `graphs/`, each with:
//! - `mod.rs`: what the recipe is, and what the folder exports
//! - `builder.rs`: the function that builds the `aerie_graph::EdgeList`
//! - `tests.rs`: its unit tests
//!
//! The correlation graph gets its own folder in step 4, once time series exist.

pub mod epsilon;
pub mod knn;
pub mod stats;
