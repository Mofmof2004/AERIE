//! Node embeddings (Mission 2): a vector per drone learned from the traffic graph alone.
//!
//! - `walks`: random walk generator (DeepWalk's input), later Node2Vec's biased walks.
//!
//! Coming next: skip-gram with negative sampling (DeepWalk), HOPE, matrix factorisation.

pub mod walks;
