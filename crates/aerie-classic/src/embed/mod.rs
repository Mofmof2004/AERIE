//! Node embeddings (Mission 2): a vector per drone learned from the traffic graph alone.
//!
//! - `walks`: random walk generator (DeepWalk's input), later Node2Vec's biased walks.
//! - `skipgram`: skip-gram with negative sampling, the learning half of DeepWalk.
//! - `deepwalk`: the two together, one call from a graph to Z.
//! - `node2vec`: DeepWalk with walks biased by p (going back) and q (moving away).
//! - `mf`: matrix factorisation of A, the simplest baseline (direct links only).
//! - `score`: what a listener reads from Z: role macro-F1, link AUC, squad NMI.
//!
//! Still missing (Mission 2 is parked, not finished): HOPE, Katz similarity on the directed
//! traffic graph factorised by SVD into source and target vectors, for experiment 3.

pub mod deepwalk;
pub mod mf;
pub mod node2vec;
pub mod score;
pub mod skipgram;
pub mod walks;
