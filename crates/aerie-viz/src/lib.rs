//! Shared pieces for AERIE's GUIs: map drawing, styles, squad summaries, missions.
//!
//! Each GUI is its own binary in `src/bin/`. Run one with
//! `cargo run --release -p aerie-viz --bin <name>`; the default is `swarm_viewer`.

pub mod grid;
pub mod launch;
pub mod m1_plot;
pub mod m1_view;
pub mod map;
pub mod mission;
pub mod squads;
pub mod style;
