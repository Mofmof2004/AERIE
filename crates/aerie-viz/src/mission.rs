//! The missions the viewer knows about. Each one gets its own view once its
//! code exists; until then the viewer says what the view will show.
//!
//! To add a mission view: mark it `available`, then draw its extra layers in
//! `main.rs` where the central panel matches on the mission.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mission {
    M0,
    M1,
    M2,
    M3,
    M4,
    M5,
    M6,
    M7,
    M8,
    M9,
    M10,
    M11,
}

impl Mission {
    pub const ALL: [Mission; 12] = [
        Mission::M0,
        Mission::M1,
        Mission::M2,
        Mission::M3,
        Mission::M4,
        Mission::M5,
        Mission::M6,
        Mission::M7,
        Mission::M8,
        Mission::M9,
        Mission::M10,
        Mission::M11,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Mission::M0 => "M0  SwarmSim",
            Mission::M1 => "M1  First Contact",
            Mission::M2 => "M2  Fingerprints",
            Mission::M3 => "M3  Clean Signal",
            Mission::M4 => "M4  Gossip",
            Mission::M5 => "M5  The Bid",
            Mission::M6 => "M6  X-Ray",
            Mission::M7 => "M7  Forecast",
            Mission::M8 => "M8  The Swarm Moves",
            Mission::M9 => "M9  Ten Thousand Drones",
            Mission::M10 => "M10 Why Drone 17?",
            Mission::M11 => "M11 Red Team",
        }
    }

    /// What this mission's view shows (or will show).
    pub fn description(self) -> &'static str {
        match self {
            Mission::M0 => {
                "The swarm on station: squads, roles, grid cells and the \
                 comm graph (epsilon graph, decision D00)."
            }
            Mission::M1 => "Which drones report, and the labels propagated to the silent ones.",
            Mission::M2 => "Embeddings learned from the traffic log, coloured by role or squad.",
            Mission::M3 => "The jamming map: true field, noisy readings and the reconstruction.",
            Mission::M4 => {
                "Gossip rounds spreading the filter output K hops, with the emission counter."
            }
            Mission::M5 => {
                "Each drone's bid for a task, the winner, and the regret against the oracle."
            }
            Mission::M6 => "The graph spectrum and the spoof detector's high-pass output.",
            Mission::M7 => "Jamming and battery forecast five steps ahead.",
            Mission::M8 => "Flight replay with link churn and the split warning.",
            Mission::M9 => "Swarms of 1k to 100k drones and sampled computation graphs.",
            Mission::M10 => "Why a drone won a bid: the edges and features that decided it.",
            Mission::M11 => "What a captured drone reveals about the rest of the swarm.",
        }
    }

    /// Whether this mission has a view yet.
    pub fn available(self) -> bool {
        matches!(self, Mission::M0 | Mission::M1)
    }
}
