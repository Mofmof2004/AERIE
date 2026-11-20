//! Opening the other GUIs from a menu. Each GUI is its own program (one file in `src/bin/`),
//! so "open" means starting that program in a separate window.

use std::path::PathBuf;
use std::process::{Command, Stdio};

/// One GUI of the workspace.
pub struct Gui {
    /// Binary name: the file name in `src/bin/`.
    pub bin: &'static str,
    pub title: &'static str,
    pub about: &'static str,
}

/// Every GUI, in menu order.
pub const GUIS: [Gui; 14] = [
    Gui {
        bin: "swarm_viewer",
        title: "Swarm viewer",
        about: "A seed's whole snapshot: swarm, jamming, live links, tasks.",
    },
    Gui {
        bin: "graph_playground",
        title: "Graph playground",
        about: "Place drones by hand: textbook ε, swarm ε (D00) and kNN graphs, kernel plot.",
    },
    Gui {
        bin: "jamming_playground",
        title: "Jamming playground",
        about: "Jammers, ridge, heatmap, readings, the link model (D03), Build your own.",
    },
    Gui {
        bin: "matrix_playground",
        title: "Matrix playground",
        about: "Place drones and see A, D, A² and P = D⁻¹A of their comm graph (Mission 1).",
    },
    Gui {
        bin: "lp_playground",
        title: "Label propagation playground",
        about: "Choose who reports, watch the labels spread round by round (Mission 1).",
    },
    Gui {
        bin: "m1_results",
        title: "Mission 1 results",
        about: "Accuracy against report share for every method and graph; runs the m1 runner.",
    },
    Gui {
        bin: "m2_results",
        title: "Mission 2 results",
        about: "Role F1, link AUC and squad NMI for DeepWalk and baselines; runs the m2 runner.",
    },
    Gui {
        bin: "skipgram_playground",
        title: "Skip-gram playground",
        about: "Watch DeepWalk learn pair by pair: walks, z and c vectors, loss (Mission 2).",
    },
    Gui {
        bin: "node2vec_playground",
        title: "Node2Vec playground",
        about: "Bend the walks with p and q, step by step; build graphs, paint labels (Mission 2).",
    },
    Gui {
        bin: "mf_playground",
        title: "Matrix factorisation playground",
        about: "Watch Z Zᵀ rebuild A epoch by epoch, with the math; hide links (Mission 2).",
    },
    Gui {
        bin: "graph_matrices_playground",
        title: "Graph matrices playground",
        about: "Build a graph, give drones values: A, D, B, L, BBᵀ, P, Bᵀx, Lx, xᵀLx (Mission 3).",
    },
    Gui {
        bin: "signal_playground",
        title: "Signal playground",
        about: "How bumpy is a signal: xᵀLx vs total variation, smoothing, the jamming field (Mission 3).",
    },
    Gui {
        bin: "regularizer_playground",
        title: "Regularizer playground",
        about: "Tikhonov, trend filtering, total variation, sparse total variation side by side (Mission 3).",
    },
    Gui {
        bin: "bias_variance_playground",
        title: "Bias–variance playground",
        about: "Experiment 1: bias², variance and total against γ, theory and simulated draws (Mission 3).",
    },
];

/// The workspace root, two levels above this crate (known at compile time).
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// The command that builds (if needed) and runs `bin` in release mode, from the
/// workspace root: always the latest code, like typing it in a terminal.
pub fn cargo_command(bin: &str) -> Command {
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--release", "-p", "aerie-viz", "--bin", bin])
        .current_dir(workspace_root())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd
}

/// Starts `bin` in its own window and returns a message for the status line. Falls back
/// to the built program next to this one when `cargo` cannot be started.
pub fn launch(bin: &str) -> String {
    if cargo_command(bin).spawn().is_ok() {
        return format!("Starting {bin} (built first if the code changed)…");
    }
    let sibling = std::env::current_exe()
        .ok()
        .map(|exe| exe.with_file_name(format!("{bin}{}", std::env::consts::EXE_SUFFIX)));
    match sibling {
        Some(path) if path.exists() => match Command::new(&path).spawn() {
            Ok(_) => format!("Started {bin} (last build: cargo was not found)."),
            Err(e) => format!("Could not start {}: {e}", path.display()),
        },
        _ => format!("Could not start {bin}: cargo not found and no built program."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_gui_in_the_menu_is_a_binary() {
        let bin_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("bin");
        for gui in &GUIS {
            assert!(
                bin_dir.join(format!("{}.rs", gui.bin)).exists(),
                "{}",
                gui.bin
            );
        }
    }

    #[test]
    fn cargo_command_runs_the_binary_from_the_workspace_root() {
        let cmd = cargo_command("jamming_playground");
        let args: Vec<_> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            [
                "run",
                "--release",
                "-p",
                "aerie-viz",
                "--bin",
                "jamming_playground"
            ]
        );
        let dir = cmd.get_current_dir().unwrap();
        assert!(dir.join("Cargo.toml").exists());
        assert!(dir.join("crates").join("aerie-viz").exists());
    }
}
