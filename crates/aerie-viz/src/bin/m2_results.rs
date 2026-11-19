//! Mission 2 results on their own: the chart and the table of `results/m2/embeddings.csv`,
//! with a button that runs the experiment (the plan's 300-drone swarm, default DeepWalk).
//! The swarm viewer's Mission 2 view shows the same next to the map, with settings.
//!
//! Run with `cargo run --release -p aerie-viz --bin m2_results`.

use aerie_classic::embed::deepwalk::DeepWalkConfig;
use aerie_classic::m2::{EXPERIMENT_BIASES, plan_config};
use aerie_viz::m2_view::{Runner, results};
use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE Mission 2 results")
            .with_inner_size([1100.0, 520.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE Mission 2 results",
        options,
        Box::new(|_cc| Ok(Box::new(Viewer::default()))),
    )
}

#[derive(Default)]
struct Viewer {
    runner: Runner,
}

impl eframe::App for Viewer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.runner.poll(ui.ctx());
        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading("Mission 2 results");
            ui.weak(
                "DeepWalk and Node2Vec (p = 1, q = 0.5 and q = 2) on one flight's traffic \
                 graph, 300 drones, test seeds 90–99, the plan's settings. Role F1 · link AUC \
                 · squad NMI, against the baselines.",
            );
            self.runner.buttons(
                ui,
                &plan_config(),
                &DeepWalkConfig::default(),
                &EXPERIMENT_BIASES,
            );
            ui.separator();
            results(ui, &self.runner.rows);
        });
    }
}
