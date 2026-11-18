//! Mission 1 results on their own: the plot and the table of `results/m1/summary.csv`,
//! with a button that runs the experiment. The swarm viewer's Mission 1 view shows the
//! same plot next to the map.
//!
//! Run with `cargo run --release -p aerie-viz --bin m1_results`.

use aerie_viz::m1_plot::{PlotState, Runner, plot, plot_controls, table};
use eframe::egui;
use swarmsim::config::SimConfig;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE Mission 1 results")
            .with_inner_size([1300.0, 850.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE Mission 1 results",
        options,
        Box::new(|_cc| Ok(Box::new(Viewer::default()))),
    )
}

#[derive(Default)]
struct Viewer {
    runner: Runner,
    plot: PlotState,
}

impl eframe::App for Viewer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.runner.poll(ui.ctx());
        egui::Panel::top("controls").show(ui, |ui| {
            ui.heading("Mission 1 results");
            self.runner.buttons(ui, &SimConfig::default(), 0.85);
            plot_controls(ui, &mut self.plot);
        });
        egui::Panel::bottom("table")
            .resizable(true)
            .default_size(220.0)
            .show(ui, |ui| {
                egui::ScrollArea::both()
                    .show(ui, |ui| table(ui, &self.runner.summary, &mut self.plot));
            });
        egui::CentralPanel::default().show(ui, |ui| plot(ui, &self.runner.summary, &self.plot));
    }
}
