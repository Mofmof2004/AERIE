//! Mission 1 runner: the whole experiment from the terminal (the swarm viewer's Mission 1
//! view runs the same sweep with a button).
//!
//! Run: cargo run --release -p aerie-classic --example m1
//!
//! Every method on every graph, for every label, report share and share of wrong reports
//! (0, 10, 25 %), over the 10 test seeds × 20 draws (`aerie_classic::m1::sweep`). Writes results/m1/runs.csv and summary.csv
//! at the repo root, and prints the main table (live graph, 10 % report) and experiment 1
//! (label propagation's accuracy against the report share, per graph).
//!
//! Scoring: accuracy on the silent drones; a drone with no guess counts as wrong.

use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

use aerie_classic::m1::{
    DRAWS, GraphKind, LabelKind, MAIN_SHARE, Progress, main_table, share_table, sweep, units,
    write_results, wrong_table,
};
use swarmsim::config::SimConfig;
use swarmsim::snapshot::TEST_SEEDS;

fn main() -> std::io::Result<()> {
    let config = SimConfig::default();
    let progress = Progress::default();
    let total = units(&TEST_SEEDS);

    let result = thread::scope(|s| {
        let job = s.spawn(|| sweep(&config, TEST_SEEDS, 0.85, &progress));
        while !job.is_finished() {
            let done = progress.done.load(Ordering::Relaxed);
            eprint!("\r{done} / {total} draws");
            thread::sleep(Duration::from_millis(250));
        }
        eprintln!("\r{total} / {total} draws");
        job.join().expect("the sweep thread panicked")
    });
    let result = result.map_err(std::io::Error::other)?;

    println!(
        "\nMain table: live graph, {:.0} % report, {} test seeds × {DRAWS} draws",
        MAIN_SHARE * 100.0,
        TEST_SEEDS.count()
    );
    println!("accuracy on the silent drones, % (in brackets: % with no guess)\n");
    print!(
        "{}",
        main_table(&result.summary, GraphKind::Live, MAIN_SHARE)
    );
    for label in LabelKind::ALL {
        println!(
            "\nExperiment 1, label = {}: label propagation accuracy, %",
            label.name()
        );
        print!("{}", share_table(&result.summary, label, "lp"));
    }

    for label in [LabelKind::Zone, LabelKind::Squad] {
        println!(
            "\nWrong reports, label = {}, live graph, {:.0} % report: accuracy on the silent \
             drones, % (in brackets: % of the lies overruled)",
            label.name(),
            MAIN_SHARE * 100.0
        );
        print!(
            "{}",
            wrong_table(&result.summary, GraphKind::Live, MAIN_SHARE, label)
        );
    }

    let dir = write_results(&result)?;
    println!("\nWrote {}", dir.display());
    Ok(())
}
