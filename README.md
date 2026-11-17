# AERIE

A graph-learning brain for a simulated 100-drone reconnaissance swarm, written in Rust.
Drones are nodes, radio links are edges, sensor readings are graph signals.
The full plan is in [Project_Plan.md](Project_Plan.md).
Design decisions and the evidence behind them are in [DECISIONS.md](DECISIONS.md).

## Layout

```
AERIE/
├── Cargo.toml              workspace: members, shared dependency versions
├── DECISIONS.md            decision log: D00 ground rules, D01 epsilon-graph tuning, ...
├── crates/
│   ├── aerie-graph/        graph storage: Edge, EdgeList, degrees, neighbours, components
│   ├── swarmsim/           the simulator (Mission 0)
│   │   ├── examples/       small programs using the crate, e.g. epsilon_stats.rs
│   │   └── src/
│   │       ├── config.rs   SimConfig: squads, arena, cells, radio ranges, k, jamming (P0, d0, ridge factor, threshold, noise)
│   │       ├── drone.rs    DroneId, SquadId, Role, Vec2, Drone
│   │       ├── spawn.rs    spawn_swarm: places the squads and drones
│   │       ├── jamming.rs  Jammer, Ridge, jam_power (falloff, ridge shadow), place_jamming (D02), true_power, readings, jammed_labels
│   │       ├── graphs.rs   declares the graph modules below
│   │       └── graphs/
│   │           ├── epsilon/   epsilon_graph (D00 rule), textbook_epsilon_graph, tests
│   │           ├── knn/       knn_graph (union rule), tests
│   │           └── stats.rs   stats(): mean degree, components, cross-squad share
│   └── aerie-viz/          GUIs (egui): one binary per GUI in src/bin/
│       └── src/bin/        swarm_viewer.rs, graph_playground.rs, jamming_playground.rs
└── target/                 build output (not committed)
```

## Everyday commands

Run them from the repo root.

| What | Command |
|---|---|
| Build everything | `cargo build` |
| Build one crate | `cargo build -p swarmsim` |
| Run all tests | `cargo test` |
| Test one crate | `cargo test -p swarmsim` |
| Run tests whose name contains a word | `cargo test -p swarmsim spawn` |
| Show `println!` output in tests | `cargo test -p swarmsim -- --nocapture` |
| Format the code | `cargo fmt` |
| Lint (style and common mistakes) | `cargo clippy --all-targets` |
| Lint one crate | `cargo clippy -p swarmsim --all-targets` |
| Check it compiles, without building binaries | `cargo check` |
| Open the docs of the whole workspace | `cargo doc --open` |

Before committing: `cargo fmt`, then `cargo clippy --all-targets`, then `cargo test`.

Crate names: `aerie-graph`, `swarmsim`, `aerie-viz`.

## Examples

Small programs in `crates/swarmsim/examples/`. Run with `--release`.

| Example | Command | What it prints |
|---|---|---|
| ε-graph stats | `cargo run --release -p swarmsim --example epsilon_stats` | one row per seed: links, mean degree, components, largest, isolated, share between squads; then the mean and the targets (D01) |

## GUIs

Always use `--release` for the GUIs: they are much smoother.

| GUI | Command |
|---|---|
| Swarm viewer (default) | `cargo run --release -p aerie-viz` |
| Swarm viewer, by name | `cargo run --release -p aerie-viz --bin swarm_viewer` |
| Graph playground | `cargo run --release -p aerie-viz --bin graph_playground` |
| Jamming playground | `cargo run --release -p aerie-viz --bin jamming_playground` |

### Swarm viewer

Shows a whole spawned swarm: squads, roles, grid cells and the comm graph (ε-graph).

- **Left panel:** mission, seed (◀ ▶ to step), cell size, squad count and size, spread, radio ranges, layers, legend.
- **Links layer:** width = weight; colour by weight, or by inside / between squads
  (links between squads stand out). Select a drone to highlight its links.
- **Map:** scroll to zoom, drag to pan, click a drone to select it, hover for details.
- **Right panel:** swarm summary, link stats from `stats()` with the targets, the selected
  drone (links, cross-squad links, weighted degree), the squads table.
- Seeds 0-79 train, 80-89 validate, 90-99 test.

### Graph playground

Place drones by hand and watch the comm graph form. It calls the real builders
`textbook_epsilon_graph`, `epsilon_graph` and `knn_graph` from `swarmsim::graphs`.

- **Modes:**
  - *Textbook*: one ε and one σ for every pair, roles ignored.
  - *Swarm (D00)*: ε = the smaller of the two radio ranges, σ = ε / 2.
  - *kNN (union)*: each drone picks its k nearest (slider), linked if either picked.
    Arrows show each drone's own picks (two arrows = mutual pick). Links longer than
    the radio range are dashed: kNN has no cut-off, they just weigh less than 0.135.
- **Canvas:**

  | Action | Effect |
  |---|---|
  | Click empty space | add a drone (role chosen in the left panel) |
  | Drag a drone | move it (snaps to 0.5 km if enabled) |
  | Click a drone | select it: its edges stay bright, σ ring shown |
  | Right-click a drone | delete it |
  | Delete key | delete the selected drone |
  | Esc | deselect |
  | Scroll | zoom around the cursor |
  | Drag empty space | pan (*Reset view* in the left panel) |

- **Presets:** *Test cases* (recon pair at 4 km, recon-relay at 6 km, relay pair at 8 km;
  swarm mode expects weights 0.278, no link, 0.206), *kNN line* (x = 0, 1, 3, 7, k = 1:
  a path), *kNN hub* (k = 1: the centre gets 4 links > 2k), *Random* (count and seed), *Clear*.
- **Right panel:** the Gaussian kernel w(d) = exp(−d² / 2σ²), cut at ε (no cut in kNN mode).
  Each edge is a dot on the curve, and you can hover the plot to read values. Below it: graph
  stats from `stats()` with ✔ / ✘ against the targets, and the edge list.
- **Squads:** each drone has a squad (picker in the left panel), shown as `S1` next to its
  number. Only used by the share-between-squads stat.
- **Dashed lines:**

  | Where | Meaning |
  |---|---|
  | Grey dashes between drones (Swarm mode) | one-way pair: in the longer range but not the shorter one, so no link |
  | Dashed edge (kNN mode) | kNN link longer than the radio range |
  | Dashed ring around the selected drone | σ ring, where the weight is 0.607 |
  | Dashed curve on the kernel plot | weight beyond ε, where no edge exists |

### Jamming playground

Place jammers and a ridge, and see the jamming power everywhere. The heatmap calls the
real `swarmsim::jamming::jam_power` at every pixel (0.2 km).

- **Tools:** *Move* (drag jammers, a ridge end or the whole ridge; drag empty space to pan),
  *Add jammer* (click), *Draw ridge* (drag from one end to the other; one ridge).
  Right-click a jammer or the ridge to delete it, Delete key removes the selected jammer,
  scroll zooms.
- **Left panel:** each jammer's P0 and position, ridge ends, physics sliders (d0, ridge
  factor, jammed threshold), how several jammers combine (sum or max, not decided yet),
  presets (*Test setup* = the unit-test geometry moved by (10, 10)), display options.
- **Map:** heatmap (linear or log scale), contours (white = jammed threshold, grey = 0.2 and
  0.1), dashed d0 ring around each jammer. Lines from each jammer to the cursor are dashed
  blue when the ridge blocks them.
- **Swarm layer:** a spawned swarm (seed); jammed drones get a red ring, failed links are red
  and dashed.
- **Place from seed:** runs `place_jamming` on the chosen seed's swarm (spawn, then place, same
  RNG, exactly like the dataset) and puts that jammer and ridge on the map. Tick *on every seed
  change* to re-place automatically while you step through seeds with ◀ ▶.
  The *min / max jammed share* sliders (`jam_frac_min`, `jam_frac_max`) and the physics
  sliders (*P0*, *d0*, *ridge factor*, *jammed threshold*) re-run the placement live as you
  drag them. The threshold is a fraction of P0, as in `place_jamming`.
- **Readings (2c):** each drone's true power comes from `true_power`, its sensor reading from
  `readings` (truth + Normal(0, noise_sd · P0)), its label from `jammed_labels`. The *noise sd* slider changes the noise live.
  *Dataset draw* is the noise the dataset gets (the seed's RNG, continued after spawn and
  `place_jamming`); *New draw* gives fresh noise for the same truth. Moving the jammer changes
  the truth but not each drone's noise.
  *Colour drones by*: role, true power, reading, or *label vs reading*:

  | Colour | Label (truth) | Reading |
  |---|---|---|
  | red | jammed | above the threshold |
  | orange | jammed | below the threshold (missed) |
  | purple | not jammed | above the threshold (false alarm) |
  | grey | not jammed | below the threshold |

  The red ring is always the label. Turn the heatmap off to see the colours better.
  Hover a drone for its truth, reading and noise.
- **Right panel:** power against distance (open ground and behind the ridge), the value table,
  the power at the cursor per jammer (clear or blocked), the effect on the swarm before
  and after jamming (links, mean degree, components, isolated drones), and readings against
  truth: counts per colour, the measured noise (mean, sd), and a scatter plot of truth (x)
  against reading (y) for every drone. Hover a dot to ring that drone on the map.

## Adding things

- **A new crate:** `cargo new crates/<name> --lib --vcs none`. The `--vcs none` flag stops it
  from creating a nested git repo. Then use `edition.workspace = true` and
  `dep = { workspace = true }` in its `Cargo.toml`.
- **A new dependency:** add the version once under `[workspace.dependencies]` in the root
  `Cargo.toml`, then `name = { workspace = true }` in the crate that uses it.
- **A new GUI:** add a file in `crates/aerie-viz/src/bin/`. It becomes a binary named after the
  file, run with `cargo run --release -p aerie-viz --bin <file_name>`.
- **A new example:** add a file in `crates/swarmsim/examples/` with a `fn main()`. Import
  with `use swarmsim::...` (not `crate::`), run with `--example <file_name>`.
- **A new decision:** add an entry `Dxx` to `DECISIONS.md` (Chose / Rejected / Evidence /
  Cost / Revisit).
- **A new graph recipe:** add a folder in `crates/swarmsim/src/graphs/` with `mod.rs`,
  `builder.rs` and `tests.rs`, and declare it with `pub mod <name>;` in `graphs.rs`.

## Reproducibility

All randomness goes through a seeded `ChaCha8Rng` passed as `&mut`.
Never use `rand::rng()` (thread-local) in simulation code.
The same seed and config always give the same swarm.
