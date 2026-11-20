# AERIE

A graph-learning brain for a simulated 100-drone reconnaissance swarm, written in Rust.
Drones are nodes, radio links are edges, sensor readings are graph signals.
The full plan is in [Project_Plan.md](Project_Plan.md).
Design decisions and the evidence behind them are in [DECISIONS.md](DECISIONS.md).

## Layout

```
AERIE/
├── Cargo.toml              workspace: members, shared dependency versions
├── DECISIONS.md            decision log: D00 ground rules, D01 ε tuning, D02 jamming, D03 link model, D04 tasks/oracle, D05 label propagation
├── RUST_NOTES.md           Rust questions and compiler mistakes, raw material for a cheat sheet
├── results/                experiment output, one folder per mission (runs.csv not committed)
├── crates/
│   ├── aerie-graph/        graph storage: Edge, EdgeList, degrees, neighbours, components
│   ├── swarmsim/           the simulator (Mission 0)
│   │   ├── examples/       small programs using the crate, epsilon_stats.rs, jamming_stats.rs
│   │   └── src/
│   │       ├── config.rs   SimConfig: squads, arena, cells, radio ranges, k, jamming (P0, d0, ridge factor, threshold, noise)
│   │       ├── drone.rs    DroneId, SquadId, Role, Vec2, Drone
│   │       ├── spawn.rs    spawn_swarm: places the squads and drones
│   │       ├── jamming.rs  Jammer, Ridge, jam_power (falloff, ridge shadow), place_jamming (D02), true_power, readings, jammed_labels, fail_links
│   │       ├── links.rs    the link model (D03): margin, LiveGraph, live_graph
│   │       ├── snapshot.rs Snapshot::build(seed): everything for one seed in the dataset order; TRAIN/VAL/TEST_SEEDS
│   │       ├── tasks.rs    TaskId, TaskKind (can_do, needs_relay, able), Task, spawn_tasks, relay_within_two_hops, CostMatrix, cost_matrix, single_winners, joint_winners, assignment_cost
│   │       ├── graphs.rs   declares the graph modules below
│   │       └── graphs/
│   │           ├── epsilon/   epsilon_graph (D00 rule), textbook_epsilon_graph, tests
│   │           ├── knn/       knn_graph (union rule), tests
│   │           └── stats.rs   stats(): mean degree, components, cross-squad share
│   ├── aerie-classic/      classical methods, no training (Mission 1)
│   │   ├── examples/m1.rs  the Mission 1 runner: every method × graph × label × share, test seeds × 20 draws
│   │   ├── examples/m2.rs  the Mission 2 runner: DeepWalk, Node2Vec (q = 0.5, 2) + baselines on the 300-drone traffic graph
│   │   └── src/
│   │       ├── lp.rs       label_propagation (iterated) and label_propagation_closed_form
│   │       ├── rwr.rs      random_walk_with_restart (iterated) and its closed form
│   │       ├── baselines.rs majority_class, nearest_reporter (hops), nearest_reporter_by_distance (km)
│   │       ├── eval.rs     Method, Problem, draw_reporters, hide, guesses, Score, score
│   │       ├── m1.rs       the Mission 1 experiment: LabelKind, GraphKind, sweep (seeds in parallel), CSV, tables
│   │       ├── m2.rs       the Mission 2 experiment: run_seed (DeepWalk, Node2Vec at each (p, q), three scores, baselines), sweep, CSV
│   │       └── embed/      Mission 2 embeddings: walks.rs (random_walks), skipgram.rs (pair_loss_and_grads, train_skipgram), deepwalk.rs (deepwalk), node2vec.rs (biased_walks, node2vec), mf.rs (matrix_factorization), score.rs (role F1, link AUC, squad NMI)
│   └── aerie-viz/          GUIs (egui): one binary per GUI in src/bin/; shared map.rs, m1_view.rs (Mission 1 view), m1_plot.rs (results plot), m2_view.rs (Mission 2 view, chart, runner)
│       └── src/bin/        swarm_viewer.rs, graph_playground.rs, jamming_playground.rs, matrix_playground.rs, lp_playground.rs, m1_results.rs, m2_results.rs, skipgram_playground.rs, node2vec_playground.rs, mf_playground.rs
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

Crate names: `aerie-graph`, `swarmsim`, `aerie-classic`, `aerie-viz`.

## Examples

Small programs in `crates/swarmsim/examples/`. Run with `--release`.

| Example | Command | What it prints |
|---|---|---|
| ε-graph stats | `cargo run --release -p swarmsim --example epsilon_stats` | one row per seed: links, mean degree, components, largest, isolated, share between squads; then the mean and the targets (D01) |
| Link stats | `cargo run --release -p swarmsim --example link_stats` | link-model calibration (D03): for several N0 and scale settings, links lost, live degree, far links kept, zone links kept, splits, stranded drones, one-way counts |
| Jamming stats | `cargo run --release -p swarmsim --example jamming_stats` | step 2 end to end, one row per seed: jammed share, relays jammed, links lost, pieces, stranded drones, readings on the wrong side; then the summary (D02) |
| Mission 1 runner | `cargo run --release -p aerie-classic --example m1` | about 1 minute (seeds in parallel). The swarm viewer's Mission 1 view runs the same sweep with a button. Every method (majority, nearest km, nearest hops, LP, RWR α 0.85) on every graph (live, ε, kNN), for every label (zone, squad, role), report share (1 % to 50 %) and share of wrong reports (0, 10, 25, 40 % of the reporters say a wrong class), test seeds 90–99 × 20 draws. Prints the main table (live graph, 10 %), experiment 1 and the wrong-reports tables (accuracy, and the share of lies each method overruled); writes `results/m1/runs.csv` and `results/m1/summary.csv`. No guess counts as wrong. |
| Mission 2 runner | `cargo run --release -p aerie-classic --example m2` | about 8 seconds (seeds in parallel). The swarm viewer's Mission 2 view runs the same with a button. DeepWalk, Node2Vec at p = 1, q = 0.5 and q = 2 (experiment 1; default walk and skip-gram settings) and matrix factorisation of A = ln(1 + messages) on the 300-drone traffic graph, test seeds 90–99, scored three ways: role macro-F1 (5-fold logistic regression), link AUC (10 % of the links hidden before training), squad NMI (k-means, k = 30). Baselines: random vectors, traffic counts (sent, received, partners; role only), majority class. Prints the table and the training pairs per epoch; writes `results/m2/embeddings.csv`. |

## GUIs

Always use `--release` for the GUIs: they are much smoother.

| GUI | Command |
|---|---|
| Swarm viewer (default) | `cargo run --release -p aerie-viz` |
| Swarm viewer, by name | `cargo run --release -p aerie-viz --bin swarm_viewer` |
| Graph playground | `cargo run --release -p aerie-viz --bin graph_playground` |
| Jamming playground | `cargo run --release -p aerie-viz --bin jamming_playground` |
| Matrix playground | `cargo run --release -p aerie-viz --bin matrix_playground` |
| Label propagation playground | `cargo run --release -p aerie-viz --bin lp_playground` |
| Mission 1 results | `cargo run --release -p aerie-viz --bin m1_results` |
| Mission 2 results | `cargo run --release -p aerie-viz --bin m2_results` |
| Skip-gram playground | `cargo run --release -p aerie-viz --bin skipgram_playground` |
| Node2Vec playground | `cargo run --release -p aerie-viz --bin node2vec_playground` |
| Matrix factorisation playground | `cargo run --release -p aerie-viz --bin mf_playground` |

### Swarm viewer

Shows a whole spawned swarm: squads, roles, grid cells and the comm graph (ε-graph). Each seed
is built in the dataset order with one RNG: `spawn_swarm`, `place_jamming`, `true_power`,
`readings`, `live_graph`, `spawn_tasks`, so the viewer shows exactly what the dataset holds.

- **Jammer, ridge, jammed zone (step 2):** the jammer (red disc), its d0 ring (dashed), the
  ridge (brown), and every drone in the jammed zone drawn in **lime** (a colour no squad, role
  or task uses). *Shade the jammed zone* fills the zone in a dark tint, darker where the jamming
  is stronger; it follows `jam_power`, so the ridge shadow shows as a gap.
- **Links shown:** *Geometric* (`epsilon_graph`, as if there were no jammer) or *Live*
  (`live_graph`, after jamming, the default). With live links, the geometric links that
  jamming killed are drawn red and dashed.
- **Tasks (step 3a):** 50 diamonds coloured by kind (O observe, T track, R relay, F fuse).
  Filled = placed near a drone, hollow = uniform. Hover a task for its roles, the number of
  drones that can take it and the nearest one. Click a task: drones whose role can take it get
  a ring in its colour, the others fade. The right panel counts tasks per kind and lists the
  5 closest drones whose role fits the selected task (click one to select it).
- **Relay rule (step 3b):** Observe and Track need a relay within 2 hops on the live graph
  (`relay_within_two_hops`). Click one of those tasks: drones whose role fits but that fail
  the rule get a faint ring and a **red cross**. The task table shows *By role* and *Able*
  (role and relay rule), the hover says how many are cut off, and the selected drone shows
  *Relay ≤ 2 hops*. On seeds 0–99 about 97 % of drones pass.
- **Costs (step 3c):** the selected task lists its 5 cheapest drones from the cost matrix,
  with the cost split into km + battery term (a flat penalty instead when not able). The cheapest (BEST)
  gets a **gold star** on the map and a gold dashed line from the task: the single-winner label.
- **Oracle labels (step 3d):** *single winner* = cheapest drone for the task alone (gold);
  *joint winner* = the Hungarian assignment over all tasks, each drone on at most one task
  (a **cyan star** and line when it is a different drone). The panel says which task took the
  single winner, and counts the tasks where the two labels differ. A task stays **open** when
  no able drone is left for it (about 1 seed in 10 has more Fuse tasks than the 15 compute
  drones). β and the penalty are in *Settings → Tasks and costs*.

- **Left panel:** mission, seed (◀ ▶ to step), cell size, squad count and size, spread, radio ranges, layers, legend.
- **Menu bar:** *Playgrounds* opens any GUI (graph playground, jamming playground, another swarm
  viewer) in its own window. It runs `cargo run --release` for it, so you always get the latest
  code; the first time it builds for a few seconds. *Settings* opens the settings window.
- **Red dashed lines:** between drones, a dead link (in range on a quiet day, killed by jamming:
  neither direction gets through). Around the jammer, the d0 ring (half power).
- **⚙ All settings…** (top of the left panel, or *Settings* in the menu bar): a window with every `SimConfig` field, in groups
  (swarm and arena, radio ranges, jamming, link model, tasks), each with *Reset group*, plus
  *Reset everything*. Changes apply live, and a line at the top shows the result: jammed share,
  links lost, mean degree, components, tasks. The arena is set in whole cells so every cell size
  stays valid. Values that would crash the simulator (a zero radius, swapped bounds) are guarded.
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

### Matrix playground

Mission 1: place drones by hand and see the matrices of their comm graph next to the map.
Links follow the swarm rule (`epsilon_graph`, D00) with Gaussian weights; the matrices come
from the real functions in `aerie-graph`.

- **A** (`adjacency_matrix`), **D** (`degree_matrix`), optional **A^k** with k from 1 to 8 (k-hop walks),
  and **P = D⁻¹A** (`transition_matrix`). Each grid has a Σ column with the row sums, and a
  check that every row of P sums to 1.
- **A is 0/1:** tick to give every link weight 1, so A only says who is linked. With the
  *Triangle + isolated* preset this gives exactly the grids of the `aerie-graph` tests.
- **Hover a cell** to light up its link on the map and read how the value is made (e.g.
  `P[1][2] = A[1][2] / D[1][1]`). Hover or select a drone to outline its row and column.
- Canvas: click to add a drone (role in the left panel), drag to move, right-click or Delete
  to remove. Presets: triangle + isolated, path of 4, star around a relay, two squads joined
  by their relays. Sliders for the two radio ranges.

### Label propagation playground

Mission 1: choose which drones report their class and watch label propagation
(`aerie_classic::lp::label_propagation`) or random walk with restart
(`aerie_classic::rwr::random_walk_with_restart`) spread it over the comm graph.

- **Method:** *Label propagation* (Y ← P′Y, reporters clamped) or *Random walk with restart*
  (F ← αPF + (1−α)Y₀, nothing clamped, α slider). Under RWR every drone gets a bar, since the
  reporters' rows can move too. The right panel shows both methods' converged accuracy side by
  side.
- **Drones:** each has a true class. Reporting drones (thick ring, **R**) are the labelled
  rows; silent drones are filled with the class colours mixed by their row of Y, with a small
  bar of Y under them, and an outer ring in their true class. Grey with **?** = no label has
  reached it yet.
- **Rounds:** slider, ◀ ▶, Play (speed in rounds per second), ⏭ for the converged answer. The
  map and the table show Y after that round; a log plot shows the largest change per round
  against `tol`.
- **Brush:** *Class k, reports*, *Class k, silent*, *Wrong report: says k* (the drone keeps its
  true class but reports k), or *Switch reports / silent*. Click a drone to paint it, click
  empty space for a new drone, drag to move, right-click to remove. 2 to 4 classes.
- **Right panel:** accuracy on the silent drones and the wrong reporters (right, wrong, no
  guess) and every row of Y with its guess and truth. Hover a row to find the drone.
- **Matrices tab:** P = D⁻¹A, then the matrix each round multiplies by: **P′** for label
  propagation (P with each reporter's row replaced by an identity row: that is the clamping) or
  **αP** for RWR, plus the restart term **(1−α)Y₀** for RWR, and Y before and after the shown
  round. Hover a cell of Y(t) to see `Σⱼ P′[i][j] · Y(t−1)[j][c]` (plus the restart cell for
  RWR) with the numbers; the cells that go into it light up.
- **Presets:** two sides joined by a bridge, path of 5, **one wrong report** (a west drone
  reports the east class between three honest reporters: LP keeps the lie, RWR at α = 0.85
  overrules it; with only one honest reporter nearby the lie wins under RWR too), **the silent squad** (experiment 4:
  a squad with no reporter takes its neighbour's class), random drones split west/east with a
  chosen share reporting.

### Swarm viewer, Mission 1 view

Pick **M1 First Contact** in the mission menu of the swarm viewer.

- **Left:** the label to propagate (zone, squad, role), the graph (live, ε, kNN), the method
  shown on the map, RWR α, the share of drones that report, and the draw (which drones
  report; fixed by seed and draw, the experiment uses draws 0–19). A button jumps to the first
  test seed (90).
- **Who reports:** *Random draw* (as in the experiment: the report share, the share of them
  that report a wrong class, and the draw number) or *Pick by hand*: click a drone on the map
  to cycle it silent → reports → **wrong report** (says the next class after its true one) →
  silent; every method reruns at once. Wrong reporters get a red ring and R instead of gold,
  and the right panel shows how many lies each method overruled. *Clear all* and *Copy the random draw* to start from.
  The experiment (Run) still uses random draws.
- **Show:** *Reports only* (who got a report through, in the class they reported; silent
  drones dark) or *Propagated labels*. The count of reporting drones is shown below.
- **Map:** each drone filled with its propagated label (class colours mixed by its row of Y;
  grey with **?** = no guess), reporters with a gold double ring and a bold **R**, wrong
  guesses crossed in red,
  optionally a ring in the true class colour. The links are the graph the method ran on.
  Hover a drone for its row of Y.
- **Right:** the list of reporting drones with what each one said (click one to select it on
  the map), the classes nobody reported, every method's accuracy on this seed and draw, and the
  selected drone's guess under each method.
- **Bottom:** *Run the experiment*: the full sweep (test seeds 90–99 × 20 draws, every
  method × graph × label × share, with the current settings and α) on a background thread
  with a progress bar, about 1 minute. The plot's x-axis is either the **share of wrong
  reports** (0 to 40 %, at a chosen report share: the default, it shows the baselines falling
  faster than LP and RWR) or the report share (at a chosen wrong share). With wrong reports the
  table also shows the share of lies each method overruled. It writes `results/m1/` and shows the plot and the table
  below. A test checks that the map's scores are the experiment's scores for the same run.

### Mission 1 results

The same plot and table as the swarm viewer's Mission 1 view, on their own, from
`results/m1/summary.csv`. *Run the experiment* runs the sweep (about 1 minute, default settings).

- **Plot:** accuracy on the silent drones against the share of drones that report (log scale).
  Colour = method, line style = graph (live solid, ε dashed, kNN dotted), with an optional
  ±1 sd band over the 200 runs. Hover a point for accuracy, sd, the share with no guess and the
  number of runs.
- **Left:** the label (zone, squad, role), the graphs and the methods to show.
- **Table:** every method on every graph at one share: accuracy ± sd (no-guess share); the best
  graph per method in bold.

### Swarm viewer, Mission 2 view

Pick *M2 Fingerprints* in the mission list. On the first visit the viewer switches to the plan's
300-drone swarm (30 squads, 70 × 35 km) if the settings are still the defaults; a button brings it
back otherwise. Each seed is embedded on a background thread (about a second in release): DeepWalk
on the full traffic graph, then the three tests of `aerie_classic::m2::run_seed`, exactly as the
runner does.

- **Left:** DeepWalk's settings (walks per drone, walk length, dimension, window, negatives,
  epochs, learning rate; *Plan's settings* resets them), Node2Vec's p and q (same walk and
  skip-gram settings; it runs next to DeepWalk), *Map uses* DeepWalk, Node2Vec or MF (matrix factorisation, run on every seed too), and what the
  map shows: **k-means
  clusters of Z** (each cluster in the colour of the squad most of its drones belong to; ring =
  true squad; cross = the drone's cluster is led by another squad), true squads, **role guessed
  from Z** or **from the traffic counts** (ring = true role, cross = wrong guess), true roles.
  *Hidden links of the link test* draws the 10 % of links the second DeepWalk run didn't see,
  dashed, from red (scored below every never-talked pair) to green (above them all).
- **Map:** the traffic graph, width = log of the messages. Hover a drone: squad, cluster, role,
  both role guesses, messages sent and received, partners.
- **Right:** this seed's role F1, link AUC and squad NMI for DeepWalk and the baselines, how many
  drones sit in a cluster led by their own squad, and for the selected drone its 8 nearest
  vectors by cosine (squad, role; click one to select it).
- **Bottom:** *Run the experiment*: test seeds 90–99 with the current swarm and walk settings,
  DeepWalk, Node2Vec at p = 1 with q = 0.5 and q = 2 (and at the p, q of the sliders),
  seeds in parallel, a few seconds. Bar chart of the three scores per method (whiskers = spread
  over seeds), the table, and the per-seed rows. Writes `results/m2/embeddings.csv`.

### Mission 2 results

The same chart and table on their own, from `results/m2/embeddings.csv`. *Run the experiment*
runs the plan's setup: 300 drones, default walk settings, DeepWalk and Node2Vec at q = 0.5
and q = 2.

### Skip-gram playground

Watch DeepWalk (`random_walks` + skip-gram with negative sampling) learn, one training pair at a
time. The trainer runs the exact loop of `train_skipgram` and calls `pair_loss_and_grads`; when
training ends it re-runs `train_skipgram` and shows ✔ if both give the same Z.

- **Left:** graph presets (two groups + bridge, three groups, ring, path, grid, star + chain,
  the 100-drone swarm's traffic graph), walk settings (walks per drone, length, seed), skip-gram
  settings (dim, window, negatives, epochs, learning rate, seed), and the training buttons:
  *Next step* / *Next pair* (Space), *Whole pair*, *Finish walk*, *Finish epoch*, *Train to the
  end*, *Play* (P) with a pairs-per-second speed.
- **9-step mode** (on by default): Space goes through one pair in 9 steps (pick u, pick v, draw
  the negatives, dot products, σ, errors g, loss, gradients, SGD step); the canvases light up
  as each step is reached, and the **Math** tab on the right shows each step's formula with the
  pair's real numbers (every negative draw, every dot product written out, the vectors before
  and after the step).
- **Graph (middle left):** the current walk (bright inside the window), u in gold, v in green,
  the negatives in red. Click a drone, then another, to add or remove the link; drag to move.
- **Embedding (middle right):** at dim 2 the real vectors, z as discs and c as rings, with lines
  from the origin to z_u and c_v. Above dim 2, with *context vectors c* ticked: Z and C together
  on the two directions that keep the most of their length, origin unmoved (uncentred PCA); the
  view says how much the plane keeps (about 70 % on two groups at dim 16, 54 % on the swarm) and
  the pair's real z_u · c_v next to the on-screen one. Unticked: Z alone, PCA around its mean.
  Arrows show the SGD step, magnified.
- **Bottom:** the pair just trained: the walk with the window, then one row per dot product
  (x = z_u·c, σ(x), target, g = σ − target, loss term) and the step sizes; the loss curve with
  the untrained level (1 + negatives)·ln 2 and the epoch boundaries.
- **Right:** the Math tab; how many drones have a same-group drone as most similar vector (and,
  on the swarm, the same for roles); tabs for the cosine similarity of Z, the co-occurrence counts of the walks
  (what skip-gram is fitted to), and the negative-sampling chances (∝ count vs ∝ count^0.75).
  Drones sorted by group, so groups show up as blocks.

### Node2Vec playground

See how p and q bend Node2Vec's random walks. The walks follow `biased_walks`' rule step for
step (same neighbour order, one random number per step): with p = q = 1 they are exactly
`random_walks`' walks (a test checks it on every preset).

- **Left:** graph presets (two groups + bridge, barbell, three wheels labelled by group or by
  role, ring, grid, the 100-drone swarm's traffic labelled by squad or by role, empty) and
  tools to build your own: 🔗 Link (click two drones: add / remove the link), ➕ Add (click
  empty space), 🖌 Label (paint with the brush label), ✖ Delete; drag moves a drone in any
  tool, right-click starts the walk there. p and q sliders (log scale, 0.1 to 10) with shortcuts,
  and a 3-drone sketch of the rule with the current percentages. *Build one walk*: with
  **Explain each step in 10 small steps** on (default), Space goes through one walk step in 10:
  where the walk is (t, v), v's neighbours, where each lies from t (d = 0, 1, 2), the bias α,
  π = weight × α, the total Z, the chances π / Z, the random number u, where u falls, the move.
  Off: Space takes a whole step. *Whole step*, *Walk to the end*, *New walk*, *Play*. Before the
  draw, moving p or q updates the step under way.
  *Many walks* settings (walks per drone, length, window, seed) and *Embedding* settings.
- **Centre:** the graph, drones in their label colour. The walk so far (numbered steps), the
  current drone (gold) and t, the one before it (violet). With the options shown, an arrow to
  every neighbour as wide as its chance: red = back to t (× 1/p), amber = also next to t (× 1),
  blue = away from t (× 1/q). Optional halos: where walks from the start drone go.
- **Right, Math:** every small step reached so far, as a formula with this walk's numbers
  (the current one framed in gold, scrolled into view): d(t, x) for each neighbour, α, each π,
  Z written out, each chance next to DeepWalk's, u, the subtractions until u goes below 0, and
  the move. **This step:** the options table (link weight, where it lies, bias, weight × bias, the
  Node2Vec chance and DeepWalk's), the chances as one bar, and for the step just taken where the
  random number fell. **Many walks:** Node2Vec against DeepWalk on the same seed: share of steps
  back / near / away, steps that change label, training pairs (window) with the same label,
  different drones per walk, hops from the start; the hops-after-k-steps curve; a q sweep (1/8
  to 8) of the same-label share and the distance. **Embedding:** skip-gram on those walks
  (Node2Vec and DeepWalk), how many drones have a same-label most similar vector, and both
  pictures (PCA above dim 2).
- **Bottom:** the walk being built and 4 walks from the start drone for each method, as chips
  in label colours; the arrow before each chip is the kind of step (↩ = back).

### Matrix factorisation playground

Watch Z Zᵀ learn to rebuild A. The trainer runs `matrix_factorization` epoch by epoch (same
start, `mf_gradient`, `mf_loss`); at the last epoch it re-runs it and shows ✔ if both give the
same Z.

- **Left:** the same presets and tools as the Node2Vec playground, plus ✂ **Hide** (click two
  linked drones: the link is 0 in A during training but kept for the link test). What A holds
  (1 per link, ln(1 + weight), the weight), the settings (dimension, learning rate, l2, epochs,
  seed: any change starts over), and training: with **Explain each epoch in 7 small steps** on,
  Space follows one drone u (right-click to pick it) through one epoch; *Whole epoch*, *+10*,
  *+100*, *Train to the end*, *Play*, *Reset*. A too-big learning rate is caught ("the loss
  blew up").
- **Centre left:** the graph. Small step 2 lights u's links with their A values; from step 3 a
  line to every other drone, then coloured by its error (green: z_u · z_v too small, pulled
  closer; red: too big, pushed away), as wide as the error.
- **Centre right:** the vectors (dim 2: themselves; above: their two main directions) with the
  origin, u's pull / push arrows and the step (magnified), and where u was before the step.
- **Right:** **Math** (u's row of A, every z_u · z_v, every e_uv, every 4 e_uv z_v, the gradient
  written out, the step and the loss before and after; largest errors first), **Matrices** (A,
  Z Zᵀ and E = Z Zᵀ − A as heatmaps, drones sorted by label), **Hidden links** (each hidden
  link's z_u · z_v, the share of never-linked pairs it beats, and the AUC).
- **Bottom:** the loss per epoch, log scale.

### Jamming playground

Place jammers and a ridge, and see the jamming power everywhere. Two tabs at the top of
the left panel: **Seed swarm** (a spawned swarm, everything below) and **Build your own**. The heatmap calls the
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
- **Link rule (2e):** *signal vs jamming (D03)* uses `live_graph`: a direction works when the
  sender's signal is at least N0 + jamming at the receiver, and a link needs both directions.
  *Either end jammed (D02)* is the old `fail_links` rule, kept for comparison. Sliders for the
  noise floor N0 and each role's signal scale (smaller = louder transmitter = resists jamming).

  **N0 works backwards from what you might expect: lower N0 = fewer links under jamming.**
  Transmit power is tied to N0, `tx = N0 · (1 + (range / scale)²)`, so that each drone still
  reaches its range on a quiet day. The margin then splits in two:

  ```
  margin = (1 + (r/s)²) / (1 + (d/s)²)  ×  N0 / (N0 + jam)
           └──── geometry only ────┘      └ jamming penalty ┘
  ```

  With no jammer the penalty is 1 and N0 cancels out: same graph for any N0. With a jammer,
  a small N0 means quiet drones that the jammer drowns easily (N0 = 0.05, jam = 0.1: margins
  fall to a third), a large N0 means loud drones (N0 = 0.2, jam = 0.1: they lose a third).
  So N0 sets how loud the drones are against the jammer.

  | Link style | Meaning |
  |---|---|
  | solid, orange to green | live link, coloured by margin (orange ≈ 1, about to break; green ≥ 10) |
  | dashed, teal, with an arrow | one-way: only sender → receiver works (the receiver is jammed) |
  | dashed, red | dead: neither direction works |

  Hover a drone for its live, geometric and one-way links, transmit power and scale.
- **Hover a link** (either tab): it is highlighted, and the tooltip shows the distance, the
  Gaussian weight, its state (live with its margin, one-way, dead), and for each direction
  the signal, N0 + jamming at the receiver, and the margin (✔ when ≥ 1).
- **Build your own tab:** place drones by hand with the *Add drone* tool (role and squad
  picked in the left panel), drag them, right-click to delete. The list shows each drone's
  role, squad and position, all editable. Each drone is labelled `#i S<squad>` on the map.
  Links come from `epsilon_graph` (no-jammer ranges) then `live_graph`, and *Write both
  margins* puts `a→b` and `b→a` on every link (✘ below 1). Jammers, the ridge and every
  slider work as in the other tab. **One-way arrows** (sender → receiver, D03 rule): every
  ordered pair is checked, and each direction that works while the reverse does not gets a
  solid arrow. Teal = a geometric link cut one way by jamming. Violet = not a geometric link,
  because the ranges differ (a relay heard by a recon drone beyond 2.5 km). Both come from
  `live_graph`, which keeps them in two lists: `cut_one_way` and `range_one_way` (D03, option b). Presets: *Paper check* (jammer at (10, 10); recon at 14
  and 15 km, relay at 17 km on y = 10), *Squad at the zone edge*, *Relay chain*, *Clear drones*.
- **Right panel:** power against distance (open ground and behind the ridge), the value table,
  the power at the cursor per jammer (clear or blocked), the effect on the swarm before
  and after jamming from `fail_links` (links, mean degree, components, isolated drones), and readings against
  truth: counts per colour, the measured noise (mean, sd), and a scatter plot of truth (x)
  against reading (y) for every drone. Hover a dot to ring that drone on the map.
  *How far each role is heard*: range against the receiver's distance to the jammer, one curve
  per role, and a table at 20, 12, 4 and 0 km with each role's transmit power. Use it to
  calibrate N0 and the scales.

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
