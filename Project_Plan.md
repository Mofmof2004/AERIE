# Project AERIE: a graph-learning brain for a drone swarm

Nov 12, 2026 · @Theo

AERIE is the learned brain of a simulated reconnaissance swarm. You build it one lecture at a time, and every slide concept becomes a working module with an experiment behind it. An aerie is an eagle's nest, and Aquila is the eagle.

## The pitch

**The scenario.** 100 drones hold watch over a contested valley 40 km wide. Each drone is a node with features. Radio links are edges. Sensor readings are graph signals. The swarm has to keep eyes on every sector, keep the network alive, and pass what it sees to a human operator. An opponent jams it, spoofs it and shoots it down the whole time. Tasks pop up ("observe sector 4", "relay for squad B") and the swarm votes on who takes them, using only what each drone hears from its neighbours.

The voting layer follows your description of the startup idea. The public [Aquila site](https://aquila-systems.fr/) only presents FleetOS as an operating system for autonomous fleets, with no technical detail to build on.

**Scope.** The swarm senses, relays, computes and survives. Strike drones and target assignment are out of scope, and any response to what the swarm finds stays with a human operator.

**The one question the whole project answers.** How much can a swarm figure out when each drone only talks to its neighbours, every transmission can be heard, and a third of the swarm may be gone by the end? This is why graph ML fits swarms so well. A graph filter of order K is exactly K rounds of neighbour-to-neighbour exchange. Your lecture 4 slides say it in three words: "Distributable: local exchanges". Every model in this course is a recipe for what drones should say to each other and what to do with what they hear.

**What you walk away with**

- **A repo** with a simulator, 11 from-scratch modules, and about 50 experiments with fixed seeds.
- **A replay demo.** One 5-minute flight where every module fires: the jamming map fills in, a spoofed drone gets caught, the network split gets predicted, a task spawns, drones bid, the winner explains itself.
- **A decision log.** One entry per mission: what went into the brain, what got rejected, and the plot that decided it.
- **A flight report.** 6 to 8 pages, paper style, the thing you show Aquila or put on your site.
- **11 exam sheets.** One page per lecture with the derivations done by hand next to your own figures.

&#91;embedded content: mission map · 4 acts, 11 missions\]

Read it bottom up. The simulator feeds every mission, and the four acts add up to a swarm that allocates its own tasks.

## How every mission works

Each mission is one lecture, one swarm problem, and one decision about the brain. All eleven follow the same five beats.

1. **Brief.** A problem the swarm has right now, stated in one line.
2. **Build.** Implement the method from scratch in Rust. Plain loops and faer for the classical methods, Burn tensors with autodiff for anything trained by gradient descent. Rust has no PyTorch Geometric to compare against, so correctness comes from tests.
3. **Experiments.** Three to six runs. Write the hypothesis in one sentence before you run. One figure per experiment.
4. **Decision.** One entry in `DECISIONS.md`: what goes into AERIE, what got rejected, which plot decided it.
5. **Exam drill.** The derivations by hand, plus the questions the slides themselves ask.

**Three rules**

- **Scratch first, tests second.** Every layer is yours. A module counts as done when its tests pass: a closed form, a finite-difference gradient check, or a tiny graph worked by hand.
- **Hypothesis before run.** Being wrong on paper first is where the learning is. Keep the wrong guesses in the log.
- **Core, then stretch.** Every mission has a core you can finish in one or two sittings. Stretch items are marked and optional.

**One warning.** Your lecture 1 slides say the written exam is 40% of the grade. Code will not write derivations for you. The exam drill is the part that pays on exam day, so do not skip it when the experiments get fun.

## Setup and conventions

These rules apply to every mission. Fix them once in week 1 so results from different missions can be compared.

**Stack**

| Need | Use |
| --- | --- |
| Language | Rust, stable toolchain, one Cargo workspace |
| Dense linear algebra, eigendecomposition, SVD, linear solves | [faer](https://docs.rs/faer/latest/faer/) |
| Graph storage | Your own edge list and CSR structs |
| Autodiff and neural layers | [Burn](https://docs.rs/crate/burn/latest) with the Autodiff backend |
| Downstream classifiers and clustering | [linfa](https://docs.rs/linfa-logistic) (linfa-logistic, linfa-clustering) |
| Assignment oracle | [pathfinding](https://docs.rs/pathfinding/4.14.0/src/pathfinding/lib.rs.html), function `kuhn_munkres_min` |
| Convex solver, as an optional check on trend filtering | [clarabel](https://docs.rs/crate/clarabel/latest) |
| Randomness | rand with an explicitly seeded generator |
| Configs and results | serde with JSON |
| Plots and GIFs | [plotters](https://docs.rs/plotters-bitmap/0.3.7/src/plotters_bitmap/bitmap.rs.html) |
| Parallel loops for samplers and Monte Carlo runs | rayon |
| Graph partitioning in M9 | Spatial k-means from linfa-clustering |
| Compute | Laptop for everything. DelftBlue only for the largest runs in M9 |

**Working in Rust**

- **Two layers of code.** The classical methods (M1, M2, M3, M6, the VAR models in M7, the samplers in M9) are plain Rust over edge lists plus faer. Only models trained by backpropagation use Burn.
- **Message passing without sparse tensors.** I found no sparse tensor type in Burn's tensor documentation. Do what PyTorch Geometric does internally: `select` the source-node rows by edge index, weight them, and `scatter` them onto the target nodes. Burn documents `select` as index-select and `scatter` as scatter-add. At 100 nodes a dense matrix product is a good cross-check.
- **Pin Burn.** It moves fast. docs.rs listed version 0.22 when I checked, and 0.21 replaced the ndarray CPU backend with a new one. Pin the version in `Cargo.toml` and read the Burn Book for that version.
- **Hand-written gradients where they are short.** Skip-gram in M2 and the least squares fits in M4 need no autodiff. Derive the gradient on paper and code it. That is exam practice for free.
- **Three kinds of test replace the library reference.** A closed form (iterative solver against the direct solve). A finite-difference gradient check on a 5-node graph. The small hand-worked example from each exam drill, as a unit test.
- **Your own metrics module.** AUC, macro-F1, NMI and NMSE are about 20 lines each. Write them once, with tests.
- **The oracle.** `kuhn_munkres_min` needs weights that implement `Ord`, so scale costs to integers. It panics when rows outnumber columns, so put tasks on rows and drones on columns.
- **Eigendecomposition.** faer's `self_adjoint_eigen` returns real eigenvalues sorted from small to large. The general `eigen` always returns complex values.
- **Plots.** Write every result to JSON first, then plot from the file. plotters makes PNGs, and animated GIFs through `BitMapBackend::gif`.
- **Seeds.** Pass a seeded generator explicitly into every function that draws random numbers. Never use the thread-local one.
- **Always build with `--release`.** Numeric loops in debug builds are far too slow to iterate on.
- **Peak memory for M9.** Wrap the global allocator with a counter, or read peak resident memory from the OS.

**Notation, matching the slides**

| Symbol | Meaning |
| --- | --- |
| N, M | Number of drones (nodes) and links (edges) |
| A, D, L | Adjacency, degree matrix, Laplacian L = D - A |
| B | Incidence matrix, with L = B B^T |
| S | Graph shift operator: A, L or a normalized version. State which one in every experiment |
| x, X | Graph signal with one value per drone, and feature matrix of size N by F |
| K | Filter order, which equals rounds of neighbour talk |
| F | Number of features per layer |
| h\_k, H | Filter coefficients, and parameter matrices of a GNN layer |
| λ, V | Eigenvalues (graph frequencies) and eigenvectors of S |

**Evaluation protocol**

- **Split by scenario, never by node.** Each SwarmSim seed is a different swarm layout and jammer placement. Train on seeds 0 to 79, validate on 80 to 89, test on 90 to 99. Every learned model is then tested on swarms it has never seen.
- **Methods with no training** (M1, M3, parts of M2 and M6) run on single snapshots. Report the mean over the 10 test seeds.
- **Five training seeds** for anything learned. Report mean and standard deviation.
- **Always run the no-graph baseline.** The same model with S = I is an MLP on node features. If the graph does not beat it, the task is too easy or the graph is wrong. Both are findings worth a log entry.
- **One metric per task type.** Accuracy or macro-F1 for node classes. AUC for link prediction and for attacks. Normalized error for signals, NMSE = ||x\_hat - x||^2 / ||x||^2. Regret for allocation, defined in Mission 5.

**Experiment card.** Every experiment folder gets a README in this form.

```
Question:    one line
Hypothesis:  written before the run
Setup:       data, model, which S, K, seeds
Result:      the number and the figure
Surprise:    what you did not expect
Slide:       lecture and slide number this illustrates
```

**Exam sheet template.** One page per lecture in `exam/`.

1. Five definitions, one line each.
2. The key equations, written from memory and then checked.
3. One full derivation by hand.
4. The slide's own questions with your answers.
5. Your best figure with a two-line reading of it.
6. One trap: a mistake you made in this mission.

## Mission 0: SwarmSim

SwarmSim is a small Rust crate that generates every dataset in the project, so you control the ground truth and can plant whatever you want to detect later. Keep plotting code out of it, in aerie-plot.

**The swarm**

| Piece | Spec |
| --- | --- |
| Drones | 100 by default. Recon 50%, RF sensing 15%, relays 20%, compute 15%. |
| Recon | Cameras. Detects contacts with a noisy confidence score. |
| RF sensing | Measures radio noise level at its position. |
| Relay | Long radio range. Holds the network together. |
| Compute | Fuses imagery from recon drones. |
| Node features | Position, velocity, battery, sensor quality, compute capacity, radio range, role one-hot. |
| Motion | Squads of about 10 patrol waypoints with simple cohesion and separation. An on-station mode freezes the formation for static-graph missions. |
| Comm graph | Three builders from lecture 1: epsilon-ball on radio range, k-nearest neighbours, and a correlation network from sensor time series. |
| Directed graph | Data flow: recon sends to relays, relays send to compute nodes. Used whenever a lecture covers directed graphs. |

**The opponent**

| Feature | What it does in SwarmSim |
| --- | --- |
| Jammers | Emitters whose power falls off smoothly with distance and is cut sharply by ridges. Links inside strong jamming fail. Jammers can move. Jamming power is the main graph signal of the project. |
| Spoofing | One drone, or a cluster of neighbours, reports false readings. |
| Attrition | Drones removed at random, by region, or most-connected first. Replacements can join. |
| Contacts | Unknown objects crossing sectors. They create "hold a track" tasks. |
| Emission cost | Every round of radio traffic raises a detectability counter. |

**Tasks the swarm votes on:** observe a sector, hold a track on a contact, relay for a cut-off squad, fuse imagery. Each task has requirements (sensor, compute, range). An oracle solves the assignment with the Hungarian algorithm and gives you labels for "who should take it".

**Outputs.** A snapshot dataset (one graph plus signals), a sequence dataset (T steps) and an event log, all written with serde. Graphs are stored as weighted edge lists, the format every later mission reads.

**Starting values.** These are numbers to begin with and tune, not measured facts about real drones.

| Parameter | Start with |
| --- | --- |
| Arena | 40 km by 20 km |
| Squads | 10 squads of 10, each with about 5 recon, 2 relays, and 3 split between RF sensing and compute |
| Radio range | 5 km for recon, RF sensing and compute. 9 km for relays |
| Link weight | exp(-d^2 / (2σ^2)) with σ at half the radio range. Zero beyond range |
| Target mean degree | 8 to 10, with one connected component before any jamming |
| Jammer power at distance d | P0 / (1 + (d / d0)^2) with d0 = 4 km. Multiplied by 0.1 behind a ridge |
| Link failure | A link fails when jamming power at either end is above 0.5 P0. In open ground that gives a dead zone of radius d0 around the jammer |
| Sensor noise | Gaussian, standard deviation 10% of P0 |
| Spoofed reading | True reading plus a constant bias of 30% of P0 |
| Flight length | 200 steps, drone speed 0.2 km per step |
| Attrition | Each drone lost with probability 0.2% per step, which removes about a third of the swarm over one flight |

**Decide before you code.** Log these as entry D00 in `DECISIONS.md`. They are cheap to settle now and expensive to change in Mission 5.

- **Link rule.** Two drones are linked when their distance is at most min(range\_i, range\_j), because a working link has to go both ways. Relays then reach each other at up to 9 km and form the backbone between squads.
- **kNN symmetry.** A kNN graph is directed. Take the union: i and j are linked when either one picks the other. The mutual version can leave drones isolated.
- **Squad spread.** A tight squad of 10 is almost a complete graph, so a mean degree of 8 to 10 can come from inside the squads alone. Also require one connected component before jamming, and aim for roughly a quarter of all links to cross between squads. The quarter is a starting target to tune. Expect squad labels to agree across almost every link.
- **IDs.** Wrap drone and squad IDs in their own types, `DroneId` and `SquadId`, so an ID can never be used as a vector index by accident. This matters once attrition starts.
- **Determinism.** Use `ChaCha8Rng` with an explicit seed and pass it as an argument. No `HashMap` or `HashSet` where iteration order reaches the output. No rayon inside the simulator.
- **Sorting floats.** `f64` is not `Ord`. Sort distances with `total_cmp`.

**The task model**

- A task is a location, a type and a list of requirements.
- The cost of drone i taking task j is c\_ij = travel time + β (1 - battery\_i), plus a large but finite penalty when the drone lacks a required capability.
- **Plant a rule that depends on the graph.** A drone qualifies for an observe or track task only if a relay sits within two hops of it on the live comm graph. Otherwise its data cannot get out. Without a rule like this, node features alone solve the task and no GNN will ever beat the MLP baseline. Mission 10 reuses this rule as ground truth.
- Store two labels per task. The single-task winner is the cheapest qualified drone for that task alone. The joint winner comes from the Hungarian algorithm over all open tasks at once. A drone cannot work out the joint winner from its view of one task, so Mission 5 trains on the first label and measures regret against the second.

**Datasets to generate**

| Dataset | Contents | Used in |
| --- | --- | --- |
| Snapshots | 100 seeds, 20 snapshots each. Graph, features, true field, noisy readings, labels | M1 to M6, M10, M11 |
| Tasks | 50 tasks per snapshot with oracle assignments | M4, M5, M10, M11 |
| On-station sequences | 100 seeds, 200 steps each, formation frozen | M7 |
| Flights | 100 seeds, 200 steps each, with motion, link loss, attrition and the event log | M8 |
| Large swarms | Single snapshots at 1k, 10k and 100k drones, arena scaled to keep the same density | M9 |

**Build order.** Only steps 1 to 3 block the first missions. Build the rest in the week that needs it.

| Step | What | Needed by | When |
| --- | --- | --- | --- |
| 1 | Static swarm, epsilon and kNN graphs, degree and component stats, a plot | M1, M3, M4 | Week 1 |
| 2 | Jamming field with one ridge, readings, link failure, zone labels | M1, M3 | Week 1 |
| 3 | Tasks, the relay rule and both oracle labels | M4, M5 | Week 1 |
| 4 | On-station sequences and the correlation graph | M1 experiment 1, M7 | Week 2 |
| 5 | Static attrition (drop a share of drones from a snapshot) and the emission counter | M4 experiments 1 and 5, M5 experiment 3 | Week 3 |
| 6 | Spoofing | M6 | Week 5 |
| 7 | Motion, flights, the event log, attrition over time | M8 | After the backbone |
| 8 | Large swarms with a grid-based neighbour search | M9 | After the backbone |

The correlation graph needs a time series, so it cannot be built in step 1. On-station sequences are cheap and need no motion: frozen drones, a slowly moving jammer and fresh noise each step.

**Watch out**

- A mean degree under 4 fragments the graph and label propagation has nothing to travel over. Over 20 and every GNN oversmooths at once.
- Store the true field and the noisy readings separately. You need the truth for every error metric.
- Keep squad, role and zone labels for every drone at every step, including drones that cannot report.
- Give every drone a stable ID that survives attrition. Mission 8 depends on it.

**Done when**

- [ ] A GIF shows the swarm on station with a jammed zone opening a hole in the comm graph.
- [ ] You can print how often neighbours share each label: squad, role, inside a jammed zone. You will need this number in Mission 1.
- [ ] Same seed gives the same dataset, byte for byte.

**Lecture 1 concepts you already hit here:** graph notation (A, D, weighted, directed), the three graph construction methods, and the three task levels (node, link, graph).

## Act I: Fundamentals (lectures 1 to 3)

Act I gets the swarm useful answers with zero neural networks. It also sets the baselines every later model has to beat.

### Mission 1: First Contact (lecture 1, label propagation)

**Brief.** Jamming has silenced most of the swarm. Only 10% of drones get a status report through. Work out the state of the other 90% with no training at all.

**Setup**

- **Task type.** Semi-supervised node classification, transductive, no trained parameters.
- **Data.** Single snapshots. Labels: inside a jammed zone (2 classes), squad (10 classes), role (4 classes).
- **Metric.** Accuracy on the unlabelled drones, averaged over the 10 test seeds and 20 random draws of which drones report.
- **Baselines.** Majority class, and the label of the nearest reporting drone by distance. The second is the nearest-neighbour approach from the slides.
- **Start with.** Tolerance 1e-6, at most 1000 iterations, α = 0.85 for RWR.

**Build**

- Transition matrix P = D^-1 A from the comm graph.
- Label propagation: Y ← P Y, then clamp the known rows, repeat until nothing moves.
- The closed form to check against: Y\_U = (I - P\_UU)^-1 P\_UL Y\_L.
- Random walk with restart: F ← α P F + (1 - α) Y0, and its closed form (1 - α)(I - αP)^-1 Y0.

**Experiments**

1. **Which graph?** Accuracy against label fraction (1% to 50%) on the epsilon, kNN and correlation graphs.
2. **Where it breaks.** Propagate "inside a jammed zone" and "squad" (neighbours agree), then "role" (neighbours disagree). Predict the accuracy of each before running.
3. **Biased reports.** Compare a random 10% of reporting drones with the realistic case, where only drones at the edge of a jammed zone can report. Measure how much accuracy drops when the labelled nodes all sit on one border.
4. **The silent squad.** One squad is fully cut off with no labelled drone inside. Find what label propagation and RWR each say about it. Sweep α and count iterations to converge.

**Watch out**

- Use the row-normalized P = D^-1 A. Each row must sum to 1, so each drone's label vector stays a distribution.
- An isolated drone has degree zero. Decide what to do with it before it becomes a division by zero.
- Clamp the known labels after every iteration. Without clamping, every drone converges to the same vector.

**Decision.** The default graph builder for AERIE, and which labels can be trusted when the reports are not a random sample.

**Exam drill**

- Run label propagation by hand on a 4-node graph for two iterations.
- Show why RWR converges: (αP)^t goes to 0 when 0 ≤ α < 1.
- Answer the slide question: is label propagation inductive or transductive? Test it. Add a drone and see what you have to redo.
- Place every task you just ran on the two axes: supervised, semi-supervised or unsupervised, and transductive or inductive.

### Mission 2: Fingerprints (lecture 2, node embeddings)

**Brief.** Switch sides for one mission. A listener records who transmits to whom and never sees the content. Learn a vector for each drone from that traffic log alone, and find out what it reveals about your own swarm.

**Setup**

- **Task type.** Unsupervised and transductive. Roles and squads are used only to score the embeddings afterwards.
- **Data.** The traffic graph of one flight: an edge between two drones weighted by how many messages they exchanged. The directed version keeps who sent to whom. Use a 300-drone swarm here, because 100 nodes give noisy results.
- **Metrics.** Macro-F1 for role with 5-fold logistic regression. AUC for link prediction. Normalized mutual information for clustering against squads.
- **Link prediction protocol.** Remove 10% of the edges before training the embeddings. Sample the same number of non-edges as negatives. Score a pair by the dot product of its embeddings.
- **Start with.** Dimension 32, walk length 20, 10 walks per node, window 5, 5 negatives per positive, plain SGD with learning rate 0.025.
- **HOPE.** Pick β smaller than 1 over the largest eigenvalue of A, or the Katz series does not converge.

**Build**

- Encoder as a lookup table Z, decoder as a dot product.
- Random walk generator, then skip-gram with negative sampling, with the gradients derived and coded by hand. That is DeepWalk.
- Node2Vec: biased walks with return parameter p and in-out parameter q.
- On the directed data-flow graph: HOPE with Katz similarity (I - βA)^-1 βA factorized by SVD into source and target embeddings. NERD-style alternating walks as stretch.
- Matrix factorization of A as the simplest baseline.

**Experiments**

1. **Squads or roles?** Node2Vec with q below 1 and q above 1. The paper says one setting finds communities and the other finds structural roles. Check which one separates squads and which one separates relays from recon drones.
2. **What the listener gets.** Four downstream tasks on the embeddings. Role classification with logistic regression: can the listener pick out relays and compute nodes? Link prediction on 10% hidden links. K-means clustering against true squads. Formation classification from mean-pooled embeddings.
3. **Follow the data.** DeepWalk against HOPE at predicting which way data flows on an edge. Source and target embeddings show where data ends up.
4. **The wall.** A new drone joins. Try to embed it. Count parameters as the swarm grows from 100 to 10,000.
5. Stretch, **cover traffic.** Add dummy transmissions. Plot the listener's accuracy against the extra emission cost.

**Watch out**

- Hiding test edges after training the embeddings leaks the answer. Hide them first.
- Say how you sample negatives: uniformly, or by degree to the power 0.75 as Word2Vec does.
- Embeddings trained on two different graphs live in two different coordinate systems. Think about what that does to the formation task in experiment 2 before you run it.
- Node2Vec with p = q = 1 must reproduce your DeepWalk numbers. Use that as the unit test.

**Decision.** Shallow embeddings become a diagnostic tool, and experiment 4 is the written reason AERIE needs GNNs. Experiment 2 is your first measurement of what the swarm leaks, which Mission 11 picks up again.

**Exam drill**

- Write the random walk objective. Explain why the softmax denominator is the problem and how negative sampling fixes it.
- State what p and q do with a 3-node sketch.
- Name ENC and DEC for DeepWalk, for matrix factorization and for HOPE.
- List the three shortcomings of shallow encoders from the slides and point to your experiment for each.

### Mission 3: Clean Signal (lecture 3, regularization)

**Brief.** Only the RF sensing drones can measure jamming power, their readings are noisy, and a third of them are blinded. Rebuild the jamming map so every drone knows where it can still talk.

**Setup**

- **Task type.** Signal reconstruction on a graph, transductive. Nothing is trained. Only γ is chosen.
- **Data.** Single snapshots. The true signal x is jamming power at every drone. The observation y is the noisy reading at RF sensing drones only, and the mask M marks who has a working sensor.
- **Graph.** Use the distance-based graph from drone positions, not the live comm graph. Jamming removes comm links exactly where the field is strongest, which is where you need the graph most.
- **Metric.** NMSE over all drones, and separately over drones with no sensor.
- **Choosing γ.** 20 values on a log grid from 0.001 to 100. Hold out 20% of the sensors and pick the γ that predicts them best.
- **For experiment 1.** With H = (I + γL)^-1 the bias is (H - I) x and the variance is σ^2 trace(H H^T). Those are the two theory curves to overlay.

**Build**

- Incidence matrix B, Laplacian L = B B^T, and the variation measure x^T L x.
- Tikhonov denoising, x\_hat = (I + γL)^-1 y. With a sensor mask M it becomes (M^T M + γL)^-1 M^T M y.
- Trend filtering, which swaps the penalty for the 1-norm of B^T x. It has no closed form, so solve it iteratively (your own ADMM, with clarabel as an optional check on a small graph).
- Stretch: graph elastic net and the Sobolev regularizer.

**Experiments**

1. **Bias and variance.** 500 noise draws for each γ. Plot bias squared, variance and total error. Overlay the theoretical curves and see if they match.
2. **Open ground against ridge shadow.** Tikhonov and trend filtering on the smooth falloff, then on the sharp edge a ridge cuts into the field. Look at what Tikhonov does to the edge.
3. **Few sensors by design.** Only 15% of drones carry the sensor, so interpolation to the other 85% is the normal case. Plot error against sensor share from 5% to 50%.
4. **Wrong graph.** Rebuild the map with the true comm graph, then a random graph with the same number of edges.

**Watch out**

- With a mask, the matrix M^T M + γL cannot be inverted if some connected component has no sensor at all. Check the components first.
- Tikhonov with x^T L x needs an undirected graph. The slides say so. Symmetrize before you start.
- Normalize the edge weights. Otherwise the same γ means something different on every graph.
- An interior-point solver such as clarabel is instant at 100 nodes and slow at 10,000. Your ADMM scales further.

**Decision.** The default map builder per terrain type, how AERIE picks γ, and how many RF sensing drones the swarm needs for a usable map.

**Exam drill**

- Derive the Tikhonov closed form by setting the gradient to zero.
- Derive the bias and the variance of x\_hat = H y for noise with variance σ^2.
- Explain in two lines why the 1-norm keeps sharp edges and the 2-norm does not.
- Recite the design guidelines for the fitting term and the regularizer, including why convexity matters.

## Act II: GNNs (lectures 4 to 6)

Act II builds the brain itself: the network that turns K rounds of neighbour talk into a bid for a task. Radio rounds have a price, and the network has to keep bidding while it loses nodes. Mission 5 is the centre of the whole project.

### Mission 4: Gossip (lecture 4, graph filters and GCNNs)

**Brief.** A drone can only talk to its neighbours, and every transmission can be heard. K rounds of neighbour talk is an emission budget. Find the smallest K that still does the job.

**Setup**

- **Shift operator.** Use the normalized adjacency S = D^-1/2 A D^-1/2. Its eigenvalues stay between -1 and 1, so S^K x does not blow up.
- **Fitting a filter.** Stack x, Sx, ..., S^K x as the columns of a matrix and solve least squares for h, with a small ridge penalty.
- **Jammer localization, the main task.** Input is the noisy jam reading at each drone. Every drone outputs one score, with a softmax over drones. The target is the drone nearest the jammer. Count a hit when the top-scoring drone is in the right squad. Train on seeds 0 to 79.
- **Collaborative filtering data.** A matrix of 100 drones by 20 task categories holding past performance scores, with 80% of entries hidden. The graph links each drone to its 10 most similar drones by correlation of observed scores. Metric is RMSE on the hidden entries.
- **Emission accounting.** One round costs 2|E| messages, one in each direction per link. A model with two layers of order K = 3 uses 6 rounds.
- **Start with.** K = 3, F = 16, 2 layers, ReLU, Adam with learning rate 0.001, 200 epochs.

**Build**

- **Gossip protocol.** Give each drone object its own value and a `send to neighbours` method. Run K rounds and accumulate h\_0 x + h\_1 Sx + ... + h\_K S^K x locally. Check it equals the matrix polynomial. Count messages and confirm O(K|E|).
- **Learned filter.** Fit the coefficients h by least squares to clean the jamming map.
- **First allocator, collaborative filtering.** A drone by task-type performance matrix with most entries missing. Build a drone similarity graph. Predict the missing scores with an order-K filter, closed-form least squares with a penalty on h. Highest predicted score takes the task.
- **Graph perceptron, then a GCNN with filter banks**, as a Burn `Module` built from `select` and `scatter` over the edge list.

**Experiments**

1. **Emission budget.** Accuracy against K from 0 to 8 on jammer localization (which squad is nearest the jammer, given only jam power readings). Plot accuracy against the detectability counter and pick the knee.
2. **Linear to deep.** Filter, perceptron and 2-layer GCNN on the same task.
3. **Shuffle test.** Permute the drone IDs. Check the output permutes the same way.
4. **Links fail.** Drop 10% to 40% of links at test time. Measure how far the filter output moves.
5. **Transfer.** Train on the full 100-drone swarm. Test on the same swarm after 30% attrition, then on a different 150-drone swarm. No retraining in either case.
6. **Parameter count.** GCNN against an MLP on the stacked signal as N grows.

**Watch out**

- Check the gossip version against the matrix version to 1e-10 before you train anything.
- An unnormalized A makes S^K x explode with K. If the loss turns into NaN, look here first.
- Keep the output per drone. A fully connected layer over all nodes at the end breaks permutation equivariance and makes transfer to another swarm impossible.
- The filter is linear in h, so the least squares fit and gradient descent must land on the same coefficients. Use that as a test.

**Decision.** K and the number of features F for the brain, under a stated emission budget.

**Exam drill**

- Answer the slide question: are graph filters inductive or transductive, and why? Experiment 5 is your evidence.
- Answer the slide question: what are ENC and DEC when the model is a graph filter?
- Show why computing S^K x costs O(K|E|) and not O(N^3).
- Write the collaborative filtering objective and say why it has a closed-form solution.
- Explain a CNN as a GCNN on a grid graph.

### Mission 5: The Bid (lecture 5, GNN architectures)

**Brief.** A sector goes dark or a contact appears. There is no command node to hand out the job, because a command node is the first thing a swarm loses. Each drone computes a bid from its K-hop view and the highest bid takes it. Train the bids to imitate the Hungarian oracle from SwarmSim.

**Setup**

- **One task at a time.** The task announcement floods through the swarm. Each drone builds its input from its own state plus task features: distance to the task, capability match, battery.
- **Output and loss.** One score per drone, softmax over drones, cross-entropy against the single-task winner from Mission 0.
- **Settling the vote.** Use max-consensus. Each drone keeps the highest (bid, drone ID) pair it has heard and rebroadcasts it. After as many rounds as the graph diameter, every drone agrees on the winner. Add those rounds to the emission count.
- **Several tasks at once.** Assign them one by one and mask drones that are already taken.
- **Regret.** Cost of the swarm's assignment minus the oracle's cost, divided by the oracle's cost. Zero is perfect. The oracle here is the joint Hungarian assignment over all open tasks.
- **Graph-level label for pooling.** "Coverable" is true when every open task has at least one qualified drone.
- **Start with.** 2 layers, 32 hidden features, Adam with learning rate 0.001. GAT with 4 heads. APPNP with α = 0.1 and 10 propagation steps.

**Build**

- **GCN** with the normalized shift operator and self-loops.
- **SGC**: propagate S^K X once, then logistic regression.
- **APPNP**: predict first, then propagate with teleport α. Note the link to RWR from Mission 1.
- **GAT**: attention score from LeakyReLU(a^T \[W h\_i || W h\_j\]), softmax over neighbours, several heads.
- **MPNN** with edge features (link quality, distance): message, aggregate, update.
- **Pooling**: global mean, sum and max, plus cluster pooling by squad, for a graph-level label "can this swarm cover the current task list".
- Stretch: an edge-varying filter (EdgeNet).

**Experiments**

1. **The bake-off.** One table: bid accuracy, regret against the oracle's assignment cost, parameter count, inference time.
2. **Depth.** 1 to 8 layers for GCN and APPNP. Plot accuracy and how similar the node embeddings become.
3. **Attrition curve.** Remove 10% to 50% of drones at test time in three ways: at random, by region, and most-connected first. Plot regret against the oracle for every architecture.
4. **What attention sees.** Plot GAT attention weights by neighbour type. Check whether GAT learns to down-weight links that cross a jammed zone. Compare with the MPNN that is given link quality directly.
5. **The fixed post.** Add a fixed command post whose identity matters. Compare a GCNN with an EdgeNet there. Then remove the post at test time, and rerun the transfer test from Mission 4.
6. **Counting.** Predict "how many recon drones are free in this sector" with mean, max and sum pooling.

**Watch out**

- There is one winner among 100 drones. A per-drone binary loss learns to say no to everyone and scores 99% accuracy. Use the softmax over drones.
- GCN uses S = D^-1/2 (A + I) D^-1/2, with self-loops added before normalizing. Without that your numbers will not match the GCN paper.
- Compare architectures at the same parameter budget and the same hop reach. Otherwise the bake-off table means nothing.
- If the no-graph MLP matches the GNNs, the planted relay rule from Mission 0 is too weak. Fix the simulator, not the model.
- Plot GAT attention from the first layer, averaged over heads. Deeper layers attend over mixed features and are hard to read.

**Decision.** The architecture of the AERIE bid network, chosen on how gracefully it degrades and not only on clean accuracy. This is the entry you will quote most.

**Exam drill**

- Write the update rule for GCN, SGC, APPNP, GAT and a generic MPNN from memory.
- Draw the computation graph of one drone for a 2-layer MPNN.
- Show permutation equivariance for a graph filter: relabelling with P gives P^T S P and the output is relabelled the same way.
- Compare parameter counts: GCNN against EdgeNet, K(M + N) + N per filter.
- List the four pooling families and the slide's answer to when you should not change the graph.

### Mission 6: X-Ray (lecture 6, spectral analysis)

**Brief.** One drone is reporting false readings. Find it from the signal alone, in the frequency domain. Then find out how a smarter opponent would hide from you.

**Setup**

- **Shift operator.** Use the Laplacian L of an undirected graph. Its eigenvectors are orthonormal and its eigenvalues are real. A small λ is a low frequency, meaning a signal that varies slowly across links.
- **Size.** At N = 100 a full eigendecomposition with `self_adjoint_eigen` is instant.
- **The simplest high-pass filter is L itself.** The value \[Lx\]\_i is how much drone i disagrees with its neighbours, and it costs one round of gossip. Start the spoof detector there.
- **Detector metric.** Flag a drone when its high-pass output is above a threshold. Score with ROC AUC over 100 trials with and without a spoof.
- **Perturbation metric.** Remove a share of edges and report ||H(S')x - H(S)x|| / ||H(S)x||.

**Build**

- Eigendecomposition of the shift operator and the graph Fourier transform.
- Sanity check: on a cycle graph your GFT must equal the DFT.
- Frequency response of a filter, h(λ) = sum of h\_k λ^k, and a numerical check of the convolution theorem.

**Experiments**

1. **Three spectra.** The jamming field, sensor noise, and one spoofed drone. See where the energy sits. Build a high-pass spoof detector from that.
2. **Colluding spoof.** Let 1, 3, 5 and 10 neighbouring drones report the same false shift. A larger group makes the lie smoother on the graph. Find the group size where your detector goes blind.
3. **Same curve.** Plot the Tikhonov response 1 / (1 + γλ) and the response of your learned filter from Mission 4 on one axis.
4. **Why transfer worked and link loss hurt.** Plot one filter's response curve and mark the eigenvalues of two different swarms on it. Then perturb the graph the way jamming does and measure output change for a low-pass and a high-pass filter.
5. **What ReLU does.** Push a band-limited signal through a ReLU. Plot the spectrum before and after.
6. **Filter bank.** Plot the responses of all F filters in layer 1 of your trained bid network.
7. Stretch: the spectrum of the directed data-flow graph.

**Watch out**

- Use faer's `self_adjoint_eigen` for symmetric matrices, not the general `eigen`. The general one always returns complex values.
- Eigenvector signs are arbitrary, and so is the basis inside a repeated eigenvalue. Compare spectra by energy per frequency, never coefficient by coefficient.
- Frequency ordering flips between operators. For L a small eigenvalue is smooth. For the normalized adjacency a large eigenvalue is smooth. Say which S you use on every plot.
- Remove the mean of the signal before the ReLU experiment, or the constant component hides everything else.

**Decision.** Which frequency bands the brain keeps, the spoof detector's threshold, and its blind spot written down in plain words.

**Exam drill**

- Derive the spectral form of the Tikhonov solution.
- State the convolution theorem for graphs.
- Answer the two slide questions: the spectral role of the filter bank and of the pointwise nonlinearity. Experiments 5 and 6 are your pictures.
- Explain the trade-off between locality in the vertex domain and in the spectral domain.

## Act III: Time (lectures 8 and 9)

Act III gives the brain a sense of time. The opponent moves, so forecasting means staying one step ahead. First the signals change while the formation holds. Then the formation itself changes.

### Mission 7: Forecast (lecture 8, graph-time learning)

**Brief.** The swarm holds station while a jammer moves along the valley. Forecast jamming power and battery level at every drone five steps ahead, so the swarm knows which links die next. Fill in dropped packets on the way.

**Setup**

- **Task type.** Node forecasting on a fixed graph. Input is the last 5 steps at every drone. Output is steps 1 to 10 ahead.
- **Data.** On-station sequences, 200 steps each. Two features per drone: jam reading and battery. Train on seeds 0 to 79.
- **Metric.** NMSE per horizon.
- **The baseline to beat.** Persistence, which predicts that the next value equals the current one.
- **Parameter counts to check.** A VAR needs N^2 = 10,000 numbers per lag. A graph VAR needs K + 1 per lag.
- **Product graph.** With a window of 5 steps it has 500 nodes. Write your own sparse Kronecker product over edge lists. It is about ten lines, and it is the exam formula in code. Use a directed path as the temporal graph for forecasting, so information only flows from past to future.
- **Parametric product graph.** S\_product = s00 (I ⊗ I) + s01 (I ⊗ S) + s10 (S\_T ⊗ I) + s11 (S\_T ⊗ S). The first index is time and the second is space.

**Build**

- **Baselines.** Per-drone autoregression, then a full VAR. Count the VAR parameters.
- **Graph VAR.** x\_t = sum over p of H\_p(S) x\_(t-p), with the filters fitted by least squares.
- **Disjoint models.** Graph then time (GCNN into a GRU), and time then graph.
- **Graph RNN.** Graph filters inside the recurrent state update.
- **Product graphs.** Build the Kronecker (S\_T ⊗ S), Cartesian (S\_T ⊗ I + I ⊗ S) and strong product from S and a temporal path graph. Then the parametric product graph with four learnable weights s\_ij.
- **Graph-time CNN** on the product graph, and graph-time Tikhonov, (I + γ L\_product)^-1 y.

**Experiments**

1. **Horizon.** Forecast error at 1 to 10 steps for every model in one plot.
2. **Twenty steps of data.** The opponent just changed behaviour and you have 20 time steps to learn from. Compare VAR and graph VAR at 20, 100 and 1000 steps.
3. **What couples what.** Train the parametric product graph with a static jammer, then a moving one. Read the learned s\_ij. Predict beforehand which weight grows when jamming moves between drones over time.
4. **Missing where it matters.** Jamming drops packets exactly where the signal is most interesting. Compare imputation when packets are lost at random and when they are lost inside the jammed zone.
5. **Space and time together.** Graph-time Tikhonov against the space-only map builder from Mission 3.

**Watch out**

- Persistence is hard to beat at horizon 1. If you lose to it badly, look for an off-by-one in your targets.
- The Kronecker order must match how you stack the signal. Build the 3-node, 2-step example by hand and compare with your code. That is also the exam drill.
- Normalize each feature with statistics from the training seeds only.
- A directed temporal graph is not symmetric. Graph-time Tikhonov needs the undirected version.

**Decision.** The forecasting module, how many steps of warning it buys, and whether a product graph earns its N times T nodes.

**Exam drill**

- Write the adjacency of the three product graphs and sketch each for a 3-node graph over 2 time steps.
- Count product graph edges and check against the slide formula.
- Parameter count: VAR against graph VAR against graph-time CNN.
- List the slide's pros and cons of disjoint models against product graph models.

### Mission 8: The Swarm Moves (lecture 9, temporal graphs)

**Brief.** Drones reposition, jamming cuts links, attrition removes nodes, replacements join. Keep the brain working, and predict the network splitting in two before it happens.

**Setup**

- **Data.** Flights with motion, link loss and attrition. One snapshot per step plus the event log.
- **Link prediction target.** Whether link i to j exists 5 steps ahead. Score only candidate pairs: drones currently linked or within 1.5 times radio range. Metrics are AUC and average precision.
- **Baselines to beat.** "The link stays as it is now", and a physics baseline that extrapolates current distance from relative velocity.
- **Split warning label.** True when the surviving drones no longer form one connected component 5 steps ahead. The classes are unbalanced, so report precision and recall, not accuracy.
- **Learned simulator.** Node state is position, velocity, role and jam reading. Edge features are relative position and distance. Predict the change in state, then add it to the current state. Train with rollouts of H = 5.
- **EvolveGCN.** Implement the two update rules as the slides write them.

**Build**

- **Two views of one log.** Export the same flight as snapshots and as an event stream (node added, node removed, edge added, edge removed).
- **Temporal link prediction.** Will link i to j exist in five steps? A relay could move before the link dies.
- **EvolveGCN**, both variants: hidden-state (node features drive the weight update) and input-output (weights evolve from their own past).
- **Supra-adjacency.** Stack snapshots into one large static graph and reuse Mission 7.
- **Encoder, processor, decoder.** A learned simulator: MLP encoders for nodes and edges, L rounds of message passing, MLP decoder to the next state. Rebuild the epsilon graph at every step. Train on H-step rollouts where predictions feed back in.

**Experiments**

1. **Churn.** Link prediction for a frozen-graph GCN, both EvolveGCN variants and the supra-adjacency model as link churn rises from motion and jamming together. Find the churn level where the static model falls over.
2. **Split warning.** Turn link predictions into a graph-level forecast: is the swarm still one connected network in five steps? Plot precision and recall against warning lead time.
3. **Rollout.** Train the simulator with a 1-step loss and with an H-step loss. Roll both out for 50 steps and plot the error.
4. **Losses as events.** Attrition is a stream of node deletions, and replacements are node insertions. Check which models keep working through both without retraining.
5. **What if.** Ask the learned simulator which squad gets cut off if relay 7 is lost. Compare with SwarmSim's real answer.
6. Stretch: classify a dead link as jammed or just out of range (anomaly detection), and predict how long a cut-off squad takes to reconnect (event time prediction).

**Watch out**

- Predict the change in state, not the state. A network asked for absolute positions learns to copy its input.
- Random drone pairs as negatives make link prediction look solved, because most pairs are far apart. Use the candidate pairs.
- The node set changes under attrition. Keep the stable drone ID and a mask of who is alive. Never reindex nodes between snapshots.
- A simulator trained only on 1-step loss drifts in rollout. Adding small noise to its inputs during training is a known fix worth trying next to the H-step loss.

**Decision.** How AERIE handles topology change, whether the split warning is reliable enough to move a relay on, and whether the learned simulator is good enough to plan with.

**Exam drill**

- Define each temporal task as a function, for example link prediction as a map from (node, node, time) to \[0, 1\].
- Snapshot-based against event-based, transductive against inductive, past against future prediction.
- Write both EvolveGCN update rules and say when each makes sense.
- Write the multi-step training loss and explain why it gives a more stable forecast.

## Act IV: Deployment (lectures 10 to 12)

Act IV asks the three questions that decide whether anyone trusts the brain. Each has a concrete person asking it: the engineer with no data centre, the operator who signs off, and the opponent holding a recovered drone.

### Mission 9: Ten Thousand Drones (lecture 10, scalability)

**Brief.** Scale the swarm from 100 to 100,000 nodes. Then take away the data centre: the swarm has to retrain on board, and a cut-off squad has to keep learning alone.

**Setup**

- **Task.** Node classification on one large swarm: is this drone inside a jammed zone, predicted from noisy readings and neighbours. Every drone has a label, so batches of any shape can be scored.
- **Data.** Large swarms at 1k, 10k and 100k drones with the arena scaled to keep the mean degree near 10.
- **What to measure.** Wall-clock time per epoch, peak memory, test accuracy, and the number of epochs needed to reach 95% of full-batch accuracy.
- **A reality check on size.** A 2-layer GCN on 100k drones still fits on a laptop, because memory grows with layers times N times F. The wall shows up with more layers or around a million nodes. That is where DelftBlue earns its place.
- **Start with.** 3 layers, 64 hidden features, 512 target drones per batch, fan-out of 10 neighbours per layer, 50 clusters with 5 merged per batch.

**Build**

- Timers and a peak-memory tracker around full-batch GCN training.
- Naive mini-batching, with a counter for the size of each node's computation graph.
- A GraphSAGE-style neighbour sampler written by hand.
- FastGCN layer-wise sampling with importance weights.
- Cluster-GCN: partition by squad or by METIS, train on induced subgraphs, then the version that merges several random clusters per batch.
- Preprocessing: compute S^k X once (SGC, SIGN) and train a plain MLP on top.

**Experiments**

1. **The scaling table.** Time per epoch, peak memory and accuracy for all methods at 1k, 10k and 100k nodes. Add a 1M-node run on DelftBlue to watch full-batch training hit the wall.
2. **Explosion.** Computation graph size against depth, with and without sampling. Sweep the fan-out.
3. **Estimator noise.** Variance of the loss estimate for FastGCN with uniform and with importance sampling.
4. **Squad-local training.** Cluster-GCN with squads as clusters is exactly a squad training on its own subgraph. Measure the share of edges lost between squads and the label skew per squad. Then merge squads that can still reach each other into one batch and measure again.
5. **Fits on a drone.** Which recipe trains inside 2 GB. Note that SGC and SIGN pay their K rounds of radio traffic once, up front, and neighbour sampling pays every batch.

**Watch out**

- Time the sampler and the model separately. In a hand-written sampler the sampling often costs more than the training step.
- Sample neighbours during training only. At test time use full neighbourhoods, or the accuracy you report is noisy.
- Squads are spatially compact, so squad partitions give batches with skewed labels. That is problem 2 from the Cluster-GCN slides. Notice it when it happens.
- Compare methods at equal epochs and at equal wall-clock time. The two comparisons tell different stories, and the slides' key points are about exactly that difference.

**Decision.** The on-board training recipe, the inference recipe, and the fallback for a squad that is cut off.

**Exam drill**

- Derive time and space complexity for full-batch GCN, neighbour sampling, FastGCN and Cluster-GCN.
- Explain neighbourhood explosion with a drawing.
- Do the slide's learning activity: which sampler do you pick when the network is very deep, and why? Experiment 2 answers it.
- State the two problems of vanilla Cluster-GCN and how sampling several clusters fixes both.

### Mission 10: Why Drone 17? (lecture 11, explainability)

**Brief.** A human operator signs off on what the swarm does and answers for it afterwards. Every bid and every spoof flag needs a reason you can prove is right.

**Setup**

- **What gets explained.** The frozen bid network from Mission 5. One instance is a snapshot, a task and the winning drone.
- **What an explanation is.** A subset of edges and features inside the winner's K-hop computation graph.
- **Ground truth.** The edges on the path from the winner to its nearest relay, plus the battery and capability features.
- **Faithfulness.** Keep only the explanation and check the decision holds. Remove the explanation and check the decision changes. For Zorro, use RDT-fidelity with noise in place of the unselected inputs.
- **Sparsity.** Selected size over total size for hard masks. Entropy of the normalized mask for soft masks.
- **Correctness.** Precision and recall of the selected edges against the planted path.
- **Evaluation set.** 200 decisions the network got right and 50 it got wrong.
- **Start with.** A size penalty near 0.005 and an entropy penalty near 1 for GNNExplainer as values to tune, 200 optimization steps, Adam with learning rate 0.01.

**Build**

- **A planted rule.** Reuse the relay rule planted in Mission 0 and add a battery threshold. A drone qualifies only with a relay within two hops and battery above the threshold, so you know the correct explanation.
- **Gradient explainers.** Saliency and gradient times input, on features and on edges.
- **GNNExplainer** from scratch: learn a soft edge mask and a soft feature mask, with size and entropy penalties.
- **Zorro**: hard masks picked greedily, scored by RDT-fidelity, with unselected inputs replaced by noise.
- Stretch: PGMExplainer (perturb, record, fit a surrogate), and a global view built by collecting local explanations.

**Experiments**

1. **The scorecard.** Faithfulness, sparsity and correctness against the planted rule, for every explainer.
2. **Stability.** Run GNNExplainer with ten seeds on the same decision. Compare the masks.
3. **Explain a flag.** Run the explainers on the spoof detector from Mission 6. A good explanation names the neighbours whose readings disagree with the flagged drone.
4. **Ten seconds.** An operator has seconds, not minutes. Plot faithfulness against explanation size and find the smallest explanation that is still faithful.
5. **Plausible but wrong.** Find a decision where an explanation looks sensible and scores badly on faithfulness. Then take a bid the network got wrong and see whether the explanation shows you why.

**Watch out**

- The network may have learned a shortcut that is not your planted rule. A faithful explainer then scores low on correctness. That gap is the difference between the two metrics, and it is the drawback of correctness the slides point out.
- Explain the winner's score relative to the runner-up. With 100 drones the raw probability is tiny and its gradients are flat.
- Apply masks as edge weights inside message passing. Deleting edges from the tensor gives no gradient to learn a mask from.
- For Zorro, draw replacement values from each feature's own distribution in the dataset.

**Decision.** The explainer that ships in the operator view, and the maximum size of what it shows.

**Exam drill**

- Local against global, post hoc against interpretable by design.
- The three post hoc families (gradient, perturbation, surrogate) with one method and one weakness each.
- Define faithfulness, sparsity (hard and soft masks), correctness and plausibility. Say why plausibility needs a faithfulness check next to it.
- Pros and cons of GNNExplainer against Zorro.
- Why an explanation lives inside the node's computation graph.

### Mission 11: Red Team (lecture 12, privacy)

**Brief.** A drone is lost and the opponent recovers it with the model on board. Find out what that model tells them about the rest of the swarm, then close the leak.

**Setup**

- **Target model.** A role classifier trained on a set of "sensitive" flights. Let it overfit a little, since that is what the attack feeds on.
- **Membership inference protocol.** Four disjoint sets of swarms: target train, target test, shadow train, shadow test. The shadow model has the same architecture as the target. The attack model is a small MLP on the sorted output probabilities, trained to tell shadow members from non-members. Metric is AUC, where 0.5 means no leak.
- **Structure attack protocol.** One explanation vector per drone. Score a pair by cosine similarity. AUC against the true adjacency on a balanced sample of links and non-links. The baseline is cosine similarity of the raw features. An attack only counts if it beats that.
- **Laplace mechanism.** Noise scale is sensitivity divided by ε. For one drone's degree, adding or removing one link changes the answer by 1.
- **PrivGNN roles.** Private data is the sensitive flights. Public data is unlabelled swarms from another area. Start with a sampling ratio of 0.2 and 5 private neighbours per query, then sweep the noise scale.
- **ε grid.** 0.1, 0.5, 1, 2, 5, 10, and no noise.

**Build**

- **Membership inference.** Train a shadow model on a similar swarm, then an attack classifier on output posteriors. Ask whether a given drone or flight was in the training data.
- **Structure leak.** Score every drone pair by cosine similarity of their feature explanations from Mission 10. Use the scores to reconstruct the comm graph.
- **Laplace mechanism.** Noise scaled to sensitivity over ε on a simple query (drone degree) and on model outputs.
- **PrivGNN.** A private teacher trained on a random subsample, queried through the k nearest private neighbours of each public node, answering with Laplace noise. A public student trained on those noisy labels is the model you release.

**Experiments**

1. **Where have you flown.** Membership inference tells the opponent whether a given flight or area was in the training data. Attack AUC for GCN, SGC, GAT and GraphSAGE against the train and test accuracy gap. The slides claim GCN and SGC leak most and GAT least. Test that claim.
2. **Topology from explanations.** Graph reconstruction AUC for each explainer from Mission 10, and how well the result locates the relays. Compare with what the traffic listener got in Mission 2. The most useful explainer may be the leakiest.
3. **Forward and rear.** PrivGNN maps cleanly onto this setting. The teacher trained on sensitive flights stays in the rear. Only the student trained on noisy labels flies forward. Plot the student's accuracy against ε.
4. **Query-time defence.** Perturb the neighbourhood at query time, as the slides suggest. Plot attack AUC against bid accuracy.
5. **Sanity check.** Run the Laplace mechanism on two neighbouring datasets 100,000 times. Confirm the output ratio stays below e^ε.

**Watch out**

- If train and test accuracy are equal, the membership attack sits at 0.5. That is the correct result and not a bug.
- Every query spends privacy budget. By the composition theorem the total ε grows with the number of public nodes you label. Track it.
- Never let shadow data touch the target model.
- The sanity check estimates a probability ratio from histograms. Use wide bins and many samples, or the empty tails will show violations that are not real.

**Decision.** Which model is allowed to fly forward, and what AERIE may send over the air and show the operator. This entry will argue with the one from Mission 10, and that tension is the point.

**Exam drill**

- Why anonymization fails, with the Netflix example.
- The definition of ε-differential privacy, sensitivity, and the Laplace scale.
- The membership inference pipeline step by step.
- PrivGNN: the role of the public graph, the composition theorem, and why subsampling amplifies privacy.
- The explanation attack and which explanation types leak most.

## The finale

The last piece of work is one replay where every module runs in the same five-minute flight over the valley. That replay, the report and the decision log are what you show people.

**The replay, beat by beat**

1. The swarm is on station. A jammer switches on. The jamming map fills in from a handful of noisy sensors (Mission 3).
2. Links inside the zone go red. The state of the silent drones is inferred from the ones still reporting (Mission 1).
3. One drone starts reporting false readings. The detector flags it, and the explanation names the three neighbours that disagree (Missions 6 and 10).
4. The forecast shows the jammed zone drifting east. The split warning fires five steps before the network would break, and a relay shifts to hold it together (Missions 7 and 8).
5. A contact enters sector 4. Drones bid within two rounds of radio traffic while the emission counter ticks up (Missions 4 and 5).
6. Six drones are lost. Bidding carries on, and the regret against the oracle stays on screen (Mission 5).
7. The operator clicks the winning drone and reads why it won (Mission 10).
8. A corner panel shows what a recovered drone would reveal right now (Mission 11).

Build it as a GIF with plotters first. A small web view with a time slider is the stretch.

**Repo layout**

```
aerie/                    Cargo workspace
  crates/
    swarmsim/             Mission 0: simulator, datasets, oracle
    aerie-graph/          edge list, CSR, shift operators, metrics
    aerie-classic/        lp, embed, regularize, spectral, var
    aerie-nn/             Burn models: filters, gnn, graphtime, temporal
    aerie-trust/          scale, explain, privacy
    aerie-plot/           plotters helpers and the GIF writer
  experiments/            one binary per experiment: m01_e1_which_graph.rs ...
  results/                JSON and figures, one folder per experiment
  exam/                   m01.md ... m11.md, one page per lecture
  DECISIONS.md
  REPORT.md
```

**A decision log entry looks like this**

```
D05. Bid network architecture
Chose:     2-layer MPNN with edge features, K = 2
Rejected:  GAT (same accuracy, 3x inference time), GCN (regret 2x higher)
Evidence:  experiments/m05_e3_attrition_curve/regret.png
Cost:      2 rounds of radio traffic per bid
Revisit:   if link quality stops being available as an edge feature
```

The values above are placeholders to show the format. Yours come from your runs.

**The flight report.** 6 to 8 pages. The problem in half a page. One section per act with its two or three best figures. A full page on what failed and why. End with the open question you would work on next with real flight logs.

## Coverage map

Every topic in your 11 decks lands in a mission. The topics below come from the slide titles. Lecture 7 was project guidance in your course outline, so it has no deck and no mission. Use this table as the checklist the night before the exam.

| Lecture | Slide topics | Mission |
| --- | --- | --- |
| 1. Introduction | Graph notation and types. Node, link and graph tasks. Correlation, epsilon and kNN graphs. Supervised, semi-supervised, unsupervised. Transductive and inductive. Adjacency and transition matrices. Random walks. Label propagation and its convergence. Random walk with restart. | M0, M1 |
| 2. Unsupervised representation learning | Encoder-decoder view. Shallow encoders. Random walk objective. Negative sampling. Word2Vec link. DeepWalk. Node2Vec. Directed graphs, node roles, alternating walks, NERD. Matrix factorization. Katz similarity and HOPE. Shortcomings. Downstream tasks. | M2 |
| 3. Graph regularization | Graph signals. Signal variation. Tikhonov and its closed form. Bias-variance trade-off. Trend filtering. Total variation with 2-norm and 1-norm. Elastic net and Sobolev. Design guidelines for fitting term and regularizer. | M3 |
| 4. Graph convolutional neural networks | Graph shift and shift sequence. Convolutional filter and its properties. Learning with filters. Graph collaborative filtering. Graph perceptron. GCNN. Why GCNNs work. Link to CNNs. Multiple features and filter banks. Transference. | M4 |
| 5. GNN architectures | GCN. SGC. APPNP. Permutation equivariance. Parameter sharing. Edge-varying filters and EdgeNet. Attention scores and GAT. Message passing and computation graphs. Pooling: node selection, clustering, zero-pad, global aggregation and its issues. | M5 |
| 6. Spectral analysis | Time as a graph and the DFT. Graph Fourier transform. Frequencies and eigendecomposition. Spectral view of regularizers. Filter frequency response. Convolution theorem. Filter types. Same filter on different graphs. Spectral locality. Role of filter banks and nonlinearity. Directed graphs. | M6 |
| 8. Graph-time learning | Spatio-temporal tasks on a fixed graph. Disjoint models both ways. VAR and graph VAR. Graph RNN. Kronecker, Cartesian, strong and parametric product graphs. Graph-time regularization, filters and CNN. Model comparison. | M7 |
| 9. Temporal graphs | Temporal tasks. Snapshot and event taxonomy. Transductive and inductive, past and future. EvolveGCN and weight evolution. Supra-adjacency. Encoder-processor-decoder. Multi-step training. | M8 |
| 10. Scalability | Full-batch time and space cost. Mini-batch construction. Neighbourhood explosion. GraphSAGE sampling. FastGCN layer-wise sampling and loss variance. Cluster-GCN, its two problems and the fix. Scaling by preprocessing (SGC, SIGN). | M9 |
| 11. Explainability | What an explanation is. Local and global. Post hoc and by design. Gradient explainers. GNNExplainer. Zorro and RDT-fidelity. PGMExplainer. Faithfulness, sparsity, correctness, plausibility. Global explanations and XGNN. | M10 |
| 12. Privacy | Why anonymization is hard. Differential privacy. Sensitivity and the Laplace mechanism. Membership inference on GNNs. Model comparison and defences. PrivGNN, composition and subsampling. Explanations leaking graph structure. | M11 |

Two topics are read-only in this plan: XGNN in lecture 11 and the real-world case studies that open lecture 1. Read those slides, no code needed.

## Order of attack: 7 weeks

You are learning Rust while you build this, so the earlier figure of 68 hours was too low. It assumed fluency. The schedule below fits 56 to 70 hours by building the backbone in Rust, covering lectures 8 to 12 on paper plus a few small experiments, and leaving the rest for after the exam. Every hour figure is a rough guess. Recalibrate after week 1.

| Week | Dates | Build in Rust | Exam drill on paper | Hours |
| --- | --- | --- | --- | --- |
| 1 | Nov 12 to 18 | M0 steps 1 to 3 | None | 10 |
| 2 | Nov 19 to 25 | M0 step 4, M1 First Contact, M3 Clean Signal | Lectures 1, 2 and 3 | 10 |
| 3 | Nov 26 to Dec 2 | M4 Gossip | Lectures 4 and 8 | 10 |
| 4 | Dec 3 to 9 | M5 The Bid | Lectures 5 and 9 | 10 |
| 5 | Dec 10 to 16 | M6 X-Ray, overflow from M5 | Lectures 6 and 10 | 9 |
| 6 | Dec 17 to 23 | Three small experiments, listed below | Lectures 11 and 12 | 9 |
| 7 | Dec 24 to 30 | Nothing | All 11 drills again, from memory | 8 |

That is 66 hours. The dates count seven weeks from November 12. Shift them if the exam falls on a different day.

**The small experiments in week 6.** All three are plain Rust with no neural network.

| Experiment | Lecture it makes concrete |
| --- | --- |
| M7 experiment 2: VAR against graph VAR by least squares | 8 |
| M9 experiment 2: computation graph size against depth, with and without sampling | 10 |
| M11 experiment 5: the Laplace sanity check | 12 |

If Mission 5 is solid by then, add gradient saliency on one bid for lecture 11.

**Why the drill column runs ahead of the build column.** Lectures 2 and 8 to 12 get little or no code before the exam, so their drills are spread over weeks 2 to 6. Every lecture then has paper time before the final revision week.

**Why this build order.** M3, M4, M5 and M6 depend on each other in that sequence. They also carry the ideas every later lecture builds on: the shift operator, graph filters, message passing and the spectral view.

**Checkpoint at the end of week 1.** If M0 steps 1 to 3 are not done, the Rust learning cost is higher than guessed. Then narrow Mission 5 to GCN and the MPNN, and Mission 6 to experiments 1 to 3.

**How to spend a week.** Two sessions of three to four hours for the build, then one shorter session a day or two later for the decision entry and the drill. Doing the drill after a gap, from memory, is what makes it stick.

**The one rule.** Never trade a drill for code. The exam is written.

**After the exam.** M2, the full versions of M7 to M11, the replay and the flight report.

**Open question**

- [ ] Are past exams or practice questions on Brightspace? If yes, map each question to a mission and do those experiments first.
