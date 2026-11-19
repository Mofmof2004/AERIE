# Decision log

One entry per decision: what went into AERIE, what got rejected, and the evidence.

---

## D00. Ground rules, settled before coding

**Link rule.** Two drones are linked when their distance is at most
min(range_i, range_j), because a working link has to go both ways. Link weight is the
Gaussian kernel exp(-d² / 2σ²) with σ = half the link range, so a pair at the edge of its
range always weighs exp(-2) ≈ 0.135. Built by `swarmsim::graphs::epsilon::epsilon_graph`.

**Textbook ε-graph kept for comparison.** `textbook_epsilon_graph` uses one ε and one σ for
every pair. It is not used for datasets. It shows what the min rule fixes: with a single ε,
a recon drone gets linked to a relay it cannot answer, or relay pairs are lost.

**kNN symmetry.** A kNN graph is directed. AERIE uses the union: i and j are linked when
either one picks the other. The mutual version can leave drones isolated.

**Graph storage.** Undirected edge list, each link stored once with src < dst. Node `i` is
the index into the swarm, not the drone ID.

**IDs.** `DroneId` and `SquadId` are their own types, so an ID cannot be used as a vector
index by accident. This matters once attrition starts.

**Determinism.** All randomness goes through `ChaCha8Rng` with an explicit seed, passed as
`&mut`. No `HashMap` or `HashSet` where iteration order reaches the output. No rayon
inside the simulator. Floats are sorted with `total_cmp`.

**Seeds.** 0–79 train, 80–89 validate, 90–99 test.

**Later steps, already settled.**
- A link fails when jamming power at either end is above 0.5 P0.
- Homophily is measured on the ε-graph before jamming, since jammed drones become isolated.
- Tasks store both oracle labels: best single drone, and the joint (Hungarian) assignment.

---

## D01. ε-graph tuning

```
Chose:     short range 2.5 km, relay range 16 km, squad spread 2 km, cells 5 x 5 km
Rejected:  the plan's starting values, 5 km / 9 km
Evidence:  cargo run --release -p swarmsim --example epsilon_stats  (seeds 0-99)
Cost:      the swarm depends on its relays; 10 of 100 seeds start disconnected
Revisit:   when jamming (step 2) or attrition cuts the relay backbone too often
```

**Targets** (Project_Plan, Mission 0): mean degree 8 to 10, one connected component before
jamming, about a quarter of links between squads.

**Result over seeds 0–99**

| | Value | Target |
| --- | --- | --- |
| Mean degree | 8.72 | 8 to 10 |
| Links between squads | 25 % | about 25 % |
| Connected seeds | 90 / 100 | all |

**Why the plan's values were rejected**

| Config | Mean degree | Connected | Between squads |
| --- | --- | --- | --- |
| 5 / 9 km, spread 1 km (plan start, seeds 0–9) | 14.8 | 1 / 10 (3.0 components on average) | 39 % |
| 5 / 9 km, spread 2 km (seeds 0–79) | ≈ 14 for non-relays | 21 / 80 | not measured |
| **2.5 / 16 km, spread 2 km (chosen, seeds 0–99)** | **8.72** | **90 / 100** | **25 %** |

With a 5 km short range, a squad scattered over a few km is close to a complete graph: each
drone gets about 9 links from its own squad alone, and nearby squads add more. The degree
overshoots the target before the squads are connected to each other.

**What the chosen values do to the structure.** This is a deliberate trade-off:
- **Relays are the backbone.** Mean degree is about 16 for relays and about 6.7 for every
  other role. Only about 2 % of links join two squads without a relay. Squads talk to each
  other through relays.
- **Squads are not cliques.** With a 2.5 km range inside a 4 x 4 km square, each squad has
  real internal structure instead of a complete graph.
- **Good for later missions.** The relay role matters in M5. Removing the most-connected
  drones first gives a strong attrition experiment. Degrees vary a lot (≈ 16 vs ≈ 7), so
  GCN normalisation matters.
- **Fragile under attack.** Losing a few relays can split the swarm. Step 2 (jamming) and
  attrition will show how fragile.

**Disconnected seeds.** They are kept in the dataset but **excluded from every experiment
that needs one component**: M1 label propagation, M3 reconstruction, M4 gossip, M6 spectrum,
M8 split warning. The check is done in code (`stats(..).n_components == 1`). Every experiment
uses the same list. *Amended by D05 for M1: label propagation handles cut-off drones (no
guess), so M1 keeps all test seeds and reports the no-guess share.*

| Split | Excluded seeds | Seeds left |
| --- | --- | --- |
| Train (0–79) | 2, 10, 18, 30, 54, 63, 79 | 73 |
| Validate (80–89) | 87 | 9 |
| Test (90–99) | 90, 92 | 8 |

There are two kinds of split:
- **One stray drone** (largest component 99): seeds 10, 18, 54, 92.
- **Whole squads cut off** (largest component 50–90): seeds 2, 30, 63, 79, 87, 90.

---

## D02. Jamming: placement, readings, labels, link failure

```
Chose:     one jammer + one ridge per seed, redrawn until 5-30 % of drones are jammed;
           labels and link failure from the true power; readings = truth + noise, not clamped
Rejected:  jammer placed near a squad; a fixed jammed share; labels from readings;
           clamped readings; removing jammed drones from the graph
Evidence:  cargo run --release -p swarmsim --example jamming_stats  (seeds 0-99)
Cost:      jamming splits the free drones on 37 / 100 seeds; few drones sit near the threshold
Revisit:   several jammers (combine rule), moving jammers (step 4), M1 if labels are too easy
```

**Units.** P0 = 1, so every power is a fraction of the jammer's power at its own position.
The threshold is `jam_threshold * jam_p0` (0.5 P0). P0 cancels out of who is jammed: doubling
it doubles every power and the threshold.

**Field** (`jam_power`). P(d) = P0 / (1 + (d / d0)²) with d0 = 4 km, so the power is half of
P0 at d0. It is multiplied by 0.1 when the straight line from the jammer to the point crosses
the ridge (`segments_cross`, strict test: touching the ridge does not shadow).

**Placement** (`place_jamming`, rejection sampling, after `spawn_swarm` with the same RNG).
- Jammer uniform in the arena, power P0.
- Ridge middle 2 to 3 km from the jammer in a random direction, the ridge roughly facing it
  (perpendicular ± 0.5 rad), 10 to 14 km long.
- Kept only if 5 % to 30 % of the drones are jammed; otherwise both are drawn again.
  Panics after 10,000 attempts so a bad config cannot loop forever.
- Rejected: placing the jammer near a squad. Rejection sampling gives the same control over
  the jammed share with fewer special cases.

**Truth, readings, labels.**
- `true_power`: the exact power at each drone. The simulator keeps it for every error metric.
- `readings`: truth + Normal(0, noise_sd · P0), noise_sd = 0.1, for **every** drone, jammed or
  not. Not clamped: a reading can be slightly negative. Drawn after placement with the same RNG.
- `jammed_labels`: jammed when the **true** power is **at or above** 0.5 P0. The plan says
  "above"; `>=` matches its "dead zone of radius d0" (exactly 0.5 P0 at d0). Equality never
  happens in practice.
- Labels come from the truth, never the readings: they are the answer key.

**Link failure** (`fail_links`). A link fails when either end is jammed (from the labels, so
from the truth). `n` is unchanged: jammed drones stay as isolated nodes, so drone `i` is still
index `i` and the labels line up.

**Several jammers.** Not needed yet: every seed has one. The jamming playground can sum or take
the max. Noise powers add, so **sum** is the likely choice. To decide when a mission needs two.

**Result over seeds 0–99**

| | Value |
| --- | --- |
| Jammed share | min 5 %, mean 9.6 %, max 19 % (the 30 % cap is never reached) |
| Relays jammed per seed | 1.9 of 20 (the same share as other drones) |
| Links lost | 13.1 % |
| Free drones split into 2+ pieces | 37 / 100 seeds (10 / 100 before jamming) |
| Free drones outside the largest piece | 3.7 per seed |
| Readings on the wrong side of the threshold | 1.8 drones per seed |

**What this means for later missions.**
- Jamming opens a real hole: about one link in eight is lost, and on a third of the seeds the
  free drones split. M8's split warning has positive examples.
- Most drones sit far from the threshold, so on average fewer than 2 per seed have a reading
  that contradicts their label. Classifying "jammed" from one's own reading is easy.
  M1 hides 90 % of the readings, which is where the graph has to do the work. If M1 still
  turns out too easy, raise noise_sd or label from a softer zone.

---

## D03. Link failure by signal against jamming

```
Chose:     a link direction works when the sender's signal ≥ N0 + jamming at the receiver;
           a link needs both. N0 = 0.5, signal scales recon 0.5, RF 0.7, relay 0.3,
           compute 1.0 km. One-way directions kept, in two lists.
Rejected:  D02's either-end rule; the plan's N0 = 0.05; the margin as the edge weight;
           one-way links only from jamming (option a); one mixed one-way list
Evidence:  cargo run --release -p swarmsim --example link_stats  (seeds 0-99)
Cost:      a second O(n²) pass for one-way directions; about 720 range one-way directions
           per seed, so the directed graph is mostly relays broadcasting
Revisit:   M9 (the extra pass needs the grid neighbour search), several jammers,
           M2 / M6 if the directed graph is dominated by relays
```

Replaces the link-failure part of D02 ("a link fails when either end is jammed"). D02's
placement, readings and zone labels stay. `fail_links` is kept for comparison only.

**Why.** Under D02 a jammed drone lost every link, even to a neighbour 100 m away, and
relays were as fragile as anything else. Real links fail when the jamming at the receiver
drowns the sender's signal: close neighbours and loud transmitters get through.

**The model** (`swarmsim::links`).

```
tx_power(role) = N0 · (1 + (range / scale)²)
signal(i→j)    = tx_power(i) / (1 + (d_ij / scale_i)²)
margin(i→j)    = signal(i→j) / (N0 + jam(j))          jam(j) = true power at j
i→j works when margin ≥ 1;  link i–j is live when both directions work
```

- **Transmit power comes from the range.** With no jammer the signal falls to N0 exactly at
  the range, so the live graph equals the geometric graph (`epsilon_graph`, D00 / D01 ranges
  unchanged). Unit test: `no_jammer_gives_the_geometric_graph`.
- **Signal scale per role** sets how well a role holds up under jamming, separately from its
  range. Smaller = louder transmitter. Relays toughest (0.3), compute weakest (1.0).
- **N0 works backwards from intuition.** Since tx is tied to N0, the margin splits into
  `geometry × N0 / (N0 + jam)`. With no jammer N0 cancels out; with a jammer a small N0 means
  quiet drones that the jammer drowns. N0 sets how loud the drones are against the jammer.
- **Jamming hits the receiver.** A drone near the jammer can be heard but cannot hear.
- **Ridges** cut the jamming (× 0.1, D02) but do not block drone-to-drone signals.
- **The zone label is unchanged** (true power ≥ 0.5 P0) and still drives placement (5–30 %).
  It no longer means the drone is cut off.

**What `live_graph` returns** (`LiveGraph`).
- `graph`: live links, **with their Gaussian weights**. The margin is ≥ 1 and unbounded, so
  using it as the weight would break label propagation, Laplacians and GCN normalisation
  (M1, M3, M6). Rejected.
- `margins`: the smaller of the two margins per live link, as an edge feature (M5).
- `cut_one_way`: geometric links cut one way by jamming (sender, receiver).
- `range_one_way`: directions between drones that are not geometric neighbours: the
  longer-range drone is heard beyond the other's range (a relay heard by a recon drone at
  5 km). Option (b): physically real, and the plan's directed graph for M2 and M6. Kept in
  its own list because it is large; a mission picks what it needs.

**Calibration over seeds 0–99** (ranges 2.5 / 16 km, default jamming)

| N0 | scales (recon, RF, relay, compute) | Links lost | Live degree | Far links kept | Zone links kept | Seeds split | Stranded | Cut one-way | Range one-way |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 (plan) | 0.5, 0.7, 0.3, 1.0 | 48 % | 4.57 | 73 % | 6 % | 100 | 39.9 | 86 | 468 |
| 0.2 | 0.5, 0.7, 0.3, 1.0 | 26 % | 6.46 | 91 % | 27 % | 92 | 14.1 | 59 | 642 |
| **0.5** | **0.5, 0.7, 0.3, 1.0** | **15 %** | **7.42** | **96 %** | **51 %** | **38** | **6.3** | **37** | **720** |
| 0.5 | 1.0, 1.0, 1.0, 1.0 | 16 % | 7.33 | 96 % | 45 % | 46 | 6.7 | 39 | 719 |
| 1.0 | 0.5, 0.7, 0.3, 1.0 | 9 % | 7.94 | 98 % | 69 % | 16 | 4.0 | 24 | 759 |
| 2.0 | 0.5, 0.7, 0.3, 1.0 | 5 % | 8.28 | 99 % | 82 % | 12 | 2.9 | 14 | 783 |

*Far*: both ends ≥ 12 km from the jammer. *Zone*: both ends in the jammed zone.
*Split*: 2+ pieces of at least 2 drones (10 seeds are split before jamming, D01).

**Reading the table.**
- **The plan's N0 = 0.05 is rejected:** the long tail thins the whole swarm (73 % of far
  links kept, every seed split) instead of opening a hole.
- **N0 is the main knob; the scales barely matter** at a given N0 (relays at 16 km are
  already near their loudest).
- **N0 = 0.5** loses 15 % of links (D02: 13 %), keeps 96 % far away, keeps half the links
  inside the zone (a squad there stays partly connected, as the plan wants), and still splits
  38 / 100 seeds (D02: 37), so M8 has positive examples.

**Departures from Project_Plan.** N0 = 0.5 instead of 0.05. The plan's range table was
written for 5 / 9 km; with D01's 2.5 / 16 km the numbers differ (see the jamming
playground's "How far each role is heard"). The plan's 3 km compute range is not adopted:
compute is made the weakest through its scale instead, so D01 stands.

---

## D04. Tasks, the relay rule and the oracle labels (Mission 0, step 3)

```
Chose:     strict role mapping; relay within 2 hops on the live graph for Observe and Track;
           c_ij = km + 3 (1 − battery), a FLAT penalty of 1000 when not able; tasks the
           Hungarian oracle can only staff with a penalised drone stay open (None)
Rejected:  a 3-hop relay rule; penalty added to the travel cost; forcing every task to get
           a drone
Evidence:  cargo test --release -p swarmsim print_ -- --ignored --nocapture  (seeds 0-99)
Cost:      the relay rule is weak (97 % of drones pass), so it barely separates drones
Revisit:   M5 (if the GNN gains nothing over an MLP, tighten the rule), M10 (ground truth)
```

**Tasks** (`spawn_tasks`). 50 per snapshot, 4 kinds. Half are placed within 3 km of a random
drone, the rest uniformly over the arena (rejection keeps them inside).

**Who can take what** (`TaskKind::can_do`, strict): Observe = recon or RF sensing,
Track = recon, Relay = relay, Fuse = compute. **Able** = the role fits and, for Observe and
Track, a relay sits within 2 hops on the live graph (`relay_within_two_hops`, a BFS of
depth 2). 2 hops matches a 2-layer GNN's receptive field (M5).

**Cost** (`cost_matrix`, tasks on rows). c_ij = distance in km + β (1 − battery_i), β = 3, so
battery adds 0 to 1.5 km. A drone that is not able costs a flat 1000. **Flat**, not "travel +
penalty": with travel added, the distance of a drone that cannot do the task decided which
task stayed open, an artefact.

**Oracle labels.** Single winner = cheapest drone for the task alone (row argmin). Joint
winner = the Hungarian assignment over all tasks (`kuhn_munkres_min` on integer metres, each
drone used at most once). A task whose only option is a penalised drone stays open.

**Result over seeds 0–99** (5000 tasks)

| | |
| --- | --- |
| Drones passing the relay rule | 97.3 % on average (min 88 %); hops to a relay 0 / 1 / 2 / more: 2000 / 6523 / 1209 / 268 |
| Tasks with no able drone | 0 |
| Cheapest able cost | median 2.9 km, p90 6.4 km |
| Tasks left open by the joint oracle | 24 (Fuse tasks outnumbering the 15 compute drones on 11 seeds) |
| Single winner shared with another task | 2590 (52 %) |
| Joint winner ≠ single winner | 1765 (35 % of staffed tasks) |
| Total cost, joint vs singles | +19 % |

**Reading it.**
- **The relay rule is weak.** With 16 km relay ranges almost every drone has a relay within
  2 hops. It is kept at 2 hops (the GNN's reach) rather than tuned now; if Mission 5 shows no
  gain from the graph, this is the first knob.
- **The two labels really differ.** Half the single winners are wanted by another task, and
  the joint oracle changes the winner of a third of the tasks. A drone cannot see that from
  one task, which is the point of Mission 5's regret measure.

---

## D05. Label propagation: default graph, default method, which labels to trust (Mission 1)

```
Chose:     propagate on the LIVE graph (the comm graph under jamming); RWR (α = 0.85) as the
           default propagation, LP's closed form as its check; propagate zone and squad,
           never role; compare against nearest-reporter by km when positions are known and
           by hops when they are not; no guess counts as wrong, split seeds kept
Rejected:  kNN (k = 5) as the default graph; the ε graph as the default; propagating role;
           dropping the seeds whose graph is split
Evidence:  cargo run --release -p aerie-classic --example m1   (test seeds 90-99 × 20 draws)
           GUI: swarm viewer, mission M1 (map, per-draw scores, Run, plots)
Cost:      on clean reports, propagation loses to "copy the nearest reporter by km"
Revisit:   M0 step 4 (correlation graph, experiment 1), M4 (features + graph), M6 (spoofed
           reports), whenever the zone label is scored again (balanced accuracy)
```

**Setup.** `aerie_classic::m1`: label propagation and RWR (closed forms; tests check they equal
the iterations), majority class, nearest reporter by km and by hops. Labels zone (2), squad
(10), role (4). Accuracy on the silent drones; a drone no report can reach gets no guess and
counts as wrong. The same draws for every method, so comparisons are paired. A share of the
reporters can say a wrong class (0, 10, 25, 40 %).

**Main table** (live graph, 10 % of drones report, no wrong reports; accuracy %)

| label | majority | nearest km | nearest hops | LP | RWR |
| --- | --- | --- | --- | --- | --- |
| zone | 91.0 | 91.4 | 89.9 | 91.4 | 91.5 |
| squad | 8.1 | **61.0** | 51.6 | 51.2 | 53.0 |
| role | **46.4** | 29.7 | 32.1 | 31.8 | 31.6 |

**Experiment 1, which graph** (LP accuracy against the report share)

| squad | 1 % | 5 % | 10 % | 20 % | 50 % |
| --- | --- | --- | --- | --- | --- |
| live | 9.0 | 31.8 | 51.2 | 75.2 | 93.5 |
| ε (no jamming) | 9.1 | 33.0 | 53.3 | 78.0 | 95.6 |
| kNN, k = 5 | 9.1 | 37.8 | 61.6 | 86.3 | 98.4 |

| zone | 1 % | 5 % | 10 % | 20 % | 50 % |
| --- | --- | --- | --- | --- | --- |
| live | 78.6 | 89.8 | 91.4 | 93.7 | 96.3 |
| ε | 79.0 | 89.8 | 92.2 | 94.5 | 96.9 |
| kNN | 21.1 | 58.4 | 77.7 | 92.2 | 98.2 |

**kNN is rejected as the default.** With k = 5 every drone's neighbours are in its own squad,
so the graph falls into 2–9 islands on the test seeds. That makes it the best graph for squad
once enough drones report (no leaks between squads), and the worst at low report shares: at
1 % reporting, 76 % of the silent drones get no guess. ε is slightly better than live, but it
is the graph without jamming, which the swarm does not have. **Live is the default.**

**Wrong reports** (live graph, 10 % report; accuracy %, in brackets % of lies overruled)

| zone | 0 % wrong | 10 % | 25 % | 40 % |
| --- | --- | --- | --- | --- |
| majority | 91.0 | 91.0 | 90.6 | 89.3 |
| nearest km | 91.4 | 83.9 | 67.4 | 58.8 |
| LP | 91.4 | **87.6** (0) | **75.2** (0) | **62.6** (0) |
| RWR | 91.5 | 87.6 (4) | 74.1 (1) | 62.5 (1) |

| squad | 0 % wrong | 10 % | 25 % | 40 % |
| --- | --- | --- | --- | --- |
| nearest km | 61.0 | 55.3 | 43.6 | 37.7 |
| LP | 51.2 | 46.5 | 37.7 | 32.7 |
| RWR | 53.0 | 48.3 | 39.2 | 34.3 |

**What it means.**
1. **The graph adds little when positions are known and reports are clean.** The ε and live
   graphs are built from distance and squads are clusters in space, so the nearest reporter
   in km already knows what the graph knows, and it never leaks: LP leaks across the ~25 % of
   links between squads and along the 16 km relay links.
2. **Propagation pays off when reports are wrong.** The nearest reporter copies one report,
   lie included; LP averages many and dilutes it. On zone LP leads nearest-by-km by 3.7 points
   at 10 % wrong and 7.8 at 25 %. On squad the gap to nearest-by-km shrinks (8 → 4 points)
   but does not close.
3. **Propagation pays off when positions are unknown.** A silent, jammed drone may not send
   its position. Then the fair baseline is nearest-by-hops, and RWR beats it on squad (53.0
   against 51.6) with no lies at all.
4. **Liars are not corrected.** LP clamps reports (0 % overruled). RWR does not clamp but
   overrules at most 4 %: with 10 % of drones reporting, a liar's neighbours are mostly
   silent, so its own restart term wins. Only a lie surrounded by honest reporters is
   outvoted (lp_playground, "One wrong report"). Propagation protects the drones *around* a
   liar, not the liar's own label.
5. **Role cannot be propagated.** Neighbours have different roles (heterophily); every
   propagation method assumes they agree, and majority wins by 15 points. Role needs features
   (M4 on).
6. **RWR over LP as the default**, by a small margin: equal or better on every label at 10 %,
   and it does not clamp. α matters: at α = 0.9 its squad accuracy fell from 53.0 to 51.4.
   The α sweep (experiment 4) is still to do.
7. **The zone numbers flatter majority.** ~90 % of drones are outside the zone, so "everyone is
   outside" scores 91 % and finds no jammed drone. Plain accuracy is the plan's metric, but
   later missions should score zone with balanced accuracy (or the recall on "jammed").

**Split seeds.** D01 said to exclude seeds whose graph is split from M1. Amended: label
propagation now handles drones no report can reach (identity row in the closed form, zero
row = no guess), so all 10 test seeds are kept and the no-guess share is reported next to
the accuracy (1 % of the silent drones on the live graph at 10 % reporting).

**Not done in Mission 1** (deferred, on purpose):
- Experiment 1 on the **correlation graph**: needs Mission 0 step 4 (time series).
- Experiment 3, **biased reports** (only drones at the zone edge report): not run as an
  experiment; the swarm viewer's *Pick by hand* mode lets it be explored on one seed. The
  plan's second question ("which labels can be trusted when the reports are not a random
  sample") is therefore still open.
- Experiment 4, **the silent squad**: seen qualitatively (lp_playground preset: LP and RWR
  both give the cut-off squad its neighbour's class); the α sweep and round counts are not
  done.
- The **exam drill** (LP by hand, RWR convergence, inductive vs transductive, task axes).
