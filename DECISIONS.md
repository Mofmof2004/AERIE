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
uses the same list.

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
