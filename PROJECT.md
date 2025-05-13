# aiarena


Below is a lean, weekend-sized design you and your son can open in an editor tonight, yet it leaves plenty of runway for bigger ideas later.

---

## 1 — The 10 000-foot architecture

| Layer | What it does | Typical file(s) |
| -------------- | -------------------------------------------------------------- | --------------------- |
| **`arena/`** | The 2-D world: coordinates, walls, hit-logic, game clock. | `arena.py` |
| **`agents/`** | *Brains* (tiny NNs) + bodies (HP, speed, weapon cooldown). | `brain.py`, `body.py` |
| **`evolve/`** | Population management, tournaments, mutation, logging. | `evo.py` |
| **`ui/`** | Optional PyGame visualizer + keyboard “spectator” controls. | `viewer.py` |
| **`storage/`** | Read/write genomes & metadata (JSON/NumPy). | `persist.py` |
| **`main.py`** | CLI entry-point: `train`, `show`, `match old_vs_new`, `ffa`, … | `main.py` |

*Everything else is glue code and tests; the whole tree is \~400 LoC on day 1.*

---

## 2 — The arena: keep it rectangular & continuous

* **Size & units:** 800 × 800 “meters”.
* **Physics:** no acceleration—just *teleport-step* position = position + velocity·Δt.
* **Walls:** simple bounce-back and 10 % speed loss.
* **Weapons:** 120° cone in front; max range = 80; cooldown = 0.6 s; damage = 10 HP.
* **Match end:** last team standing or 60 s timeout (draw).
* **Time step:** fixed 50 Hz (20 ms) → deterministic, replayable.

---

## 3 — Agent I/O spec (a.k.a. the “gamepad” for the NN)

**Inputs** (all normalised to \[-1, 1]) – 14 numbers total:

1. Forward distance to nearest **enemy** (±1 if none in sight)
2. Bearing to that enemy (sin θ, cos θ)
3. Forward distance to nearest **ally** (or ±1)
4. Bearing to that ally (sin θ, cos θ)
5. Own health / 100
6. Weapon ready? (1 = yes, –1 = cooling)
7. x-velocity, y-velocity (clamped)
8. Bias neuron = 1

**Outputs** – 4 real values, each thresholded:

| NN output | ≥ 0 → action | < 0 → action |
| --------- | -------------- | ------------ |
| o₁ | thrust forward | thrust back |
| o₂ | strafe left | strafe right |
| o₃ | rotate left | rotate right |
| o₄ | fire | hold fire |

---

## 4 — A “fits-in-your-head” neural net

```python
# brain.py
import numpy as np

class TinyNet:
def __init__(self, w_in=None, w_out=None):
self.w_in = w_in if w_in is not None else np.random.uniform(-1, 1, (16, 14))
self.w_out = w_out if w_out is not None else np.random.uniform(-1, 1, (4, 16))

def __call__(self, x):
h = np.tanh(self.w_in @ x) # hidden layer (16 neurons)
y = np.tanh(self.w_out @ h) # outputs in (-1, 1)
return y
```

*14 × 16 + 16 × 4 = 320 parameters → tiny genome, fast to mutate.*

---

## 5 — Evolution loop (µ + λ, tournament style)

1. **Population** = 64 genomes.
2. **Tournaments**: each genome plays four 1 v 1 matches versus random opponents; fitness = Σ wins + 0.1 · (HP left – HP enemy).
3. **Selection**: top 8 elites copy unchanged; remaining 56 produced by:

* **Mutation only** (90 %): add `N(0, σ²)` with σ = 0.2.
* **Crossover** (10 %): uniform mix of two parents.
4. **Diversity guard**: if stdev(fitness) < 0.03 → increase σ.
5. **Checkpoint** every 10 generations: write best genome, CSV log.

> **Tip:** run headless `main.py train --iters 2000 --nolimit` to crunch overnight (≈10 M game-steps ≈ 3 min on a laptop).

---

## 6 — Storing & replaying agents

```txt
storage/
├─ g00000_best.npz # w_in, w_out, fitness
├─ g00010_best.npz
└─ match_2025-05-13T10-12-11.jsonl # sequence of arena snapshots
```

* Load any `.npz` as a brain.
* The replay file is a newline-delimited JSON list of `{t, state}` for easy diff-viewing and future plotting.

---

## 7 — Game modes

| Mode | How to run | Extra rules |
| ---------------------- | --------------------------- | -------------------------------- |
| **1 v 1** | `main.py match a.npz b.npz` | — |
| **Free-for-All (≥ 3)** | `--ffa` | no teams, last alive wins |
| **Team 3 v 3** | `--teams 3` | agents see ally bearing/distance |

### Emergent team play?

For team battles you *don’t* hard-code co-op moves. Simply:

* identical input spec (they know ally location),
* **shared team reward** (whole team’s win/loss copied to each member’s fitness).

Natural division-of-labour often appears around generation 100-200: e.g. one chaser, one flanker, one camper.

---

## 8 — Suggested day-by-day roadmap

| Day | Evening goal | Stretch |
| --------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **0 (tonight)** | Initialise repo, draw arena box in PyGame, spawn a *manual* arrow-keys agent to chase a dummy target. | — |
| **1** | Drop in `TinyNet`; evolve 20 generations headless; view fitness curve in matplotlib. | Add JSONL replay & simple playback. |
| **2** | Add 1 v 1 UI; pit “gen 0 vs gen 200”. | Add FFA & 3 v 3 flags; start looking for emergent tactics. |
| **Weekend** | Polish: CLI flags, ZIP checkpoints, README. | Sound effects, HP bar sprites, web leaderboard. |

Total code under 1000 lines; runs on any Python ≥ 3.10 with only **NumPy** and **PyGame**.

---

## 9 — Fun extensions for later

* **Speciation / NEAT-style topology evolution** – bolt-on versus rewrite.
* **Curriculum** – shrink arena, lower HP, then gradually relax to full rules.
* **Vision-only input** – 16-ray lidar, no high-level bearings → richer emergence.
* **Swap to JAX / PyTorch** for GPU training (same API, just replace dot-products).
* **Online ladder** – agents upload genomes, server runs daily round-robin.

---

### Final checklist

* [ ] Arena loop is deterministic → reproducible replays
* [ ] All randomness behind a single `rng = np.random.default_rng(seed)`
* [ ] Evolution parameters configurable via CLI / YAML for easy tinkering
* [ ] Clear docs so your son can tweak constants and immediately “see” impact

Have fun watching your digital gladiators learn to circle-strafe and kite—there’s nothing like the first time an emergent wing-man saves its buddy!