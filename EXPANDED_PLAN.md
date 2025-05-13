Here’s a “next-stage roadmap” that keeps the core of your project intact but aims to pull richer behaviour out of the same \~1 kLoC code base.

---

### 1 — Determinism first (the bedrock of good evolution)

* **Replace ad-hoc randomness with a single `rng`.**

  * `AgentBody.get_effective_cooldown_time()` calls `random.uniform(…)` from Python’s global RNG, which is never seeded, so two identical runs of the same genome diverge immediately .
  * Pass `rng` down from `EvolutionOrchestrator` (already has `self.rng`) into every arena step or make a small utility `rand.py` that exposes one shared `np.random.default_rng(seed)` instance.
* **Log the seed with every checkpoint** so you can replay a whole training run exactly.

---

### 2 — Tighten the fitness signal

| Problem                                                               | Quick fix                                                                                          | Stretch                                                                |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Fitness uses coarse ±1 win/loss plus HP bonus, giving flat gradients  | Add **damage-dealt** and **time-alive** terms (e.g. `+0.01*damage` and `+0.001*ticks_survived`)    | Multi-objective (NSGA-II) or **novelty search** to promote exploration |
| Agents learn to **camp** at spawn points                              | Randomise start positions & headings each match (still deterministic by seeding)                   | Curriculum: start with large HP, reduce over generations               |
| Evaluation noise                                                      | Evaluate each genome against *K* fixed “benchmark” genomes from earlier gens before computing rank | Elo or TrueSkill ladder that is updated online                         |

---

### 3 — Give the brains something to chew on

* **Richer perception:** swap the high-level “nearest enemy” inputs for a 12-ray lidar (distances only). The NN now has to **build its own notion of enemy vs wall** instead of receiving it for free.
* **Continuous outputs:** instead of sign-tests, map `tanh` outputs directly to thrust (-1..1), strafe, rotation velocity and trigger-probability (fire when > 0.5). That lets networks learn *how much* to rotate, not just left vs right.
* **Add memory with two lines of code:** change `TinyNet` to `hidden_size=16`, `output_size=4+hidden_size`, feed the last 16 outputs back as recurrent inputs (simple Elman RNN). Cheap, but often enough for flanking or bait-and-switch tricks.

---

### 4 — Evolve smarter, faster

1. **Self-adaptive mutation σ**
   Maintain a per-genome `sigma` and mutate it with log-normal noise (`sigma *= exp(τ·N(0,1))`). Keeps exploration alive without global hand-tuning.
2. **Tournament selection** instead of full sorting: pick 5 genomes at random, keep the best, repeat until new pop is full. Preserves diversity and scales to larger pops.
3. **Vectorised headless arena**
   Right now every agent runs Python loops. Re-implement `Arena.update()` in NumPy (positions as `(N,2)` arrays, cosine/sine look-ups vectorised). On a laptop this gives 10-50× more frames per second, letting you run thousands of generations overnight.

---

### 5 — Game-play tweaks that unlock emergent tactics

* **Ammo or energy**: firing consumes 5 units; regenerates when not shooting. Forces timing decisions.
* **Obstacles**: sprinkle static rectangles; they break line-of-sight and promote flanking.
* **Pick-ups**: a single health pack in the middle means agents must weigh aggression vs retreat.
* **True team mode**: you already compute ally bearings . Flip the switch by spawning two 3-man teams and setting *shared* fitness = team’s mean.

Watch agents invent “charger & sniper” roles around gen 150!

---

### 6 — Tooling & UX

* **Replay viewer** is half-finished in `storage/replay.py` .  Add a CLI flag `main.py replay ./storage/replays/xyz.jsonl` that streams the JSONL back through `Viewer`.
* **Live charts**: write the CSV you already print to `evo/stats.csv`, then add a tiny Matplotlib live-reload script so your son can see peaks and plateaus as training runs.
* **Crash-safe checkpoints**: after each generation, `persist.save_genome()` already runs .  Also pickle the whole `EvolutionOrchestrator` (population + rng state). Resume with `--resume path/to/checkpoint.pkl`.

---

### 7 — Suggested two-week timeline

| Day   | Dev focus                                  | Play focus                                      |
| ----- | ------------------------------------------ | ----------------------------------------------- |
| 1-2   | Deterministic RNG refactor; seed logging   | Verify identical replay hashes                  |
| 3-4   | Fitness refactor + start-pos randomisation | Run 100 gens; inspect heat-maps of movement     |
| 5-6   | Continuous outputs; adjust action scaling  | Manual duels vs gen 0 & gen 200                 |
| 7-9   | Vectorised arena; benchmark fps            | Let it run overnight to 2 000 gens              |
| 10-12 | Obstacles + ammo; update perception rays   | Observe new tactics; clip replays for YouTube   |
| 13-14 | Self-adaptive σ and tournament selection   | Long run; evaluate against best previous season |

---

### 8 — Future gems (rainy-weekend upgrades)

* **Switch the network to JAX** – same maths, GPU-accelerated, still pure NumPy API.
* **NEAT-Py port** – topology evolution unlocks attention-like skip connections without back-prop.
* **Online league** – small FastAPI server that accepts `.npz` uploads and auto-ranks them in nightly round-robins.

---

Tackle the items roughly in this order: determinism → fitness shaping → perception/output richness → evolutionary plumbing → environment variety. Each layer you add makes the next one more fruitful. Happy gladiator-breeding!
