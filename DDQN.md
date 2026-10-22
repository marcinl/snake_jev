# Double DQN for Snake

A primer on Double DQN (DDQN) and how it fits into `snake_jev.py` as the second layer of the hybrid controller (d3-edge → DDQN → search).

> **Note:** this design is implemented in numpy in `ddqn.py` (train with `python ddqn.py`), and `snake_jev.py --ddqn ddqn_weights.npz` plays the hybrid controller. `test_ddqn.py` checks the examples in sections 3, 4 and 7. Apart from those, the numbers below are hand-worked or illustrative, chosen to show the mechanics. They are not measured results; see the README for those.

The document follows one concrete move through every stage: the board it starts from (state `s`), the move chosen (action `a`), the reward `r`, the board it leads to (`s'`), and how the networks learn from it.

## 1. The vocabulary

Reinforcement learning describes the game as a loop between an **agent** (the snake's brain) and an **environment** (the board):

| Term | Meaning in Snake |
|---|---|
| **State `s`** | What the agent sees before a move: a list of numbers describing the board around the head (section 4) |
| **Action `a`** | The move it picks: `left`, `straight` or `right`, relative to its heading (section 3) |
| **Reward `r`** | A number the game hands back after the move: +10 apple, −10 death, −0.01 otherwise (section 5) |
| **Next state `s'`** | The board after the move |
| **Episode** | One game, from start to death, a full board, or starvation |
| **Transition** | One recorded step `(s, a, r, s', done)`; `done` says whether the episode ended |
| **Discount `γ`** | How much a reward one step later is worth compared with one now; we use `γ = 0.9` |

The agent's goal is to maximise the **return**: the sum of all future rewards, each discounted by how far away it is:

```
return = r₁ + γ·r₂ + γ²·r₃ + …
```

With `γ = 0.9`, an apple 1 step away is worth `10 · 0.9 = 9.0` now, 5 steps away `10 · 0.9⁵ ≈ 5.9`, and 20 steps away `10 · 0.9²⁰ ≈ 1.2`. Nearer apples are worth more, which is what makes the agent hurry.

## 2. Q-learning, DQN and Double DQN

### Q-values

A **Q-value** `Q(s, a)` is the agent's estimate of the return if it takes action `a` in state `s` and plays well afterwards. With three actions, each state has three Q-values, and the agent plays the largest:

```
Q(s, left) = 3.1    Q(s, straight) = 1.4    Q(s, right) = 0.7    →  play left
```

Learning means making these estimates accurate. After each move, the estimate for the move just played is nudged toward a **target** built from what actually happened:

```
target = r + γ · value(s')        value(s') = how good the next state is
loss   = (Q(s, a) − target)²
```

If the move ended the game (`done = True`), there is no future, and `target = r`.

### DQN: a neural network instead of a table

Snake has far too many board positions to store a Q-value for each one. **DQN** (Deep Q-Network) uses a small neural network: the input is the state's feature vector, and the output is one Q-value per action. Similar positions produce similar features, so what the network learns in one position carries over to positions it has never seen.

DQN adds two stabilisers, without which training tends to diverge:

- **Replay buffer.** Every transition is stored in a buffer (e.g. the last 50,000). Training samples random batches from it instead of learning only from the latest move. Consecutive moves are highly correlated; random sampling breaks that up, and each transition is reused many times.
- **Target network.** A second copy of the network, `Q_target`, computes `value(s')`. It is frozen and only refreshed from the learning network, `Q_online`, every K steps. Without it, every update to `Q_online` would also shift the targets it is chasing.

Plain DQN uses `value(s') = max over a' of Q_target(s', a')`.

### What "Double" fixes

Taking the max over noisy estimates is biased upwards. If three moves are really worth 2.5 each, but the estimates have errors of ±0.4, the max will usually land on whichever error happened to be positive, at around 2.9. Repeated across millions of updates, this makes the agent believe some moves are better than they are. In Snake, that typically shows up as confident circling, or heading into pockets the network has overrated.

Double DQN splits the job between the two networks:

- the **online** network **chooses** the next action: `a* = argmax over a' of Q_online(s', a')`
- the **target** network **scores** it: `value(s') = Q_target(s', a*)`

A noisy overestimate in one network is unlikely to be repeated in the other, so a move only gets a high value if both networks rate it well. Section 7 works through a full example.

## 3. Actions: relative, not absolute

The DDQN chooses among **left / straight / right**, relative to where the snake is heading. These are converted to the game's absolute moves before `Game.step` is called:

| Heading | `left` | `straight` | `right` |
|---|---|---|---|
| UP | LEFT | UP | RIGHT |
| RIGHT | UP | RIGHT | DOWN |
| DOWN | RIGHT | DOWN | LEFT |
| LEFT | DOWN | LEFT | UP |

Why relative:

- **Reversing is impossible by construction.** With four absolute actions, one would always be a wasted no-op.
- **One policy instead of four.** "Turn away from the wall ahead" is the same decision whichever way the snake faces. With absolute actions, the network would have to learn it separately for each heading.

## 4. The state `s`: 17 numbers

Before each move, the board is converted into 17 numbers. All of them are measured relative to the head, and none depends on the board size, so a network trained on a 10×8 board can also play on 20×15.

| # | Group | Features | Values |
|---|---|---|---|
| 0–2 | Immediate danger | wall or body directly `straight`, `left`, `right` (i.e. `is_safe` is false) | 0 / 1 |
| 3–6 | Heading | one-hot UP, DOWN, LEFT, RIGHT | 0 / 1 |
| 7–10 | Food direction | food is up, down, left, right of the head (two can be 1 at once) | 0 / 1 |
| 11–13 | Tail reachable | `Game.tail_reachable` after `straight`, `left`, `right` | 0 / 1 |
| 14–16 | Free space | `Game.free_space` after each action ÷ the snake's length, capped at 1.0 | 0.0 – 1.0 |

The first three groups are the classic Snake DQN state. The last two reuse the step 1 safety helpers. They let the network see that a move leads into a small region before the snake is actually trapped. Without them, a DQN can only learn about traps from the deaths they cause, which is slow and unreliable.

### Example state `s`

```
#########
#.......#   row 0
#..F....#   row 1     food F at (2,1)
#...TSH.#   row 2     head H (5,2), body S (4,2), tail T (3,2), heading RIGHT
#.......#   row 3
#.......#   row 4
#########             7 × 5 board, 35 cells (columns 0–6)
```

Heading RIGHT, so `straight` = RIGHT → (6,2), `left` = UP → (5,1), `right` = DOWN → (5,3). All three cells are empty. The food is 1 row up and 3 columns left of the head. The snake is 3 cells long. After any of the three moves, the head can reach 33 cells, far more than its length of 3, so all three free-space values are capped at 1.0.

**Why divide by the snake's length rather than the board size?** What matters is whether the body fits in the region the head is entering. A region at least as large as the snake (1.0) is roomy; one half its size (0.5) is a likely trap. Those thresholds mean the same thing on every board. A fraction of the board would not: a 5-cell pocket is 6% of a 10×8 board but under 2% of a 20×15 one.

### One network for every board size

The network never sees the grid itself, only these 17 numbers, and all of them are relative to the head and independent of the board size. The same trained weights load and play on any N × M board. To make sure it has met the situations each size produces, training picks a random board between 8×6 and 20×15 for each episode.

```
s = [ 0, 0, 0,          danger: straight, left, right
      0, 0, 0, 1,       heading: UP, DOWN, LEFT, RIGHT
      1, 0, 1, 0,       food: up, down, left, right
      1, 1, 1,          tail reachable: straight, left, right
      1.0, 1.0, 1.0 ]   free space: straight, left, right
```

Nothing is dangerous here, so the only features that separate the moves are the food features. A trained network should prefer `left` (UP), which turns toward the food.

## 5. Rewards

| Event | Reward | Episode ends? |
|---|---|---|
| Eat an apple | **+10** | no (yes if it fills the board: a win) |
| Hit a wall or the body | **−10** | yes |
| Any other move | **−0.01** | no |
| 2·N·M moves without an apple (starvation) | no extra reward | cut off, but treated as *not* terminal (see below) |

### Why these values

**+10 for an apple.** This is the goal. The exact number is arbitrary; what matters is how it compares with the others. Q-values end up roughly in the range −10 to +40, which keeps the network's outputs and gradients at a comfortable scale.

**−10 for dying.** Death has to cost roughly as much as an apple is worth, otherwise the network learns to gamble.
- Much smaller (e.g. −1): a risky dash for an apple is "worth it" even if it often kills the snake.
- Much larger (e.g. −100): the agent becomes so afraid of dying that it circles in open space forever and never goes for food near walls or its body.
- At −10, losing the game cancels roughly one apple's worth of reward, so the agent goes for food when it can, but not at any price.

**−0.01 per step.** A small push against wasting time. Without it, an agent that circles forever loses nothing, and circling is exactly the failure d3-edge shows today. It is deliberately tiny:
- One step costs 1/1000 of an apple, so the agent never prefers death to a long detour.
- Over a full starvation run on a 10×8 board (160 steps), the total is `160 × −0.01 = −1.6`. That is a real cost, but much smaller than one apple (+10) or dying (−10).
- With discounting, a whole infinite future of step costs is worth at most `−0.01 / (1 − 0.9) = −0.1`. So the step cost tips close decisions toward shorter routes without overriding anything that matters.

**Starvation is a cut-off, not a death.** When an episode hits the 2·N·M limit, the transition is stored with `done = False`. The snake didn't do anything fatal on that move; we simply stopped watching. Marking it terminal would teach the network that ordinary safe moves are suddenly worth nothing, which is wrong and makes learning noisier.

**What we deliberately leave out (for now): a reward for getting closer to the food.** Many Snake tutorials add something like +0.1 for each step toward the apple. It speeds up early learning, but it rewards exactly the greedy behaviour that traps long snakes, working against the tail-reachability features. If training turns out to be slow, the safe version is *potential-based* shaping, which provably doesn't change which policy is best. It isn't needed for the first version.

### Example: the reward over one episode

An episode lasting 120 moves, eating 3 apples and then hitting a wall:

```
3 apples          3 × (+10)    = +30.00
1 death           1 × (−10)    = −10.00
116 other moves   116 × (−0.01) = −1.16
total (undiscounted)            = +18.84
```

The training log will report this undiscounted total per episode, which is the easiest number to watch going up.

## 6. Networks: shape and initialisation

### Shape

Both networks have the same shape: a small multi-layer perceptron (MLP).

```
input (17) → dense 64 + ReLU → dense 64 + ReLU → dense 3 (linear)
              Q(s, left)   Q(s, straight)   Q(s, right)
```

That is `17·64+64 + 64·64+64 + 64·3+3 = 5,507` weights, about 22 KB as float32. The output layer is linear (no activation), because Q-values can be negative.

### How `Q_online` is initialised

`Q_online` starts with **random weights**, using the standard scheme for ReLU networks:

- **Hidden layers:** He initialisation, `W ~ Normal(0, √(2 / fan_in))`, where `fan_in` is the number of inputs to the layer (17 for the first, 64 for the second). This keeps the size of the signal roughly constant from layer to layer, so it doesn't vanish or blow up.
- **Output layer:** the same, scaled down by ×0.01, so every initial Q-value is close to 0.
- **Biases:** all 0.
- **Seeded** (`numpy.random.default_rng(seed)`), so a training run can be repeated exactly.

Starting near 0 means the untrained network has **no opinion**: for the example state `s`, it might output something like

```
Q_online(s) ≈ [ 0.004, −0.002, 0.001 ]     left, straight, right
```

That's fine, because at the start of training the agent's moves are chosen at random anyway (ε-greedy, below). The near-zero start also doesn't bias the agent toward or away from any move before it has seen a single reward.

### How `Q_target` is initialised and kept up to date

`Q_target` starts as an **exact copy** of `Q_online`: same weights, same biases. The two are identical at step 0 and only drift apart as `Q_online` trains.

Every **K = 1,000 training steps**, the weights are copied across again (`Q_target ← Q_online`). Between copies, `Q_target` is frozen: it receives no gradient updates.

- Smaller K means targets follow the learning network more closely, which is faster but less stable.
- Larger K is more stable but slower to propagate what has been learned.

1,000 is a common starting point for a network this size. (An alternative is a *soft* update every step, `Q_target ← 0.995·Q_target + 0.005·Q_online`. It has a similar effect and can be tried if hard copies cause training to wobble.)

### Exploration: ε-greedy

If the agent always played its current best guess, it would never try anything new. With probability ε it plays a random action, otherwise its highest Q-value. ε starts at 1.0 (fully random) and decays to 0.05 over the first ~60% of training. Early episodes explore and fill the replay buffer; later ones mostly exploit what has been learned. Training only starts once the buffer holds at least 1,000 transitions, so the first batches aren't drawn from a handful of moves.

## 7. Worked example: one full transition

### The move

The agent is in the example state `s` from section 4 and plays **`a = left`** (UP). The head moves from (5,2) to (5,1). No apple, no collision, so **`r = −0.01`** and **`done = False`**.

```
before (s)               after (s')
#########                #########
#.......#                #.......#
#..F....#                #..F..H.#     head (5,1), heading UP
#...TSH.#                #....TS.#     body (5,2), tail (4,2)
#.......#                #.......#
#.......#                #.......#
#########                #########
```

In `s'` the snake is heading UP, so `straight` = UP, `left` = LEFT and `right` = RIGHT (from the table in section 3). The food is now level with the head and 3 columns left. All moves are safe and keep everything connected:

```
s' = [ 0, 0, 0,   1, 0, 0, 0,   0, 0, 1, 0,   1, 1, 1,   1.0, 1.0, 1.0 ]
```

The transition `(s, left, −0.01, s', False)` goes into the replay buffer. Later, a training batch samples it.

### Computing the target

Partway through training, the two networks give these Q-values for `s'`:

| Action in s' | Absolute move | Q_online(s') | Q_target(s') |
|---|---|---|---|
| straight | UP | 2.1 | 2.3 |
| left | LEFT (toward food) | **3.4** | 2.6 |
| right | RIGHT (away from food) | 1.0 | **2.9** |

```
Plain DQN:   value(s') = max(2.3, 2.6, 2.9) = 2.9
             target    = −0.01 + 0.9 · 2.9 = 2.60

Double DQN:  a*        = argmax Q_online(s') = left   (3.4)
             value(s') = Q_target(s', left)   = 2.6
             target    = −0.01 + 0.9 · 2.6 = 2.33
```

Plain DQN takes `Q_target`'s single optimistic estimate for `right`, even though that move leads away from the food, and the online network rates it the worst of the three. Double DQN only trusts `left`, the move the online network actually prefers, and scores it with the target network's more modest 2.6.

### The update

Suppose `Q_online(s, left)` is currently 2.0. The error is `2.0 − 2.33 = −0.33`, so the loss for this sample is `0.33² ≈ 0.11`. One Adam step (learning rate ~0.001), averaged over the batch of 64, moves `Q_online(s, left)` slightly upward toward 2.33.

Only the output for the action actually played (`left`) gets a gradient. The Q-values for `straight` and `right` in `s` aren't pushed toward anything from this sample, because we don't know what would have happened had they been played.

### The same state with different outcomes

The target depends on what the move led to. For the same `s` with a different `s'`:

| Outcome of the move | r | done | target |
|---|---|---|---|
| Empty cell (above) | −0.01 | no | −0.01 + 0.9 · Q_target(s', a*) = 2.33 |
| Ate the apple, Q_target(s', a*) = 1.5 | +10 | no | 10 + 0.9 · 1.5 = 11.35 |
| Hit a wall | −10 | yes | −10 (no future term) |
| Starvation cut-off on this move | −0.01 | stored as no | same as an empty cell |

## 8. Training loop (offline, no Ollama)

```
Q_online ← random weights (seeded);  Q_target ← copy of Q_online
replay   ← empty buffer (capacity 50,000);  ε ← 1.0

for each episode:
    reset game with a fresh seed;  s ← features(game)
    while not done and not starved:
        a = random action with probability ε, else argmax Q_online(s)
        play a  →  r, s', done
        replay.add(s, a, r, s', done)
        if len(replay) ≥ 1,000:
            batch = 64 random transitions from replay
            a*     = argmax Q_online(s'_batch)            # online network chooses
            target = r + γ · Q_target(s'_batch, a*) · (1 − done)   # target network scores
            one Adam step on (Q_online(s_batch, a_batch) − target)²
        every K = 1,000 steps:  Q_target ← Q_online
        s ← s'
    decay ε;  log episode reward and score
save Q_online → ddqn_weights.npz
```

Only `Q_online` is saved. At play time, there's no exploration and no target network: the game computes `s`, runs `Q_online`, and plays the highest-scoring action that passes the tail check.

## 9. Where it sits in the hybrid controller

The DDQN is not trusted on its own. Like d3-edge, its move is accepted only if the tail stays reachable afterwards:

```
1. d3-edge   – if the route since the last apple is new and its move keeps the tail reachable
2. DDQN      – else, its highest-Q move that keeps the tail reachable
3. search    – else, BFS to the food (or follow the tail) until the next apple
```

The DDQN is deterministic as well, so it could circle just like d3-edge. To stop that, its move is also rejected if it would put the head back on a (cell, heading) the head has already had since the last apple. When all three of its moves fail these checks, search plays.

### Example decision trace (illustrative output format)

```
step  head   d3-edge  ddqn        played  source  reason
 41   (5,2)  UP       left/UP     UP      model   new route, tail reachable
 57   (8,6)  LEFT     right/DOWN  DOWN    ddqn    route repeated at (8,6) RIGHT
 88   (2,3)  DOWN     left/RIGHT  UP      search  DOWN, RIGHT: tail unreachable; BFS to food (length 7)
...
score=14 decisions=212 sources: model=61% ddqn=27% search=12%
```

The source shares in the last line are the key diagnostic. If `model` falls too low, d3-edge isn't really playing the game and the prompt needs more work.

## 10. Known limitations

- **Tail reachability is a rule of thumb, not a guarantee.** On a nearly full board the snake can still trap itself, and search with the starvation limit is the final backstop.
- **The DDQN only sees local features.** It's good at "don't box yourself in" and "head roughly toward the food", but poor at long-range planning, which is why search sits underneath it.
- **The hyperparameters (γ, K, ε schedule, reward sizes, layer widths) are standard starting points, not tuned values.** If training is unstable, check them first.
- **The source article reports a single game (seed 7).** The benchmark mode will compare model-only, DDQN-only, search-only and hybrid over many seeds before we draw conclusions.
