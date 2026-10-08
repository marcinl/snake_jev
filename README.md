# Snake × Jev

Snake played by a Jev-style System One decision model ([TypeSafe](https://typesafe.ai/)) served locally by [Ollama](https://ollama.com/blog/ollama-now-supports-jev-style-decision-models) (0.35+).

On every game tick the game sends the board to Ollama's `/v1/systemone` endpoint as text. It asks one typed question, which direction to move, and plays the answer.

**Status:** the integration works end to end. The default `tev1:0.8b` model plays poorly, as described under [Findings](#findings-with-tev108b). Try `--model nimble` (9B) for stronger play.

## Quick start

Prerequisites: Python 3.10+, Ollama 0.35+ running locally, and a Jev-style model:

```bash
ollama pull tev1:0.8b
```

```bash
pip install -r requirements.txt
```

This installs [pygame-ce](https://pyga.me/), the community fork of pygame; the code still says `import pygame`. Don't install `pygame` and `pygame-ce` together, because both provide the `pygame` package and overwrite each other. If you see `NotImplementedError: font module not available`, run `pip uninstall pygame` and then `pip install pygame-ce`. (pygame 2.6.1 has no Python 3.14 wheel, and building it from source leaves out fonts unless SDL_ttf is installed.)

```bash
python3 snake_jev.py --cols 20 --rows 15
```

| Flag | Default | Purpose |
|---|---|---|
| `--cols` / `--rows` | 20 / 15 | Board size N × M (walls not counted) |
| `--model` | `tev1:0.8b` | Any Jev-style Ollama model, e.g. `nimble`, `tev1` |
| `--host` | `http://localhost:11434` | Ollama server |
| `--no-assist` | off | Send only the raw board, without the hints the game works out |
| `--no-guard` | off | Always play the model's top choice, even if it crashes |
| `--headless` | off | Console only, no PyGame window; prints a summary line |
| `--verbose` | off | Headless: print every frame and decision |
| `--dump` | off | Print the request for the first tick and exit |
| `--fps` | 10 | Upper limit on ticks per second (the model's speed is usually the real limit) |
| `--cell` | 28 | Size of one cell in pixels |
| `--seed` | random | Fixes food placement so runs can be repeated |

In the window, **R** restarts after game over and **Esc** quits.

Run the tests (no Ollama or pygame needed):

```bash
python3 -m unittest -v
```

## How it works

1. **Game logic (no model involved).** An N columns × M rows grid. The snake starts with length 2, with its head at `(N//2, M//2)`, its tail one cell to the left, and heading RIGHT. Food spawns on a random free cell. Reverse moves are ignored, as in classic Snake.
2. **Each game tick.** Build a text description of the board, send one `POST /v1/systemone` request, and get back a typed `choice` answer.
3. **Safety check ("guard").** The answer includes a probability for every move. If the model's top move would crash, the game plays its most likely safe move instead. `--no-guard` turns this off.
4. **Display.** A PyGame window (or `--headless`). The game runs at the model's pace, up to the `--fps` limit. The status bar shows the score, the move played, the model's confidence and the time each decision took.

A game also ends if the snake goes 2·N·M moves without eating. This stops a model that loops forever.

## What gets sent to the model each tick

Jev only accepts text, so every value in `state` is a string. This is the first tick on an 8×6 board (`--dump --cols 8 --rows 6 --seed 1`):

```json
{
  "model": "tev1:0.8b",
  "state": {
    "rules": "Classic Snake. The snake moves one cell per tick. Hitting a wall (#) or its own body ends the game. Eating food (F) grows the snake by one.",
    "board_size": "8 columns x 6 rows (walls not counted)",
    "board": "##########\n#........#\n#F.......#\n#........#\n#...TH...#\n#........#\n#........#\n##########",
    "legend": "H = head, S = body, T = tail, F = food, . = empty, # = wall",
    "heading": "the snake is moving RIGHT",
    "snake_length": "2",
    "head_position": "column 4, row 3 (0-based, row 0 at top)",
    "food_position": "column 0, row 1",
    "food_relative_to_head": "food is 2 row(s) UP and 4 column(s) LEFT from the head",
    "next_cell": "UP: free; DOWN: free; LEFT: not allowed (reverse); RIGHT: free"
  },
  "questions": {
    "move": {
      "type": "choice",
      "instructions": "Which direction should the snake's head move next? Move towards the food and never into a wall or the snake's body.",
      "criteria": {
        "UP": "Move the head one row up",
        "DOWN": "Move the head one row down",
        "LEFT": "Move the head one column left",
        "RIGHT": "Move the head one column right"
      }
    }
  }
}
```

The response looks like this:

```json
{
  "model": "tev1:0.8b",
  "answers": {
    "move": {
      "type": "choice",
      "choice": "UP",
      "probabilities": {"UP": 0.61, "DOWN": 0.12, "LEFT": 0.18, "RIGHT": 0.09},
      "confidence": 0.21
    }
  },
  "usage": {"input_tokens": 512, "output_tokens": 3}
}
```

(The numbers above are illustrative; the format is the real one.)

- **How it scales with N × M:** the `board` field is (N+2)·(M+2) characters, so the input grows with board area. Everything else stays the same size.
- **Hints:** `food_relative_to_head` and `next_cell` are worked out by the game, not the model. Small models are weak at reading positions off an ASCII grid; the hints turn the task into reading a sentence. `--no-assist` removes them.

## Findings with `tev1:0.8b`

Measured on Ollama 0.35.1 with a 10×8 board:

| Setup | Result |
|---|---|
| Guard on, hints on | 0–3 food per game over 5 seeds; mostly loops until the 2·N·M limit; 60–105 ms per decision |
| Guard off | Crashes within 5–7 moves |
| Hints off | Ate nothing in 3 games |
| Moves as straight/left/right (first version) | Stuck in a 2×2 loop, always choosing "right" |
| Moves as UP/DOWN/LEFT/RIGHT (current) | Moved towards the food in 22 of 30 test positions, against 17 of 30 |
| Four yes/no ("Noul") questions per tick instead of one Choice | Always safe (30 of 30), but went towards the food less often (13 vs 16 of 30); not adopted |

UP/DOWN/LEFT/RIGHT likely works better because the answer options then use the same words as the hints. The main limit now is the model's size, not how the question is asked.

## Implementation notes

- **Calls Ollama directly over HTTP, not through `typesafe-sdk`.** The SDK has `TypeSafeClient(base_url=..., model=...)`, but its docs don't say how `base_url` maps onto Ollama's `/v1/systemone` path. Calling the endpoint directly needs no packages other than pygame. To use the SDK, replace the `JevClient` class in `snake_jev.py`.
- **pygame is imported only when the window opens**, so `--headless`, `--dump` and the tests work without it.

## Files

| File | Contents |
|---|---|
| `snake_jev.py` | Game logic, query building, Ollama client, guard, PyGame and headless front ends |
| `test_snake_jev.py` | Unit tests for game rules, request shape and guard behaviour |
| `requirements.txt` | `pygame-ce` |

## References

- [Ollama: Jev-style decision models](https://ollama.com/blog/ollama-now-supports-jev-style-decision-models)
- [TypeSafe](https://typesafe.ai/)
- [TypeSafe docs: state](https://docs.typesafe.ai/concepts/state.md)
- [TypeSafe Python sync client](https://docs.typesafe.ai/sdk/python/api/clients/sync.md)
