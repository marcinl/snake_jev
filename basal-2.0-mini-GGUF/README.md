---
license: apache-2.0
language:
- pl
- en
base_model: speakleash/Bielik-1.5B-v3.0-Instruct
library_name: gguf
pipeline_tag: text-generation
tags:
- typed-decisions
- decision-making
- classification
- calibration
- polish
- gguf
---

# basal-2.0-mini-GGUF

[![GitHub](https://img.shields.io/badge/GitHub-rkinas%2Fbasal%20%E2%80%94%20inference%20engine-black?logo=github)](https://github.com/rkinas/basal)
[![Website](https://img.shields.io/badge/web-basal.si5.pl-0a7d5a)](https://basal.si5.pl/)
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97-basal--2.0%20collection-yellow)](https://huggingface.co/collections/Remek/basal-20-6acd1712a8e443c86cd576fc)

GGUF files (Q8_0 and Q5_K_M) for Ollama and llama.cpp of [basal-2.0-mini](https://huggingface.co/Remek/basal-2.0-mini) (1.5B, base Bielik-1.5B-v3.0-Instruct). Same model, same typed-decision API through `basal-serve`; the calibration files this folder ships are listed below.

## Measured against bf16

**Agreement** is the share of decisions where the port picks the same option as the bf16 release model on the development set dev v2 (MLX rows: every 2nd dev v2 decision, 1,700 decisions). Bars for this tier: agreement 2-10 >= 0.9161, agreement at reference margin >= 0.2 >= 0.9858, accuracy within 1 point of bf16, ECE at most +0.01. All ports of this tier:

| package | agreement with bf16 (options 2-10) | agreement at margin >= 0.2 | accuracy diff | ECE diff | size | speed | gate |
|---|---|---|---|---|---|---|---|
| [FP8](https://huggingface.co/Remek/basal-2.0-mini-FP8) | 0.9333 | 0.9937 | -0.0077 | -0.0032 | 1.8 GB | 103.5 dec/s c16, c1 p50 14 ms (H100) | PASS |
| [GGUF Q8_0](https://huggingface.co/Remek/basal-2.0-mini-GGUF) | 0.9769 | 1.0000 | -0.0004 | -0.0019 | 1.6 GB | not measured in this release | PASS |
| [GGUF Q5_K_M](https://huggingface.co/Remek/basal-2.0-mini-GGUF) | 0.9339 | 0.9931 | -0.0006 | -0.0016 | 1.1 GB | not measured in this release | PASS |
| [MLX 8-bit](https://huggingface.co/Remek/basal-2.0-mini-MLX-8bit) | 0.9792 | 1.0000 | -0.0025 | +0.0024 | 1.7 GB | 93 ms c1 p50, 223 ms for 5 questions (Mac) | PASS |

The ports are decision-layer checks against bf16, not new benchmarks; the model's own results are on [basal-2.0-mini](https://huggingface.co/Remek/basal-2.0-mini).

## Use

Files: `basal-2.0-mini-Q8_0.gguf` (recommended), `basal-2.0-mini-Q5_K_M.gguf` (smaller), `Modelfile.Q8_0`, `Modelfile.Q5_K_M`, `basal.json` and the calibration files below. The basal server runs in front of Ollama or llama.cpp: it renders basal's own prompt and reads the log-probabilities of the option codes. Ollama 0.12.11 or newer:

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install "basal @ https://github.com/rkinas/basal/archive/refs/tags/v2.0.0.tar.gz"
hf download Remek/basal-2.0-mini-GGUF --local-dir basal-2.0-mini-GGUF
cd basal-2.0-mini-GGUF && ollama create basal-2.0-mini:q8_0 -f Modelfile.Q8_0 && cd ..
basal-serve --model ./basal-2.0-mini-GGUF --mode ollama --ollama-model basal-2.0-mini:q8_0 --port 8000
```

`--model` points at the downloaded folder for the tokenizer, chat template and calibration files; Ollama serves the weights. Ollama returns at most 20 log-probabilities per position, so questions with more than 20 options need llama.cpp:

```bash
llama-server -m basal-2.0-mini-GGUF/basal-2.0-mini-Q8_0.gguf -c 4096 --port 8080
basal-serve --model ./basal-2.0-mini-GGUF --mode llamacpp --llamacpp-url http://127.0.0.1:8080 --port 8000
```

Without a separate server, `--mode gguf` loads the file in process (`basal[gguf]`, llama-cpp-python): `basal-serve --model ./basal-2.0-mini-GGUF --mode gguf --gguf basal-2.0-mini-GGUF/basal-2.0-mini-Q8_0.gguf --port 8000`. A plain `ollama run` gives free text only: the typed answers come from `basal-serve` in front of it. `--http-parallel N` (with `OLLAMA_NUM_PARALLEL` / `llama-server -np`) serves concurrent clients.

**Context.** Ollama mode reads at most 4,096 tokens per prompt (basal-serve sends `num_ctx 4096`, which overrides the Modelfile), and llama.cpp mode reads what `llama-server -c` allows; longer states are cut in the middle (`usage.truncated`). For long documents use `llama-server -c 8192` or `--mode gguf`.

Speed: not measured in this release. GGUF is the slow path for batches: it is for desktops and laptops, not servers.

**Engine.** The same basal engine v2.0.0 also serves the experimental basal-omni-1.0 models (images and audio): [README](https://github.com/rkinas/basal#images-and-audio-basal-omni-experimental).

## Calibration

`CALIBRATION.json` in this folder is the bf16 model's calibration, the default for every mode without a fit of its own. basal-serve `--mode ollama` and `--mode llamacpp` load `CALIBRATION.<engine>.json` for their engine when the folder has it, else `CALIBRATION.json`. `--mode gguf` (in process) loads `CALIBRATION.json`, the bf16 fit; pass `--calibration <folder>/CALIBRATION.llamacpp.json` to use the llama.cpp fit. A calibration fitted on bf16 readouts does not carry over exactly to a quantised port (an outside test of
basal-1.5 on Ollama saw 2.3% real error at the 1% threshold with the bf16 file), so a port-specific fit, where it passed,
uses the bf16 model's method on that port's own readouts of the same calibration splits, one option order:

- `basal-serve --mode ollama`: Ships `CALIBRATION.ollama.json`, fitted on Ollama (GGUF) readouts of this port (same method as the bf16 fit, one option order); dev v2 ECE 0.0376 with the bf16 fit -> 0.0358 with this fit (in-sample), 0.0383 held-out (leave-one-family-out).
- `basal-serve --mode llamacpp`: Ships `CALIBRATION.llamacpp.json`, fitted on llama.cpp (GGUF) readouts of this port (same method as the bf16 fit, one option order); dev v2 ECE 0.0333 with the bf16 fit -> 0.0362 with this fit (in-sample), 0.0393 held-out (leave-one-family-out).

Where a port uses the bf16 calibration, treat its confidence thresholds as approximate and fit your own with
`basal-calibrate` when they matter.

## Results of the model

See the model card [basal-2.0-mini](https://huggingface.co/Remek/basal-2.0-mini) for Werdykt, JDI, LEK, PL-Holdout and PL-Real numbers against basal-1.5 and the evaluation notes. This port is not scored separately on those benchmarks.

## Limitations

- Validate on your own documents before relying on it: hidden and held-out sets do not cover every domain.
- Polish world knowledge of a small model is limited: put the relevant facts in the state. Legal rules change.
- Exact arithmetic and thresholds are a weak spot of one-pass decisions: compute them in code (or `"facts": "auto"`).
- The calibration is cautious on unseen tasks: no 1% threshold is certified in the shipped files, so `"max_error": 0.01`
  on an `act` or `tool` question defers every decision; use 0.05, or certify thresholds on your own data with `basal-calibrate`.
- `act` and `multi` over many labels are experimental. Up to 255 options per question (Ollama: 20); the model's context is
  32,768 tokens (mini: 8,192), but Ollama mode reads at most 4,096 tokens per prompt and llama.cpp mode what `llama-server -c` allows (see Use). The 2.0 models ship no evidence head.
- Decisions with serious consequences for people should be reviewed by a person.

## Citation

```bibtex
@software{kinas2026basal20,
  title   = {basal-2.0: typed-decision models and inference engine for Polish and English},
  author  = {Kinas, Remigiusz},
  year    = {2026},
  version = {2.0.0},
  url     = {https://github.com/rkinas/basal}
}
```

## License and attribution

Apache-2.0. Fine-tuned from [speakleash/Bielik-1.5B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct) (Apache-2.0). Bielik is a project of SpeakLeash and ACK Cyfronet AGH.
