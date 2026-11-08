# FraudLM v2 (GPT-2 route): build plan

Owner: Thông (106212681). Historical build plan created Fri 06/11/2026 03:55 (UTC+7).
Current device setup and operational commands are in `DEVICE_GUIDE.md` and `DEVELOPMENT.md`.

Hard deadline: A2 submission Sun 08/11 23:59 (team target 20:00). Report freeze Sat 07/11 20:00.
The A2 core (classifier + clustering) always has priority; this LLM is the HD "additional ML" extra.

---

## 1. Decisions (LOCKED = ask Thông before changing)

| # | Decision | Status |
|---|---|---|
| D1 | Route = Raschka (2024) pipeline: own GPT-2 architecture code (Stage 1) → load OpenAI GPT-2 pretrained weights, **no pretraining** (Stage 2) → **LoRA instruction fine-tuning** (Stage 3). No classifier fine-tuning. | LOCKED |
| D2 | Base model **GPT-2 XL (1558M)**. Fallback **GPT-2 Large (774M)**, then Medium (355M), chosen only by the G2 feasibility gate. Same code, config switch only. | LOCKED (fallback by gate) |
| D3 | Model code is ours in plain PyTorch. `transformers` model classes / `AutoModel` / `peft` are **not** allowed in `src`. `transformers` may be used **only inside a test** to check our logits against a reference. | LOCKED |
| D4 | Tokenizer = `tiktoken` "gpt2" (50,257 tokens, EOS `<|endoftext|>` id 50256). Context 1,024 tokens. | LOCKED |
| D5 | Fine-tuning data = Alpaca-style JSON (Section 4). **The old "no LLM-generated data" rule is dropped (Thông, 06/11).** Use whatever gives the best data; record the origin of every example in its `source` field. | LOCKED |
| D6 | Product: single-turn terminal chatbot that explains a classifier result and answers banking/fraud questions for a **fraud analyst**. Decision support only: never approves, rejects or blocks (OOS07). UI is A3. | LOCKED |
| D7 | Never use `customer_age`, `income`, `employment_status` (or `x1`, `x2`, `month`, `device_fraud_count`) as a reason in any data, prompt or answer. | LOCKED |
| D8 | Name: **FraudLM**. Do not reuse another team's product name, code or data files. | LOCKED |
| D9 | G2 result (06/11, Kaggle T4): **GPT-2 XL**, fp32 base, gradient checkpointing on, micro-batch 4 × accum 2, lr 1e-4, r16/α32, 2 epochs (775 tok/s, peak 10.75 GB, projected 2.18 h). | DECIDED by G2 |
| D10 | F10 age proxy `date_of_birth_distinct_emails_4w`: **kept** as an explanation factor (it is a real model input); reported as a limitation (Spearman −0.456 with age). Decided by the lead under Thông's delegation 06/11; Thông may overturn (rebuild + retrain). | DECIDED (lead) |
| D11 | Kaggle runs require explicit owner approval before publication through the Kaggle CLI. | DECIDED |

---

## 2. Folder layout and ownership (only the owner writes in a folder)

```
llm_ver2/
  PLAN.md, DEVELOPMENT.md, TEAM_LOG.md  development notes and technical history
  environment.yml, README_dev.md      lead
  training_model/
    gpt_model.py                      model-builder
    lora.py                           model-builder
    train_sft.py, sft_data.py         trainer
  utils/
    load_gpt2_weights.py, generate.py model-builder
    prompt.py                         data-builder   (the ONE prompt template; everybody imports it)
    guard.py                          chat-builder
  preprocess_data/                    data-builder   (scripts that build the dataset)
  preprocessed/                       data-builder   (intermediate tables, e.g. ml_scores.csv)
  dataset/                            data-builder   (final Alpaca JSON splits + DATA_CARD notes)
  kaggle/                             trainer        (notebook / kernel script + metadata, no secrets)
  checkpoint/                         trainer        (gitignored weights; pretrained cache under checkpoint/pretrained/)
  evaluation/                         evaluator      (scripts, results JSON/CSV, figures)
  chat.py                             chat-builder
  tests/                              each owner writes tests for its own module (test_<module>.py)
  reviews/                            reviewer       (one review file per gate)
```

---

## 3. Interface contracts (do not change a signature without the lead's OK; log it in TEAM_LOG.md)

**Model (`training_model/gpt_model.py`)**
- `GPT_CONFIGS: dict[str, dict]` keys `"gpt2-small" | "gpt2-medium" | "gpt2-large" | "gpt2-xl"`; each dict has
  `vocab_size=50257, context_length=1024, emb_dim, n_heads, n_layers, drop_rate, qkv_bias=True`.
- `class GPTModel(nn.Module)`: `__init__(cfg: dict)`, `forward(idx: LongTensor[B,T]) -> logits FloatTensor[B,T,V]`.
  Tied output head is allowed (GPT-2 ties `wte` and the head).
- Optional `gradient_checkpointing: bool` attribute (trainer may need it on a 16 GB T4).

**Weights (`utils/load_gpt2_weights.py`)**
- `load_pretrained_gpt2(size: str, cache_dir="checkpoint/pretrained", device="cpu", dtype=torch.float32) -> GPTModel`
  Downloads OpenAI GPT-2 weights (safetensors from the Hugging Face hub repo `openai-community/<size>`, via
  plain HTTP or `huggingface_hub`), maps them into `GPTModel` (split `c_attn` into Q/K/V, transpose Conv1D weights),
  and asserts every parameter was filled.

**Generation (`utils/generate.py`)**
- `generate(model, idx, max_new_tokens, context_size=1024, temperature=0.0, top_k=None, eos_id=50256) -> LongTensor`
- `text_to_ids(text) -> LongTensor[1, T]` (batch dim added, Raschka style; **`len()` of it is 1, not the token
  count**: use `.shape[1]`) / `ids_to_text(ids)` with tiktoken gpt2, single row only (ValueError for B>1).
  Decode generated tokens with `ids_to_text_until_eos(ids_row, eos_id=50256)`; pass only the generated part
  (batched generation pads finished rows with EOS).
- Dtype rule (G1 review): load the base in its final dtype via `load_pretrained_gpt2(dtype=...)` **before**
  `apply_lora`; never `.half()`/`.to(dtype)` after it (that casts the LoRA params). Loss on `logits.float()`.
  Merge LoRA in fp32, then cast; or keep the adapter unmerged.

**LoRA (`training_model/lora.py`)**
- `apply_lora(model, rank: int, alpha: float, targets=("attn", "ff")) -> int` (freezes base, returns trainable count)
- `lora_state_dict(model) -> dict` (adapter weights only), `load_lora(model, state_dict)`, `merge_lora(model)`.

**Prompt (`utils/prompt.py`)**, Alpaca template, the only place it is defined:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}          <- whole "### Input:" block omitted when input is empty

### Response:
{output}<|endoftext|>
```
- `format_prompt(instruction: str, input: str = "") -> str` (ends with `"### Response:\n"`)
- `format_example(ex: dict) -> tuple[str, str]` returns (prompt, response_with_eos).
- `SYSTEM_RULES: str` short policy line placed at the top of every `input` for grounded tasks.

**Dataset (`dataset/`)**: `fraudlm_sft_train.json`, `fraudlm_sft_val.json`, `fraudlm_sft_test.json`
- JSON list of `{"id", "instruction", "input", "output", "task", "source"}`.
  `task` ∈ `explain_score | dataset_fact | policy | banking_qa | general`; `source` = origin (dataset name +
  licence, `template`, or `llm:<model>`).
- Every prompt+response ≤ 1,024 tokens (target ≤ 512). Test split also holds unseen question phrasings.

**ML grounding table (`preprocessed/ml_scores.csv`)**, built by data-builder from the team repo outputs:
- `app_id, month, score, threshold, flagged, label, top_factors` (`top_factors` = JSON list of
  `{"feature","direction","contribution"}`, **non-protected features only**).
- LLM training examples use **development months 0–5 rows only**; test months 6–7 rows are for the demo/evaluation.

**Checkpoints (`checkpoint/<run_id>/`)**: `lora_adapter.pt`, `run_config.json`, `train_log.csv`
(`step, epoch, train_loss, val_loss, lr, tokens_seen, elapsed_s`), `samples.txt` (generations each eval).

---

## 4. Fine-tuning data mix (data-builder decides the final sizes after G2 gives the throughput)

| Task | How it is made | Why |
|---|---|---|
| `explain_score` | Programmatic from `ml_scores.csv`: context (score, threshold, flagged, top non-protected factors) in `input`; many question phrasings (hand-written + LLM paraphrases); answers built from templates so **numbers are exact** | Core skill, faithful numbers |
| `dataset_fact` | Programmatic from `../reports/tables/*.csv` (EDA) and repo metrics | "What does the data show" with exact numbers |
| `policy` | Templates + paraphrases: refuse approve/reject/block; never cite protected attributes; say "not in the context" | Safety, OOS07 |
| `banking_qa` | Public banking customer-support / fraud Q&A datasets (licence checked, recorded) | Banking knowledge |
| `general` | Small share of a public instruction set (e.g. Alpaca-cleaned / Dolly) | Keep it fluent |

Starting target: 8k–15k train examples, 500 val, 500 test. Check against the G2 time estimate.

---

## 5. Build order and gates (a step starts only when the gate before it is PASSED by reviewer)

| Gate | Deliverable | Owner | Pass condition | Target |
|---|---|---|---|---|
| G0 | Scaffold, `environment.yml`, conda env builds | lead | `python -c "import torch, tiktoken, safetensors"` works | Fri 05:00 |
| G1 | Stage 1 model + weight loading + generation + LoRA | model-builder | Tests: shapes, causal mask, **GPT-2 small loss on a fixed paragraph in 3–4.5 (random ≈ 10.8)**, coherent greedy text from "Every effort moves you", LoRA leaves base frozen and `merge_lora` gives same logits | Fri 08:00 |
| G2 | Kaggle feasibility | trainer (+ Thông runs on Kaggle) | XL loads on T4/P100, 20 LoRA steps run, peak memory and s/step logged → projected full-run time ≤ 5 h, else drop to Large | Fri 10:00 |
| G3 | Dataset v1 | data-builder | Schema valid, no protected-attribute reasons, token lengths OK, splits disjoint, sources/licences listed | Fri 11:00 |
| G4 | Training script + full Kaggle run | trainer | Masked loss (prompt tokens = −100), resumable checkpoints, val loss falls; run finished | Fri 20:00 (retry window to Sat 06:00) |
| G5 | Evaluation + chat demo | evaluator, chat-builder | Metrics on test split vs baselines (B0 template responder, B1 base GPT-2 without fine-tune); `python chat.py` works on a test-month application | Sat 14:00 |
| G6 | Final audit | reviewer | All rules in Section 6 hold; numbers in results files match what the report will quote | Sat 18:00 |

**Status 06/11 ~16:00:** G0–G4 PASS. G5 and G6 PASS WITH FIXES (`reviews/G5_G6_review.md`, non-blocking).
Run `xl_r16_v1` trained (Kaggle `fraudlm-train` v1) and evaluated (Kaggle `fraudlm-eval` v1).

---

## 6. Development conventions

1. Read `PLAN.md` and the last 30 lines of `TEAM_LOG.md` before starting. Write only in folders you own.
2. Each code file starts with a header: purpose, author "FraudLM v2 team (COS30049.2-1)", "Architecture follows
   Raschka, S. (2024) *Build a Large Language Model (From Scratch)*, Manning; code repo Apache-2.0" where it applies.
3. Comments explain *why*, short and plain. Thông must be able to explain every line in a demo.
4. Test on CPU with `gpt2-small` and tiny data before anything goes to Kaggle. Long jobs: resumable, log to disk, print ETA, fail loudly.
5. Seed 42 everywhere. Never touch test months 6–7 for any training decision.
6. Never write secrets (Kaggle token, HF token) into any file, log or message. Never run `kaggle kernels push`
   or any upload without Thông's explicit OK in the conversation.
7. Record completed work in `TEAM_LOG.md`: time, role, changes, files, test results and open issues.
8. No prose for the submitted report/README; give facts, numbers and bullet notes. Thông writes the sentences.
