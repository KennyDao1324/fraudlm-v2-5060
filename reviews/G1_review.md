# G1 review: model-builder (Stage 1 model, weight loading, generation, LoRA)

Reviewer: reviewer agent. Date: 2026-11-06 04:54 (UTC+7). Files reviewed: `training_model/gpt_model.py`,
`training_model/lora.py`, `utils/load_gpt2_weights.py`, `utils/generate.py`, `tests/test_gpt_model.py`,
`tests/test_load_weights.py`, `tests/test_lora.py`, `tests/test_generate.py`.

## Verdict: **PASS WITH NOTES** (no blocking issue; G2 can start)

Every G1 pass condition in PLAN §5 and every acceptance test in `model-builder.md` holds, after accepting the
two tolerance deviations below. The notes are small fixes plus warnings for trainer and chat-builder.

Full suite re-run by the reviewer (`.venv/Scripts/python -m pytest tests -q -s`):

```
28 passed in 12.62s
```

The reviewer's own extra checks were run from a scratch script outside the repo
(`...\scratchpad\g1_checks.py`). They are not part of the repo, and the results are quoted below.

## Decisions on the two deviations

| # | Deviation | Decision | Reason / evidence |
|---|---|---|---|
| D-1 | HF reference: fp32 max logit diff 1.14e-4 (misses atol 1e-4); strict check moved to fp64 (3.69e-13), with fp32 kept at < 1e-3 | **ACCEPT** | GPT-2 logits are about 100 in size, so 1.14e-4 is a relative error of about 1e-6, which is fp32 rounding from different kernels and summation order. At fp64 the diff is 3.69e-13. That shows the mapping (c_attn split order, Conv1D transposes, LN, GELU, tying) is exact; a real mapping error would show up as more than 1e-2. The intent of the 1e-4 check is met. Reproduced: `[G1] max \|ours - HF\| logit difference: float32 1.14e-04, float64 3.69e-13` (tests/test_load_weights.py:107-109). |
| D-2 | Real-weight `merge_lora`: diff 9.9e-5, so the test uses atol 1e-4 / rtol 1e-5; strict 1e-5 only on the tiny model | **ACCEPT**, with optional fix N4 | Reviewer re-ran the same merge in fp32 and fp64 on gpt2-small (r16, a32, B ~ N(0, 0.02)). fp32: max diff 1.26e-4, max \|logit\| 120, max relative 3.3e-6. fp64: **6.5e-13**. So the merge maths is exact and the fp32 diff is only rounding. On this input atol 1e-4 alone would fail (1.26e-4); the test passes only because rtol 1e-5 × \|logit\| ≈ 1.2e-3 is added. The combined tolerance is the correct one for ~100-sized logits. The strict 1e-5 check still runs on the tiny model (tests/test_lora.py:112). |

## Check table

| Check | Result | Evidence |
|---|---|---|
| Test suite | PASS | `28 passed in 12.62s` |
| Output shape [B,T,50257] | PASS | tests/test_gpt_model.py:34-40 |
| Causal mask | PASS | SDPA `is_causal=True` with T_q == T_k (gpt_model.py:64-68). Test changes positions 8-11 only, and logits 0-7 stay equal to 1e-6 (test_gpt_model.py:43-53). No KV cache exists, so the top-left alignment of `is_causal` when T_q != T_k cannot cause a problem. |
| gpt2-small loss on a fixed paragraph in 3.0-4.5 | PASS | `paragraph tokens=181 pretrained loss=3.3192 random-init loss=10.9533` |
| Greedy "Every effort moves you" fluent | PASS | `'Every effort moves you forward.\n\nThe first step is to understand the importance of your work.\n\nThe second step is to understand the'` |
| HF reference | PASS (D-1 accepted) | see above |
| Weight tying | PASS | gpt_model.py:163, re-tied after `assign` at load_gpt2_weights.py:149. Reviewer check: `out_head.weight is tok_emb.weight` and same `data_ptr` after loading in fp32 and fp16. |
| c_attn split / transposes | PASS | `torch.chunk(..., 3, dim=-1)` on the (in, 3·out) Conv1D weight gives Q, K, V in HF order, then `.T` (load_gpt2_weights.py:81-88). c_proj, c_fc and mlp.c_proj are transposed, LN and biases are not. Proven exact by fp64 3.69e-13. Every tensor is shape-checked (lines 131-140), and leftover meta tensors are caught (lines 152-155). |
| Parameter counts | PASS | small 124,439,808 / medium 354,823,168 / large 774,030,080 / xl 1,557,611,200 (tied head counted once) |
| LoRA: only adapters trainable, B=0 gives base output, merge gives same logits | PASS | test_lora.py:39-46, 58-67, 101-112, 115-138. r16 a32 trainable: small 2,654,208 / xl 22,118,400 |
| **Gradient checkpointing + frozen base + LoRA** | PASS (reviewer check) | gpt2-small, r8, train mode, real loss. `gc=False` and `gc=True` give the same loss (4.39738). Both give 144 params with grad, **0 non-LoRA params with grad**, and 0 all-zero grads. Max grad diff between ckpt and plain = **0.0**. With a fresh adapter (B=0) under ckpt, \|grad B\| = 3.37e3 and \|grad A\| = 0, as expected. `use_reentrant=False` (gpt_model.py:194) avoids the frozen-input trap. Not covered by a repo test (N3). |
| bf16 / fp16 base + fp32 LoRA (CPU) | PASS (reviewer check) | Load gpt2-small with `dtype=bf16`: loss 2.8396 (fp32 2.8322). With `fp16`: loss 2.8335. With apply_lora, ckpt and backward, LoRA params and their grads stay **fp32**, grads are non-zero, logits are finite, and an AdamW step runs. fp32 base + `torch.autocast("cpu", bf16)` also works, with fp32 LoRA grads. GPU fp16 + GradScaler could not be tested here; that is for G2. |
| Memory: meta + `assign=True` | PASS with note N5 | No random-init allocation; assign swaps tensors in. Measured on gpt2-small, fp32 to CPU: peak commit rises by about 1.14 GB (2.4× the 475 MB of parameters). The cause is `load_file` (523 MB) plus fp32 `.contiguous()` copies of the transposed views. Scaled to XL, this is about 14-15 GB of CPU RAM when loading fp32 onto the CPU, and about 6-7 GB plus one tensor when `device="cuda"`. Fine on Kaggle (about 29 GB RAM; trainer to confirm in G2), but not on a 16 GB laptop. |
| Contracts PLAN §3 | PASS with note N1 | `GPT_CONFIGS` keys and fields OK. `load_pretrained_gpt2(size, cache_dir="checkpoint/pretrained", device="cpu", dtype=torch.float32)` OK. `generate(model, idx, max_new_tokens, context_size=1024, temperature=0.0, top_k=None, eos_id=50256)` OK. `apply_lora(model, rank, alpha, targets=("attn","ff")) -> int`, `lora_state_dict`, `load_lora`, `merge_lora` OK. `gradient_checkpointing` attribute OK. The forward argument is named `in_idx`, not `idx` (N1). |
| `text_to_ids` returns [1,T] LongTensor | ACCEPT, contract wording to update (N2) | PLAN §3:70 gives no return type. Raschka's `text_to_token_ids` also returns [1,T], so this choice is defensible. Nobody calls it outside the model-builder's own tests: the data-builder counts tokens with its own tiktoken call (preprocess_data/common.py:94). The trap is that `len(text_to_ids(s))` is 1. |
| No `transformers` outside tests | PASS | Only `pytest.importorskip("transformers")` at tests/test_load_weights.py:85. Not in environment.yml. No `peft` or `AutoModel` anywhere in src. |
| File headers (PLAN §6 rule 2) | PASS | All 8 files have purpose, team author and source attribution; the 4 src files also have the Raschka credit. Tests leave out Raschka, which is fine because rule 2 says "where it applies". |
| Comments short and explainable | PASS with note N6 | Code reads top to bottom and has no clever abstractions. One comment is inaccurate (N6). |
| Seed 42 | PASS | Every random test seeds 42. Unseeded tests do not depend on randomness. Library code does not seed, which is correct; scripts own the seed. |
| Secrets | PASS | Grep for `token=`, `api_key`, `KAGGLE_`, `hf_<id>`, `secret`, `password` in training_model/, utils/, tests/: no hits |
| Protected attributes | N/A at G1 | No data or templates in G1 scope |
| Numbers traceable | PASS | Every number in the model-builder's TEAM_LOG line is printed by the test suite (`-s`), and the reviewer reproduced them |
| TEAM_LOG hygiene | NOTE (N8) | The model-builder's line is stamped 05:40, but local time at review was 04:54 (+07:00) |

## Fixes (none blocking; do before G4, or note for the named agent)

N1. `training_model/gpt_model.py:180`: rename `forward(self, in_idx)` to `forward(self, idx)` to match the
   PLAN §3 contract (`forward(idx: LongTensor[B,T])`), or have the lead change the contract wording. Positional
   calls work either way; only `model(idx=...)` breaks.

N2. PLAN §3 line 70 (lead): change the wording to `text_to_ids(text) -> LongTensor[1,T]` (batch of one, EOS allowed);
   `ids_to_text(ids: [1,T] | [T] | list[int]) -> str`. Trainer, chat-builder and evaluator must not use
   `len(text_to_ids(...))` as a token count; use `.shape[1]`, or `get_tokenizer().encode(...)` for plain lists.

N3. `tests/test_lora.py` (new test, model-builder): add the reviewer's check as a regression test. Use the tiny
   model with `apply_lora` + `train()` + `gradient_checkpointing=True` vs `False`, and assert the same loss, LoRA
   grads equal (atol 1e-6), and `p.grad is None` for every non-LoRA param. Today only the full-trainable model is
   tested with checkpointing (tests/test_gpt_model.py:77-94).

N4. `tests/test_lora.py:134-138` (optional): as in test_load_weights, add an fp64 merge assertion on the real
   weights (`copy.deepcopy(model).double()` before apply_lora, then cast the LoRA layers to double too). Then
   assert `atol=1e-8`. Reviewer measured 6.5e-13. This keeps a strict check on real weights.

N5. `utils/load_gpt2_weights.py:143` (optional, memory): use `v.to(device=device, dtype=dtype).contiguous()`
   (convert and move first, then make contiguous on the target device), and `del hf_state` once `ours` is built.
   When `device="cuda"`, this stops a full set of fp32 contiguous copies being made in CPU RAM. Not needed if
   G2 shows Kaggle RAM is enough.

N6. `training_model/lora.py:32`: the comment "same as nn.Linear's default init" is inaccurate. A has shape
   (in_dim, rank), so `kaiming_uniform_` uses fan_in = rank and gives A ~ U(±1/√rank) (±0.25 at r=16). nn.Linear,
   and PEFT's (rank, in) layout, use fan_in = in_dim (±0.025 for XL). The code matches Raschka appendix E exactly,
   so keep it, but say "Raschka appendix E init". **Trainer note:** with this init the LoRA update per Adam step is
   about √(in_dim/rank) ≈ 10× larger than with PEFT defaults, so start from a modest LR (Raschka uses 5e-5 for
   LoRA) and watch the G2 loss for spikes.

N7. `utils/generate.py:35-39`: `ids_to_text` on a [B>1, T] tensor raises `TypeError: 'list' object cannot be
   interpreted as an integer`. It also decodes EOS padding as the literal text `<|endoftext|>` (reviewer:
   `'The<|endoftext|><|endoftext|>'`). Raise a clear `ValueError` for B>1 and mention it in the docstring.
   **Chat-builder and evaluator:** cut each row at its first EOS before decoding. Also note `generate.py:73-80`.
   A row that finishes early gets its EOS appended, but the last row to finish does not. This is documented in
   the docstring; downstream code should strip EOS rather than rely on whether it is there.

N8. TEAM_LOG.md (model-builder): log the real clock time. The 05:40 entry was written before 04:54 local time.

## Warnings for other agents (no G1 code change needed)

- **Trainer:** load the base in its final dtype through `load_pretrained_gpt2(..., dtype=...)` *before*
  `apply_lora`. Never call `model.half()` or `model.to(torch.float16)` after `apply_lora`: the reviewer
  confirmed this turns the LoRA params into fp16 (`{torch.float16}`), and then GradScaler's `unscale_` will fail
  or the updates will be rounded away. `model.to(device)` is safe. Compute the loss on `logits.float()`.
- **Chat-builder:** `merge_lora` into a half-precision base rounds the update into fp16/bf16. Measured after one
  AdamW step: max logit diff 0.25 (fp16) and 4.9 (bf16) at logit size ~300, which is about 1-2 ulp of the logit
  dtype. For the demo, either merge in fp32 and then cast, or keep the adapter unmerged.
- **Trainer (G2):** `cache_dir` is relative to the working directory (`/kaggle/working` on Kaggle). XL fp32
  loaded onto the CPU needs about 14-15 GB of CPU RAM at peak (N5).
