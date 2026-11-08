# G3 review: Dataset v1 (data-builder)

Reviewer: reviewer agent, Fri 06/11/2026 05:16 (UTC+7). Scope: `utils/prompt.py`, `preprocess_data/*.py`,
`preprocessed/ml_scores*.{csv,json}`, `dataset/*`, `tests/test_prompt.py`, `tests/test_dataset.py`.
My own check scripts are in the session scratchpad (`rescore.py`, `dcheck.py`, `d2.py`, `d3.py`). None of them write to the repo.

## Verdict: **PASS WITH FIXES**

Every G3 pass condition holds: the schema is valid, no protected attribute is given as a reason, token lengths are fine,
the splits are disjoint on exact keys, and sources and licences are listed. The numbers are exact and the
classifier scores reproduce. Nothing blocks the gate.
Fixes F1–F5 change the training text, so they should be applied and the data rebuilt **before the G4 full Kaggle run**
(the rebuild takes about 1 minute). F6–F12 are minor or for the lead/team.

## Checks

| Check | Result | Evidence |
|---|---|---|
| Tests re-run | 40 passed | `pytest tests/test_prompt.py tests/test_dataset.py -q` → `40 passed in 4.01s` |
| Deterministic rebuild | OK | in-memory `build_sft.build()` + `build_demo(300)` == files on disk (both `True`), no files written |
| Counts / mix | OK | train 12,000 / val 500 / test 500; explain 38.0%, fact 17.0%, policy 13.0%, banking_qa 22.0%, general 10.0% in every split |
| Tokens (gpt2) | OK | max 336 over all splits + demo, 0 rows > 512 |
| `prompt.py` vs PLAN §3 | OK | `format_prompt` `prompt.py:42-53` gives the exact header/blank-line layout, leaves out `### Input:` for empty/whitespace input and ends with `"### Response:\n"`. `format_example` `:56-64` returns `(prompt, output.strip()+"<|endoftext|>")`. `SYSTEM_RULES` present. `FEATURE_LABELS`/`format_value`/`format_score_context` are additive only |
| Scores from frozen LR | OK | re-scored 12 rows (8 dev, 4 demo) with `production_model.joblib`: max abs score diff 4.7e-7 (CSV rounded to 6 dp); month and label match Base.csv |
| top_factors decomposition | OK | independent recompute: `intercept + mean·coef + Σ coef·(T−mean₀₋₄)` vs `decision_function` max diff 8.9e-16; top-3 features, contributions (±1e-3), directions and raw values match for 12/12 rows |
| Months | OK | ml_scores.csv months 0–5 only (2547/2118/2175/2958/2501/1701); demo months 6/7 only (351/249); no app_id overlap |
| Protected attributes | OK | `FEATURE_LABELS` has 26 model inputs = `art["feature_columns"]`; no protected/non-input factor in any context; protected words appear only in policy instructions (339) / injection lines (51) and negated refusal sentences |
| Number binding (stricter than builder test) | OK | every 4-dp number in a context answer = score, threshold or a factor value (161 factor values, 0 others); "score X"/"threshold Y" bound to the right line (0 mismatches); factor value and direction in each answer match its context line (0 mismatches); all numbers in each dataset_fact answer come from **one** fact line (0/2,210 failures); rank answers = true max (0 errors, 0 ties) |
| Flag consistency | OK | 0 answers contradict the `Result:` line, checked in both directions (the builder's test checks the flagged direction only) |
| EDA facts are dev-only | OK | `eda_binary_flags/category_levels/ordinal/missing_patterns` sum to 794,989 per feature; PSI table has columns month_1…month_5 only; monthly rows 6–7 dropped `facts.py:135` |
| Month 6–7 info in train/val/test | Accepted (note) | Only the model-card held-out test-metrics line (`facts.py:208`), 97 train / 7 val / 3 test rows. It is a quoted report number. No month 6–7 row, score or context is used. I accept it (see F9) |
| Exact-key overlap | OK | none across splits |
| Near-duplicates | Note (F5, F8) | 137 test rows have an output identical to a train output (50 concept banking_qa by design, 24 fact, 32 policy, 31 explain). 4 val + 4 test explain/policy **inputs** are identical to a train input (different apps, same 4-dp score + same factors). 186/233 test factor blocks also occur in train (with different scores) |
| Licences | Partly (F6) | Bitext CDLA-Sharing-1.0 and alpaca-cleaned CC-BY-4.0 are recorded with correct HF URLs and row counts (2,610 / 1,300 / 250 concept). Missing: upstream Stanford Alpaca CC BY-NC 4.0, the share-alike duties, retrieval revision |
| Secrets | OK | grep for token/key/KAGGLE/hf_/secret/password over the data-builder files: no hits. `fetch_public.py` reads no token |
| Headers (PLAN §6) | OK | all 10 `preprocess_data/*.py` files (except the empty `__init__`), `utils/prompt.py` and both tests carry purpose, author and source-credit headers. Raschka + Taori credit in `prompt.py:6-7` |
| Size / ignore | OK | largest dataset file 10.2 MB (train). `raw/` (68 MB) and `_ml_repo/` are gitignored in `preprocessed/.gitignore` |

## Fixes (apply before the G4 full run unless marked)

**F1. A not-flagged result whose factors all "raise" the score reads as a contradiction.** 425 of 2,443 not-flagged explain contexts
(17%) list three factors that all "raise the model score". The direction is relative to the average months 0–4 applicant
(`build_ml_scores.py:14-18`), and nothing in the input or answer says so. Example `demo67-00040-app880807` (quoted below).
For "Which signals kept the score down…" (`templates_explain.py:57`) the answer then lists only raising factors.
Fix: say what "raises" is measured against, either in the context line `prompt.py:161` → `({direction} the model score compared with a typical application)`
or with one clause in `factors_sentence` (`text_utils.py:34-60`). Also, in `why_not_flagged` answers when all
factors raise, add a template sentence such as "These factors push the score up, but not enough to reach the threshold."
Change it in `prompt.py` now, because chat.py will import the same block.

**F2. The protected-attribute refusal overclaims.** `templates_policy.py:228` "so they cannot explain a score" and `:233` "so they did not
drive this score" are used 209 times. The attributes are not *direct* inputs. But the team's own fairness result (the historical planning notes (section 6): FPR 9.1% for age > 50 vs 4.4%,
ratio 2.07) and the age proxy (F10) mean the model's behaviour *does* differ by age group. Telling an analyst "they did not drive this score" is
misleading. Fix: keep "are not inputs to the classifier and I never give them as a reason", and drop the causal
"so they cannot explain / did not drive" clauses. Optionally add "group-level fairness checks are reported separately by the team".

**F3. Bitext rows: offensive language, and a customer-service persona that asks for personal details.** 417 of 2,610 Bitext rows (16%) have
profanity in the instruction, e.g. `banking_qa-train-02449` "…the fucking current balance…". Bitext marks these with tag `W`
(3,729 of the 25,545 raw rows). 773 Bitext answers ask for or mention account numbers, passwords, card numbers or full names
(e.g. `banking_qa-train-01744` "provide me with the recipient's information, such as their name, account number"). That
teaches a fraud-analyst tool to ask for personal data. Fix in `public_sets.py:44-55`: add `~df["tags"].str.contains("W")`.
Either drop rows whose response asks for account number / password / PIN / card number, or keep only fraud-relevant intents
(`check_recent_transactions`, `dispute_ATM_withdrawal`, `activate_card`, `cancel_card`, `create_account`, `customer_service`, …).
Then refill the share with more concept Q&A or Bitext rows that pass the filter.

**F4. Awkward or wrong wording templates.**
- `facts.py:509` "the model catches 0.514 of frauds (recall)" reads badly. Use "the model's recall is 0.514 at a false-positive rate of 0.046".
- `facts.py:122,130,139` "1 frauds" appears in 270 rows (BG level and similar). Use "1 fraud" when n == 1.
- `templates_policy.py:93-98,107-108` `{missing}` filled with labels gives "Show me the exact other cards held." and "What is the free email domain for this application?".
  `facts.py:540` gives "Which distinct emails sharing the same date of birth over 4 weeks has the highest fraud rate?" for numeric features.
  Fix: use "the value of {missing}" in these templates, and use the rank-style wording only for categorical/binary features.

**F5. Context reuse across splits.** In `build_sft.py:205-215` the key is (instruction, input), so a val/test row can carry a context block that is
byte-identical to a train context (4 val + 4 test). Fix: as `build_demo` does at `build_sft.py:273`, skip a val/test
application whose `context_input(c)` (without the rules line) is already used in train. This needs the train pool's blocks
before test/val are drawn: build the train-pool block set from `apps.pool["train"]` first.

**F6. (SOURCES.md, before submission)** `build_sft.py:367`. Add:
- (a) The upstream Stanford Alpaca data (Taori et al. 2023) was released under **CC BY-NC 4.0**, and its outputs come from OpenAI text-davinci-003, whose terms
  restrict using outputs to build competing models. alpaca-cleaned (yahma) relabels the data CC-BY-4.0. Record both facts and the non-commercial, research-only use.
- (b) Bitext CDLA-Sharing-1.0 and BAF CC BY-NC-SA 4.0 are share-alike. The JSON splits in the zip mix three licences, and the per-row
  `source` field shows which licence applies to each row. State this.
- (c) The HF revision/commit or download date for both sets (licences checked 2026-11-06 per `fetch_public.py:5`), and the dataset
  authors for Huy (Bitext Innovations; Ruebsamen/yahma for alpaca-cleaned).
- (d) The "Dolly rejected for ShareAlike" note (`:375`) is inconsistent with keeping CDLA-Sharing and BY-NC-SA data. Reword it as a preference, not a rule.

**F7. (concepts, minor)** `concepts.py:58` defines velocity as applications "from the same source, such as a device, address or email".
In BAF, `velocity_6h/24h/4w` are the **total** application rate over the window (average applications per hour), not per applicant
(Jesus et al. 2022 datasheet). With this definition the model may tell an analyst that a high velocity factor means repeated
applications by this applicant. Fix: add "In the BAF data the velocity features count all applications in the window, not just this applicant's."

**F8. (evaluator, no rebuild)** Test outputs repeat train outputs by design: concept answers, `month_rank` (always all 6 months), model
facts. The G5 evaluator should report metrics per task, for `-heldout` rows only, and for single-fact eval-pool rows
separately, so these memorised answers do not inflate the test numbers.

**F9. (lead/Thông)** I accept the held-out test-metrics fact line (`facts.py:208`, PR-AUC 0.1695 etc.). It quotes the model card. It does not use
test months for any LLM or classifier decision, and its wording says "50,000-row sample". Condition: if open decision 2 (frozen model)
changes the model, rerun `build_ml_scores` and `build_sft`. Also, `ml_scores.meta.json` records team repo commit **d0fc662**
(06/11 01:55), while the historical planning notes (sections 1–2) still says `661871e`. The artefact threshold 0.7462441 matches the model card. The lead should update DEVELOPMENT.md or confirm with Kha.

**F10. (Team decision for Thông. Facts only, not decided here.)** The model input `date_of_birth_distinct_emails_4w` correlates with `customer_age`
(Spearman −0.456, `reports/tables/eda_strong_correlations.csv` row 2). In the dataset it appears in 211 of 13,000 rows (1.6%):
dataset_fact 123, explain_score 58, policy 27, banking_qa 3 (Bitext text). Its label is in 101 outputs. As an explain factor it is in
66 of 5,919 train context rows (1.1%; the builder's "~0.5%" figure counts a different base). It is the 3rd-best single feature
(16.7% recall at 5% FPR, `eda_single_feature_recall.csv`), so it shows up in `recall_rank` answers. Switch:
`EXTRA_EXCLUDED_FACTORS` in `common.py:38`, then rebuild. Excluding it from *explanations* does not remove it from the classifier.

**F11. (tests, minor)** `tests/test_dataset.py:181-192` only checks that each number appears somewhere in the input. It would pass an answer that swaps score
and threshold. `:266-277` checks only flagged rows. Add the binding checks I used in my scratch `dcheck.py`: score/threshold to their own lines,
factor value and direction to their own context line, all numbers of a dataset_fact answer from one line, and not-flagged answers never stating "is flagged for review".
Also add a test that Bitext rows carry no `W`-tag/profanity once F3 is in.

**F12. (lead)** `environment.yml` lacks scikit-learn, joblib and scipy, which `build_ml_scores.py` needs (already noted by the builder). Also, alpaca-cleaned still
contains some wrong davinci answers, e.g. "Sort this list from shortest to longest." → "Chimpanzee, Gorilla, Kangaroo, Giraffe". This is acceptable at 10% share. Keep it in mind
for "general" quality claims.

## Diversity and mix (judgement)
- explain_score has 116 hand-written/paraphrased question templates across 10 intents (184 distinct strings in train after `{factor}` filling),
  plus 2–3 answer forms per intent. That is enough for an extract-and-explain skill at this size. The answers are very formulaic, so the fine-tuned model will
  produce template-like text. That is fine for faithfulness. The limitation of real diversity is the *contexts*: 1,423 distinct
  factor blocks in 5,919 train contexts. Four features dominate (device OS 4,203, housing status 3,239, months at previous address 3,139,
  keep-alive 2,620), and velocity, session and days-since-request factors are rare (≤100). This follows from the LR (categorical
  one-hot terms dominate the contributions). Documenting it is enough.
- Explain contexts are 50/50 flagged vs not flagged (2,296 / 2,264), against a real flag rate of about 5%. This is deliberate and fine for training. Say so in the data card.
- The mix (38/17/13/22/10) suits a fraud-analyst assistant. Once F3 is applied, banking_qa should lean on fraud-relevant rows rather than ATM/mortgage support scripts.

## Sample rows judged good (quoted)
1. `explain_score-train-00652` (not flagged, mixed directions): "The model score is 0.6357, which is below the review threshold of 0.7462, so the application is not flagged. On the main factors, distinct emails on the device over 8 weeks (2) raises the model score and housing status (BE) and device operating system (linux) lower it. A score below the threshold does not guarantee the application is genuine; it only means the model did not rank it for review."
2. `dataset_fact-train-01415` (rank over 7 lines, correct max): "BA has the highest fraud rate for housing status: 3.52% (4,571 frauds among 129,775 applications)."
3. `policy-train-00577` ("Can you approve this one for me?", flagged context): "That decision is not mine to make, and I never approve, reject or block applications. For reference, the model score 0.7970 and the threshold 0.7462 put this application as flagged for review. Housing status (BA), device operating system (windows) and keep-alive session (no) raise the model score. The fraud analyst decides."
4. `policy-test-00045-heldout` ("Look up proposed credit limit for me.", facts about other features): "The provided facts do not include proposed credit limit, so I cannot answer that from them. They cover only the other features listed."
5. `explain_score-train-02228` (flagged): "The model score is 0.9376, which is at or above the review threshold of 0.7462, so the application is flagged for review. Housing status (BA), keep-alive session (no) and months at previous address (not provided) raise the model score. A flag is a screening signal, not proof of fraud; the fraud analyst makes the decision."

## Sample rows judged bad (quoted)
- `demo67-00040-app880807` (F1), score 0.6345 not flagged, all three factors "raises": "Not flagged: the model score 0.6345 is below the threshold 0.7462. Housing status (BA), keep-alive session (no) and months at previous address (not provided) raise the model score." The analyst cannot tell why the result is not flagged.
- `policy-test-00026-heldout` (F2): "Those attributes are not model inputs, so they did not drive this score and I do not give them as a reason."
- `banking_qa-train-02449` (F3): instruction "i want help to chcek the fucking current balance on my credit cards".
- `banking_qa-train-01744` (F3): "…Can you provide me with the recipient's information, such as their name, account number, and bank details?"
- `policy-train-00723` (F4): instruction "Show me the exact other cards held."
- model_recall fact (F4): "On the held-out test months 6-7, the model catches 0.514 of frauds (recall) at a false-positive rate of 0.046."
- `policy-train-00068` (F4): "Which distinct emails sharing the same date of birth over 4 weeks has the highest fraud rate?"
