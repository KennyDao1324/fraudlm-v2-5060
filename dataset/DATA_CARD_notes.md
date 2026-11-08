# DATA_CARD_notes (facts only)

- Seed 42; tokenizer tiktoken gpt2; examples above 512 tokens are dropped.
- ml_scores source: frozen (model logistic_regression, threshold 0.7462, artefact sha256 4cb8eb3bef9861aa...).
- explain_score and policy contexts use development months 0-5 applications only; months 6-7 appear only in fraudlm_demo_months67.json.
- dataset_fact: for single-fact questions the target fact row comes from a train pool (80%) in train and from an eval pool (20%) in val and test (hash split); distractor lines and ranking questions use all rows.
- Application pools (hash of app_id): train 84%, val 8%, test 8%; no application is shared across pools. Val / test applications whose printed context equals a train context were dropped first (val 90, test 118 applications).
- Test ids ending in -heldout use question wordings that never occur in train or val.
- Fields: id, instruction, input, output, task, source. source is `template`, `synthetic:generated-qa-v1` (question wording or concept answers) or a public dataset name with licence.
- Answers of explain_score, dataset_fact and policy are template-built; every number is copied from the input.
- Factor direction (raises / lowers the model score) is measured against a typical application, the average applicant of months 0-4; the factor lines and the answers say so.

## Repeats and sampling (read before reporting test metrics)

- Test outputs identical to a train output: 144 of 500 (by task: {'banking_qa': 50, 'explain_score': 43, 'policy': 28, 'dataset_fact': 23}); val: 94 of 500 (by task: {'explain_score': 31, 'dataset_fact': 25, 'policy': 25, 'banking_qa': 13}). These are by design (concept answers, month ranking, model facts); report metrics per task and for -heldout rows separately.
- Sampling: explain_score contexts are drawn about 50/50 flagged / not flagged (train share flagged 0.504); the real flag rate of the frozen model on months 0-5 is 0.0752 (5% false-positive budget on validation month 5). ml_scores.csv holds 14000 rows, 7000 flagged.
- Not-flagged train explain_score contexts whose three factors all raise the score: 402; their answers say the factors are not enough to reach the threshold.

## Context diversity

- Train explain_score + policy contexts: 5919; distinct factor blocks: 1424.
- Most frequent factors (name, count): [('device operating system', 4208), ('housing status', 3240), ('months at previous address', 3138), ('keep-alive session', 2607), ('home phone valid', 1664), ('other cards held', 670), ('name-email similarity', 573), ('distinct emails on the device over 8 weeks', 420)].
- Test context rows whose factor block also occurs in train (with a different score): 175 of 236.

| Split | Rows | explain_score | dataset_fact | policy | banking_qa | general | held-out wording | tokens mean / p95 / max |
|---|---|---|---|---|---|---|---|---|
| train | 12000 | 4560 | 2040 | 1560 | 2640 | 1200 | 0 | 205.5 / 275 / 336 |
| val | 500 | 190 | 85 | 65 | 110 | 50 | 0 | 200.7 / 275 / 332 |
| test | 500 | 190 | 85 | 65 | 110 | 50 | 200 | 192.1 / 273 / 332 |
