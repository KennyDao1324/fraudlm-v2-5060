# Development guide

FraudLM uses a custom GPT-2 architecture, pretrained weights and LoRA instruction fine-tuning. It supports analyst review, not independent fraud classification or customer decisions.

- Follow `DEVICE_GUIDE.md` for this branch's interpreter, hardware and run commands. The Mac profile uses CPU; the RTX 5060 profile uses CUDA.
- `PLAN.md` records the original architecture decisions, module interfaces and validation gates. `TEAM_LOG.md` preserves technical run notes; historical hardware and model references may differ from the current device profile.
- Test model changes on `gpt2-small` and small inputs first. Keep long runs resumable and record seeds, configuration, data hashes and checkpoint hashes.
- Preserve source attribution, licences and each training example's provenance. The `source` field is data lineage, not a code-author label.
- Keep adapters, private datasets, pretrained caches and credentials out of Git. Follow `DISTRIBUTION.md` before sharing them.
- Separate raw generations from deterministic calculations and guard replacements in evaluation. Previously reviewed examples are regression checks, not a new blind test.
- Never use protected attributes as reasons for a fraud decision. Keep score/threshold facts grounded in the supplied classifier context.
