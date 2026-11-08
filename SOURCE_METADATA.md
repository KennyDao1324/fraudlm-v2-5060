# Generated-example source metadata

`synthetic:generated-qa-v1` identifies generated example wording and concept answers. It is a data-origin label, not the running model or an API dependency. These examples are not relabelled as human-authored. Detailed original-origin records are retained separately by the project owner.

This migration changes source metadata only. Instructions, input contexts, answers, IDs, tasks, row order, trained weights, predictions and measured scores remain unchanged. Dataset build manifests report the new actual file checksums.

`source_tag_migration.json` beside each migrated dataset records the original/current byte checksums, row counts and a canonical checksum of every non-source field. Existing run configs and historic evaluation records keep their original identities; they are not rewritten to claim training on newly serialized files.

The evaluator and resume path accept the original identity only after verifying both the migrated file checksum and unchanged training payload. Unrelated content changes fail validation. New training runs record actual current hashes. Keep each migration sidecar with its dataset when transferring an existing run to another device. Newly generated datasets need no historical migration sidecar.

For older private input folders on another device, use `python preprocess_data/normalize_source_labels.py --data-dir dataset_multisource --backup-dir <outside-project-folder>`. Use a separate fresh backup folder for each dataset directory. The utility refuses to merge distinct origins and never labels generated examples as human-authored.
