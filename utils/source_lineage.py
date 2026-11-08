"""Verify source-only metadata migrations without changing recorded training-run identities."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def recorded_sha256(path):
    """Return the original file identity only for an exactly verified metadata-only migration.

    Actual file hashes remain available through file_sha256. Existing run configurations
    retain their original hashes; new training runs record their current actual hashes.
    """
    path = Path(path)
    actual = file_sha256(path)
    sidecar = path.parent/'source_tag_migration.json'
    if not sidecar.exists():
        return actual
    manifest = json.loads(sidecar.read_text())
    entry = manifest.get('files',{}).get(path.name)
    if entry is None or actual == entry['original_sha256']:
        return actual
    if actual != entry['current_sha256']:
        raise ValueError('Source metadata migration checksum mismatch: '+str(path))
    rows = json.loads(path.read_text())
    payload = [{k:v for k,v in row.items() if k != 'source'} for row in rows]
    fingerprint = hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if len(rows) != entry['rows'] or fingerprint != entry['training_payload_sha256']:
        raise ValueError('Training payload differs from the verified source-only migration')
    return entry['original_sha256']


def recorded_pair(train_path, val_path):
    return recorded_sha256(train_path)[:16]+'-'+recorded_sha256(val_path)[:16]
