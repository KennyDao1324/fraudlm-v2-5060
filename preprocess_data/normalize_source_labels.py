"""Migrate a single legacy generated-source label, preserving originals outside the project."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ALIAS = 'synthetic:generated-qa-v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def migrate(directory, backup):
    directory, backup = Path(directory).resolve(), Path(backup).resolve()
    if backup.is_relative_to(ROOT):
        raise ValueError('Keep the original provenance backup outside the project')
    paths = sorted(directory.glob('fraudlm_*.json'))
    originals = {path:path.read_bytes() for path in paths}
    labels = {row.get('source') for data in originals.values() for row in json.loads(data)
              if isinstance(row.get('source'),str) and row['source'].startswith('llm:')}
    if not labels:
        return {'status':'already normalized','directory':str(directory)}
    if len(labels) != 1:
        raise ValueError('Multiple generated-source origins found; do not collapse them into one label')
    origin = labels.pop()
    backup.mkdir(parents=True,exist_ok=False)
    (backup/'origin.json').write_text(json.dumps({'original_source_label':origin,'alias':ALIAS,
        'origin_type':'Generated example wording/concept answers; not human-authored.'},indent=2))
    entries = {}
    for path, original in originals.items():
        before = json.loads(original)
        after = [dict(row,source=ALIAS) if row.get('source') == origin else row for row in before]
        payload = lambda rows:[{k:v for k,v in row.items() if k != 'source'} for row in rows]
        assert payload(before) == payload(after)
        updated = original.replace(origin.encode(),ALIAS.encode())
        assert json.loads(updated) == after
        if updated == original:
            continue
        (backup/path.name).write_bytes(original)
        path.write_bytes(updated)
        entries[path.name] = dict(original_sha256=digest(original),current_sha256=digest(updated),rows=len(after),
            training_payload_sha256=digest(json.dumps(payload(after),sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()))
    sidecar = directory/'source_tag_migration.json'
    if sidecar.exists():
        (backup/sidecar.name).write_bytes(sidecar.read_bytes())
    sidecar.write_text(json.dumps({'schema':1,'change':'Source-only metadata alias; all non-source fields unchanged.',
                                  'alias':ALIAS,'files':entries},indent=2))
    manifest = directory/'build_manifest.json'
    if manifest.exists():
        original = manifest.read_bytes()
        (backup/manifest.name).write_bytes(original)
        info = json.loads(original.replace(origin.encode(),ALIAS.encode()))
        for name,item in info.get('files',{}).items():
            path = directory/name
            if path.exists():
                item.update(bytes=path.stat().st_size,sha256=digest(path.read_bytes()))
        for name in info.get('generated_file_sha256',{}):
            path = directory/name
            if path.exists():
                info['generated_file_sha256'][name] = digest(path.read_bytes())
        manifest.write_text(json.dumps(info,indent=2))
    return {'status':'migrated','files':len(entries),'directory':str(directory),'backup':str(backup)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--backup-dir',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.data_dir,args.backup_dir),indent=2))


if __name__ == '__main__':
    main()
