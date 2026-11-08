import hashlib
import json

import pytest

from preprocess_data.normalize_source_labels import migrate
from utils.source_lineage import file_sha256, recorded_sha256, recorded_pair


def test_metadata_migration_preserves_payload_and_recorded_identity(tmp_path):
    data = tmp_path/'data'
    data.mkdir()
    originals = {}
    for split in ('train','val'):
        path = data/f'fraudlm_sft_{split}.json'
        path.write_text(json.dumps([dict(id=split,instruction='Explain.',input='Score: 0.2',
            output='Analyst review only.',task='explain_score',source='llm:generator-v0')]))
        originals[path.name] = file_sha256(path)
    result = migrate(data,tmp_path/'originals')
    assert result['status'] == 'migrated'
    for name, old in originals.items():
        path = data/name
        assert file_sha256(path) != old
        assert recorded_sha256(path) == old
        assert json.loads(path.read_text())[0]['source'] == 'synthetic:generated-qa-v1'
    assert recorded_pair(data/'fraudlm_sft_train.json',data/'fraudlm_sft_val.json') == originals['fraudlm_sft_train.json'][:16]+'-'+originals['fraudlm_sft_val.json'][:16]
    path.write_text(path.read_text().replace('Analyst review only.','A changed answer.'))
    with pytest.raises(ValueError,match='checksum'):
        recorded_sha256(path)


def test_unmigrated_file_keeps_its_actual_identity(tmp_path):
    path = tmp_path/'data.json'
    path.write_text('[]')
    assert recorded_sha256(path) == file_sha256(path)


def test_several_origins_are_not_silently_collapsed(tmp_path):
    path = tmp_path/'fraudlm_sft_train.json'
    path.write_text(json.dumps([dict(source='llm:generator-v0'),dict(source='llm:generator-v1')]))
    with pytest.raises(ValueError,match='Multiple'):
        migrate(tmp_path,tmp_path/'backup')


def test_existing_training_run_resumes_after_source_only_migration(tmp_path,monkeypatch):
    import torch
    from training_model import train_sft as trainer
    from training_model.gpt_model import GPTModel
    tiny = dict(vocab_size=50257,context_length=128,emb_dim=16,n_heads=2,n_layers=1,drop_rate=0.,qkv_bias=True)
    def model(*args,**kwargs):
        torch.manual_seed(12)
        return GPTModel(tiny)
    monkeypatch.setattr(trainer,'load_pretrained_gpt2',model)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    data = tmp_path/'data'
    data.mkdir()
    rows = [dict(id=f'row{i}',instruction='Explain.',input='Score: 0.2',output='Review only.',
                 task='explain_score',source='llm:generator-v0') for i in range(4)]
    for split in ('train','val'):
        (data/f'fraudlm_sft_{split}.json').write_text(json.dumps(rows))
    original = trainer.data_sha256(data/'fraudlm_sft_train.json',data/'fraudlm_sft_val.json')
    args = ['--size','gpt2-small','--lora_rank','2','--lora_alpha','4','--data_dir',str(data),
        '--out_dir',str(tmp_path),'--run_id','continued','--resume','','--max_len','128',
        '--max_steps','2','--micro_batch','1','--grad_accum','1','--eval_at_start','off',
        '--eval_every','2','--save_every','1','--sample_tokens','1','--log_every','1']
    with pytest.raises(trainer.SimulatedKill):
        trainer.main(args+['--debug_crash_at_step','2'])
    migrate(data,tmp_path/'original-data')
    assert trainer.main(args+['--resume','auto']) == 'done'
    config = json.loads((tmp_path/'continued/run_config.json').read_text())
    assert config['data_sha256'] == original
    assert config['actual_data_sha256'] != original
