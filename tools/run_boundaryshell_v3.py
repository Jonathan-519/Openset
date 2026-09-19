#!/usr/bin/env python3
"""TaxoLocal-v3: preflight, two-stage training, known-only calibration, test, pack."""
import argparse
import json
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from boundaryshell.protocol import audit, write_json, training_fingerprint, prepare_known


def effective_config(path, seed):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    cfg['seed'] = seed
    cfg['data']['seed'] = seed
    if cfg['data'].get('sampler', {}).get('name') == 'hierarchical_episode':
        cfg['data']['sampler']['seed'] = seed
        cfg['data']['sampler']['holdout_seed'] = seed
    if cfg['model']['prec'] != 'fp32':
        raise ValueError('v3 uses fp32 prompt training; set model.prec: fp32')
    if cfg['boundaryshell']['epochs'] < 1 or cfg['classifier']['epochs'] < 1:
        raise ValueError('Epoch counts must be positive')
    return cfg


def bind_run(run, cfg):
    run.mkdir(parents=True, exist_ok=True)
    config_path = run / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != cfg:
            raise ValueError('Run configuration changed; use a new --run-dir')
    else:
        write_json(config_path, cfg)
    report_path = Path(cfg['data']['train']).parent / 'prepare_report.json'
    if report_path.exists():
        write_json(run / 'prepare_report.json', json.loads(report_path.read_text()))
    fingerprint = training_fingerprint(cfg)
    manifest_path = run / 'training_inputs.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != fingerprint:
        raise ValueError('Run inputs changed; use a new --run-dir')
    if not manifest_path.exists():
        write_json(manifest_path, fingerprint)


def pack(run):
    if not run.is_dir():
        raise FileNotFoundError(run)
    paths = sorted(p for p in run.rglob('*') if p.is_file() and p.suffix in {'.json', '.jsonl', '.log', '.yml', '.yaml', '.txt'})
    if not paths:
        raise ValueError('No reports in run')
    destination = run.parent / (run.name + '_review.tar.gz')
    with tarfile.open(destination, 'w:gz') as archive:
        for path in paths:
            archive.add(path, arcname=str(Path(run.name) / path.relative_to(run)), recursive=False)
    print('Review archive: {}'.format(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'preflight', 'train', 'calibrate', 'test', 'pack'])
    parser.add_argument('--config', default='configs/Zooplankton_Taxonomic_Tree/TaxoLocal_v3_BoundaryShell.yml')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--stage', choices=['both', 'classifier', 'open'], default='both')
    parser.add_argument('--check-image-content', action='store_true', help='Check cross-split SHA256 duplicates; no test scores')
    args = parser.parse_args()
    os.chdir(ROOT)
    run = Path(args.run_dir or 'runs/boundaryshell_v3/seed_{}'.format(args.seed)).resolve()
    if args.command == 'pack':
        pack(run)
        return
    cfg = effective_config(args.config, args.seed)
    if args.command == 'prepare':
        print(json.dumps(prepare_known(cfg), ensure_ascii=False, indent=2))
        return
    bind_run(run, cfg)
    if args.command == 'preflight':
        report = audit(cfg, args.check_image_content)
        write_json(run / 'preflight.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    import torch
    from boundaryshell import runtime
    if not torch.cuda.is_available():
        raise RuntimeError('Real CLIP training/calibration/testing needs CUDA. Unit tests run on CPU.')
    runtime.seed_all(args.seed)
    device = torch.device('cuda')
    if args.command == 'train':
        if args.stage in ('both', 'classifier'):
            runtime.train_classifier(cfg, run, device)
        if args.stage in ('both', 'open'):
            runtime.train_open(cfg, run, device)
    elif args.command == 'calibrate':
        runtime.calibrate(cfg, run, device)
    elif args.command == 'test':
        runtime.test(cfg, run, device)


if __name__ == '__main__':
    main()
