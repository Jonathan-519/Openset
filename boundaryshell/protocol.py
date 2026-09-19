"""Manifest integrity and disjoint known selection/calibration partitions."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def partition_known(paths, labels, seed=1729):
    """Stable stratified split; calibration rows never select checkpoints."""
    groups = defaultdict(list)
    for i, (path, label) in enumerate(zip(paths, labels)):
        groups[int(label)].append((hashlib.sha256((str(seed) + ':' + str(path)).encode()).hexdigest(), i))
    selection, calibration = [], []
    for label, rows in sorted(groups.items()):
        if len(rows) == 1:
            calibration.append(rows[0][1])
            continue
        indices = [i for _, i in sorted(rows)]
        mid = len(indices) // 2
        selection.extend(indices[:mid])
        calibration.extend(indices[mid:])
    if not selection or not calibration:
        raise ValueError('Need nonempty known selection and calibration partitions')
    return sorted(selection), sorted(calibration)


def read_manifest(path, root):
    rows = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) != 3:
                raise ValueError('Expected path,label,index in {}'.format(path))
            rows.append((str((Path(root) / row[0]).resolve()), int(row[1]), row[0]))
    if not rows:
        raise ValueError('Empty manifest: {}'.format(path))
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError('Duplicate paths in {}'.format(path))
    return rows


def audit(cfg, check_images=False):
    data = cfg['data']
    roots = {'train': data['data_root'], 'val_known': data['data_root'],
             'test_known': data['data_root'],
             'test_intra': data.get('near_test_root', data.get('full_data_root')),
             'test_extra': data.get('ood_test_root', data.get('ood_root'))}
    seen = {}
    contents = {}
    report = {'splits': {}, 'image_content_checked': check_images,
              'test_scores_loaded': False, 'new_blind_test_claim_allowed': False}
    train_species = set()
    for split, root in roots.items():
        rows = read_manifest(data[split], root)
        for path, label, _ in rows:
            if split in ('train', 'val_known', 'test_known') and not 0 <= label < data['num_known_leaves']:
                raise ValueError('Invalid known label: {}'.format(path))
            if split == 'test_extra' and label != -1:
                raise ValueError('Extra label must be -1')
            if path in seen:
                raise ValueError('Path overlap: {} / {}: {}'.format(seen[path], split, path))
            seen[path] = split
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            if split == 'train':
                train_species.add(Path(path).parent.name)
            if split == 'test_intra' and Path(path).parent.name in train_species:
                raise ValueError('Near-test species appears in training: {}'.format(path))
            if check_images:
                digest = sha256(path)
                if digest in contents and contents[digest][0] != split:
                    raise ValueError('Cross-split image duplicate: {} / {}'.format(contents[digest], path))
                contents[digest] = (split, path)
        report['splits'][split] = {'count': len(rows), 'manifest_sha256': sha256(data[split])}
    report['hierarchy_sha256'] = sha256(data['hierarchy'])
    val = read_manifest(data['val_known'], data['data_root'])
    a, b = partition_known([r[2] for r in val], [r[1] for r in val], cfg['boundaryshell']['partition_seed'])
    report['known_validation_partition'] = {'selection_count': len(a), 'calibration_count': len(b),
                                             'selection_indices': a, 'calibration_indices': b}
    return report


def training_fingerprint(cfg):
    """No test manifest or test image access by train/calibration code."""
    return {k: sha256(cfg['data'][k]) for k in ('train', 'val_known', 'hierarchy')}


def prepare_known(cfg):
    """Keep test lists untouched; remove exact duplicates from known train/val.

    Priority: test > validation > training. Test image bytes are hashed only
    for contamination auditing, never used as training features or targets.
    """
    data = cfg['data']
    sources = cfg['source_known_manifests']
    output_paths = [Path(data[s]) for s in ('train', 'val_known')]
    report_path = Path(data['train']).parent / 'prepare_report.json'
    if report_path.exists() and all(p.exists() for p in output_paths):
        report = json.loads(report_path.read_text())
        expected_inputs = {k: sha256(sources[k] if k in sources else data[k])
                           for k in report['input_manifest_sha256']}
        expected_outputs = {k: sha256(data[k]) for k in ('train', 'val_known')}
        if expected_inputs == report['input_manifest_sha256'] and expected_outputs == report['output_manifest_sha256']:
            return report
        raise ValueError('Prepared inputs/outputs changed; use a new manifest directory')
    for path in output_paths:
        if path.exists():
            raise FileExistsError('Prepared manifest exists: {}; no overwrite'.format(path))
    seen = {}
    removed = []
    outputs = {}
    roots = {'test_known': data['data_root'], 'test_intra': data['near_test_root'],
             'test_extra': data['ood_test_root'], 'val_known': data['data_root'], 'train': data['data_root']}
    inputs = {}
    for split in ('test_known', 'test_intra', 'test_extra', 'val_known', 'train'):
        source = sources[split] if split in sources else data[split]
        inputs[split] = sha256(source)
        rows = read_manifest(source, roots[split])
        kept = []
        for path, label, relative in rows:
            digest = sha256(path)
            if digest in seen:
                old = seen[digest]
                old_known = old['split'] in ('train', 'val_known', 'test_known')
                new_known = split in ('train', 'val_known', 'test_known')
                if old_known != new_known or (old_known and old['label'] != label):
                    raise ValueError('Exact duplicate has conflicting status/label: {} and {}'.format(old['path'], path))
                if split.startswith('test'):
                    if old['split'] != split:
                        raise ValueError('Locked test splits overlap; cannot change test silently')
                    # Preserve within-test multiplicity; report it explicitly.
                    removed.append({'action': 'retained_locked_test_duplicate', 'split': split,
                                    'path': relative, 'duplicate_of': old['path']})
                else:
                    removed.append({'action': 'excluded_from_new_manifest', 'split': split,
                                    'path': relative, 'duplicate_of': old['path'], 'reason': 'exact_sha256'})
                    continue
            else:
                seen[digest] = {'split': split, 'path': relative, 'label': label}
            kept.append((relative, label))
        if split in sources:
            counts = defaultdict(int)
            for _, label in kept:
                counts[label] += 1
            outputs[split] = kept
    # Replenish tiny validation classes only from clean training data. The
    # test partition is never moved. Reserve >=2 train images per class.
    for label in range(data['num_known_leaves']):
        have = sum(y == label for _, y in outputs['val_known'])
        need = max(0, 2 - have)
        candidates = sorted((r for r in outputs['train'] if r[1] == label),
                            key=lambda r: hashlib.sha256(r[0].encode()).hexdigest())
        if len(candidates) < need + 2:
            raise ValueError('Not enough disjoint train/val images for class {}'.format(label))
        for row in candidates[:need]:
            outputs['train'].remove(row)
            outputs['val_known'].append(row)
            removed.append({'action': 'reserved_train_image_for_validation', 'path': row[0], 'label': label})
    # Complete all validation before writing either manifest.
    for split, rows in outputs.items():
        path = Path(data[split])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows((p, label, i) for i, (p, label) in enumerate(rows))
    report = {'input_manifest_sha256': inputs, 'counts': {s: len(r) for s, r in outputs.items()},
              'image_files_modified': False, 'test_manifests_modified': False,
              'test_image_use': 'exact_hash_contamination_audit_only',
              'removed_or_retained_duplicates': removed,
              'output_manifest_sha256': {s: sha256(data[s]) for s in outputs}}
    write_json(Path(data['train']).parent / 'prepare_report.json', report)
    return report
