"""CLIP integration, staged learning, and a frozen full-cascade router."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .core import OpenProjectionHead, fit_support, synthesize_shell, boundary_loss, distances, calibrate_known
from .protocol import sha256, write_json, write_jsonl, partition_known, training_fingerprint


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_torch(path, device='cpu'):
    # Only load locally created or trusted project checkpoints.
    return torch.load(path, map_location=device, weights_only=True)


def metadata(loaders, device):
    from train_taxosafe import build_hier_meta
    return build_hier_meta(loaders['param_names'], loaders['leaf_nodes'],
                           loaders['intnl_nodes'], loaders['sublabels'].to(device),
                           device, expected_num_leaves=None)


def build(cfg, splits, device, checkpoint=None):
    from loader import get_dataloader
    from models import get_model
    loaders = get_dataloader(cfg['data'], splits, cfg['data']['batch_size'])
    meta = metadata(loaders, device)
    if meta['num_leaves'] != cfg['data']['num_known_leaves']:
        raise ValueError('Configured leaf count differs from hierarchy')
    # MaPLe initializes its shallow projection in fp16 even for fp32 config.
    model = get_model(cfg['model'], meta['leaf_names']).float().to(device)
    if checkpoint:
        state = load_torch(checkpoint, device)
        state = state.get('state_dict', state)
        model.load_state_dict(state, strict=True)
    return model, loaders, meta


def partition_loader(loader, indices):
    # Preserve original dataset indices used by score collection.
    return torch.utils.data.DataLoader(loader.dataset, batch_size=loader.batch_size,
        sampler=indices, num_workers=loader.num_workers, collate_fn=loader.collate_fn,
        pin_memory=True, drop_last=False)


def validation_parts(cfg, loader):
    a, b = partition_known(loader.dataset.data, loader.dataset.target,
                            cfg['boundaryshell']['partition_seed'])
    return partition_loader(loader, a), partition_loader(loader, b), a, b


def collect(model, loader, meta, device, status='known'):
    from taxosafe_eval_utils import collect_score_records
    model.eval()
    return collect_score_records(model, loader, status, meta, device,
                                  include_image_features=True, include_vectors=True)


def tensors(rows, device):
    x = torch.tensor([r['image_feature'] for r in rows], device=device, dtype=torch.float32)
    y = torch.tensor([r['true_leaf'] for r in rows], device=device, dtype=torch.long)
    return x, y


def train_classifier(cfg, run, device):
    seed_all(int(cfg.get('seed', 1)))
    from engine_taxosafe import score_label_sets
    from losses.taxosafe_loss import morphology_pool, taxonomy_weighted_contrastive_loss
    stage = run / 'classifier'
    stage.mkdir(exist_ok=True)
    if (stage / 'best.pth').exists():
        raise FileExistsError('Classifier exists; use --stage open or a new run directory')
    model, loaders, meta = build(cfg, ['train', 'val_known'], device)
    params = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_('prompt_learner' in name or 'VPT' in name)
        if parameter.requires_grad:
            params.append(parameter)
    selection, _, a, b = validation_parts(cfg, loaders['val_known'])
    write_json(run / 'validation_partition.json', {'selection_indices': a, 'calibration_indices': b,
        'seed': cfg['boundaryshell']['partition_seed'], 'source_sha256': sha256(cfg['data']['val_known'])})
    opt_cfg = cfg['classifier']
    optimizer = torch.optim.SGD(params, lr=opt_cfg['lr'], momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt_cfg['epochs'])
    best, stale, history = -1., 0, []
    for epoch in range(opt_cfg['epochs']):
        sampler = loaders.get('train_batch_sampler')
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        model.train()
        loss_sum, count = 0., 0
        for value in loaders['train']:
            image, y = value[0].to(device), value[1].long().to(device)
            p = meta['leaf_to_parent'][y]
            scores, _, reps = score_label_sets(model, image,
                {'leaf': meta['leaf_names'], 'parent': meta['parent_names']}, True)
            loss = F.cross_entropy(scores['leaf'], y) + opt_cfg['parent_weight'] * F.cross_entropy(scores['parent'], p)
            if opt_cfg['morphology_weight']:
                morphology = morphology_pool(reps['spatial'], reps['text']['leaf'], p, meta)
                loss = loss + opt_cfg['morphology_weight'] * taxonomy_weighted_contrastive_loss(morphology, y, p)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite classifier loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            count += len(y)
        scheduler.step()  # exactly once per epoch
        rows = collect(model, selection, meta, device)
        accuracy = float(np.mean([r['global_pred_leaf'] == r['true_leaf'] for r in rows]))
        event = {'epoch': epoch + 1, 'loss': loss_sum / count, 'selection_closed_accuracy': accuracy,
                 'learning_rate': optimizer.param_groups[0]['lr']}
        history.append(event)
        print(json.dumps(event), flush=True)
        write_jsonl(stage / 'history.jsonl', history)
        if accuracy > best:
            best, stale = accuracy, 0
            torch.save(model.state_dict(), stage / 'best.pth')
            write_json(stage / 'selection.json', dict(event, checkpoint_selection='known_selection_only',
                training_fingerprint=training_fingerprint(cfg), calibration_rows_evaluated=False))
        else:
            stale += 1
        if stale >= opt_cfg['patience']:
            break
    print('Classifier finished. Best known selection accuracy: {:.4%}'.format(best), flush=True)


def support_settings(cfg):
    b = cfg['boundaryshell']
    return dict(quantile=b['support_quantile'], shrinkage=b['shrinkage'], min_radius=b['min_radius'])


@torch.no_grad()
def project_all(head, x, batch=1024):
    return torch.cat([head(v) for v in x.split(batch)])


def score_rows(rows, head, support, device):
    x = torch.tensor([r['image_feature'] for r in rows], dtype=torch.float32, device=device)
    with torch.no_grad():
        z = project_all(head, x)
        ratios = distances(z, support['centers']) / support['radius'][None, :]
        leaf = torch.tensor([r['global_pred_leaf'] for r in rows], device=device)
        local = -ratios.gather(1, leaf[:, None]).squeeze(1)
        manifold = -(distances(z, support['parent_centers']) / support['parent_radius'][None, :]).min(1).values
    semantic = torch.tensor([r['parent_score'] for r in rows], device=device)
    return local, semantic, manifold


def train_open(cfg, run, device):
    seed_all(int(cfg.get('seed', 1)) + 10000)
    b = cfg['boundaryshell']
    out = run / 'open'
    out.mkdir(exist_ok=True)
    if (out / 'head.pth').exists():
        raise FileExistsError('Open model exists; use a new run directory')
    checkpoint = run / 'classifier/best.pth'
    model, loaders, meta = build(cfg, ['train_reference', 'val_known'], device, checkpoint)
    for p in model.parameters():
        p.requires_grad_(False)
    selection, _, _, _ = validation_parts(cfg, loaders['val_known'])
    train_rows = collect(model, loaders['train_reference'], meta, device)
    select_rows = collect(model, selection, meta, device)
    x, y = tensors(train_rows, device)
    del model, loaders
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    mapping = meta['leaf_to_parent']
    raw_support = fit_support(x, y, mapping, **support_settings(cfg))
    head_args = dict(input_dim=x.shape[1], output_dim=b['output_dim'], hidden_dim=b['hidden_dim'])
    head = OpenProjectionHead(**head_args).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=b['lr'], weight_decay=1e-4)
    correct = torch.tensor([r['global_pred_leaf'] == r['true_leaf'] for r in select_rows], device=device)
    train_history = []
    # Fixed epoch count: no unknown dev or calibration scores select the head.
    for epoch in range(b['epochs']):
        head.eval()
        support = fit_support(project_all(head, x), y, mapping, **support_settings(cfg))
        synthetic, parent, source, stats = synthesize_shell(x, y, raw_support,
            mode=b['neighborhood'], k=b['neighbors'], margin=b['shell_margin'],
            noise=b['shell_noise'], attempts=b['shell_attempts'])
        if len(synthetic) < b['min_synthetic']:
            write_json(out / 'generator_failure.json', dict(stats, epoch=epoch + 1))
            raise RuntimeError('Too few valid shell examples; see open/generator_failure.json. No silent mixup fallback.')
        head.train()
        events = []
        weights = 1. / torch.bincount(y)[y].float()
        balanced_ids = torch.multinomial(weights, len(x), replacement=True)
        for ids in balanced_ids.split(b['batch_size']):
            u_ids = torch.randint(len(synthetic), (len(ids),), device=device)
            loss, info = boundary_loss(head, x[ids], y[ids], synthetic[u_ids], parent[u_ids], support,
                reject_margin=b['reject_margin'], parent_weight=b['parent_weight'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite boundary loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            events.append(dict(info, loss=float(loss.detach())))
        event = {'epoch': epoch + 1, **stats,
                 **{key: float(np.mean([e[key] for e in events])) for key in events[0]}}
        train_history.append(event)
        print(json.dumps(event), flush=True)
        write_jsonl(out / 'history.jsonl', train_history)
    head.eval()
    support = fit_support(project_all(head, x), y, mapping, **support_settings(cfg))
    scores = score_rows(select_rows, head, support, device)
    selection_diagnostic = {'closed_accuracy': float(correct.float().mean()), 'head_selection': 'fixed_epoch_count'}
    try:
        selection_diagnostic.update(calibrate_known(*scores, correct, **cfg['calibration']))
    except ValueError as error:
        selection_diagnostic.update(gate_passed=False, reason=str(error))
    payload = {'head_args': head_args, 'state_dict': head.cpu().state_dict(),
        'support': {k: v.cpu() for k, v in support.items()},
        'classifier_sha256': sha256(checkpoint), 'training_fingerprint': training_fingerprint(cfg),
        'leaf_names': meta['leaf_names'], 'parent_names': meta['parent_names']}
    torch.save(payload, out / 'head.pth')
    write_json(out / 'summary.json', {'selection_diagnostic': selection_diagnostic,
        'generator_space': 'frozen_classifier_embedding', 'inference_space': 'learned_open_embedding',
        'head_args': head_args, 'train_count': len(x), 'classifier_sha256': payload['classifier_sha256'],
        'real_unknown_loaded': False, 'test_loaded': False,
        'class_support': [{'name': meta['leaf_names'][c], 'count': int(support['counts'][c]),
                          'radius': float(support['radius'][c])} for c in range(len(mapping))]})


def load_open(cfg, run, device):
    payload = load_torch(run / 'open/head.pth', device)
    if payload['classifier_sha256'] != sha256(run / 'classifier/best.pth'):
        raise ValueError('Classifier changed after open-head training')
    if payload['training_fingerprint'] != training_fingerprint(cfg):
        raise ValueError('Training/validation/hierarchy changed after open-head training')
    head = OpenProjectionHead(**payload['head_args']).to(device)
    head.load_state_dict(payload['state_dict'], strict=True)
    head.eval()
    return head, payload['support'], payload


def stripped(rows):
    return [{k: v for k, v in r.items() if k not in ('image_feature', 'leaf_cosine', 'parent_cosine')}
            for r in rows]


def calibrate(cfg, run, device):
    out = run / 'calibration'
    out.mkdir(exist_ok=True)
    if (out / 'thresholds.json').exists():
        raise FileExistsError('Frozen thresholds already exist')
    head, support, _ = load_open(cfg, run, device)
    model, loaders, meta = build(cfg, ['val_known'], device, run / 'classifier/best.pth')
    _, calibration_loader, _, _ = validation_parts(cfg, loaders['val_known'])
    rows = collect(model, calibration_loader, meta, device)
    scores = score_rows(rows, head, support, device)
    correct = torch.tensor([r['global_pred_leaf'] == r['true_leaf'] for r in rows], device=device)
    report_rows = stripped(rows)
    for i, row in enumerate(report_rows):
        row.update(local_knownness_score=float(scores[0][i]), root_semantic=float(scores[1][i]),
                   root_manifold=float(scores[2][i]))
    write_jsonl(out / 'validation_scores.jsonl', report_rows)
    try:
        thresholds = calibrate_known(*scores, correct, **cfg['calibration'])
    except ValueError as error:
        write_json(out / 'gate_failure.json', {'gate_passed': False, 'reason': str(error),
            'known_closed_accuracy': float(correct.float().mean()), 'test_loaded': False})
        raise
    thresholds['metadata'] = {'classifier_sha256': sha256(run / 'classifier/best.pth'),
        'head_sha256': sha256(run / 'open/head.pth'), 'training_fingerprint': training_fingerprint(cfg),
        'calibration_split': 'heldout_half_of_val_known', 'real_unknown_loaded': False,
        'test_loaded': False, 'new_blind_test_claim_allowed': False}
    write_json(out / 'thresholds.json', thresholds)
    print(json.dumps(thresholds, indent=2), flush=True)


def route(rows, scores, thresholds, support, parent_names, leaf_names):
    local, semantic, manifold = [v.detach().cpu().tolist() for v in scores]
    mapping = support['leaf_to_parent'].cpu().tolist()
    outputs = stripped(rows)
    for i, row in enumerate(outputs):
        leaf = row['global_pred_leaf']
        rejected_root = semantic[i] < thresholds['semantic_threshold'] and manifold[i] < thresholds['manifold_threshold']
        accepted = not rejected_root and local[i] >= thresholds['local_threshold']
        # Leaf classification stays global. Its taxonomy parent is consistent
        # by construction; a rejected leaf falls back to the semantic parent.
        parent = mapping[leaf] if accepted else row['pred_parent']
        kind = 'global_unknown' if rejected_root else ('known' if accepted else 'intra_unknown')
        row.update(prediction_type=kind, candidate_leaf=leaf, candidate_parent=parent,
                   candidate_parent_name=parent_names[parent], candidate_leaf_name=leaf_names[leaf],
                   parent=None if rejected_root else parent, leaf=leaf if accepted else None,
                   local_knownness_score=local[i], root_semantic=semantic[i], root_manifold=manifold[i],
                   root_knownness_score=max(semantic[i] - thresholds['semantic_threshold'],
                                            manifold[i] - thresholds['manifold_threshold']))
        if row['true_parent'] is not None:
            row['true_parent_name'] = parent_names[row['true_parent']]
        if row['true_leaf'] is not None:
            row['true_leaf_name'] = leaf_names[row['true_leaf']]
    return outputs


def test(cfg, run, device):
    from metrics_open import evaluate_open_set
    out = run / 'test'
    out.mkdir(exist_ok=True)
    if (out / 'metrics.json').exists() or (out / 'predictions.jsonl').exists():
        raise FileExistsError('Test outputs already exist; they will not be overwritten')
    threshold_path = run / 'calibration/thresholds.json'
    thresholds = json.loads(threshold_path.read_text())
    if not thresholds.get('gate_passed') or thresholds['metadata']['test_loaded'] is not False:
        raise ValueError('No valid known-only calibration gate')
    if thresholds['metadata']['head_sha256'] != sha256(run / 'open/head.pth'):
        raise ValueError('Open model changed after calibration')
    head, support, payload = load_open(cfg, run, device)
    if thresholds['metadata']['classifier_sha256'] != payload['classifier_sha256']:
        raise ValueError('Calibration classifier hash mismatch')
    model, loaders, meta = build(cfg, ['test_known', 'test_intra', 'test_extra'], device,
                                  run / 'classifier/best.pth')
    if payload['leaf_names'] != meta['leaf_names'] or payload['parent_names'] != meta['parent_names']:
        raise ValueError('Label ordering changed')
    outputs = []
    for split, status in [('test_known', 'known'), ('test_intra', 'intra'), ('test_extra', 'extra')]:
        rows = collect(model, loaders[split], meta, device, status)
        outputs.extend(route(rows, score_rows(rows, head, support, device), thresholds,
                             support, meta['parent_names'], meta['leaf_names']))
    metrics = evaluate_open_set(*[[r for r in outputs if r['status'] == s] for s in ('known', 'intra', 'extra')])
    # Correct the generic OSCR candidate convention: detection quality must
    # use the fixed global classifier, independent of the threshold's fallback.
    from metrics_open import _near_detection_metrics
    oscr_rows = [dict(r, candidate_parent=int(support['leaf_to_parent'][r['global_pred_leaf']])) for r in outputs]
    metrics['near_open_set'] = _near_detection_metrics(
        [r for r in oscr_rows if r['status'] == 'known'], [r for r in oscr_rows if r['status'] == 'intra'])
    metrics['known']['semantic_parent_accuracy'] = float(np.mean([
        r['pred_parent'] == r['true_parent'] for r in outputs if r['status'] == 'known']))
    acc = metrics['known']['end_to_end_leaf_accuracy']
    closed = metrics['known']['global_leaf_accuracy']
    metrics['research_gate'] = {'known_e2e_strictly_above_90': acc > cfg['calibration']['target'],
        'drop_within_budget': closed - acc <= cfg['calibration']['max_drop'] + 1e-12,
        'passed': acc > cfg['calibration']['target'] and closed - acc <= cfg['calibration']['max_drop'] + 1e-12}
    metrics['metadata'] = {'classifier_sha256': payload['classifier_sha256'],
        'head_sha256': sha256(run / 'open/head.pth'), 'thresholds_sha256': sha256(threshold_path),
        'protocol': cfg['protocol'], 'new_blind_test_claim_allowed': False,
        'test_manifest_sha256': {s: sha256(cfg['data'][s]) for s in ('test_known', 'test_intra', 'test_extra')},
        'rate_unit': 'fraction_[0,1]', 'calibration_does_not_guarantee_test_accuracy': True}
    write_jsonl(out / 'predictions.jsonl', outputs)
    write_json(out / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
