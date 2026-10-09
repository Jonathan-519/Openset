"""Independent root-first calibration on frozen, TRAIN-fitted evidence.

Staged root selection sees root scores and immutable reference anchors only.
Leaf selection cannot alter its state. Joint selection is an explicit control,
not an independent root calibration. All unsuccessful fits remain executable.
"""
import copy
import math
from collections import defaultdict

import numpy as np

from taxosafe_boundary import calibration as boundary
from taxosafe_discovery import calibration as discovery
from taxosafe_recovery import calibration as recovery
from taxosafe_dcbs.protocol import normalized_name

base, membership = discovery.base, discovery.membership
DEFAULT_SETTINGS = dict(root_known_target=.92, root_near_target=.85, seed=1)
SCHEMA_VERSION = "morphology_root_first_v1"
ROOT_SCHEMA = "morphology_root_state_v1"
CONTEXT = dict(validation_scope="exploratory_calibration_conditional_on_frozen_morphology_evidence",
               independent_model_level_validation=False, confirmatory_validation=False,
               test_used_for_fitting=False, development_reused_for_method_design=True)
METHODS = ("root_method", "leaf_method", "candidate_policy")
LEAF_ORDER = ("four gates; otherwise known>90% and precision>90%; otherwise known>90% with precision first; "
              "otherwise maximum known count. Within first two morphologys: near-source macro, known count, precision, "
              "higher leaf threshold. Root state is immutable.")


def validate_settings(settings=None):
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("Morphology settings must be a mapping")
    result = dict(DEFAULT_SETTINGS, **(settings or {}))
    if set(result) != set(DEFAULT_SETTINGS):
        raise ValueError("Unexpected Morphology setting")
    if type(result['seed']) is not int or not 0 <= result['seed'] < 2**31:
        raise ValueError("Morphology seed must be an integer in [0,2**31)")
    for key in ('root_known_target', 'root_near_target'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != DEFAULT_SETTINGS[key]:
            raise ValueError("Morphology coverage settings are preregistered")
        result[key] = float(value)
    return result


def _policy(policy):
    if policy not in ('staged', 'joint'):
        raise ValueError("Unknown Morphology policy")


def _finite(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError, TypeError):
        return False


def _data(records, meta):
    rows = list(records)
    p, c, mapping = discovery._meta(meta)
    values = defaultdict(list)
    methods = None
    required = {'root_score', 'parent_scores', 'leaf_scores', 'candidate_parent', 'candidate_leaf', 'anchor_parent', 'anchor_leaf', *METHODS}
    for row in rows:
        d = row.get('morphology')
        if not isinstance(d, dict) or set(d) != required:
            raise ValueError("Invalid Morphology evidence schema")
        if not _finite(d['root_score']):
            raise ValueError("Morphology root score must be finite")
        current = {key: d[key] for key in METHODS}
        if any(not isinstance(value, str) or not value.strip() for value in current.values()) or (methods is not None and methods != current):
            raise ValueError("Morphology method labels must be consistent nonempty strings")
        methods = current
        for key, count in [('parent_scores', p), ('leaf_scores', c)]:
            if not isinstance(d[key], (list, tuple)) or len(d[key]) != count or not all(_finite(x) for x in d[key]):
                raise ValueError("Morphology vector must be finite and taxonomy aligned: " + key)
            values[key].append(d[key])
        for key, count in [('candidate_parent', p), ('candidate_leaf', c), ('anchor_parent', p), ('anchor_leaf', c)]:
            if isinstance(d[key], bool) or not isinstance(d[key], (int, np.integer)) or not 0 <= d[key] < count:
                raise ValueError("Invalid Morphology candidate or anchor")
            values[key].append(d[key])
        if mapping[d['candidate_leaf']] != d['candidate_parent'] or mapping[d['anchor_leaf']] != d['anchor_parent']:
            raise ValueError("Morphology leaf and parent paths must be consistent")
        values['root_score'].append(d['root_score'])
    data = {key: np.asarray(value, dtype=int if key in ('candidate_parent','candidate_leaf','anchor_parent','anchor_leaf') else float)
            for key, value in values.items()}
    if rows:
        data['leaf_score'] = data['leaf_scores'][np.arange(len(rows)), data['candidate_leaf']]
        data['parent_score'] = data['parent_scores'][np.arange(len(rows)), data['candidate_parent']]
    data['methods'] = methods
    return data


def _inputs(known, near, extra, meta):
    groups = [list(known), list(near), list(extra)]
    p, c, mapping = discovery._meta(meta)
    if any(not group for group in groups):
        raise ValueError("Morphology fit requires known, near and extra DEV")
    rows = sum(groups, [])
    _data(rows, meta)
    seen = {}
    for status, group in zip(base.STATUSES, groups):
        for row in group:
            if row.get('status') != status or row.get('split') != 'val_' + status:
                raise ValueError("DEV splits only; test fitting is prohibited")
            if status != 'extra' and (type(row.get('true_parent')) is not int or not 0 <= row['true_parent'] < p):
                raise ValueError("Invalid Morphology true parent")
            if status == 'known' and (type(row.get('true_leaf')) is not int or not 0 <= row['true_leaf'] < c or mapping[row['true_leaf']] != row['true_parent']):
                raise ValueError("Invalid Morphology true leaf")
            digest, evidence = base._digest(row), discovery._hash(row['morphology'])
            if digest in seen and seen[digest] != evidence:
                raise ValueError("Duplicate image has conflicting Morphology evidence")
            seen[digest] = evidence
    return sorted(base.unique_records(rows), key=base._digest), len(rows)


def _hash_state(state, field):
    return discovery._hash({k: v for k, v in state.items() if k != field})


def reference_path(records, meta):
    """Canonical immutable reference path shared with evidence collection.

    An inconsistent original leaf is reselected only within its original
    parent, using the original root/parent/leaf ordered log probabilities.
    Source baseline decoders and supplied records are not changed.
    """
    records = list(records)
    tree, parent_count, _, mapping = base._scores(records, meta)
    selected = membership.candidate_scores(records, meta)
    parent, leaf = selected['parent'].copy(), selected['leaf'].copy()
    original_leaf = leaf.copy()
    for i, p in enumerate(parent):
        children = np.flatnonzero(mapping == p)
        if not len(children):
            raise ValueError("Reference path has a parent without known leaf children")
        if mapping[leaf[i]] != p:
            leaf[i] = children[tree[i, 1 + parent_count + children].argmax()]
    return parent, leaf, original_leaf, tree, mapping


def validate_router(router, meta):
    fields = {'schema_version','decoder','meta','settings','policy','methods','root_threshold','leaf_threshold',
        'root_state','root_state_sha256','comparison','fit_image_sha256','evidence_sha256','input_record_count','unique_image_count',
        'duplicate_record_count','fit_completed','fit_splits','status','targets_passed','best_effort','baseline_fallback','router_sha256'}.union(CONTEXT)
    if not isinstance(router, dict) or set(router) != fields or router.get('schema_version') != SCHEMA_VERSION or router.get('decoder') != 'morphology_root_first' or router.get('meta') != meta:
        raise ValueError("Morphology router schema or taxonomy changed")
    discovery._meta(meta)
    _policy(router.get('policy'))
    settings = validate_settings(router.get('settings'))
    root = router.get('root_state')
    root_fields = {'schema_version','meta','settings','policy','root_method','root_input_sha256','fit_image_sha256','threshold',
        'requested_constraints','effective_constraints','candidate_ceiling','structural_infeasible','selection','leaf_scores_used',
        'rerouted_candidates_used','root_state_sha256'}
    if (not isinstance(root, dict) or set(root) != root_fields or root.get('schema_version') != ROOT_SCHEMA or root.get('meta') != meta
            or root.get('settings') != settings or root.get('policy') != router['policy']
            or root.get('root_state_sha256') != _hash_state(root, 'root_state_sha256')
            or router.get('root_state_sha256') != root.get('root_state_sha256')):
        raise ValueError("Morphology root state binding changed")
    ids=router['fit_image_sha256']
    if (not isinstance(ids,list) or not ids or any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in ids)
            or ids!=sorted(set(ids)) or root['leaf_scores_used'] is not (router['policy']=='joint')
            or root['rerouted_candidates_used'] is not (router['policy']=='joint')):
        raise ValueError("Morphology fitting identities or selection dependency changed")
    constraints=[]
    for key in ('requested_constraints','effective_constraints','candidate_ceiling'):
        value=root[key]
        if (not isinstance(value,dict) or set(value)!= {'known','near','near_sources'} or not isinstance(value['near_sources'],dict)
                or any(type(x) is not int or not 0<=x<=len(ids) for x in [value['known'],value['near'],*value['near_sources'].values()])
                or any(not isinstance(s,str) or not s for s in value['near_sources'])):
            raise ValueError("Invalid Morphology root coverage constraints")
        constraints.append(value)
    requested,effective,ceiling=constraints
    if (set(requested['near_sources'])!=set(ceiling['near_sources']) or set(effective['near_sources'])!=set(ceiling['near_sources'])
            or any(effective[k]!=min(requested[k],ceiling[k]) for k in ('known','near'))
            or any(effective['near_sources'][s]!=min(requested['near_sources'][s],ceiling['near_sources'][s]) for s in ceiling['near_sources'])
            or root['structural_infeasible'] is not (requested!=effective)):
        raise ValueError("Morphology structural coverage cap changed")
    if (router['fit_completed'] is not True or router['baseline_fallback'] is not False
            or type(router['targets_passed']) is not bool or router['best_effort'] is not (not router['targets_passed'])
            or router['status']!=('feasible' if router['targets_passed'] else 'best_effort')
            or any(type(router[k]) is not int or router[k]<0 for k in ('input_record_count','unique_image_count','duplicate_record_count'))
            or router['unique_image_count']!=len(ids) or router['input_record_count']-len(ids)!=router['duplicate_record_count']):
        raise ValueError("Morphology fitted router status or record counts changed")
    methods = router.get('methods')
    if not isinstance(methods, dict) or set(methods) != set(METHODS) or any(not isinstance(v, str) or not v.strip() for v in methods.values()):
        raise ValueError("Invalid Morphology router methods")
    if root.get('root_method') != methods['root_method'] or root.get('fit_image_sha256') != router.get('fit_image_sha256'):
        raise ValueError("Morphology root method or fitting identities changed")
    if (not _finite(router.get('root_threshold')) or not _finite(router.get('leaf_threshold'))
            or router['root_threshold'] != root.get('threshold') or router.get('settings') != settings
            or router.get('comparison') != 'root_score>=root_threshold;then_selected_leaf_score>=leaf_threshold'
            or router.get('router_sha256') != _hash_state(router, 'router_sha256')):
        raise ValueError("Morphology thresholds or router checksum changed")


def decode_records(records, router, meta):
    validate_router(router, meta)
    records = list(records)
    data = _data(records, meta)
    if not records:
        return []
    if data['methods'] != router['methods']:
        raise ValueError("Morphology inference methods differ from fitted router")
    rt, lt = router['root_threshold'], router['leaf_threshold']
    result = []
    for i, record in enumerate(records):
        p, c = int(data['candidate_parent'][i]), int(data['candidate_leaf'][i])
        rs, ls = float(data['root_score'][i]), float(data['leaf_score'][i])
        margin = ls-lt
        if not np.isfinite(margin):
            raise ValueError("Morphology leaf margin overflow")
        if rs < rt:
            kind, parent, leaf, node = 'global_unknown', None, None, 0
        elif ls >= lt:
            kind, parent, leaf, node = 'known', p, c, 1 + len(meta['parent_names']) + c
        else:
            kind, parent, leaf, node = 'intra_unknown', p, None, 1 + p
        row = dict(record)
        row.update(prediction_type=kind, parent=parent, leaf=leaf, output_node=node,
            candidate_parent=p, candidate_leaf=c, leaf_candidate_parent=p, route_parent=p,
            candidate_parent_name=meta['parent_names'][p], candidate_leaf_name=meta['leaf_names'][c],
            anchor_parent=int(data['anchor_parent'][i]), anchor_leaf=int(data['anchor_leaf'][i]),
            root_pass=rs >= rt, selected_root_score=rs, selected_leaf_score=ls,
            selected_parent_score=float(data['parent_score'][i]), raw_selected_leaf_score=ls,
            raw_selected_parent_score=float(data['parent_score'][i]),
            root_knownness_score=rs, local_knownness_score=ls, local_known_margin=margin,
            root_threshold=rt, local_threshold=lt, leaf_threshold=lt,
            root_state_sha256=router['root_state_sha256'], decoder='morphology_root_first', policy=router['policy'],
            comparison=router['comparison'], root_score_type='independent_morphology_score',
            local_score_type='selected_leaf_score_conditional_on_root_pass',
            score_note='Continuous raw evidence scores, not probabilities; failed root cannot be rescued by a leaf')
        result.append(row)
    return result


def _groups(rows):
    result = defaultdict(list)
    for i, row in enumerate(rows):
        key = row['true_leaf'] if row['status'] == 'known' else normalized_name(str(row.get('source') or 'unspecified'))
        result[(row['status'], key)].append(i)
    return result


def _root_fit(rows, meta, settings, before):
    # Deliberately do not call _data here: root selection reads neither leaf
    # evidence nor the potentially rerouted candidate path.
    score = np.asarray([r['morphology']['root_score'] for r in rows], dtype=float)
    known = np.array([r['status'] == 'known' for r in rows]); near = np.array([r['status'] == 'intra' for r in rows])
    ck = known & np.array([r['morphology']['anchor_leaf'] == r.get('true_leaf') for r in rows])
    cn = near & np.array([r['morphology']['anchor_parent'] == r.get('true_parent') for r in rows])
    requested = dict(known=(92*int(known.sum())+99)//100, near=(85*int(near.sum())+99)//100, near_sources={})
    ceilings = dict(known=int(ck.sum()), near=int(cn.sum()), near_sources={})
    groups = _groups(rows)
    old_pass = np.array([r['prediction_type'] != 'global_unknown' for r in before])
    for (status, source), indices in groups.items():
        if status == 'intra':
            requested['near_sources'][source] = int((old_pass[indices] & cn[indices]).sum())
            ceilings['near_sources'][source] = int(cn[indices].sum())
    effective = dict(known=min(requested['known'], ceilings['known']), near=min(requested['near'], ceilings['near']),
        near_sources={s:min(value, ceilings['near_sources'][s]) for s,value in requested['near_sources'].items()})
    grid = discovery._boundaries(score)
    accepted = score[None,:] >= grid[:,None]
    safe = ((accepted & ck).sum(1) >= effective['known']) & ((accepted & cn).sum(1) >= effective['near'])
    for (status, source), indices in groups.items():
        if status == 'intra':
            safe &= (accepted[:,indices] & cn[indices]).sum(1) >= effective['near_sources'][source]
    if not safe.any():
        raise AssertionError("All-accept root must meet structurally capped coverage")
    threshold = float(grid[np.flatnonzero(safe)[-1]])
    inputs = [dict({k:r.get(k) for k in ('split','status','source','true_leaf','true_parent')},
                   image_sha256=base._digest(r),
                   **{k:r['morphology'][k] for k in ('root_score','anchor_parent','anchor_leaf','root_method')}) for r in rows]
    state = dict(schema_version=ROOT_SCHEMA, meta=copy.deepcopy(meta), settings=dict(settings), policy='staged',
        root_method=rows[0]['morphology']['root_method'], root_input_sha256=discovery._hash(inputs),
        fit_image_sha256=[base._digest(r) for r in rows], threshold=threshold,
        requested_constraints=requested, effective_constraints=effective, candidate_ceiling=ceilings,
        structural_infeasible=requested != effective, selection='highest_threshold_meeting_capped_anchor_coverage_and_near_source_support',
        leaf_scores_used=False, rerouted_candidates_used=False)
    state['root_state_sha256'] = _hash_state(state,'root_state_sha256')
    return state, grid, int(safe.sum())


def _enumerate(rows, data, roots, meta, before):
    leaf_grid = discovery._boundaries(data['leaf_score'])
    km,nm,em = [np.array([r['status'] == s for r in rows]) for s in base.STATUSES]
    kc = km & np.array([data['candidate_leaf'][i] == r.get('true_leaf') for i,r in enumerate(rows)])
    nc = nm & np.array([data['candidate_parent'][i] == r.get('true_parent') for i,r in enumerate(rows)])
    groups = _groups(rows)
    old_correct = np.array([r['prediction_type']=='global_unknown' if r['status']=='extra' else
        r['prediction_type']=='intra_unknown' and r['parent']==r.get('true_parent') for r in before])
    old_macro = {s:float(np.mean([old_correct[ix].mean() for (status,_),ix in groups.items() if status==s])) for s in ('intra','extra')}
    root_pass = data['root_score'][None,:] >= roots[:,None]
    counts, macros = [], []
    for lt in leaf_grid:
        leaves = root_pass & (data['leaf_score'][None,:] >= lt)
        parents = root_pass & ~leaves
        correct = (leaves & kc) | (parents & nc) | (~root_pass & em)
        counts.append(np.column_stack([x.sum(1) for x in (leaves & kc, parents & nc, ~root_pass & em, leaves)]))
        macros.append(np.column_stack([np.mean([correct[:,ix].mean(1) for (s,_),ix in groups.items() if s==status],axis=0) for status in base.STATUSES]))
    counts, macros = np.vstack(counts), np.vstack(macros)
    precision = np.divide(counts[:,0],counts[:,3],out=np.zeros(len(counts)),where=counts[:,3]!=0)
    return dict(counts=counts, macros=macros, precision=precision, quality=(macros.sum(1)+precision)/4.,
        regret=np.maximum(0,np.maximum(old_macro['intra']-macros[:,1],old_macro['extra']-macros[:,2])),
        root=np.tile(roots,len(leaf_grid)),leaf=np.repeat(leaf_grid,len(roots)),leaf_grid=leaf_grid,
        totals=[int(x.sum()) for x in (km,nm,em)])


def _choose_leaf(values):
    k,n,e,l = values['counts'].T
    four,kp,known = boundary._masks(values['counts'],values['totals'])
    tier,mask = ('four_gates',four) if four.any() else ('known_precision',kp) if kp.any() else ('known_only',known) if known.any() else ('maximum_known',k==k.max())
    ppv,near = values['precision'],values['macros'][:,1]
    keys = (values['leaf'],ppv,k,near)
    if tier=='known_only': keys += (ppv,)
    if tier=='maximum_known': keys += (k,)
    ids = np.flatnonzero(mask)
    index = int(ids[np.lexsort(tuple(x[ids] for x in keys))[-1]])
    return index, dict(selected_tier=tier,selection_rule=LEAF_ORDER,
        feasible_counts=dict(four_gates=int(four.sum()),known_precision=int(kp.sum()),known=int(known.sum()),maximum_known=int(k.max())))


def _root_report(rows, data, state, before, frozen):
    accepted = data['root_score'] >= state['threshold']
    groups = _groups(rows)
    coverage = {}
    for status in base.STATUSES:
        ix = [i for i,r in enumerate(rows) if r['status']==status]
        coverage[status] = dict(total=len(ix),accepted=int(accepted[ix].sum()),rejected=int((~accepted[ix]).sum()))
    ck=np.array([r['status']=='known' and r['morphology']['anchor_leaf']==r.get('true_leaf') for r in rows])
    cn=np.array([r['status']=='intra' and r['morphology']['anchor_parent']==r.get('true_parent') for r in rows])
    coverage.update(correct_known_anchor_retained=int((accepted&ck).sum()),correct_near_parent_anchor_retained=int((accepted&cn).sum()))
    near,extra={},{}
    for (status,source),ix in groups.items():
        if status=='intra':
            near[source]=dict(total=len(ix),all_accepted=int(accepted[ix].sum()),correct_parent_retained=int((accepted[ix]&cn[ix]).sum()),
                requested=state['requested_constraints']['near_sources'][source],effective=state['effective_constraints']['near_sources'][source])
        if status=='extra':
            old=sum(before[i]['prediction_type']=='global_unknown' for i in ix);new=int((~accepted[ix]).sum())
            extra[source]=dict(total=len(ix),d05_rejected=old,rejected=new,not_worse=new>=old)
    effective=state['effective_constraints'];requested=state['requested_constraints']
    met=lambda constraints: (coverage['correct_known_anchor_retained']>=constraints['known']
        and coverage['correct_near_parent_anchor_retained']>=constraints['near']
        and all(near[s]['correct_parent_retained']>=n for s,n in constraints['near_sources'].items()))
    return dict(state=state,root_state_sha256=state['root_state_sha256'],threshold=state['threshold'],
        requested_constraints=state['requested_constraints'],effective_constraints=state['effective_constraints'],
        structural_infeasible=state['structural_infeasible'],candidate_ceiling=state['candidate_ceiling'],
        coverage=coverage,per_near_source=near,extra_source_comparison=extra,
        extra_sources_preserved=all(x['not_worse'] for x in extra.values()),
        effective_coverage_passed=bool(met(effective)),requested_coverage_passed=bool(met(requested)),
        extra_gate_passed=10*coverage['extra']['rejected']>9*coverage['extra']['total'],
        root_selection_uses_leaf_scores=not frozen,frozen_before_leaf=frozen,
        coverage_scope='original reference anchors; rerouted candidate potential is reported separately by leaf_stage',
        coverage_is_final_accuracy_guarantee=False)


def fit_router(known,near,extra,meta,settings=None,policy='staged',reference_records=None,d05_records=None):
    _policy(policy);settings=validate_settings(settings)
    rows,input_count=_inputs(known,near,extra,meta);data=_data(rows,meta)
    before,d05=recovery._d05(rows,d05_records,meta)
    if d05 is None: raise ValueError("Morphology calibration requires D05 records/router")
    original,_=discovery._reference(rows,reference_records,meta)
    if original is not None:
        anchor_parent,anchor_leaf,_,_,_=reference_path(original,meta)
        if any(r['morphology']['anchor_parent']!=int(anchor_parent[i]) or r['morphology']['anchor_leaf']!=int(anchor_leaf[i]) for i,r in enumerate(rows)):
            raise ValueError("Morphology anchors differ from the canonical original reference path")
    root,root_grid,safe_count=_root_fit(rows,meta,settings,before)
    search_roots=np.array([root['threshold']]) if policy=='staged' else root_grid
    values=_enumerate(rows,data,search_roots,meta,before)
    if policy=='joint':
        index,selection=boundary._choose(values['counts'],values['quality'],values['regret'],values['root'],values['leaf'],values['totals'])
        root.update(policy='joint',threshold=float(values['root'][index]),selection='joint_root_leaf_boundary_KP_control',
                    leaf_scores_used=True,rerouted_candidates_used=True)
        root['root_state_sha256']=_hash_state(root,'root_state_sha256')
    else: index,selection=_choose_leaf(values)
    feasible=bool(boundary._masks(values['counts'][[index]],values['totals'])[0][0])
    router=dict(schema_version=SCHEMA_VERSION,decoder='morphology_root_first',meta=copy.deepcopy(meta),settings=settings,policy=policy,
        methods=data['methods'],root_threshold=root['threshold'],leaf_threshold=float(values['leaf'][index]),
        root_state=root,root_state_sha256=root['root_state_sha256'],
        comparison='root_score>=root_threshold;then_selected_leaf_score>=leaf_threshold',
        fit_image_sha256=[base._digest(r) for r in rows],evidence_sha256=discovery._hash([r['morphology'] for r in rows]),
        input_record_count=input_count,unique_image_count=len(rows),duplicate_record_count=input_count-len(rows),
        fit_completed=True,fit_splits=['val_known','val_intra','val_extra'],status='feasible' if feasible else 'best_effort',
        targets_passed=feasible,best_effort=not feasible,baseline_fallback=False,**CONTEXT)
    router['router_sha256']=_hash_state(router,'router_sha256')
    predictions=decode_records(rows,router,meta);report=base.evaluate_records(predictions,meta)
    if [report['counts'][k] for k in ('known_correct','intra_correct','extra_correct','leaf_outputs')]!=values['counts'][index].tolist():
        raise AssertionError("Morphology search and inference disagree")
    paired=discovery._paired(original,predictions,meta);recover=recovery._recovery(before,predictions,meta)
    root_report=_root_report(rows,data,root,before,policy=='staged')
    root_pass=data['root_score']>=root['threshold']
    known_possible=sum(bool(root_pass[i]) and r['status']=='known' and int(data['candidate_leaf'][i])==r.get('true_leaf') for i,r in enumerate(rows))
    near_possible=sum(bool(root_pass[i]) and r['status']=='intra' and int(data['candidate_parent'][i])==r.get('true_parent') for i,r in enumerate(rows))
    report.update(**CONTEXT,status=router['status'],best_effort=router['best_effort'],paired_audit=paired,
        known_count_preserved=None if paired is None else paired['known_count_preserved'],known_recovery=recover,recovery_passed=recover['passed'],
        morphology_policy=dict(name=policy,settings=settings,**selection,baseline_fallback=False,test_allowed_after_failed_gates=True),
        root_stage=root_report,leaf_stage=dict(root_state_sha256=root['root_state_sha256'],**selection,
            root_admitted_correct_candidate_known=known_possible,root_admitted_correct_candidate_near=near_possible,
            actual_candidate_known_ceiling=sum(r['status']=='known' and int(data['candidate_leaf'][i])==r.get('true_leaf') for i,r in enumerate(rows)),
            actual_candidate_near_ceiling=sum(r['status']=='intra' and int(data['candidate_parent'][i])==r.get('true_parent') for i,r in enumerate(rows)),
            additional_correct_known_loss=known_possible-report['counts']['known_correct'],
            additional_correct_near_loss=near_possible-report['counts']['intra_correct'],
            extra_correct_invariant_to_leaf_threshold=True,root_threshold_changed_by_leaf=policy=='joint'),
        exact_threshold_search=dict(root_grid=root_grid.tolist(),searched_root_grid=search_roots.tolist(),leaf_grid=values['leaf_grid'].tolist(),
            candidate_count=len(values['counts']),root_coverage_feasible_count=safe_count,
            includes_all_accept_and_reject=True,root_selection_includes_all_accept_and_reject=True,
            leaf_selection_includes_all_accept_and_reject=True,joint_cartesian_grid_exhaustive=policy=='joint',
            root_fixed_for_leaf=policy=='staged',scope='all finite empirical boundaries on this policy allowed axes; no population guarantee'))
    for k in ('input_record_count','unique_image_count','duplicate_record_count'): report[k]=router[k]
    discovery._hash(report)
    return router,report


def crossfit_audit(known,near,extra,meta,settings=None,policy='staged',reference_records=None,d05_records=None):
    _policy(policy);settings=validate_settings(settings)
    rows,_=_inputs(known,near,extra,meta)
    _,reference=discovery._reference(rows,reference_records,meta);_,d05=recovery._d05(rows,d05_records,meta)
    if reference is None or d05 is None:
        return dict(**CONTEXT,passed=False,recovery_passed=False,complete=False,status='not_evaluable',reason='Raw reference and D05 bundles required')
    target={base._digest(r):r for r in rows};original={base._digest(r):r for r in reference['records']};old={base._digest(r):r for r in d05['records']}
    folds=discovery._folds(rows,settings);before=[];d05_before=[];after=[]
    for fold in folds:
        fit,held=fold['fit_image_sha256'],fold['held_image_sha256']
        fitted=[target[h] for h in fit];rf=[original[h] for h in fit];df=[old[h] for h in fit]
        groups=lambda rs:[[r for r in rs if r['status']==s] for s in base.STATUSES]
        if not held or any(not group for group in groups(fitted)):
            fold.update(status='not_evaluable',reason='empty held or missing fit status');continue
        ref_router=membership.calibrate(*groups(rf),meta,dict(reference['calibration_settings'],source_loo=False))
        d05_router,_=discovery.fit_router(*groups(df),meta,d05['router']['settings'],'global')
        router,report=fit_router(*groups(fitted),meta,settings,policy,
            dict(records=rf,router=ref_router,calibration_settings=reference['calibration_settings']),dict(records=df,router=d05_router))
        b=membership.decode_records([original[h] for h in held],ref_router,meta)
        d=discovery.decode_records([old[h] for h in held],d05_router,meta)
        a=decode_records([target[h] for h in held],router,meta)
        before.extend(b);d05_before.extend(d);after.extend(a)
        fold.update(status='completed',reference_fit_image_sha256=ref_router['fit_image_sha256'],d05_fit_image_sha256=d05_router['fit_image_sha256'],
            target_fit_image_sha256=router['fit_image_sha256'],reference_calibration_sha256=ref_router['calibration_sha256'],
            d05_router_sha256=d05_router['router_sha256'],router_sha256=router['router_sha256'],root_state_sha256=router['root_state_sha256'],
            root_threshold=router['root_threshold'],leaf_threshold=router['leaf_threshold'],fit_targets_passed=report['targets_passed'],
            fit_morphology_policy=report['morphology_policy'],root_stage=report['root_stage'],leaf_stage=report['leaf_stage'],
            held_report=discovery._paired(b,a,meta),held_known_recovery=recovery._recovery(d,a,meta))
    identities=[base._digest(r) for r in after]
    if len(identities)!=len(set(identities)):raise AssertionError("Morphology OOF evaluates image twice")
    complete=set(identities)==set(target)
    paired=discovery._paired(before,after,meta) if after else None
    recover=recovery._recovery(d05_before,after,meta) if after else dict(passed=False,status='not_evaluable')
    recover['passed']=bool(complete and recover['passed'])
    scored=None if paired is None else paired['selected']
    return dict(**CONTEXT,schema_version='morphology_crossfit_v1',policy=policy,folds=folds,
        passed=bool(complete and paired and scored['targets_passed'] and paired['known_count_preserved']),
        recovery_passed=recover['passed'],complete=complete,status='completed' if complete else 'not_evaluable',
        unique_image_count=len(rows),evaluated_image_count=len(after),report=scored,
        counts=None if paired is None else dict(reference=paired['reference']['counts'],selected=scored['counts']),
        paired_audit=paired,known_count_preserved=None if paired is None else paired['known_count_preserved'],
        known_recovery=recover,reference_predictions=before,d05_predictions=d05_before,predictions=after,
        output_used_for_threshold_selection=False,held_data_used_for_thresholds=False,source_protection_is_diagnostic=True,
        qualification_rule='complete conditional OOF; four gates; known count at least fold-refit original reference',
        interpretation='Frozen model and TRAIN evidence; all root/leaf thresholds and both reference routers refitted using each fit partition only; no conformal guarantee')
