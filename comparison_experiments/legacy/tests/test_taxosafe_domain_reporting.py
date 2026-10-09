"""Report contracts use actual saved artifacts, without pretending to train models."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from taxosafe_domain import protocol, reporting, runner
from taxosafe_support import calibration as base

META = dict(leaf_names=['a', 'b', 'c', 'd'], parent_names=['P', 'Q'], leaf_to_parent=[0, 0, 1, 1])


def records(stage='val', pass_gates=False, domain=True):
    result = []
    for status in base.STATUSES:
        for i in range(2):
            name = stage + '_' + status + str(i)
            kind = 'known' if status == 'known' else 'intra_unknown' if status == 'intra' and pass_gates else 'global_unknown'
            row = dict(image_sha256=hashlib.sha256(name.encode()).hexdigest(), split=stage+'_'+status,
                path=name+'.png', source=str(i) if status == 'known' else status+str(i), status=status,
                true_leaf=i if status == 'known' else None, true_parent=0 if status != 'extra' else None,
                prediction_type=kind, leaf=i if kind == 'known' else None, parent=0 if kind != 'global_unknown' else None,
                candidate_leaf=i, candidate_parent=0)
            if domain:
                row['domain'] = dict(root_score=float(i), parent_scores=[1., 0.], leaf_scores=[1., 2., 0., 0.],
                    candidate_parent=0, candidate_leaf=i, anchor_parent=0, anchor_leaf=i,
                    root_method='dual', leaf_method='conditional', candidate_policy='reference_path')
            result.append(row)
    return result + [dict(copy.deepcopy(result[0]), path='alias.png')]


class DomainReporting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'suite'
        self.cfg = copy.deepcopy(protocol.DEFAULTS)
        self.snapshot = runner._initialize(self.root, self.cfg, {'binding':{'directory':'immutable_D05'}}, 'cpu')

    def _file(self, directory, name, value):
        path = directory / name
        if isinstance(value, bytes):
            path.write_bytes(value)
        elif name.endswith('.jsonl'):
            protocol.write_records(path, value)
        else:
            protocol.write_json(path, value)
        return dict(path=name, sha256=protocol.file_hash(path))

    def _stage(self, arm_id, stage, pass_gates=False, passed=False, rows=None):
        directory = self.root / 'arms' / arm_id / stage
        directory.mkdir(parents=True)
        arm = next(a for a in self.cfg['arms'] if a['id'] == arm_id)
        receipt = dict(schema_version=protocol.SCHEMA_VERSION, arm_id=arm_id, stage=stage,
            signature=self.snapshot['signature'], source_binding=self.snapshot['source_binding'], artifacts={})
        if stage == 'training':
            receipt.update(optimizer_steps=0, weight_source=arm.get('weight_source', arm_id),
                fit_report={'synthetic_reporting_fixture':True})
            receipt['model'] = self._file(directory, 'model.pth', b'report fixture only')
            receipt['artifacts']['model'] = receipt['model']
        else:
            rows = records('val' if stage == 'calibration' else 'test', pass_gates,
                           domain=arm_id not in ('H00_reference', 'H01_d05')) if rows is None else rows
            full = None if arm_id == 'H00_reference' else dict(passed=passed, recovery_passed=False,
                validation_scope='conditional_threshold_crossfit', folds=[{'predictions':[{'large':'fixture'}]}])
            compact, digest = reporting.compact_crossfit(full), None
            if stage == 'calibration':
                artifact = self._file(directory, 'crossfit_audit.json', full)
                receipt['artifacts']['crossfit'] = artifact
                digest = artifact['sha256']
            else:
                previous = protocol.read_json(self.root / 'arms' / arm_id / 'calibration/completed.json')
                compact = previous['summary']['crossfit_audit']
                digest = previous['summary']['crossfit_audit_sha256']
                receipt['calibration_crossfit_audit_sha256'] = digest
            report = dict(base.evaluate_records(rows, META), schema_version='domain_evaluation_v1',
                root_stage_outcomes=reporting.root_stage_summary(rows, META), crossfit_audit=compact,
                crossfit_audit_sha256=digest, domain_policy={'name':arm['policy'], 'reason':'synthetic reporting fixture'})
            receipt.update(meta=META, summary=report, targets_passed=report['targets_passed'])
            router = {'root_state_sha256':'shared-staged-root' if arm['policy'] == 'staged' else 'joint-root'}
            for key, name, value in [('predictions','predictions.jsonl',rows),('scores','scores.jsonl',rows),
                                    ('report','report.json',report),('router','router.json',router)]:
                receipt['artifacts'][key] = self._file(directory, name, value)
            if stage == 'test':
                receipt['calibration_receipt_sha256'] = protocol.file_hash(self.root / 'arms' / arm_id / 'calibration/completed.json')
        protocol.write_json(directory / 'completed.json', receipt)
        runner._complete_stage(self.root, arm_id, stage, self.snapshot)
        return receipt

    def _all_dev(self, pass_arms=(), crossfit_arms=(), failed=()):
        for arm in self.cfg['arms']:
            arm_id = arm['id']
            if arm_id in failed:
                runner._failure(self.root, arm_id, 'training', 'synthetic unavailable model')
                continue
            self._stage(arm_id, 'training')
            self._stage(arm_id, 'calibration', arm_id in pass_arms, arm_id in crossfit_arms)

    def test_root_outcomes_keep_root_errors_separate_and_ignore_alias_weight(self):
        rows = records(pass_gates=True)
        rows[0].update(prediction_type='global_unknown', leaf=None, parent=None)
        rows[-1] = dict(rows[0], path='alias.png')
        rows[1].update(leaf=0)
        rows[2].update(prediction_type='known', leaf=0, parent=0)
        result = reporting.root_stage_summary(rows, META)
        self.assertEqual((result['known_count'],result['near_count'],result['extra_count']), (2,2,2))
        self.assertEqual(result['known_root_rejected'], 1)
        self.assertEqual(result['known_leaf_wrong_after_root'], 1)
        self.assertEqual(result['unknown_leaf_false_accept_after_root'], 1)
        self.assertEqual(result['extra_root_rejected'], 2)
        self.assertEqual(result['near_parent_correct_after_root'], 1)

    def test_same_hash_domain_conflict_fails_closed(self):
        rows = records()
        rows[-1]['domain']['root_score'] += 1
        with self.assertRaisesRegex(ValueError, 'conflicting domain evidence'):
            reporting.root_stage_summary(rows, META)

    def test_compact_crossfit_drops_full_fold_predictions_but_preserves_decision(self):
        full = dict(passed=False,recovery_passed=True,folds=[{'predictions':list(range(1000))}],
            predictions=list(range(1000)), known_recovery=dict(passed=True,checks={'x':True},counts={'k':2},paired_audit=list(range(1000))))
        compact = reporting.compact_crossfit(full)
        self.assertEqual(compact['fold_count'], 1)
        self.assertEqual(compact['known_recovery'], {'passed':True,'checks':{'x':True},'counts':{'k':2}})
        self.assertNotIn('folds', compact)
        self.assertNotIn('predictions', compact)
        self.assertLess(len(str(compact)), 300)

    def test_deployment_needs_dev_four_gates_known_count_and_conditional_crossfit(self):
        self._all_dev(pass_arms=['H08_dual_conditional'], crossfit_arms=[])
        frozen = reporting.freeze_dev_selection(self.root)
        self.assertEqual(frozen['recommendation_arm_id'], 'H00_reference')
        row = next(r for r in frozen['exploratory_development_ranking'] if r['arm_id']=='H08_dual_conditional')
        self.assertTrue(row['targets_passed'])
        self.assertFalse(row['dev_crossfit_passed'])
        self.assertFalse(row['qualified'])
        self.assertFalse((self.root / 'cache/test').exists())

    def test_passing_four_gates_and_oof_still_requires_reference_known_count(self):
        initial = records(pass_gates=True)[:-1]
        before = [dict(copy.deepcopy(initial[i % 2]), image_sha256=hashlib.sha256(('known'+str(i)).encode()).hexdigest())
                  for i in range(20)] + initial[2:]
        after = copy.deepcopy(before)
        after[0]['leaf'] = 1
        def value(rows):
            return ({'meta':META}, dict(base.evaluate_records(rows,META),crossfit_audit={'passed':True}), rows)
        result = reporting._entry('H08_dual_conditional',value(after),value(before),'H00_reference','development')
        self.assertTrue(result['targets_passed'])
        self.assertTrue(result['dev_crossfit_passed'])
        self.assertFalse(result['known_count_preserved'])
        self.assertFalse(result['qualified'])
        self.assertTrue(reporting._entry('H08_dual_conditional',value(before),value(before),
                                        'H00_reference','development')['qualified'])

    def test_twelve_arm_report_keeps_failures_null_test_descriptive_and_zero_step_statfit(self):
        self._all_dev(failed=['H11_dual_rank16'])
        frozen = reporting.freeze_dev_selection(self.root)
        before = protocol.file_hash(self.root / 'dev_selection.json')
        for arm in self.cfg['arms']:
            if arm['id'] != 'H11_dual_rank16':
                self._stage(arm['id'], 'test', pass_gates=arm['id']=='H08_dual_conditional')
        summary = reporting.summarize_suite(self.root)
        self.assertEqual(summary['declared_arm_count'], 12)
        self.assertEqual(summary['completed_test_count'], 11)
        self.assertEqual(summary['recommendation_arm_id'], frozen['recommendation_arm_id'])
        self.assertEqual(protocol.file_hash(self.root / 'dev_selection.json'), before)
        row = next(r for r in summary['all_arms'] if r['arm_id']=='H04_subspace_root')
        self.assertEqual(row['training_execution'], 'statistical_TRAIN_fit')
        self.assertEqual(row['optimizer_steps'], 0)
        self.assertIn('inherited_dev_crossfit_passed', row)
        self.assertNotIn('test_crossfit_passed', row)
        fail = summary['all_arms'][-1]
        self.assertIsNone(fail['test_known_end_to_end_leaf_accuracy'])
        self.assertIsNone(fail['dev_known_root_rejected'])
        self.assertIn('unavailable', fail['failure_reason'])
        self.assertFalse(summary['production_recommendation_uses_test'])
        self.assertEqual(summary['best_exploratory_test_arm']['arm_id'], 'H08_dual_conditional')
        self.assertFalse(summary['best_exploratory_test_arm']['qualified'])
        self.assertNotIn('crossfit_audit_passed', summary['best_exploratory_test_arm'])
        self.assertTrue(summary['diagnostics']['test']['root_state_comparison']['staged_root_states_exact'])

    def test_candidate_changes_are_separate_from_confidence_and_terminal_changes(self):
        before = records(pass_gates=True)
        after = copy.deepcopy(before)
        after[0].update(candidate_parent=1, candidate_leaf=2, prediction_type='global_unknown', parent=None, leaf=None)
        after[0]['domain'].update(candidate_parent=1,candidate_leaf=2,candidate_policy='domain_parent_reference_child')
        after[-1] = dict(copy.deepcopy(after[0]), path='alias.png')
        after[1].update(prediction_type='global_unknown',parent=None,leaf=None)
        after[1]['domain']['root_score'] += .2
        result = reporting._candidate_comparison(before, after)
        self.assertEqual(result['candidate_path_changed'], 1)
        self.assertEqual(result['candidate_changed_terminal_changed'], 1)
        self.assertEqual(result['same_candidate_terminal_changed'], 1)
        self.assertEqual(result['confidence_vectors_changed'], 1)
        self.assertEqual(result['known_candidate_leaf_damaged'], 1)

    def test_shared_root_controls_reject_score_or_staged_state_drift_allow_joint_threshold(self):
        fingerprints = {key:{'root_score_sha256':'same','root_state_sha256':'staged'} for key in
            ('H06_dual_root','H08_dual_conditional','H09_dual_reroute','H10_dual_joint')}
        fingerprints['H10_dual_joint']['root_state_sha256'] = 'joint-different'
        self.assertTrue(reporting._root_invariants(fingerprints)['staged_root_states_exact'])
        for key in ('root_state_sha256','root_score_sha256'):
            wrong = copy.deepcopy(fingerprints)
            wrong['H09_dual_reroute'][key] = 'changed'
            with self.assertRaisesRegex(ValueError, 'Shared domain root'):
                reporting._root_invariants(wrong)

    def test_changed_full_crossfit_artifact_rejected(self):
        arm = 'H02_d05_staged'
        self._stage(arm, 'calibration')
        folder = self.root / 'arms' / arm / 'calibration'
        full = folder / 'crossfit_audit.json'
        full.write_text('{}')
        with self.assertRaises(ValueError):
            reporting._stage(self.root, arm, 'calibration', self.snapshot)

    def test_coherently_rehashed_root_counts_must_reproduce_predictions(self):
        arm = 'H02_d05_staged'
        receipt = self._stage(arm, 'calibration')
        folder = self.root / 'arms' / arm / 'calibration'
        report = protocol.read_json(folder / 'report.json')
        report['root_stage_outcomes']['known_root_rejected'] += 1
        receipt['summary'] = report
        receipt['artifacts']['report'] = self._file(folder, 'report.json', report)
        protocol.write_json(folder / 'completed.json', receipt)
        (folder / 'stage_binding.json').unlink()
        runner._complete_stage(self.root, arm, 'calibration', self.snapshot)
        with self.assertRaisesRegex(ValueError, 'root-stage outcomes'):
            reporting._stage(self.root, arm, 'calibration', self.snapshot)

    def test_crossfit_compact_cannot_claim_success_against_full_audit(self):
        arm = 'H02_d05_staged'
        receipt = self._stage(arm, 'calibration')
        folder = self.root / 'arms' / arm / 'calibration'
        report = protocol.read_json(folder / 'report.json')
        report['crossfit_audit']['passed'] = True
        receipt['summary'] = report
        receipt['artifacts']['report'] = self._file(folder, 'report.json', report)
        protocol.write_json(folder / 'completed.json', receipt)
        (folder / 'stage_binding.json').unlink()
        runner._complete_stage(self.root, arm, 'calibration', self.snapshot)
        with self.assertRaisesRegex(ValueError, 'compact decision'):
            reporting._stage(self.root, arm, 'calibration', self.snapshot)


if __name__ == '__main__':
    unittest.main()
