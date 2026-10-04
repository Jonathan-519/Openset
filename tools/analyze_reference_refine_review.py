#!/usr/bin/env python3
"""Reproduce the 2026-10-03 reference/refinement audit from review archives.

This is a read-only diagnostic: it never tunes a threshold or trains a model.
Only the Python standard library is required. Both input roots must contain
archive_manifest.json and run/. Hash checks cover the packaged text artifacts;
they cannot verify absent images, weights, or the training feature cache.
The Chinese narrative is a dated review and refuses other experiment files;
use --json-only to compute the generic paired diagnostics for another run.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tarfile


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def unique_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["image_sha256"]].append(row)
    for group in grouped.values():
        first = group[0]
        for other in group[1:]:
            for field in ("status", "true_parent", "true_leaf", "prediction_type", "candidate_parent", "candidate_leaf"):
                if first[field] != other[field]:
                    raise ValueError("Duplicate content disagrees on " + field)
    return {key: group[0] for key, group in grouped.items()}, {
        "record_count": len(rows), "unique_count": len(grouped),
        "duplicate_count": len(rows) - len(grouped),
        "duplicate_paths": [[row["path"] for row in group] for group in grouped.values() if len(group) > 1],
    }


def verify_manifest(root):
    manifest = read_json(root / "archive_manifest.json")
    for item in manifest["files"]:
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe manifest path")
        content = (root / path).read_bytes()
        if len(content) != item["bytes"] or sha256(content) != item["sha256"]:
            raise ValueError("Manifest mismatch: " + str(path))
    return {"schema": manifest["schema"], "created_at_utc": manifest["created_at_utc"],
            "verified_files": len(manifest["files"]), "verified": True,
            "images_present": manifest["includes_images"], "checkpoints_present": manifest["includes_checkpoints"]}


def correct(row):
    if row["status"] == "known":
        return (row["prediction_type"] == "known" and row["candidate_leaf"] == row["true_leaf"]
                and row["candidate_parent"] == row["true_parent"])
    if row["status"] == "intra":
        return row["prediction_type"] == "intra_unknown" and row["candidate_parent"] == row["true_parent"]
    return row["prediction_type"] == "global_unknown"


def four_counts(rows):
    rows = list(rows)
    result = {status: sum(row["status"] == status for row in rows) for status in ("known", "intra", "extra")}
    result.update({status + "_correct": sum(row["status"] == status and correct(row) for row in rows)
                   for status in ("known", "intra", "extra")})
    result["leaf_outputs"] = sum(row["prediction_type"] == "known" for row in rows)
    return result


def pair_auc(positive, negative):
    if not positive or not negative:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def score_analysis(rows):
    rows = list(rows)
    known = [r for r in rows if r["status"] == "known"]
    near = [r for r in rows if r["status"] == "intra"]
    result = {}
    for field in ("baseline_local_knownness_score", "reconstruction_score", "local_knownness_score"):
        result[field] = {"known_vs_near_auroc": pair_auc([r[field] for r in known], [r[field] for r in near])}
    result["root_knownness_score"] = {"taxonomy_vs_extra_auroc": pair_auc(
        [r["root_knownness_score"] for r in rows if r["status"] != "extra"],
        [r["root_knownness_score"] for r in rows if r["status"] == "extra"])}
    return result


def error_partition(rows):
    result = {}
    for status in ("known", "intra", "extra"):
        group = [r for r in rows if r["status"] == status]
        details = Counter()
        for r in group:
            if correct(r):
                details["correct"] += 1
            elif r["prediction_type"] == "global_unknown":
                details["incorrect_root_rejection"] += 1
            elif r["prediction_type"] == "intra_unknown":
                details["incorrect_parent_fallback"] += 1
            else:
                details["incorrect_leaf_acceptance"] += 1
        result[status] = {"count": len(group), "exclusive_outcomes": dict(details),
                          "prediction_type_counts": dict(Counter(r["prediction_type"] for r in group))}
        if status != "extra":
            result[status]["candidate_parent_wrong"] = sum(r["candidate_parent"] != r["true_parent"] for r in group)
            result[status]["correct_parent_but_root_rejected"] = sum(
                r["candidate_parent"] == r["true_parent"] and r["prediction_type"] == "global_unknown" for r in group)
        if status == "known":
            result[status]["candidate_leaf_correct"] = sum(r["candidate_leaf"] == r["true_leaf"] for r in group)
            result[status]["correct_leaf_candidate_rejected"] = sum(
                r["candidate_leaf"] == r["true_leaf"] and r["prediction_type"] != "known" for r in group)
        if status == "intra":
            eligible = sum(r["candidate_parent"] == r["true_parent"] and r["prediction_type"] != "global_unknown" for r in group)
            result[status]["fixed_parent_max_correct_fallback_count"] = eligible
            result[status]["fixed_parent_max_correct_fallback_rate"] = eligible / len(group)
    return result


def paired_analysis(baseline, refined):
    if set(baseline) != set(refined):
        raise ValueError("Paired content sets differ")
    changed = []
    decision_matches = parent_matches = candidate_matches = raw_matches = 0
    for key, a in baseline.items():
        b = refined[key]
        for field in ("status", "true_leaf", "true_parent"):
            if a[field] != b[field]:
                raise ValueError("Paired metadata mismatch")
        decision_matches += (a["prediction_type"], a["candidate_parent"], a["candidate_leaf"]) == (b["prediction_type"], b["candidate_parent"], b["candidate_leaf"])
        parent_matches += (a["prediction_type"] == "global_unknown") == (b["prediction_type"] == "global_unknown")
        candidate_matches += (a["candidate_parent"], a["candidate_leaf"]) == (b["candidate_parent"], b["candidate_leaf"])
        raw_matches += a["support_evidence"] == b["support_evidence"] and a["log_probs"] == b["log_probs"]
        if a["prediction_type"] != b["prediction_type"]:
            changed.append({"path": a["path"], "source": a["source"], "status": a["status"],
                            "before": a["prediction_type"], "after": b["prediction_type"],
                            "correct_before": correct(a), "correct_after": correct(b),
                            "candidate_leaf": a["candidate_leaf_name"], "reconstruction_score": b["reconstruction_score"]})
    return {"unique_images": len(baseline), "decision_matches": decision_matches,
            "parent_gate_matches": parent_matches, "candidate_matches": candidate_matches,
            "exact_raw_evidence_matches": raw_matches, "changed": changed,
            "correctness_regressions": sum(r["correct_before"] and not r["correct_after"] for r in changed),
            "correctness_improvements": sum(not r["correct_before"] and r["correct_after"] for r in changed)}


def compact_source_loo(router):
    return [{"status": f["status"], "held_source": f["held_source"],
             "fit_count": f["fit_count"], "baseline_refit_on_fit_sources_only": f["baseline_refit_on_fit_sources_only"],
             "baseline_held_metrics": f["baseline_held_metrics"], "refined_held_metrics": f["held_metrics"]}
            for f in router["source_loo"]["folds"]]


def compare_prior_archives(directory, roots):
    result = []
    for path in sorted(directory.glob("trial_1_review_20261003T074938*.gz")):
        with tarfile.open(path, "r:gz") as archive:
            manifests = [m for m in archive.getmembers() if m.name == "archive_manifest.json"]
            if len(manifests) != 1:
                raise ValueError("Archive missing unique manifest")
            manifest = json.loads(archive.extractfile(manifests[0]).read())
            label = "refine" if manifest["schema"] == "taxosafe_refine_review_v1" else "reference"
            current = read_json(roots[label] / "archive_manifest.json")
            old = {f["path"]: (f["bytes"], f["sha256"]) for f in manifest["files"]}
            new = {f["path"]: (f["bytes"], f["sha256"]) for f in current["files"]}
            result.append({"archive": path.name, "sha256": sha256(path.read_bytes()), "run": label,
                           "same_packaged_file_list_and_hashes_as_latest": old == new})
    return result


def analyze(reference, refine, prior_directory=None):
    roots = {"reference": reference, "refine": refine}
    result = {"schema": "reference_refine_trial1_diagnostic_v1",
              "scope": "Read-only uploaded experiment audit; TEST informs error diagnosis and subsequent method design and is no longer an untouched final test set; this script fits no numerical parameters or thresholds.",
              "archive_integrity": {label: verify_manifest(root) for label, root in roots.items()},
              "input_artifact_sha256": {}, "runs": {}, "paired": {}, "score_diagnostics": {}}
    rowsets = {}
    for label, root in roots.items():
        result["runs"][label] = {}
        rowsets[label] = {}
        for split, metric_path, row_path, gate_path in (
            ("development", "calibration/development_metrics.json", "calibration/development_predictions.jsonl", "calibration/validation_report.json"),
            ("test", "test/metrics.json", "test/predictions.jsonl", "test/gates.json"),
        ):
            rows, dedup = unique_rows(root / "run" / row_path)
            rowsets[label][split] = rows
            metrics = read_json(root / "run" / metric_path)
            gates = read_json(root / "run" / gate_path)
            counts = four_counts(rows.values())
            if counts != gates["counts"]:
                raise ValueError("Recomputed gate counts differ: " + label + " " + split)
            result["runs"][label][split] = {"metrics": metrics, "counts_recomputed": counts,
                                            "gates": {k: gates[k] for k in ("metrics", "checks", "requirements", "targets_passed")},
                                            "deduplication": dedup, "error_partition": error_partition(rows.values())}
            result["runs"][label][split]["source_macro_rates"] = {
                "near_correct_fallback": sum(m["correct_fallback_rate"] for m in metrics["per_intra_species"].values()) / len(metrics["per_intra_species"]),
                "extra_root_rejection": sum(m["global_unknown_recall"] for m in metrics["per_extra_source"].values()) / len(metrics["per_extra_source"]),
            }
            for path in (metric_path, row_path, gate_path):
                result["input_artifact_sha256"][label + "/" + path] = sha256((root / "run" / path).read_bytes())
    for split in ("development", "test"):
        result["paired"][split] = paired_analysis(rowsets["reference"][split], rowsets["refine"][split])
        result["score_diagnostics"][split] = score_analysis(rowsets["refine"][split].values())
    binding = read_json(refine / "run/refinement/source_binding.json")
    source_receipts = {path: sha256((reference / "run" / path).read_bytes()) == digest
                       for path, digest in binding["receipt_sha256"].items()}
    if not all(source_receipts.values()):
        raise ValueError("Refinement was not bound to the provided reference")
    baseline_test = read_json(refine / "run/test/baseline_metrics.json")
    baseline_dev = read_json(refine / "run/calibration/baseline_metrics.json")
    result["source_consistency"] = {
        "bound_source_receipts_match": source_receipts,
        "router_file_matches": sha256((reference / "run/calibration/router.json").read_bytes()) == binding["router_sha256"],
        "baseline_test_metrics_exact": baseline_test == result["runs"]["reference"]["test"]["metrics"],
        "baseline_development_metrics_exact": baseline_dev == result["runs"]["reference"]["development"]["metrics"],
        "recorded_development_reproduction": read_json(refine / "run/calibration/baseline_reproduction.json"),
        "recorded_test_preservation": read_json(refine / "run/test/preservation.json"),
        "limitation": "Binary checkpoints, feature cache and source images are not in these archives and were not independently rerun.",
    }
    rt = read_json(reference / "run/training/completed.json")
    ft = read_json(refine / "run/refinement/training_report.json")
    fr = read_json(refine / "run/calibration/router.json")
    result["training"] = {"reference": {k: rt[k] for k in (
        "best_epoch", "known_validation", "gradient_splits", "support_splits", "test_used_for_fitting", "unknown_images_used_for_gradients", "model_cost")},
        "refine": {k: v for k, v in ft.items() if k != "split"}}
    histories = [json.loads(s) for s in (reference / "run/training/train.jsonl").read_text().splitlines() if s]
    result["training"]["reference"]["epochs_completed"] = len(histories)
    result["training"]["reference"]["training_seconds_recorded"] = histories[-1]["elapsed_seconds"]
    result["training"]["refine"]["split"] = {k: v for k, v in ft["split"].items()
                                               if k not in ("fit_indices", "validation_indices", "fit_hashes", "validation_hashes")}
    result["training"]["refine"]["fit_count"] = len(ft["split"]["fit_indices"])
    result["training"]["refine"]["validation_count"] = len(ft["split"]["validation_indices"])
    train_hashes = set(ft["split"]["fit_hashes"]) | set(ft["split"]["validation_hashes"])
    dev_hashes = set(rowsets["refine"]["development"])
    test_hashes = set(rowsets["refine"]["test"])
    result["split_separation"] = {
        "train_dev_image_overlap": len(train_hashes & dev_hashes),
        "train_test_image_overlap": len(train_hashes & test_hashes),
        "dev_test_image_overlap": len(dev_hashes & test_hashes),
        "near_dev_test_source_overlap": sorted(
            {r["source"] for r in rowsets["refine"]["development"].values() if r["status"] == "intra"}
            & {r["source"] for r in rowsets["refine"]["test"].values() if r["status"] == "intra"}),
        "extra_dev_test_source_overlap": sorted(
            {r["source"] for r in rowsets["refine"]["development"].values() if r["status"] == "extra"}
            & {r["source"] for r in rowsets["refine"]["test"].values() if r["status"] == "extra"}),
    }
    species_groups = defaultdict(list)
    for row in rowsets["refine"]["test"].values():
        if row["status"] == "intra":
            species_groups[row["source"]].append(row)
    result["near_species_candidate_limits"] = {
        name: {"sample_count": len(rows),
               "correct_parent_candidates": sum(r["candidate_parent"] == r["true_parent"] for r in rows),
               "fixed_parent_gate_eligible": sum(r["candidate_parent"] == r["true_parent"] and r["prediction_type"] != "global_unknown" for r in rows)}
        for name, rows in sorted(species_groups.items())}
    result["calibration"] = {"parent_threshold": fr["fixed_parent_threshold"], "leaf_threshold": fr["leaf_threshold"],
                             "reconstruction_threshold": fr["reconstruction_threshold"], "baseline_fallback": fr["baseline_fallback"],
                             "preservation_audit": fr["preservation_audit"], "source_loo": compact_source_loo(fr)}
    result["timing"] = {label: read_json(root / "run/test/inference_timing.json") for label, root in roots.items()}
    if prior_directory:
        result["prior_archive_comparison"] = compare_prior_archives(prior_directory, roots)
    return result


def pct(value):
    return "—" if value is None else "{:.2f}%".format(100 * value)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def markdown_report(d):
    expected = {
        "reference/calibration/development_predictions.jsonl": "5f60ed0de0ff7629aa557877d5e19388cb2db46594dfcd1bf2debe0da96478ff",
        "reference/test/predictions.jsonl": "5ea22b68da9459dcecf09670efa7a9750d0b198c284e92b3cf21b6bf8f1cacaf",
        "refine/calibration/development_predictions.jsonl": "3f5d946378e08b0896c83dcb87abd988c05fd3a9d9b1039f77b8004a7f0d8f30",
        "refine/test/predictions.jsonl": "9521b144cb17063bda537a22d9f9246c467a5a6ccb455f9f0181e4def684f033",
    }
    if any(d["input_artifact_sha256"].get(path) != digest for path, digest in expected.items()):
        raise ValueError("The dated Chinese narrative only covers the supplied 2026-10-03 experiment. Use --json-only for other runs.")
    base = d["runs"]["reference"]
    ref = d["runs"]["refine"]
    lines = ["# Trial 1：reference 与第一次语义重构精修的实验审计（2026-10-03）", "",
             "本报告对应 07:55 UTC 上传的两份压缩包，按图像内容 SHA-256 去重。审计脚本未利用 TEST 拟合数值参数或阈值；本轮 TEST 已用于错误诊断和后续方法设计，不能视为未触碰的最终测试集。完整机器可读结果见同名 JSON；复现脚本为 `tools/analyze_reference_refine_review.py`。", "",
             "## 结论", "",
             "精修带来小幅拒识收益，同时损失已知准确率，整体正确数没有增加，尚未达到验收标准。重训 reference 的主要 TEST 指标与此前较优版本一致；本轮不是训练中断或权重误用。", "",
             "TEST 已知正确数 423→419，近域正确父类回退 124→128，域外正确根拒识保持 220；总正确数两者均为 767/926。新增模块拒绝了 12 个原叶输出：4 个原本正确的已知样本、1 个原已知错分类、4 个近域未知和 3 个域外未知。域外从叶退至父仍是错误，并未成为正确根拒识。", "",
             "## 数据与结果可信范围", ""]
    for label in ("reference", "refine"):
        a = d["archive_integrity"][label]
        lines.append("- `{}`：{} 个打包文本文件的字节数与 SHA-256 全部通过；生成时间 {}。".format(label, a["verified_files"], a["created_at_utc"]))
    lines += ["- TRAIN 1610 张；DEV 219 已知 + 69 近域 + 88 域外 = 376 张；TEST 原清单 927 行、去重后 926 张（450 + 204 + 272），已知中有 1 行内容重复。",
              "- 精修绑定的 reference 配置、输入审计和阶段完成凭据全部匹配；DEV 与 TEST 的候选、原始输出分数以及父级根门逐图一致，打包的基线 metrics 也完全匹配。",
              "- 从已记录图像摘要重算，TRAIN/DEV/TEST 两两图像内容交集均为 0；DEV 与 TEST 的近域物种集合、域外来源集合也分别无交集。",
              "- 压缩包不含真实图像、检查点和特征缓存；这里只能核验记录与相互绑定，不能替代重新加载权重运行。", ""]
    if d.get("prior_archive_comparison"):
        lines.append("07:49 上传包及其重复副本中的文件清单和每个文件摘要，与 07:55 对应运行包完全一致；这是重新打包，不是新增实验。")
        lines.append("")
    lines += ["## 四项验收指标", ""]
    rows = []
    keys = [("已知端到端叶准确率", "known_end_to_end_leaf_accuracy", ">90%"),
            ("近域未知正确父类回退率", "intra_correct_fallback_rate", "≥85%"),
            ("域外未知根拒识率", "extra_global_unknown_recall", ">90%"),
            ("开放世界叶级精确率", "open_world_accepted_leaf_precision", ">90%")]
    for name, key, target in keys:
        b, f = base["test"]["gates"]["metrics"][key], ref["test"]["gates"]["metrics"][key]
        rows.append([name, pct(b), pct(f), "{:+.2f} 个百分点".format(100 * (f - b)), target,
                     "通过" if ref["test"]["gates"]["checks"][key] else "未通过"])
    lines += [table(["指标", "reference TEST", "refine TEST", "变化", "标准", "refine 判定"], rows), "",
              "分母分别为：已知 419/450、近域 128/204、域外 220/272、开放世界叶精确率 419/497。最后一项包括未知误接收，不能用仅在已知类中计算的 419/421=99.52% 代替。", "",
              "近域至少还需增加 46 个正确父回退，域外至少还需增加 25 个正确根拒识。若保持 419 个正确叶输出不变，叶输出总量需从 497 降至不超过 465，即至少消除 32 个错误叶输出，精确率才能严格大于 90%。这是计数要求，不代表仅调阈值可做到。", "",
              "### DEV 与 TEST 对照", ""]
    rows = [[name, pct(base["development"]["gates"]["metrics"][key]), pct(ref["development"]["gates"]["metrics"][key]),
             pct(base["test"]["gates"]["metrics"][key]), pct(ref["test"]["gates"]["metrics"][key])] for name, key, _ in keys]
    lines += [table(["指标", "reference DEV", "refine DEV", "reference TEST", "refine TEST"], rows), "",
              "DEV 已知正确数保持 198/219，但 TEST 损失 4 个正确样本，说明 DEV 聚合保护不是未来样本保证。DEV 新增 5 个近域正确回退全部来自 Centropages_tenuiremis；留一来源分析中仅该来源增加 4 个正确回退，其余三个近域来源均无增加。", "",
              "## 其他指标及定义", ""]
    fields = [("候选父类准确率（拒识前）", "known", "parent_accuracy"), ("全局叶分类准确率（拒识前）", "known", "global_leaf_accuracy"),
              ("父分支内叶候选准确率（拒识前）", "known", "branch_restricted_leaf_accuracy"),
              ("候选叶 macro-F1（拒识前）", "known", "macro_f1"), ("候选叶 balanced accuracy（拒识前）", "known", "balanced_accuracy"),
              ("已知叶输出覆盖率", "known", "known_leaf_coverage"), ("已知样本中已接收叶精确率", "known", "accepted_leaf_precision"),
              ("已知误退父级率", "known", "under_specification_rate"), ("已知误退根率", "known", "known_global_rejection_rate"),
              ("近域误接收为已知叶率", "intra", "over_specification_error_rate"), ("近域误退根率", "intra", "intra_global_rejection_rate"),
              ("近域候选父类定位准确率", "intra", "parent_localization_accuracy"),
              ("域外错误通过父级根门率", "extra", "false_parent_acceptance_rate"), ("域外误接收叶率", "extra", "false_known_leaf_rate"),
              ("整体最深可靠节点准确率", "overall", "deepest_reliable_taxon_accuracy"), ("三种状态平均节点准确率", "overall", "macro_deepest_reliable_taxon_accuracy"),
              ("未知过度细分风险", "overall", "over_specification_risk")]
    lines += [table(["指标", "reference TEST", "refine TEST"], [[name, pct(base["test"]["metrics"][g][k]), pct(ref["test"]["metrics"][g][k])] for name, g, k in fields]), "",
              "候选类指标未经过拒识门，不代表最终识别正确率。macro-F1 与 balanced accuracy 按候选叶计算；整体 macro 节点准确率按 known/intra/extra 三种状态等权，不是按物种等权。未知过度细分风险统计近域叶误接收以及域外所有非根输出。已知选择风险（已接收已知中的错误比例）从 3/426=0.70% 降到 2/421=0.48%，但覆盖率下降；开放世界叶选择风险仍有 78/497=15.69%，不可混淆两种分母。", "",
              "### 排序质量与工作点", ""]
    fields = [("近域 AUROC（当前组合路由分数）", "near_open_set", "auroc"), ("近域 AUPR-known", "near_open_set", "aupr_known"),
              ("近域 AUPR-unknown", "near_open_set", "aupr_unknown"), ("近域 FPR@95%TPR（越低越好）", "near_open_set", "fpr95"),
              ("近域 OSCR", "near_open_set", "oscr"), ("根门 AUROC", "extra", "auroc"), ("根门 AUPR（谱系内为正）", "extra", "aupr"),
              ("根门 FPR@95%TPR（越低越好）", "extra", "fpr95")]
    lines += [table(["指标", "reference TEST", "refine TEST"], [[name, pct(base["test"]["metrics"][g][k]), pct(ref["test"]["metrics"][g][k])] for name, g, k in fields]), ""]
    rows = [[field, pct(d["score_diagnostics"]["development"][field]["known_vs_near_auroc"]),
             pct(d["score_diagnostics"]["test"][field]["known_vs_near_auroc"])] for field in ("baseline_local_knownness_score", "reconstruction_score", "local_knownness_score")]
    lines += [table(["独立重算的 known-vs-near AUROC", "DEV", "TEST"], rows), "",
              "精修 local_knownness_score 是 membership logit 裕量与重构裕量的最小值，二者量纲不同；它不是纯重构分数，也不是校准概率。单独计算纯重构分数后，其 AUROC 仍低于原 membership，表明新增证据整体分离能力不足。工作点的近域回退率小幅提升，并不意味着全排序质量提升。根门 AUROC 较高但固定工作点召回仅 80.88%，尾部重叠与路由约束仍明显。", "",
              "TEST 近域 CCR@FPR=1%/5%/10% 分别由 59.78%/78.22%/85.33% 降为 26.22%/60.67%/78.89%。低误接收区间尤其不适合直接采用当前组合分数。", "",
              "### 95% Wilson 区间（单个固定工作点）", ""]
    intervals = ref["test"]["metrics"]["confidence_intervals_95"]
    lines += [table(["指标", "refine 点估计", "95% 区间"], [[name, pct(ref["test"]["gates"]["metrics"][key]),
               pct(intervals[key]["low"]) + "–" + pct(intervals[key]["high"])] for name, key, _ in keys]), "",
              "这些是单指标区间，不是两个版本之差的置信区间，也未校正多次在同一 TEST 上开发的选择偏差。仅一个 trial，不能声称差异具有稳定统计优势。", "",
              "## 逐物种近域拒识", "",
              "正确拒识在本任务中要求输出正确父类。退到根或错误父类，虽然没有输出已知物种，也不能计作正确。", ""]
    rows = []
    for name, m in base["test"]["metrics"]["per_intra_species"].items():
        n = ref["test"]["metrics"]["per_intra_species"][name]
        total = m["sample_count"]
        rows.append([name, total, "{}/{} ({})".format(round(m["correct_fallback_rate"] * total), total, pct(m["correct_fallback_rate"])),
                     "{}/{} ({})".format(round(n["correct_fallback_rate"] * total), total, pct(n["correct_fallback_rate"])),
                     round(n["over_specification_error_rate"] * total), round(n["intra_global_rejection_rate"] * total),
                     round(n["wrong_parent_rate"] * total)])
    lines += [table(["物种", "N", "reference 正确父回退", "refine 正确父回退", "refine 误接收叶", "refine 误退根", "候选父类错误*"], rows), "",
              "*候选父类错误与根拒识/叶误接收可重叠，不可与其他列直接相加。Oithona_plumifera 的 32 张中，14 张候选父错、21 张退根，最终仅 4 张回退正确父类；仅固定候选而放开根门时，其父回退理论上限为 18/32=56.25%，连根门也冻结则仅剩 7/32=21.88% 的上限。增加叶门无法修复父候选或根门错误。Turritopsis_nutricula 占近域 TEST 的 114/204=55.88%，总样本微平均会弱化其他来源问题。近域逐来源等权正确回退率由 51.10% 升到 54.83%，仍明显低于微平均 62.75%。", "",
              "## 逐来源域外拒识", ""]
    rows = []
    for name, m in base["test"]["metrics"]["per_extra_source"].items():
        n = ref["test"]["metrics"]["per_extra_source"][name]
        total = n["sample_count"]
        rows.append([name, total, "{}/{} ({})".format(round(n["global_unknown_recall"] * total), total, pct(n["global_unknown_recall"])),
                     "{}→{}".format(round(m["false_known_leaf_rate"] * total), round(n["false_known_leaf_rate"] * total)),
                     n["prediction_type_counts"].get("intra_unknown", 0)])
    lines += [table(["来源", "N", "两版正确根拒识", "误接收叶 reference→refine", "refine 错退父"], rows), "",
              "Fish_larva 最弱：41 张中仅 21 张根拒识，20 张错误通过根门；其中 18 张被选入 Euphausiacea，2 张进入 Sagittoidea。精修仅将两张叶输出降到父级，不能改变根拒识率。两版域外逐来源等权根拒识率均为 82.19%。", "",
              "## 全部已知物种端到端准确率", ""]
    rows = []
    for name, m in base["test"]["metrics"]["per_known_leaf"].items():
        n = ref["test"]["metrics"]["per_known_leaf"][name]
        total = n["sample_count"]
        a, b = m["accepted_correct_leaf_count"], n["accepted_correct_leaf_count"]
        rows.append([name, total, "{}/{} ({})".format(a, total, pct(m["end_to_end_leaf_accuracy"])),
                     "{}/{} ({})".format(b, total, pct(n["end_to_end_leaf_accuracy"])), "{:+d}".format(b-a)])
    lines += [table(["已知物种", "N", "reference", "refine", "正确数变化"], rows), "",
              "已知损失集中在 Copepoda：Acartia hongi 3 张、Calanus sinicus 1 张。Oithona similis 的一个错误叶分类被改成父回退，但已知端到端仍错误，不能记作已知准确率提升。", "",
              "## 错误所在层级与硬上限", ""]
    rows = []
    for split in ("development", "test"):
        a = ref[split]["error_partition"]["intra"]
        rows.append([split, a["count"], a["candidate_parent_wrong"], a["correct_parent_but_root_rejected"],
                     a["fixed_parent_max_correct_fallback_count"], pct(a["fixed_parent_max_correct_fallback_rate"])])
    lines += [table(["集合", "近域N", "候选父错", "父候选正确但退根", "冻结父路由最大正确数", "冻结父路由上限"], rows), "",
              "DEV 最大 52/69=75.36%，低于 85% 标准；这已经证明只修改叶判别而冻结父级路由不可能通过 DEV 全部门槛。TEST 上限 175/204=85.78%，仅比至少 174 个的验收计数多 1 个，几乎没有叶门犯错余量。该上限只用于错误定位，不用于选择 TEST 阈值。", "",
              "refine TEST 的互斥错误：已知 419 正确 + 2 错叶 + 22 退父 + 7 退根；近域 128 正确父回退 + 49 错接收叶 + 2 错父回退 + 25 退根；域外 220 正确退根 + 27 错叶 + 25 错父。", "",
              "## 训练、校准及计算开销", "",
              "reference 训练到第 75 轮，选中第 50 轮，累计训练日志约 6772.89 秒；最终权重选择仅使用 val_known。精修使用 1610×512 的冻结 fine 特征，1297 张拟合、313 张 TRAIN 内验证；376833 个可训练参数，基线可训练参数为 0。第 6 轮达到最佳内验证 NLL=0.01237、准确率 99.36%，第 14 轮早停。", "",
              "内验证图像来自原 TRAIN，冻结 backbone 已见过这些数据；报告明确 frozen_backbone_independent_validation=false、strict_unseen_class_evaluation=false。因此很低的 NLL 只能证明在这些已知特征上拟合良好，不能证明未知物种泛化。训练梯度记录仅来自 TRAIN，未知图像未用于梯度；DEV 负责门限，TEST 未用于本次既有模型拟合。", "",
              "精修训练特征提取约 22.59 秒、模块拟合约 5.58 秒。TEST 总推理记录 reference 15.54 秒、refine 15.56 秒，单次且有调度/IO噪声，不能解释为可靠速度提升或损失。", "",
              "## 当前问题与优化所需条件", "",
              "1. 父级判别必须能够改善：未知近邻错误退根与域外错误留父同时存在，单向收紧或放松同一个全局门会互相伤害。需要父级相容证据与叶级已知证据分离，并针对同父未知验证泛化。",
              "2. 新叶证据应具备真正的类外分离能力：当前按已知分类 NLL 选出的全局 fine 重构器在纯重构 AUROC 上明显弱于原证据，不能仅延长训练或添加更强拒识阈值。",
              "3. 校准应评估来源泛化、类不平衡和已知尾部风险：DEV 增益集中在单一来源；应报告来源宏平均、逐父类/逐物种错误和独立留源结果，避免大来源支配选择。",
              "4. 候选保持的好处应保留：reference 的父候选已知准确率 99.33%、分支内叶候选 97.11%，已有判别能力较强。新增拒识证据应独立于候选识别头，且在没有可靠增益时保留基线。",
              "5. 目前 TEST 已被多轮用于研究诊断。后续性能声明仍需固定协议、多个随机种子及尚未参与设计的新测试来源验证；本报告未据 TEST 拟合任何阈值。", "",
              "本报告仅解释已完成的两次运行。新代码或文献方案的潜在收益不属于这份实验结果，必须另行拟合、校准和测试后确认。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--refine", type=Path, required=True)
    parser.add_argument("--prior-archives", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--json-only", action="store_true", help="Generic numerical diagnostics without the dated Chinese narrative")
    args = parser.parse_args()
    result = analyze(args.reference, args.refine, args.prior_archives)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.json_only:
        md_path.write_text(markdown_report(result), encoding="utf-8")
    print(json_path)
    if not args.json_only:
        print(md_path)


if __name__ == "__main__":
    main()
