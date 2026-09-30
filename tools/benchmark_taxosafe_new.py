#!/usr/bin/env python3
"""Compare frozen v11/new scoring on the same known validation manifest.

No training, calibration, test image access, or threshold changes. This measures
end-to-end scoring, not optimizer-step time; training timing is in train.jsonl.
"""
import argparse
import contextlib
import copy
import io
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed new run")
    parser.add_argument("--v11-run-dir", required=True, help="Completed unchanged v11 run")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", help="Default: new run/speed_comparison.json")
    args = parser.parse_args()
    if min(args.batch_size, args.repeats, args.warmup) < 1:
        parser.error("batch-size, repeats and warmup must be positive")
    import torch
    from taxosafe_dcbs import pipeline as old
    from taxosafe_support import pipeline as new
    from taxosafe_support.protocol import read_json, read_split, audit_rows, write_json, resolve, run_lock
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU and both completed checkpoints are required")
    directory, old_directory = resolve(args.run_dir), resolve(args.v11_run_dir)
    output = Path(args.output).resolve() if args.output else directory / "speed_comparison.json"
    if output.exists():
        parser.error("Output already exists; provide a fresh --output")
    cfg = read_json(directory / "training/config.json")
    old_cfg = read_json(old_directory / "training/config.json")
    device = torch.device("cuda")
    encoder, evidence, trained, bank = new.load_trained(cfg, directory, device)
    backbone, heads, old_trained, old_support = old.load_trained(old_cfg, old_directory, device)
    if trained["meta"] != old_trained["meta"]:
        raise ValueError("The two checkpoints have different taxonomies")
    meta = trained["meta"]
    rows = read_split(cfg, "val_known", meta)
    old_rows = read_split(old_cfg, "val_known", meta)
    if [(r["image_sha256"], r["true_leaf"]) for r in rows] != [
            (r["image_sha256"], r["true_leaf"]) for r in old_rows]:
        raise ValueError("Compare the same ordered known-validation images only")
    audit_rows({"val_known": rows}, forbidden_hashes=trained["audit"]["train"]["image_hashes"])
    audit_rows({"val_known": old_rows}, forbidden_hashes=old_trained["audit"]["train"]["image_hashes"])
    scoring_cfg, old_scoring_cfg = copy.deepcopy(cfg), copy.deepcopy(old_cfg)
    scoring_cfg["data"]["eval_batch_size"] = old_scoring_cfg["data"]["eval_batch_size"] = args.batch_size
    calls = {
        "new": lambda: new.collect({"val_known": rows}, scoring_cfg, meta, encoder, evidence, bank, device),
        "v11": lambda: old.collect({"val_known": old_rows}, old_scoring_cfg, meta, backbone, heads, old_support, device),
    }
    seconds = {"new": [], "v11": []}
    with run_lock(directory), torch.no_grad():
        for iteration in range(args.warmup + args.repeats):
            # Alternate ordering to reduce cache/temperature ordering bias.
            for name in (("v11", "new") if iteration % 2 == 0 else ("new", "v11")):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    calls[name]()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                if iteration >= args.warmup:
                    seconds[name].append(elapsed)
        medians = {key: statistics.median(value) for key, value in seconds.items()}
        ratio = medians["new"] / medians["v11"]
        report = {"scope": "known_validation_end_to_end_scoring_only", "test_images_read": False,
                  "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
                  "batch_size": args.batch_size, "unique_images": len(rows), "seconds": seconds,
                  "median_seconds": medians, "new_over_v11_time": ratio,
                  "new_not_slower_in_this_measurement": ratio <= 1.0,
                  "images_per_second": {k: len(rows)/v for k, v in medians.items()},
                  "new_checkpoint": trained["checkpoint"]["sha256"],
                  "v11_checkpoint": old_trained["checkpoint"]["sha256"],
                  "note": "Repeated inference timing, not proof of training-speed parity or accuracy."}
        write_json(output, report)
    print("new/v11 median scoring time: {:.4f}; report: {}".format(ratio, output))


if __name__ == "__main__":
    main()
