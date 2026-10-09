"""Optional separate training/calibration/TEST commands for the new suite."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
from . import protocol,runner,reporting


def run_phase(phase):
    parser=argparse.ArgumentParser(description="C00 evidence-guard suite: "+phase)
    parser.add_argument("--run-dir",required=True,type=Path)
    parser.add_argument("--config",default=protocol.DEFAULT_CONFIG)
    parser.add_argument("--reference-run-dir",default=protocol.DEFAULT_SOURCE)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=parser.parse_args()
    suite=protocol.resolve(str(args.run_dir))
    if phase=="training":
        cfg=protocol.effective_config(protocol.resolve(args.config))
        source=protocol.resolve(args.reference_run_dir)
        audit=runner.preflight(cfg,source,suite,args.device)
        if not audit["raw_images_valid"]:raise ValueError(str(audit["raw_image_audits"]))
        runner._initialize(suite,cfg,runner._source(source),args.device)
    else:
        cfg=protocol.validate_config(protocol.read_json(runner._regular(suite/"config.json")))
        source=Path(protocol.read_json(suite/"source_binding.json")["directory"])
        snapshot=runner._verify_snapshot(suite,cfg,runner._source(source))
        if snapshot["device"]!=args.device:raise ValueError("Device differs from suite")
    failures=[]
    try:
        with protocol.run_lock(suite):
            if phase=="training":
                ok=[runner._execute(suite,cfg,source,None,"cache_"+stage,args.device) for stage in ("train","development")]
                if not all(ok):raise ValueError("Feature preparation failed; see cache logs")
            if phase=="test":
                reporting.freeze_dev_selection(suite)
                if not runner._execute(suite,cfg,source,None,"cache_test",args.device,resume=True):
                    raise ValueError("TEST feature preparation failed")
            for arm in cfg["arms"]:
                name=arm["id"]
                required="training" if phase=="calibration" else "calibration" if phase=="test" else None
                dependency=runner._dependency(arm)
                unavailable=[]
                if required and not (suite/"arms"/name/required/runner.STAGE_MARKER).is_file():
                    unavailable.append(name+"/"+required)
                if dependency and not (suite/"arms"/dependency/"training"/runner.STAGE_MARKER).is_file():
                    unavailable.append(dependency+"/training")
                if unavailable:
                    if not (suite/"arms"/name/"failure.json").is_file():
                        runner._failure(suite,name,"dependency","Required stage unavailable: "+", ".join(unavailable))
                    failures.append(name);continue
                if (suite/"arms"/name/phase).exists() and not (suite/"arms"/name/phase/runner.STAGE_MARKER).is_file():
                    failures.append(name);continue
                if not runner._execute(suite,cfg,source,name,phase,args.device,resume=True):failures.append(name)
            if phase=="calibration":reporting.freeze_dev_selection(suite)
            if phase=="test":reporting.summarize_suite(suite)
    finally:
        if phase=="test" and (suite/"snapshot.json").is_file():
            from tools.pack_taxosafe_evidence_guard_review import pack
            stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            pack(suite,suite.with_name(suite.name+"_review_"+stamp+".tar.gz"))
    if failures:raise SystemExit(2)
