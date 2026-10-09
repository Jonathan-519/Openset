"""Decode every tracked image without editing it; writes a compact QA summary."""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from audit_repository import inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-root", required=True, help="Full clone containing prepro/raw")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.image_root).resolve()
    paths = [name for name, item in inventory().items() if item["kind"] == "image"]

    def inspect(name):
        try:
            with Image.open(root / name) as image:
                image.load()
                return {"path": name, "size": image.size, "mode": image.mode}
        except Exception as exc:
            return {"path": name, "error": type(exc).__name__ + ": " + str(exc)}

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(inspect, paths))
    good = [r for r in records if "size" in r]
    sizes = np.asarray([r["size"] for r in good])
    report = {"tracked_image_count": len(paths), "decoded_count": len(good),
              "errors": [r for r in records if "error" in r],
              "colour_modes": dict(Counter(r["mode"] for r in good)),
              "dimensions_min_median_max": np.quantile(sizes, [0, .5, 1], axis=0).tolist() if len(good) else None,
              "min_side_below_224": int(np.sum(sizes.min(1) < 224)) if len(good) else None,
              "scope": "Every tracked image fully decoded; no human biological relabeling or near-duplicate guarantee."}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
