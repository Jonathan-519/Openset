"""Build Fold 1's known-species view without replacing existing paths."""
import argparse
import json
from pathlib import Path


PARENTS = [
    "Amphipoda",
    "Appendiculata",
    "Cladocera",
    "Copepoda",
    "Euphausiacea",
    "Medusae",
    "Sagittoidea",
]


def build_known_view(full_root, fold_file, view_root):
    full_root = Path(full_root).resolve()
    view_root = Path(view_root).resolve()

    with open(fold_file, "r", encoding="utf-8") as f:
        split = json.load(f)

    # Validate every source and existing target before creating any links.
    pending = []
    for parent in PARENTS:
        parent_view = view_root / parent
        for species in split[parent]["known"]:
            if Path(species).name != species or species in (".", ".."):
                raise ValueError("Invalid species directory: {}".format(species))
            source = full_root / parent / species
            target = parent_view / species

            if not source.is_dir():
                raise FileNotFoundError(source)

            if target.exists() or target.is_symlink():
                if target.resolve() != source.resolve():
                    raise FileExistsError(
                        "Existing view points elsewhere; inspect manually: {} -> {} "
                        "(expected {})".format(target, target.resolve(), source)
                    )
                continue
            pending.append((source, target))

    for source, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)
    return len(pending)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, default=root / "raw/Zooplankton_Taxonomic_Tree")
    parser.add_argument("--fold-file", type=Path, default=root / "splits/Zooplankton_Taxonomic_Tree/fold1.json")
    parser.add_argument("--view-root", type=Path, default=root / "views/Zooplankton/fold1_known")
    args = parser.parse_args()
    count = build_known_view(args.full_root, args.fold_file, args.view_root)
    print("Created {} species links; existing links verified.".format(count))
