from pathlib import Path
import re
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent

PREPRO_ROOT = PROJECT_ROOT / "prepro"

TAXONOMY_ROOT = (
    PREPRO_ROOT
    / "raw"
    / "Zooplankton_Taxonomic_Tree"
)

OOD_ROOT = (
    PREPRO_ROOT
    / "raw"
    / "Zooplankton_OOD"
)

OUTPUT_ROOT = (
    PREPRO_ROOT
    / "data"
    / "Zooplankton_Taxonomic_Tree"
)

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def list_images(folder):
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    images = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    ]

    return sorted(images, key=natural_key)


def save_rows(filename, rows):
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_ROOT / filename

    if output_path.exists():
        backup_path = output_path.with_suffix(
            output_path.suffix + ".bak"
        )

        if not backup_path.exists():
            shutil.copy2(
                output_path,
                backup_path,
            )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        for index, (path, label) in enumerate(rows):
            stream.write(
                "{},{},{}\n".format(
                    path,
                    label,
                    index,
                )
            )

    print(
        "{}：{} 张".format(
            filename,
            len(rows),
        )
    )


def build_intra_rows(specification):
    rows = []

    for parent_id, parent_name, species_name in specification:
        species_dir = (
            TAXONOMY_ROOT
            / parent_name
            / species_name
        )

        for image_path in list_images(species_dir):
            # full_data_root: prepro
            relative_path = image_path.relative_to(
                PREPRO_ROOT
            ).as_posix()

            rows.append(
                (relative_path, parent_id)
            )

    return rows


def build_extra_rows(split_name):
    rows = []
    split_root = OOD_ROOT / split_name

    if not split_root.is_dir():
        raise FileNotFoundError(split_root)

    images = sorted(
        (
            path
            for path in split_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: natural_key(path),
    )

    for image_path in images:
        # ood_root: .
        relative_path = image_path.relative_to(
            PROJECT_ROOT
        ).as_posix()

        rows.append(
            (relative_path, -1)
        )

    return rows


val_intra_specification = [
    (
        3,
        "Copepoda",
        "Centropages_tenuiremis",
    ),
    (
        5,
        "Medusae",
        "Eirene",
    ),
]

test_intra_specification = [
    (
        3,
        "Copepoda",
        "Acartia_pacifica",
    ),
    (
        3,
        "Copepoda",
        "Oithona_similis",
    ),
    (
        5,
        "Medusae",
        "Turritopsis_nutricula",
    ),
]


val_intra_rows = build_intra_rows(
    val_intra_specification
)

test_intra_rows = build_intra_rows(
    test_intra_specification
)

oe_train_rows = build_extra_rows(
    "oe_train"
)

val_extra_rows = build_extra_rows(
    "val_extra"
)

test_extra_rows = build_extra_rows(
    "test_extra"
)


expected_counts = {
    "val_intra": 405,
    "test_intra": 326,
    "oe_train": 13,
    "val_extra": 126,
    "test_extra": 138,
}

actual_counts = {
    "val_intra": len(val_intra_rows),
    "test_intra": len(test_intra_rows),
    "oe_train": len(oe_train_rows),
    "val_extra": len(val_extra_rows),
    "test_extra": len(test_extra_rows),
}

if actual_counts != expected_counts:
    raise ValueError(
        "数据数量不符合预期。\n"
        "期望：{}\n"
        "实际：{}".format(
            expected_counts,
            actual_counts,
        )
    )


save_rows(
    "gt_val_intra.txt",
    val_intra_rows,
)

save_rows(
    "gt_test_intra.txt",
    test_intra_rows,
)

save_rows(
    "gt_oe_train.txt",
    oe_train_rows,
)

save_rows(
    "gt_val_extra.txt",
    val_extra_rows,
)

save_rows(
    "gt_test_extra.txt",
    test_extra_rows,
)