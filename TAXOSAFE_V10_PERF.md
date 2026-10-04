# TaxoSafe-v10 performance branch

This branch upgrades the v9_rebuild pipeline for higher end-to-end open-set
performance without adding external training images.

## Protocol

- Locked test manifests are unchanged and never loaded during training/calibration.
- Original v9 development unknowns are stratified 60/40 within species/source.
- 60% near-unknown development images -> train_intra only.
- 40% near-unknown development images -> val_intra router calibration only.
- 60% global-OOD development images -> oe_train only.
- 40% global-OOD development images -> val_extra router calibration only.
- Checkpoint selection remains val_known leaf accuracy only.

## Main changes

1. Real near-unknown supervision:
   - true-parent cross entropy;
   - explicit local-unknown classification against known children;
   - margin forcing local-unknown above the strongest known child.
2. Existing pseudo-open training strengthened with:
   - local-unknown prompt loss;
   - taxonomy-weighted contrastive loss;
   - sibling-boundary manifold mixup.
3. Development-only OE for global OOD.
4. TRAIN-only normalized leaf prototypes + ViM residual.
5. Learned root fusion for (known + near unknown) versus global OOD.
6. Learned local fusion for known leaf versus unseen sibling.
7. Joint held-out threshold search maximizing macro deepest-reliable accuracy
   with a minimum known end-to-end accuracy constraint.

## Run

```bash
cd /home/ubuntu/hdd/data/qz/Openset
git checkout taxosafe-v10-perf

python prepro/build_taxosafe_v10_dev_splits.py

python -m unittest tests.test_taxosafe_v10_protocol
python -m unittest tests.test_taxosafe_suite

for SEED in 1 2 3 4 5; do
  python train_taxosafe.py \
    --config configs/Zooplankton_Taxonomic_Tree/Zooplankton_Taxonomic_Tree_v10_perf.yml \
    --trial "$SEED" --seed "$SEED"

  python calibrate_taxosafe_v10.py \
    --config configs/Zooplankton_Taxonomic_Tree/Zooplankton_Taxonomic_Tree_v10_perf.yml \
    --trial "$SEED"

  python test_taxosafe_v10.py \
    --config configs/Zooplankton_Taxonomic_Tree/Zooplankton_Taxonomic_Tree_v10_perf.yml \
    --trial "$SEED"
done
```

## Acceptance targets relative to v9 trial_1

Treat these as experiment gates, not guaranteed outcomes:

- closed known leaf accuracy >= 90%;
- known end-to-end leaf accuracy >= 70%;
- near-unknown OSER <= 15%;
- near-unknown correct fallback >= 70%;
- global-OOD recall >= 75%;
- global-OOD false-known-leaf rate <= 7.5%;
- overall deepest-reliable-taxon accuracy >= 70%;
- open-world accepted-leaf precision >= 85%.

Report mean/std across at least five seeds. Do not retune on locked test.
