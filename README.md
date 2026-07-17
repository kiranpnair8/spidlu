# Spi-dLU RQ1 Phase 1

Phase 1 of RQ1 evaluates whether Spi-dLU preserves utility when inserted into
pretrained Hugging Face causal language models. It does not evaluate watermark,
signature-transfer, or ownership-detection metrics.

## Variants

- `ann_original`: load the original pretrained model and evaluate directly.
- `spidlu`: replace the original MLP activation with Spi-dLU and run the
  configured alignment budget.
- `ann_compute_matched`: keep the original ANN activation and run the same
  data order, optimizer, scheduler, seed, update count, token budget,
  validation schedule, and checkpoint rule as `spidlu`.
- `quantized_activation`: replace the same activation with a non-spiking
  straight-through quantized activation with levels matched to Spi-dLU temporal
  resolution where practical.

## Run

```bash
python scripts/run_phase1.py --config configs/phase1_rq1.yaml
```

Smoke mode uses tiny dataset slices and one training step:

```bash
python scripts/run_phase1.py --config configs/phase1_rq1.yaml --smoke
```

Slurm:

```bash
sbatch jobs/run_phase1.sh
```

## Tests

```bash
pip install -r requirements.txt
pytest tests/test_phase1.py
```

The legacy scratch transformer and signature/watermark scripts are retained
until the Phase 1 smoke and surgery gates pass in the target environment.
