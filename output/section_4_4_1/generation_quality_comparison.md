# Table A. Generation quality comparison

Reference: 247 real anomaly training windows. All methods evaluated on 750 windows from saved 1,000-window pools. Lower is closer for every error metric.

| Method | N | Mean difference | Std difference | Correlation difference | ACF difference |
|---|---:|---:|---:|---:|---:|
| GT-GAN | 750 | 0.77791 | 0.33782 | 0.45179 | 0.26604 |
| Diffusion | 750 | 0.10910 | 0.29683 | 0.35399 | 0.28500 |
| Masking GT-GAN | 750 | 0.00520 | 0.00152 | 0.00210 | 0.00251 |
| Masking Diffusion | 750 | 0.01006 | 0.01236 | 0.01352 | 0.00473 |
| FMD (default filter) | 750 | 0.05229 | 0.02231 | 0.02339 | 0.00922 |

Non-filtered methods: random sampling without replacement, seed 42. FMD: default filter selects 750 candidates from the full Masking Diffusion pool. No model retraining.

ACF: pooled within-window autocorrelation at observation lags 1-10, averaged in absolute difference over the five input features. No products cross window boundaries. See the sibling JSON for full definitions and per-feature values.

These are training-reference similarity results, not proof of independent generalization, diversity, or downstream detection improvement.
