# Algorithm Improvement Roadmap (Probing Pass)

## Current bottleneck diagnosis

- Promoted runs: **0**
- Candidate runs: **1**
- Reject runs: **295**
- Mean ML quality score: **0.4727**
- Pass-ML rate: **0.0000**
- Mean stress Sharpe (20 bps): **-0.8877**

## Best near-term policy profile from parallel sweep

- min_reliability: **0.45**
- min_stress_sharpe: **0.2**
- min_ml_quality: **0.35**
- Expected candidates under this profile: **30**
- Near-promote average ML gap: **0.3500**

## Top 10 upgrade actions (impact-ordered)

1. **Fix ML-quality gate bottleneck**: add calibrated probability outputs + CV completeness checks for every model run.
2. **Raise CV coverage to 100%**: enforce minimum fold coverage and fail closed when `cv_metrics.csv` is missing.
3. **Improve feature stack**: add regime, volatility, spread, and higher-moment features with rolling stability diagnostics.
4. **Econometric robustness**: add walk-forward and purged time-series CV to reduce leakage and inflation of in-sample quality.
5. **Time-series forecasting upgrades**: compare baseline vs AR/ARX/state-space/sequence models with identical transaction-cost assumptions.
6. **Technical-analysis signal hygiene**: keep only signals with out-of-sample incremental information value and low collinearity.
7. **Risk-first deployment gates**: require positive stressed return and stress Sharpe under cost/slippage perturbation scenarios.
8. **Parallel compute architecture**: execute model/backtest grid in process-level pools and shard by asset/family for faster iteration.
9. **Business governance layer**: add scorecards (expected capacity, turnover constraints, implementation friction) before promotion.
10. **Automated rescue loop**: retrain top near-promote runs with targeted ML-gap closure and re-gate automatically.

## Near-promote rescue candidates

- `bt_crypto_5y_SOL-USD_rf_wv_dwt_db4`
- `bt_eurusd_sweep_EURUSD=X_hgb_baseline`
- `bt_eurusd_sweep_EURUSD=X_mlp_baseline`
- `bt_eurusd_sweep_EURUSD=X_hgb_stops`
- `bt_eurusd_sweep_EURUSD=X_mlp_stops`
- `bt_crypto_5y_ETH-USD_hgb_wv_dwt_db6`
- `bt_crypto_5y_SOL-USD_hgb_wv_dwt_db4`
- `bt_crypto_5y_SOL-USD_mlp_wv_dwt_db4`
- `bt_crypto_5y_SOL-USD_rf_proba_hmm_wavelet`
- `bt_crypto_5y_ETH-USD_mlp_wv_dwt_db4`
