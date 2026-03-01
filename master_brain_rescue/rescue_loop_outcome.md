# Rescue Loop Outcome Summary

- Rescue jobs executed: **1**
- Rescue job failures: **0**
- Backtests scored before: **296**
- Backtests scored after: **300**
- Promoted runs before: **0**
- Promoted runs after: **0**

## Next tuning focus

- If promotes remain 0, prioritize ML quality uplift (CV completeness + calibration) for rescued variants.
- Iterate with lower-variance features and stronger probability calibration, then rerun this rescue loop.
- Keep stress and risk gates strict while lifting model-quality reliability.

- Comparison CSV: `C:\Users\dunnm\source\repos\finance-scrapers-portfolio\master_brain_rescue\rescue_before_after_comparison.csv`
- Rescue execution CSV: `C:\Users\dunnm\source\repos\finance-scrapers-portfolio\master_brain\rescue_execution_report.csv`
