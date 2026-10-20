# Migration from the original pipeline

| Original | Updated |
|---|---|
| Download_Data.ipynb (exploratory) | steps/s01_download.py |
| Cleaning_Redlining_Analysis_Modelling.ipynb (exploratory) | steps/s02_clean.py plus preparation/model transforms |
| EDA notebook exploration | steps/s03_eda.py |
| steps/s01_prepare.py | steps/s04_prepare.py |
| steps/s02_models.py | steps/s05_models.py |
| steps/s03_fairness.py | steps/s06_fairness.py |
| steps/s04_same_finances.py | steps/s07_same_finances.py |
| steps/s05_neighbourhood.py | steps/s08_neighbourhood.py |
| steps/s06_pricing.py | steps/s09_pricing.py |
| steps/s07_shap.py | steps/s10_shap.py |
| steps/s08_report.py | steps/s11_report.py |

Root configuration, shared helpers, runner and publisher use the new names.
Download is explicit. Cleaning outputs feed EDA and preparation; preparation
writes the categorical Parquet contract used by the downstream models/analyses.
The original model families, fairness metrics, adjusted analyses, matching,
pricing and SHAP computations remain, with integration fixes documented in README.

This is not numerical equivalence with the old pipeline: sampling (when a new
sample is downloaded), demographic definitions, removal of global percentile
caps, signed logarithms and CV-based selection can change results. Reports now
present computed tables and neutral descriptions rather than assuming a fixed
direction of disparity or reusing old performance numbers. The tabbed publisher
is rebuilt from a shared section manifest. Original notebooks are archived,
not converted cell-for-cell; some exploratory charts remain only in notebooks.

The active scientific code is still exploratory: no claim of a complete
methodological audit or completed mitigation experiment is made.
