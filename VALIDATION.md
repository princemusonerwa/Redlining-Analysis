# Validation performed

- Python syntax compilation passed for active source files.
- Five focused tests passed: seeded sampling without replacement and chunk independence; conservative cleaning and duplicate flags; explicit non-Hispanic reference grouping; signed-log handling of negative income; runnable entrypoints for every analysis stage.
- End-to-end synthetic smoke run passed with 6,000 artificial raw records, all six model families, fairness, regression ladder, matching, neighborhood analysis, pricing, SHAP, Markdown/HTML reporting and the tabbed publisher.
- All referenced report tables and figures were present.
- A second unchanged run skipped all ten completed analysis stages and republished successfully.
- A standalone downstream request rejected missing prerequisites. Raw-byte changes altered the resume fingerprint.

No live HMDA API download or real-data/full-size modeling run was performed. The smoke run uses deliberately reduced model sizes, two CV folds and fewer resamples. Synthetic outputs and datasets are excluded from the distributed package. Dependency pins are retained from the supplied project; use a clean Python 3.11/3.12 virtual environment. The shared test environment emitted third-party binary/deprecation warnings but completed the run successfully. This validates integration, not scientific conclusions or the full statistical methodology.
