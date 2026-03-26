# Comparison of Hyperparameter Optimization (HPO) Samplers

This report compares two different samplers used in Optuna for Hyperparameter Optimization (HPO) for the Random Forest model: **TPE (Tree-structured Parzen Estimator)** and **RandomSampler**.

Both samplers were run for 20 trials to optimize `n_estimators` (50 to 200) and `max_depth` (2 to 15), maximizing the F1 score.

## 1. Best Metric Achieved
- **TPE Sampler:** Achieved a best F1 score of **0.79994** (Parameters: `n_estimators` = 161, `max_depth` = 15).
- **Random Sampler:** Achieved a best F1 score of **0.79920** (Parameters: `n_estimators` = 106, `max_depth` = 15).

**Conclusion on Metric:** TPE achieved a slightly better metric because it adapts its search based on previous successful trials, eventually finding a slightly more optimal parameter combination towards the end of its run (Trial 17). 

## 2. Stability of Convergence
- **TPE Sampler:** TPE shows a more structured exploration approach. While it investigates different areas (causing some drops in metric), it visibly focuses on the most promising areas (high `max_depth`), yielding consistent high scores around ~0.799 in multiple trials (Trial 0, Trial 5, Trial 11, Trial 13, Trial 17).
- **Random Sampler:** The Random sampler is completely unstable by definition. It randomly bounces between high and low scores (from 0.354 to 0.799). It happened to find a good score at Trial 0 by pure chance but failed to improve upon it iteratively during the rest of the run.

**Conclusion on Stability:** TPE is far more stable and directed in its search, converging consistently on the best hyperparameters, whereas Random Sampler relies purely on luck.

## 3. Speed of Convergence / Execution Time
- **TPE Sampler:** Total execution time for 20 trials was ~**76 seconds**.
- **Random Sampler:** Total execution time for 20 trials was ~**55 seconds**.

**Conclusion on Speed:** Random Sampler is faster per trial because it involves zero overhead in deciding the next parameters to test. TPE takes slightly longer overall because the algorithm has to compute the probability distributions and mathematically select the next best parameters based on previous results. However, TPE's computational cost provides a better quality solution.

## Summary
For this scenario, **TPE is the superior choice**. Despite taking about 20 seconds longer over 20 trials, it actively learns from past evaluations, providing a slightly better best metric and mathematically directed exploration, contrary to Random's blind guessing.
