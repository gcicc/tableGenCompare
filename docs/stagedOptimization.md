Next round of improvement: Group Sequential Optuna hyperparameter optimization

1. For some reason CTGAN is failing on STG-Driver-liver-train.ipynb.  Let's fix this first

Then let's create a STG-Driver-breast-cancer2.ipynb file based on STG-Driver-breast-cancer.ipynb, but let's have section 4 accomplish the following in the new version:

1. Run pilot hyperparameter optimization - let's use a default of 15 per model.
2. Produce output as we currently do.
3. Note the time it took to run the pilot and provide user with 
a. estimate of how many trials each model would run in an hour
b. offer the user the oppurtunity to provide 
  i) a common number of trials to complement the pilot
  ii) user specifies a different number of trials for each algorithm
  ii) based on user specified common compute time for each model, use 3a to set the number trials for each model.  E.g., I allocate 90 minutes to each model; based on number of models completed in 15 minutes, make the adjustment.
4. The user might want to dedicate more time/trials. E.g., we might run a batch on Monday, inspect, run another batch, inspect.  We want optimization to pick up where it left off and continue to improve our estimate of how many trials each model would run in an hour after each batch report. 
5. If there can be some assessment/recommendation offered about adding more trials, that would help the user decide when they've hit the point of diminishing returns.
6. Finally, the current output is still very verbose. Let's get the trial report down to one line and let's also alert which trial we are on: e.g., Cumulative training trial number 24: report Combined Score: X.XXXX (Similarity: X.XXXX, Accuracy: X.XXXX) Best Combined Score so far: X.XXXX
7. 