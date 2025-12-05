# Minor fixes and stage setting

2.1: Correlation_heatmap.png: when there are a large number of columns, we need to reduce font size and suppress the text for very large number of columns: overplotting makes it difficult to see numbers.
5.1: trts_comprehensive_analysis.png: When overall performance metrics are close to one, the text above the barchart overlaps with the title.  Could y-axis limits be increased to allow for text to fit within the graphic when values are at the max value of 1?

2.2: feature_distribution.png: when there are a large number of columns, the graphic becomes very long leading to awkward dimensions for copying and pasting.  Perhaps limit the number of histograms displayed per file and use multiple files.  Perhaps 3x2 grids? How else can we solve this problem?
3.1: correlation_comparison.png: when there are a large number of columns, we need to reduce font size and suppress the text for very large number of columns: overplotting makes it difficult to see numbers. Compared with task 2.1, this will be similar, but adjustments should take into consideration that this graphic juxtaposes two correlation heatmaps.  
3.2: distribution_comparison.png. Similar to 2.2, when there are a large number of columns, the graphic becomes very long leading to awkward dimensions for copying and pasting.  Perhaps limit the number of histograms displayed per file and use multiple files.  Perhaps 3x2 grids? How else can we solve this problem?  Let us find a common solution.

2.3: Create a readme.md file for the Section 2 folder.  In it provide a stock explanation of what each file provides.  This is intended to orient a reviewer to output - so this text can routinely accompany the production of Section 2 output. 
3.3: Create a readme.md file for the Section 3 folder.  In it provide a stock explanation of what each file provides.  In this section, some .csv files should have column definitions stated. In particular:
3.3a: evaluation_summary.csv - provide a verbal description (no formulas) for each of the column names
3.3b: statistical_similarity.csv - provide a verbal description (no formulas) for each of the column names
3.3c: pca_comparison_with_outcome.png - provide some explanation and guidance on how to interpret the plots. A couple of sentences for each plot.



# Advanced 
3.4. Let us consider additions to the evaluation_summary.csv.
3.4a: Can we include some mode collapse detection? E.g., some flag can be thrown if say all synthetic columns are Males, while real data contains both males and females.  Perhaps another .csv file can be created to provide details - e.g., columns X, Y, and Z exhibit evidence of mode collapse impacting the variety of output.  
3.4b: are there higher-order similarlity metrics we could include, e.g., mutual information or some copula metrics?
3.5: trts_detailed_results.csv. Currently we have Accuracy and training time, let us expand:
Accuracy
Sensitivity
Recall
True Positive Rate (TPR)
Specificity
True Negative Rate (TNR)
Precision
Positive Predictive Value (PPV)
Negative Predictive Value (NPV)
False Positive Rate (FPR)
False Negative Rate (FNR)
False Discovery Rate (FDR)
False Omission Rate (FOR)
F1 Score
F_\beta Score
Balanced Accuracy
Youden’s J Statistic
Matthews Correlation Coefficient (MCC)
Fowlkes–Mallows Index (FMI)
ROC Curve
Area Under the ROC Curve (AUC, AUROC)
Precision–Recall Curve
Area Under the Precision–Recall Curve (AUPRC)
Average Precision (AP)
Brier Score
Calibration Curve / Reliability Diagram
Prevalence (Base Rate)
Predicted Positive Rate
Cohen’s Kappa (for multi‑class / agreement)

Additionally, I want to include metrics associated with privacy: 
Nearest neighbor distance (synthetic to real) – detects potential memorization or near duplicates of real individuals
Nearest neighbor distance (real to synthetic) evaluates how closely synthetic data shadow specific real individuals
Membership inference risk. – attack tries to infer whether a real record was in the training set based on access to synthetic data or the generator
Attribute inference risk – given partial attributes of an individual, attacker uses synthetic data/model to infer hidden attributes
Uniqueness/re-identification rate – fraction of synthetic rows that are exact or near exact matches to real rows under a defined matching rule
Effective sample size/diversity metrics – number of unique rows, entropy of key categorical variables or related measures of diversity in synthetic data


# Review of objective functions

What can we do to introduce early stopping?

Create a readme.md file for this folder.  In it provide a stock explanation of what each file provides.  This is intended to orient a reviewer to output - so this text can routinely accompany the production of Section 2 output. 
