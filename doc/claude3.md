

Next, We will recreate multi_model_breat_cancer_demo_hypertune_v2.ipynb with these changes:
1. Sections of multi_model_breat_cancer_demo_hypertune_v2.ipynb will have a 1 to 1 correspondance with sections of multi_model_breat_cancer_demo_hypertune.ipynb
2. Upgrade the hyperparameter spaces definitions of each model by leveraging examples found in hypertuning_ed.md for hyperparameters and sample optuna implementation. Have an agent review the hyperparameter spaces for each model and make adjustments so they are robust for various tables we might encounter generally.  Remove reference to demo_epochs (this parameter should be tuned). Let demo_samples be equal to size of original data set by default.
3. Let us update the objective function used by optuna.  Read Phase1_Breast_cancer_enhanced_GANerAid.ipynb - we want to employ the 60% similarity (which itself is a 60% 40% combination of univariate and bivariate similarity. These should be based on earth mover's distance.  If in bivariate case, if EMD is not avail (even after help from agent to locate), use euclidean distance between correlation matricies associated with read and synthetic data) and 40% accuracy (which is an average of accuracy metrics associated with TSTR/TRTR and TRTS/TRTR). Ensure the similarity component and the accuracy component are scaled to 1 so that the optuna has preference for models with larger objective values (and these will be naturally capped at 1).  PROVIDE ME A CRITIQUE OF MY PROPOSAL AS WE ITERATE.
4. Add a markdown chunk to provide a summary of the hyperparameter spaces to be studied before running the optuna optimization. This should precede the chunk where we run the optimization.
5. For Each model output graphics associated with discriminator and generator history to file.   
5a. Read Phase1_breast_cancer_enhanced_GANerAID.ipynb.  For each model, output a graphic along the lines of ENHANCED OPTMIZATION ANALYSIS to file.
6. Read GANerAID_Demo_Notebook.ipynb and add graphics found in EVALUATION REPORT (evaluation_report.plot_evaluation_metrics)
     * Ensure all output from OPTIMIZED GANERAID EVALUATION METRICS VISUALIZATION is included in final comparison and these are included in the notebook's final section
     * Ensure ENHANCED STATISTICAL ANALYSIS - OPTIMIZED MODEL is included in final comparison and provided in notebook's final section
7. In the end multi_model_breat_cancer_demo_hypertune_v2.ipynb should be apppriately renamed.
8. Include an appendix that provides some conceptual descriptions of the 5 models and how they differ. Have an agent research if there are certain settings where one model might perform better. Include references to seminal papers on the 5 methods. Include references to python packages that we are leveraging.
9.  Include an appendix detailing how optuna works, perhaps taking CTGAN implementation as an example. Include appropriate references.
10. Include an appendix that describes the objective function we are using. Provide some conceptual description of EMD, and other metrics employed. Include appropriate references.
11. Include an appendix that discusses the rationale for hyperparameter space choices in general and again, use CTGAN as an example to detail.


Desired outcome: Production ready notebook that I can run to identify best model for generating synthetic Breast Cancer Data.  Appedices are intended to bring team members up to speed with important concepts related to how the algorithms and optimizations work.

Once we have a this working example, I want to clean up this branch of superfluous files. 

Next Step: create a new a new file adjusting for alzheimers_disease.csv

And so on with the other .csv files in /data

We will work in a staged fashion.  Once you've completed your read of this let me know.