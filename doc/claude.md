# Clinical Synthetic Data Generation Framework - GANerAide generalization

## Overview

First: read doc/GANerAid_Demo_Notebook.ipynb and doc/clinical_synth_demo.ipynb
Second: to get understanding of typical columns in tables we'll work with look at sample datasets in the /data folder
Third: Look at the structure of this current project and make changes as necessary to the file structure. 

Critical: Read what is here first to understand what is intended. 

Goal: Create a new ipynb along the lines of doc/GANerAid_Demo_Notebook.ipynb that will take in a .csv file.  Add placeholder sections for pre-processing.  Regarding the graphics and numeric summaries, I want to include the summaries used here: doc/GANerAid_Demo_Notebook.ipynb and notebooks/clinical_synth_demo.ipynb but I want things organized effectively.  Also, Let's make graphics and summaries simple along the lines of doc/GANerAid_Demo_Notebook.ipynb.  Consider small additions to code such as ability to optionally output figures and table summaries.

Phase 0:  Outline tasks needed to complete. Deliver to user the complement of visualizations and tables to be produced - broken down by EDA, similarity, accuracy, classification metrics etc.
Phase 1: Create a .ipynb file to work with Breast_cancer_data (essentially providing an update to GANerAid_Demo_Notebook.ipynb) 
Phase 2: Tailor changes and create a new .ipynb file for Pakistani_Diabetes_Dataset.  In this version, focus on making call out for where user must note location of file - we'll want to generalize this.
Phase 3: Read C:\Users\gcicc\claudeproj\tableGenCompare\notebooks\Phase2_Pakistani_Diabetes_Enhanced_GANerAid.ipynb
Brainstorm how we can generalize the initial bit where user loads and pre-processes files. Let's think about the steps needed - like renaming columns, etc. Again, add call outs for the user and prompt the user for background information and reference on their incoming data set.  In this way, the next version of .ipynb file should be setting agnostic.  (However, for purpose of demo, let us use C:\Users\gcicc\claudeproj\tableGenCompare\doc\liver_train.csv)
Phase 4: Read compareModels.py and the file built in Phase 3.  Our next step is to create an .ipynb file to compare CTGAN, TVAE (Tabular VAE), CopulaGAN, TableGAN, GANerAid.  The code in compareModels.py is just a guide.  The final .ipynb file should be user freindly.  Also, if large blocks of code could be sources from elsewhere this would be preferable. Focus should be on EDA, simulation setup (let there be a table summarizing simulation plan), sections detailing the best of each tuned model, section contrasting model similarity and performance metrics, etc.  
Phase 5: As you author this .ipynb employ multiple agents that will ensure some harmonization with previous version, that hyperparemter space is reasonably exhaustive (though offer suggestions on where to par back).  What is most important is the organization of the displays for review. You audience is a clinical development team looking to assess the value of synthetic data.
Phase 6: First delete any existing reference and work conducted to date on Phase 6.  We need to start from scratch - this includes existing phase 6 .ipynb file.  Review and report the features of of the .ipynb files in the notebooks directory.  Let us itemize the union of features and outputs found. The plan will be to advance the file created in Phase 5 to make it more holistic with regard to output previously suggested. Create a long .ipynb file that includes steps to bring in data (in this case, let's use pakistani diabetes).  Main steps should include: loading data, data pre-processing - including missing data handling with MICE, visualization and tabular summaries of incoming data.  Section to setup simulation specs including a place where hyperparmeter space can be augmented.  Section to run simulation. Identification and re-run of models with best hypertuning. Section to report visualization and tabular comparisons with original data for each model.  Section to provide final identification of best of the data generation approaches.  In this stage everything should be self contained in a single .ipynb file.  Again, let us tailor this specifically to pakistani data set.  First provide an outline of tasks for me to review and approve.


