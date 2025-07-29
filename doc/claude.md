# Clinical Synthetic Data Generation Framework - GANerAide generalization

## Overview

First: read doc/GANerAid_Demo_Notebook.ipynb and doc/clinical_synth_demo.ipynb
Second: to get understanding of typical columns in tables we'll work with look at sample datasets in the /data folder
Third: Look at the structure of this current project and make changes as necessary to the file structure. 

Critical: Read what is here first to understand what is intended. 

Goal: Create a new ipynb along the lines of doc/GANerAid_Demo_Notebook.ipynb that will take in a .csv file.  Add placeholder sections for pre-processing.  Regarding the graphics and numeric summaries, I want to include the summaries used here: doc/GANerAid_Demo_Notebook.ipynb and notebooks/clinical_synth_demo.ipynb but I want things organized effectively.  Also, Let's make graphics and summaries simple along the lines of doc/GANerAid_Demo_Notebook.ipynb.  Consider small additions to code such as ability to optionally output figures and table summaries.

Phase 0:  Outline tasks needed to complete. Deliver to user the complement of visualizations and tables to be produced - broken down by EDA, similarity, accuracy, classification metrics etc.
Phase 1: Create a .ipynb file to work with Breast_cancer_data (essentially providing an update to GANerAid_Demo_Notebook.ipynb) 
Phase 2: Tailor changes and create a new .ipynb file for Pakistani_Diabetes_Dataset
Phase 3: Read C:\ForGit\PyTorchUltimateMaterial\GC-conditionalGAN\LiverCode2.ipynb with emphasis on preprocessing steps through One-Hot encoding of categorical data + MICE.  Build a new .ipynb file along the lines of our update to GANerAid_Demo_Notebook.ipynb, adding a section to address the pre-processing step
Phase 4: Read compareModels.py and the file built in Phase 3.  Our next step is to create an .ipynb file to compare CTGAN, TVAE (Tabular VAE), CopulaGAN, TableGAN, GANerAid.  The code in compareModels.py is just a guide.  The final .ipynb file should be user freindly.  Also, if large blocks of code could be sources from elsewhere this would be preferable. Focus should be on EDA, simulation setup (let there be a table summarizing simulation plan), sections detailing the best of each tuned model, section contrasting model similarity and performance metrics, etc.


