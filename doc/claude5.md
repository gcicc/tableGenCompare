Note: TableGAN has been causing multiple issues. 
1. Start new branch called remove-tableGAN.

I want to remove all reference to TableGAN from C:\Users\gcicc\claudeproj\tableGenCompare\Clinical_Synthetic_Data_Generation_Framework.ipynb and supporting files. This should include instances of TableGAN in section 1.4. Similarly remove instances of TableGAN from section 2.5.  Have an agent review this notebook and all files supporting this notebook to ensure that references to TableGAN - including descriptions of it in the appendix have been removed.
Desired outcome: All Sells of section 1 run without error, Section 1 ans section 2 subsection numbering should reflect the removal of TableGAN.
WE WILL PAUSE HERE FOR USER TO TEST

3.  I wish to add to Section 1.  In particular, let us add CTAB-GAN and CTAB-GAN+.  Use https://github.com/Team-TUD/CTAB-GAN.git and https://github.com/Team-TUD/CTAB-GAN-Plus.git.  
4. Reorder the subsections of section 1: CTGAN, CTAB-GAN, CTAB-GAN+, GANerAide, CouplaGAN, TVAE and renumber appropriately.
Desired outcome: All Sells of section 1 run without error, Section 1 reorganized as described.  
WE WILL PAUSE HERE FOR USER TO TEST
5. Similarly, I wish to complement the hyperparameter subsections along the lines of CTGAN in Section 2.  In this way, CTAB-GAN and CTAB-GAN+ become part of the model comparison exercise we are build - serving as 2 replacement models for TableGAN.
WE WILL PAUSE HERE FOR USER TO TEST
6.  Reporting and model comparison, should now account for the update to the collection of models we're using.