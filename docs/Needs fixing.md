
We are fixing errors in Section 4 of these notebooks. Let's underscore that sections are developed in a common fashion across notebooks.  Main difference is just in the data set that used.  Note that Section 3 runs 6 demos of synthetic data generation models. Section 4 is simply performing hyperparameter opimization.  

Corrections should be made in setup.py when possible so notebooks are as similar as possible in sections 3,4, 5.

Note that section 2 loads the data, performs multiple imputation and then samples 5000 rows from that imputed data set.  So in the liver notebook below, it would appear that the wrong dataset is being used.  Consider what was done in C:\Users\gcicc\claudeproj\tableGenCompare\SynthethicTableGenerator-Pakistani.ipynb to ensure this dataset was employed and see if that fix should be applied across the rest of the note books.

## Files Affected
- `C:\Users\gcicc\claudeproj\tableGenCompare\SynthethicTableGenerator-Alzheimer.ipynb`

All working as expected to end of section 4 and produced all output as expected. 


- `C:\Users\gcicc\claudeproj\tableGenCompare\SynthethicTableGenerator-BreastCancer.ipynb`
All working as expected to end of section 4 and produced all output as expected. 


- `C:\Users\gcicc\claudeproj\tableGenCompare\SynthethicTableGenerator-Liver.ipynb`
CTGAN CHUNK_040
[WARNING] Could not process column Gender of the patient: y contains previously unseen labels: 'nan'
CTAB-GAN CHUNK_042
[WARNING] Could not process column Gender of the patient: y contains previously unseen labels: 'nan'

CHUNK_044
🔄 CTAB-GAN+ Trial 1: epochs=300, batch_size=128, test_ratio=0.150
ERROR	src.models.implementations.ctabganplus_model:ctabganplus_model.py:train()- CTAB-GAN+ training failed: unsupported operand type(s) for //: 'NoneType' and 'int'
Traceback (most recent call last):
  File "c:\Users\gcicc\claudeproj\tableGenCompare\src\models\implementations\ctabganplus_model.py", line 180, in train
    self._ctabganplus_model.fit()
  File "c:\Users\gcicc\claudeproj\tableGenCompare\src\models\implementations\..\..\..\CTAB-GAN\model\ctabgan.py", line 59, in fit
    self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"],
  File "c:\Users\gcicc\claudeproj\tableGenCompare\./CTAB-GAN\model\synthesizer\ctabgan_synthesizer.py", line 626, in fit
    layers_G = determine_layers_gen(self.gside, self.random_dim+self.cond_generator.n_opt, self.num_channels)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\gcicc\claudeproj\tableGenCompare\./CTAB-GAN\model\synthesizer\ctabgan_synthesizer.py", line 459, in determine_layers_gen
    layer_dims = [(1, side), (num_channels, side // 2)]
                                            ~~~~~^^~~
TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\gcicc\AppData\Local\Temp\ipykernel_22280\4130900770.py", line 45, in ctabganplus_objective
    result = model.train(data,
            ^^^^^^^^^^^^^^^^^
  File "c:\Users\gcicc\claudeproj\tableGenCompare\src\models\implementations\ctabganplus_model.py", line 218, in train
    raise RuntimeError(f"Training failed: {e}")
RuntimeError: Training failed: unsupported operand type(s) for //: 'NoneType' and 'int'
[I 2025-09-18 00:53:46,470] Trial 0 finished with value: 0.0 and parameters: {'epochs': 300, 'batch_size': 128, 'test_ratio': 0.15}. Best is trial 0 with value: 0.0.
❌ Trial 1 failed: Training failed: unsupported operand type(s) for //: 'NoneType' and 'int'

CHUNK_046
[WARNING] Could not process column Gender of the patient: y contains previously unseen labels: 'nan'

CHUNK_048
[WARNING] Could not process column Gender of the patient: y contains previously unseen labels: 'nan'

CHUNK_050
[WARNING] Could not process column Gender of the patient: y contains previously unseen labels: 'nan'

- `C:\Users\gcicc\claudeproj\tableGenCompare\SynthethicTableGenerator-Pakistani.ipynb`
All working as expected to end of section 4 and produced all output as expected. 
