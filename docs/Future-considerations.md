1.  Bring these notebooks into AWS environment. Increase number of trials in subsection of section 4.
   - Idea: Make n_trials a global variable that is defined at top of section 4 and used commonly for each method.
2.  Examples here do not offer a multi-level categorical endpoint. Such a dataset would allow us to verify one-hot encoding.
3.  Missingness... we are imputing values with MICE and using that imputed data to create a complete synthetic data.  Other approaches might add addition binary columns to note missingness
4. Need to review output of Section 4 with ntrials = 50 or 100 to really assess the value of hyperparmeter optimization and downstream impact in section 5

