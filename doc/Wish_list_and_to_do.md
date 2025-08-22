1. Once note book runs with first data set through end of section 4, we should run it for other data sets (4 notebooks runs in total).  Liver dataset will employ MICE
2.
3. This notebook is getting too long.  We need some solution - perhaps we can move python definitions outside of notebook as source these.
4. Alternative - look into Marimo: https://marimo.io/ - as alternative to juypter notebook
5. Review output of each cell junk... determine what can be trimmed from output. E.g., many messages can be deleted or simplified.
6. Review production of tables, graphics. Ensure these are returning what is expected and if not, work to resolve.
7. Determine if want to drop some algorithms, add other algorithms.
8. Review approach to hyperparameter tuning - note it is currently a combination of similarity (itself a combo of univariate and bivariate metrics) and utility (here a measure of accuracy) - this is what is employed to identified, say, the best CTGAN hyperparameter configuration
9. Review approach to identifying best of the best.  Whilst hyperparameter tuning is based on similarity and accuracy, we might apply different criteria in this stage (e.g., utility might include measures of accuracy, precision, recall, f2, etc.)
10. Determine if we can add functionality to notebook (marimo is a reactive notebook - so there is potential...) E.g., user checks off the models to contrast and code only executes for those models
11. Greg's been leveraging claude code on home computer... can we accomplish same in AWS
12. 
