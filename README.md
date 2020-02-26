# normative-super-learner
A tool to build and interrogate a normative prediction model<sup>1</sup> for any continuous, real-valued variable. 

## Implementation:

**Dependencies**: Python3, Joblib, Scikit-Learn, Numpy, Pandas, SciPy, Statsmodels  

**Input**:  
```
python3 normative-super-learner.py <trainCSVFile> <evalCSVFile> <predCSVFile> <model> <kCV_paramSearch> <numIter_paramSearch> <numSims_MB> <numSims_NPS> <pThresh> <numCores> <prefix>
```  

- *\<trainCSVFile\>*: file path to the normative training set in CSV format  
- *\<evalCSVFile\>*: file path to the normative evaluation/validation set in CSV format  
- *\<predCSVFile\>*: file path to the atypical prediction/test set in CSV format  
- *\<model\>*: submodel/s algorithms to use in the Super Learner, currently available: "randForest", "mlp", "svm", "glm", "gradBoost", "all"  
- *\<kCV_paramSearch\>*: number of cross-validation folds to use when tuning hyperparameters in randomized search  
- *\<numIter_paramSearch\>*: number of test iterations to use when tuning hyperparameters in randomized search  
- *\<numSims_MB\>*: number of random initiations/simulations to use during model building  
- *\<numSims_NPS\>*: number of random initiations/simulations to use during noise perturbation sensitivity (NPS) analysis  
- *\<pThresh\>*: significance threshold for paired samples Wilcoxon tests and false discovery rate correction  
- *\<numCores\>*: number of cores to use for parallel computing  
- *\<prefix\>*: string for prefix to append to all output files  
   
Example:  
```
python3 normative-super-learner.py trainFile.csv evalFile.csv predFile.csv all 10 1000 100 1000 0.05 50 outputPrefix
```

Notes on CSV file format:  
- Rows should be one per instance/subject
- Column 1 (index 0) contains the instance/subject ID
- Column 2 contains the continuous, real-valued label to be predicted
- Columns 3...N contain the feature values, with the column header containing the feature label
- For the \<predCSVFile\>, Column 3 instead contains the atypical group labels, and Columns 4...N contain the feature values
  
**Output**:  
{prefix}\_errors\_{PRED_VAR_NAME}\_cv{kCV_paramSearch}\_simsNPS{numSims_NPS}.csv
- Normative model deviations for each instance in the normative evalution and atypical prediction/test sets
- Rows: individuals/instances
- Columns: subject ID (SUBJ), normative/atypical group label (GROUP), value to predict (LABEL), normative deviation (ERR)

{prefix}\_featInfNPS\_{PRED_VAR_NAME}\_cv{kCV_paramSearch}\_simsNPS{numSims_NPS}.csv
- Feature influence statistics for each feature/column in the input CSV files
- Rows: features
- Columns: feature label (FEAT), normative/atypical group label (GROUP), direction of feature influence (DIR; either "INCREASE" or "DECREASE"), median of perturbed deviation distribution (MEDIAN), differnce between median of perturbed deviation distribution and median of true deviation distribution (DIFF), Wilcoxon statistic (STAT), effect size of comparison (RSQUARED), raw probability value of comparison (NEG_P), raw probability value of by-chance perturbed deviation distribution compared to true deviation distribution (POS_P), FDR corrected NEG_P value (NEG_P_FDR), FDR corrected POS_P value (POS_P_FDR)

{prefix}\_modelPerformances\_{PRED_VAR_NAME}\_cv{kCV_paramSearch}\_simsNPS{numSims_NPS}.csv
- Super learner and submodel performance metrics
- Rows: super learner and all submodels
- Columns: algorithm name (MODEL), mean absolute error on the normative evaluation/validation set (EVAL_MAE), performance difference relative to by-chance performance (DIFF_FROM_CHANCE), correlation coefficient from Pearson testing prediction versus true label (PEARSON_R), probability value of the correlation coefficient (R_P)

{prefix}\_superLearner\_{PRED_VAR_NAME}\_cv{kCV_paramSearch}
- Saved Super Learner model output from Joblib: contains a dictionary with keys 'models' (trained submodels from Scikit-Learn) and 'coefficients' (coefficients associated with predictions from each submodel)
- Can be accessed with *joblib.load(superLearnerFile)*

## Analysis Summary:
The analysis pipeline consists of:

### Normative Modeling
------
1) Multiple machine learning algorithms\* (submodels) are tuned and trained on a normative sample displaying some baseline reference phenotype (e.g. "healthy", "no disease", "typically-developing")<sup>1</sup>. 
2) Prediction models are aggregated into a super learner<sup>2,3</sup>, a linear combination of submodel predictions optimizing the same loss function, implemented as a ridge regression model with no intercept. 
3) The super learner is tuned, trained on the full normative training sample, and evaluated using Pearson correlation on a normative evaluation set.
4) All optimized/trained submodels and the super learner are saved (using *JobLib*) and performance statistics are output.

\*Available models include random forest, gradient boosting machine (boosted trees), multilayer perceptron, support vector machine, and ridge regression linear model. All algorithms are implemented using *NumPy*, *SciPy*, and *SciKit-Learn*, with many more to come.

### Calculate Atypical Deviations 
------
1) The normative super learner is used to make predictions in an 'atypical' sample displaying some phenotype-of-interest (eg. "symptomatic", "disease present"). Multiple atypical samples (phenotypes) can be predicted simultaneously.
2) Predictions for the atypical sample/s are used to calculate phenotype-specific deviations from the normative prediction (e.g. BrainAGE<sup>4</sup>)
3) Atypical sample predictions and deviations are output.

### Feature Influence on Deviations
------
1) A univariate noise perturbation sensitivity (NPS)<sup>5</sup> analysis is used to interogate the magnititue and direction of feature influence on atypical deviations from the normative model. NPS is a feature-wise metric representing how sensitive group-level deviations from normative prediction are to the atypical phenotype.
2) Feature influence is thresholded using the paired-samples Wilcoxon test (perturbed deviation distribution vs. true deviation distribution) corrected for multiple comparisons using the Benjamini and Hochberg method. Features influencing non-significant differences in deviation distribution (based on the phenotype-of-interest OR by-chance) are considered non-influencial.
3) All feature influence scores, direction of influence (increasing or decreasing deviations from the norm), and associated descriptive statistics (distribution medians, Wilcoxon statistic, effect size) are output locally.

More detailed information to come! For specific questions, contact tjkeding@gmail.com.


## License
© Taylor J. Keding, 2020. Licensed under the General Public License v3.0 (GPLv3).
This program comes with ABSOLUTELY NO WARRANTY; This is free software, and you are welcome to redistribute it under certain conditions (contact tjkeding@gmail.com for more details).


## Acknowledgements & Contributions
Special thanks to Justin Russell Ph.D., Josh Cisler Ph.D., and Jerry Zhu Ph.D. (University of Wisconsin-Madison) for their input on implementation and best-practices. normative-super-learner is open for improvements and maintenance. Your help is valued to make the package better for everyone!


# References
**(1)** Marquand, A.F., Kia, S.M., Zabihi, M. et al. Conceptualizing mental disorders as deviations from normative functioning. *Molecular Psychiatry* 24, 1415–1424 (2019). DOI: 10.1038/s41380-019-0441-1

**(2)** van der Laan, M.J., Polley, E.C., & Hubbard, A.E. Super learner. *Statistical Applications in Genetics and Molecular Biology*. 6,25 (2007). DOI: 10.2202/1544-6115.1309

**(3)** Naimi, A.I. & Balzer, L.B. Stacked Generalization: An Introduction to Super Learning. *European Journal of Epidemiology*. 33, 459–464 (2018). DOI: 10.1007/s10654-018-0390-z

**(4)** Franke, K. & Gaser, C. Ten Years of BrainAGE as a Neuroimaging Biomarker of Brain Aging: What Insights Have We Gained? *Frontiers in Neurology*. 10, (2019). DOI: 10.3389/fneur.2019.00789

**(5)** Saltelli, A. Sensitivity analysis for importance assessment. *Risk Analysis*. 22, 3:579-90 (2002). DOI: 10.1111/0272-4332.00040
