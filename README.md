# normative-super-learner
A tool to build and interrogate a normative prediction model<sup>1</sup> for any continuous, real-valued variable. 

The analysis pipeline consists of:

Normative Modeling
------
1) Multiple machine learning algorithms\* (submodels) are tuned and trained on a normative sample displaying some baseline reference phenotype (e.g. "healthy", "no disease", "typically-developing")<sup>1</sup>. 
2) Prediction models are aggregated into a super learner<sup>2</sup>, a linear combination of submodel predictions optimizing the same loss function, implemented as a ridge regression model with no intercept. 
3) The super learner is tuned, trained on the full normative sample, and evaluated (Pearson correlation) using a normative evaluation set.
4) All trained submodels and the super learner are saved (using *JobLib*) and performance statistics are output locally.

\*Available models include random forest, gradient boosting machine (boosted trees), multilayer perceptron, support vector machine, and ridge regression linear model. All algorithms are implemented using *NumPy*, *SciPy*, and *SciKit-Learn*, with many more to come.

Calculate Atypical Deviations 
------
1) The normative super learner is used to make predictions in a 'atypical' sample displaying some phenotype-of-interest (eg. "symptomatic", "diaease present"). Multiple atypical samples (phenotypes) can be predicted simultaneously.
2) Predictions for the atypical sample/s are used to calculate phenotype-specific deviations from the norm (e.g. BrainAGE<sup>3</sup>)
3) Atypical sample predictions and deviations from the norm are output locally.

Feature Influence on Deviations
------
1) A univariate noise perturbation sensitivity (NPS) analysis is used to interogate the magnititue and direction of feature influence on atypical deviations from the normative model. NPS is a feature-wise metric that represents how sensitive group-level deviations in prediction are to the atypical phenotype.
2) Feature influence is thresholded using the paired-samples Wilcoxon test (perturbed deviation distribution vs. true deviation distribution) which is corrected for multiple comparisons using the Benjamini and Hochberg False-Discovery Rate (FDR)<sub>3</sub>. Features influencing non-significant differences in deviation distribution (based on the phenotype-of-interest OR by-chance) are considered non-influencial.
3) The feature influence score, direction of influence (increasing or decreasing deviations from the norm), and associated descriptive statistics (distribution medians, Wilcoxon statistic, effect size) are output locally.
