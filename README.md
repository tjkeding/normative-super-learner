# normative-super-learner
A tool to build and interrogate a normative prediction model<sup>1</sup> for any continuous, real-valued variable. 

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
1) A univariate noise perturbation sensitivity (NPS) analysis is used to interogate the magnititue and direction of feature influence on atypical deviations from the normative model. NPS is a feature-wise metric representing how sensitive group-level deviations from normative prediction are to the atypical phenotype.
2) Feature influence is thresholded using the paired-samples Wilcoxon test (perturbed deviation distribution vs. true deviation distribution) corrected for multiple comparisons using the Benjamini and Hochberg method<sup>3</sup>. Features influencing non-significant differences in deviation distribution (based on the phenotype-of-interest OR by-chance) are considered non-influencial.
3) All feature influence scores, direction of influence (increasing or decreasing deviations from the norm), and associated descriptive statistics (distribution medians, Wilcoxon statistic, effect size) are output locally.

More detailed information to come! For specific questions, contact tjkeding@gmail.com.


## License
© Taylor J. Keding, 2020. Licensed under the General Public License v3.0 (GPLv3).
This program comes with ABSOLUTELY NO WARRANTY; This is free software, and you are welcome to redistribute it under certain conditions (contact tjkeding@gmail.com for more details).

## Acknowledgements & Contributions
Special thanks to Justin Russell Ph.D., Josh Cisler Ph.D., and Jerry Zhu Ph.D. (University of Wisconsin-Madison) for their input on implementation and best-practices. normative-super-learner is open for improvements and maintenance. Your help is valued to make the package better for everyone!

# References
------
(1) Marquand, A.F., Kia, S.M., Zabihi, M. et al. Conceptualizing mental disorders as deviations from normative functioning. *Molecular Psychiatry* 24, 1415–1424 (2019). DOI: 10.1038/s41380-019-0441-1

(2) van der Laan, M.J., Polley, E.C., & Hubbard, A.E. Super learner. *Statistical Applications in Genetics and Molecular Biology*. 6,25 (2007). DOI: 10.2202/1544-6115.1309

(3) Naimi, A.I. & Balzer, L.B. Stacked Generalization: An Introduction to Super Learning. *European Journal of Epidemiology*. 33, 459–464 (2018). DOI: 10.1007/s10654-018-0390-z

(4) Franke, K. & Gaser, C. Ten Years of BrainAGE as a Neuroimaging Biomarker of Brain Aging: What Insights Have We Gained? *Frontiers in Neurology* 10, (2019). DOI: 10.3389/fneur.2019.00789
