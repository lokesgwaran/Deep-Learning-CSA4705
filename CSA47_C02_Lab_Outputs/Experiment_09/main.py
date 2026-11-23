prevalence,sensitivity,specificity=.01,.95,.90
p_positive=prevalence*sensitivity+(1-prevalence)*(1-specificity)
print('P(disease | positive):',prevalence*sensitivity/p_positive)
