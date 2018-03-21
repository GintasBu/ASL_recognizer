# ASL_recognizer
Sign language recognizer. HMM models in combination with NLP n-gram model

Trained HMM models using three different methods to evaluate model results: 
log likelihood using cross-validation folds (CV), Bayesian Information Criterion (BIC), Discriminative Information Criterion (DIC).
WER rate was varying between 48 to 63 depending on the set of features and the model selection method.
In the optional Part 4, designed and implemented a hybrid HMM/NLP n-gram model. The final WER was 21% for a test set.
