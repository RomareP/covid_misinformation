# Spotting SARS-COV-2 Misinformation in Italy
Misinformation is not a new phenomenon but the popularity and ubiquity of social media speed up that the amount and velocity at which information is produced and spreads greatly outpaces our ability to evaluate whether it is correct and unbiased. This is especially important in healthcare, where misinformation can influence attitudes and health behaviours that can lead to harm.
## In this project, we make the following contributions:
1. We collect Italian scenario tweets on Covid-19 <br>
2. We quantify the ratio between high and low credibility information on Twitter <br>
3. We evaluate different machine learning classifiers to discriminate HCM and LCM using two features extraction modalities <br>

## Dataset
Dataset of 190.000 Italian tweets from March; <br>
Scraping articles linked in tweets, if presents; <br>
Lists of High Credibility Media (HCM) and Low Credibility Media (LCM); <br>
HCM articles = 2192 + LCM articles 447 = unbalanced dataset. <br>

## Features Extraction
Two ways:
1. Frequency–Inverse Document Frequency (TF-IDF)
2. Stylometry

## Models
1. Logistic Regression, 
2. K Nearest Neighbors, 
3. Decision Tree, 
4. Random Forest, 
5. Support Vector Machine.

## Evaluation
Recall/Sensitivity/TPR = LCM is classified as LCM. <br>
Specificity/TNR = HCM is classified as HCM. <br>
FPR = HCM is classified as LCM. <br>
