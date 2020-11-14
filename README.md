# Euroleague Basketball Data Analysis and Prediction

This repository includes and an end-to-end methodology for building machine learning algorithms for predicting Euroleague Basketball game outcomes.

The methodology and results are discussed in detail in this article published on arxiv.org, entitled "[Descriptive and Predictive Analysis of Euroleague Basketball Games and the Wisdom of Basketball Crowds](https://arxiv.org/abs/2002.08465)"

The repository consists of the following modules (which represent the logical steps in the modelling process)

*   `data-collection`
*   `feature-extraction`
*   `descriptive-analysis`
*   `model-selection`
*   `feature-selection`
*   `model-validation`

Data extraction and storage settings are specified in the `settings/` directory

## Data Collection

Data is collected through scraping [Euroleague](https://www.euroleague.net/)'s official website.

In data collection, there are three scripts for collecting three types of data:

*   Team statistics per game, such as offence, defense scores, rebounds, steals, assists, rebounds, etc., for each team (row) in every game in a season.
*   Game results. Each row corresponds to a game in season. Teams and final scores are given.
*   Standing data. Each row corresponds to the standing of a team in the table at the end of the round. All rounds are included.

To collect the data for a season the user should run the script with the input console argument being the end year of the season, i.e. for season 2017-2018, execute

`$ python data-collection/scrap_game_stats.py -s 2018`

Similarly for the collection of the other data types.

Data is stored in the directory specified in the `settings/data_collection.json` file.

## Feature Extraction

Features are extracted from the data collected. Features are split in two main categories:

*   Match-level features. Every observation (row) corresponds to a match in a season. Features include average offence, average defense, form, etc., for each team in a game.
*   Team-level features. Every observation (row) corresponds to team in game in a season. Features include average offence, average defense, etc., for each team in a game.

To extract features, run the script with the input console argument being the end year of the season, i.e. for season 2017-2018, execute

`$ python feature-extraction/extract_features.py -s 2018`

Feature files are stored in the directory and with name patterns specified in the `settings/feature_extraction.json` file.

## Descriptive Analysis

After data collection and feature extraction, we perform Explanatory Data Analysis, a descriptive analysis of the datasets. Here, we focus on the distribution of score points for the home and away teams, winning and losing teams and the probability of winning as a function of the points scored. A jupyter notebook and a script are provided. Both do the same tasks.

## Model Selection

In model selection we compare a number of classification algorithms in different settings (e.g. match vs team level features) and perform hyper-parameter tuning.

The accuracy, weighted accuracy and ROC-AUC scores are recorded and later compared for identifying the best performing algorithm.

The `gridsearch_cross_validation.py` covers most cases. The other two scripts are left there for legacy (they shouldn't be used). Some setting parameters are hard-coded, so the user should edit those before running the script. More algorithms can be added and more hyper-parameters can be tuned if necessary.

The script has been split into sections so that it can be converted to a notebook using [`nbconvert`](https://nbconvert.readthedocs.io/en/latest/).

## Feature Selection

Different methods of feature selection are performed in this module.

*   Filter methods, including the mutual information, chi-squared and ANOVA F-statistic. Results are being evaluated for a chosen algorithm using k-fold cross-validation on the training test.

*   Feature transformation methods using Principal Component Analysis (PCA). Increasing numbers of principal components results are being evaluated for a chosen algorithm using k-fold cross-validation on the training test.

*   Wrapper method for feature selection, i.e. subsets of features are generated and evaluated using a chosen algorithm and its hyper-parameters. Here, as the number of features is relative small, we are able to generate all possible combinations of features. If the number of features grows large, a different approach should be adopted, the Sequential Forward Selection, see
`feature_selection_wrapper_sfs.py` script.

As before, Some setting parameters are hard-coded, so the user should edit those before running the script.

## Model Validation

After choosing the best performing algorithm, feature selection and tuning to its optimal hyper-parameters, we validate the final model(s) on the test set.

The validation of the model is performed in a jupyter notebook, see `validation.ipynb`. The notebook also includes assessment of the wisdom of the crowds model and comparison.

The directory also includes a script for assessing the performance of a few benchmark models. These are:
1. Home team always wins
2. F4 teams (i.e. teams that reached the F4 in the previous season) always win when playing with a non-F4 team, otherwise home team always wins.
3. Persistence model, teams that won in the previous round win, if both teams have won, home team wins.
4. Standing model, team higher in the standings wins.
5. Panathinaikos always wins, otherwise home team always wins
6. Random model.

To Run the benchmark models, execute

`$ python model-validation/benchmarks.py`
