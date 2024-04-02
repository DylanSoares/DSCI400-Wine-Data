# DSCI 400 Machine Learning Final Project

Dylan Soares

## WIP README

Project to create graphs for a presentation, train a model and generate predictions.
The documentation for this is WIP or provided as in. This project should of likely been done as a classification.
Time being a critical aspect of this assignment, improvements could of been made. Expect mistakes.

## Dataset of Choice

Wine Quality Dataset [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)<br>
Using only the white wines.

## Running

**Generate your own models:**

1. Run the `regr_white_model_selection.py` to generate models.
    1. Adjust hyperparameter grids to your preference
    2. Adjust the state if you wish

**Using the packaged presentation models:**

1. Extract `models.zip` from `/output/models/`

**Using the predictor UI:**

1. Open the `/output/plots/<DATE_TIME>_regr_model_errors.png`
   > <DATE_TIME> will match with the model you wish to use
2. Use the plot to decide which model(s) to use
   > Generally, lower metrics is better for all but R^2\
3. Run `white_wine_predictor.py`
4. Select the chosen model(s) .pkl file, found in `/output/models/`
5. Fill in the wine sample data
6. Click `"Predict"` to make the final predictions.
