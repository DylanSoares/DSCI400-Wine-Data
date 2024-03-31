import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

data = pd.read_csv('winequality-white.csv', sep=';')

X = data.drop(columns=['quality'])
y = data['quality']

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet Regression': ElasticNet(),
    'Support Vector Machine': SVR(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42),
    'CatBoost': CatBoostRegressor(random_state=42)
    # 'Neural Network': MLPRegressor(max_iter=500, early_stopping=True, random_state=42) ITS NOT WORTH IT
}

pca_params = {
    'pca__n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10]
}

model_params = {
    'Linear Regression': {},
    'Ridge Regression': {'model__alpha': [0.1, 1, 10]},
    'Lasso Regression': {'model__alpha': [0.1, 1, 10]},
    'ElasticNet Regression': {'model__alpha': [0.1, 1, 10], 'model__l1_ratio': [0.1, 0.5, 0.9]},
    'Support Vector Machine': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']},
    'Decision Tree': {'model__max_depth': [None, 5, 10, 20, 30]},
    'Random Forest': {'model__n_estimators': [200, 300, 500]},
    'Gradient Boosting': {'model__n_estimators': [200, 300, 500], 'model__learning_rate': [0.01, 0.1, 0.5]},
    'K-Nearest Neighbors': {'model__n_neighbors': [3, 5, 10]},
    'XGBoost': {'model__n_estimators': [100, 200, 300, 500], 'model__learning_rate': [0.01, 0.1, 0.5]},
    'LightGBM': {'model__n_estimators': [100, 200, 300, 500], 'model__learning_rate': [0.01, 0.1, 0.5]},
    'CatBoost': {'model__iterations': [100, 200, 300, 500], 'model__learning_rate': [0.01, 0.1, 0.5]},
    # 'Neural Network': {'model__hidden_layer_sizes': [(50,), (100,), (50, 50)], 'model__alpha': [0.0001, 0.001, 0.01],
    #                    'model__learning_rate_init': [0.0001, 0.001, 0.01], 'model__solver': ['lbfgs', 'adam']}
}

scoring = {
    'MSE': 'neg_mean_squared_error',
    'MAE': 'neg_mean_absolute_error'
}

results_list = []


def main():
    train_models()
    plot_best_model_info()
    plot_mse_pca()
    plot_models_error()
    export_model()


def export_model():
    best_model = grid_search.best_estimator_
    best_model_name = results.loc[results['MSE'].idxmin(), 'Model']
    best_model_params = grid_search.best_params_

    # Export the best model
    joblib.dump(best_model, f"output/model_outputs/regr_{best_model_name}_model.pkl")
    print(f"Best model 'regr_{best_model_name}' exported successfully.")

    with open(f"output/model_outputs/regr_{best_model_name}_params.txt", "w") as f:
        f.write(str(best_model_params))
    print(f"Best model parameters saved to 'regr_{best_model_name}_params.txt'.")


def train_models():
    global grid_search, results
    total_models = len(models)
    total_pca_components = len(pca_params['pca__n_components'])
    total_combinations = total_models * total_pca_components

    # Outer progress bar for models
    model_progress_bar = tqdm(total=total_combinations, desc='Models', )

    for model_name, model in models.items():
        for pca_n_components in pca_params['pca__n_components']:
            # Define pipeline with PCA and model
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=pca_n_components)),
                ('model', model)
            ])

            # Define inner cross-validation for hyperparameter tuning
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(pipeline, param_grid=model_params[model_name], cv=inner_cv, scoring=scoring,
                                       refit='MSE')
            grid_search.fit(X, y)

            # Extract best model
            best_model = grid_search.best_estimator_

            # Perform outer cross-validation to evaluate performance
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores_mse = cross_val_score(best_model, X, y, cv=outer_cv, scoring='neg_mean_squared_error')
            scores_mae = cross_val_score(best_model, X, y, cv=outer_cv, scoring='neg_mean_absolute_error')

            # Calculate additional error metrics
            y_pred = best_model.predict(X)
            mse_mean = -np.mean(scores_mse)
            mae_mean = -np.mean(scores_mae)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            msle = mean_squared_log_error(y, y_pred)
            mape = mean_absolute_percentage_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Append results to list
            results_list.append({
                'Model': model_name,
                'PCA Components': pca_n_components,
                'MSE': mse_mean,
                'MAE': mae_mean,
                'RMSE': rmse,
                'MSLE': msle,
                'MAPE': mape,
                'R^2': r2
            })

            model_progress_bar.update(1)

    model_progress_bar.close()
    results = pd.DataFrame(results_list)


def plot_best_model_info():
    best_model_name = results.loc[results['MSE'].idxmin(), 'Model']
    best_model_params = grid_search.best_params_

    # Create a string with the best model info
    info_str = f"Best Model: {best_model_name}\n"
    info_str += f"Best Model Parameters: {best_model_params}"

    # Create a text plot
    plt.figure(figsize=(16, 9))
    plt.text(0.5, 0.5, info_str, ha='center', va='center', fontsize=12, color='white')
    plt.axis('off')

    # Save the plot as PNG
    plt.savefig("output/regr_best_model_info.png", bbox_inches='tight', transparent=True)


def plot_mse_pca():
    # Apply scaling to quality values
    scaled_quality = StandardScaler().fit_transform(np.array(y).reshape(-1, 1))
    scaled_min_quality = scaled_quality.min()
    scaled_max_quality = scaled_quality.max()

    heatmap_data = results.pivot(index='Model', columns='PCA Components', values='MSE')
    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, cmap="flare", fmt=".3f", cbar=False)
    plt.title('Mean Squared Error across Models and PCA Components\nScaled Quality Range: {:.2f} - {:.2f}'.format(
        scaled_min_quality, scaled_max_quality), color='white')
    plt.xlabel('PCA Components', color='white')
    plt.ylabel('Model', color='white')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.savefig(f"output/plots/regr_mse_across_PCA.png", transparent=True, bbox_inches='tight')


def plot_models_error():
    # Apply scaling to quality values
    scaled_quality = StandardScaler().fit_transform(np.array(y).reshape(-1, 1))
    scaled_min_quality = scaled_quality.min()
    scaled_max_quality = scaled_quality.max()

    best_models_results = results.loc[results.groupby('Model')['MSE'].idxmin()]
    heatmap_data = pd.DataFrame({
        'Model': best_models_results['Model'],
        'MAE': best_models_results['MAE'],
        'MSE': best_models_results['MSE'],
        'RMSE': best_models_results['RMSE'],
        'MSLE': best_models_results['MSLE'],
        'MAPE': best_models_results['MAPE'],
        'R^2': best_models_results['R^2']
    })

    # Create heatmap
    heatmap_data.set_index('Model', inplace=True)
    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, cmap="flare", fmt=".3f", cbar=False)

    plt.title('Error Metrics across Models\nScaled Quality Range: {:.2f} - {:.2f}'.format(
        scaled_min_quality, scaled_max_quality), color='white')

    plt.xlabel('Error Type', color='white')
    plt.ylabel('Model', color='white')
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], labels=['MAE', 'MSE', 'RMSE', 'MSLE', 'MAPE', 'R^2'],
               color='white')
    plt.yticks(rotation=0, color='white')
    plt.savefig(f"output/plots/regr_model_errors.png", transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    main()
