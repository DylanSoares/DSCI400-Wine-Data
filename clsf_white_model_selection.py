'''
This doesnt really work yet. It would be cool, but don't have the time.
'''

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

data = pd.read_csv('winequality-white.csv', sep=';')

X = data.drop(columns=['quality'])
y = data['quality']

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(random_state=42)
}

pca_params = {
    'pca__n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10]
}

model_params = {
    'Logistic Regression': {'model__C': [0.1, 1, 10], 'model__penalty': ['l1', 'l2'], 'model__solver': ['liblinear', 'saga']},
    'Decision Tree': {'model__max_depth': [None, 5, 10, 20], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4]},
    'Random Forest': {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 5, 10, 20], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4]},
    'Gradient Boosting': {'model__n_estimators': [50, 100, 200], 'model__learning_rate': [0.01, 0.1, 0.5], 'model__max_depth': [3, 5, 7], 'model__min_samples_split': [2, 5, 10]},
    'SVM': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf'], 'model__gamma': ['scale', 'auto']},
    'K-Nearest Neighbors': {'model__n_neighbors': [3, 5, 10], 'model__weights': ['uniform', 'distance']},
    'XGBoost': {'model__n_estimators': [50, 100, 200], 'model__learning_rate': [0.01, 0.1, 0.5], 'model__max_depth': [3, 5, 7]},
    'LightGBM': {'model__n_estimators': [50, 100, 200], 'model__learning_rate': [0.01, 0.1, 0.5], 'model__max_depth': [3, 5, 7]},
    'CatBoost': {'model__iterations': [50, 100, 200], 'model__learning_rate': [0.01, 0.1, 0.5], 'model__depth': [3, 5, 7]}
}

results_list = []


def main():
    train_models()
    plot_best_model_info()
    plot_pca_components()
    plot_models_performance()
    export_model()


def export_model():
    best_model = grid_search.best_estimator_
    best_model_name = results.loc[results['Accuracy'].idxmax(), 'Model']
    best_model_params = grid_search.best_params_

    # Export the best model
    joblib.dump(best_model, f"output/model_outputs/clsf_{best_model_name}_model.pkl")
    print(f"Best model '{best_model_name}' exported successfully.")

    with open(f"output/model_outputs/clsf_{best_model_name}_params.txt", "w") as f:
        f.write(str(best_model_params))
    print(f"Best model parameters saved to '{best_model_name}_params.txt'.")


def train_models():
    global grid_search, results
    total_models = len(models)
    total_pca_components = len(pca_params['pca__n_components'])
    total_combinations = total_models * total_pca_components

    # Outer progress bar for models
    model_progress_bar = tqdm(total=total_combinations, desc='Models')

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
            grid_search = GridSearchCV(pipeline, param_grid=model_params[model_name], cv=inner_cv,
                                       scoring='accuracy')
            grid_search.fit(X, y)

            # Extract best model
            best_model = grid_search.best_estimator_

            # Perform outer cross-validation to evaluate performance
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(best_model, X, y, cv=outer_cv, scoring="accuracy", n_jobs=-1)

            # Calculate confusion matrix
            y_pred = best_model.predict(X)
            cm = confusion_matrix(y, y_pred)

            # Calculate F1 score
            f1 = f1_score(y, y_pred, average='macro')

            # Append results to list
            results_list.append({
                'Model': model_name,
                'PCA Components': pca_n_components,
                'Accuracy': accuracy_score(y, y_pred),
                'Recall': recall_score(y, y_pred, average='macro'),
                'F1 Score': f1,
            })

            model_progress_bar.update(1)

    model_progress_bar.close()
    results = pd.DataFrame(results_list)


def plot_best_model_info():
    best_model_name = results.loc[results['Accuracy'].idxmax(), 'Model']
    best_model_params = grid_search.best_params_

    # Create a string with the best model info
    info_str = f"Best Model: {best_model_name}\n"
    info_str += f"Best Model Parameters: {best_model_params}"

    # Create a text plot
    plt.figure(figsize=(16, 9))
    plt.text(0.5, 0.5, info_str, ha='center', va='center', fontsize=12, color='white')
    plt.axis('off')

    # Save the plot as PNG
    plt.savefig("output/plots/clsf_best_model_info.png", bbox_inches='tight', transparent=True)


def plot_pca_components():
    heatmap_data = results.pivot(index='Model', columns='PCA Components', values='Accuracy')
    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, cmap="flare", fmt=".3f", cbar=False)
    plt.title('Accuracy across Models and PCA Components', color='white')
    plt.xlabel('PCA Components', color='white')
    plt.ylabel('Model', color='white')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.savefig(f"output/plots/clsf_accuracy_across_PCA.png", transparent=True, bbox_inches='tight')


def plot_models_performance():
    best_models_results = results.loc[results.groupby('Model')['Accuracy'].idxmax()]

    # Create heatmap data with selected classification metrics
    heatmap_data = pd.DataFrame({
        'Model': best_models_results['Model'],
        'Accuracy': best_models_results['Accuracy'],
        'Recall': best_models_results['Recall'],
        'F1 Score': best_models_results['F1 Score']
    })

    # Set the index to 'Model'
    heatmap_data.set_index('Model', inplace=True)

    # Create the heatmap
    plt.figure(figsize=(16, 9))
    sns.heatmap(heatmap_data, annot=True, cmap="flare", fmt=".3f", cbar=False)

    # Set the title
    plt.title('Classification Metrics across Models', color='white')

    # Set labels and ticks
    plt.xlabel('Metric', color='white')
    plt.ylabel('Model', color='white')
    plt.xticks(rotation=45, color='white')
    plt.yticks(rotation=0, color='white')

    # Save the plot
    plt.savefig(f"output/plots/clsf_model_metrics.png", transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    main()
