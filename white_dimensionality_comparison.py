import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

data = pd.read_csv("winequality-white.csv", sep=";")
X = data.drop(columns=["quality"])
y = data["quality"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def main():
    variance_ratio()
    correlation_matrix()
    correlation_bar()
    rf_forest_correlation()


def variance_ratio():
    # Standardize the features
    # Perform PCA
    pca = PCA()
    pca.fit(X_scaled)
    # Calculate explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    # Plot explained variance ratio
    plt.figure(figsize=(16, 9))
    plt.bar(range(len(explained_variance)), explained_variance, align='center', color="palevioletred")
    plt.ylabel('Explained Variance Ratio', color='white')
    plt.xlabel('Principal Component Index', color='white')
    plt.title('Explained Variance Ratio by Principal Component', color='white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.savefig(f"output/variance_ratio.png", transparent=True, bbox_inches='tight')


def correlation_matrix():
    # Perform dimensionality reduction
    num_components = len(X.columns)
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X_scaled)
    # Calculate correlation matrix
    corr_matrix = data.corr()
    # Plot correlation matrix heatmap
    plt.figure(figsize=(16, 9))
    sns.heatmap(corr_matrix, annot=True, cmap='flare', fmt=".2f", linewidths=0.5, cbar=False, alpha=0.7)
    plt.title('Correlation Matrix Heatmap', color='white', fontsize=16, pad=20)
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.savefig(f"output/correlation_matrix.png", transparent=True, bbox_inches='tight')


def correlation_bar():
    # Calculate correlations
    correlations = X.corrwith(y)

    # Plot correlations
    plt.figure(figsize=(16, 9))
    correlations.plot(kind='bar', color='palevioletred')
    plt.title('Correlation of Features with Quality', color='white')
    plt.xlabel('Features', color='white')
    plt.ylabel('Correlation', color='white')
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.savefig(f"output/correlation_barchart.png", transparent=True, bbox_inches='tight')


def rf_forest_correlation():
    rf = RandomForestRegressor()
    rf.fit(X, y)
    importance = rf.feature_importances_

    # Sort feature importance's in descending order
    sorted_indices = np.argsort(importance)[::-1]
    sorted_importance = importance[sorted_indices]
    sorted_features = X.columns[sorted_indices]

    # Plot feature importance
    plt.figure(figsize=(16, 9))
    plt.barh(range(len(sorted_importance)), sorted_importance, align='center', color='palevioletred')
    plt.yticks(range(len(sorted_importance)), sorted_features, color='white')
    plt.xlabel('Feature Importance', color='white')
    plt.ylabel('Features', color='white')
    plt.title('Feature Importance for Quality Prediction using Random Forest', color='white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.savefig(f"output/feature_importance.png", transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    main()
