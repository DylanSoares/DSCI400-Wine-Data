import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.core.defchararray import capitalize


def main():
    # Read white wine data
    white_wine_data = pd.read_csv('winequality-white.csv', sep=';')

    # Numerical variables:
    num_cols_char = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                     "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

    num_cols_char1 = ["free sulfur dioxide", "total sulfur dioxide", "density", "sulphates", ]
    num_cols_char2 = ["citric acid", "pH", "fixed acidity", "volatile acidity"]
    num_cols_char3 = ["residual sugar", "chlorides", "alcohol"]

    # Output graphs
    wine_quality_graph(white_wine_data, "quality")
    compare_characteristics(white_wine_data, num_cols_char1, "Chars_Comp_First.png")
    compare_characteristics(white_wine_data, num_cols_char2, "Chars_Comp_Second.png")
    compare_characteristics(white_wine_data, num_cols_char3, "Chars_Comp_Third.png")
    compare_individual_characteristics(white_wine_data, num_cols_char)

    print("All visualizations complete")


# visualizing categorical wine characteristics and their relationship to wine quality
def wine_quality_graph(entries, field):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    sns.countplot(x=field, hue="quality", data=entries, ax=ax, palette="flare")
    ax.set_title(f'White Wine by {field}', fontsize=14, color='white')
    ax.set_xlabel(capitalize(field), color='white')
    ax.set_ylabel("Number of Wines", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f"output/plots/Wine_Quality.png", transparent=True, bbox_inches='tight')


# visualizing numerical wine characteristics and their relationship to wine quality
def compare_characteristics(entries, fields, filename):
    n_rows = len(fields)
    n_cols = 2

    # Calculate the number of plots needed
    n_plots = n_rows // n_cols + (1 if n_rows % n_cols > 0 else 0)

    fig, axes = plt.subplots(n_plots, n_cols, figsize=(16, 9), dpi=300)
    fig.suptitle("White Wine Quality by Characteristics\n", fontsize=16, fontweight='bold', color='white')

    for i, col in enumerate(fields):
        ax = axes[i // n_cols, i % n_cols]
        sns.histplot(data=entries, x=col, hue="quality", palette="Set1", kde=True, ax=ax, element="step")

        ax.set_xlabel(capitalize(col), color='white')
        ax.set_ylabel("Number of Wines", fontsize="medium", color='white')
        ax.set_title(f'Wine Quality by {capitalize(col)}', fontsize=14, color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # If the number of plots * n_cols is greater than the number of rows, remove the last subplot
    if n_plots * n_cols > n_rows:
        fig.delaxes(axes.flatten()[n_rows])
    plt.savefig(f"output/plots/{filename}", transparent=True, bbox_inches='tight')


def compare_individual_characteristics(entries, fields):
    for col in fields:
        fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
        fig.suptitle("Wine Quality by Characteristics\n", fontsize=16, fontweight='bold', color='white')
        sns.histplot(data=entries, x=col, hue="quality", palette="Set1", kde=True, ax=ax, element="step")
        ax.set_xlabel(capitalize(col), color='white')
        ax.set_ylabel("Number of Wines", fontsize="medium", color='white')
        ax.set_title(f'White Wine Quality by {capitalize(col)}', fontsize=14, color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(f"output/plots/{col}.png", transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    main()
