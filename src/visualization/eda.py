import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno

from src.config import TARGET_COLUMN, BINARY_VARS, ORDINAL_VARS


sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})


def plot_missing_values(X: pd.DataFrame) -> None:
    """
    Plot missing value patterns using missingno.

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix.
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    # Graphique 1 : Bar chart — % de valeurs présentes par colonne
    msno.bar(X, ax=axes[0], color='#4C72B0', fontsize=9)
    axes[0].set_title('Complétude par colonne', fontweight='bold')

    # Graphique 2 : Matrix — visualisation ligne par ligne
    msno.matrix(X, ax=axes[1], fontsize=9, sparkline=False)
    axes[1].set_title('Matrice des valeurs manquantes', fontweight='bold')

    plt.suptitle('Analyse des valeurs manquantes', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()



def plot_target_distribution(train_df: pd.DataFrame) -> None:
    """
    Plot class distribution for the binary target in the training set.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    counts = train_df[TARGET_COLUMN].value_counts().sort_index()

    bars = ax.bar(
        ["Non diabétique\n(no + pré-diabète)", "Diabétique"],
        counts.values,
        color=["#FFB300", "#7C61F4"],
        edgecolor="white",
        width=0.5,
    )

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{value:,}\n({value / len(train_df) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Distribution de la variable cible (Train)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Nombre d'observations")
    ax.set_ylim(0, max(counts.values) * 1.15)

    plt.tight_layout()
    plt.show()


def plot_bmi_distribution_by_class(train_df: pd.DataFrame) -> None:
    """
    Plot BMI distribution for each target class.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    labels_class = {0: "Non diabétique", 1: "Diabétique"}
    colors = {0: "steelblue", 1: "coral"}

    for ax, target_class in zip(axes, [0, 1]):
        subset = train_df[train_df[TARGET_COLUMN] == target_class]["BMI"]

        ax.hist(subset, bins=45, color=colors[target_class], edgecolor="white", alpha=0.85)
        ax.axvline(subset.mean(), color="red", linestyle="--", linewidth=1.8,
                   label=f"Moyenne: {subset.mean():.1f}")
        ax.axvline(subset.median(), color="black", linestyle=":", linewidth=1.8,
                   label=f"Médiane: {subset.median():.1f}")

        ax.set_title(f"BMI – {labels_class[target_class]}", fontweight="bold")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Fréquence")
        ax.legend(fontsize=9)

    plt.suptitle("Distribution du BMI par classe", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_binary_features_vs_target(train_df: pd.DataFrame) -> None:
    """
    Plot diabetes rate by binary feature modality.
    """
    fig, axes = plt.subplots(3, 5, figsize=(19, 11))
    axes = axes.flatten()

    for index, column in enumerate(BINARY_VARS):
        ax = axes[index]

        cross_tab = pd.crosstab(
            train_df[column],
            train_df[TARGET_COLUMN],
            normalize="index",
        ) * 100

        for target_class in [0, 1]:
            if target_class not in cross_tab.columns:
                cross_tab[target_class] = 0

        cross_tab = cross_tab[[0, 1]]

        cross_tab.plot(
            kind="bar",
            ax=ax,
            color=["#FFB300", "#7C61F4"],
            edgecolor="white",
            linewidth=0.8,
            legend=(index == 0),
        )

        ax.set_title(column, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("% par modalité" if index % 5 == 0 else "")
        ax.set_xticklabels(["Non", "Oui"], rotation=0, fontsize=8)
        ax.set_ylim(0, 115)

        if index == 0:
            ax.legend(["Non diabétique", "Diabétique"], fontsize=7)

    for hidden_index in range(len(BINARY_VARS), len(axes)):
        axes[hidden_index].set_visible(False)

    plt.suptitle("Taux de diabète par variable binaire (Train)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_ordinal_boxplots(train_df: pd.DataFrame) -> None:
    """
    Plot boxplots of ordinal features by target class.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for index, column in enumerate(ORDINAL_VARS):
        ax = axes[index]

        data_0 = train_df[train_df[TARGET_COLUMN] == 0][column].astype(float)
        data_1 = train_df[train_df[TARGET_COLUMN] == 1][column].astype(float)

        boxplot = ax.boxplot(
            [data_0, data_1],
            labels=["Non diabétique", "Diabétique"],
            patch_artist=True,
        )

        boxplot["boxes"][0].set_alpha(0.7)
        boxplot["boxes"][1].set_alpha(0.7)

        for median in boxplot["medians"]:
            median.set_color("black")
            median.set_linewidth(2)

        ax.set_title(column, fontsize=10, fontweight="bold")
        ax.set_ylabel(column)

    plt.suptitle("Distribution des variables ordinales par classe (Train)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_age_diabetes_rate(train_df: pd.DataFrame) -> None:
    """
    Plot diabetes rate by age group.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    age_ct = pd.crosstab(
        train_df["Age"].astype(int),
        train_df[TARGET_COLUMN],
        normalize="index",
    ) * 100

    if 1 not in age_ct.columns:
        age_ct[1] = 0

    ax.plot(age_ct.index, age_ct[1], marker="o", linewidth=2.5, markersize=7, color="#002486")
    ax.fill_between(age_ct.index, age_ct[1], alpha=0.15)

    ax.set_xticks(age_ct.index)

    age_labels = [
        "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
        "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"
    ]
    ax.set_xticklabels(age_labels, rotation=45, ha="right", fontsize=9)

    ax.set_title("Taux de diabète par tranche d'âge (Train)", fontsize=12, fontweight="bold")
    ax.set_ylabel("% diabétiques")
    ax.set_xlabel("Tranche d'âge")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_target_correlations(train_df: pd.DataFrame) -> pd.Series:
    """
    Compute and plot Pearson correlations between features and target.

    Returns
    -------
    pd.Series
        Sorted correlations with the target.
    """
    correlations = (
        train_df.astype(float)
        .corr()[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#123953" if value > 0 else "#FFBF00" for value in correlations.values]

    ax.barh(correlations.index, correlations.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Corrélation des features avec Diabetes_binary (Train)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Coefficient de Pearson")

    plt.tight_layout()
    plt.show()

    return correlations


def plot_correlation_matrix(train_df: pd.DataFrame) -> None:
    """
    Plot the full correlation matrix of the training set.
    """
    fig, ax = plt.subplots(figsize=(14, 11))

    corr_matrix = train_df.astype(float).corr().round(2)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6.5},
        linewidths=0.3,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title("Matrice de corrélation – Train set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()