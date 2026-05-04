import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix


class VisualizationManager:
    """Generate and save report-ready plots for the Parkinson's detection project."""

    def __init__(self, output_dir="outputs", style="seaborn-v0_8-darkgrid"):
        """Set up the output directory and matplotlib style."""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("seaborn-darkgrid")

    def _save(self, fig, filename):
        """Save a figure as a PNG file and close it."""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return path

    def plot_label_distribution(self, data, target_column="status", filename="label_distribution.png"):
        """Plot a bar chart showing the count of healthy vs Parkinson's samples."""
        counts = data[target_column].value_counts().sort_index()
        labels = ["Healthy (0)", "Parkinson's (1)"]
        colors = ["#4CAF50", "#F44336"]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)

        for bar, count in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title("Label Distribution", fontsize=14, fontweight="bold")
        ax.set_ylabel("Number of Samples")
        ax.set_ylim(0, max(counts.values) * 1.15)
        plt.show()
        return self._save(fig, filename)

    def plot_confusion_matrix(self, cm, filename="confusion_matrix.png"):
        """Plot the confusion matrix as a heatmap."""
        cm_array = np.array(cm)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Healthy", "Predicted Parkinson's"],
            yticklabels=["Actual Healthy", "Actual Parkinson's"],
            ax=ax,
        )
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_ylabel("Actual Label")
        ax.set_xlabel("Predicted Label")
        plt.show()
        return self._save(fig, filename)
    
    def plot_feature_correlation(self, data, filename="feature_correlation.png"):
        """Plot a heatmap of the correlation between all the features."""
        corr = data.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        plt.imshow(corr, cmap='viridis', aspect='auto', interpolation='nearest')
        plt.colorbar()
        ax.set_title("Correlation Heatmap", fontsize=16)
        ax.grid(False)

        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)

        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                value = corr.iloc[i, j]
                text_color = "white" if value <= 0.0 else "black"
                plt.text(j, i, f"{value:.2f}", ha='center', va='center', color=text_color, fontsize=8)

        plt.tight_layout()
        plt.show()
        return self._save(fig, filename)
    
    def plot_pairplot(self, data, cols_to_plot, filename ="pairplot.png"):
        """Pairplot of Key Variables for Parkinson’s Classification"""

        color_map = {0: "tab:blue", 1: "tab:red"}
        colors = data['status'].map(color_map).values


        fig = plt.figure(figsize=(12, 10))
        axs = scatter_matrix(
            data[cols_to_plot],
            figsize=(12, 10),
            diagonal="hist",
            c=colors,
            alpha=0.6,
            s=12,
            marker="o",
        )

        for ax in axs.ravel():
            ax.tick_params(axis="both", labelsize=7)


        handles = [
            Patch(color="tab:blue", label="Healthy (0)"),
            Patch(color="tab:red",  label="Parkinson’s (1)")
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=10)

        fig.suptitle(
            "Pairplot of Key Variables for Parkinson’s Classification",
            fontsize=22,
            y=0.97)
        plt.tight_layout()
        plt.show()
        return self._save(fig, filename)
    

    def plot_feature_importance(self, data, target_column="status", filename="feature_importance.png"):
        """Plot the absolute correlation of each feature with the target as a proxy for importance.
        """
        feature_cols = [col for col in data.columns if col not in ["name", target_column]]
        importance = (
            data[feature_cols]
            .corrwith(data[target_column])
            .abs()
            .sort_values(ascending=True)
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance)))
        ax.barh(importance.index, importance.values, color=colors, edgecolor="white")
        ax.set_title("Feature Importance (Correlation with Target)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Absolute Correlation")
        ax.set_xlim(0, importance.values.max() * 1.15)

        for i, value in enumerate(importance.values):
            ax.text(value + 0.005, i, f"{value:.3f}", va="center", fontsize=8)
        plt.show()
        return self._save(fig, filename)