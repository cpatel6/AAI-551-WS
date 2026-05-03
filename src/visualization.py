import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
        """Plot the confusion matrix as a heatmap.

        Parameters
        ----------
        cm : list of lists or 2-D array-like
            Confusion matrix returned by ``model.evaluate()``.
        """
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

    def plot_roc_curve(self, y_true, y_scores, filename="roc_curve.png"):
        """Plot the ROC curve and display the AUC score.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_scores : array-like
            Decision function scores or predicted probabilities for the positive class.
        """
        from sklearn.metrics import roc_auc_score, roc_curve

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Random classifier")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        return self._save(fig, filename)

    def plot_cross_validation_scores(self, cv_scores, filename="cross_validation_scores.png"):
        """Plot cross-validation fold scores together with mean and standard deviation.

        Parameters
        ----------
        cv_scores : array-like
            Scores returned by ``sklearn.model_selection.cross_val_score``.
        """
        cv_scores = np.array(cv_scores)
        folds = list(range(1, len(cv_scores) + 1))
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(folds, cv_scores, color="#42A5F5", edgecolor="white", label="Fold score")
        ax.axhline(mean_score, color="#E53935", linestyle="--", lw=2, label=f"Mean = {mean_score:.3f}")
        ax.fill_between(
            [0.5, len(cv_scores) + 0.5],
            mean_score - std_score,
            mean_score + std_score,
            alpha=0.15,
            color="#E53935",
            label=f"±1 std ({std_score:.3f})",
        )

        ax.set_xticks(folds)
        ax.set_xticklabels([f"Fold {i}" for i in folds])
        ax.set_ylim(max(0, cv_scores.min() - 0.05), min(1.0, cv_scores.max() + 0.05))
        ax.set_ylabel("Score")
        ax.set_title("Cross-Validation Scores", fontsize=14, fontweight="bold")
        ax.legend()
        return self._save(fig, filename)

    def plot_feature_distributions(self, data, target_column="status", top_n=6, filename="feature_distributions.png"):
        """Plot KDE distributions of the top features split by class label.

        Parameters
        ----------
        top_n : int
            Number of most-correlated features to visualise.
        """
        feature_cols = [col for col in data.columns if col not in ["name", target_column]]
        top_features = (
            data[feature_cols]
            .corrwith(data[target_column])
            .abs()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )

        ncols = 3
        nrows = (len(top_features) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
        axes = axes.flatten()

        healthy = data[data[target_column] == 0]
        parkinsons = data[data[target_column] == 1]

        for i, feature in enumerate(top_features):
            sns.kdeplot(healthy[feature], ax=axes[i], color="#4CAF50", fill=True, alpha=0.4, label="Healthy")
            sns.kdeplot(parkinsons[feature], ax=axes[i], color="#F44336", fill=True, alpha=0.4, label="Parkinson's")
            axes[i].set_title(feature, fontsize=10, fontweight="bold")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Density")
            axes[i].legend(fontsize=8)

        for j in range(len(top_features), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Feature Distributions by Class", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_model_comparison(self, results, filename="model_comparison.png"):
        """Plot a grouped bar chart comparing metrics across models or configurations.

        Parameters
        ----------
        results : dict
            Mapping of model name to a dict of metric name → value.
            Example::

                {
                    "SVM (RBF)":    {"accuracy": 0.95, "f1_score": 0.96},
                    "SVM (Linear)": {"accuracy": 0.93, "f1_score": 0.94},
                }
        """
        model_names = list(results.keys())
        metric_names = list(next(iter(results.values())).keys())

        x = np.arange(len(model_names))
        bar_width = 0.8 / len(metric_names)
        colors = plt.cm.tab10(np.linspace(0, 0.5, len(metric_names)))

        fig, ax = plt.subplots(figsize=(max(7, len(model_names) * 2), 5))

        for i, metric in enumerate(metric_names):
            values = [results[model][metric] for model in model_names]
            offset = (i - len(metric_names) / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, values, bar_width * 0.9, label=metric.capitalize(), color=colors[i])
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
        ax.legend()
        return self._save(fig, filename)
