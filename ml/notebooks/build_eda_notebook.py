from pathlib import Path

import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parent / "01_eda.ipynb"


def _markdown(source: str):
    return nbf.v4.new_markdown_cell(source)


def _code(source: str):
    return nbf.v4.new_code_cell(source)


def build_notebook() -> "nbf.NotebookNode":
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
    }
    notebook.cells = [
        _markdown(
            "# Coal GCV — Exploratory Data Analysis\n\n"
            "Distribution, correlation, and relationship analysis of the four "
            "proximate-analysis features against GCV."
        ),
        _code(
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import seaborn as sns\n\n"
            "sys.path.insert(0, str(Path.cwd().parent))\n"
            "from src.config import CLEAN_DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN\n\n"
            "df = pd.read_csv(CLEAN_DATA_PATH)\n"
            "df.describe()"
        ),
        _markdown("## Feature distributions"),
        _code(
            "fig, axes = plt.subplots(2, 2, figsize=(10, 8))\n"
            "for ax, column in zip(axes.flatten(), FEATURE_COLUMNS):\n"
            "    sns.histplot(df[column], kde=True, ax=ax)\n"
            "    ax.set_title(column)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _markdown("## Correlation heatmap"),
        _code(
            "correlation = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()\n"
            "plt.figure(figsize=(6, 5))\n"
            "sns.heatmap(correlation, annot=True, cmap=\"coolwarm\", vmin=-1, vmax=1)\n"
            "plt.title(\"Correlation between proximate features and GCV\")\n"
            "plt.show()"
        ),
        _markdown("## GCV vs. each feature"),
        _code(
            "fig, axes = plt.subplots(2, 2, figsize=(10, 8))\n"
            "for ax, column in zip(axes.flatten(), FEATURE_COLUMNS):\n"
            "    ax.scatter(df[column], df[TARGET_COLUMN], alpha=0.3, s=10)\n"
            "    ax.set_xlabel(column)\n"
            "    ax.set_ylabel(TARGET_COLUMN)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _markdown(
            "## Observations\n\n"
            "- Fixed Carbon typically shows the strongest positive correlation with "
            "GCV; Moisture and Ash typically show negative correlation, consistent "
            "with combustion chemistry.\n"
            "- The four features sum to 100% by construction (proximate analysis "
            "convention).\n"
            "- Findings here inform the Analytics dashboard page."
        ),
    ]
    return notebook


def main() -> None:
    notebook = build_notebook()
    nbf.write(notebook, str(NOTEBOOK_PATH))
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
