from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib.patches import Rectangle


class QualityReporter:
    """This class is responsible for quality reporting a dataframe with samples.
    It has methods to print statistical outputs and save figures to disk.

    Examples
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [2, 4, 2, 3, 4, 3, 2, 3, 4, 5],
    ...                    "b": [2, 4, 2, 4, 2, 4, 2, 3, 2, 3],
    ...                    "c": list("asdfsdfsdf")})
    >>> variables = {"a": "Normal(0, 1)", "b": "Expon(1)", "c":"Discrete()"}
    >>> quality_reporter = QualityReporter(df, variables=variables)
    >>> quality_reporter.print_numeric()
    ================ CONTINUOUS PARAMETERS ================
       mean       std  min  10%  50%  90%  max
    a   3.2  1.032796  2.0  2.0  3.0  4.1  5.0
    b   2.8  0.918937  2.0  2.0  2.5  4.0  4.0
    >>> quality_reporter.print_discrete()
    ================ DISCRETE PARAMETERS ================
    | c   |   proportion |
    |:----|-------------:|
    | s   |          0.3 |
    | d   |          0.3 |
    | f   |          0.3 |
    | a   |          0.1 |
    """

    def __init__(self, df: pd.DataFrame, variables: Mapping[str, str]) -> None:
        """Initialize QualityReporter with dataframe and variable descriptions.

        Args:
            df: DataFrame containing the samples
            variables: Dictionary mapping variable names to their distribution descriptions
                      e.g., {"COST": "normal(0, 1)", ...}
        """
        self.df: pd.DataFrame = df.loc[:, list(variables.keys())]
        self.variables: dict[str, str] = dict(variables)
        assert not self.df.empty

    def print_numeric(self) -> None:
        """Print statistics for all numerical columns."""
        df_numeric = self.df.select_dtypes(include="number")
        if df_numeric.empty:
            return

        print("=" * 16, "CONTINUOUS PARAMETERS", "=" * 16)

        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", None
        ):
            print(
                df_numeric.describe(percentiles=[0.1, 0.5, 0.9]).T.drop(
                    columns=["count"]
                )
            )

    def print_discrete(self) -> None:
        """Print statistics for all discrete (non-numerical) columns."""
        df_non_numeric = self.df.select_dtypes(exclude="number")
        if df_non_numeric.empty:
            return

        print("=" * 16, "DISCRETE PARAMETERS", "=" * 16)

        for column in df_non_numeric.columns:
            print(self.df[column].value_counts(normalize=True).round(3).to_markdown())

    @staticmethod
    def _create_output_dir(output_dir: Path | None) -> Path | None:
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
        return None

    def plot_columns(self, output_dir: Path | None = None) -> None:
        """Loop through all columns and plot them, saving to disk if
        `output_dir` is given and exists.

        Args:
            output_dir: Optional directory path to save plots. If None, plots
                       are not saved to disk.
        """
        df_numeric = self.df.select_dtypes(include="number")
        df_non_numeric = self.df.select_dtypes(exclude="number")

        print("=" * 16, "GENERATING VARIABLE PLOTS", "=" * 16)

        # Create output directory if specified
        output_path = self._create_output_dir(output_dir)

        # Plot numeric columns
        for column in df_numeric.columns:
            fig, ax = self.plot_numeric(
                series=self.df[column],
                var_name=column,
                var_description=self.variables[column],
            )

            if output_path is not None:
                filename = output_path / f"{column}.png"
                fig.savefig(filename, dpi=200)
                print(f" - Saved variable: {filename}")

            plt.close(fig)

        # Plot discrete columns
        for column in df_non_numeric.columns:
            fig, ax = self.plot_discrete(
                series=self.df[column],
                var_name=column,
                var_description=self.variables[column],
            )

            if output_path is not None:
                filename = output_path / f"{column}.png"
                fig.savefig(filename, dpi=200)
                print(f" - Saved file: {filename}")

            plt.close(fig)

    @staticmethod
    def plot_numeric(
        series: pd.Series, var_name: str, var_description: str
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a plot for a single numeric column, returning (fig, ax).

        Args:
            series: Pandas series containing the numeric data
            var_name: Name of the variable
            var_description: Description of the variable distribution

        Returns:
            Tuple of (matplotlib Figure, matplotlib Axes)
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=series, kde=True, stat="density", bins="auto", ax=ax)
        ax.set_title(f"{var_name}\n{var_description}", fontsize=10)

        # Add rugplot
        ax.scatter(
            series.to_numpy(),
            np.zeros(len(series)),
            marker="|",
            color="black",
            alpha=0.5,
        )

        # Add average and quantiles to the plot
        ax.axvline(x=series.mean(), color="black", ls="-", alpha=0.8)
        for q_i in series.quantile(q=[0.1, 0.5, 0.9]).values:
            ax.axvline(x=q_i, color="black", ls="--", alpha=0.8)

        ax.grid(True, ls="--", alpha=0.5)
        fig.tight_layout()

        return fig, ax

    @staticmethod
    def plot_discrete(
        series: pd.Series, var_name: str, var_description: str
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a plot for a single discrete column, returning (fig, ax).

        Args:
            series: Pandas series containing the discrete data
            var_name: Name of the variable
            var_description: Description of the variable distribution

        Returns:
            Tuple of (matplotlib Figure, matplotlib Axes)
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        # Calculate normalized proportions
        value_counts = series.value_counts(normalize=False)

        # Create DataFrame for seaborn
        plot_data = pd.DataFrame(
            {var_name: value_counts.index, "proportion": value_counts.values}
        )

        # Use seaborn barplot with normalized values
        sns.barplot(data=plot_data, x=var_name, y="proportion", ax=ax)

        ax.set_title(f"{var_name}\n{var_description}", fontsize=10)
        ax.set_ylabel("Proportion")

        # Add percentage labels on bars
        for p in ax.patches:
            rect = cast(Rectangle, p)
            percentage = f"{100 * rect.get_height():.1f}%"
            ax.annotate(
                percentage,
                (rect.get_x() + rect.get_width() / 2.0, rect.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.grid(True, ls="--", alpha=0.5, axis="y")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()

        return fig, ax

    def print_correlation(self, corr_name: str, df_corr: pd.DataFrame) -> None:
        """Print information about desired and achieved correlation matrices.

        Args:
            corr_name: Name of the correlation group
            df_corr: DataFrame containing the desired correlation matrix
        """
        assert np.allclose(df_corr.values, df_corr.values.T)
        assert df_corr.shape[0] == df_corr.shape[1]

        # Import here to avoid circular imports
        from semeio.fmudesign.create_design import (  # noqa: PLC0415
            nearest_correlation_matrix,
        )

        # Get lower triangular indices
        idx_low_triang = np.tril_indices_from(df_corr.values, k=-1)
        corr = self.df[df_corr.columns].select_dtypes(include="number").corr()
        # Do not print if only a single variable remains
        if corr.shape[1] == 1:
            return

        print(
            "=" * 16,
            f"CORRELATION_GROUP {corr_name!r}: {list(df_corr.columns)}",
            "=" * 16,
        )

        print("Desired correlation:")
        print(df_corr.round(2))

        nearest_corr = nearest_correlation_matrix(
            df_corr.values, weights=None, eps=1e-6, verbose=False
        )
        diffs = nearest_corr[idx_low_triang] - df_corr.values[idx_low_triang]
        corr_rmse = np.sqrt(np.mean(diffs**2))
        if corr_rmse > 1e-2:
            print(f"The desired correlation matrix is not valid => {corr_rmse=:.2f}")
            print("Closest valid correlation matrix (used as target):")
            print(
                pd.DataFrame(
                    nearest_corr, columns=df_corr.columns, index=df_corr.index
                ).round(2)
            )
        else:
            print("The desired correlation matrix is valid")

        print("Correlation in samples:")
        print(corr.round(2))

    def plot_correlation(
        self,
        corr_name: str,
        df_corr: pd.DataFrame,
        output_dir: Path | None = None,
        show: bool = False,
    ) -> None:
        """Plot correlation group of variables.

        Args:
            corr_name: Name of the correlation group
            df_corr: DataFrame containing the correlation structure to plot
            output_dir: Optional directory path to save plots
            show: Whether or not to show the matplotlib figure
        """

        def corrfunc(
            x: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            **kwargs: Any,  # noqa: ANN401
        ) -> None:
            # Add correlations and grid to plots
            r, _ = sp.stats.pearsonr(x, y)
            ax = plt.gca()
            ax.annotate(
                r"$\rho=$" + f"{r:.2f}",
                xy=(0.05, 0.95),
                xycoords=ax.transAxes,
            )
            ax.grid(True, ls="--", alpha=0.5)

        def add_grid(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            ax = plt.gca()
            ax.grid(True, ls="--", alpha=0.5)

        df = self.df[df_corr.columns].select_dtypes(include="number")

        # Do not plot if only a single variable remains
        if df.shape[1] == 1:
            return

        pairgrid = sns.PairGrid(df)

        pairgrid.map_upper(sns.kdeplot)
        pairgrid.map_upper(add_grid)
        pairgrid.map_diag(sns.histplot, bins="auto", kde=True)

        pairgrid.map_lower(sns.scatterplot, s=10, alpha=0.6)
        pairgrid.map_lower(corrfunc)

        # Add rugplots to diagonal plots
        for i, var in enumerate(df.columns):
            pairgrid.diag_axes[i].scatter(  # type: ignore[index]
                df[var].to_numpy(),
                np.zeros(len(df)),
                marker="|",
                color="black",
                alpha=0.5,
            )

        output_path = self._create_output_dir(output_dir)
        if output_path is not None:
            filename = output_path / f"{corr_name}.png"
            pairgrid.savefig(filename, dpi=200)
            print(f" - Saved correlation: {filename}")

        if show:
            plt.show()

        plt.close(pairgrid.fig)


if __name__ == "__main__":
    # Testing an experimenting with this class is easier with an example,
    # rather than trying to formally test the design of output plots
    # using units tests and the like. Therefore an example is included.

    # Create sample data
    rng = np.random.default_rng(42)
    n_samples = 500

    # Generate correlated data
    mean = [0, 0]
    cov = [[1, 0.7], [0.7, 1]]
    correlated_data = rng.multivariate_normal(mean, cov, n_samples)

    df = pd.DataFrame(
        {
            "COST": correlated_data[:, 0] * 100 + 1000,
            "EFFICIENCY": correlated_data[:, 1] * 0.1 + 0.8,
            "MATERIAL": rng.choice(
                ["Steel", "Aluminum", "Titanium"], n_samples, p=[0.5, 0.3, 0.2]
            ),
        }
    )

    variables = {
        "COST": "Normal(1000, 100²)",
        "EFFICIENCY": "Normal(0.8, 0.1²)",
        "MATERIAL": "Discrete([Steel, Aluminum, Titanium])",
    }

    # Create QualityReporter
    quality_reporter = QualityReporter(df, variables)

    # 1. Plot all columns
    quality_reporter.plot_columns()

    # 2. Plot individual numeric variable
    fig, ax = quality_reporter.plot_numeric(df["COST"], "COST", variables["COST"])
    plt.show()
    plt.close(fig)

    # 3. Plot individual discrete variable
    fig, ax = quality_reporter.plot_discrete(
        df["MATERIAL"], "MATERIAL", variables["MATERIAL"]
    )
    plt.show()
    plt.close(fig)

    # 4. Correlation analysis and plotting
    correlation_matrix = pd.DataFrame(
        [[1.0, 0.7], [0.7, 1.0]],
        index=["COST", "EFFICIENCY"],
        columns=["COST", "EFFICIENCY"],
    )

    quality_reporter.print_correlation("corr1", correlation_matrix)
    quality_reporter.plot_correlation("corr1", correlation_matrix, show=True)
