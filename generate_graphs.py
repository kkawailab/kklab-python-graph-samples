#!/usr/bin/env python3
"""
generate_graphs.py

Create a library of sample charts and save the resulting PNG files under ``images/``.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # imported for 3D plots


# All generated images live in this directory so we can change it easily in one place.
IMAGE_DIR = "images"


def ensure_dir(path):
    """Create ``path`` if it does not already exist."""

    os.makedirs(path, exist_ok=True)


def line_plot():
    """Plot a sine wave to showcase a basic line plot."""

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.title("Line Plot")
    plt.savefig(f"{IMAGE_DIR}/line_plot.png")
    plt.close()


def scatter_plot():
    """Scatter random points to demonstrate color mapping."""

    np.random.seed(0)  # Keep the random numbers stable for reproducible images.
    x = np.random.rand(50)
    y = np.random.rand(50)

    plt.scatter(x, y, c=x, cmap="viridis", marker="o")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot")
    plt.colorbar()
    plt.savefig(f"{IMAGE_DIR}/scatter_plot.png")
    plt.close()


def bar_chart():
    """Display categorical data with a vertical bar chart."""

    categories = ["A", "B", "C", "D"]
    values = [10, 24, 36, 18]

    plt.bar(categories, values, color="skyblue")
    plt.title("Bar Chart")
    plt.savefig(f"{IMAGE_DIR}/bar_chart.png")
    plt.close()


def histogram():
    """Visualize a distribution using a histogram."""

    data = np.random.randn(1000)

    plt.hist(data, bins=30, color="gray", edgecolor="black")
    plt.title("Histogram")
    plt.savefig(f"{IMAGE_DIR}/histogram.png")
    plt.close()


def box_plot():
    """Compare spread and outliers for several datasets."""

    data = [np.random.randn(100) + i for i in range(4)]

    plt.boxplot(data)
    plt.title("Box Plot")
    plt.savefig(f"{IMAGE_DIR}/box_plot.png")
    plt.close()


def pie_chart():
    """Show proportions of a whole with a pie chart."""

    labels = ["Apple", "Banana", "Cherry", "Date"]
    sizes = [30, 15, 45, 10]

    plt.pie(sizes, labels=labels, autopct="%1.1f%%")
    plt.title("Pie Chart")
    plt.savefig(f"{IMAGE_DIR}/pie_chart.png")
    plt.close()


def image_display():
    """Render a tiny random image using a color map."""

    img = np.random.rand(10, 10)

    plt.imshow(img, cmap="viridis")
    plt.colorbar()
    plt.title("Image Display")
    plt.savefig(f"{IMAGE_DIR}/image_display.png")
    plt.close()


def filled_contour_plot():
    """Draw a filled contour plot from a Gaussian-like surface."""

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2))

    plt.contourf(X, Y, Z, levels=20, cmap="coolwarm")
    plt.colorbar()
    plt.title("Filled Contour Plot")
    plt.savefig(f"{IMAGE_DIR}/filled_contour_plot.png")
    plt.close()


def quiver_plot():
    """Illustrate vector fields using arrows (quiver plot)."""

    Y, X = np.mgrid[-3:3:100j, -3:3:100j]
    U = -1 - X**2 + Y
    V = 1 + X - Y**2

    plt.quiver(X, Y, U, V)
    plt.title("Quiver Plot")
    plt.savefig(f"{IMAGE_DIR}/quiver_plot.png")
    plt.close()


def polar_plot():
    """Plot data on polar axes for circular patterns."""

    theta = np.linspace(0, 2 * np.pi, 100)
    r = 1 + np.sin(4 * theta)

    ax = plt.subplot(projection="polar")
    ax.plot(theta, r)
    ax.set_title("Polar Plot")
    plt.savefig(f"{IMAGE_DIR}/polar_plot.png")
    plt.close()


def surface_plot():
    """Render a 3D surface whose height depends on distance from the origin."""

    X = np.linspace(-5, 5, 50)
    Y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(X, Y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_title("3D Surface Plot")
    plt.savefig(f"{IMAGE_DIR}/3d_surface_plot.png")
    plt.close()


def regression_plot():
    """Plot a scatterplot with a fitted regression line."""

    np.random.seed(1)
    x = np.linspace(0, 10, 50)
    noise = np.random.normal(scale=1.2, size=x.size)
    y = 1.8 * x + 2.5 + noise

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept

    plt.scatter(x, y, color="tab:blue", alpha=0.75, label="Data")
    plt.plot(x, y_fit, color="tab:orange", linewidth=2, label="Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Regression Plot")
    plt.legend()
    plt.savefig(f"{IMAGE_DIR}/regression_plot.png")
    plt.close()


def residuals_plot():
    """Show residuals against fitted values in a two-panel layout."""

    np.random.seed(2)
    x = np.linspace(0, 12, 60)
    noise = np.random.normal(scale=1.5, size=x.size)
    y = 2.2 * x - 1.0 + noise

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    residuals = y - y_fit

    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    axes[0].scatter(x, y, color="tab:blue", alpha=0.75, label="Data")
    axes[0].plot(x, y_fit, color="tab:orange", linewidth=2, label="Fit")
    axes[0].set_ylabel("y")
    axes[0].set_title("Residuals Plot")
    axes[0].legend()

    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].scatter(x, residuals, color="tab:green", alpha=0.75)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Residuals")

    plt.tight_layout()
    plt.savefig(f"{IMAGE_DIR}/residuals_plot.png")
    plt.close()


def correlation_heatmap():
    """Visualize a correlation matrix as a heatmap."""

    np.random.seed(3)
    data = np.random.randn(200, 4)
    data[:, 1] = 0.6 * data[:, 0] + 0.4 * data[:, 1]
    data[:, 2] = -0.5 * data[:, 0] + 0.3 * data[:, 2]

    corr = np.corrcoef(data, rowvar=False)

    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(4), ["Var1", "Var2", "Var3", "Var4"])
    plt.yticks(range(4), ["Var1", "Var2", "Var3", "Var4"])
    plt.title("Correlation Heatmap")
    plt.savefig(f"{IMAGE_DIR}/correlation_heatmap.png")
    plt.close()


def main():
    """Generate every sample figure in sequence."""

    ensure_dir(IMAGE_DIR)  # Make sure the output directory exists first.

    line_plot()
    scatter_plot()
    bar_chart()
    histogram()
    box_plot()
    pie_chart()
    image_display()
    filled_contour_plot()
    quiver_plot()
    polar_plot()
    surface_plot()
    regression_plot()
    residuals_plot()
    correlation_heatmap()


if __name__ == "__main__":
    main()
