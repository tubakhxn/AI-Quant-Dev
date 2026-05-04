# Urban Data Modeling with AI
# Fully standalone script

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde

# 1. Generate synthetic urban data
def generate_urban_data(n_samples=500, random_state=42):
    np.random.seed(random_state)
    # Distance from city center (km)
    distance = np.random.uniform(0, 20, n_samples)
    # Population density (people per sq km), higher near center, add noise
    density = 10000 * np.exp(-distance/7) + np.random.normal(0, 800, n_samples)
    density = np.clip(density, 500, None)
    # Housing price ($/sq m), depends on both distance and density, add noise
    price = 5000 * np.exp(-distance/10) + 0.2 * density + np.random.normal(0, 500, n_samples)
    price = np.clip(price, 1000, None)
    return distance, density, price

# 2. Fit regression model (polynomial)
def fit_polynomial_regression(x, y, degree=3):
    model = make_pipeline(PolynomialFeatures(degree), RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(x.reshape(-1, 1), y)
    return model

# 3. Plotting functions
def plot_scatter_with_regression(x, y, xlabel, ylabel, title, model=None, smooth=True):
    # This function now accepts an axis argument for subplotting
    import matplotlib.pyplot as plt
    def inner(ax):
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x_sorted, y_sorted, z_sorted = x[idx], y[idx], z[idx]
        ax.scatter(x_sorted, y_sorted, c=z_sorted, cmap='Blues', s=40, edgecolor='none', alpha=0.7, label='Data')
        if model is not None and smooth:
            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = model.predict(x_fit.reshape(-1, 1))
            ax.plot(x_fit, y_fit, color='red', lw=2.5, label='Regression')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    return inner

def plot_heatmap(x, y, xlabel, ylabel, title):
    def inner(ax):
        sns.kdeplot(x=x, y=y, cmap="Reds", fill=True, thresh=0.05, alpha=0.7, ax=ax)
        ax.scatter(x, y, s=15, color='blue', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    return inner

# Optional: 3D surface plot
def plot_3d_surface(distance, density, price):
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import griddata
    def inner(ax):
        xi = np.linspace(distance.min(), distance.max(), 40)
        yi = np.linspace(density.min(), density.max(), 40)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((distance, density), price, (xi, yi), method='cubic')
        surf = ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.7)
        ax.set_xlabel('Distance from Center (km)')
        ax.set_ylabel('Population Density')
        ax.set_zlabel('Housing Price')
        ax.set_title('3D Surface: Price vs Distance & Density')
    return inner

# Main execution
def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Generate data
    distance, density, price = generate_urban_data()
    # Fit regression models
    model_dist_price = fit_polynomial_regression(distance, price, degree=3)
    model_density_price = fit_polynomial_regression(density, price, degree=3)

    # Create subplots: 2D plots (3) + 3D plot (1)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    # 1. Scatter: Distance vs Price
    ax1 = fig.add_subplot(gs[0, 0])
    plot_scatter_with_regression(distance, price, 'Distance from Center (km)', 'Housing Price ($/sq m)', 'Distance vs Housing Price', model_dist_price)(ax1)
    # 2. Scatter: Density vs Price
    ax2 = fig.add_subplot(gs[0, 1])
    plot_scatter_with_regression(density, price, 'Population Density', 'Housing Price ($/sq m)', 'Density vs Housing Price', model_density_price)(ax2)
    # 3. Heatmap: Density vs Price
    ax3 = fig.add_subplot(gs[1, 0])
    plot_heatmap(density, price, 'Population Density', 'Housing Price ($/sq m)', 'Density vs Housing Price (Heatmap)')(ax3)
    # 4. 3D Surface plot
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    plot_3d_surface(distance, density, price)(ax4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
