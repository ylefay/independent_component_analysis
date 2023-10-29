from mva_independent_component_analysis.fast_ica.preprocessing import demeaning, whitening
from mva_independent_component_analysis.fast_ica.fastica import fast_ica
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax.numpy as jnp
mpl.rcParams['figure.dpi'] = 1000

# Reading the data.
spy_data = pd.read_csv("./data/sp500/sp500_per_sectors.csv", index_col=0)
timestamp = spy_data.index
sectors = spy_data.columns
# Computing the log-returns.
returns = np.log((spy_data / spy_data.shift(1)).dropna())
# Doing some postprocessing to fit the format.
returns = returns.to_numpy().T

# Postprocessing the data for the fast_ica algorithm.
returns, m = demeaning(returns)
whitened_returns, L = whitening(returns)
# Running the fast_ica algorithm.
W = fast_ica(whitened_returns, whitened_returns.shape[0])
# Computing the independent components.
independent_components = m + jnp.linalg.inv(L) @ W.T @ whitened_returns
# Plotting the independent components.
fig, axs = plt.subplots(W.shape[0], 2)
for i in range(W.shape[0]):
    axs[i, 0].plot(independent_components[i], label=sectors[i], linewidth=0.5)
    axs[i, 1].plot(returns[i], label=sectors[i], linewidth=0.5)
plt.savefig("./sp500.png", dpi=1000)