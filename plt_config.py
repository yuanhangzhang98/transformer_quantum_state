import matplotlib
import matplotlib.pyplot as plt

try:
    matplotlib.use("Qt5Agg")
except:
    pass
params = {
    # "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Arial",
    "figure.figsize": (8, 5),
    "figure.dpi": 80,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    'axes.titlesize': 24,
    "legend.fontsize": 20,
    "lines.linewidth": 2,
    'lines.markersize': 10,
    # "lines.markeredgewidth": 0.8,
    # "lines.markersize": 5,
    # "lines.marker": "o",
    # "patch.edgecolor": "black",
}
plt.rcParams.update(params)
plt.style.use("seaborn-deep")

# color = [[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.9290, 0.6940, 0.1250],\
#     [0.4940, 0.1840, 0.5560], [0.4660, 0.6740, 0.1880], [0.3010, 0.7450, 0.9330],\
#     [0.6350, 0.0780, 0.1840], [0, 0, 1], [0, 0.5, 0], [1, 0, 0], [0, 0.75, 0.75],\
#     [0.75, 0, 0.75], [0.75, 0.75, 0], [0.25, 0.25, 0.25]]

color = plt.rcParams['axes.prop_cycle'].by_key()['color']

print('Customized matplotlib configuration loaded.')
