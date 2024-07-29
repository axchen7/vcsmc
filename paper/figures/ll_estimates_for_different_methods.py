import matplotlib.pyplot as plt

# Data
datasets = ["DS1", "DS2", "DS3", "DS4", "DS5", "DS6", "DS7"]
vanilla_csmc = [
    -8306.76,
    -27884.37,
    -35381.01,
    -15019.21,
    -8940.62,
    -8029.51,
    -11013.57,
]
vaiphy = [-7490.54, -31203.44, -33911.13, -13700.86, -8464.77, -7157.84, -9462.21]
phi_csmc = [-7290.36, -30568.49, -33798.06, -13582.24, -8367.51, -7013.83, -9209.18]
hyp_smc = [-7688.9, -27029.39, -34272.38, -14141.12, -8293.67, -7080.0, -9469.56]

# Standard deviations (actual values from the image)
vanilla_csmc_std = [166.27, 226.60, 218.18, 100.61, 46.44, 83.67, 113.49]
vaiphy_std = [0, 3e-12, 7e-12, 0, 0, 0, 1e-12]
phi_csmc_std = [7.23, 31.34, 6.62, 35.08, 8.87, 16.99, 18.03]
hyp_smc_std = [36.3, 72.52, 46.68, 23.26, 17.88, 65.26, 36.3]

# Create subplots
fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # 2x4 grid

# Flatten the array of axes for easy iteration
axs = axs.flatten()  # type: ignore

# Plot data
for i, dataset in enumerate(datasets):
    means = [vanilla_csmc[i], vaiphy[i], phi_csmc[i], hyp_smc[i]]
    std_devs = [vanilla_csmc_std[i], vaiphy_std[i], phi_csmc_std[i], hyp_smc_std[i]]
    labels = ["Vanilla CSMC", "VaiPhy", "φ-CSMC", "Hyp SMC"]

    for j, mean in enumerate(means):
        axs[i].errorbar(j + 1, mean, yerr=std_devs[j], fmt="o", capsize=5)

    axs[i].set_title(dataset)
    axs[i].set_xticks(range(1, 5))
    axs[i].set_xticklabels(labels)
    axs[i].set_ylabel("Log Likelihood Estimate")

    # Calculate Y range
    all_data = [mean - std for mean, std in zip(means, std_devs)] + [
        mean + std for mean, std in zip(means, std_devs)
    ]
    y_min = min(all_data) - 100
    y_max = max(all_data) + 100
    axs[i].set_ylim([y_min, y_max])

# Remove any empty subplots (in case we have more subplots than datasets)
for j in range(len(datasets), len(axs)):
    fig.delaxes(axs[j])

# Add overall title
fig.suptitle("Log Likelihood Estimates for Different Methods", fontsize=16)

# Adjust layout with more padding between columns and no extra space around the outside edge
plt.subplots_adjust(wspace=0.4, hspace=0.2, left=0.05, right=0.95, top=0.9, bottom=0.05)

# Save the figure as an svg
plt.savefig("ll_estimates_for_different_methods.svg", format="svg")

# Show the plot
plt.show()
