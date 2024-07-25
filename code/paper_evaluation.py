import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the objective log data
objective_log = np.load("objective_log.npy")

# Calculate the mean and the percentiles for the hull curve
mean_values = np.mean(objective_log, axis=1)
lower_boundary = np.min(objective_log, axis=1)
upper_boundary = np.max(objective_log, axis=1)

# Set the color
kit_deep_blue = "#144466"
see_through_blue = "#14446680"  # Adding 80 to make it semi-transparent

# Set seaborn theme to white
sns.set_theme(style="white")

# Plot the mean and the hull curve
plt.figure(figsize=(3, 2))

# Plot the mean
sns.lineplot(x=range(objective_log.shape[0]), y=mean_values, color=kit_deep_blue, label='Mean')

# Plot the hull curve
plt.fill_between(range(objective_log.shape[0]), lower_boundary, upper_boundary, color=see_through_blue, label='particle range')

plt.xlabel("Solver iteration")
plt.ylabel("Objective function")
plt.legend()

plt.show()
