import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
file = "Twins_photo_0.01_train_curve.xlsx"
df = pd.read_excel(file)

# Create figure
fig, ax1 = plt.subplots(figsize=(12, 6), dpi=1000)

# Set general font
plt.rcParams['font.family'] = 'Palatino Linotype'

# Plot Loss
ax1.plot(df['iter'], df['loss'], 'dodgerblue', label='Loss', linewidth=2)
ax1.set_xlabel('Iterations', fontsize=30)
ax1.set_ylabel('Loss', color='black', fontsize=30)
ax1.tick_params(axis='both', labelsize=30)
ax1.tick_params(axis='y', labelcolor='black')

# Create a second axis for mIoU
ax2 = ax1.twinx()
ax2.plot(df['iter'], df['mIoU'], 'darkorange', label='mIoU', linewidth=2)
ax2.set_ylabel('mIoU (%)', color='black', fontsize=30)
ax2.tick_params(axis='y', labelsize=30, labelcolor='black')

# Title 
# plt.title('Training Curve: Loss and mIoU vs Iteration', fontsize=18)

# Legend inside the plot (on the right, without overlapping the axis)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines + lines2, labels + labels2,
           loc='center right', bbox_to_anchor=(0.88, 0.5),
           fontsize=30, frameon=True)

# Adjust layout
plt.tight_layout()
plt.show()


