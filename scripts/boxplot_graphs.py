from openpyxl import load_workbook
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
import numpy as np

# Save and plot quality settings
mpl.rcParams['savefig.dpi'] = 900  # Save figure quality
mpl.rcParams['figure.dpi'] = 150   # Improve quality in interactive Plot Pane

# Apply Palatino Linotype font to the entire plot
rcParams['font.family'] = 'Palatino Linotype'
plt.rcParams['font.size'] = 16  # Font size

# LOAD THE EXCEL FILE AND SPECIFIC SHEET
excel_path = r'C:\Users\maximilianovelezmont\OneDrive - usach.cl\000 Doctorado\00 Tesis\00 Publicacion Domain Generalization\RESULTADOS_14_07.xlsx'
wb = load_workbook(excel_path, data_only=True)
sheet = wb['mIoU best mIoU check']  # Make sure this is the correct sheet name

# READ CELL RANGES FOR EACH MODEL
mobilenet_dirt = [cell[0].value for cell in sheet['C4:C103']]
segformer_dirt = [cell[0].value for cell in sheet['G4:G103']]
twins_dirt = [cell[0].value for cell in sheet['K4:K103']]
resnet_dirt = [cell[0].value for cell in sheet['O4:O103']]

# BUILD DATAFRAME ONLY FOR DIRT CONDITION
data = []

for val in mobilenet_dirt:
    if val is not None:
        data.append({'Condition': 'Dirt', 'Model': 'Mobilenet', 'mIoU': val})
for val in segformer_dirt:
    if val is not None:
        data.append({'Condition': 'Dirt', 'Model': 'Segformer', 'mIoU': val})
for val in twins_dirt:
    if val is not None:
        data.append({'Condition': 'Dirt', 'Model': 'Twins', 'mIoU': val})
for val in resnet_dirt:
    if val is not None:
        data.append({'Condition': 'Dirt', 'Model': 'Resnet', 'mIoU': val})

df = pd.DataFrame(data)

# SEPARATE DATA BY MODEL
mobilenet_vals = df[df['Model'] == 'Mobilenet']['mIoU'].values
segformer_vals = df[df['Model'] == 'Segformer']['mIoU'].values
twins_vals = df[df['Model'] == 'Twins']['mIoU'].values
resnet_vals = df[df['Model'] == 'Resnet']['mIoU'].values

# CUSTOM POSITIONS FOR EACH MODEL
positions = [1, 2, 3, 4]

# CREATE BOXPLOT
fig, ax = plt.subplots(figsize=(10, 6))
box = ax.boxplot(
    [mobilenet_vals, segformer_vals, twins_vals, resnet_vals],
    positions=positions,
    widths=0.5,
    patch_artist=True
)

# BOX COLORS
colors = ['lightskyblue', 'peachpuff', 'orchid', 'yellowgreen']  # Mobilenet, Segformer, Twins, Resnet
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# MEDIANS IN THICK RED
for median in box['medians']:
    median.set_color('red')
    median.set_linewidth(2)

# X-AXIS LABELS
ax.set_xticks(positions)
ax.set_xticklabels(['Mobilenet', 'Segformer', 'Twins', 'Resnet'])

# STYLE AND LABELS
ax.set_ylabel('mIoU (%)')
# ax.set_title('mIoU Comparison under Dirt Condition')
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# DISPLAY IMPORTANT BOXPLOT STATISTICS
print("📊 Descriptive statistics per model:")
print("=" * 50)
for model in df['Model'].unique():
    values = df[df['Model'] == model]['mIoU'].dropna().values
    if len(values) > 0:
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        median = np.median(values)
        min_val = np.min(values)
        max_val = np.max(values)
        mean = np.mean(values)
        std = np.std(values)

        print(f"\n🔹 Model: {model}")
        print(f"   N = {len(values)}")
        print(f"   Min     : {min_val:.2f}")
        print(f"   Q1 (25%): {q1:.2f}")
        print(f"   Median  : {median:.2f}")
        print(f"   Q3 (75%): {q3:.2f}")
        print(f"   Max     : {max_val:.2f}")
        print(f"   Mean    : {mean:.2f}")
        print(f"   Std Dev : {std:.2f}")




