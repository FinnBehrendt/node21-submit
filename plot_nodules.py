import json
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# --- Config ---
image_path = "/path/to/file.mha"
json_path = "path/to/outputs.json"

# --- Load image ---
img = sitk.ReadImage(image_path)
img_array = sitk.GetArrayFromImage(img)[0]  # shape: [1, H, W] -> [H, W]

# --- Load JSON ---
with open(json_path, 'r') as f:
    data = json.load(f)

# --- Plot ---
fig, ax = plt.subplots()
ax.imshow(img_array, cmap="gray")

for box in data["boxes"]:
    if box["probability"] < 0.1: # only plot boxes with probability >= 0.1
        continue
    corners = [(x, y) for x, y, z in box["corners"]]  # ignore z (always 0)
    poly = Polygon(corners, closed=True, edgecolor='red', linewidth=1.5, fill=False)
    ax.add_patch(poly)
    cx = sum([pt[0] for pt in corners]) / 4
    cy = sum([pt[1] for pt in corners]) / 4
    ax.text(cx, cy, f"{box['probability']:.2f}", color='yellow', fontsize=8)

plt.title("Detections")
plt.axis("off")
plt.tight_layout()
plt.show()