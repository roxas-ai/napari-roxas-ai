"""
Original code by github user triyan-b https://github.com/triyan-b
"""

import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

pixels_per_um = 1.5333
pixels_per_um = 2.2675

# Read and preprocess the image
index = 0
img_file = "clusters.png"
img_file = sorted(glob.glob("CropsForTesting/*.png"))[index]

path = Path(img_file)
print(f"Reading image {path.name}")
img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2GRAY)
img_3c = cv2.merge([img, img, img])
print("Read image with shape", img.shape)

cells = []
clusters = []
lumen_contours, hierarchy = cv2.findContours(
    img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
print(f"Found {len(lumen_contours)} lumen contours")
for i, c in enumerate(lumen_contours[:]):
    M = cv2.moments(c)
    if (m00 := M["m00"]) != 0:
        cx = int(M["m10"] / m00)
        cy = int(M["m01"] / m00)
        cells.append({"centroid": (cx, cy)})
    else:
        print(
            i,
            "Lumen centroid not found: contour not segmented properly near",
            np.mean(c, axis=0)[0],
        )
        continue

# Distance transform + dilation according to separation threshold + normalization
separation_threshold_um = 5
dist_orig = cv2.distanceTransform(
    np.uint8(cv2.bitwise_not(img)), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
)
dist = cv2.normalize(dist_orig, 0, 1.0, cv2.NORM_MINMAX)
_, dist_thresholded = cv2.threshold(
    dist_orig / pixels_per_um,
    separation_threshold_um,
    np.max(dist_orig),
    cv2.THRESH_BINARY,
)
dist_thresholded = (dist_thresholded - np.min(dist_thresholded)) / (
    np.max(dist_thresholded) - np.min(dist_thresholded)
)

num_clusters, clusters = cv2.connectedComponents(
    np.bitwise_not((dist_thresholded * 255).astype("uint8")), connectivity=8
)
print(f"Number of clusters: {num_clusters - 1}")

np.random.seed(42)
colored_labels = np.zeros((*clusters.shape, 3), dtype=np.uint8)
for cluster in range(1, num_clusters):  # Skip background
    colored_labels[clusters == cluster] = np.random.randint(0, 255, 3)
colored_labels = cv2.bitwise_and(colored_labels, img_3c)

for cell in cells:
    cluster = clusters[cell["centroid"][::-1]]
    assert cluster != 0
    cell.update({"cluster": cluster})


# Create the figure
fig, ax = plt.subplots(1, 4, figsize=(16, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Plot the initial images
ax[0].set_title("Original Binary Image")
ax[0].imshow(img, cmap="gray")
ax[0].axis("off")

ax[1].set_title("Distance transform")
ax[1].imshow(dist, cmap="binary_r")
ax[1].axis("off")

ax[2].set_title("Thresholded distance transform")
dist_img = ax[2].imshow(dist_thresholded, cmap="binary")
ax[2].axis("off")

ax[3].set_title("Clusters")
result_img = ax[3].imshow(colored_labels)
ax[3].axis("off")

# Add a slider to control the threshold value
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # [left, bottom, width, height]
threshold_slider = Slider(
    ax_slider,
    "Threshold",
    0,
    2 * separation_threshold_um,
    valinit=separation_threshold_um,
    valstep=0.1,
)


# Update function for the slider
def update(val):
    _, dist_thresholded = cv2.threshold(
        dist_orig / pixels_per_um, val, np.max(dist_orig), cv2.THRESH_BINARY
    )
    dist_thresholded = (dist_thresholded - np.min(dist_thresholded)) / (
        np.max(dist_thresholded) - np.min(dist_thresholded)
    )
    num_clusters, clusters = cv2.connectedComponents(
        cv2.bitwise_not((dist_thresholded * 255).astype("uint8")),
        connectivity=8,
    )
    print(f"Number of clusters: {num_clusters - 1}")
    colored_labels = np.zeros((*clusters.shape, 3), dtype=np.uint8)
    np.random.seed(42)
    for label in range(1, num_clusters):  # Skip background
        colored_labels[clusters == label] = np.random.randint(0, 255, 3)
    colored_labels = cv2.bitwise_and(colored_labels, img_3c)

    for cell in cells:
        cluster = clusters[cell["centroid"][::-1]]
        assert cluster != 0
        cell.update({"cluster": cluster})

    dist_img.set_data(dist_thresholded)
    result_img.set_data(colored_labels)
    fig.canvas.draw_idle()


# Attach the update function to the slider
threshold_slider.on_changed(update)

# Show the interactive plot
plt.show()
