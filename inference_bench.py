import numpy as np
import torch
import cv2
import time

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_detections(img, detections, with_keypoints=True):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)

    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])

    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor="r", facecolor="none",
                                 alpha=detections[i, 16])
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1,
                                        edgecolor="lightskyblue", facecolor="none",
                                        alpha=detections[i, 16])
                ax.add_patch(circle)

    plt.show()

def save_benchmark(timecost, modulename, torch_device="cpu"):
    cost_np = np.array(timecost)
    print("==>> cost_np.shape: ", cost_np.shape)
    print("==>> type(cost_np): ", type(cost_np))
    import csv
    csvFile = open(torch_device + "_analyze.csv", 'w', newline='')
    writer = csv.writer(csvFile)
    writer.writerow([torch_device, "mean", "std", "min", "max", "count"])
    for i, name in enumerate(modulename):
        writer.writerow([name, cost_np[:,i].mean(), cost_np[:,i].std(), cost_np[:,i].min(), cost_np[:,i].max(), cost_np[:,i].shape[0]])
    np.savetxt(torch_device + "_timecost.csv", cost_np, delimiter=",")

from blazeface import BlazeFace

front_net = BlazeFace().to(device)
front_net.load_weights("blazeface.pth")
front_net.load_anchors("anchors.npy")
back_net = BlazeFace(back_model=True).to(device)
back_net.load_weights("blazefaceback.pth")
back_net.load_anchors("anchorsback.npy")

# Optionally change the thresholds:
front_net.min_score_thresh = 0.75
front_net.min_suppression_threshold = 0.3

t1 = time.time()

img = cv2.imread("1face.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

front_detections = front_net.predict_on_image(img)

t2 = time.time()
print("blazeface timecost:", (t2 - t1) * 1000.0, 'ms')
timecost = []
for i in range(100):
    t1 = time.time()
    front_detections = front_net.predict_on_image(img)
    t2 = time.time()
    timecost.append([(t2 - t1) * 1000.0])
    # print("blazeface timecost:", (t2 - t1) * 1000.0, 'ms')

save_benchmark(timecost, ["BlazeFace"], "cpu")

front_detections.shape

front_detections

plot_detections(img, front_detections)
