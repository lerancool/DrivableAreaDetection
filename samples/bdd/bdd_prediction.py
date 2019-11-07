import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import bdd_train

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/bdd/"))  # To find local version


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
BDD_MODEL_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_bdd_0010.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "datasets/bdd/test")


class InferenceConfig(bdd_train.BddConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create Model and Load Trained Weights
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(BDD_MODEL_PATH, by_name=True)

# Class names
class_names = ['BG', 'drivable area']

# Run Object Detection
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]

if ".DS_Store" in file_names:
    file_names.remove('.DS_Store')

for image_name in file_names:
    # Run detection

    image = skimage.io.imread(os.path.join(IMAGE_DIR, image_name))

    results = model.detect([image], verbose=1)
    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

print(file_names)
