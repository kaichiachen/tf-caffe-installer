# coding: utf-8

import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Pdf")
from matplotlib import pyplot as plt
import scipy.misc

# Here are the imports from the object detection module.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

import time
begin = time.time()

# # Model preparation 
MODEL_NAME = '/home/ubuntu/models/research/object_detection/models/model/ssd_inception_v2_coco_2017_11_17'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(MODEL_NAME + '/frozen_inference_graph.pb')
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/ubuntu/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = '/home/ubuntu/frames'
imgfiles = [f for f in listdir(PATH_TO_TEST_IMAGES_DIR) if isfile(join(PATH_TO_TEST_IMAGES_DIR, f)) and f.lower().endswith(('.jpg','jpeg','.png'))]
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, f) for f in imgfiles ]

IMAGE_SIZE = (12, 8)

end = time.time()
print("preparation cost: {} secs".format(end - begin))

array_num_person = []
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for f in imgfiles:
            image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, f)
            start = time.time()
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            image_np = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4)
            fin = time.time()
            print("analysis cost: {}secs, image: {}".format(fin - start, f))
            scipy.misc.imsave('/home/ubuntu/tf_detect_output/'+f, image_np)
