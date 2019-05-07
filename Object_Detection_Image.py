import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# 工作目录:object_detection
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

# 已训练的模型文件
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'test.jpg'

# 获取当前的路径
CWD_PATH = os.getcwd()

# 模型 + 模型查找表(要更换检测的对象,换的就是这个)
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# 有多少个类
NUM_CLASSES = 6

# 加载表
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# 读入图,创建会话.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# 输入类型是标准图
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# 输出类型是分布矩阵
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# 层次关系是权重优先
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# 检测数量
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# 用OpenCV读入一个图片,当然有摄像头可以用VideoCapture()
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)

# 跑一下识别
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# 在识别区画框
vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.60)

# 输出图片
cv2.imshow('Object detector', image)

# 按任意键关闭
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
