import cv2
import numpy as np
import os, os.path
import rospy
import tensorflow as tf

from styx_msgs.msg import TrafficLight
import rospy
import tensorflow as tf
import numpy as np
import cv2
import os
from functools import partial

import os
import rospy

from PIL import Image, ImageDraw


class TLClassifier(object):

    def __init__(self, is_debug, threshold):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        model = os.path.join(os.getcwd(), 'light_classification/train/models/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb')
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.session = tf.Session(config=config, graph=self.detection_graph)
        self.category_id = 10 # Traffic light iamge category id
        self.detect_thr = threshold
        self.is_debug = is_debug


    def crop_traffic_light(self, box, image):
        """
        :param box: numpy.ndarray, 4-elements vector
        :param image: numpy.ndarray, camera image
        :return: cropped traffic light
        """
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        im_width, im_height = image_pil.size
        draw = ImageDraw.Draw(image_pil)
        ymin, xmin, ymax, xmax = box
        # Delete frame with 5px from box
        (left, right, top, bottom) = (xmin * im_width + 5, xmax * im_width + 5,
                                      ymin * im_height - 5, ymax * im_height + 5)
        traffic_light = image_pil.crop([int(left), int(top), int(right), int(bottom)])
        return traffic_light

    def get_tensorflow_classifier(self, image):
        """
        Get a tensorflow classifier and return classification results
        :param image: numpy array
        :return: [boxes, scores, classes, num]
        """
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.session.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        return boxes, scores, classes, num


    def detect_traffic_light(self, image_crop):
        """
        :param cropped traffic light image
        :return: color of the traffic light
        """
        image_np = np.array(image_crop)
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # convert to hsv
        image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        # mask of red (0,50,50) ~ (10, 255,255) and (170, 50, 50) ~ (180, 255, 255)
        mask_red1 = cv2.inRange(image_hsv, (0, 50, 50), (10, 255, 255))
        mask_red2 = cv2.inRange(image_hsv, (170, 50, 50), (180, 255, 255))
        mask_red = mask_red1 + mask_red2

        red_pixels = cv2.countNonZero(mask_red)

        # mask of yellow in hsv (16, 50, 50) ~ (35, 255, 255)
        mask_yellow = cv2.inRange(image_hsv, (16, 50, 50), (35, 255, 255))

        yellow_pixels = cv2.countNonZero(mask_yellow)

        # mask of green in hsv (36,50,50) ~ (70, 255,255)
        mask_green = cv2.inRange(image_hsv, (36, 50, 50), (70, 255, 255))

        green_pixels = cv2.countNonZero(mask_green)

	if self.is_debug:
            rospy.loginfo("Red pixels: {}; Yellow pixels: {}; Green pixels: {}".format(red_pixels,yellow_pixels,green_pixels))

        if red_pixels > 70:
            return TrafficLight.RED
        elif green_pixels > 70:
            return TrafficLight.GREEN
        elif yellow_pixels > 70:
            return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args: image (cv::Mat): image containing the traffic light
        Returns: int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if self.is_debug:
            rospy.loginfo("------\n")

        # get classification from tensorflow model
        boxes, scores, classes, tmp = self.get_tensorflow_classifier(image)

        # find all occurrences whith probability is greater then the threshold
        max_prob_tl_idx = []
        for idx, score in enumerate(scores[0]):
            if score >= self.detect_thr and classes[0][idx] == self.category_id:
                max_prob_tl_idx.append(idx)
                if self.is_debug:
                    rospy.loginfo("idx: {}; score: {}".format( idx, score ))

        max_prob_tl_boxes = boxes[0][max_prob_tl_idx]

        traffic_lights = []
        for idx, box in enumerate(max_prob_tl_boxes):
            if self.is_debug:
                box_h = (box[2] - box[0]) * image.shape[0]
                box_w = (box[3] - box[1]) * image.shape[1]
                rospy.loginfo("Box h: {}; w: {}".format(box_h, box_w))

            traffic_light_image = self.crop_traffic_light(box, image)
            traffic_light_color = self.detect_traffic_light(traffic_light_image)
            if traffic_light_color != TrafficLight.UNKNOWN:
                traffic_lights.append(traffic_light_color)

        if len(traffic_lights) > 0:
            return max(set(traffic_lights), key=traffic_lights.count)
        else:
            return TrafficLight.UNKNOWN
