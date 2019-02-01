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

    def __init__(self):
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
        self.detect_thr = 0.4


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
        # Score is shown on the result image, together with the class label.
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


    #def detect_pixel_ratio(self, image_hsv, mask, intensity_threshold):
    #    bool_mask = mask > 0
    #    template = np.zeros_like(image_hsv, np.uint8)
    #    template[bool_mask] = image_hsv[bool_mask]
    #    # convert resulting image to grayscale
    #    template_rgb = cv2.cvtColor(template, cv2.COLOR_HSV2RGB)
    #    template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
    #   target_pixels = len(np.where(template_gray >= intensity_threshold)[0])
    #    other_pixels = len(np.where(template_gray < intensity_threshold)[0])
    #    return target_pixels, other_pixels


    def detect_traffic_light(self, image_crop):
        """
        :param cropped traffic light image
        :return: color of the traffic light
        """
        image_np = np.array(image_crop)
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # convert to hsv
        image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        intensity_threshold = 100
        # mask of red (0,50,50) ~ (10, 255,255) and (170, 50, 50) ~ (180, 255, 255)
        mask_red1 = cv2.inRange(image_hsv, (0, 50, 50), (10, 255, 255))
        mask_red2 = cv2.inRange(image_hsv, (170, 50, 50), (180, 255, 255))
        mask_red = mask_red1 + mask_red2
        #red_pixels, not_red_pixels = self.detect_pixel_ratio(image_hsv,
        #                                                     mask_red,
        #                                                     intensity_threshold)

        r = cv2.countNonZero(mask_red)

        #red_pixels_percent = 1. * red_pixels / (red_pixels + not_red_pixels)

        # mask of yellow in hsv (16, 50, 50) ~ (35, 255, 255)
        mask_yellow = cv2.inRange(image_hsv, (16, 50, 50), (35, 255, 255))
        #yellow_pixels, not_yellow_pixels = self.detect_pixel_ratio(image_hsv,
        #                                                           mask_yellow,
        #                                                           intensity_threshold)
        #yellow_pixels_percent = 1. * yellow_pixels / (yellow_pixels + not_yellow_pixels)

        y = cv2.countNonZero(mask_yellow)

        # mask of green in hsv (36,50,50) ~ (70, 255,255)
        mask_green = cv2.inRange(image_hsv, (36, 50, 50), (70, 255, 255))
        #green_pixels, not_green_pixels = self.detect_pixel_ratio(image_hsv,
        #                                                         mask_green,
        #                                                         intensity_threshold)
        #green_pixels_percent = 1. * green_pixels / (green_pixels + not_green_pixels)

        g = cv2.countNonZero(mask_green)

        rospy.loginfo("R: {}; Y: {}; G: {}".format(r,y,g))

        #threshold_percent = 0.020

        #if red_pixels_percent > threshold_percent:
        if r > 70:
            #rospy.loginfo("Detect  RED tl with  {} probability. {} / {}".format( red_pixels_percent, red_pixels, (red_pixels + not_red_pixels) ) )
            return TrafficLight.RED
        #elif yellow_pixels_percent > threshold_percent:
        elif y > 70:
            #rospy.loginfo("Detect  YELLOW tl with  {} probability".format( yellow_pixels_percent ) )
            return TrafficLight.YELLOW
        #elif green_pixels_percent > threshold_percent:
        elif g > 70:
            #rospy.loginfo("Detect  GREEN tl with  {} probability".format( green_pixels_percent ) )
            return TrafficLight.GREEN
        else:
            #rospy.loginfo("Can't detect traffic light color. Max probability: {}".format( max( red_pixels_percent, yellow_pixels_percent, green_pixels_percent ) ) )
            return TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args: image (cv::Mat): image containing the traffic light
        Returns: int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        rospy.loginfo("------\n")
        # get classification from tensorflow model
        boxes, scores, classes, tmp = self.get_tensorflow_classifier(image)

        # find all occurrences where probability is greater then the threshold
        max_prob_tl_idx = []
        for idx, score in enumerate(scores[0]):
            if score >= self.detect_thr and classes[0][idx] == self.category_id:
                max_prob_tl_idx.append(idx)
                #rospy.loginfo("idx: {}; score: {}".format( idx, score ))
        max_prob_tl_boxes = boxes[0][max_prob_tl_idx]

        traffic_lights = []
        for idx, box in enumerate(max_prob_tl_boxes):
            box_h = (box[2] - box[0]) * image.shape[0]
            box_w = (box[3] - box[1]) * image.shape[1]
            #rospy.loginfo("Box h: {}; w: {}".format(box_h, box_w))
            traffic_light_image = self.crop_traffic_light(box, image)
            traffic_light_color = self.detect_traffic_light(traffic_light_image)
            if traffic_light_color != TrafficLight.UNKNOWN:
                traffic_lights.append(traffic_light_color)

        if len(traffic_lights) > 0:
            return max(set(traffic_lights), key=traffic_lights.count)
        else:
            return TrafficLight.UNKNOWN
