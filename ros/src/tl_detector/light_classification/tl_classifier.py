import rospy
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime

class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier
        if is_site:
            fronzen_inf_graph = r'light_classification/model/site/frozen_inference_graph.pb'
        else:
            fronzen_inf_graph = r'light_classification/model/sim/frozen_inference_graph.pb'

        self.graph = tf.Graph()
        self.threshold = 0.5

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(fronzen_inf_graph, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name(
                'num_detections:0')

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        with self.graph.as_default():
            img_expand = np.expand_dims(image, axis=0)
            start = datetime.datetime.now()
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: img_expand})
            end = datetime.datetime.now()
            elapsed = end - start
            #print elapsed.total_seconds()

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        rospy.loginfo('Traffice Light Detection Result ---')
        rospy.loginfo('  elapsed sec (ms): %d', elapsed.microseconds/1000)
        rospy.loginfo('  score: %f', scores[0])
        # rospy.loginfo('  class: %d', classes[0])

        if scores[0] > self.threshold:
            if classes[0] == 1:
                rospy.loginfo('  *** Green ***')
                return TrafficLight.GREEN
            elif classes[0] == 2:
                rospy.loginfo('  *** RED RED ***')
                return TrafficLight.RED
            elif classes[0] == 3:
                rospy.loginfo('  *** Yellow ***')
                return TrafficLight.YELLOW        

        rospy.loginfo('  --- UNKOWN ---')
        return TrafficLight.UNKNOWN


