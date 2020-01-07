#!/usr/bin/env python

"""
    Openpose python wrapper: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/modules/python_module.md
    Code based on: https://github.com/geaxgx/tello-openpose
    Modify MODEL_FOLDER to point to the directory where the models are installed
"""
from __future__ import print_function
import cv2
import sys
import argparse
from collections import namedtuple
import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage  # topics /hires/image_raw, /hires/image_raw/compressed
from cv_bridge import CvBridge
from math import atan2, degrees, sqrt, pi

try:
    sys.path.append('/usr/local/python')
    sys.path.append('/home/swarm/openpose/build/python/openpose')
    #from openpose import pyopenpose as op
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

#MODEL_FOLDER = "/home/nvidia/openpose/models/"
MODEL_FOLDER = "/home/swarm/openpose/models/"

def distance (A, B):
    """
        Calculate the square of the distance between points A and B
    """
    return int(sqrt((B[0]-A[0])**2 + (B[1]-A[1])**2))

def angle (A, B, C):
    """
        Calculate the angle between segment(A,p2) and segment (p2,p3)
    """
    if A is None or B is None or C is None:
        return None
    return degrees(atan2(C[1]-B[1],C[0]-B[0]) - atan2(A[1]-B[1],A[0]-B[0]))%360

def vertical_angle (A, B):
    """
        Calculate the angle between segment(A,B) and vertical axe
    """
    if A is None or B is None:
        return None
    return degrees(atan2(B[1]-A[1],B[0]-A[0]) - pi/2)

# map body part point number to name and vice versa
body_kp_id_to_name = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "MidHip",
    9: "RHip",
    10: "RKnee",
    11: "RAnkle",
    12: "LHip",
    13: "LKnee",
    14: "LAnkle",
    15: "REye",
    16: "LEye",
    17: "REar",
    18: "LEar",
    19: "LBigToe",
    20: "LSmallToe",
    21: "LHeel",
    22: "RBigToe",
    23: "RSmallToe",
    24: "RHeel",
    25: "Background"}


body_kp_name_to_id = {v: k for k, v in body_kp_id_to_name.items()}


# for body, create pairs-->line segments between location data points
Pair = namedtuple('Pair', ['p1', 'p2', 'color'])
color_right_side = (0,255,0)
color_left_side = (0,0,255)
color_middle = (0,255,255)
color_face = (255,255,255)

pairs_head = [
    Pair("Nose", "REye", color_right_side),
    Pair("Nose", "LEye", color_left_side),
    Pair("REye", "REar", color_right_side),
    Pair("LEye", "LEar", color_left_side)
]

pairs_upper_limbs = [
    Pair("Neck", "RShoulder", color_right_side),
    Pair("RShoulder", "RElbow", color_right_side),
    Pair("RElbow", "RWrist", color_right_side),
    Pair("Neck", "LShoulder", color_left_side),
    Pair("LShoulder", "LElbow", color_left_side),
    Pair("LElbow", "LWrist", color_left_side)
]

pairs_lower_limbs = [
    Pair("MidHip", "RHip", color_right_side),
    Pair("RHip", "RKnee", color_right_side),
    Pair("RKnee", "RAnkle", color_right_side),
    Pair("RAnkle", "RHeel", color_right_side),
    Pair("MidHip", "LHip", color_left_side),
    Pair("LHip", "LKnee", color_left_side),
    Pair("LKnee", "LAnkle", color_left_side),
    Pair("LAnkle", "LHeel", color_left_side)
]

pairs_spine = [
    Pair("Nose", "Neck", color_middle),
    Pair("Neck", "MidHip", color_middle)
]

pairs_feet = [
    Pair("RAnkle", "RBigToe", color_right_side),
    Pair("RAnkle", "RHeel", color_right_side),
    Pair("LAnkle", "LBigToe", color_left_side),
    Pair("LAnkle", "LHeel", color_left_side)
]


pairs_body = pairs_head + pairs_upper_limbs + pairs_lower_limbs + pairs_spine + pairs_feet

# map face data points to id number
# DONE: TODO: translate all f strings to python2: '{} {}'.format('one', 'two')
# or maybe not: https://github.com/asottile/future-fstrings
face_kp_id_to_name = {}
for i in range(17):
    face_kp_id_to_name[i] = "Jaw{}".format(i+1)
    #Jaw1 --> Jaw17
for i in range(5):
    face_kp_id_to_name[i+17 ] = "REyebrow{}".format(5-i)
    face_kp_id_to_name[i+22] = "LEyebrow{}".format(i+1)
for i in range(6):
    face_kp_id_to_name[(39-i) if i<4 else (45-i)] = "REye{}".format(i+1)
    face_kp_id_to_name[i+42] = "LEye{}".format(i+1)
face_kp_id_to_name[68] = "REyeCenter"
face_kp_id_to_name[69] = "LEyeCenter"
for i in range(9):
    face_kp_id_to_name[27+i] = "Nose{}".format(i+1)
for i in range(12):
    face_kp_id_to_name[i+48] = "OuterLips{}".format(i+1)
for i in range(8):
    face_kp_id_to_name[i+60] = "InnerLips{}".format(i+1)

face_kp_name_to_id = {v: k for k, v in face_kp_id_to_name.items()}
# why the chosen dict entries? ...Look on OpenPose Demo - Output

# for face, create pairs-->line segments between location data points
pairs_jaw = [ Pair("Jaw{}".format(i+1), "Jaw{}".format(i+2), color_face) for i in range(16)]
pairs_nose = [ Pair("Nose{}".format(i+1), "Nose{}".format(i+2), color_face) for i in range(3)] + [ Pair("Nose{}".format(i+1), "Nose{}".format(i+2), color_face) for i in range(4,8)]

pairs_left_eye = [ Pair("LEye{}".format(i+1), "LEye{}".format(i+2), color_face) for i in range(5)] + [Pair("LEye6","LEye1",color_face)]
pairs_right_eye = [ Pair("REye{}".format(i+1), "REye{}".format(i+2), color_face) for i in range(5)] + [Pair("REye6","REye1",color_face)]
pairs_eyes = pairs_left_eye + pairs_right_eye

pairs_left_eyebrow = [ Pair("LEyebrow{}".format(i+1), "LEyebrow{}".format(i+2), color_face) for i in range(4)]
pairs_right_eyebrow = [ Pair("REyebrow{}".format(i+1), "REyebrow{}".format(i+2), color_face) for i in range(4)]
pairs_eyesbrow = pairs_left_eyebrow + pairs_right_eyebrow

pairs_outer_lips = [ Pair("OuterLips{}".format(i+1), "OuterLips{}".format(i+2), color_face) for i in range(11)] + [Pair("OuterLips12","OuterLips1",color_face)]
pairs_inner_lips = [ Pair("InnerLips{}".format(i+1), "InnerLips{}".format(i+2), color_face) for i in range(7)] + [Pair("InnerLips8","InnerLips1",color_face)]
pairs_mouth = pairs_outer_lips + pairs_inner_lips


pairs_face = pairs_jaw + pairs_nose + pairs_eyes + pairs_eyesbrow + pairs_mouth

# 
class OP:
    @staticmethod
    # def distance_kps(kp1,kp2):
    #     x1,y1,c1 = kp1
    #     x2,y2,c2 = kp2
    #     if c1 > 0 and c2 > 0:
    #         return abs(x2-x1)+abs(y2-y1)
    #     else: 
    #         return 0
    def distance_kps(kp1,kp2):
        # kp1 and kp2: numpy array of shape (3,): [x,y,conf]
        # 
        x1,y1,c1 = kp1
        x2,y2,c2 = kp2
        if kp1[2] > 0 and kp2[2] > 0:
            return np.linalg.norm(kp1[:2]-kp2[:2])
        else: 
            return 0
    @staticmethod
    def distance (p1, p2):
            """
                Distance between p1(x1,y1) and p2(x2,y2)
            """
            return np.linalg.norm(np.array(p1)-np.array(p2))

    def __init__(self, number_people_max=-1, min_size=-1, openpose_rendering=False, face_detection=False, frt=0.4, hand_detection=False, debug=False):
        """
        openpose_rendering : if True, rendering is made by original Openpose library. Otherwise rendering is to the
        responsability of the user (~0.2 fps faster)
        """
        #self.veh = ns[1:-1]  # Remove leading/trailing slash

        self.openpose_rendering = openpose_rendering
        self.min_size = min_size
        self.debug = debug
        self.face_detection = face_detection
        self.hand_detection = hand_detection
        self.frt = frt
        
        params = dict()
        params["model_folder"] = MODEL_FOLDER
        params["model_pose"] = "BODY_25"
        params["number_people_max"] = number_people_max
        if not self.openpose_rendering:
            params["render_pose"] = 0
        if self.face_detection:
            params["face"] = True
            params["face_render_threshold"] = self.frt
        # if self.hand_detection:
            params["hand"] = True
        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
        self.datum = op.Datum()

        # Convert ROS images to cv::Mat
        self.br = CvBridge()
        # Publish rendered images
        self.pubImage = rospy.Publisher("openpose_image/compressed", CompressedImage, queue_size=1, latch=True)

        self.time_last_img = rospy.get_rostime()
        self.img_freq = 0.8  # s TODO: parameter in launch file?


    def eval(self, frame):
        '''
        for one image, set poseKeypoints & faceKeypoints & nb_persons
        return nb_persons, body_kps, face_kps
        '''
        self.frame = frame

        self.datum.cvInputData = frame.copy()  # cv2.imread("/home/nvidia/Desktop/image.png")
        self.opWrapper.emplaceAndPop([self.datum])
        
        if self.openpose_rendering: # publish image to publisher node
            img = self.br.cv2_to_compressed_imgmsg(self.datum.cvOutputData)#, encoding="bgr8")
            self.pubImage.publish(img)

        if self.datum.poseKeypoints.shape: # When no person is detected, shape = (), else (nb_persons, 25, 3)
            self.body_kps = self.datum.poseKeypoints
            self.face_kps = self.datum.faceKeypoints
            # the code to sort persons by size was bugged, so it has been deleted
            self.nb_persons = len(self.body_kps)

            if self.face_detection:
                self.face_kps = self.face_kps[order]
                self.face_kps = self.face_kps[big_enough]
                filter=self.face_kps[:,:,2]<self.frt
                self.face_kps[filter] = 0

        else:
            self.nb_persons = 0
            self.body_kps = []
            self.face_kps = []
        
        return self.nb_persons,self.body_kps, self.face_kps

    def draw_pairs_person(self, frame, kps, kp_name_to_id, pairs, person_idx=0, thickness=3, color=None):
        """
            Draw on 'frame' pairs of keypoints for one person in frame
        """
        person = kps[person_idx]
        for pair in pairs:
            p1_x,p1_y,p1_conf = person[kp_name_to_id[pair.p1]]
            p2_x,p2_y,p2_conf = person[kp_name_to_id[pair.p2]]
            if p1_conf != 0 and p2_conf != 0:
                col = color if color else pair.color
                cv2.line(frame, (p1_x, p1_y), (p2_x, p2_y), col, thickness)

    def draw_pairs(self, frame, kps, kp_name_to_id, pairs, thickness=3, color=None):
        """
            Draw on 'frame' pairs of keypoints for all people ini frame
        """
        for person_idx in range(self.nb_persons):
            self.draw_pairs_person(frame, kps, kp_name_to_id, pairs, person_idx, thickness=thickness, color=color)


    def draw_body(self, frame, pairs=pairs_body, thickness=3, color=None):
        """
            Draw on 'frame' pairs of body keypoints for all people in frame
        """
        self.draw_pairs(frame, self.body_kps, body_kp_name_to_id, pairs, thickness, color)

    def draw_face(self, frame, pairs=pairs_face, thickness=2, color=None):
        """
            Draw on 'frame' pairs of face keypoints for all people in frame
        """
        self.draw_pairs(frame, self.face_kps, face_kp_name_to_id, pairs, thickness, color)

    def draw_eyes_person (self, frame, person_idx=0):
        """
            Draw on 'frame' pairs of right/left eyes for one person in frame
        """
        eyes_status = self.check_eyes(person_idx=person_idx)
        # right eye
        if eyes_status in [1,3]:
            color = (0,200,230)
        else:
            color = (230,230,0)
        self.draw_pairs_person(frame,self.face_kps,face_kp_name_to_id,pairs_right_eye,person_idx,2,color)
        
        # left eye
        if eyes_status in [2,3]:
            color = (0,200,230)
        else:
            color = (230,230,0)
        self.draw_pairs_person(frame,self.face_kps,face_kp_name_to_id,pairs_left_eye,person_idx,2,color)


    def draw_eyes (self, frame):
        """
            Draw on 'frame' pairs of right/left eyes for all people in frame
        """
        for person_idx in range(self.nb_persons):
            self.draw_eyes_person(frame, person_idx)

    def get_body_kp(self, kp_name="Neck", person_idx=0):
        """
            Return the coordinates of a keypoint named 'kp_name' of the person of index 'person_idx' (from 0), or None if keypoint not detected
        """
        try:
            kps = self.datum.poseKeypoints[person_idx]
        except:
            print("get_body_kp: invalid person_idx '{}'".format(person_idx))
            return None
        try:
            x,y,conf = kps[body_kp_name_to_id[kp_name]]
        except:
            print("get_body_kp: invalid kp_name '{}'".format(kp_name))
            return None
        if x or y:
            return (int(x),int(y))
        else:
            return None

    def get_face_kp(self, kp_name="Nose_Tip", person_idx=0):
        """
            Return the coordinates of a keypoint named 'kp_name' of the face of the person of index 'person_idx' (from 0), or None if keypoint not detected
        """
        try:
            kps = self.datum.faceKeypoints[person_idx]
        except:
            print("get_face_kp: invalid person_idx '{}'".format(person_idx))
            return None
        try:
            x,y,conf = kps[face_kp_name_to_id[kp_name]]
        except:
            print("get_face_kp: invalid kp_name '{}'".format(kp_name))
            return None
        if x or y:
            return (int(x),int(y))
        else:
            return None
    
    def length(self, pairs, person_idx=0, coefs = None):
        """
            Calculate the mean of the length of the pairs in the list 'pairs' for the person of index 'person_idx' (from 0)
            If one (or both) of the 2 points of a pair is missing, the number of pairs used to calculate the average is decremented of 1
        """
        # Not used in this script.  Why useful?
        if coefs is None:
            coefs = [1] * len(pairs)

        person = self.body_kps[person_idx]

        l_cum = 0
        n = 0
        for i,pair in enumerate(pairs):
            l = self.distance_kps(person[body_kp_name_to_id[pair.p1]], person[body_kp_name_to_id[pair.p2]])
            if l != 0:
                l_cum += l * coefs[i]
                n += 1
        if n>0:
            return l_cum/n
        else:
            return 0
    
    def check_eyes(self, person_idx=0):
        """
            Check if the person whose index is 'person_idx' has his eyes closed
            Return :
            0 if both eyes are open,
            1 if only right eye is closed
            2 if only left eye is closed
            3 if both eyes are closed            
        """

        eye_aspect_ratio_threshold = 0.2  # If ear < threshold, eye is closed

        reye_closed = False
        reye1 = self.get_face_kp("REye1", person_idx=person_idx)
        reye2 = self.get_face_kp("REye2", person_idx=person_idx)
        reye3 = self.get_face_kp("REye3", person_idx=person_idx)
        reye4 = self.get_face_kp("REye4", person_idx=person_idx)
        reye5 = self.get_face_kp("REye5", person_idx=person_idx)
        reye6 = self.get_face_kp("REye6", person_idx=person_idx)
        if reye1 and reye2 and reye3 and reye4 and reye5 and reye6:
            right_eye_aspect_ratio = (self.distance(reye2, reye6)+self.distance(reye3, reye5))/(2*self.distance(reye1, reye4))
            if right_eye_aspect_ratio < eye_aspect_ratio_threshold:
                reye_closed = True
                print("RIGHT EYE CLOSED")
        
        leye_closed = False
        leye1 = self.get_face_kp("LEye1", person_idx=person_idx)
        leye2 = self.get_face_kp("LEye2", person_idx=person_idx)
        leye3 = self.get_face_kp("LEye3", person_idx=person_idx)
        leye4 = self.get_face_kp("LEye4", person_idx=person_idx)
        leye5 = self.get_face_kp("LEye5", person_idx=person_idx)
        leye6 = self.get_face_kp("LEye6", person_idx=person_idx)
        if leye1 and leye2 and leye3 and leye4 and leye5 and leye6:
            left_eye_aspect_ratio = (self.distance(leye2, leye6)+self.distance(leye3, leye5))/(2*self.distance(leye1, leye4))
            if left_eye_aspect_ratio < eye_aspect_ratio_threshold:
                leye_closed = True
                print("LEFT EYE CLOSED")
        if reye_closed:
            if leye_closed:
                return 3
            else:
                return 1
        elif leye_closed:
            return 2
        else:
            return 0

    def check_pose(self):
        """
            Check if we detect a pose in the body detected by Openpose.
            The origin is in the top left corner. Right is +x, down is +y.
            9 poses coded: LEFT/RIGHT_ARM_UP_OPEN/CLOSED, LEFT/RIGHT_HAND_ON_LEFT_EAR,
            HANDS_ON_EARS, CLOSE_HANDS_UP, HANDS_ON_NECK
        """
    
        neck = self.get_body_kp("Neck")
        r_wrist = self.get_body_kp("RWrist")
        l_wrist = self.get_body_kp("LWrist")
        r_elbow = self.get_body_kp("RElbow")
        l_elbow = self.get_body_kp("LElbow")
        r_shoulder = self.get_body_kp("RShoulder")
        l_shoulder = self.get_body_kp("LShoulder")
        r_ear = self.get_body_kp("REar")
        l_ear = self.get_body_kp("LEar")
        
        self.shoulders_width = distance(r_shoulder,l_shoulder) if r_shoulder and l_shoulder else None
        print("Shoulders width ", self.shoulders_width)


        vert_angle_right_arm = vertical_angle(r_wrist, r_elbow)
        vert_angle_left_arm = vertical_angle(l_wrist, l_elbow)

        left_hand_up = neck and l_wrist and l_wrist[1] < neck[1]
        right_hand_up = neck and r_wrist and r_wrist[1] < neck[1]
        print("left_hand_up", left_hand_up)
        print("right_hand_up", right_hand_up)

        if right_hand_up:
            if not left_hand_up:
                # Only right arm up
                if r_ear and (r_ear[0]-neck[0])*(r_wrist[0]-neck[0])>0:
                # Right ear and right hand on the same side
                    if vert_angle_right_arm:
                        if vert_angle_right_arm < -15:
                            return "RIGHT_ARM_UP_OPEN"
                        if 15 < vert_angle_right_arm < 90:
                            return "RIGHT_ARM_UP_CLOSED"
                elif l_ear and self.shoulders_width and distance(r_wrist,l_ear) < self.shoulders_width/4:
                    # Right hand close to left ear
                    return "RIGHT_HAND_ON_LEFT_EAR"
            else:
                # Both hands up
                # Check if both hands are on the ears
                if r_ear and l_ear:
                    ear_dist = distance(r_ear,l_ear)
                    if distance(r_wrist,r_ear)<ear_dist/3 and distance(l_wrist,l_ear)<ear_dist/3:
                        return("HANDS_ON_EARS")
                # Check if boths hands are closed to each other and above ears 
                # (check right hand is above right ear is enough since hands are closed to each other)
                if self.shoulders_width and r_ear:
                    near_dist = self.shoulders_width/1.7
                    if r_ear[1] > r_wrist[1] and distance(r_wrist, l_wrist) < near_dist :
                        return "CLOSE_HANDS_UP"

        else:
            if left_hand_up:
                # Only left arm up
                if l_ear and (l_ear[0]-neck[0])*(l_wrist[0]-neck[0])>0:
                    # Left ear and left hand on the same side
                    if vert_angle_left_arm:
                        if vert_angle_left_arm < -15:
                            return "LEFT_ARM_UP_CLOSED"
                        if 15 < vert_angle_left_arm < 90:
                            return "LEFT_ARM_UP_OPEN"
                elif r_ear and self.shoulders_width and distance(l_wrist,r_ear) < self.shoulders_width/4:
                    # Left hand close to right ear
                    return "LEFT_HAND_ON_RIGHT_EAR"
            else:
                # Both wrists under the neck
                if neck and self.shoulders_width and r_wrist and l_wrist:
                    near_dist = self.shoulders_width/3
                    if distance(r_wrist, neck) < near_dist and distance(l_wrist, neck) < near_dist :
                        return "HANDS_ON_NECK"

        return None


    def imageCB(self, data):
        if rospy.get_rostime() - self.time_last_img <= rospy.Duration.from_sec(self.img_freq):
            return 
        self.time_last_img = rospy.get_rostime()

        #frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")
        
        #img_str = np.fromstring(data.data, np.uint8)
        #ncol,nrow = data.size()
        #nparr = np.fromstring(img_str, dtype=np.uint8).reshape(nrow,ncol,3)

        
        np_arr = np.fromstring(data.data, np.uint8)
        print(np_arr.shape)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        print(frame)
        


        #cv2.imshow("Rendering", frame)
        #cv2.waitKey(0)
        nb_persons,body_kps,face_kps = self.eval(frame)
        
        #self.draw_body(frame)
        #if self.face_detection: 
        #    self.draw_face(frame)
        #    self.draw_eyes(frame)
        #cv2.imshow("Rendering", frame)
        #cv2.waitKey(0)

        print("Number of persons found:", nb_persons)
        if nb_persons == 0:
            return

        # Interpret the keypoints returned by openpose
        self.pose = self.check_pose() # self.w,self.h)
        if self.pose:
            # We trigger the associated action... TODO
            print("Pose:", self.pose)
            print
        else:
            print("No pose was detected")



if __name__ == '__main__':
    try:
        rospy.init_node('openpose_node')        # TODO: get ros params
        number_people_max = 1  # limit the number of people detected
        face = False  # enable face keypoint detection
        frt = 0.4  # face rendering threshold
        rendering = True  # show the original rendering made by Openpose lib
        my_op = OP(openpose_rendering=rendering, number_people_max=number_people_max, min_size=60, face_detection=face, frt=frt)
        #my_op.w = 640
        #my_op.h = 480
        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, my_op.imageCB)
        #cv2.destroyAllWindows()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass





