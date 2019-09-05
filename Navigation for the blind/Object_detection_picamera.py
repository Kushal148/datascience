
import time
import RPi.GPIO as GPIO
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
TRIG = 23
echo = 24



IM_WIDTH = 640
IM_HEIGHT = 480
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'
sys.path.append('..')

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_complete_label_map.pbtxt')

NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX



def dist_mes():
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(echo, GPIO.IN)
    
    

    GPIO.output(TRIG, GPIO.LOW)
    print("waiting...")
    time.sleep(0.1)

    GPIO.output(TRIG, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIG, GPIO.LOW)

    pulse_start = 0
    pulse_end = 0

    while (GPIO.input(echo) == GPIO.LOW):
        pulse_start = time.time()
    while (GPIO.input(echo) == GPIO.HIGH):
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    
    d = round(distance)
    
    print("Distance : ", d, "centimeters")
    os.system('espeak " distance is {} centimeters"'.format(str(d)))
    
    
def object_detector(frame):
    global detected_inside, detected_outside
    global inside_counter, outside_counter
    global pause, pause_counter

    frame_expanded = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.40)



    

    if (((int(classes[0][0]) == 2) or(int(classes[0][0]) == 0) or (int(classes[0][0] == 1) or (int(classes[0][0]) == 3) or (int(classes[0][0]) == 4) or (int(classes[0][0]) == 6) or (int(classes[0][0]) == 7) or (int(classes[0][0]) == 8) or (int(classes[0][0]) == 10) or (int(classes[0][0]) == 11) or (int(classes[0][0]) == 13) or (int(classes[0][0]) == 15) or (int(classes[0][0]) == 17) or (int(classes[0][0]) == 18) or (int(classes[0][0]) == 21) or (int(classes[0][0]) == 27) or (int(classes[0][0]) == 28) or (int(classes[0][0]) == 31) or (int(classes[0][0]) == 33) or (int(classes[0][0]) == 44) or (int(classes[0][0]) == 62) or (int(classes[0][0]) == 73) or (int(classes[0][0]) == 74) or (int(classes[0][0]) == 76)))):

        time.sleep(0.5)

        if(int(classes[0][0]==0)):
            print('------------------')

        if(int(classes[0][0]==1)):
               print('person')
               os.system("espeak 'person ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==2)):
               print('bicycle')
               os.system("espeak 'bicycle ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==3)):
               print('car')
               os.system("espeak 'car ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==4)):
               print('motorcycle')
               os.system("espeak 'motorcycle ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==6)):
               print('bus')
               os.system("espeak 'bus ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==7)):
               print('train')
               os.system("espeak 'train ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==8)):
               print('truck')
               os.system("espeak 'truck ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==10)):
               print('traffic light')
               os.system("espeak 'traffic light ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==11)):
               print('fire hydrant')
               os.system("espeak 'fire hydrant ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==13)):
               print('stop sign')
               os.system("espeak 'stop sign ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==15)):
               print('bench')
               os.system("espeak 'bench ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==17)):
               print('cat')
               os.system("espeak 'cat ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==18)):
               print('dog')
               os.system("espeak 'dog ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==21)):
               print('cow')
               os.system("espeak 'cow ahead'")
               dist_mes()
               time.sleep(1)
               
        if(int(classes[0][0]==27)):
               print('backpack')
               os.system("espeak 'backpack ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==28)):
               print('umbrella')
               os.system("espeak 'umbrella ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==31)):
               print('handbag')
               os.system("espeak 'handbag ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==33)):
               print('suitcase')
               os.system("espeak 'suitcase ahead'")
               dist_mes()
               time.sleep(1)
               
        if(int(classes[0][0]==62)):
               print('chair')
               os.system("espeak 'chair ahead'")
               dist_mes()
               time.sleep(1)

        if(int(classes[0][0]==73)):
               print('laptop')
               os.system("espeak 'laptop ahead'")
               dist_mes()
               time.sleep(1)
           
        if(int(classes[0][0]==74)):
               print('mouse')
               os.system("espeak 'mouse ahead'")
               dist_mes()
               time.sleep(1)
        
        if(int(classes[0][0]==76)):
               print('keyboard')
               os.system("espeak 'keyboard ahead'")
               dist_mes()
               time.sleep(1)

        
     
   
    return frame


###----------------- Picamera-------------- ###
if camera_type == 'picamera':
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame = object_detector(frame)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()
        
cv2.destroyAllWindows()
