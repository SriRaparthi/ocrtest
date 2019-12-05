import os
import re
import cv2
import sys
import dlib
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import pytesseract
import face_recognition as fr
import warnings
warnings.filterwarnings("ignore")
from utils import label_map_util
from utils import visualization_utils as vis_util
# from ktpocr.extractor import KTPOCR
from flask import jsonify
import argparse
import ocrconfig as cfg


MODEL_NAME = 'model'
# IMAGE_NAME = '..\\OCR-Final\\photos2\\21ktp.jpg'
# SELFIE_IMAGE_NAME = '..\\OCR-Final\\photos2\\21photo.jpg'
CWD_PATH = os.getcwd()
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')
# Path to image
# PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
# PATH_TO_SELFIE_IMAGE = os.path.join(CWD_PATH,SELFIE_IMAGE_NAME)
# Number of classes the object detector can identify
NUM_CLASSES = 1
# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)
    # Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
def getimage(PATH_TO_IMAGE):
      try:
            if os.path.isfile(PATH_TO_IMAGE):
                  image = plt.imread(PATH_TO_IMAGE)
                  print('KTP Image Loaded')
                  msg2 = 'KTP image loaded'
                  return image,PATH_TO_IMAGE,msg2
            else:
                  print ("The file " + PATH_TO_IMAGE + " does not exist.")
                  msg2 = 'Image file does not exist'
                  return None,PATH_TO_IMAGE,msg2
      except OSError as e:
            print("Invalid image file!")
            raise OSError("Invalid image file! :" + str(e))
      except IOError as e:
            print("Invalid image file!")
            raise IOError("Invalid image file! :" + str(e))
      except e:
            raise(e)

def getselfie_img(PATH_TO_SELFIE_IMAGE):
      try:
            if os.path.isfile(PATH_TO_SELFIE_IMAGE):
                  selfie_img = plt.imread(PATH_TO_SELFIE_IMAGE)
                  print('Selfie Image Loaded')
                  msg2 = 'Selfie Image Loaded'
            else:
                  print ("The file " + PATH_TO_SELFIE_IMAGE + " does not exist.")
                  msg2 = 'Image file does not exist'
            return selfie_img,PATH_TO_SELFIE_IMAGE,msg2
      except OSError as e:
            print("Invalid image file!")
            raise OSError("Invalid image file! :" + str(e))
      except IOError as e:
            print("Invalid image file!")
            raise IOError("Invalid image file! :" + str(e))
      
            
def selfieimageproc(selfie_img):
      #DETECTING FACE FROM THE KTP AND CROPPING
      #Loading 194 face landmarks trained data file into predictor path
      predictor_path = cfg.predictor_path
      detector = dlib.get_frontal_face_detector()
      predictor = dlib.shape_predictor(predictor_path)
      selfie_img = cv2.cvtColor(selfie_img, cv2.COLOR_RGB2BGR)
      dets = detector(selfie_img, 1)
      msg = None
      print("Number of faces detected: {}".format(len(dets)))
      if len(dets) == 1:
      
            for k, d in enumerate(dets):
                  cr_points_ktp = (d.left() , d.top() , d.right() , d.bottom() )
                  # Get the landmarks/parts for the face in box d.
                  shape = predictor(selfie_img, d)
                  print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                            shape.part(1)))
                  print('Selfie validation successful')
                                    
                  global selfiescore
                  global selfieresult
                  selfie_face = selfie_img.copy()
                  selfie_face = cv2.cvtColor(selfie_face, cv2.COLOR_RGB2BGR)
                  selfie_face = Image.fromarray(selfie_face)
                  selfie_face = selfie_face.crop(cr_points_ktp)
                  dets, scores, idx = detector.run(selfie_img, 1, -1)
                  selfiescore= scores[0]
                  if selfiescore > 1:
                        selfieresult = 'PASS'
                        selfie_msg = 'Selfie passes the Threshold check'
                  else:
                        selfieresult = 'FAIL'
                        selfie_msg = 'Selfie fails the Threshold check'

                               
      elif len(dets) == 0:
            print('No face detected,please reupload selfie picture')
            selfie_msg = 'No face detected,please reupload selfie picture'
            return None,"FAIL",selfie_msg
      else:
            print('More than one face detected in the selfie, please upload only single face selfie image')
            selfie_msg = 'More than one face detected in the selfie, please upload only single face selfie image'
            return None,"FAIL",selfie_msg
    
      return selfiescore,selfieresult,selfie_msg

def ktpimageproc(image):
      image = np.rot90(image)
      print('Image rotation successful')
      image_expanded = np.expand_dims(image, axis=0)
      print(image_expanded.shape)
      (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded}
            )
      offset_height = offset_width = target_height = target_width = 20
      im_height, im_width, channels = image.shape

      try:
            coordinates = vis_util.return_coordinates(
                                    image,
                                    np.squeeze(boxes),
                                    np.squeeze(classes).astype(np.int32),
                                    np.squeeze(scores),
                                    category_index,
                                    use_normalized_coordinates=True,
                                    line_thickness=8,
                                    min_score_thresh=0.60
                              )
            print(coordinates[0][4])
      except:
            #No KTP Card found. Please reupload image
            print('No KTP Card found. Please reupload image')
            ktpresult = "FAIL"
            ktpscore = None
            ktp_msg = 'No KTP Card found. Please reupload image'
            return ktpscore,ktpresult,ktp_msg
      
      if coordinates[0][4] > 90:
            ymin = coordinates[0][0]
            ymax = coordinates[0][1]
            xmin = coordinates[0][2]
            xmax = coordinates[0][3]
            print('Box detected Accurately')
      else:
            #Box detection accuracy not qualified, Please reupload image
            print('Box detection accuracy not qualified, Please reupload image')
            ktpresult = "FAIL"
            ktpscore = None
            ktp_msg = 'Box detection accuracy not qualified, Please reupload image'
            return ktpscore,ktpresult,ktp_msg
      img2 = Image.fromarray(image)
      global cropped_image
      cropped_image = img2.crop((xmin, ymin, xmax, ymax))
      cropped_image = np.array(cropped_image)
      cv2.imwrite("test_images/ktp_crop/cropped_ktp.jpg", cropped_image)
      #DETECTING FACE FROM THE KTP AND CROPPING
      #Loading 194 face landmarks trained data file into predictor path
      predictor_path = cfg.predictor_path
      detector = dlib.get_frontal_face_detector()
      predictor = dlib.shape_predictor(predictor_path)
      dets = detector(cropped_image, 1)
      print("Number of faces detected: {}".format(len(dets)))
      if len(dets) == 1:
            for k, d in enumerate(dets):
                  cr_points_ktp = (d.left() , d.top() , d.right() , d.bottom() )
                  # Get the landmarks/parts for the face in box d.
                  shape = predictor(cropped_image, d)
                  print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                            shape.part(1)))
                  print('Success, Only one face detected')
                  ktp_face = cropped_image.copy()
                  ktp_face = cv2.cvtColor(ktp_face, cv2.COLOR_RGB2BGR)
                  ktp_face = Image.fromarray(ktp_face)
                  ktp_face = ktp_face.crop(cr_points_ktp)
                  ktp_face_img = np.array(ktp_face)
                  cv2.imwrite("test_images/ktp_face/ktp_face.jpg", ktp_face_img)
                  dets, scores, idx = detector.run(cropped_image, 1, -1)
                  # np.savetxt('scores.txt', scores[0], delimiter=',')
                  for i, d in enumerate(dets):
                        print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))
                  # if scores[0] > 0.05:
                  #       print('Success, Face cropped and Image qualifies the threshold value')
                  # else:
                  #       print('Please reupload, Image doesnt qualify the threshold value')
                  # f = open("scores.txt","a+")
                  # for i in range(1):
                  #       f.write(IMAGE_NAME + ","+ str(scores[0]) + "\n" )
                  #       f.close()
                  ktpscore = scores[0]
                  if ktpscore > 1:
                        ktpresult = 'PASS'
                        print('Success, Face cropped and Image qualifies the threshold value')
                        ktp_msg = 'Image qualifies the threshold value'
                        return ktpscore,ktpresult,ktp_msg
                  else:
                        ktpresult = 'FAIL'
                        print('KTP Image is not clear. Please reupload')
                        ktp_msg = 'KTP Image is not clear. Please reupload'
                        return None,"FAIL",ktp_msg
      elif len(dets) == 0:
            print('No face detected,please reupload KTP image')
            ktp_msg = 'No face detected,please reupload KTP image'
            return None,"FAIL",ktp_msg
      else:
            print('More than one face detected in the image, please upload only KTP image')
            ktp_msg='More than one face detected in the image, please upload only KTP image'
            return None,"FAIL",ktp_msg
      
def facematching(ktp_face_img,selfie_face_img):
      image1_encoding = fr.face_encodings(ktp_face_img)
      image2_encoding = fr.face_encodings(selfie_face_img)
      if len(selfie_face_img)> 0:
            image1_encoding = selfie_face_img[0]
      else:
            print('No selfie face found')
      if len(ktp_face_img) > 0:
            image2_encoding = ktp_face_img[0]
      else:
            print('No ktp face found')
      results = fr.compare_faces([image1_encoding], image2_encoding)
      print('FACEMATCHING:')
      print(results)

      if results[0] == True:
            print("Face Matching with the KTP Successful")
      else:
            print("Face does not match with the KTP face")

#TEXT EXTRACTION USING OCR
def ktptextextract(cropped_image):
      gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

      ## (2) Threshold
      th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)

      ## (3) Detect
      result = pytesseract.image_to_string((threshed), lang="ind")

      ## (5) Normalize
      for word in result.split("\n"):
            if "”—" in word:
                  word = word.replace("”—", ":")
      
      #normalize NIK
      if "NIK" in word:
            nik_char = word.split()
      if "D" in word:
            word = word.replace("D", "0")
      if "?" in word:
            word = word.replace("?", "7") 
      if ":" in word:
            word = word.replace(":", "")
      print("Tesseract output from KTP:")  
      ktpnum = re.findall(r'\d+',result)
      NIK = max(ktpnum,key=len)
      res = list(map(int, ktpnum))
      return NIK
      # window_name = 'Cropped'
      # cv2.imshow(window_name,ktp_face_img)
      # cv2.waitKey()
      



