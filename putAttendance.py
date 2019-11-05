# USAGE
# python putAttendance.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle --image received/

# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import time
import datetime
from pymongo import MongoClient

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
#print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

client = MongoClient(port=27017)
db = client.record

# load our serialized face embedding model from disk
#print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
#image = cv2.imread(args["image"])

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
d = datetime.date.today().timetuple()
date = str(d[2]) + ":"  + str(d[1]) + ":" + str(d[0])
day = days[d[6]]
directory = args["image"]

def getPeriod() :
	now = datetime.datetime.now()
	p = now.replace(hour = 9, minute = 25)
	if now < p :
		return 1
	p = now.replace(hour = 10, minute = 15)
	if now < p :
		return 2
	p = now.replace(hour = 11, minute = 25)
	if now < p :
		return 3
	p = now.replace(hour = 12, minute = 15)
	if now < p :
		return 4
	p = now.replace(hour = 14, minute = 0)
	if now < p :
		return 5
	p = now.replace(hour = 14, minute = 50)
	if now < p :
		return 6
	p = now.replace(hour = 15, minute = 50)
	if now < p :
		return 7
	p = now.replace(hour = 16, minute = 45)
	if now < p :
		return 8

period = getPeriod()

attendees = {}
for per in le.classes_ :
	attendees[per] = []

# loop over the detections
for i in range(0, 100):
	time.sleep(.4)
	try:
		image = cv2.imread(directory +str(i) + ".png")
	except:
		continue
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]
	
	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()
	
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
	
		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the
			# face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	
			# extract the face ROI
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
	
			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue
	
			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
				(0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()
	
			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]
			#print(name)
			timestamp = time.strftime('%H:%M:%S')
			attendees[name].append([timestamp])
			#print(result)
			
	
			# draw the bounding box of the face along with the associated
			# probability
			#text = "{}: {:.2f}%".format(name, proba * 100)
			#y = startY - 10 if startY - 10 > 10 else startY + 10
			#cv2.rectangle(image, (startX, startY), (endX, endY),
			#	(0, 0, 255), 2)
			#cv2.putText(image, text, (startX, y),
			#	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	
# show the output image
	#cv2.imshow("Image", image)
	#cv2.waitKey(0)

for key, value in attendees.iteritems():
	#print key, value
	if len(value) > 5 :
		record = { 'name': key, 'time' : value[0][0], 'date' : date, 'day' : day, 'period'  : period}
		result = db.attendance.insert_one(record)
		
print("\nAttendance Updated");


