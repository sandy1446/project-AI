# USAGE
# python yolo_1.py --input ../example_videos/janie.mp4 --output ../output_videos/yolo_janie.avi --yolo yolo-coco --display 0
# python yolo_1.py --input /home/sandy/Desktop/project/opencv-dnn-gpu-examples/example_videos/s2.mp4 --output yolo_sample2.avi --display 1 --use-gpu 1

# import the necessary packages
from imutils.video import FPS
import numpy as np
import pandas as pd
import argparse
import cv2
import os
import time
from os import system, name
import json
import csv
from datetime import datetime
from firebase import firebase

firebase=firebase.FirebaseApplication("https://quickstart-1598859311837.firebaseio.com/",None)

def clear():

    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')


def Scale(xinput):
	f = open("/home/sandy/Desktop/project/S mean.txt", 'r')
	# print("a")
	all_lines = [[float(num) for num in line.split()] for line in f]
	x, s = zip(*all_lines)
	# print(x)
	x = sorted(x)
	y = sorted(s, reverse=True)
	if xinput <= x[0]:
		x1 = x[0]
		x2 = x[1]
		y1 = y[0]
		y2 = y[1]
	else:
		for i in range(len(x)):
			try:
				y1[i+1]
			except:
				x1 = x[i-1]
				y1 = y[i-1]
				x2 = x[i]
				y2 = y[i]
				break
			if xinput >= x1[i] and xinput <= x1[i+1]:
				x1 = x[i]
				y1 = y[i]
				x2 = x[i+1]
				y2 = y[i+1]
				break
	c = xinput
	d = (c-x1)*(y2-y1)/(x2-x1)+y1
	'''x1=x[0]
	y1=y[0]
	x2=x[len(x)-1]
	y2=y[len(y)-1]
	c=xinput
	print(c)

	d=(c-x1)*(y2-y1)/(x2-x1)+y1
	print(d)'''
	return(d)
	f.close()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="1.mp4",
                help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="output_videos/yolo_vec.avi",
                help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not output frame should be displayed")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applyong non-maxima suppression")
ap.add_argument("-u", "--use-gpu", type=bool, default=1,
                help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
LABELS = open("classes.names").read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(
    "darknet-yolov3.cfg", "darknet-yolov3_final.weights")

# check if we are going to use GPU
if args["use_gpu"]:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the width and height of the frames in the video file
W = None
H = None

# initialize the video stream and pointer to output video file, then
# start the FPS timer
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
#f = open("boundary.txt", 'w')
#w_f = "SN;Class IDs;X coordinate;Y coordinate;Time\n"
#f.write(w_f)
cx = []
cy = []
cm = []
m = 0
k = 0
W_XYZ = []
detail = []

# loop over frames from the video file stream
while True:
	
	fps = FPS().start()
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
								 swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	count_person = 0
	count_car = 0
	count_bus = 0
	count_truck = 0

    # loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				cx.append(centerX)
				cy.append(centerY)

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				m += 1
	cm.append(m)
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
							args["threshold"])
	font = cv2.FONT_HERSHEY_SIMPLEX

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
									   confidences[i])
			cv2.putText(frame, text, (x, y - 5),
						cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
			if LABELS[classIDs[i]] == "car":
				count_car += 1
			if LABELS[classIDs[i]] == "truck" or LABELS[classIDs[i]] == "bus":
				count_truck += 1
			if LABELS[classIDs[i]] == "person" or LABELS[classIDs[i]] == "motorbike":
				count_person += 1
			cv2.line(frame, (cx[i], cy[i]),
					 (cx[i]+1, cy[i]+1), (0, 255, 255), 1)

	tot_car = "car : "+str(count_car)
	tot_truck = "truck : "+str(count_truck)
	tot_person = "Motorbike : "+str(count_person)
	cv2.putText(frame, "Counting System:", (W-400, 20),
				font, 0.75, (0, 0, 255), 1)
	if count_car > 0:
		cv2.putText(frame, tot_car, (W-400, 50), font, 0.5, (0, 0, 255), 1)
	if count_truck > 0:
		cv2.putText(frame, tot_truck, (W-400, 80), font, 0.5, (0, 0, 255), 1)
	if count_person > 0:
		cv2.putText(frame, tot_person, (W-400, 110), font, 0.5, (0, 0, 255), 1)

    # check to see if the output frame should be displayed to our
    # screen

    # if an output video file path has been supplied and the video
    # writer has not been initialized, do so now

	if len(idxs) > 0:
		workingFolder = "/home/sandy/Desktop/project/data1"
		s_describe = np.loadtxt(workingFolder+'/s mean.txt', delimiter=',')
		s_mean = np.mean(s_describe)
		cameramxt = np.loadtxt(
			workingFolder+'/Camera Matrix.txt', delimiter=',')
		inverse_cameramxt = np.linalg.inv(cameramxt)
		# print(inverse_cameramxt)
		translation_vector = np.loadtxt(
			workingFolder+'/Translation vector.txt', delimiter=',')
		tvec = np.array(translation_vector, dtype=np.float32).reshape((3, 1))
		# print("\n",tvec)
		inverse_R_mtx = np.loadtxt(
			workingFolder+'/Inverse R.txt', delimiter=',')
		now = datetime.now()
        # print(now.microsecond)
		C_time = now.strftime("%H:%M:%S") + \
			str(now.microsecond*1e-6).replace('0.', '.')
		for i in range(len(boxes)):
			if LABELS[classIDs[i]] in ("Car", "truck", "Bus", "person", "motorbike"):
				s = Scale(float(cy[i]))
				uv_1 = np.array([[cx[i], cy[i], 1]], dtype=np.float32)
				uv_1 = uv_1.T
				print("1 : ", uv_1)
				suv_1 = np.array(
					[s*uv_1[0], s*uv_1[1], s*uv_1[2]], dtype=np.float32)
				print("\n2 : ", suv_1)
				xyz_c = inverse_cameramxt.dot(suv_1)
				# print(xyz_c)
				xyz_c = xyz_c-tvec
				# print("\n",xyz_c[0])
				XYZ = inverse_R_mtx.dot(xyz_c)
                # print("\n",xyz_c)
				XYZ = np.array(XYZ, dtype="float")
				text = str(round(float(XYZ[0]), 5)) +","+str(round(float(XYZ[1]), 5))
				data=(
					str(i+1),
					LABELS[classIDs[i]],
					float(XYZ[0]),
					float(XYZ[1]),
					C_time
					)
				detail+=data
				#result=firebase.post('/quickstart-1598859311837/Realtime' , detail)
				#print(result)
				#detail = str(i+1)+";"+LABELS[classIDs[i]]+";"+str(
					#float(XYZ[0]))+","+str(float(XYZ[1]))+";"+C_time+"\n"
				#f.write(detail)
				# print(detail)
				cv2.putText(frame, text, (cx[i], cy[i]),
							font, 0.25, (255, 255, 255), 1)
				cv2.putText(frame, str(len(boxes)+1), (W-400, 130),
							font, 0.5, (0, 0, 255), 1)
                # time.sleep()
                # cv2.waitKey(0)
		k += 1
		print(k)
		if k%50==0:
			result=firebase.post('/quickstart-1598859311837/Realtime' , detail)
			print(result)
			detail=[]
		clear()
	# update the FPS counter
	fps.update()
	fps.stop()
	s = 'FPS = '+str("{:.2f}".format(fps.fps()))
	bottomLeftCornerOfText = (5, 20)
	fontScale = 0.5
	fontColor = (120, 200, 100)
	lineType = 1
	cv2.putText(frame, s, bottomLeftCornerOfText,
				font, fontScale, fontColor, lineType)
	cx.clear()
	cy.clear()
	W_XYZ.clear()

	if args["display"] > 0:
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			f.close()
			break
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
								 (frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)

#read_file = pd.read_csv(r'boundary.txt')
#read_file.to_csv(r'boundary_1.csv', index=";")
# stop the timer and display FPS information
