import cv2 as cv
import time
import numpy as np
import sys
cap = cv.VideoCapture('train.mp4')
length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output_train.avi',fourcc, 20.0, (640,480))
frame_count=0
while(cap.isOpened()):
	_,prev = cap.read()
	hsv = np.zeros_like(prev)
	hsv[...,1] = 255
	prev = cv.cvtColor(prev,cv.COLOR_BGR2GRAY)
	frame_count = frame_count+1
	break
while(cap.isOpened()):
	_,next = cap.read()
	frame_count = frame_count+1
	print('[',frame_count,'/',length,']')
	if frame_count>length:
		print('reached end of video file... exiting....')
		break
	next = cv.cvtColor(next,cv.COLOR_BGR2GRAY)
	flow = cv.calcOpticalFlowFarneback(prev,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
	flow = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
	out.write(flow)

	prev = next
cap.release()
out.release()
cv.destroyAllWindows()
