import cv2 as cv
import time
import numpy as np
import sys
cap = cv.VideoCapture('train.mp4')
length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
outb = cv.VideoWriter('output_train_b.avi',fourcc, 20.0, (640,480))
outg = cv.VideoWriter('output_train_g.avi',fourcc, 20.0, (640,480))
outr = cv.VideoWriter('output_train_r.avi',fourcc, 20.0, (640,480))

frame_count=0
while(cap.isOpened()):
	_,prev = cap.read()
	# cv.imshow('sample',prev[0][0][:])
	b,g,r = cv.split(prev)

	# hsv = np.zeros_like(prev)
	hsvb = np.zeros_like(prev)
	hsvg = np.zeros_like(prev)
	hsvr = np.zeros_like(prev)
	hsvb[..., 1] = 255
	hsvg[..., 1] = 255
	hsvr[..., 1] = 255

	# prev = cv.cvtColor(prev,cv.COLOR_BGR2GRAY)
	prevb,prevg,prevr = b,g,r
	frame_count = frame_count+1
	break

while(cap.isOpened()):
	_,next = cap.read()
	nextb,nextg,nextr = cv.split(next)
	frame_count = frame_count+1

	print('[',frame_count,'/',length,']')
	if frame_count>length:
		break
	# next = cv.cvtColor(next,cv.COLOR_BGR2GRAY)

	flowb = cv.calcOpticalFlowFarneback(prevb,nextb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	flowg = cv.calcOpticalFlowFarneback(prevg,nextg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	flowr = cv.calcOpticalFlowFarneback(prevr,nextr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	magb, angb = cv.cartToPolar(flowb[...,0], flowb[...,1])
	magg, angg = cv.cartToPolar(flowg[...,0], flowg[...,1])
	magr, angr = cv.cartToPolar(flowr[...,0], flowr[...,1])

	hsvb[...,0] = angb*180/np.pi/2
	hsvb[...,2] = cv.normalize(magb,None,0,255,cv.NORM_MINMAX)
	flowb = cv.cvtColor(hsvb,cv.COLOR_HSV2BGR)

	hsvg[..., 0] = angg * 180 / np.pi / 2
	hsvg[..., 2] = cv.normalize(magg, None, 0, 255, cv.NORM_MINMAX)
	flowg = cv.cvtColor(hsvg,cv.COLOR_HSV2BGR)

	hsvr[..., 0] = angr * 180 / np.pi / 2
	hsvr[..., 2] = cv.normalize(magr, None, 0, 255, cv.NORM_MINMAX)
	flowr = cv.cvtColor(hsvr,cv.COLOR_HSV2BGR)

	outb.write(flowb)
	outg.write(flowg)
	outr.write(flowr)

	prevb = nextb
	prevg = nextg
	prevr = nextr

cap.release()
outb.release()
outg.release()
outr.release()