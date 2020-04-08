"""
Saves a series of snapshots with the current camera as snapshot_<width>_<height>_<nnn>.jpg

Arguments:
    --f <output folder>     default: current folder
    --n <file name>         default: snapshot
    --w <width px>          default: none
    --h <height px>         default: none

Buttons:
    q           - quit
    space bar   - save the snapshot
    
  
"""

import cv2
import time
import sys
import argparse
import os
import imutils
import numpy as np

__author__ = "Tiziano Fiorenzani"
__date__ = "01/06/2018"


def save_snaps(width=0, height=0, name="snapshot", folder=".", raspi=False):
    kernel=np.array([[-1,-1,-1],[-1, 9,-1],[-1,-1,-1]])
    if raspi:
        os.system('sudo modprobe bcm2835-v4l2')

    cap = cv2.VideoCapture(0)
    if width > 0 and height > 0:
        print("Setting the custom Width and Height")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
            # ----------- CREATE THE FOLDER -----------------
            folder = os.path.dirname(folder)
            try:
                os.stat(folder)
            except:
                os.mkdir(folder)
    except:
        pass

    nSnap   = 0
    w       = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h       = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fileName    = "%s/%s_%d_%d_" %(folder, name, w, h)
    	
    while True:
        ret, frame = cap.read()
        rotate=imutils.rotate(frame,0)
        cv2.imshow('camera', rotate)
        gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpened=cv2.filter2D((rotate),-1,kernel)
        gray_sharp=cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        cv2.imshow('sharpened',gray_frame)
        cv2.imshow('sharpened1',gray_sharp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            print("Saving image ", nSnap)
            cv2.imwrite("%s%d.jpg"%(fileName, nSnap), rotate)
            nSnap += 1
    cap.release()
    cv2.destroyAllWindows()




def main():
    # ---- DEFAULT VALUES ---
    SAVE_FOLDER = "."
    FILE_NAME = "snapshot"
    FRAME_WIDTH = 0
    FRAME_HEIGHT = 0

    # ----------- PARSE THE INPUTS -----------------
    parser = argparse.ArgumentParser(
        description="Saves snapshot from the camera. \n q to quit \n spacebar to save the snapshot")
    parser.add_argument("--folder", default=SAVE_FOLDER, help="Path to the save folder (default: current)")
    parser.add_argument("--name", default=FILE_NAME, help="Picture file name (default: snapshot)")
    parser.add_argument("--dwidth", default=FRAME_WIDTH, type=int, help="<width> px (default the camera output)")
    parser.add_argument("--dheight", default=FRAME_HEIGHT, type=int, help="<height> px (default the camera output)")
    parser.add_argument("--raspi", default=False, type=bool, help="<bool> True if using a raspberry Pi")
    args = parser.parse_args()

    SAVE_FOLDER = args.folder
    FILE_NAME = args.name
    FRAME_WIDTH = args.dwidth
    FRAME_HEIGHT = args.dheight


    frame=save_snaps(width=args.dwidth, height=args.dheight, name=args.name, folder=args.folder, raspi=args.raspi)
	#sharpened=cv2.filter2D(imutils.rotate(frame,180),-1,kernel)
    #cv2.waitKey(0)
    #cv2.imshow('sharpened',sharpened)
    #print("Files saved")

if __name__ == "__main__":
    main()



