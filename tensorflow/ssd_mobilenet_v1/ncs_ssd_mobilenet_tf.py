#! /usr/bin/env python3

import argparse, time
import numpy as np
import cv2 as cv

from imutils.video import VideoStream
from imutils.video import FPS
from mvnc import mvncapi as mvnc

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-g', '--graph', required=True, help='Path to input graph file')
    ap.add_argument('-c', '--confidence', default=.5, help='Confidence threshold')
    ap.add_argument('-d', '--display', type=int, default=0,
                    help='Switch to display image on screen')
    ap.add_argument('-s', '--size', type=int, default=300,
                    help='Size of the image for inference')
    return vars(ap.parse_args())

def open_ncs_device():
    devices = mvnc.EnumerateDevices()
    if len(devices) is 0:
        print("No NCS devices found, exiting...")
        quit()

    print("Found {} device(s), only device[0] will be used".format(len(devices)))
    device = mvnc.Device(devices[0])
    device.OpenDevice()

    return device

def display_frame(frame):
    name = 'Camera Output'
    cv.namedWindow(name)
    cv.moveWindow(name, 20, 20)
    cv.imshow(name, frame)

def frame_loop(args, vs):
    fps = FPS().start()

    while True:
        try:
            frame = vs.read()
            image = frame.copy()

            # Display the frame to the screen when asked
            if args["display"] > 0:
                display_frame(image)

                # if the 'q' key was pressed, break from the loop
                key = cv.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            # update the FPS counter
            fps.update()

        # If "ctrl+c" is pressed in the terminal, break from the loop
        except KeyboardInterrupt:
            print('CTRL+C key pressed, exiting...')
            break

        # If there's a problem reading a frame, break gracefully
        except AttributeError:
            print('Could not acquire a new frame, exiting...')
            break

    fps.stop()

    return fps

def main():
    args = parse_args()

    print("Starting the video stream...")
    vs = VideoStream().start()
    time.sleep(1)

    print('Opening NCS device....')
    device = open_ncs_device()

    print("Starting the frame loop...")
    fps = frame_loop(args, vs)

    # Destroy all windows if we are displaying them and stop the video stream
    if args["display"] > 0:
        cv.destroyAllWindows()
    vs.stop()

    # Release device and graph
    device.CloseDevice()

    # Display FPS information
    print("Elapsed time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS:  {:.2f}".format(fps.fps()))

if __name__ == "__main__":
    main()
