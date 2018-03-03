#! /usr/bin/env python3

from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import numpy as np
import time
import cv2 as cv

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-g', '--graph', required=True, help='Path to input graph file')
    ap.add_argument('-c', '--confidence', default=.5, help='Confidence threshold')
    ap.add_argument('-d', '--display', type=int, default=0,
                    help='Switch to display image on screen')
    ap.add_argument('-s', '--size', type=int, default=300,
                    help='Size of the image for inference')
    return vars(ap.parse_args())

def main():
    args = parse_args()

    print("Starting the video stream...")
    vs = VideoStream().start()
    time.sleep(1)

    print("Starting the frame loop...")
    fps = frame_loop(args, vs)

    # Destroy all windows if we are displaying them and stop the video stream
    if args["display"] > 0:
        cv.destroyAllWindows()
    vs.stop()

    # Display FPS information
    print("Elapsed time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS:  {:.2f}".format(fps.fps()))

def frame_loop(args, vs):
    fps = FPS().start()

    while True:
        try:
            frame = vs.read()
            image = frame.copy()

            # Display the frame to the screen when asked
            if args["display"] > 0:
                cv.imshow("Output", image)

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

if __name__ == "__main__":
    main()
