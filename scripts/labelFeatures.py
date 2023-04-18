import cv2 as cv
import numpy as np

import datetime
import os

## Test ##
# print(str(datetime.datetime.now().utcnow()).replace(" ", "_"))
# print("videos/drone.mp4".split('/')[-1].strip('.mp4'))
##########


class SIFTLabeler:
    def __init__(self, working_dir: str = '.'):

        self.extractor = cv.SIFT_create()
        self.working_dir = working_dir

        pass

    def labelFromVideo(self, path: str = '/dev/video1'):

        # Create a label for the images used in matching
        label = ""

        if path == '/dev/video1':
            # Make label a lamda function
            label = lambda _ : str(datetime.datetime.now().utcnow()).replace(" ", "_")
        else:
            label = path.strip('/')[-1].strip('.mp4')[0]

        # Open the video stream
        cap = cv.VideoCapture(path)

        if cap.isOpened() == False:
            print("Video stream is not opened")

        # Buffer to store the images
        images = []

        # Sequence number to save the images
        seq = 0

        while cap.isOpened():

            # Fill the buffer with 2 images
            if len(images) == 0:
                # Image grab
                frame = self.__grab_image__(cap, images, 2)
                if frame != 0:
                    print('Error in capturing frames')
                    return

            # Match and label features
            self.matchFeatures(images[0], images[1])

        # Release the vid capture
        cap.release()

        # Close all frames
        cv.destroyAllWindows()

    def __grab_image__(self, stream, buffer, num_caps=1):

        itr = 0
        while itr < num_caps:

            # Capture frame
            ret, frame = stream.read()

            if ret == True:
                # Skip the next 4 images
                for _ in range(4):
                    _, _ = stream.read()

                buffer.append(frame)
                return 0
            else:
                return -1

    def matchFeatures(self, img1, img2):

        # Images turned to grayscale
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # Extract features of both images
        kp1, des1 = self.extractor.detectAndCompute(img1, None)
        kp2, des2 = self.extractor.detectAndCompute(img2, None)


    def extractFeatures(self, img):
        pass


# cap = cv.VideoCapture('videos/drone.mp4')

# Loop through mp4
