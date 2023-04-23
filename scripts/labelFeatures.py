import cv2 as cv
import numpy as np
import random

import datetime
import os

## Test ##
# print(str(datetime.datetime.now().utcnow()).replace(" ", "_"))
# print("videos/drone.mp4".split('/')[-1].strip('.mp4'))
##########

class SIFTLabeler:
    # Constructor
    def __init__(self, working_dir: str = '.'):

        # SIFT Feature Extractor
        self.extractor = cv.SIFT_create()

        # Validate the save path
        if os.path.exists(working_dir):
            print(f'Valid path ({os.path.abspath(working_dir)})')
            # os.path.
            self.working_dir = working_dir

        # Date to hold for learning
        self.x = None
        self.y = None

    def __save_data(self, name):
        np.save(f'{os.path.abspath(self.working_dir)}{name}_features.npy', self.x)
        np.save(f'{os.path.abspath(self.working_dir)}{name}_validation.npy', self.y)

    def labelFromVideo(self, path: str = '/dev/video1'):

        # Create a label for the images used in matching
        label = ""

        if path == '/dev/video1':
            # Make label a lamda function
            label = lambda _ : str(datetime.datetime.now().utcnow()).replace(" ", "_")
        else:
            label = path.split('/')[-1].strip('.mp4')

        print(f'Prefix: {label}')
        print(f'{self.working_dir}')
        print(f'save path {os.path.abspath(self.working_dir)}')
        exit(0)
        # Open the video stream
        cap = cv.VideoCapture(path)

        if cap.isOpened() == False:
            print("Video stream is not opened")

        # Buffer to store the images
        images = []

        # Sequence number to save the images
        seq = 0
        itr = 0
        while cap.isOpened():

            if itr == 2:
                break

            print(f'Processing images pairs sequence {seq}')

            # Fill the buffer with 2 images
            if len(images) == 0:
                # Image grab
                ret = self.__grab_image__(cap, images, 2)
                if ret != 0:
                    print('Error in capturing frames')
                    return
            else:
                ret = self.__grab_image__(cap, images)
                if ret != 0:
                    print('Error in capturing frames')
                    return

            # Match and label features
            self.matchFeatures(images[0], images[1])
            images.pop(0)

            seq+=1
            itr+=1

        # Release the vid capture
        cap.release()

        # Close all frames
        cv.destroyAllWindows()

        # Save the data
        self.__save_data(label)

    def __grab_image__(self, stream, buffer, num_caps=1):

        itr = 0
        while itr < num_caps:

            # Capture frame
            ret, frame = stream.read()

            if ret == True:
                # Skip the next 4 images
                for _ in range(2):
                    _, _ = stream.read()

                buffer.append(frame)
                itr+=1
                # return 0
            else:
                return -1

        return 0

    def matchFeatures(self, img1, img2):

        # Images turned to grayscale
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # Extract features of both images
        print('Extracting features... ')
        kp1, des1 = self.extractor.detectAndCompute(img1, None)
        kp2, des2 = self.extractor.detectAndCompute(img2, None)
        print('Extraction finsihed!')

        # Use the flann based matcher to make training date
        print('Matching features... ')

        # Set up the keypoint matcher.
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply ratio test to filter out ambiguous matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        print('Finished matching!')

        # Create the feature and validation datasets
        x = np.array([])
        y = np.array([])
        for m in good_matches:
            # save the positive match
            np.append(x, des1[m.queryIdx] + des2[m.trainIdx])
            np.append(y, 1.0)

        # Get the descriptors that don't have any matches
        unmatched_des1 = []
        unmatched_des2 = []
        for i, d in enumerate(des1):
            matched = False
            for match in good_matches:
                if i == match.queryIdx:
                    matched = True
                    break
            if not matched:
                unmatched_des1.append(d)

        for i, d in enumerate(des2):
            matched = False
            for match in good_matches:
                if i == match.trainIdx:
                    matched = True
                    break
            if not matched:
                unmatched_des2.append(d)


        # Take 50 random samples from the unmatched descriptors and
        #   label them as false
        if len(unmatched_des1) >= 50 and len(unmatched_des2) >= 50:
            unmatched_des1 = np.array(random.sample(unmatched_des1, 50))
            unmatched_des2 = np.array(random.sample(unmatched_des2, 50))
        else:
            if len(unmatched_des1) > len(unmatched_des2):
                unmatched_des1 = np.array(unmatched_des1[:len(unmatched_des2)])
                unmatched_des2 = np.array(unmatched_des2)
            else:
                unmatched_des1 = np.array(unmatched_des1)
                unmatched_des2 = np.array(unmatched_des2[:len(unmatched_des1)])

        for i, _ in enumerate(unmatched_des1):
            np.append(x, unmatched_des1[i] + unmatched_des2[i])
            np.append(y, 0.0)

        if isinstance(self.x, np.ndarray):
            np.append(self.x, x)
        else:
            self.x = x

        if isinstance(self.y, np.ndarray):
            np.append(self.y, y)
        else:
            self.y = x

        return (x, y)

if __name__ == '__main__':

    lblr = SIFTLabeler('../imgs/')
    lblr.labelFromVideo('../videos/drone.mp4')
