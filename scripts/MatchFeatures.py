import os
import torch
import cv2 as cv
import numpy as np

from SIFTmodel import SIFTnn

import argparse
import time

def loadModel(path: str='') -> torch.nn:
    if len(path) == 0:
        print('Please provide a path to the model weights (*.pth)')
    model = SIFTnn()
    model.load_state_dict(torch.load(os.path.abspath(path)))
    model.eval()
    return model

def extractAndMatch(img1, img2, model):

    # Load imgs or video stream
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Extract features
    kp1, des1 = extractor.detectAndCompute(gray1, None)
    kp2, des2 = extractor.detectAndCompute(gray2, None)

    # matches = sorted(matches, key=lambda val: val.distance)

    # Create a list of tuples containing the keypoint, its descriptor, and its response value
    kp_des_1 = [(kp, desc, kp.response) for kp, desc in zip(kp1, des1)]
    kp_des_2 = [(kp, desc, kp.response) for kp, desc in zip(kp2, des2)]

    # Sort the keypoints by their response value in descending order
    sorted_features1 = sorted(kp_des_1, key=lambda x: x[2], reverse=True)
    sorted_features2 = sorted(kp_des_2, key=lambda x: x[2], reverse=True)

    # Num features to use
    num_feat = 50
    sorted_features1 = sorted_features1[:num_feat]
    sorted_features2 = sorted_features2[:num_feat]

    # Reusing variables
    kp1 = []
    kp2 = []
    des1 = []
    des2 = []

    for kp, des, _ in sorted_features1:
        kp1.append(kp)
        des1.append(des)
    for kp, des, _ in sorted_features2:
        kp2.append(kp)
        des2.append(des)

    # create BFMatcher object
    bf_start = time.time()
    bf = cv.BFMatcher()

    # Match descriptors
    matches = bf.match(np.array(des1),np.array(des2))
    bf_end = time.time()

    # Metric testing
    model_times = []


    # Use the model to match features
    # 0(n^2) = infinite loop
    model.double()
    th = 0.8
    itr = 0
    matches = []

    model_match_start = time.time()
    for queryIdx, d1 in enumerate(des1):
        match = None
        val = None
        for trainIdx, d2 in enumerate(des2):
            # Time the model
            model_start = time.time()
            # Use the model
            output = model(torch.from_numpy(np.array(d1)), torch.from_numpy(np.array(d2)))
            model_end = time.time()
            model_times.append(model_end - model_start)


            if output[0] > th:
                # Calculate the distance between the two descriptors
                dist = np.linalg.norm(d1 - d2)
                if val == None: # no match
                    val = output[0]
                    match = cv.DMatch(queryIdx, trainIdx, dist)
                else:
                    if output[0] > val: # better match
                        val = output[0]
                        match = cv.DMatch(queryIdx, trainIdx, dist)

        if match != None:
            matches.append(match)

    model_match_end = time.time()

    # Feature parameters
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

    # Show the image
    cv.namedWindow('name', cv.WINDOW_NORMAL)
    img3 = cv.resize(img3, (960,540))
    cv.imshow('name', img3)

    # Finished
    cv.waitKey(0)
    cv.destroyAllWindows()

    model_time_avg = sum(model_times)/len(model_times)

    # Print metrics
    print('-'*15 + f'Metrics for the top {num_feat} feature descriptors' + '-'*15)
    print(f'(Model) Average time to run model : \t\t{model_time_avg}s')
    # print(f'(Model) Time to run model and match pairs : \t{model_match_end - model_match_start}s')
    print("")
    print(f'(BF) Time to run brute force matcher : \t\t{bf_end - bf_start}s')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', nargs='+', default=[], help="Path of 2 images to match")
    parser.add_argument('-m', '--model', help="Path to model(.pth) file")
    args = parser.parse_args()

    if len(args.images) != 2:
        print('Two relative paths to an image must be provided')
        parser.print_help()

    if args.model == None:
        print('Relative path to model required')
        parser.print_help()

    # openCV objects
    extractor = cv.SIFT_create()

    # Load model
    model = loadModel(args.model)
    img1 = cv.imread(args.images[0])
    img2 = cv.imread(args.images[1])

    extractAndMatch(img1, img2, model)



