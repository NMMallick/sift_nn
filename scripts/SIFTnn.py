import os

import cv2 as cv
import torch
import torch.nn as nn
import numpy as np

from SIFTmodel import SIFTnn


def loadModel(path: str='.') -> torch.nn:
    model = SIFTnn()
    model.load_state_dict(torch.load(os.path.abspath(path)))
    model.eval()
    return model

if __name__ == '__main__':

    # openCV objects
    extractor = cv.SIFT_create()

    # Load model
    model = loadModel('new_model.pth')

    # Load imgs or video stream
    img1 = cv.imread('../imgs/img1.jpg')
    img2 = cv.imread('../imgs/img2.jpg')

    # Extract features
    kp1, des1 = extractor.detectAndCompute(img1, None)
    kp2, des2 = extractor.detectAndCompute(img2, None)

    # Use the model to match features
    # 0(n^2) = infinite loop
    model.double()
    itr = 0
    for d1 in enumerate(des1):
        for d2 in enumerate(des2):
            output = model(torch.from_numpy(d1[1]), torch.from_numpy(d2[1]))
            print(f'd1 : {d1[0]}\td2 : {d2[0]}\tval : {output[0]}')

            input()


    cv.waitKey(0)
    cv.destroyAllWindows()
