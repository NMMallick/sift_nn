import cv2 as cv
import numpy as np
import random

import datetime
import os

import SIFTnn as snn

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader, random_split

class SIFTnnLabeler:
    # Constructor
    def __init__(self, working_dir: str = '.'):

        # If we care to label data and train at the same time
        self.__model__ = None

        # Set up SIFT Feature Extractor
        self.extractor = cv.SIFT_create()

        # Set up flann based matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

        # Validate the save path (TODO) either fuck off with this or fix it
        if os.path.exists(working_dir):
            print(f'Valid path ({os.path.abspath(working_dir)})')
            # os.path.
            self.working_dir = working_dir

        # Date to hold for learning
        self.x = []
        self.y = []

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

            if itr == 20:
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
                # Skip the next 5 images
                for _ in range(5):
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
        _, des1 = self.extractor.detectAndCompute(img1, None)
        _, des2 = self.extractor.detectAndCompute(img2, None)
        print('Extraction finsihed!')

        # Use the flann based matcher to make training data
        print('Matching features... ')
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Apply ratio test to filter out ambiguous matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        print('Finished matching!')

        des1 = des1.tolist()
        des2 = des2.tolist()

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

        # Create the feature and validation datasets
        num_pairs = 0
        for m in good_matches:
            # save the positive match
            self.x.append(des1[m.queryIdx] + des2[m.trainIdx])
            self.y.append(1.0)

            for i, d2 in enumerate(des2):
                if i == m.trainIdx:
                    continue
                self.x.append(des1[m.queryIdx] + d2)
                self.y.append(0.0)
                num_pairs += 1

            for i, d1 in enumerate(des1):
                if i == m.queryIdx:
                    continue
                self.x.append(des2[m.trainIdx] + d1)
                self.y.append(0.0)
                num_pairs += 1

        # Take 50 random samples from the unmatched descriptors and
        #   label them as false
        min = len(unmatched_des1) if len(unmatched_des1) < len(unmatched_des2) else len(unmatched_des2)
        unmatched_des1 = random.sample(unmatched_des1, min)
        unmatched_des2 = random.sample(unmatched_des2, min)

        for i, _ in enumerate(unmatched_des1):
            self.x.append(unmatched_des1[i] + unmatched_des2[i])
            self.y.append(0.0)
            num_pairs+=1

        print(f'Number of pairs saved : {num_pairs}/{len(self.x)}')

    def labelAndTrain(self, path, model=None, video=False):

        # Prepare optimizer and criterion and create a model
        self.__model__ = snn.SIFTnn() if model == None else self.__load_model(model)
        optimizer = optim.Adam(self.__model__.parameters(), lr=0.001)
        criterion = nn.BCELoss() # binary cross entropy loss function

        # Open the video stream
        cap = cv.VideoCapture(path) if video else None
        if cap != None and cap.isOpened() == False:
            print("Video stream is not opened")
            exit(1)

        # Buffer to store the images
        images = []

        # Populate the images list if we aren't reading from a video stream
        if not video:
            if len(path) != 2:
                print(f'({self.labelAndTrain.__name__}) path parameter needs a path to two images i.e ["/path/to/img1", "/path/to/img2"]')
                exit(1)
            images.append(cv.imread(path[0]))
            images.append(cv.imread(path[1]))

        # Sequence number to save the images
        seq = 0
        itr = 0
        while len(images) == 2 or video:

            # Fill the buffer with 2 images
            if video:
                if not cap.isOpened(): # kill the loop if vid stream is done
                    break
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

            print(f'Processing images pairs sequence {seq}')

            # Match and label features
            self.matchFeatures(images[0], images[1])

            # Use the x and y data to match the features
            self.__train_model(model=self.__model__,
                               optimizer=optimizer,
                               criterion=criterion)
            images.pop(0)

            seq+=1
            itr+=1

        # Release the vid capture
        cap.release()

        # Close all frames
        cv.destroyAllWindows()

        # Save the model
        self.__save_model(self.__save_model)

    def __train_model(self, **kwargs):

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Create the model, optimizer and loss criteria
        model = kwargs['model']
        optimizer = kwargs['optimizer']
        criterion = kwargs['criterion']

        x = torch.from_numpy(self.x)
        y = torch.from_numpy(self.y)

        dataset = TensorDataset(x, y)

        # Define train and test sizes
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

        # Train model
        epoch = 0
        running_loss = 0.0
        model.double()

        for x_batch, y_batch in train_loader:
            for i, x in enumerate(x_batch):
                # zero the parameter gradients
                optimizer.zero_grad()
                # x.to(device)
                # forward + backward + optimize
                outputs = model(x[:128], x[128:])
                # print(f'pred : {outputs[0]}\temp : {y_batch[i]}')
                loss = criterion(outputs[0], y_batch[i])
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 32 == 31: # print every 100 batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
                epoch+=1

        # evaluate the model on the test data
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                for i, x in enumerate(x_batch):
                    outputs = model(x[:128], x[128:])
                    predicted = (outputs[0] > 0.5).float()
                    total += 1
                    if predicted == y_batch[i]:
                        correct += 1

        print('Accuracy of the network on the %d test descriptor pairs: %d %%' % (total, 100 * correct / total))

        # Save the model (this may be extremely redundant but I don't know any better)
        self.__model__ = model

    def __save_data(self, name):

        # print(self.x)
        x = np.array(self.x)
        y = np.array(self.y)

        np.save(f'{os.path.abspath(self.working_dir)}/{name}_features.npy', x)
        np.save(f'{os.path.abspath(self.working_dir)}/{name}_validation.npy', y)

    def __load_model(self, model):
        model = snn.SIFTnn()
        model.load_state_dict(torch.load(os.path.abspath(model)))
        model.eval()
        return model

    def __save_model(self, model, path: str='model.pth'):
        torch.save(model.state_dict(), path)

if __name__ == '__main__':

    lblr = SIFTnnLabeler('.')
    # lblr.labelFromVideo('../videos/drone.mp4')

    lblr.labelAndTrain('../videos/drone.mp4', video=True)