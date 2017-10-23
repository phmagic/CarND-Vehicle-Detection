import numpy as np
import cv2
from skimage.feature import hog
from collections import deque
import math

from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

# Fast HOG calculation, used in pipeline
def cv_extract_hog_features(img, cspace='RGB',
                         orient=9, pix_per_cell=8,
                         cell_per_block=2, hog_channel='ALL'):
    # apply color conversion if other than 'RGB'

    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else:
        feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            opencv_hog = cv2.HOGDescriptor((img.shape[0],img.shape[1]), 
                (pix_per_cell*cell_per_block,pix_per_cell*cell_per_block), 
                (pix_per_cell,pix_per_cell), (pix_per_cell, pix_per_cell), orient)
            hog_features.append(opencv_hog.compute(feature_image[:, :, channel]))
        hog_features = np.ravel(hog_features)
    else:
        opencv_hog = cv2.HOGDescriptor((img.shape[0],img.shape[1]), 
                (pix_per_cell*cell_per_block,pix_per_cell*cell_per_block), 
                (pix_per_cell,pix_per_cell), (pix_per_cell, pix_per_cell), orient)
        hog_features = opencv_hog.compute(feature_image[:, :, hog_channel])
    # Return list of feature vectors
    return hog_features

# Used for previewing HOG visuals
def get_hog_features(img, orientations, pixels_per_cell, cells_per_block, vis=False, feature_vec=False):
    if vis:
        features, hog_image = hog(img, orientations=orientations,
                                  pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                  cells_per_block=(cells_per_block, cells_per_block),
                                  transform_sqrt=False,
                                  visualise=vis,
                                  feature_vector=feature_vec)
        return features, hog_image

    else:
        return hog(img, orientations=orientations,
                   pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                   cells_per_block=(cells_per_block, cells_per_block),
                   transform_sqrt=False,
                   visualise=vis,
                   feature_vector=feature_vec)


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def get_bboxes(labels):
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        bboxes.append(bbox)
    return bboxes


class Vehicle():
    def __init__(self, box):
        self.box = box
        self.centroid = (0,0)
        self.size = 0
        self.missing_duration = 0
        self.ratio = 0

class VehicleTracker():
    def __init__(self, 
                clf, scaler, color_space, orientations, pix_per_cell, cell_per_block, 
                hist_bins, render=False):
        self.frames = deque(maxlen=8)
        self.frame_number = 0
        self.vehicles = []
        self.render = render
        self.clf = clf
        self.scaler = scaler
        self.color_space = color_space
        self.orientations = orientations
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hist_bins = hist_bins

    def search_windows(self, img, windows, 
                    clf, scaler, color_space='RGB', 
                    hist_bins=32,
                    hist_range=(0, 256), 
                    orient=9,
                    pix_per_cell=8, 
                    cell_per_block=2):
        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window using single_img_features()
            hog_features = cv_extract_hog_features(test_img, cspace=color_space,
                                                   orient=orient, pix_per_cell=pix_per_cell,
                                                   cell_per_block=cell_per_block, hog_channel='ALL')
            hist_features = color_hist(test_img, nbins=hist_bins)
            # spatial_features = bin_spatial(test_img, size=spatial_size)
            features = np.hstack((hog_features, hist_features))

            # 5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            # 6) Predict using your classifier (with probabilities)
            prob = clf.predict_proba(test_features)
            # Only save the window if the classifier is sure it's a car
            if prob[0][1] > 0.8:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows

    def find_cars_windows(self, img):
        hot_windows = []
        window_sizes = [slide_window(img, x_start_stop=[650, img.shape[1]],  y_start_stop=[390, 490], 
                            xy_window=(96, 96), xy_overlap=(0.8, 0.7)),
                        slide_window(img, x_start_stop=[750, img.shape[1]], y_start_stop=[400, 620], 
                                xy_window=(128, 128), xy_overlap=(0.7, 0)),
                        slide_window(img, x_start_stop=[900, img.shape[1]], y_start_stop=[400, 650], 
                                xy_window=(144, 144), xy_overlap=(0.8, 0)),
                        slide_window(img, x_start_stop=[1000, img.shape[1]], y_start_stop=[400, 650], 
                                xy_window=(192, 192), xy_overlap=(0.8, 0))
        ]
        
        for size in window_sizes:
            hot = self.search_windows(img, size, 
                                clf=self.clf, 
                                scaler=self.scaler, 
                                color_space=self.color_space, 
                                orient=self.orientations, 
                                pix_per_cell=self.pix_per_cell, 
                                cell_per_block=self.cell_per_block, 
                                hist_bins=self.hist_bins)
            hot_windows += hot
        return hot_windows
        
    def process(self, img):
        input_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        boxes = self.find_cars_windows(input_img)
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat,boxes)
        self.frames.append(heat)
        avg_frame = np.mean(np.array(self.frames), axis=0)
        heat = apply_threshold(avg_frame,0.25)
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        bboxes = get_bboxes(labels)
        for box in bboxes:
            # Filter out bad boxes
            # Split wide boxes
            width = (box[1][0] - box[0][0])
            height = (box[1][1] - box[0][1])
            ratio = width / height
            cv2.rectangle(img, box[0], box[1], (0, 0, 255), 4)
            
            self.update_vehicle(box)
            if (ratio > 2.2):
                box_1 = (box[0], (box[1][0] - width//2, box[1][1]))
                box_2 = ((box_1[1][0] + 10, box_1[0][1]), (box_1[1][0] + width//2, box_1[1][1]))
                self.update_vehicle(box_1)
                self.update_vehicle(box_2)
            else:
                self.update_vehicle(box)
            
        
        existing_vehicles = []
        # Remove old vehicles
        for vehicle in self.vehicles:
            # Is vehicle too far in the horizon
            # Or disappeared off the side?
            vehicle.missing_duration += 1
            if vehicle.missing_duration < 48:
                existing_vehicles.append(vehicle)
                
            
        self.vehicles = existing_vehicles
        
        self.frame_number += 1
        if self.render:
            return self.draw(img)
    
    def draw(self, img):
        for vehicle in self.vehicles:
            cv2.rectangle(img, vehicle.box[0], vehicle.box[1], (255, 255, 0), 4)
#             cv2.putText(img, "{} {:d}".format(vehicle.missing_duration, vehicle.centroid[1]), \
#                  (vehicle.box[0][0],vehicle.box[0][1]), cv2.FONT_HERSHEY_PLAIN, 2.5, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(img, "Cars: {}".format(len(self.vehicles)), \
                 (img.shape[1]//2,120), cv2.FONT_HERSHEY_PLAIN, 2.5, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, "Frame: {:d}".format(self.frame_number), \
                 (img.shape[1]//2,160), cv2.FONT_HERSHEY_PLAIN, 2.5, (255,255,255), 2, cv2.LINE_AA)
        
        return img
    
    def update_vehicle(self, box):
        width = (box[1][0] - box[0][0])
        height = (box[1][1] - box[0][1])
        ratio = width / height
        size = width*height
        centroid = (box[0][0] + width//2, box[0][1] + height//2)
        # Check for weird shapes
        normal_shape = (ratio > 0.5 and ratio < 2.2)
        if not normal_shape:
            return
        # Find the closest vehicle to the box
        closest_vehicle = None
        for vehicle in self.vehicles:
            this_distance = math.sqrt(math.pow((vehicle.centroid[0] - centroid[0]), 2) + 
                                     math.pow((vehicle.centroid[1] - centroid[1]), 2))
            size_diff = math.fabs(size - vehicle.size) / vehicle.size
            if closest_vehicle is None:
                if this_distance < 100:
                    closest_vehicle = vehicle
            else:
                closest_distance = math.sqrt(math.pow((closest_vehicle.centroid[0] - centroid[0]), 2) + 
                                     math.pow((closest_vehicle.centroid[1] - centroid[1]), 2))
                if closest_distance > this_distance:
                    closest_vehicle = vehicle
        
        vehicle = Vehicle(box)         
        if closest_vehicle is None:
            self.vehicles.append(vehicle)
            vehicle.box = box
        else:
            vehicle = closest_vehicle
            if vehicle.missing_duration > 5:
                vehicle.box = box
            else:
                box_start_x = int((0.95*vehicle.box[0][0] + 0.05*box[0][0]))
                box_start_y = int((0.95*vehicle.box[0][1] + 0.05*box[0][1]))
                box_end_x = int((0.95*vehicle.box[1][0] + 0.05*box[1][0]))
                box_end_y = int((0.95*vehicle.box[1][1] + 0.05*box[1][1]))
                vehicle.box = ((box_start_x, box_start_y), (box_end_x, box_end_y))
                
        vehicle.size = size
        vehicle.centroid = centroid
        vehicle.missing_duration = 0
        vehicle.ratio = ratio
