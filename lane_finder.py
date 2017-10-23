import pickle
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import glob

class Line():
    def __init__(self):
        self.frame = 0
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')


class LaneFinder():
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    DISTORTED_LEFT = 320
    DISTORTED_RIGHT = 920

    warp_src = np.float32([
        [250, 680],
        [593, 450],
        [685, 450],
        [1050, 680]
    ])

    warp_dest = np.float32([
        [DISTORTED_LEFT, FRAME_HEIGHT],
        [DISTORTED_LEFT, 0],
        [DISTORTED_RIGHT, 0],
        [DISTORTED_RIGHT, FRAME_HEIGHT]
    ])

    ym_per_pix = 30 / FRAME_HEIGHT
    xm_per_pix = 3.7 / (DISTORTED_RIGHT - DISTORTED_LEFT)

    def __init__(self, camera_mtx, camera_dist, render=False):
        self.mtx = camera_mtx
        self.dist = camera_dist
        self.left_line = Line()
        self.right_line = Line()
        self.vehicle_center_m = 0
        self.render = render
        self.radius_of_curve = 0
        self.lane_img = None

    def process(self, img):
        left = self.left_line
        right = self.right_line
        img = self.cal_cam(img, self.mtx, self.dist)
        binary = LaneFinder.threshold(img)
        binary_warped = LaneFinder.perspective_transform(binary)
        left.frame += 1
        right.frame += 1
        should_recalc = (left.best_fit is None or right.best_fit is None)

        if (should_recalc == False):
            left_fit, right_fit = LaneFinder.find_lines_from_previous(binary_warped,
                                                           np.mean(left.recent_xfitted),
                                                           np.mean(right.recent_xfitted))
            if left_fit is None or right_fit is None:
                should_recalc = True
            elif (left_fit[0] * right_fit[0] < 0):
                should_recalc = True
            elif (left_fit[0] * left.best_fit[0] < 0) or (right_fit[0] * right.best_fit[0] < 0):
                should_recalc = True

        if should_recalc:
            left_fit, right_fit = LaneFinder.find_lines_from_scratch(binary_warped)

        obj_lane_width = LaneFinder.DISTORTED_RIGHT - LaneFinder.DISTORTED_LEFT
        lane_width_rejection = 0.2

        # Sanity checks for lane distance
        lane_left_bottom = LaneFinder.quad(left_fit, LaneFinder.FRAME_HEIGHT)
        lane_right_bottom = LaneFinder.quad(right_fit, LaneFinder.FRAME_HEIGHT)
        calculated_lane_width_bottom = lane_right_bottom - lane_left_bottom
        lane_left_top = LaneFinder.quad(left_fit, 0)
        lane_right_top = LaneFinder.quad(right_fit, 0)
        calculated_lane_width_top = lane_right_top - lane_left_top

        use_previous = False

        # Diverging lines, use previous lane findings
        if (np.absolute(calculated_lane_width_bottom - calculated_lane_width_top) > 200):
            use_previous = True
        elif (np.absolute(calculated_lane_width_bottom - obj_lane_width) > lane_width_rejection * obj_lane_width):
            use_previous = True
        elif (np.absolute(calculated_lane_width_top - obj_lane_width) > 2 * lane_width_rejection * obj_lane_width):
            use_previous = True

        if use_previous and left.current_fit is not None:
            left_fit = 0.7 * left.current_fit + 0.3 * left_fit
            right_fit = 0.7 * right.current_fit + 0.3 * right_fit

        if should_recalc:
            left.best_fit, right.best_fit = left_fit, right_fit
        else:
            left.best_fit = (left.best_fit + left_fit) / 2
            right.best_fit = (right.best_fit + right_fit) / 2

        left.radius_of_curvature = LaneFinder.curve_radius(LaneFinder.real_world_coeffs(left_fit),
                                                           LaneFinder.FRAME_HEIGHT * LaneFinder.ym_per_pix)
        right.radius_of_curvature = LaneFinder.curve_radius(LaneFinder.real_world_coeffs(right_fit),
                                                            LaneFinder.FRAME_HEIGHT * LaneFinder.ym_per_pix)

        lane_left = LaneFinder.quad(left_fit, LaneFinder.FRAME_HEIGHT)
        lane_right = LaneFinder.quad(right_fit, LaneFinder.FRAME_HEIGHT)

        camera_center = LaneFinder.FRAME_WIDTH / 2

        self.vehicle_center_m = (camera_center - (lane_left + lane_right) / 2) * LaneFinder.xm_per_pix

        y = np.linspace(0, img.shape[0] - 1, binary_warped.shape[0])

        left.recent_xfitted.append(LaneFinder.quad(left_fit, y))
        right.recent_xfitted.append(LaneFinder.quad(right_fit, y))

        if left.current_fit is not None:
            left.diffs = left_fit - left.current_fit
            right.diffs = right_fit - right.current_fit

        left.current_fit = left_fit
        right.current_fit = right_fit

        lane_img = LaneFinder.fill_lane(binary_warped, left.best_fit, right.best_fit)
        lane_img = LaneFinder.perspective_transform(lane_img, src=LaneFinder.warp_dest, dest=LaneFinder.warp_src)

        self.radius_of_curve = np.mean([left.radius_of_curvature, right.radius_of_curvature])
        self.lane_img = lane_img

        if self.render:
            self.draw(img)

    def draw(self, img):
        draw_img = cv2.addWeighted(img, 1, self.lane_img, 0.3, 0)
        cv2.putText(draw_img, "Vehicle center: {:.2f}m".format(self.vehicle_center_m), \
                    (LaneFinder.FRAME_WIDTH // 2, 40), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(draw_img,
                    "Radius of curve: {:.1f}m".format(self.radius_of_curve), \
                    (LaneFinder.FRAME_WIDTH // 2, 80), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 2, cv2.LINE_AA)
        return draw_img

    @staticmethod
    def cal_cam(img, mtx, dist):
        return cv2.undistort(img, mtx, dist, None, mtx)

    @staticmethod
    def region_of_interest(img):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)
        vertices = np.array([[(100, img.shape[0]), (img.shape[1] * .4, img.shape[0] * .5),
                              (img.shape[1] * .6, img.shape[0] * .5), (img.shape[1], img.shape[0])]], dtype=np.int32)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    @staticmethod
    def threshold(img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        l_channel = hls[:, :, 1]

        # B is the Blue - Yellow spectrum of the LAB color space
        # Filtering for yellow lines
        b_channel = lab[:, :, 2]
        b_mask = np.zeros_like(b_channel)
        b_mask[(l_channel >= 100) & (b_channel >= 150)] = 1

        v_channel = hsv[:, :, 2]
        v_mask = np.zeros_like(v_channel)
        v_mask[(v_channel > 220)] = 1

        #     # Filtering for  saturated lines
        s_channel = hls[:, :, 1]
        sobel_sx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)
        abs_sobel_sx = np.absolute(sobel_sx)
        scaled_sobel_sx = np.uint8(255 * abs_sobel_sx / np.max(abs_sobel_sx))
        # Threshold mask
        sx_mask = np.zeros_like(scaled_sobel_sx)
        sx_mask[(scaled_sobel_sx >= 40) & (scaled_sobel_sx <= 100)] = 1

        stacked = np.zeros_like(s_channel)
        stacked[(b_mask == 1) | (sx_mask == 1) | (v_mask == 1)] = 1
        stacked = LaneFinder.region_of_interest(stacked)

        return stacked

    @staticmethod
    def perspective_transform(img, src=warp_src, dest=warp_dest):
        height, width = img.shape[0], img.shape[1]
        M = cv2.getPerspectiveTransform(src, dest)
        warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)
        return warped

    @staticmethod
    def quad(coeffs, val):
        return coeffs[0] * val ** 2 + coeffs[1] * val + coeffs[2]

    @staticmethod
    def find_lines_from_scratch(binary_warped):
        height, width = binary_warped.shape
        histogram = np.sum(binary_warped[height // 2:, :], axis=0)
        output = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        n_windows = 9  # gives us a 200 x 80 sliding window
        window_height = np.int(height // n_windows)

        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # Find the peaks at the left and right side
        midpoint = np.int(histogram.shape[0] // 2)
        # Start with the peaks of the histogram on the bottom of the image
        leftx_current = np.argmax(histogram[:midpoint])
        rightx_current = np.argmax(histogram[midpoint:]) + midpoint

        left_lane_inds = []
        right_lane_inds = []

        for windex in range(n_windows):
            win_y_low = height - (windex + 1) * window_height
            win_y_high = height - windex * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(output,
                          (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high),
                          (0, 255, 255,), 3)
            cv2.rectangle(output,
                          (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high),
                          (0, 255, 255,), 3)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                              (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                               (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        actual_left_x, actual_right_x = nonzero_x[left_lane_inds], nonzero_x[right_lane_inds]
        lefty, righty = nonzero_y[left_lane_inds], nonzero_y[right_lane_inds]

        left_fit = np.polyfit(lefty, actual_left_x, 2)
        right_fit = np.polyfit(righty, actual_right_x, 2)

        return left_fit, right_fit

    @staticmethod
    def find_lines_from_previous(binary_warped, last_left_x, last_right_x):
        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        margin = 200

        left_lane_inds = ((nonzero_x > (last_left_x - margin)) &
                          (nonzero_x < (last_left_x + margin)))
        right_lane_inds = ((nonzero_x > (last_right_x - margin)) &
                           (nonzero_x < (last_right_x + margin)))

        output = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        actual_left_x, actual_right_x = nonzero_x[left_lane_inds], nonzero_x[right_lane_inds]
        actual_left_y, actual_right_y = nonzero_y[left_lane_inds], nonzero_y[right_lane_inds]

        if len(actual_left_x) < 100 or len(actual_right_x) < 100:
            return None, None

        left_fit = np.polyfit(actual_left_y, actual_left_x, 2)
        right_fit = np.polyfit(actual_right_y, actual_right_x, 2)

        return left_fit, right_fit

    @staticmethod
    def fill_lane(binary_warped, left_fit, right_fit):
        output = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        output = np.zeros_like(output)
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fit_x = LaneFinder.quad(left_fit, ploty)
        right_fit_x = LaneFinder.quad(right_fit, ploty)

        left = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])
        # Flip the value order here to make a filled in shape, instead of going from bottom left, to top right,
        # Go from top left -> bottom left -> bottom right -> top right
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x,
                                                            ploty])))])
        pts = np.hstack((left, right))
        cv2.fillPoly(output, np.int_([pts]), (0, 255, 0))
        return output

    @staticmethod
    def curve_radius(coeffs, val):
        return ((1 + (2 * coeffs[0] * val + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    @staticmethod
    def real_world_coeffs(coeffs):
        # denominator is the width of the lane in our warped image
        ploty = np.linspace(0, LaneFinder.FRAME_HEIGHT - 1, num=LaneFinder.FRAME_HEIGHT)
        fit_x = LaneFinder.quad(coeffs, ploty)
        fit = np.polyfit(ploty * LaneFinder.ym_per_pix, fit_x * LaneFinder.xm_per_pix, 2)
        return fit