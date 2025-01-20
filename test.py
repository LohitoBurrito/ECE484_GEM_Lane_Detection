import time
import math
import numpy as np
import cv2
import copy
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology

import torch

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    select_device,scale_coords,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result, LoadImages



class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        # self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True

        self._device = 'cpu'
        self.yolo_seg = torch.jit.load('data/weights/yolopv2.pt', map_location=torch.device(self._device))
        self.device = select_device(self._device)

        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.yolo_seg = self.yolo_seg.to(self.device)
        if self.half:
            self.yolo_seg.half()  # to FP16  
        self.yolo_seg.eval()


    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)


    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to unint8, then apply threshold to get binary image

        ## TODO

        # Convert image to gray scale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # # Gaussiam blur the image
        # img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

        binary_output = cv2.Canny(img_gray, 100, 200)/255

        ####

        return binary_output


    def color_thresh(self, img):

        # hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # lower_white = np.array([0, 200, 0])  # Adjusted HLS range
        # upper_white = np.array([255, 255, 255])

        # mask = cv2.inRange(hls, lower_white, upper_white)
        # filt_img = cv2.bitwise_and(img, img, mask=mask)
        # binary_output = cv2.cvtColor(filt_img, cv2.COLOR_BGR2GRAY)

        # return binary_output / 255

        # High values across R, G, B channels indicate white
        r, g, b = cv2.split(img)
        mask = (r > 200) & (g > 200) & (b > 200)
        return mask.astype(np.float32)

    def remove(self, img):
        # Target RGB value to change to black and tolerance
        target_r, target_g, target_b = 150, 150, 140
        tolerance = 7  # Adjust as needed

        # Create a mask for pixels in the target RGB range
        r, g, b = cv2.split(img)
        target_mask = (
            (r >= target_r - tolerance) & (r <= target_r + tolerance) &
            (g >= target_g - tolerance) & (g <= target_g + tolerance) &
            (b >= target_b - tolerance) & (b <= target_b + tolerance)
        )

        # Create a copy of the image to modify
        result_img = img.copy()

        # Set pixels matching the mask to black (0, 0, 0)
        result_img[target_mask] = [0, 0, 0]

        return cv2.bilateralFilter(img, d=25, sigmaColor=50, sigmaSpace=20)
        
    def lane_detect(self, img, depth_map):
        imgsz = 640
        source = 'data/frame.png'
        cv2.imwrite(source, img)
        stride = 32

        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        dataset.__iter__()
        self.yolo_seg(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.yolo_seg.parameters())))  # run once    

        path, img, im0, vid_cap = next(dataset)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        [pred,anchor_grid],seg,ll= self.yolo_seg(img)
        pred = split_for_trace_model(pred,anchor_grid)
        pred = non_max_suppression(pred, 0.3, 0.45, classes=None, agnostic=False)

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Process detections
        det = pred[0]

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                plot_one_box(xyxy, im0, line_thickness=3)

        # Print time (inference)
        show_seg_result(im0, (da_seg_mask,ll_seg_mask), is_demo=True)

        return im0, 1.0
    
    def color_thresh_red(self, img):
        # Convert the image to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # Define red thresholds in HLS
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for both red ranges and combine them
        mask1 = cv2.inRange(hls, lower_red1, upper_red1)
        mask2 = cv2.inRange(hls, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply the mask to get the filtered image
        filt_img = cv2.bitwise_and(img, img, mask=mask)

        # Convert to grayscale and normalize to binary output
        binary_output = cv2.cvtColor(filt_img, cv2.COLOR_BGR2GRAY)

        return binary_output / 255

    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.

        ## TODO
        SobelOutput = self.gradient_thresh(img=img)
        ColorOutput = self.color_thresh(img=img)
        ####

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1
        # binaryImage[ColorOutput==1] = 1
        # Remove noise from binary image
        # ColorOutput = morphology.remove_small_objects(ColorOutput.astype('bool'),min_size=50,connectivity=2)
        binaryImage = binaryImage.astype(np.uint8)

        return binaryImage


    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        h, w = len(img), len(img[0])

        points_src = np.float32([[660, 490], [660, h], [w-460, 490], [w-50, h]])  # longer box
        margin_reduction = 200
        points_dest = np.float32([[margin_reduction, 0], [margin_reduction, h-1], [w - margin_reduction, 0], [w - margin_reduction, h-1]])

        M = cv2.getPerspectiveTransform(points_src, points_dest)

        Minv = np.linalg.inv(M)

        warped_img = cv2.warpPerspective(np.float32(img), M, (w, h))
        
        return points_src[0], points_src[1], points_src[2], points_src[3], warped_img, M, Minv
    
    def get_endpoints(self, img):

        height, width, _ = img.shape

        start_x, start_y, end_x, end_y, point_x, point_y = -1, -1, -1, -1, -1, -1

        for row_idx in range(height-1, -1, -1):

            row = img[row_idx]
            non_zero_idx = np.where(np.all(row != (0,0,0), axis=1))[0]

            if len(non_zero_idx) > 0 and start_x == -1:
                start_x = non_zero_idx[0]
                start_y = row_idx
                point_x = non_zero_idx[0]
                point_y = row_idx
            
            elif len(non_zero_idx) > 0:
                point_x = non_zero_idx[0]
                point_y = row_idx
            
            elif len(non_zero_idx) > 0 and end_x == -1 and start_x != -1:
                end_x = point_x
                end_y = point_y

                return start_x, start_y, end_x, end_y

        return start_x, start_y, point_x, point_y

    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:

                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original imageimg/output_image_3.png
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


img = cv2.imread('test17.png')

node = lanenet_detector()

lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

bilateral_img = cv2.bilateralFilter(lab_image, d=25, sigmaColor=50, sigmaSpace=20)

blended_image = cv2.cvtColor(bilateral_img, cv2.COLOR_LAB2BGR)

yolo_lane_img, avg_depth = node.lane_detect(blended_image, None)

cv2.imshow('img', yolo_lane_img)

cv2.waitKey(0)

cv2.destroyAllWindows()

binary_img = node.color_thresh_red(yolo_lane_img)

cv2.imshow('img', binary_img)

cv2.waitKey(0)

cv2.destroyAllWindows()

top_left, bottom_left, top_right, bottom_right, warped_img, M, Minv = node.perspective_transform(binary_img)

cv2.imshow('img', warped_img)

cv2.waitKey(0)

cv2.destroyAllWindows()

warped_img = (warped_img > 0).astype(np.uint8) * 255

cv2.imshow('img', warped_img)

cv2.waitKey(0)

cv2.destroyAllWindows()

bgr_warped = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR)

start_x, start_y, end_x, end_y = node.get_endpoints(bgr_warped)

cv2.line(bgr_warped, (start_x, start_y), (end_x, end_y), (0, 0, 255), 3)

cv2.imshow('img', bgr_warped)

cv2.waitKey(0)

cv2.destroyAllWindows()

points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

perspective_img = cv2.polylines(img, np.int32([points]), isClosed=True, color=(0,0, 255), thickness=2)

cv2.imshow('img', perspective_img)

cv2.waitKey(0)

cv2.destroyAllWindows()