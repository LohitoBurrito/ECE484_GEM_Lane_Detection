import os
os.environ["YOLO_VERBOSE"] = "False"
import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz, get_actual_heading
from Line import Line
# from lane.detect import detect
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology
from ultralytics import YOLO

import torch

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    select_device,scale_coords,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result, LoadImages


ANGLE_STEP_SMALL = np.pi / 16
ANGLE_STEP_MEDIUM = np.pi / 8
ANGLE_STEP_LARGE = np.pi / 4
MIN_ANGLE_SLIGHT = -2 * np.pi
MAX_ANGLE_SLIGHT = 2 * np.pi
MIN_ANGLE_REG = -3.5 * np.pi
MAX_ANGLE_REG = 3.5 * np.pi
MIN_ANGLE_VEL = 3.5
MAX_ANGLE_VEL = 5.0


# GEM PACMod Header
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd

class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.pub_yolo = rospy.Publisher("lane_detection/yolo", Image, queue_size=1)  # Publisher for YOLO output

        # Steering angle request publisher (TO DO)
        self.steer_rqst_pub = rospy.Publisher('/pacmod/as_rx/steering_rqst', PositionWithSpeed, queue_size=1)
        self.steer_rqst = PositionWithSpeed()
        self.steer_rqst.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_rqst.angular_velocity_limit = 2.0 # radians/second

        self._device = '0'
        self.yolo_seg = torch.jit.load('data/weights/yolopv2.pt')
        self.device = select_device(self._device)

        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.yolo_seg = self.yolo_seg.to(self.device)
        if self.half:
            self.yolo_seg.half()  # to FP16  
        self.yolo_seg.eval()

        # Acceleration request publisher (TO DO)
        self.accel_rqst_pub = rospy.Publisher('/pacmod/as_rx/accel_rqst', PacmodCmd, queue_size=1)
        self.accel_rqst = PacmodCmd()
        self.accel_rqst.enable = True
        self.accel_rqst.clear  = False
        self.accel_rqst.ignore = False
        self.accel_rqst.f64_cmd = 0.0
        self.heading = 0.0

        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True
        self.count = 0
        self.GEM = False
        self.bag0011 = False
        self.bag0056 = False
        self.bag0830 = False
        self.bag0484 = False
        self.GEMe4 = True # use this setting whenever running real GEM vehicle
        self.model = YOLO("yolov8n.pt")  # Load YOLO model
        self.depth_image = None  # Store the latest depth image

        if self.GEM:
            self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        elif self.bag0830 or self.GEMe4:
            # self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_callback, queue_size=1)
            # self.sub_depth = rospy.Subscriber('/zed2/zed_node/depth/depth_registered', Image, self.depth_callback, queue_size=1)
            self.sub_depth = rospy.Subscriber('/oak/stereo/image_raw', Image, self.depth_callback, queue_size=1)
            self.sub_image = rospy.Subscriber('/oak/rgb/image_raw', Image, self.img_callback, queue_size=1)
        else:
            self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)


    def depth_callback(self, data):
        try:
            # Convert ROS depth image to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)


    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(f"Image callback error: {e}")
            return
        
        raw_img = cv_image.copy()
        temp_img = cv_image.copy()
        # temp = detect(np.array(temp_img), temp_img.shape)
        mask_image, bird_image = self.detection(raw_img)
        yolo_img, avg_depth = self.run_yolo(raw_img, self.depth_image)

        if avg_depth is not None:
            self.accel_rqst.f64_cmd = float(avg_depth)
        else:
            self.accel_rqst.f64_cmd = float(100000)

        self.accel_rqst_pub.publish(self.accel_rqst)

        if mask_image is not None and bird_image is not None:
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)
        
        if yolo_img is not None:
            yolo_img_msg = self.bridge.cv2_to_imgmsg(yolo_img, 'bgr8')
            self.pub_yolo.publish(yolo_img_msg)
        
        if avg_depth is not None:
            print(f"Published average person depth: {avg_depth:.2f} meters")
    

    def run_yolo(self, img, depth_map):
        """
        Run YOLO object detection on the input image and annotate the result with bounding boxes.
        Extract depth values for the person class.
        """
        results = self.model(img)  # Run YOLO inference
        annotated_img = results[0].plot()  # Get annotated image with bounding boxes

        person_depths = []

        for result in results[0].boxes:
            bbox = result.xyxy[0].cpu().numpy().astype(int)
            class_id = int(result.cls.cpu().numpy())
            confidence = float(result.conf.cpu().numpy())

            # Check if the detected class is 'person' (usually class_id = 0 for COCO)
            if class_id == 0 and confidence > 0.6:
                x1, y1, x2, y2 = bbox
                # Extract the depth values from the depth_map
                person_region = depth_map[y1:y2, x1:x2]
                person_depths.extend(person_region.flatten())
                # print(person_depths)

        # Compute the average depth if there are valid values
        avg_depth = np.nanmean(person_depths)/1000 if person_depths else None
        # print(avg_depth)  

        return annotated_img, avg_depth
    

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

        # Gaussiam blur the image
        img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

        # Use cv2.Sobel() to find derievatives for both X and Y Axis
        ddepth = cv2.CV_16S

        grad_x = cv2.Sobel(img_blur, ddepth, 1, 0)
        grad_y = cv2.Sobel(img_blur, ddepth, 0, 1)

        # Use cv2.addWeighted() to combine the results
        grad_combined = cv2.addWeighted(grad_x, 0.3, grad_y, 0.3, 0.0)

        # Convert each pixel to unint8, then apply threshold to get binary image
        abs_grad = cv2.convertScaleAbs(grad_combined)

        binary_output = cv2.inRange(abs_grad, thresh_min, thresh_max) / 255

        ####

        return binary_output


    def color_thresh(self, img):
        #1. Convert the image from RGB to HSL
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass

        ## TODO

        # Method 2: RGB-based filtering (only looking at white pixels)
        # lower_white = np.array([200, 200, 200]) 
        # upper_white = np.array([255, 255, 255]) 

        # mask = cv2.inRange(img, lower_white, upper_white)
        # filt_img = cv2.bitwise_and(img, img, mask=mask)
        # binary_output = cv2.cvtColor(filt_img, cv2.COLOR_BGR2GRAY)


        # Method 3: More strict HLS-filtering
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 200, 0])  # Adjusted HLS range
        upper_white = np.array([255, 255, 255])

        mask = cv2.inRange(hls, lower_white, upper_white)
        filt_img = cv2.bitwise_and(img, img, mask=mask)
        binary_output = cv2.cvtColor(filt_img, cv2.COLOR_BGR2GRAY)

        ####

        return binary_output/255


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
        # Remove noise from binary image
        # binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)
        ColorOutput = morphology.remove_small_objects(ColorOutput.astype('bool'),min_size=50,connectivity=2)


        return ColorOutput


    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        h, w = len(img), len(img[0])

        points_src = np.float32([[550, 550], [550, h], [w-430, 550], [w-260, h]])  # longer box
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
                if self.heading < 0:
                    point_x = non_zero_idx[-1]
                else:
                    point_x = non_zero_idx[0]
                point_y = row_idx
            
            elif len(non_zero_idx) > 0 and end_x == -1 and end_y != -1:
                end_x = point_x
                end_y = point_y

                return start_x, start_y, end_x, end_y
            
        print(start_x, start_y, end_x, end_y)

        return start_x, start_y, point_x, point_y
    
    def front2steer(self, f_angle):
        # if(f_angle > 90):
        #     f_angle = 90
        # if (f_angle < -90):
        #     f_angle = -90
        # if (f_angle > 0):
        #     steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        # elif (f_angle < 0):
        #     f_angle = -f_angle
        #     steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        # else:
        #     steer_angle = 0.0
        # return steer_angle
        f_deg = f_angle * 180 / math.pi
        if f_deg < -50:
            return -3.5 * math.pi
        elif f_deg > 50:
            return 3.5 * math.pi

        print ((f_deg / 75) * 3.5)
        return (f_deg / 75) * 3.5 * math.pi
    
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
    
    # def color_thresh_red(self, img):
    #     # Convert the image to HLS color space
    #     hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    #     # Define red thresholds in HLS
    #     lower_red1 = np.array([0, 50, 50])
    #     upper_red1 = np.array([10, 255, 255])
    #     lower_red2 = np.array([170, 50, 50])
    #     upper_red2 = np.array([180, 255, 255])

    #     # Create masks for both red ranges and combine them
    #     mask1 = cv2.inRange(hls, lower_red1, upper_red1)
    #     mask2 = cv2.inRange(hls, lower_red2, upper_red2)
    #     mask = cv2.bitwise_or(mask1, mask2)

    #     # Apply the mask to get the filtered image
    #     filt_img = cv2.bitwise_and(img, img, mask=mask)

    #     # Convert to grayscale and normalize to binary output
    #     binary_output = cv2.cvtColor(filt_img, cv2.COLOR_BGR2GRAY)

    #     return binary_output / 255

    def color_thresh_red(self, img):
        # Define the BGR range for detecting red, with a small margin for tolerance
        lower_red = np.array([0, 0, 100])  # Lower bound of "red"
        upper_red = np.array([80, 80, 255])  # Upper bound of "red"

        # Create a mask for red pixels in the specified BGR range
        mask = cv2.inRange(img, lower_red, upper_red)

        # Apply the mask to the original image
        filt_img = cv2.bitwise_and(img, img, mask=mask)

        # Convert the filtered image to grayscale
        binary_output = cv2.cvtColor(filt_img, cv2.COLOR_BGR2GRAY)

        # Normalize the output to [0, 1]
        return binary_output / 255



    def detection(self, img):
        # cv2.imwrite('../img/output_image_' + str(self.count) + '.png', img)

        # binary_img = self.combinedBinaryImage(img)

        # # cv2.imwrite('../binary/output_image_'+ str(self.count) +'.png', binary_img.astype(np.uint8) * 255)

        # img_birdeye, M, Minv = self.perspective_transform(binary_img)
        
        # # cv2.imwrite('../image_top/output_image_'+ str(self.count) +'.png', img_birdeye * 255)
        # self.count += 1

        # if not self.hist:
        #     # Fit lanvehicle_drivers/gem_gnss_control/scripts/lane/test6.pnge without previous result
        #     ret = line_fit(img_birdeye)
        #     left_fit = ret['left_fit']
        #     right_fit = ret['right_fit']
        #     nonzerox = ret['nonzerox']
        #     nonzeroy = ret['nonzeroy']
        #     left_lane_inds = ret['left_lane_inds']
        #     right_lane_inds = ret['right_lane_inds']
        # else:
        #     # Fit lane with previous result
        #     # if not self.detected:
        #     ret = line_fit(img_birdeye)

        #     if ret is not None:
        #         left_fit = ret['left_fit']
        #         right_fit = ret['right_fit']
        #         nonzerox = ret['nonzerox']
        #         nonzeroy = ret['nonzeroy']
        #         left_lane_inds = ret['left_lane_inds']
        #         right_lane_inds = ret['right_lane_inds']

        #         left_fit = self.left_line.add_fit(left_fit)
        #         right_fit = self.right_line.add_fit(right_fit)

        #         self.detected = True

        #     # Annotate original image
        #     bird_fit_img = None
        #     combine_fit_img = None
        #     if ret is not None:
        #         bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
        #         combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
        #     # else:
        #         # print("Unable to detect lanes")

        #     actual_heading = get_actual_heading()

        #     # Steering angle request publishing
        #     if actual_heading == 4:                     # hard left turn
        #         self.steer_rqst.angular_position = min(self.steer_rqst.angular_position + ANGLE_STEP_MEDIUM, MAX_ANGLE_REG)
        #         self.steer_rqst.angular_velocity_limit = MIN_ANGLE_VEL
        #     if actual_heading == 3:                     # slight left turn
        #         self.steer_rqst.angular_position = min(self.steer_rqst.angular_position + ANGLE_STEP_SMALL, MAX_ANGLE_SLIGHT)
        #         self.steer_rqst.angular_velocity_limit = MIN_ANGLE_VEL
        #     elif actual_heading == 0:                   # hard right turn
        #         self.steer_rqst.angular_position = max(self.steer_rqst.angular_position - ANGLE_STEP_MEDIUM, MIN_ANGLE_REG)
        #         self.steer_rqst.angular_velocity_limit = MIN_ANGLE_VEL
        #     elif actual_heading == 1:                   # slight right turn
        #         self.steer_rqst.angular_position = max(self.steer_rqst.angular_position - ANGLE_STEP_SMALL, MIN_ANGLE_SLIGHT)
        #         self.steer_rqst.angular_velocity_limit = MIN_ANGLE_VEL
        #     elif actual_heading == 2:                   # straight
        #         self.steer_rqst.angular_velocity_limit = MAX_ANGLE_VEL
        #         if self.steer_rqst.angular_position > 0:
        #             self.steer_rqst.angular_position = max(self.steer_rqst.angular_position - ANGLE_STEP_LARGE, 0.0)
        #         elif self.steer_rqst.angular_position < 0:
        #             self.steer_rqst.angular_position = min(self.steer_rqst.angular_position + ANGLE_STEP_LARGE, 0.0)
        #         else:
        #             self.steer_rqst.angular_position = 0.0
                    
        #     self.steer_rqst_pub.publish(self.steer_rqst)

        #     return combine_fit_img, bird_fit_img

        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        bilateral_img = cv2.bilateralFilter(lab_image, d=20, sigmaColor=11, sigmaSpace=50)

        # lab_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2)
        blended_image = cv2.cvtColor(bilateral_img, cv2.COLOR_LAB2BGR)

        yolo_lane_img, avg_depth = self.lane_detect(blended_image, None)

        binary_img = self.color_thresh_red(yolo_lane_img)

        top_left, top_right, bottom_right, bottom_left, warped_img, M, Minv = self.perspective_transform(binary_img)

        warped_img = (warped_img > 0).astype(np.uint8) * 255

        bgr_warped = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR)

        start_x, start_y, end_x, end_y = self.get_endpoints(bgr_warped)

        gray_img = cv2.cvtColor(bgr_warped, cv2.COLOR_BGR2GRAY)         
        _, mask = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

        non_black_pixels = np.count_nonzero(mask)
        
        print("NONZERO: ", non_black_pixels)

        if (non_black_pixels <= 6000):

            print("GOT HERE")
            if (self.heading >= 0):
                self.steer_rqst.angular_position = 3.5 * np.pi
            else:
                self.steer_rqst.angular_position = -3.5 * np.pi
            return bgr_warped, yolo_lane_img

        self.heading = np.arctan2(start_y - end_y, end_x - start_x) - (np.pi / 2)

        print(f"Actual heading: {self.heading * 180 / np.pi:.2f} degrees")

        steering_angle = self.front2steer(self.heading)

        self.steer_rqst.angular_position = steering_angle

        # print(f"Steering angle request: {steering_angle * 180 / np.pi:.2f} degrees")

        self.steer_rqst_pub.publish(self.steer_rqst)

        points = np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.int32)

        perspective_img = cv2.polylines(img, np.int32([points]), isClosed=True, color=(0,0, 255), thickness=2)

        return bgr_warped, yolo_lane_img
        


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)