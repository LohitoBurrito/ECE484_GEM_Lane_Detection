import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from scipy.ndimage import median_filter

# from combined_thresh import combined_thresh
# from perspective_transform import perspective_transform

# feel free to adjust the parameters in the code if necessary

noise_thresh = 750
straight_thresh = 150
hard_turn_thresh = 220
l_ct = 0
r_ct = 0
current_state = []
actual_heading = 2


def line_fit(binary_warped):
	"""
	Find and fit lane lines
	"""

	global noise_thresh, straight_thresh, hard_turn_thresh, l_ct, r_ct, current_state, actual_heading

	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
	# Create an output image to draw on and visualize the result
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[100:midpoint]) + 100
	rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint
	# Choose the number of sliding windows
	nwindows = 24
	# Set height of windows
	window_height = int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	first_win_left = 0
	first_win_right = 0
	last_win_left = 0
	last_win_right = 0
	max_right = 0
	max_left = 0

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		##TO DO

		tp_l_1 = (leftx_current - margin, window * window_height)
		bt_r_1 = (leftx_current + margin, (window + 1) * window_height)
		tp_l_2 = (rightx_current - margin, window * window_height)
		bt_r_2 = (rightx_current + margin, (window + 1) * window_height)
		y_low = binary_warped.shape[0] - (window+1) * window_height
		y_high = binary_warped.shape[0] - (window) * window_height

		####
		# Draw the windows on the visualization image using cv2.rectangle()
		##TO DO

		out_img = cv2.rectangle(out_img, tp_l_1, bt_r_1, (0, 255, 0), 3)
		out_img = cv2.rectangle(out_img, tp_l_2, bt_r_2, (0, 255, 0), 3)

		####
		# Identify the nonzero pixels in x and y within the window
		##TO DO

		left = np.where((nonzerox >= tp_l_1[0]) & (nonzerox < bt_r_1[0]) & (nonzeroy >= y_low) & (nonzeroy < y_high))[0]
		right = np.where((nonzerox >= tp_l_2[0]) & (nonzerox < bt_r_2[0]) & (nonzeroy >= y_low) & (nonzeroy < y_high))[0]

		####
		# Append these indices to the lists
		##TO DO
	
		left_lane_inds.append(np.array(left))
		right_lane_inds.append(np.array(right))

		if (first_win_left == 0) and (first_win_right == 0) and (len(left) > 0) and (len(right) > 0):
			first_win_left = int(np.mean(nonzerox[left]))
			first_win_right = int(np.mean(nonzerox[right]))
		else:
			temp1 = int(np.mean(nonzerox[left])) if (len(left) > 0) else -1
			temp2 = int(np.mean(nonzerox[right])) if (len(right) > 0) else -1
			if (temp1 != -1 and np.abs(first_win_left - temp1) > max_left):
				last_win_left = temp1
				max_left = np.abs(first_win_left - temp1)
			
			if (temp2 != -1 and np.abs(first_win_right - temp2) > max_right):
				last_win_right = temp2
				max_right = np.abs(first_win_right - temp2)

		####
		# If you found > minpix pixels, recenter next window on their mean position
		##TO DO

		if len(left) > minpix:
			leftx_current = int(np.mean(nonzerox[left]))

		if len(right) > minpix:
			rightx_current = int(np.mean(nonzerox[right]))


	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each using np.polyfit()
	# If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
	# the second order polynomial is unable to be sovled.
	# Thus, it is unable to detect edges.
	try:
	##TODO
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
	####
	except TypeError:
		# print("Unable to detect lanes")
		return None

	diff_left = last_win_left - first_win_left
	abs_diff_left = np.abs(last_win_left - first_win_left)
	diff_right = last_win_right - first_win_right
	abs_diff_right = np.abs(last_win_right - first_win_right)

	if (abs_diff_left >= abs_diff_right) and (straight_thresh < abs_diff_left < noise_thresh):
		# print("Left lane diff: ", abs_diff_left)
		if diff_left > 0:
			r_ct += 1
			current_state.append(1)
			# print("Right turn detected, ", r_ct)
		else:
			l_ct += 1
			current_state.append(-1)
			# print("Left turn detected, ", l_ct)
	elif (abs_diff_right >= abs_diff_left) and (straight_thresh < abs_diff_right < noise_thresh):
		# print("Right lane diff: ", abs_diff_right)
		if diff_right > 0:
			r_ct += 1
			current_state.append(1)
			# print("Right turn detected, ", r_ct)
		else:
			l_ct += 1
			current_state.append(-1)
			# print("Left turn detected, ", l_ct)
	else:
		# print("No turn")
		# print("Left diff =", diff_left)	
		# print("Right diff =", diff_right)
		current_state.append(0)

	filtered_state = (median_filter(np.array(current_state), size = 6))
	left_count = np.sum(filtered_state == -1)
	right_count = np.sum(filtered_state == 1)
	zero_count = np.sum(filtered_state == 0)
	current_max = max(left_count, right_count, zero_count)

	if current_max == left_count:
		if max(abs_diff_left, abs_diff_right) > hard_turn_thresh:	# hard left turn
			actual_heading = 4
			print("Hard left turn")
		else:														# slight left turn
			actual_heading = 3
			print("Slight left turn")
	elif current_max == right_count:
		if max(abs_diff_left, abs_diff_right) > hard_turn_thresh:	# hard right turn
			actual_heading = 0
			print("Hard right turn")
		else:														# slight right turn
			actual_heading = 1
			print("Slight right turn")
	else:
		actual_heading = 2
		print("Straight")

	if len(current_state) > 18:
		current_state = []

	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret


def get_actual_heading():
	global actual_heading
	return actual_heading


def tune_fit(binary_warped, left_fit, right_fit):
	"""
	Given a previously fit line, quickly try to find the line based on previous lines
	"""
	# Assume you now have a new warped binary image
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# If we don't find enough relevant points, return all None (this means error)
	min_inds = 10
	if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
		return None

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret


def viz1(binary_warped, ret, save_file=None):
	"""
	Visualize each sliding window location and predicted lane lines, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	out_img = ret['out_img']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	if save_file is None:
		plt.show()
	else:
		plt.savefig(save_file)
	plt.gcf().clear()


def bird_fit(binary_warped, ret, save_file=None):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	margin = 100  # NOTE: Keep this in sync with *_fit()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	plt.imshow(result)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)

	# cv2.imshow('bird',result)
	# cv2.imwrite('bird_from_cv2.png', result)

	# if save_file is None:
	# 	plt.show()
	# else:
	# 	plt.savefig(save_file)
	# plt.gcf().clear()

	return result


def final_viz(undist, left_fit, right_fit, m_inv):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# Generate x and y values for plotting
	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	# Convert arrays to 8 bit for later cv to ros image transfer
	undist = np.array(undist, dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result
