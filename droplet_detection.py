import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypylon import pylon 
import json

from pypylon_opencv_viewer import BaslerOpenCVViewer
from Fluigent.SDK import fgt_init, fgt_close
from Fluigent.SDK import fgt_get_sensorRange, fgt_get_sensorValue, fgt_get_pressure, fgt_get_pressureRange
from Fluigent.SDK import fgt_set_customSensorRegulation
from Fluigent.SDK import fgt_set_pressure
from simple_pid import PID

#camera stuff
serial_number = "23280459"
camera = None
info = None
converter = None

#constants
WIDTH = 672
HEIGHT = 512

CAMERA_RESOLUTION = 0.30
CALIBRATION_TIME = 20

MAX_POSSIBlE_RADIUS = 500
MIN_POSSIBLE_RADIUS = 0

# optimize these paramaters with heuristics during calibration phase
accumulator_size = 200000
dp = (CAMERA_RESOLUTION * 10 ** 6) / accumulator_size
accum_threshold = 10
scale = 1
delta = 0
ddepth = cv2.CV_16S
gradients = []
gradient_value = 140
grad_threshold = 80
delta = 15
k_min_dist = 50
line_sim_thresh = 100
start = -1
minDist = 10
gradient_value = 140
grad_threshold = 100
minRadius = 5
maxRadius = 60
tube_width = MAX_POSSIBlE_RADIUS
PRESSURE_MIN = None
PRESSURE_MAX = None
KP = 0.5
KI = 0.5
pid = None
k_pressure_radius = 0.5
all_circle_points = []

np_filter_lines = np.zeros((1, 8), dtype=int)
filter_lines_just_create = True

#collect data
data = {}
data['radius'] = []
data['time_step'] = []
tube_widths = []
errors = []



def pressure_configuation():
  fgt_init()

  pressureInfoArray = fgt_get_pressureChannelsInfo()
  for i, pressureInfo in enumerate(pressureInfoArray):
      print('Pressure channel info at index: {}'.format(i))
      print(pressureInfo)


def droplet_regulation(tgt_radius, current_pressure):
    pressure_error = tgt_radius - data['radius'][-1]
    return min(max(current_pressure + k_rad_press * pressure_error, fgt_get_pressure(0) / 3), PRESSURE_MAX)




def log(data):
    print(data)
    df = pd.DataFrame(data)
    df = df.sort_values(by='time_step', ascending=True)
    df.plot(kind='line', x='time_step', y='radius')
    plt.ylim(0, 100)
    plt.show()
    plt.savefig('droplet_results.png')
    print(df)
    print(tube_widths)


#find the perpendicular line thorugh vector projection
def min_dist_lines(a1, b1, a2, b2):

    proj_a1_v = np.dot(a1 - a2, b2 - a2) / np.dot(b2 - a2, b2 - a2)

    dist_squared = np.linalg.norm(a1 - a2) ** 2  -  proj_a1_v ** 2

    return np.math.sqrt(dist_squared)

def get_camera_info():
  for i in pylon.TlFactory.GetInstance().EnumerateDevices():
      if i.GetSerialNumber() == serial_number:
          return i
  else:
      print('Camera with {} serial number not found'.format(serial_number))
      return None


    global filter_lines_just_create
    global np_filter_lines
    if lines is not None:
        line_pairs_dots = np.zeros((len(lines), len(lines)))

        #find the pairs of lines that are the most paralell
        
        for i in range(len(lines)):
            line_1 = lines[i][0]
            for j in range(len(lines)):
                print('hi')
                if i <= j:
                    line_pairs_dots[i,j] = 0
                else:
                    line_2 = lines[j][0]
                    #print(line_1)
                    #print(line_2)
                    

                    a1 = np.array(line_1[:2])
                    b1 = np.array(line_1[2:])

                    a2 = np.array(line_2[:2])
                    b2 = np.array(line_2[2:])


                    #print(b1 - a1)
                    #print(b2 - a2)

                    v1 = np.subtract(b1, a1)
                    v2 = np.subtract(b2, a2)
                    
                #print(v1.shape)
                    #print(v2.shape)
                    line_pairs_dots[i,j] = np.dot(v1, v2)
            


        print(line_pairs_dots)
        dot_pairs_indices= np.argsort(-1 * line_pairs_dots.flatten())
        print(dot_pairs_indices)
        print('NUM OPTIONS ', len(dot_pairs_indices))

            
        


            #go through the most parallel
        for index in dot_pairs_indices.flatten():
            i = index // len(lines)
            j = index - (i * len(lines))


            print((i, j))
            line_1 = lines[i][0]
            line_2 = lines[j][0]

            a1 = np.array(line_1[:2])
            b1 = np.array(line_1[2:])

            a2 = np.array(line_2[:2])
            b2 = np.array(line_2[2:])

            tube_length  = None
            if abs(a1[1] - a2[1]) > line_sim_thresh:
                if 1:
                    if abs(b1[1] - b2[1]) > line_sim_thresh:
                        if 1:
                            if(filter_lines_just_create):
                                np_filter_lines[0,:] = np.array([[a1[0], a1[1], b1[0], b1[1], a2[0], a2[1], b2[0], b2[1]]])
                                filter_lines_just_create = False
                            else:
                                np_filter_lines = np.append(np_filter_lines, [[a1[0], a1[1], b1[0], b1[1], a2[0], a2[1], b2[0], b2[1]]], axis=0)


            
                            print(a1.shape)
                            tube_length = min_dist_lines(a1, b1, a2, b2)

                            print("TUBE LENGTH", tube_length)
                            #print("NP FILTER LINES")
                            #print(np_filter_lines)
                            
        
            if tube_length:
                current_tube_length = tube_length
                #cv2.line(output_img, (line_1[0], line_1[1]), (line_1[2], line_1[3]), (255,0, 0),2)
                #cv2.line(output_img, (line_1[0], line_1[1]), (line_1[2], line_1[3]), (0,0, 255),2)

                current_tube_length = tube_length
                v = b2 - a2
                v_norm = np.linalg.norm(v)
                if v_norm:
                    proj_a1_v = np.dot(a1 - a2, v) / (np.linalg.norm(v) ** 2)

                #draw the perpendicular line

                
                #cv2.line(output_img, (a1[0], a1[1]), (a2[0] + int((proj_a1_v  * v)[0]), a2[1] + int((proj_a1_v * v)[1])), (0, 255, 0), 2)

            
            #bin the coords to get the most common cord

    line_par_filtered = np_filter_lines

    new_filter = [0,0,0,0,0,0,0,0]
    if line_par_filtered.shape != (1,8):


        for i in range(8):
            coordinate_col = line_par_filtered[:,i]
            coordinate_bins = np.argsort(-1 * np.bincount(coordinate_col))

            print("COORD BINS")
            print(coordinate_bins)
            if coordinate_bins.any():
                most_common_coord = coordinate_bins[0]
                new_filter[i] = most_common_coord

                if i >= 4:
                    j = 1
                    while abs(new_filter[i-4] - most_common_coord) < line_sim_thresh and j <= most_common_coord:
                        most_common_coord = coordinate_bins[j]
                        j += 1
                    new_filter[i] = most_common_coord
                    


                line_par_filtered = line_par_filtered[np.where(line_par_filtered[:,i] == most_common_coord)]
    
    if new_filter != [0,0,0,0,0,0,0,0]:
        cv2.line(output_img, (new_filter[0], new_filter[1]), (new_filter[2], new_filter[3]), (255, 0, 0), 2)
        cv2.line(output_img, (new_filter[4], new_filter[5]), (new_filter[6], new_filter[7]), (0, 255, 0), 2)
        a1 = np.array([new_filter[0], new_filter[1]])
        b1 = np.array([new_filter[2], new_filter[3]])

        a2 = np.array([new_filter[4], new_filter[5]])
        b2 = np.array([new_filter[6], new_filter[7]])
        
        v = b2 - a2
        if np.linalg.norm(v) != 0:
            proj_a1_v = np.dot(a1 - a2, v) / (np.linalg.norm(v) ** 2)
            
            #draw the perpendicular line
            cv2.line(output_img, (a1[0], a1[1]), (a2[0] + int((proj_a1_v  * v)[0]), a2[1] + int((proj_a1_v * v)[1])), (0, 0, 255), 2)
            cv2.imshow('output_lines_1', output_img)


        return new_filter
    else:
        return None

def droplet_detection(circles, output_img):
    assert start != -1

    global gradients
    global minDist
    global minRadius
    global maxRadius
    global grad_threshold
    global gradient_value
    global data

    radius_sum = 0
    gradient_sum = 0
    sum_gradient_avg = 0
    grad_count = 0
    grad_circle_count = 0

    if circles is not None:
        minSensor, maxSensor = 0, 400

        # draw the circles
        for (x_d, y_d, r_d) in circles[0]:
            if time.time() - start < CALIBRATION_TIME:
                gradient_sum = 0
                gradient1_sum = 0
                grad_count = 0
                grad1_count = 0
                for theta in range(360):
                    circle_x = int(x_d + r_d * np.cos(theta))
                    circle_y = int(y_d + r_d * np.sin(theta))

                    # if missed by rounding error check this too
                    circle_x1 = int(x_d + r_d * np.cos(theta)) + 1
                    circle_y1 = int(y_d + r_d * np.sin(theta)) + 1

                    if circle_x < len(grad) and circle_y < len(grad[circle_x]):
                        circle_grad_val = grad[circle_x][circle_y]
                        # rather than using threshold, figure out a way to get like values in the 75th percentile of all grad vals or smt
                        if circle_grad_val > grad_threshold:
                            print("Circle grad val ", grad[circle_x][circle_y])
                            gradient_sum += grad[circle_x][circle_y]
                            grad_count += 1
                            all_circle_points.append([circle_x, circle_y])
                    if circle_x1 < len(grad) and circle_y1 < len(grad[circle_x1]):
                        circle_grad_val = grad[circle_x1][circle_y1]
                        if circle_grad_val > grad_threshold:
                            print("Circle grad val ", grad[circle_x1][circle_y1])
                            gradient_sum += grad[circle_x1][circle_y1]
                            grad1_count += 1
                            all_circle_points.append([circle_x1, circle_y1])

                if grad_count > 0:
                    sum_gradient_avg += gradient_sum / grad_count
                    grad_circle_count += 1
                if grad1_count > 0:
                    sum_gradient_avg += gradient1_sum / grad1_count
                    grad_circle_count += 1

            radius_sum += r_d
            x_d, y_d, r_d = int(x_d), int(y_d), int(r_d)
            # outer_circle
            cv2.circle(output_img, (x_d, y_d), r_d, (0, 255, 0), 4)
            # center dot
            cv2.rectangle(output_img, (x_d - 2, y_d - 2), (x_d + 2, y_d + 2), (255, 0, 0), -1)

        if time.time() - start < CALIBRATION_TIME and sum_gradient_avg != 0:
            gradients.append(sum_gradient_avg / grad_circle_count)
            #print("Gradients ", gradients)
            gradient_value = np.sum(gradients) / len(gradients)

        data['time_step'].append(time.time() - start)
        data['radius'].append(sum(circles[0][:, 2]) / len(circles[0]))

        #fgt_set_customSensorRegulation(data['radius'][-1], 100, 400, 0)
        # update the paramaters for the circles
        minRadius = int(data['radius'][-1]) - delta
        maxRadius = int(data['radius'][-1]) + delta
        minDist = k_min_dist / ((int(data['radius'][-1])) ** 2)
    cv2.imshow('output_img', output_img)



def camera_configuration():
  global info
  global camera
  global converter
  
  info = get_camera_info()

  if info is not None:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
    camera.Open()

  camera.AcquisitionFrameRate.SetValue(40)
  camera.PixelFormat.GetValue()
  camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
  converter = pylon.ImageFormatConverter()
  converter.OutputPixelFormat = pylon.PixelType_BGR8packed
  converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

  
# VERY IMPORTANT STEP! To use Basler PyPylon OpenCV viewer you have to call .Open() method on you camera
camera_configuration()

if camera is not None:
  current_tube_length = 0
  tube_edges = None
  boxes = []

  start = time.time()
  pressure_val = 200
  PRESSURE_MIN, PRESSURE_MAX = fgt_get_pressureRange(1)
  while camera.IsGrabbing():
    #fgt_init()
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)

        output_img = image.GetArray()
        cv2.imshow('output', output_img)
        # output_img1 = frame.copy()
        # image recognition calibration (attempt to minimize the time)
        if time.time() - start < CALIBRATION_TIME:
          den_img = cv2.fastNlMeansDenoisingColored(output_img, None, 10, 10, 7, 21)
          blur_img = cv2.GaussianBlur(den_img, (3, 3), 0)
          gray_img = cv2.cvtColor(den_img, cv2.COLOR_BGR2GRAY)
          cv2.imshow("gray", gray_img)

          # Calculate the gradient mag vector at all to determine the change in intensities for all values
          grad_x = cv2.Sobel(gray_img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
          grad_y = cv2.Sobel(gray_img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
          abs_grad_x = cv2.convertScaleAbs(grad_x)
          abs_grad_y = cv2.convertScaleAbs(grad_y)
          grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

          ret, threshed_img = cv2.threshold(gray_img, 230, 255, cv2.THRESH_BINARY)
          contours = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

  
          #change to minAreaRect later
          for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x,y, x+w,y+h])

          
          left, top = np.min(np.asarray(boxes), axis=0)[:2]
          right, bottom = np.max(np.asarray(boxes), axis=0)[2:]

          cv2.rectangle(output_img, (left,top), (right,bottom), (255, 0, 0), 2)

          #calibrate these if needed to get the lines faster through testing
          #edges = cv2.Canny(gray_img, 50, 300, 4)
          #lines = cv2.HoughLinesP(edges, rho=1, theta = np.pi/180, threshold =50, minLineLength = WIDTH / 2, maxLineGap = 20)

          #display the tube edges (not accurate immediately, will calibirate and achieve the tube lines)
          #tube_edges = get_tube_edges(lines, output_img)

        gray_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(image=gray_img, method=cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                    param1=gradient_value, minRadius=minRadius, maxRadius=maxRadius)

        droplet_detection(circles, output_img)
        #fgt_set_pressure(1, droplet_regulation(target_radius, fgt_get_pressure(1)))



        if cv2.waitKey(1) & 0xFF == ord('q'):
            grabResult.Release()
            camera.StopGrabbing()
            #out.release()
            cv2.destroyAllWindows()
            break

        grabResult.Release()

  #fgt_set_pressure(0, 0)
  camera.Close()
  #fgt_close()

  log(data)

# print(data)
# df = pd.DataFrame(data)
# df = df.sort_values(by='time_step', ascending=True)
# print(df)
# df.plot(kind='line',x='time_step', y='radius')
# plt.ylim(0,100)
# plt.show()
