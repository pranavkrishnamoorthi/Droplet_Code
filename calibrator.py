import sys
import time
import signal
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypylon import pylon
import json

from pypylon_opencv_viewer import BaslerOpenCVViewer
from Fluigent.SDK import fgt_init, fgt_close
from Fluigent.SDK import fgt_get_sensorRange, fgt_get_sensorValue, fgt_get_pressure, fgt_get_pressureRange, fgt_get_pressureChannelsInfo
from Fluigent.SDK import fgt_set_customSensorRegulation
from Fluigent.SDK import fgt_set_pressure
from simple_pid import PID
from datetime import datetime

now = datetime.now()


# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

# camera stuff
serial_number = "23280459"
camera = None
info = None
converter = None

# constants
WIDTH = 672
HEIGHT = 512

CAMERA_RESOLUTION = 0.30
CALIBRATION_TIME = 0

MAX_POSSIBlE_RADIUS = 500
MIN_POSSIBLE_RADIUS = 0

OIL = 0
WATER = 1

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
k_min_dist = 25
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
KP = 4
KI = 3
KD = 1
bias = 0
sse_bias = 0
pid = None
k_rad_press = 0.5
all_circle_points = []
ratio_height_rad = 1
current_height = 0
target_ratio = 0.3
CHANNEL_WIDTH = 200
height_10x = 393
np_filter_lines = np.zeros((1, 8), dtype=int)
filter_lines_just_create = True
pid_has_been_set_up = False

pid_off = False
delta_pressure_thresh = 200
# collect data
data = {}
data['radius'] = []
data['time_step'] = []
data['water_pressure'] = []
data['oil_pressure'] = []
data['diamater'] = []
data['flow_rate'] = []

tube_widths = []
errors = []

# actual_radius is the actual target radius in microns

actual_diamater_lst = [20 + 10 * i for i in range(3)]
actual_radius_lst = [d / 2 for d in actual_diamater_lst]
radius_index = 0
last_time = -1
actual_radius = actual_radius_lst[radius_index]
hold_time = 0

upper_path = 'droplet_regulation_data/'
video_filename = upper_path + 'video' + dt_string + '.mp4'
plot_filename = upper_path + 'plot' + dt_string + '.png'
data_filename = upper_path + 'data' + dt_string + '.csv'

# output testing
xfourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(video_filename, xfourcc, 40, (WIDTH, HEIGHT))

new_pressure = None

iter_num = 0
first_run = True
current_radius = 0

current_oil_pressure = None
pressure_step = None
# set up the PID

# calibration variables
first = True
calibrated = False
calibrate_iter_num = 0
calibrate_new_pressure = 0
calibrate_pressure_step = 0
slope = 0
intercept = 0


def pid_setup(tgt_radius):
    global pid
    pid = PID(KP, KI, KD, setpoint=tgt_radius + sse_bias)
    pid.sample_time = 0.1

# set up the pressure configuration


def pressure_configuation():
    global new_pressure

    # fgt_init()
    """

  pressureInfoArray = fgt_get_pressureChannelsInfo()
  for i, pressureInfo in enumerate(pressureInfoArray):
      print('Pressure channel info at index: {}'.format(i))
      print(pressureInfo)
  """

    new_pressure = fgt_get_pressure(WATER)


def gradual_water_pressure(current_pressure, target_pressure):
    # gradually change water pressure

    while current_pressure > target_pressure:
        current_pressure -= min(5, current_pressure-target_pressure)
        fgt_set_pressure(WATER, current_pressure)
        time.sleep(5)
    while current_pressure < target_pressure:
        current_pressure += min(5, target_pressure-current_pressure)
        fgt_set_pressure(WATER, current_pressure)
        time.sleep(5)


def calibration(current_pressure, base_pressure, upper_pressure):
    global first, calibrate_iter_num, calibrate_new_pressure, calibrate_pressure_step, calibrated, slope, intercept, data
    # on first run gradually reset the water pressure to base pressure
    if first and current_pressure != base_pressure:
        gradual_water_pressure(current_pressure, target_pressure)
        first = False
        return
    # After collecting enough data points appy ridge regression and calculate slope and intercept
    if calibrate_pressure_step >= upper_pressure:
        df = pd.DataFrame(data)
        data_rad = np.array(df['radius'])
        data_water = np.array(df['water_pressure'])
        data_oil = np.array(df['oil_pressure'])
        data_ratio = data_water/data_oil
        y = data_rad.reshape(-1, 1)
        X = np.hstack((data_ratio.reshape(-1, 1),
                       np.ones((len(data_ratio), 1))))
        clf = Ridge(alpha=1.0, fit_intercept=False)clf.fit(X, y)
        clf.fit(X, y)
        slope = clf.coef_[0][0]
        intercept = clf.coef_[0][1]
        calibrated = True
        return
    # switching to next iteration
    elif calibrate_iter_num < 1500:
        calibrate_new_pressure = calibrate_pressure_step
        calibrate_iter_num += 1

    # swtiching to next pressure
    elif pressure_step < 135 and abs(current_pressure-calibrate_pressure_step) <= 0.5:
        # time.sleep(5)
        calibrate_iter_num = 0
        calibrate_pressure_step = calibrate_pressure_step+5
        calibrate_new_pressure = calibrate_pressure_step
        print("Setting Pressure: " + str(calibrate_new_pressure))
    calibrate_new_pressure = min(max(new_pressure, PRESSURE_MIN), PRESSURE_MAX)
    if (calibrate_iter_num % 50 == 0):
        print("\tCurrent Pressure: "+str(calibrate_new_pressure))
        print("\tIter: "+str(calibrate_iter_num))

    # set pressure
    fgt_set_pressure(WATER, calibrate_new_pressure)

    # if new pressure delay to allow for convergence of pressure
    if calibrate_iter_num == 1:
        time.sleep(5)


# Iterates through pressures in increments of 5 and collects 100 data points


def pressure_test(current_pressure):
    global first_run
    global iter_num
    global new_pressure
    global pressure_step
    global data
    if first_run == True:
        new_pressure = current_pressure  # change to current val
        iter_num = 0
        pressure_step = current_pressure
        first_run = False
    elif iter_num < 1500 or pressure_step > 135:
        new_pressure = pressure_step
        iter_num += 1
    elif pressure_step < 135 and abs(current_pressure-pressure_step) <= 0.5:
        # time.sleep(5)
        iter_num = 0
        pressure_step = pressure_step+5
        new_pressure = pressure_step
        print("Setting Pressure: " + str(new_pressure))
    new_pressure = min(max(new_pressure, PRESSURE_MIN), PRESSURE_MAX)
    if (iter_num % 50 == 0):
        print("\tCurrent Pressure: "+str(new_pressure))
        print("\tIter: "+str(iter_num))
    fgt_set_pressure(WATER, new_pressure)


def log(data):
    # print(data[max(50,len(data)-50):])
    # convert from radii in pixels to actual
    data['diamater'] = [2 * ratio_height_rad * r for r in data['radius']]
    df = pd.DataFrame(data)
    df = df.sort_values(by='time_step', ascending=True)
    df.plot(kind='line', x='time_step', y='diamater')
    df.plot(kind='line', x='time_step', y='flow_rate')

    plt.xlabel('time (sec)')
    plt.ylabel('diameter (μm)')
    plt.ylim(0, 100)

    df.to_csv(data_filename)
    plt.savefig(plot_filename)

    plt.show()

    print(df)
    print(tube_widths)


# find the perpendicular line thorugh vector projection
def min_dist_lines(a1, b1, a2, b2):
    proj_a1_v = np.dot(a1 - a2, b2 - a2) / np.dot(b2 - a2, b2 - a2)
    dist_squared = np.linalg.norm(a1 - a2) ** 2 - proj_a1_v ** 2
    return np.math.sqrt(dist_squared)


def get_camera_info():
    for i in pylon.TlFactory.GetInstance().EnumerateDevices():
        if i.GetSerialNumber() == serial_number:
            return i
    else:
        print('Camera with {} serial number not found'.format(serial_number))
        return None

# CHT


def droplet_detection(circles, output_img):
    assert start != -1

    global gradients
    global minDist
    global minRadius
    global maxRadius
    global grad_threshold
    global gradient_value
    global data
    global pid_has_been_set_up
    global target_radius
    global ratio_height_rad
    global pid
    global radius_index
    global last_time
    global new_pressure
    global actual_radius
    global pid_off

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
                            print("Circle grad val ",
                                  grad[circle_x1][circle_y1])
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
            cv2.rectangle(output_img, (x_d - 2, y_d - 2),
                          (x_d + 2, y_d + 2), (255, 0, 0), -1)

        if time.time() - start < CALIBRATION_TIME and sum_gradient_avg != 0:
            gradients.append(sum_gradient_avg / grad_circle_count)
            #print("Gradients ", gradients)
            gradient_value = np.sum(gradients) / len(gradients)

        data['time_step'].append(time.time() - start)
        data['radius'].append(sum(circles[0][:, 2]) / len(circles[0]))
        data['water_pressure'].append(fgt_get_pressure(WATER))
        data['oil_pressure'].append(fgt_get_pressure(OIL))
        current_radius = sum(circles[0][:, 2]) / len(circles[0])

        if time.time() - start > CALIBRATION_TIME and CHANNEL_WIDTH != 0:
            if not pid_has_been_set_up:
                print('slkdfjklsjdfl')
                ratio_height_rad = CHANNEL_WIDTH / height_10x
                target_radius = actual_radius / ratio_height_rad
                pid_setup(target_radius)
                pid_has_been_set_up = True

            if pid_has_been_set_up:

                if len(data['radius']) != 0:
                    error = abs(
                        actual_radius_lst[radius_index] - ratio_height_rad * data['radius'][-1])
                    if error < 2 and radius_index < len(actual_radius_lst) - 1:
                        #pid.proportional_on_measurement = True
                        print("RADIUS CHANGE")
                        last_time = time.time()
                        radius_index += 1
                        target_radius = actual_radius_lst[radius_index] / \
                            ratio_height_rad
                        pid.setpoint = target_radius
                        #pid.auto_mode = False
                        pid_off = True

                    if last_time > 0 and time.time() - last_time > 10:
                        print('changing output')
                        #pid.set_auto_mode(True, last_output = data['radius'][-1])
                        last_time = -1
                        pid_off = False

                if not calibrated:
                    calibration(fgt_get_pressure(WATER), 100, 150)
                # pressure_test(fgt_get_pressure(WATER))
                # fgt_set_pressure(WATER, new_pressure) #incorporate min max to prevent crashing
                a = 1

        #fgt_set_customSensorRegulation(data['radius'][-1], 100, 400, 0)
        # update the paramaters for the circles
        if (len(data['radius']) > 0 and int(data['radius'][-1]) != 0):
            minRadius = int(data['radius'][-1]) - delta
            maxRadius = int(data['radius'][-1]) + delta
            minDist = k_min_dist / ((int(data['radius'][-1])) ** 2)
    cv2.imshow('output_img', output_img)
    # log(data)
    out.write(output_img)


# set up the camera
def camera_configuration():
    global info
    global camera
    global converter

    info = get_camera_info()

    if info is not None:
        camera = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateDevice(info))
        camera.Open()

    camera.AcquisitionFrameRate.SetValue(40)
    camera.PixelFormat.GetValue()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


# MAIN CODE
# VERY IMPORTANT STEP! To use Basler PyPylon OpenCV viewer you have to call .Open() method on you camera
camera_configuration()


def sigint_handler(signal, frame):
    print('Interrupted')
    log(data)
    out.write(output_img)
    sys.exit(0)


signal.signal(signal.SIGINT, sigint_handler)
if camera is not None:
    current_tube_length = 0
    tube_edges = None
    boxes = []

    start = time.time()
    PRESSURE_MIN, PRESSURE_MAX = 53, 150
    pressure_configuation()
    while camera.IsGrabbing():
        # fgt_init()
        grabResult = camera.RetrieveResult(
            5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)

            output_img = image.GetArray()
            cv2.imshow('output', output_img)
            # output_img1 = frame.copy()
            # image recognition calibration (attempt to minimize the time)
            if time.time() - start < CALIBRATION_TIME:
                den_img = cv2.fastNlMeansDenoisingColored(
                    output_img, None, 10, 10, 7, 21)
                blur_img = cv2.GaussianBlur(den_img, (3, 3), 0)
                gray_img = cv2.cvtColor(den_img, cv2.COLOR_BGR2GRAY)
                cv2.imshow("gray", gray_img)

                # Calculate the gradient mag vector at all to determine the change in intensities for all values
                grad_x = cv2.Sobel(gray_img, ddepth, 1, 0, ksize=3,
                                   scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
                grad_y = cv2.Sobel(gray_img, ddepth, 0, 1, ksize=3,
                                   scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

            '''
          ret, threshed_img =  cv2.threshold(gray_img,255,255,cv2.THRESH_TOZERO_INV)
          contours = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        #change to minAreaRect later
          for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x,y, x+w,y+h])


          if len(boxes) > 0 and not pid_has_been_set_up:
            left, top = np.min(np.asarray(boxes), axis=0)[:2]
            right, bottom = np.max(np.asarray(boxes), axis=0)[2:]
            cv2.rectangle(output_img, (left,top), (right,bottom), (255, 0, 0), 2)
            current_height = abs(bottom - top)
        '''

            gray_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
            t1 = time.time()
            circles = cv2.HoughCircles(image=gray_img, method=cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                       param1=gradient_value, minRadius=minRadius, maxRadius=maxRadius)
            t2 = time.time()

            if circles is not None:
                data['flow_rate'].append(len(circles[0]) / (t2 - t1))
                droplet_detection(circles, output_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                grabResult.Release()
                camera.StopGrabbing()
                out.release()
                cv2.destroyAllWindows()
                break

            grabResult.Release()
            # out.release()

    #fgt_set_pressure(0, 0)
    camera.Close()
    # fgt_close()

    log(data)

# print(data)
# df = pd.DataFrame(data)
# df = df.sort_values(by='time_step', ascending=True)
# print(df)
# df.plot(kind='line',x='time_step', y='radius')
# plt.ylim(0,100)
# plt.show()
