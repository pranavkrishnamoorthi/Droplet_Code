import sys
import time
import signal
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypylon import pylon
import json

from sklearn.linear_model import Ridge
from pypylon_opencv_viewer import BaslerOpenCVViewer
from Fluigent.SDK import fgt_init, fgt_close
from Fluigent.SDK import fgt_get_sensorRange, fgt_get_sensorValue, fgt_get_pressure, fgt_get_pressureRange, fgt_get_pressureChannelsInfo
from Fluigent.SDK import fgt_set_customSensorRegulation
from Fluigent.SDK import fgt_set_pressure
from simple_pid import PID
from datetime import datetime

from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import pickle

from threading import Thread, Event
import tkinter as tk

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
CALIBRATION_TIME = 10

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
PRESSURE_MIN, PRESSURE_MAX = 100, 130
KP = 0
KI = 0
KD = 0
bias = 10
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
model_created = False
# collect data
data = {}
data['radius'] = []
data['time_step'] = []
data['water_pressure'] = []
data['oil_pressure'] = []
data['diameter'] = []
pid_testing = {}
pid_testing['Kp'] = []
pid_testing["start_time"] = []

calibration_data = {}
calibration_data['radius'] = []
calibration_data['time_step'] = []
calibration_data['water_pressure'] = []
calibration_data['oil_pressure'] = []
calibration_data['diameter'] = []

tube_widths = []
errors = []
no_circles_count = 0

# trained polynomial model


pressures_test = [90 + i*0.1 for i in range(500)]

# actual_radius is the actual target radius in microns

actual_diameter_lst = [20 + 10 * i for i in range(3)]
actual_radius_lst = [d / 2 for d in actual_diameter_lst]
radius_index = 0
last_time = -1
actual_radius = actual_radius_lst[radius_index]
hold_time = 0

upper_path = 'droplet_regulation_data/'
video_filename = upper_path + 'video' + dt_string + '.mp4'
plot_filename = upper_path + 'plot' + dt_string + '.png'
data_filename = upper_path + 'data' + dt_string + '.csv'
calibration_filename = upper_path + 'calibration_data' + dt_string + '.csv'
pid_filename = upper_path + 'pid' + dt_string + '.csv'
# output testing
xfourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(video_filename, xfourcc, 40, (WIDTH, HEIGHT))
output_image = None

iter_num = 0
first_run = False
previous_radius = None
previous_variance = None

current_oil_pressure = None
Kp_step = 0
# set up the PID

# calibration variables
first = True
calibrated = False
calibrate_iter_num = 0
calibrate_new_pressure = 0
calibrate_pressure_step = PRESSURE_MIN
slope = 0
intercept = 0


stop_time = time.time()

# control testing variables
stable_step = 0

# set up the PID


def pid_setup(tgt_radius):
    global pid
    pid = PID(KP, KI, KD, setpoint=tgt_radius)
    # pid.output_limits = (-10, 10)
    pid.sample_time = 1

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


def calibration(current_pressure, base_pressure, upper_pressure):
    # print('Called Calib')
    global first, calibrate_iter_num, calibrate_new_pressure, calibrate_pressure_step, calibrated, slope, intercept, data, calibration_data, target_radius
    # copy data into calibration set
    if len(data['radius']) > 0:
        calibration_data['radius'].append(data['radius'][-1])
        calibration_data['time_step'].append(data['time_step'][-1])
        calibration_data['water_pressure'].append(data['water_pressure'][-1])
        calibration_data['oil_pressure'].append(data['oil_pressure'][-1])
    # calibration_data['diameter'].append(data['diameter'][-1])
    # on first run gradually reset the water pressure to base pressure
    if first and current_pressure != base_pressure:
        print("Starting Calibration")
        calibrate_pressure_step = base_pressure
        print("Base Pressure: " + str(base_pressure))
        fgt_set_pressure(WATER, base_pressure)
        first = False
        return
    # After collecting enough data points appy ridge regression and calculate slope and intercept
    if calibrate_pressure_step >= upper_pressure:
        calibration_data['diameter'] = [
            2 * ratio_height_rad * r for r in calibration_data['radius']]
        df = pd.DataFrame(calibration_data)
        data_rad = np.array(df['radius'])
        data_water = np.array(df['water_pressure'])
        data_oil = np.array(df['oil_pressure'])
        data_ratio = data_water/data_oil
        # FLIPPED DATA X and Y SHOULD BE SWITCHED (FIXED)
        y = data_ratio.reshape(-1, 1)
        X = np.hstack((data_rad.reshape(-1, 1),
                       np.ones((len(data_rad), 1))))
        clf = Ridge(alpha=1.0, fit_intercept=False)
        clf.fit(X, y)
        slope = clf.coef_[0][0]
        intercept = clf.coef_[0][1]
        print("Slope: " + str(slope))
        print("Intercept: "+str(intercept))
        calibrated = True
        print('CALIBRATION DONE')
        target_radius = data_rad[-1]
        print("Setting target radius: ", target_radius)
        return
    # switching to next iteration
    elif calibrate_iter_num < 1000:
        calibrate_new_pressure = calibrate_pressure_step
        calibrate_iter_num += 1

    # swtiching to next pressure
    elif calibrate_pressure_step < upper_pressure:
        print("Stepping up calibration")
        time.sleep(2)
        calibrate_iter_num = 0
        calibrate_pressure_step = calibrate_pressure_step+5
        calibrate_new_pressure = calibrate_pressure_step
        print("Setting Pressure: " + str(calibrate_new_pressure))

    # prevent bounds exceeding
    calibrate_new_pressure = min(
        max(calibrate_new_pressure, PRESSURE_MIN), PRESSURE_MAX)

    # printing stuff no functionality here
    if (calibrate_iter_num % 50 == 0):
        print("\tCurrent Calib Pressure: "+str(calibrate_new_pressure))
        print("\tIter: "+str(calibrate_iter_num))

    # set pressure
    fgt_set_pressure(WATER, calibrate_new_pressure)

    # if new pressure delay to allow for convergence of pressure
    if calibrate_iter_num == 1:
        time.sleep(5)


def pid_test(current_pressure, current_radius):
    global first_run
    global iter_num
    global new_Kp
    global Kp_step
    global data, pid, pid_testing
    pid.Kp = 0.10
    # pid.Kd = 0.1
    pid.Ki = 0.05
    new_pressure = current_pressure + pid(current_radius)
    new_pressure = min(max(new_pressure, PRESSURE_MIN), PRESSURE_MAX)
    # print("PID output: ", pid(current_radius))
    # print("Radius: ", current_radius)
    # print("New Pressure: ", new_pressure)
    fgt_set_pressure(WATER, new_pressure)
    # if first_run == True:
    #     Kp_step = 0  # change to current val
    #     # pid.Kp = Kp_step
    #     iter_num = 0
    #     pid_testing['Kp'].append(Kp_step)
    #     pid_testing['start_time'].append(data['time_step'][-1])
    #     print("PID Output: ", pid(current_radius))
    #     print("Current Pressure: ", current_pressure)
    #     new_pressure = current_pressure + pid(current_radius)
    #     if new_pressure < PRESSURE_MIN or new_pressure > PRESSURE_MAX:
    #         print("Hitting bounds thresh")
    #     new_pressure = min(max(new_pressure, PRESSURE_MIN), PRESSURE_MAX)
    #     fgt_set_pressure(WATER, new_pressure)

    #     first_run = False
    # elif Kp_step > 0.005:
    #     print("Done with PID testing")
    #     log_pid()
    #     start_pid.clear()
    # elif iter_num < 1500:
    #     iter_num += 1
    #     output = pid(current_radius)
    #     new_pressure = current_pressure + output
    #     if new_pressure < PRESSURE_MIN or new_pressure > PRESSURE_MAX:
    #         print("Hitting bounds thresh")
    #     new_pressure = min(max(new_pressure, PRESSURE_MIN), PRESSURE_MAX)
    #     fgt_set_pressure(WATER, new_pressure)
    # elif Kp_step < 0.05:
    #     # time.sleep(5)
    #     Kp_step += 0.001
    #     pid.Kp = Kp_step
    #     pid_testing['Kp'].append(Kp_step)
    #     pid_testing['start_time'].append(data['time_step'][-1])
    #     iter_num = 0


# perform the droplet_regulation
# return increment amount and time of increment
def droplet_regulation(tgt_radius, current_pressure):
    global target_ratio
    global new_pressure
    global stop_time
    assert target_ratio != 0

    radius_error = tgt_radius - data['radius'][-1]
    # pid_result = pid(data['radius'][-1])
    # new_pressure = intercept + pid_result * slope
    # Model is trained in terms of water/oil ratio so need to multiply by oil pressure
    heuristic_pressure = (intercept + tgt_radius*slope)*fgt_get_pressure(OIL)

    if abs(radius_error) > 5:
        if radius_error > 0 and heuristic_pressure > current_pressure:
            new_pressure = min(current_pressure + 20, heuristic_pressure)
            stop_time = time.time()
        elif radius_error < 0 and heuristic_pressure < current_pressure:
            new_pressure = max(current_pressure - 20, heuristic_pressure)
            stop_time = time.time()
        elif radius_error > 0:
            new_pressure = current_pressure + 5
        else:
            new_pressure = current_pressure - 5
    elif abs(radius_error) > 3:
        if radius_error > 0:
            new_pressure = current_pressure + min(abs(radius_error)*slope, 3)
        else:
            new_pressure = current_pressure - min(abs(radius_error)*slope, 3)
    elif abs(radius_error) > 1:
        if radius_error > 0:
            new_pressure = current_pressure + 1
        else:
            new_pressure = current_pressure - 1

    new_pressure = min(max(new_pressure, PRESSURE_MIN), PRESSURE_MAX)

# output data


def log_pid():
    cdf = pd.Dataframe(pid_testing)
    cdf.to_csv(pid_filename)


def log(data):
    # print(data)
    # convert from radii in pixels to actual
    global calibration_data
    print("Logging")
    data['diameter'] = [2 * ratio_height_rad * r for r in data['radius']]
    df = pd.DataFrame(data)
    df = df.sort_values(by='time_step', ascending=True)
    df.plot(kind='line', x='time_step', y='diameter')

    # calibration data
    calibration_data['diameter'] = [
        2 * ratio_height_rad * r for r in calibration_data['radius']]
    cdf = pd.DataFrame(calibration_data)
    cdf = cdf.sort_values(by='time_step', ascending=True)
    cdf.plot(kind='scatter', x='time_step', y='diameter')
    cdf.to_csv(calibration_filename)

    plt.xlabel('time (sec)')
    plt.ylabel('diameter (μm)')
    plt.ylim(0, 100)

    df.to_csv(data_filename)
    plt.savefig(plot_filename)

    plt.show()

    print(df)
    print(tube_widths)


def get_camera_info():
    for i in pylon.TlFactory.GetInstance().EnumerateDevices():
        if i.GetSerialNumber() == serial_number:
            return i
    else:
        print('Camera with {} serial number not found'.format(serial_number))
        return None

# CHT


def droplet_detection(circles, output_img, gray_img):
    assert start != -1

    global gradients, minDist, minRadius, maxRadius, grad_threshold, gradient_value
    global data, pid_has_been_set_up, target_radius, ratio_height_rad, pid, radius_index
    global last_time, new_pressure, actual_radius, pid_off, model_created
    global previous_radius, previous_variance
    global stop_time
    global stable_step, no_circles_count

    radius_sum = 0
    gradient_sum = 0
    sum_gradient_avg = 0
    grad_count = 0
    grad_circle_count = 0

    if circles is None:
        no_circles_count += 1

    # if no_circles_count > 20:
        # make the rest of the codee

    if circles is not None:
        no_circles_count = 0

        minSensor, maxSensor = 0, 400

        # draw the circles
        # TODO: if no circles for 3 seconds,reset to previous working radius

        for i in range(len(circles[0])):
            x, y, r = circles[0][i]
            x, y, r = int(x), int(y), int(r)
            outer_edge_sum = 0
            for the_ten in range(360):
                theta = the_ten * 10
                outer_edge = r
                max_diff = 0
                x1, y1 = x, y
                for delt in range(0, 10):
                    r_new = r + delt
                    x2 = int(x + r_new * np.cos(theta))
                    y2 = int(y + r_new * np.sin(theta))
                    if y2 < 512 and x2 < 512 and x1 < 512 and y1 < 512:
                        if gray_img[x2][y2] - gray_img[x1][y1] > max_diff:
                            max_diff = gray_img[x2][y2] - gray_img[x1][y1]
                            outer_edge = r_new
                        x1, y1 = x2, y2
                    else:
                        break
                outer_edge_sum += outer_edge
            circles[0][i][2] = outer_edge_sum / 360

            radius_sum += r
            # outer_circle
            cv2.circle(output_img, (x, y), r, (0, 255, 0), 4)
            # center dot
            cv2.rectangle(output_img, (x - 2, y - 2),
                          (x + 2, y + 2), (255, 0, 0), -1)

        # if not calibrated and (previous_radius is None or previous_variance < 15):
        #     calibration(fgt_get_pressure(WATER), PRESSURE_MIN, PRESSURE_MAX)

        if len(data['radius']) > 4:
            # consider using an average of the previous 10 points?
            previous_radius = data['radius'][-1]
            previous_variance = np.var(data['radius'][:-5])
        else:
            previous_variance = None
        data['time_step'].append(time.time() - start)
        # FIND WAY TO USE RADIUS HEIGHT RATIO HERE (double check)
        # data['radius'].append(sum(circles[0][:, 2]) / len(circles[0]))
        data['radius'].append(np.median(circles[0][:, 2] / ratio_height_rad))
        # (sum(circles[0][:, 2]) / len(circles[0]))/ratio_height_rad)
        data['water_pressure'].append(fgt_get_pressure(WATER))
        data['oil_pressure'].append(fgt_get_pressure(OIL))

        if start_pid.is_set():
            pid_test(fgt_get_pressure(WATER), data['radius'][-1])

        print('\tCurrent Radius: ', data['radius'][-1])
        if calibrated and CHANNEL_WIDTH != 0:
            print('CALIBRATION DONE')
            if not pid_has_been_set_up:
                print('PID Set up')
                ratio_height_rad = CHANNEL_WIDTH / height_10x
                # target_radius = actual_radius_lst[0] / ratio_height_rad
                target_radius = actual_radius_lst[0]
                pid_setup(target_radius)
                pid_has_been_set_up = True

            if pid_has_been_set_up:
                print("TARGET RADIUS: ", target_radius)

                if len(data['radius']) != 0:

                    print(data['radius'][-1])
                    error = abs(target_radius - data['radius'][-1])
                    if error < 1 and radius_index < len(actual_radius_lst) - 1:
                        if stable_step == 1000:
                            print("RADIUS CHANGE")
                            last_time = time.time()
                            radius_index += 1
                            # target_radius = actual_radius_lst[radius_index] / \
                            #     ratio_height_rad
                            target_radius = actual_radius_lst[radius_index]
                            stable_step = 0
                            pid.setpoint = target_radius
                        else:
                            stable_step += 1
                        #pid.proportional_on_measurement = True

                    if last_time > 0 and time.time() - last_time > 5:
                        print('changing output')
                        pid.set_auto_mode(True, last_output=data['radius'][-1])
                        last_time = -1

                    # if time.time() - stop_time > 5:
                    #     droplet_regulation(
                    #         target_radius, fgt_get_pressure(WATER))
                    #     # new_pressure
                    #     fgt_set_pressure(WATER, new_pressure)

                    if previous_radius is None or previous_variance < 15:
                        droplet_regulation(
                            target_radius, fgt_get_pressure(WATER))
                        # new_pressure
                        fgt_set_pressure(WATER, new_pressure)

                    else:
                        stop_time = 0

                    a = 1

        #fgt_set_customSensorRegulation(data['radius'][-1], 100, 400, 0)
        # update the paramaters for the circles
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


def main(run_event, start_event):
    global camera, start, PRESSURE_MIN, PRESSURE_MAX, output_image

    if camera is not None:
        current_tube_length = 0
        tube_edges = None
        boxes = []

        start = time.time()
        pressure_configuation()

        while camera.IsGrabbing() and run_event.is_set():
            # fgt_init()
            grabResult = camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)

                output_img = image.GetArray()
                cv2.imshow('output', output_img)
                # output_img1 = frame.copy()
                # image recognition calibration (attempt to minimize the time)
                # ???
                gray_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(image=gray_img, method=cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                           param1=gradient_value, minRadius=minRadius, maxRadius=maxRadius)

                if start_event.is_set():
                    droplet_detection(circles, output_img, gray_img)

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

        # log(data)


def getPoint():
    if len(data['radius']) > 0:
        return data['radius'][-1]
    return 0


class Example(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, master=root)
        self.canvas = tk.Canvas(
            self, background="black", height=250, width=300)
        self.canvas.pack(side="top", fill="both", expand=True)

        # create lines for velocity and torque
        self.line = self.canvas.create_line(0, 0, 0, 0, fill="red")

        # start the update process
        self.update_plot()

    def update_plot(self):
        global x
        v = getPoint()
        # x = x+0.1
        self.add_point(self.line, v)
        self.canvas.xview_moveto(1.0)
        self.after(100, self.update_plot)

    def add_point(self, line, y):
        coords = self.canvas.coords(line)
        x = coords[-2] + 5
        coords.append(x)
        coords.append(y)
        coords = coords[-200:]  # keep # of points to a manageable size
        self.canvas.coords(line, *coords)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


window = tk.Tk()
controlFrame = tk.Frame(window)
graphFrame = Example(window)

buttonStart = tk.Button(controlFrame, text="Start Detection")
buttonStop = tk.Button(controlFrame, text="Stop Detection")
pidStart = tk.Button(controlFrame, text="Start PID")
pidStop = tk.Button(controlFrame, text="Stop PID")
label = tk.Label(controlFrame, text="Enter a radius value")
entry = tk.Entry(controlFrame)
button = tk.Button(controlFrame, text="Set Radius")
labelOil = tk.Label(controlFrame, text="Enter oil pressure")
entryOil = tk.Entry(controlFrame)
buttonOil = tk.Button(controlFrame, text="Set Oil Pressure")
buttonQuit = tk.Button(
    controlFrame, text="Exit Controller", command=window.destroy)

start_event = Event()
start_pid = Event()


def radiusClick(event):
    global target_radius
    try:
        # TODO: buffer how fast target radius actually gets to a big drop maybe like increments of 5 or 10
        target_radius = float(entry.get())
        print("Set target radius: ", target_radius)
    except:
        return


def oilClick(event):
    try:
        new_oil = float(entryOil.get())
        fgt_set_pressure(OIL, new_oil)
        print("Set oil pressure: ", new_oil)
    except:
        return


def quitClick(event):
    global window, data, out
    window.destroy()


def startClick(event):
    global start_event
    start_event.set()


def stopClick(event):
    global start_event
    start_event.clear()


def pidStartClick(event):
    global start_pid, first_run
    first_run = True
    pid_setup(target_radius)
    start_pid.set()


def pidStopClick(event):
    global start_pid, first_run, iter_num
    start_pid.clear()
    log_pid()
    first_run = True
    iter_num = 0


window.geometry("500x200")
button.bind('<Button>', radiusClick)
# buttonQuit.bind('<Button>', quitClick)
buttonOil.bind('<Button>', oilClick)
buttonStart.bind('<Button>', startClick)
buttonStop.bind('<Button>', stopClick)
pidStart.bind('<Button>', pidStartClick)
pidStop.bind('<Button>', pidStopClick)
label.pack()
entry.pack()
labelOil.pack()
entryOil.pack()
buttonStart.pack()
buttonStop.pack()
button.pack()
buttonOil.pack()
pidStart.pack()
pidStop.pack()
buttonQuit.pack()

controlFrame.pack()
graphFrame.pack(side="top", fill="both", expand=True)

if __name__ == '__main__':
    run_event = Event()
    run_event.set()
    t1 = Thread(target=main, args=(run_event, start_event))
    t1.daemon = True
    t1.start()
    try:
        window.mainloop()
        start_event.clear()
        run_event.clear()
    except:
        start_event.clear()
        run_event.clear()
        print("hehe")
    log(data)
    out.write(output_img)
    start_event.clear()
    run_event.clear()


# print(data)
# df = pd.DataFrame(data)
# df = df.sort_values(by='time_step', ascending=True)
# print(df)
# df.plot(kind='line',x='time_step', y='radius')
# plt.ylim(0,100)
# plt.show()
