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
from Fluigent.SDK import fgt_get_sensorRange, fgt_get_sensorValue
from Fluigent.SDK import fgt_set_customSensorRegulation
from Fluigent.SDK import fgt_set_pressure

# maybe use cypython to optimize furth1er if neeßded

# fill this in based on camera used (megaPixels)
# current camera: Basler ace acA640-750uc Color USB 3.0 Camera

#MAKE ALL THE CONSTANTS CAPS AFTERWARDS
width = 672
height = 512

# viewer.set_configuration(VIEWER_CONFIG_RGB_MATRIX)
# viewer.show_interactive_panel(window_size=(1000,2000))
'''
img = viewer.get_image()
cv2.imshow('sample', img)
'''


CAMERA_RESOLUTION = 0.30
#cap = cv2.VideoCapture(0)



filename = 'droplet_variable_output.mp4'


# camera stats
'''
if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

else:
    print("Camera hasn't opened")
'''



# output testing



# optimize these paramaters with heuristics during calibration phase
calibration_time = 2
accumulator_size = 200000
dp = (CAMERA_RESOLUTION * 10 ** 6) / accumulator_size

# these get optimized throughout the program


maxPossibleRadius = 500
minPossibleRadius = 0


accum_threshold = 10
# to determine if we are tracking a circle (will be better optimized for less noisy input)


# figure out how to compute the gradient
# gradient computation:

scale = 1
delta = 0
ddepth = cv2.CV_16S
gradients = []
gradient_value = 140

# this is an estimate, will figure out how to calc soon
grad_threshold = 80

data = {}
data['radius'] = []
data['time_step'] = []

delta = 15
k_min_dist = 50

def log(data):
    print(data)
    df = pd.DataFrame(data)
    df.plot(kind='line', x='time_step', y='radius')
    plt.ylim(0, 100)
    plt.show()
    plt.savefig('droplet_results.png')
    print(df)

 

def channel_edges(lines):
    #analyze the lines that are closest to horizontal
    #among those determine which are close to parallel
    theta_thresh = 5

    
    #horizontal_lines = [line for line in lines[0] if abs(line[1] - 90) < theta_thresh]
    #for r1, t1, in 
    
    """
    theta_threshold = 5
    #pairs of parallel lines
    parallel_lines = []
    for r1, t1, in lines[0]:
        parallel_lines.extend([((r1,t1),(r2,t2)) for r2,t2 in lines[0] if abs(t1-t2) < theta_threshold])

    #filter out pairs of lines that aren't parallel enough (won't be necessary generally)
    while len(parallel_lines) > 1:
        parallel_lines= [filter(lambda x : x[0][0] - x[1][0] < theta_threshold, parallel_lines)]
        theta_threshold -= 0.1
    """



#lines are vectors of the form [radius, theta]
def min_dist_lines(a1, b1, a2, b2):
    
    proj_a1_v = abs(np.dot(a1 - a2, b2 - a2)) / (np.linalg.norm(b2 - a2) ** 2)

    dist_squared = np.linalg.norm(a1 - a2) ** 2  -  proj_a1_v ** 2

    return np.math.sqrt(dist_squared)

'''
with open('view_config.json') as f:
    view_config = json.load(f)
  '''
serial_number = "23280459"

info = None
for i in pylon.TlFactory.GetInstance().EnumerateDevices():
    if i.GetSerialNumber() == serial_number:
        info = i
        break
else:
    print('Camera with {} serial number not found'.format(serial_number))

# VERY IMPORTANT STEP! To use Basler PyPylon OpenCV viewer you have to call .Open() method on you camera
if info is not None:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
    camera.Open()

    viewer = BaslerOpenCVViewer(camera)
    #viewer.set_configuration(view_config)
    #viewer.show_interactive_panel(window_size=(width, height))


    #viewer.set_impro_function(detection, own_window=True)



    # make this proportional to frequency as well as
    minDist = 10

    gradient_value = 140

    # this is an estimate, will figure out how to calc soon
    grad_threshold = 100
    minRadius = 5
    maxRadius = 60


    lines = None
    # output_img = img
    # output_img1 = frame.copy()

    camera.AcquisitionFrameRate.SetValue(40)

    camera.PixelFormat.GetValue()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    #xfourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #out = cv2.VideoWriter(filename, xfourcc, 40, (width, height))

    current_tube_length = 0

    while camera.IsGrabbing():
        fgt_init()

        start = time.time()

        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            #bgr_img = frame = np.ndarray(shape=(height, width, 3), dtype=np.uint16)


            image = converter.Convert(grabResult)
            #frame = np.ndarray(buffer=image.GetBuffer(), shape=(image.GetHeight(), image.GetWidth(), 3), dtype=np.uint16)
            # bgr_img[:, :, 0] = frame[:, :, 2]
            #bgr_img[:, :, 1] = frame[:, :, 1]
            #bgr_img[:, :, 2] = frame[:, :, 0]

            
            
            output_img = image.GetArray()
            cv2.imshow('output', output_img)
            # output_img1 = frame.copy()

            # calibration

            # threshold to determine if a line is parallel or not

            if time.time() - start < calibration_time:

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

                edges = cv2.Canny(gray_img, 50, 300, 3)
                lines = cv2.HoughLinesP(edges, rho=1, theta = np.pi/180, threshold = 150, minLineLength = 100, maxLineGap = 20)
  

                if lines is not None:
                    #print(lines)

                    line_pairs_dots = np.zeros((len(lines), len(lines)))

                    #find the pairs of lines that are the most paralell
                    
                    for i in range(len(lines)):
                        line_1 = lines[i][0]
                        for j in range(len(lines)):
                            #print('hi')
                            if i == j:
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
                        cv2.line(output_img, (line_1[0] ,line_1[1]), (line_1[2], line_1[3]), (255, 0, 0), 2)
                        cv2.imshow('output_lines', output_img)

                    ##print(line_pairs_dots)
                    dot_pairs_indices= np.argsort(-1 * line_pairs_dots.flatten())[:5]
                    #print(dot_pairs_indices)
                    #print('NUM OPTIONS ', len(dot_pairs_indices))

                        


                        #go through the most parallel
                    for index in dot_pairs_indices.flatten():
                        i = index // len(lines)
                        j = index - (i * len(lines))


                        #print((i, j))
                        line_1 = lines[i][0]
                        line_2 = lines[j][0]

                        a1 = np.array(line_1[:2])
                        b1 = np.array(line_1[2:])

                        a2 = np.array(line_2[:2])
                        b2 = np.array(line_2[2:])


                        
                        #print(a1.shape)
                        tube_length = min_dist_lines(a1, b1, a2, b2)

                        #print("TUBE LENGTH", tube_length)
                        
                    
                        if tube_length >= current_tube_length:
                            current_tube_length = tube_length
                            cv2.line(output_img, (line_1[0], line_1[1]), (line_1[2], line_1[3]), (255,0, 0),2)
                            cv2.line(output_img, (line_1[0], line_1[1]), (line_1[2], line_1[3]), (0,0, 255),2)

                            current_tube_length = tube_length
                            v = b2 - a2
                            proj_a1_v = np.dot(a1 - a2, v) / (np.linalg.norm(v) ** 2)

                            #draw the perpendicular line

                            
                            cv2.line(output_img, (a1[0], a1[1]), (a2[0] + int((proj_a1_v  * v)[0]), a2[1] + int((proj_a1_v * v)[1])), (0, 255, 0), 2)
                            cv2.imshow("output_lines", output_img)

                ''' 
                edges = cv2.Canny(gray_img, 50, 250, apertureSize=3)
                linesP = cv2.HoughLinesP(edges, rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 100,maxLineGap = 10)


                if lines is not None:
                    print(lines)
                    for line in lnies:
                            cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.imshow("output_line", output_img)

                '''
                     
                     





            # need to figure out a faster way of denoising the image to get more circles
            # den_img = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            
            gray_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

            circles = cv2.HoughCircles(image=gray_img, method=cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                        param1=gradient_value, minRadius=minRadius, maxRadius=maxRadius)

            radius_sum = 0
            gradient_sum = 0
            sum_gradient_avg = 0
            grad_count = 0
            grad_circle_count = 0
            if circles is not None:
                minSensor, maxSensor = 0, 400

                # draw the circles
                for (x_d, y_d, r_d) in circles[0]:

                    if time.time() - start < calibration_time:
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
                            if circle_x1 < len(grad) and circle_y1 < len(grad[circle_x1]):
                                circle_grad_val = grad[circle_x1][circle_y1]
                                if circle_grad_val > grad_threshold:
                                    print("Circle grad val ", grad[circle_x1][circle_y1])
                                    gradient_sum += grad[circle_x1][circle_y1]
                                    grad1_count += 1

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

                if time.time() - start < calibration_time and sum_gradient_avg != 0:
                    gradients.append(sum_gradient_avg / grad_circle_count)
                    #print("Gradients ", gradients)
                    gradient_value = np.sum(gradients) / len(gradients)

                data['time_step'].append(time.time() - start)
                data['radius'].append(sum(circles[0][:, 2]) / len(circles[0]))

                fgt_set_customSensorRegulation(data['radius'][-1], 100, 400, 0)
                # update the paramaters for the circles
                minRadius = int(data['radius'][-1]) - delta
                maxRadius = int(data['radius'][-1]) + delta

                # bounds
                # if minRadius < minPossibleRadius:
                #     minRadius = minPossibleRadius + delta
                # if maxRadius > maxPossibleRadius:
                #     maxRadius = maxPossibleRadius - delta

                minDist = k_min_dist / (int(data['radius'][-1]))
                cv2.imshow('output_img', output_img)
               # out.write(output_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                grabResult.Release()
                camera.StopGrabbing()
                #out.release()
                cv2.destroyAllWindows()
                break
            
            
        grabResult.Release()

fgt_set_pressure(0, 0)
camera.Close()
fgt_close()

print(data)
df = pd.DataFrame(data)
df = df.sort_values(by='time_step', ascending=True)
print(df)

df.plot(kind='line',x='time_step', y='radius')
plt.ylim(0,100)
plt.show()

'''

def detection(img):

    # make this proportional to frequency as well as
    minDist = 10

    gradient_value = 140

    # this is an estimate, will figure out how to calc soon
    grad_threshold = 100
    minRadius = 5
    maxRadius = 60
    start = time.time()

    lines = None
    output_img = img
    # output_img1 = frame.copy()

    # calibration

    # threshold to determine if a line is parallel or not

    if time.time() - start < calibration_time:
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

        lines = cv2.HoughLines(grad, 1, np.pi/180, 150, 0, 0)


        print(lines)
        print(lines[0][0][0])
        # for rho, theta in lines[0]:
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     x1 = int(x0 + 1000 * (-b))
        #     y1 = int(y0 + 1000 * (a))
        #     x2 = int(x0 - 1000 * (-b))
        #     y2 = int(y0 - 1000 * (a))
        #
        #     cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)





    # need to figure out a faster way of denoising the image to get more circles
    # den_img = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    gray_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(image=gray_img, method=cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                param1=gradient_value, minRadius=minRadius, maxRadius=maxRadius)

    radius_sum = 0
    gradient_sum = 0
    sum_gradient_avg = 0
    grad_count = 0
    grad_circle_count = 0
    if circles is not None:

        # draw the circles
        for (x_d, y_d, r_d) in circles[0]:

            if time.time() - start < calibration_time:
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
                    if circle_x1 < len(grad) and circle_y1 < len(grad[circle_x1]):
                        circle_grad_val = grad[circle_x1][circle_y1]
                        if circle_grad_val > grad_threshold:
                            print("Circle grad val ", grad[circle_x1][circle_y1])
                            gradient_sum += grad[circle_x1][circle_y1]
                            grad1_count += 1

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

        if time.time() - start < calibration_time and sum_gradient_avg != 0:
            gradients.append(sum_gradient_avg / grad_circle_count)
            print("Gradients ", gradients)
            gradient_value = np.sum(gradients) / len(gradients)

        data['time_step'].append(time.time() - start)
        data['radius'].append(sum(circles[0][:, 2]) / len(circles[0]))

        # update the paramaters for the circles
        minRadius = int(data['radius'][-1]) - delta
        maxRadius = int(data['radius'][-1]) + delta

        # bounds
        # if minRadius < minPossibleRadius:
        #     minRadius = minPossibleRadius + delta
        # if maxRadius > maxPossibleRadius:
        #     maxRadius = maxPossibleRadius - delta

        minDist = k_min_dist / (int(data['radius'][-1]))

'''
'''
            cv2.imshow('gray_img', gray_img)
            cv2.imshow('output_img', output_img)
            out.write(output_img)
'''
'''      
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break
'''  
'''
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    '''


'''
main()
'''

