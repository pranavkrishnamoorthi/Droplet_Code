import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# maybe use cypython to optimize further if needed

# fill this in based on camera used (megaPixels)
# current camera: Basler ace acA640-750uc Color USB 3.0 Camera
CAMERA_RESOLUTION = 0.30



cap = cv2.VideoCapture('droplet_variable_size.mp4')
filename = 'droplet_variable_output.mp4'

# camera stats
if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# output testing
xfourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(filename, xfourcc, fps, (width, height))

# optimize these paramaters with heuristics during calibration phase
calibration_time = 3
accumulator_size = 200000
dp = (CAMERA_RESOLUTION * 10 ** 6) / accumulator_size

# these get optimized throughout the program


maxPossibleRadius = 0
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
grad_threshold = 100

data = {}
data['radius'] = []
data['time_step'] = []

delta = 15
k_min_dist = 80

def log(data):
    print(data)
    df = pd.DataFrame(data)
    df.plot(kind='line', x='time_step', y='radius')
    plt.ylim(0, 100)
    plt.show()
    plt.savefig('droplet_results.png')
    print(df)



def channel_edges(lines):
    theta_threshold = 5
    #pairs of parallel lines
    parallel_lines = []
    for r1, t1, in lines[0]:
        parallel_lines.extend([((r1,t1),(r2,t2)) for r2,t2 in lines[0] if abs(t1-t2) < theta_threshold])

    #filter out pairs of lines that aren't parallel enough (won't be necessary generally)
    while len(parallel_lines) > 1:
        parallel_lines= [filter(lambda x : x[0][0] - x[1][0] < theta_threshold, parallel_lines)]
        theta_threshold -= 0.1



#lines are vectors of the form [radius, theta]
def min_dist_lines(l1, l2):
    dist = 0
    # #maxPossibleRadius is the distance between the top and the bottom lines
    # #(droplet should never exceed this)
    r1, theta1 = l1
    r2, theta2 = l2
    x1, y1 = r1 * np.cos(theta1), r1 * np.sin(theta1)
    x2, y2 = r2 * np.cos(theta2), r2 * np.cos(theta2)
    dist = max(int(np.math.sqrt((x2 - x1)**2 + (y2 - y1)**2)), maxPossibleRadius)
    # cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)


def main():


    # make this proportional to frequency as well as
    minDist = 10

    gradient_value = 140

    # this is an estimate, will figure out how to calc soon
    grad_threshold = 100
    minRadius = 5
    maxRadius = 60
    start = time.time()

    lines = None

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            output_img = frame.copy()
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

            cv2.imshow('gray_img', gray_img)
            cv2.imshow('output_img', output_img)
            out.write(output_img)

        print(fps)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    log(data)

main()
