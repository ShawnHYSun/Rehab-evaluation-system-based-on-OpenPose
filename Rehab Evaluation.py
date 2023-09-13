# Author: Haoyuan Sun|Shawn - Research Assistant at ECE department in University of Florida, Gainesville, FL

import sys
import cv2
import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
import tkinter as tk
from tkinter import messagebox

try:
    # Import Openpose
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(r'input your path')
        os.environ['PATH']  = os.environ['PATH'] + r';input your path' \
                                                   r'\input your path'
        import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this '
              'Python script in the right folder?')
        raise e

    try:
        # Array to store angle differences
        # We suppose that the data in the 1st video is standard dataset
        # and the data in the 2nd video is testing dataset.
        angle_diffs = []
        standard_dataset = []
        testing_dataset = []
        data1 = np.array([])    # data1 is the array for standard dataset
        data2 = np.array([])    # data2 is the array for testing dataset
        n = 1                   # counting number

        # Function to calculate angle between three points
        def calculate_angle(x1, y1, x2, y2, x3, y3):
            # Calculate dot product of vector P1P2 and vector P2P3
            dot_product = (x2 - x1) * (x3 - x2) + (y2 - y1) * (y3 - y2)

            # Calculate length of vector P1P2 and vector P2P3
            len1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            len2 = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

            # Calculate angle between vector P1P2 and vector P2P3
            angle_rad = math.acos(dot_product / (len1 * len2))

            # Convert radians to degrees
            angle_deg = math.degrees(angle_rad)
            return angle_deg

        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--video_path1", default="../input your path", help="Path to the first video file")
        parser.add_argument("--video_path2", default="../input your path", help="Path to the second video file")
        args = parser.parse_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../models/"

        # Initialize OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Open the video files
        cap1 = cv2.VideoCapture(args.video_path1)
        cap2 = cv2.VideoCapture(args.video_path2)

        while cap1.isOpened() and cap2.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            # Process frames from both videos
            datum1 = op.Datum()
            datum1.cvInputData = frame1
            opWrapper.emplaceAndPop(op.VectorDatum([datum1]))

            datum2 = op.Datum()
            datum2.cvInputData = frame2
            opWrapper.emplaceAndPop(op.VectorDatum([datum2]))

            # Display the frames with detected keypoints
            cv2.imshow("OpenPose - Video 1", datum1.cvOutputData)
            cv2.imshow("OpenPose - Video 2", datum2.cvOutputData)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # TODO: Calculate angles for left arm, right arm, left leg, right leg in 1st video
            left_shoulder_x1, left_shoulder_y1 = datum1.poseKeypoints[0][5][:2]
            left_elbow_x1, left_elbow_y1 = datum1.poseKeypoints[0][6][:2]
            left_wrist_x1, left_wrist_y1 = datum1.poseKeypoints[0][7][:2]
            left_arm_angle1 = calculate_angle(left_shoulder_x1, left_shoulder_y1, left_elbow_x1, left_elbow_y1,
                                                  left_wrist_x1, left_wrist_y1)

            right_shoulder_x1, right_shoulder_y1 = datum1.poseKeypoints[0][2][:2]
            right_elbow_x1, right_elbow_y1 = datum1.poseKeypoints[0][3][:2]
            right_wrist_x1, right_wrist_y1 = datum1.poseKeypoints[0][4][:2]
            right_arm_angle1 = calculate_angle(right_shoulder_x1, right_shoulder_y1, right_elbow_x1, right_elbow_y1,
                                                   right_wrist_x1, right_wrist_y1)

            left_hip_x1, left_hip_y1 = datum1.poseKeypoints[0][11][:2]
            left_knee_x1, left_knee_y1 = datum1.poseKeypoints[0][12][:2]
            left_ankle_x1, left_ankle_y1 = datum1.poseKeypoints[0][13][:2]
            left_leg_angle1 = calculate_angle(left_hip_x1, left_hip_y1, left_knee_x1, left_knee_y1, left_ankle_x1,
                                                  left_ankle_y1)

            right_hip_x1, right_hip_y1 = datum1.poseKeypoints[0][8][:2]
            right_knee_x1, right_knee_y1 = datum1.poseKeypoints[0][9][:2]
            right_ankle_x1, right_ankle_y1 = datum1.poseKeypoints[0][10][:2]
            right_leg_angle1 = calculate_angle(right_hip_x1, right_hip_y1, right_knee_x1, right_knee_y1,
                                                   right_ankle_x1, right_ankle_y1)

            # TODO: Calculate angles for the second video
            left_shoulder_x2, left_shoulder_y2 = datum2.poseKeypoints[0][5][:2]
            left_elbow_x2, left_elbow_y2 = datum2.poseKeypoints[0][6][:2]
            left_wrist_x2, left_wrist_y2 = datum2.poseKeypoints[0][7][:2]
            left_arm_angle2 = calculate_angle(left_shoulder_x2, left_shoulder_y2, left_elbow_x2, left_elbow_y2,
                                                  left_wrist_x2, left_wrist_y2)

            right_shoulder_x2, right_shoulder_y2 = datum2.poseKeypoints[0][2][:2]
            right_elbow_x2, right_elbow_y2 = datum2.poseKeypoints[0][3][:2]
            right_wrist_x2, right_wrist_y2 = datum2.poseKeypoints[0][4][:2]
            right_arm_angle2 = calculate_angle(right_shoulder_x2, right_shoulder_y2, right_elbow_x2, right_elbow_y2,
                                                   right_wrist_x2, right_wrist_y2)

            left_hip_x2, left_hip_y2 = datum2.poseKeypoints[0][11][:2]
            left_knee_x2, left_knee_y2 = datum2.poseKeypoints[0][12][:2]
            left_ankle_x2, left_ankle_y2 = datum2.poseKeypoints[0][13][:2]
            left_leg_angle2 = calculate_angle(left_hip_x2, left_hip_y2, left_knee_x2, left_knee_y2, left_ankle_x2,
                                                  left_ankle_y2)

            right_hip_x2, right_hip_y2 = datum2.poseKeypoints[0][8][:2]
            right_knee_x2, right_knee_y2 = datum2.poseKeypoints[0][9][:2]
            right_ankle_x2, right_ankle_y2 = datum2.poseKeypoints[0][10][:2]
            right_leg_angle2 = calculate_angle(right_hip_x2, right_hip_y2, right_knee_x2, right_knee_y2,
                                                   right_ankle_x2, right_ankle_y2)

            # Calculate the differences in angles between the two videos
            diff_left_arm_angle = abs(left_arm_angle1 - left_arm_angle2)
            diff_right_arm_angle = abs(right_arm_angle1 - right_arm_angle2)
            diff_left_leg_angle = abs(left_leg_angle1 - left_leg_angle2)
            diff_right_leg_angle = abs(right_leg_angle1 - right_leg_angle2)
            print("diff_left_arm_angle:{:.2f}, diff_right_arm_angle:{:.2f}, diff_left_leg_angle:{:.2f}, "
                  "diff_right_leg_angle:{:.2f}".format(diff_left_arm_angle, diff_right_arm_angle,
                                                       diff_left_leg_angle, diff_right_leg_angle))
            print('{0} times calculation'.format(n))

            # Append the data to the array
            standard_dataset.append((left_arm_angle1, right_arm_angle1, left_leg_angle1, right_leg_angle1))
            testing_dataset.append((left_arm_angle2, right_arm_angle2, left_leg_angle2, right_leg_angle2))
            angle_diffs.append((diff_left_arm_angle, diff_right_arm_angle, diff_left_leg_angle, diff_right_leg_angle))

            data_point1 = np.array([left_arm_angle1, right_arm_angle1, left_leg_angle1, right_leg_angle1])
            data_point2 = np.array([left_arm_angle2, right_arm_angle2, left_leg_angle2, right_leg_angle2])
            data1 = np.append(data1, data_point1)
            data2 = np.append(data2, data_point1)
            n += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # todo: RANSAC
        # reshape the array
        data1 = data1.reshape(-1, 4)
        data2 = data2.reshape(-1, 4)
        print('data1:', data1, '\n\n\n')
        print('data2:', data2, '\n\n\n')

        # Use RANSAC for linear fitting
        model = RANSACRegressor()
        model.fit(data1[:, 0].reshape(-1, 1), data1[:, 1])

        # Calculate inliers
        threshold = 0.1  # Threshold for prediction error
        predictions = model.predict(data2[:, 0].reshape(-1, 1))
        inliers = np.abs(predictions - data2[:, 1]) < threshold
        inliers_ratio = np.sum(inliers) / len(data2)

        # Visualize fitting result
        plt.scatter(data1[:, 0], data1[:, 1], color='blue', label='Data 1')
        plt.scatter(data2[:, 0], data2[:, 1], color='red', label='Data 2')
        plt.plot(data1[:, 0], model.predict(data1[:, 0].reshape(-1, 1)), color='green', label='RANSAC Fit')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('RANSAC Fit and Data')
        plt.show()

        # Output inliers ratio
        print("\nInliers Ratio:", inliers_ratio)


        # todo: DTW scoring
        def calculate_distance(point1, point2):
            # Calculate the distance between two data points
            return np.linalg.norm(point1 - point2)  # Euclidean distance


        def calculate_dtw_score(array1, array2):
            distance_matrix = np.zeros((len(array1), len(array2)))

            # Fill the distance matrix
            for i in range(len(array1)):
                for j in range(len(array2)):
                    distance_matrix[i, j] = calculate_distance(array1[i], array2[j])

            dtw_matrix = np.zeros((len(array1) + 1, len(array2) + 1))
            dtw_matrix[0, 1:] = np.inf
            dtw_matrix[1:, 0] = np.inf

            # Fill the DTW matrix
            for i in range(1, len(array1) + 1):
                for j in range(1, len(array2) + 1):
                    cost = distance_matrix[i - 1, j - 1]
                    dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

            dtw_score = dtw_matrix[-1, -1]

            # Backtrack to find corresponding feature points
            i, j = len(array1), len(array2)
            correspondences = []
            while i > 0 and j > 0:
                correspondences.append((i - 1, j - 1))
                min_value = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
                if min_value == dtw_matrix[i - 1, j]:
                    i -= 1
                elif min_value == dtw_matrix[i, j - 1]:
                    j -= 1
                else:
                    i -= 1
                    j -= 1

            correspondences.reverse()

            # Calculate similarity score based on corresponding feature points and DTW distance
            similarity_score = 1.0 / (1.0 + dtw_score)  # A simple scoring method, adjust as needed
            return similarity_score, correspondences

        # data
        data_array1 = data1
        data_array2 = data2

        # Calculate similarity score and corresponding feature points
        similarity_score, correspondences = calculate_dtw_score(data_array1, data_array2)

        print("Similarity Score:", similarity_score)
        print("Correspondences:", correspondences)

        # Function to show the similarity results using a messagebox
        def show_similarity_results():
            similarity_score, feature_point_matches = calculate_dtw_score(data_array1, data_array2)

            result_text = f"Similarity Score: {similarity_score:.2f}\n\nFeature Point Matches:\n"
            for idx, (point1, point2) in enumerate(feature_point_matches, start=1):
                result_text += f"Pair {idx}: Data Point 1: {point1}, Data Point 2: {point2}\n"

            # Create a messagebox to display the results
            messagebox.showinfo("Similarity Results", result_text)


        # Create the main tkinter window
        root = tk.Tk()
        root.title("DTW Similarity Results")

        # Create a button to show the similarity results
        show_results_button = tk.Button(root, text="Show Similarity Results", command=show_similarity_results)
        show_results_button.pack(padx=20, pady=20)

        # Run the main loop
        root.mainloop()

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(e)
        sys.exit(-1)
except Exception as e:
    print(e)
    sys.exit(-1)
