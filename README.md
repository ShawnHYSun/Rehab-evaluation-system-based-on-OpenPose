# Rehab_Evaluation

This code serves the purpose of analyzing and comparing two rehab training video sources(1 standard video and 1 training video) by tracking and measuring the angles of various body parts (left arm, right arm, left leg, right leg) in each frame of the videos using OpenPose library to estimate the keypoint positions of various body parts in each frame of the two videos.

Let's break down its workflow step by step:

· Importing Libraries: The code starts by importing necessary libraries such as numpy, math, argparse, openpose, opencv, and matplotlib.

· Initialization: It initializes various variables and data structures, including angle_diffs, standard_dataset, testing_dataset, data1, data2, and n. These will be used to store angle differences and data from two video sources.

· Angle Calculation Function: The calculate_angle function is defined, which calculates the angle between three given points using trigonometry.

· Argument Parsing: It sets up argument parsing using argparse to accept paths for two video files (video_path1 and video_path2). These paths are provided as command-line arguments when the script is run.

· Custom Parameters: Custom parameters for the OpenPose library are defined in the params dictionary. These parameters specify the model folder location.

· OpenPose Initialization: OpenPose is configured and initialized using the specified parameters.

· Video Capture: The code opens two video files (video_path1 and video_path2) for processing using OpenCV's VideoCapture.

· Video Processing Loop: It enters a loop where frames from both video sources are read one by one. OpenPose is used to detect keypoints in each frame.

· Angle Calculation: For each frame, the code calculates angles for various body parts (left arm, right arm, left leg, right leg) in both video sources. It computes differences in these angles between the two videos and stores them.

· Data Storage: The calculated angles and angle differences are stored in various data structures (standard_dataset, testing_dataset, angle_diffs, data1, and data2).

· RANSAC Linear Fitting: The code uses RANSAC (Random Sample Consensus) regression to fit a linear model between data points from data1 and data2. It calculates inliers and an inliers ratio and visualizes the fitting result.

· DTW Scoring: Dynamic Time Warping (DTW) is used to calculate a similarity score between data1 and data2. The DTW distance matrix is computed, and backtracking is performed to find corresponding feature points. A similarity score is calculated based on these correspondences.

· GUI Display: A graphical user interface (GUI) using Tkinter is created. It includes a button to show the DTW similarity results.

· Similarity Results Display: When the "Show Similarity Results" button is pressed, it displays the DTW similarity score and the corresponding feature point matches in a messagebox.

· Cleanup: After processing is complete or if an error occurs, the video capture is released, and OpenCV windows are destroyed.

· Exception Handling: The code includes exception handling to handle errors gracefully.
