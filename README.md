A rehabilitation training evaluation software based on action flow similarity comparison algorithm was designed to solve the problem of non-standard patient training without the guidance of a therapist during the rehabilitation training process. By recording standard rehabilitation training videos in advance, the patient's real-time rehabilitation training videos are processed through algorithms, compared with standard rehabilitation training videos, and training evaluation is given through the human-computer interaction page.  
  
In the context of active rehabilitation training, trainees should follow a set of standard rehabilitation exercises prescribed by a physician. Evaluating the training outcomes accurately involves real-time collection of the trainee's rehabilitation movements and comparing them with the standard exercises. Designing an effective algorithm for comparing the similarity of motion sequences is crucial in providing a reasonable assessment of the rehabilitation exercise proficiency. The following will elaborate on the algorithm for motion flow similarity comparison.  
  
**The algorthm is the combination of OpenPose, Dynamic Time Wrapping and Random Sample Consensus.**
  
**OpenPose**  
OpenPose is a sophisticated system that integrates advanced deep learning techniques and computer vision methodologies for precise human pose estimation and tracking. At its core are Convolutional Neural Networks (CNNs), a powerful class of deep learning models adept at extracting intricate features from images, essential for accurately recognizing patterns and structures related to human poses. The architecture employs a multi-stage approach, progressively refining pose estimations using higher resolutions and detailed information.  
A key innovation of OpenPose is the utilization of Part Affinity Fields (PAFs), which are vector fields indicating the likelihood of connections between body parts. These PAFs, combined with heatmap representations of body parts, enable the model to assemble a complete pose by associating and linking body joints. The system excels at real-time processing, thanks to optimized algorithms, parallel processing, and hardware acceleration. Additionally, it incorporates tracking algorithms and temporal information for robust and consistent pose estimation across video frames, effectively handling occlusions and tracking errors. OpenPose is engineered for hardware optimization, ensuring efficiency across a range of devices, from standard CPUs and GPUs to specialized accelerators. This amalgamation of technologies makes OpenPose a powerful tool for diverse applications, enabling comprehensive understanding and analysis of human movements and interactions.  
![image](https://github.com/ShawnHYSun/Images/blob/main/Pipeline.png)  
Figure 1. Overall Pipeline of OpenPose.  
  
**Dynamic Time Wrapping (DTW)**  
DTW was first used in speech recognition and got verified. Action flow signals and speech signals share significant similarities. They both exhibit specific patterns with minor variations based on the same pattern, demonstrating randomness and uncertainty. In the case of speech, different individuals speaking the same sentence produce varying sounds, yet we can understand the content. Similarly, in movements, different individuals performing the same action may have varying speeds and amplitudes, but we can still differentiate the action. Therefore, applying the DTW algorithm to motion sequence comparison is reasonable. The following simply explains the principles of the DTW algorithm.  
DTW is a technique that compares and aligns two sequences, considering variations in speed or timing. It aims to find the best match by allowing for slight shifts or warps in the sequences, making it useful for comparing signals like speech or motion, where slight variations are expected. The algorithm constructs a distance matrix and identifies the optimal path with the least cumulative distance, allowing for local differences and variations.  
![image](https://github.com/ShawnHYSun/Images/blob/main/DTW.png)  
Figure 2. DTW Schematic Diagram.  
The following illustrates the application of DTW on action flow similarity comparison. First use parameters that obtained by OpenPose to describe a posture. The angles between joint points are selected from the human body model as parameters to describe the human body's upper and lower limb posture. For the two postures 𝑋 {𝑎1 − 𝑎i} and 𝑌 {𝑏1 − 𝑏i}, 𝑎 and 𝑏 are the posture 𝑋, Y joint angles. Define action 𝑋, Y distance 𝑑(x, 𝑦) = Σwi|ai-bi|. Let 𝑋1, 𝑋2, 𝑋3, ⋯, 𝑋𝑀 represent the standard action flow sequence and 𝑌1, 𝑌2, 𝑌3, ⋯, 𝑌𝑁 represent the actual collected action flow sequence. 𝑀 and 𝑁 represent the length of the standard action flow sequence and the actual action flow sequence respectively. In order to find the correct time matching sequence, in actual operation we need to construct an M That is, the difference between the action at time i on the X sequence and the action at time j on the Y sequence. The smaller the distance, the higher the similarity between the two actions. Calculate all (i, j) to fill the entire matrix. The DTW algorithm boils down to finding the shortest distance path from point (0,0) to point (M, N) in the matrix. The i and j corresponding to the point passed by the path are the matching time points.  
  
**Random Sample Consensus (RANSAC)**  
RANSAC is an algorithm commonly used in computer vision and related fields to estimate parameters of a mathematical model from a dataset containing potential outliers. It does so by iteratively selecting random subsets of the data to fit a model and then evaluates the quality of the model based on how many data points agree with it within a defined threshold. This iterative process helps to robustly estimate the model even when a significant portion of the data is noisy or incorrect.  
RANSAC provides a reliable way to find the best-fit model parameters from a dataset by focusing on the most consistent data points while effectively disregarding outliers.  
  
**DTW&RANSAC Combined Algorithm**  
For rehabilitation training actions, it is assumed that all rehabilitation actions are at a uniform speed, so the motion flow sequence can be considered to be a linear model. The DTW algorithm will try to find the feature points corresponding to the two sequences. Therefore, even if the rehabilitation action is completely different from the standard action, the DTW algorithm will calculate the distance of an action stream and give the final score. Therefore, the RANSAC algorithm can be used as a priori algorithm using the DTW algorithm. That is, if the RANSAC algorithm is used to linearly fit the motion flow and a model that meets a certain accuracy cannot be obtained, it is considered that the training action mode is inconsistent with the standard action mode and the prescribed training action is not being performed, so there is no need to use the DTW algorithm for calculation. If the RANSAC algorithm obtains a model that meets a certain accuracy, it is considered that the training action pattern is consistent with the standard action pattern, is a linear movement, and the prescribed training action is being performed, and then uses the RANSAC algorithm combined with the time factor for scoring. The specific process is shown in the figure.  
![image](https://github.com/ShawnHYSun/Images/blob/main/Process.png)  
Figure 3. Process of Combined Algorithm.  


  
**References**  
[1]	Cao, Z., Simon, T., Wei, S.E. and Sheikh, Y., 2017. Realtime multi-person 2d pose estimation using part affinity fields. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7291-7299).  
[2]	Muda, L., Begam, M. and Elamvazuthi, I., 2010. Voice recognition algorithms using mel frequency cepstral coefficient (MFCC) and dynamic time warping (DTW) techniques. arXiv preprint arXiv:1003.4083.  
[3]	Keogh, E. and Ratanamahatana, C.A., 2005. Exact indexing of dynamic time warping. Knowledge and information systems, 7, pp.358-386.  
[4]	Fischler, M.A. and Bolles, R.C., 1981. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6), pp.381-395.  
