# GazeEstimator
A C++ implemented gaze estimator based on [ETH-XGaze](https://github.com/xucong-zhang/ETH-XGaze).

This program first reads an face image and normalize it. Then a previously trained model from [ETH-XGaze](https://github.com/xucong-zhang/ETH-XGaze) is loaded with libtorch and the normalized image is fed to the model for gaze estimation.

 
## Requirements
[opencv](https://opencv.org/)

[libtorch](https://pytorch.org/get-started/locally/)

[dlib](http://dlib.net/)   

This implementation is tested on windows 10 2004 with dlib 19.21, opencv 4.5.0 and libtorch 1.6.0 Release Version.

## usage
Please first set paths to dlib, opencv and libtorch accordingly in [CMakeLists.txt](./CMakeLists.txt). Then just build with CMake.

To use GPU for inference, please set USE_CUDA=true in [gaze_estimator.cpp](./gaze_estimator.cpp) to move the model and input to CUDA.

