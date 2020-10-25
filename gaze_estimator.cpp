#include "normalizer.hpp"

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

// OpenCV Headers
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

// C++ Standard Libraries
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <memory>
#include <ctime>
#include <filesystem>


//libtorch
#include <torch/script.h> // One-stop header.

using namespace dlib;
using namespace std;
using namespace cv;
using namespace torch::indexing;

void MatType(Mat inputMat)
{
    int inttype = inputMat.type();

    string r, a;
    uchar depth = inttype & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (inttype >> CV_CN_SHIFT);
    switch (depth) {
    case CV_8U:  r = "8U";   a = "Mat.at<uchar>(y,x)"; break;
    case CV_8S:  r = "8S";   a = "Mat.at<schar>(y,x)"; break;
    case CV_16U: r = "16U";  a = "Mat.at<ushort>(y,x)"; break;
    case CV_16S: r = "16S";  a = "Mat.at<short>(y,x)"; break;
    case CV_32S: r = "32S";  a = "Mat.at<int>(y,x)"; break;
    case CV_32F: r = "32F";  a = "Mat.at<float>(y,x)"; break;
    case CV_64F: r = "64F";  a = "Mat.at<double>(y,x)"; break;
    default:     r = "User"; a = "Mat.at<UKNOWN>(y,x)"; break;
    }
    r += "C";
    r += (chans + '0');
    cout << "Mat is of type " << r << " and should be accessed with " << a << endl;

}

void draw_gaze(cv::Mat& img, float pitch, float yaw)
{
    int h = img.size[0];
    int w = img.size[1];
    double length = w / 2.0;
    double dx = -length * sin(yaw) * cos(pitch);
    double dy = -length * sin(pitch);
    arrowedLine(img, Point(h / 2, w / 2),
        Point(round(h / 2 + dx), round(w / 2 + dy)), Scalar(0, 0, 255),
        2, LINE_AA, 0.2);
}

int main()
{
    string content_dir = std::filesystem::current_path().string() +  "/../content/";// path to the directory containing data
    string img_file = content_dir + "cam00.JPG";// input image
    string cam_file = content_dir + "cam00.xml";// camera calibration
    string face_file = content_dir + "face_model.txt";// 3d face model 
    string dat_file = content_dir + "shape_predictor_68_face_landmarks.dat"; // parameters of dlib face landmarks model
    string model_file = content_dir + "xgaze.pt";// gaze estimation model exported by pytorch

    Mat img = imread(img_file);
    cv_image<bgr_pixel> image(img);
    frontal_face_detector face_detector = get_frontal_face_detector();
    shape_predictor predictor;
    deserialize(dat_file) >> predictor;
    cout << "deserialized" << endl;
    std::vector<dlib::rectangle> detected_faces = face_detector(image, 1);
    cout << "detect finished" << endl;
    if (detected_faces.size() == 0)
    {
        cout << "no face detected" << endl;
        return 0;
    }
    cout << "face detected: " << detected_faces[0] << endl;
    full_object_detection shape = predictor(image, detected_faces[0]);
    std::vector<cv::Point> landmarks;
    for (int i = 0; i < shape.num_parts(); i++)
    {
        landmarks.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
    }

    // load camera info
    FileStorage fs;
    fs.open(cam_file, FileStorage::READ);
    Mat camera_matrix, camera_distortion;
    fs["Camera_Matrix"] >> camera_matrix;                                      // Read cv::Mat
    fs["Distortion_Coefficients"] >> camera_distortion;
    
    // load face model 
    std::ifstream face_fs(face_file);
    Mat face_model_load;
    std::string line;
    int rows = 0;
    cout << "start loading face model" << endl;
    while (!face_fs.eof())
    {
        getline(face_fs, line);
        if (line == "")
            break;
        std::istringstream stream(line);

        char sep = ' '; //comma!
        double x;
        // read *both* a number and a comma:
        while (stream >> x) {
            face_model_load.push_back(x);
        }
        rows++;
    }
    face_model_load = face_model_load.reshape(1, rows);
    std::vector<int> landmark_use = {20, 23, 26, 29, 15, 19};
    Mat face_model = Mat::zeros(6, 3, CV_64F);
    for (int i = 0; i < landmark_use.size(); i++)
    {
        face_model_load.row(landmark_use[i]).copyTo(face_model.row(i));
    }
    cout << "face model: " << face_model << endl;

    cout << "estimate head pose" << endl;
    ////estimate head pose
    opengaze::Normalizer normalizer;
    normalizer.setCameraMatrix(camera_matrix);
    normalizer.setFaceModel(face_model);
    opengaze::Sample sample;
    landmark_use = { 36, 39, 42, 45, 31, 35 };
    for (int p = 0; p < 6; p++)
    { 
        sample.face_data.landmarks[p] = landmarks[landmark_use[p]];
    }
    normalizer.estimateHeadPose(sample.face_data.landmarks, sample);
    cv::Mat warpMat;
    Mat img_normalized = normalizer.normalizeFace(img, sample, warpMat);
    imwrite("img_normed.png", img_normalized);
    
    //normalize landmarks
    std::vector<cv::Point2f> landmarks_used, landmarks_warped;
    for (int i = 0; i < 6; i++)
        landmarks_used.push_back(sample.face_data.landmarks[i]);
    perspectiveTransform(landmarks_used, landmarks_warped, warpMat);
    cout << "landmarks warped:" << landmarks_warped << endl;
    cout << "warp mat:" << warpMat << endl;

    //load model
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_file);
        //module.to(at::kCUDA); //uncomment to inference with gpu, first inference will be slower than cpu due to warmup
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::cout << "model loading ok\n";
    // inference
    // Create a vector of inputs.
    cv::Mat input_var;
    cv::cvtColor(img_normalized, input_var, cv::COLOR_BGR2RGB);
    input_var.convertTo(input_var, CV_32FC1);
    cv::Scalar img_mean = cv::Scalar(0.485, 0.456, 0.406) * 255;
    cv::Scalar img_std = cv::Scalar(0.229, 0.224, 0.225) * 255;
    input_var = (input_var - img_mean) / img_std;
    at::Tensor tensor_image = torch::from_blob(input_var.data, {1, input_var.rows, input_var.cols, 3}/*, options*/).permute({ 0, 3, 1, 2 });
    tensor_image = tensor_image.toType(at::kFloat);
    std::vector<torch::jit::IValue> inputs;
    //tensor_image = tensor_image.to(at::kCUDA);//if the model loaded is cuda, then input tensor need to be cuda
    inputs.push_back(tensor_image);

    // Execute the model and turn its output into a tensor.
    std::cout << "model inference:" << std::endl;
    at::Tensor output = module.forward(inputs).toTensor();
    float theta = output[0][0].item<float>();
    float phi = output[0][1].item<float>();
    cout << "predicted gaze(pitch, yaw): " << theta << ' ' << phi << endl;

    Point3f gaze_vec((-1.0f) * cos(theta) * sin(phi), 
        (-1.0f) * sin(theta),
        (-1.0f) * cos(theta) * cos(phi));
    cout << "predicted gaze vector: " << gaze_vec << endl;

    for (int i = 0; i < landmarks_warped.size(); i++)
        cv::circle(img_normalized, landmarks_warped[i], 5, Scalar(0, 0, 255), -1);
    draw_gaze(img_normalized, theta, phi);
    imwrite("output.png", img_normalized);
}

