#ifndef NORMALIZER_HPP
#define NORMALIZER_HPP

#include <opencv2/opencv.hpp>
#include "data.hpp"

namespace opengaze{
class Normalizer {

public:
    Normalizer();
    ~Normalizer();

    void estimateHeadPose(const cv::Point2f *landmarks, opengaze::Sample &sample);

    void setCameraMatrix(cv::Mat input);

    void loadFaceModel(std::string path);
    void setFaceModel(cv::Mat facemat);

    void setParameters(int focal_length, int distance, int img_w, int img_h);

    cv::Mat normalizeFace(cv::Mat input_image, Sample &sample);
    cv::Mat normalizeFace(cv::Mat input_image, Sample& sample, cv::Mat& warpMat);

    std::vector<cv::Mat> normalizeEyes(cv::Mat input_image, Sample &sample);

    cv::Mat cvtToCamera(cv::Point3f input, const cv::Mat cnv_mat);

private:
    cv::Mat camera_matrix_;
    std::vector<cv::Point3f> face_model_;
    cv::Mat face_model_mat_, cam_norm_;
    float focal_norm_, distance_norm_;
    cv::Size roiSize_norm_;
};


}




#endif //NORMALIZER_HPP
