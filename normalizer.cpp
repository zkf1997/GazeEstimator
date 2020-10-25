#include "normalizer.hpp"

using namespace cv;
using namespace std;

namespace opengaze {

Normalizer::Normalizer() {
    // parameters for data normalization
    focal_norm_ = 960;
    distance_norm_ = 600; // 600 500 1000
    roiSize_norm_ = cv::Size(224, 224); // 224 448
    cam_norm_ = (Mat_<float>(3,3) << focal_norm_, 0, roiSize_norm_.width/2, 0, focal_norm_, roiSize_norm_.height/2.0f, 0, 0, 1.0f);
}

Normalizer::~Normalizer() {}

void Normalizer::setParameters(int focal_length, int distance, int img_w, int img_h){
    // parameters for data normalization
    focal_norm_ = focal_length;
    distance_norm_ = distance; // 600 500 1000
    roiSize_norm_ = cv::Size(img_w, img_h); // 224 448
    cam_norm_ = (Mat_<float>(3,3) << focal_norm_, 0, roiSize_norm_.width/2, 0, focal_norm_, roiSize_norm_.height/2.0f, 0, 0, 1.0f);
}

// convert vector from normalization space to camera coordinate system
cv::Mat Normalizer::cvtToCamera(cv::Point3f input, const Mat cnv_mat) {
    // convert to the original camera coordinate system
    Vec3f gaze_v(input.x, input.y, input.z);
    // apply de-normalization
    Mat gaze_v_cam = cnv_mat.inv() * Mat(gaze_v);
    gaze_v_cam = gaze_v_cam / norm(gaze_v_cam);

    return gaze_v_cam;
}

cv::Mat Normalizer::normalizeFace(Mat input_image, opengaze::Sample &sample, cv::Mat& warpMat) {
    // get the face center in 3D space
    Mat HR;
    cv::Rodrigues(sample.face_patch_data.head_r, HR);
    Mat HT = repeat(sample.face_patch_data.head_t, 1, 6);
    Mat Fc;
    add(HR*face_model_mat_.t(), HT, Fc);

    float distance = (float)norm(sample.face_patch_data.face_center); // original distance
    float z_scale = distance_norm_ / distance; // scaling factor
    cv::Mat scaleMat;
    scaleMat = (Mat_<float>(3,3) << 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, z_scale);// scaling matrix
    scaleMat.convertTo(scaleMat, CV_32F);

    // get the look_at matrix
    Mat hRx = HR.col(0);
    Mat forward = sample.face_patch_data.face_center /distance;
    Mat down = forward.cross(hRx);
    down = down / norm(down);
    Mat right = down.cross(forward);
    right = right / norm(right);

    // rotation matrix
    sample.face_patch_data.face_rot = Mat(3, 3, CV_32F);
    right.copyTo(sample.face_patch_data.face_rot.col(0));
    down.copyTo(sample.face_patch_data.face_rot.col(1));
    forward.copyTo(sample.face_patch_data.face_rot.col(2));
    sample.face_patch_data.face_rot = sample.face_patch_data.face_rot.t(); // there is no scaling
    sample.face_patch_data.face_rot.convertTo(sample.face_patch_data.face_rot, CV_32F);

    warpMat = cam_norm_ * (scaleMat * sample.face_patch_data.face_rot) * camera_matrix_.inv();// transformation matrix
    // crop image and copy the equalized image
    Mat face_patch;
    warpPerspective(input_image, face_patch, warpMat, roiSize_norm_);

    return face_patch;
}

vector<cv::Mat> Normalizer::normalizeEyes(cv::Mat input_image, Sample &sample){
    vector<cv::Mat> eye_images;

    Mat img_gray;
    cvtColor(input_image, img_gray, COLOR_BGR2GRAY);

    Mat eye_center;
    Mat* eye_rot;
    for (int i=0; i<2; ++i) {
        if (i==0){
            eye_center = sample.eye_data.leye_pos;
            eye_rot = &sample.eye_data.leye_rot;
        }
        else {
            eye_center = sample.eye_data.reye_pos;
            eye_rot = &sample.eye_data.reye_rot;
        }


        float distance = (float)norm(eye_center);
        float z_scale = distance_norm_ / distance;

        Mat scaleMat;
        scaleMat = (Mat_<float>(3,3) << 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, z_scale);// scaling matrix
        scaleMat.convertTo(scaleMat, CV_32F);

        // get the look_at matrix
        Mat HR;
        cv::Rodrigues(sample.face_patch_data.head_r, HR);
        Mat hRx = HR.col(0);
        Mat forward = eye_center/distance;
        Mat down = forward.cross(hRx);
        down = down / norm(down);
        Mat right = down.cross(forward);
        right = right / norm(right);

        // rotation matrix
        *eye_rot = Mat(3, 3, CV_32F);
        right.copyTo(eye_rot->col(0));
        down.copyTo(eye_rot->col(1));
        forward.copyTo(eye_rot->col(2));
        *eye_rot = eye_rot->t(); // there is no scaling

        Mat warpMat = cam_norm_ * (scaleMat * *eye_rot) * camera_matrix_.inv();// transformation matrix
        // crop image and copy the equalized image
        Mat eye_patch, eye_patch_equal;
        warpPerspective(img_gray, eye_patch, warpMat, roiSize_norm_);
        equalizeHist(eye_patch, eye_patch_equal);
        eye_images.push_back(eye_patch_equal);

    }
    eye_rot = nullptr;
    return eye_images;
}

void Normalizer::loadFaceModel(std::string path) {
    string face_model_file_path = path + "/content/model/face_model.yml";
    //
    cout << endl << "Loading 3D face model for head pose estimation from : " << face_model_file_path << endl;
    FileStorage fs;
    if (!fs.open(face_model_file_path, FileStorage::READ)) {
        cout << "Cannot load the 3D face model!" << endl;
        exit(EXIT_FAILURE);
    }
    fs["face_model"] >> face_model_mat_;
    for(int p=0; p<6; ++p)
        face_model_.emplace_back(Point3d(face_model_mat_.at<float>(0,p),
                                         face_model_mat_.at<float>(1,p),
                                         face_model_mat_.at<float>(2,p)));
    fs.release();
}

void Normalizer::setFaceModel(cv::Mat facemat)
{
    face_model_mat_ = facemat;
    face_model_mat_.convertTo(face_model_mat_, CV_32F);
    for (int p = 0; p < 6; ++p)
        face_model_.emplace_back(Point3d(face_model_mat_.at<float>(p, 0),
            face_model_mat_.at<float>(p, 1),
            face_model_mat_.at<float>(p, 2)));
}

// estimate head pose via model fitting
void Normalizer::estimateHeadPose(const Point2f *landmarks, opengaze::Sample &sample) {
    Mat zero_dist = Mat::zeros(1, 5, CV_64F);
    vector<Point2d> landmarks_orig(landmarks,
                                   landmarks + 6);
    cv::Mat head_r, head_t;
    camera_matrix_.convertTo(camera_matrix_, CV_64F); // input must be double type
    solvePnP(face_model_, landmarks_orig, camera_matrix_, zero_dist, head_r, head_t, false, SOLVEPNP_EPNP);
    solvePnP(face_model_, landmarks_orig, camera_matrix_, zero_dist, head_r, head_t, true);
    head_r.convertTo(sample.face_patch_data.head_r, CV_32F);
    head_t.convertTo(sample.face_patch_data.head_t, CV_32F);
    camera_matrix_.convertTo(camera_matrix_, CV_32F);

    // get the face center in 3D space
    Mat HR;
    cv::Rodrigues(sample.face_patch_data.head_r, HR);
    Mat HT = repeat(sample.face_patch_data.head_t, 1, 6);
    Mat Fc;
    add(HR*face_model_mat_.t(), HT, Fc);
    Mat two_eye_center = (Fc.col(0) + Fc.col(1) + Fc.col(2) + Fc.col(3)) / 4.0;
    Mat nose_center = (Fc.col(4) + Fc.col(5)) / 2.0;
    Mat face_center = (two_eye_center + nose_center) / 2.0; // face center
    face_center.copyTo(sample.face_patch_data.face_center); // copy to output
    sample.face_patch_data.face_center.convertTo(sample.face_patch_data.face_center, CV_32F);

    Mat le = 0.5*(Fc.col(2) + Fc.col(3)); // left eye
    le.copyTo(sample.eye_data.leye_pos);
    sample.eye_data.leye_pos.convertTo(sample.eye_data.leye_pos, CV_32F);
    Mat re = 0.5*(Fc.col(0) + Fc.col(1)); // right eye
    re.copyTo(sample.eye_data.reye_pos);
    sample.eye_data.reye_pos.convertTo(sample.eye_data.reye_pos, CV_32F);

}

void Normalizer::setCameraMatrix(cv::Mat input) {
    camera_matrix_ = input;
    camera_matrix_.convertTo(camera_matrix_, CV_32F);
}

}