/*
 * PnPProblem.h
 *
 *  Created on: Mar 28, 2014
 *      Author: Edgar Riba
 */

#ifndef PNPPROBLEM_H_
#define PNPPROBLEM_H_

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include "Mesh.h"
//#include "ModelRegistration.h"

class PnPProblem
{
public:
    explicit PnPProblem();  // custom constructor
    virtual ~PnPProblem();

    //bool backproject2DPoint(const Mesh *mesh, const cv::Mat& K, const cv::Point2f &point2d, cv::Point3f &point3d);
    //bool intersect_MollerTrumbore(Ray &R, Triangle &T, double *out);
    //std::vector<cv::Point2f> verify_points(Mesh *mesh, const cv::Mat& K);
    //cv::Point2f backproject3DPoint(const cv::Point3f &point3d, const cv::Mat& K);
    //bool estimatePose(const std::vector<cv::Point3f> &list_points3d, const std::vector<cv::Point2f> &list_points2d, const cv::Mat& K, int flags);
    void estimatePoseRANSAC( const std::vector<cv::Point3f> &list_points3d, const std::vector<cv::Point2f> &list_points2d, const cv::Mat& K, cv::Mat& rvec, cv::Mat& tvec, int flags, cv::Mat &inliers, int iterationsCount, float reprojectionError, double confidence, bool useExtrinsicGuess = false);

    /*cv::Mat get_R_matrix() const { return R_matrix_; }
    cv::Mat get_t_matrix() const { return t_matrix_; }
    cv::Mat get_P_matrix() const { return P_matrix_; }

    void set_P_matrix( const cv::Mat &R_matrix, const cv::Mat &t_matrix);*/

private:
    ///** The computed rotation matrix */
    //cv::Mat R_matrix_;
    ///** The computed translation matrix */
    //cv::Mat t_matrix_;
    ///** The computed projection matrix */
    //cv::Mat P_matrix_;
};

#endif /* PNPPROBLEM_H_ */
