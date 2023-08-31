/*
 * PnPProblem.cpp
 *
 *  Created on: Mar 28, 2014
 *      Author: Edgar Riba
 */

#include <iostream>
#include <sstream>

#include "PnPProblem.h"

#include <opencv2/calib3d/calib3d.hpp>

/* Functions for Möller-Trumbore intersection algorithm */
static cv::Point3f CROSS(cv::Point3f v1, cv::Point3f v2)
{
    cv::Point3f tmp_p;
    tmp_p.x =  v1.y*v2.z - v1.z*v2.y;
    tmp_p.y =  v1.z*v2.x - v1.x*v2.z;
    tmp_p.z =  v1.x*v2.y - v1.y*v2.x;
    return tmp_p;
}

static double DOT(cv::Point3f v1, cv::Point3f v2)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

static cv::Point3f SUB(cv::Point3f v1, cv::Point3f v2)
{
    cv::Point3f tmp_p;
    tmp_p.x =  v1.x - v2.x;
    tmp_p.y =  v1.y - v2.y;
    tmp_p.z =  v1.z - v2.z;
    return tmp_p;
}

/* End functions for Möller-Trumbore intersection algorithm */

// Function to get the nearest 3D point to the Ray origin
static cv::Point3f get_nearest_3D_point(std::vector<cv::Point3f> &points_list, cv::Point3f origin)
{
    cv::Point3f p1 = points_list[0];
    cv::Point3f p2 = points_list[1];

    double d1 = std::sqrt( std::pow(p1.x-origin.x, 2) + std::pow(p1.y-origin.y, 2) + std::pow(p1.z-origin.z, 2) );
    double d2 = std::sqrt( std::pow(p2.x-origin.x, 2) + std::pow(p2.y-origin.y, 2) + std::pow(p2.z-origin.z, 2) );

    if(d1 < d2)
    {
        return p1;
    }
    else
    {
        return p2;
    }
}

// Custom constructor given the intrinsic camera parameters

PnPProblem::PnPProblem()
{
    
}

PnPProblem::~PnPProblem()
{
    // TODO Auto-generated destructor stub
}


// Estimate the pose given a list of 2D/3D correspondences with RANSAC and the method to use

void PnPProblem::estimatePoseRANSAC( const std::vector<cv::Point3f> &list_points3d, // list with model 3D coordinates
                                     const std::vector<cv::Point2f> &list_points2d,     // list with scene 2D coordinates
                                     const cv::Mat& K, cv::Mat& rvec, cv::Mat& tvec,
                                     int flags, cv::Mat &inliers, int iterationsCount,  // PnP method; inliers container
                                     float reprojectionError, double confidence, bool useExtrinsicGuess)    // Ransac parameters
{
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);  // vector of distortion coefficients
    //cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);          // output rotation vector
    //cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);    // output translation vector

    //bool useExtrinsicGuess = false;   // if true the function uses the provided rvec and tvec values as
    // initial approximations of the rotation and translation vectors

    cv::solvePnPRansac( list_points3d, list_points2d, K, distCoeffs, rvec, tvec,
                        useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                        inliers, flags );

    //Rodrigues(rvec, R_matrix_); // converts Rotation Vector to Matrix
    //t_matrix_ = tvec;           // set translation matrix

    //this->set_P_matrix(R_matrix_, t_matrix_); // set rotation-translation matrix

}
