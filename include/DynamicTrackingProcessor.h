#ifndef SEMANTIC_SLAM_DYNAMIC_TRACKING_H
#define SEMANTIC_SLAM_DYNAMIC_TRACKING_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <SLAM.h>
#include <WebAPI.h>
#include <Utils.h>
#include <ConcurrentMap.h>
#include <ConcurrentSet.h>
#include <ConcurrentVector.h>

#include <opencv2/calib3d.hpp>
#include "PnPProblem.h"
#include "RobustMatcher.h"

namespace EdgeSLAM {
	class ObjectBoundingBox;
	class ObjectNode;
	class ObjectTrackingResult;
	class ObjectTrackingFrame;
	class Frame;
}
namespace SemanticSLAM {

	class DynamicTrackingProcessor {
	public:
		static RobustMatcher rmatcher;
		static cv::KalmanFilter KFilter;
		static PnPProblem pnp_detection, pnp_detection_est;
	public:
		static void Init();
		static int ObjectTracking(EdgeSLAM::SLAM* SLAM, std::string name,EdgeSLAM::ObjectBoundingBox* pNewBox, EdgeSLAM::ObjectNode* pObject, EdgeSLAM::ObjectTrackingResult* pTrackRes, const cv::Mat& newframe, const cv::Mat& K, cv::Mat& P);
		static int ObjectRelocalization(EdgeSLAM::ObjectBoundingBox* pNewBox, EdgeSLAM::ObjectNode* pObject, const cv::Mat& newframe, const cv::Mat& K, cv::Mat& P);
		static void PoseRelocalization(EdgeSLAM::ObjectBoundingBox* pNewBox, std::set<EdgeSLAM::ObjectBoundingBox*> setNeighBoxes, const cv::Mat& newframe, const cv::Mat& K, cv::Mat& P);
		static void MatchTestByFrame(EdgeSLAM::Frame* pNewFrame, std::set<EdgeSLAM::ObjectBoundingBox*> setNeighBoxes, const cv::Mat& newframe, const cv::Mat& K, cv::Mat& P);
		static int MatchTest(EdgeSLAM::ObjectBoundingBox* pNewBox, std::set<EdgeSLAM::ObjectBoundingBox*> setNeighBoxes, const cv::Mat& newframe, const cv::Mat& K, cv::Mat& P);
		static void MatchTest(EdgeSLAM::ObjectBoundingBox* pNewBox, EdgeSLAM::ObjectBoundingBox* pNeighBox, const cv::Mat& newframe, const cv::Mat& neighframe, const cv::Mat& K);
	private:
		//Parameter
		static bool mbFastMatch;
	private:
		//kalman filter
		static void initKalmanFilter(cv::KalmanFilter& KF, int nStates, int nMeasurements, int nInputs, double dt);
		static void updateKalmanFilter(cv::KalmanFilter& KF, cv::Mat& measurements, cv::Mat& translation_estimated, cv::Mat& rotation_estimated);
		static void fillMeasurements(cv::Mat& measurements, const cv::Mat& translation_measured, const cv::Mat& rotation_measured);
	private:
		//utils
		static void createFeatures(const std::string& featureName, int numKeypoints, cv::Ptr<cv::Feature2D>& detector, cv::Ptr<cv::Feature2D>& descriptor);
		static cv::Ptr<cv::DescriptorMatcher> createMatcher(const std::string& featureName, bool useFLANN);
		static cv::Mat rot2euler(const cv::Mat& rotationMatrix);
		static cv::Mat euler2rot(const cv::Mat& euler);
	private:
		//draw
		static void draw2DPoints(cv::Mat image, std::vector<cv::Point2f>& list_points, cv::Scalar color);
		static void draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f>& list_points2d);
		static void drawArrow(cv::Mat image, cv::Point2i p, cv::Point2i q, cv::Scalar color, int arrowMagnitude = 9, int thickness = 1, int line_type = 8, int shift = 0);
	};
}


#endif