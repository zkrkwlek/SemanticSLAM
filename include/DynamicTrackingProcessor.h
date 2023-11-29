#ifndef SEMANTIC_SLAM_DYNAMIC_TRACKING_H
#define SEMANTIC_SLAM_DYNAMIC_TRACKING_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <SLAM.h>
#include <WebAPI.h>
#include <Utils.h>
#include <ConcurrentMap.h>
#include <ConcurrentSet.h>
#include <ConcurrentVector.h>

#include "PnPProblem.h"
#include "RobustMatcher.h"
#include "ThreadPool.h"

namespace EdgeSLAM {
	class ObjectBoundingBox;
	class ObjectNode;
	class ObjectTrackingResult;
	class ObjectTrackingFrame;
	class Frame;
	class KeyFrame;
	class Map;
}
namespace SemanticSLAM {
	class DynamicObjectMap;
	class DynamicTrackingProcessor {
	public:
		static RobustMatcher rmatcher;
		static PnPProblem pnp_detection, pnp_detection_est;
	public:
		static void Init();
		static void UpdateKalmanFilter(EdgeSLAM::ObjectNode* pObject, int nPnP, cv::Mat _Pcw, cv::Mat& _Pco, cv::Mat& Pwo);
		static void UpdateConfidence(EdgeSLAM::ObjectTrackingFrame* pFrame, cv::Mat Pcw, cv::Mat Pco, cv::Mat Pwo, cv::Mat Ow, cv::Mat K);
		static void ObjectMapping(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void ObjectTracking(ThreadPool::ThreadPool* POOL, EdgeSLAM::SLAM* SLAM, std::string user, EdgeSLAM::Frame* frame, const cv::Mat& img, int id);
		static void ObjectMapGeneration(EdgeSLAM::SLAM* SLAM, std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs, std::set<EdgeSLAM::ObjectBoundingBox*> spNewBBs, std::set<EdgeSLAM::ObjectBoundingBox*> setNeighObjectBBs, EdgeSLAM::Map* MAP);
		static void CreateObjectMapPoint(EdgeSLAM::KeyFrame* pKF1, EdgeSLAM::KeyFrame* pKF2, EdgeSLAM::ObjectBoundingBox* pBB1, EdgeSLAM::ObjectBoundingBox* pBB2, float minThresh, float maxThresh, EdgeSLAM::Map* pMap, EdgeSLAM::ObjectNode* pObjMap);
		static int ObjectRelocalization(EdgeSLAM::ObjectBoundingBox* pNewBox, EdgeSLAM::ObjectNode* pObject, EdgeSLAM::ObjectTrackingResult* pTrackRes, const cv::Mat& newframe, const cv::Mat& K, cv::Mat& P);
		static void drawBoundingBox(cv::Mat& imgage, const cv::Mat& Pco, const cv::Mat& K, float radx, float rady, float radz);
	private:
		//Parameter
		static bool mbFastMatch;
	private:
		//tracking
		static int ObjectTracking2(EdgeSLAM::SLAM* SLAM, EdgeSLAM::ObjectTrackingResult* pTrackRes, EdgeSLAM::Frame* frame, const cv::Mat& newframe, int fid, const cv::Mat& Pcw, const cv::Mat& K);

	private:
		//utils
		static void createFeatures(const std::string& featureName, int numKeypoints, cv::Ptr<cv::Feature2D>& detector, cv::Ptr<cv::Feature2D>& descriptor);
		static cv::Ptr<cv::DescriptorMatcher> createMatcher(const std::string& featureName, bool useFLANN);
	private:
		//draw
		static void draw2DPoints(cv::Mat image, std::vector<cv::Point2f>& list_points, cv::Scalar color);
		static void draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f>& list_points2d);
		static void drawArrow(cv::Mat image, cv::Point2i p, cv::Point2i q, cv::Scalar color, int arrowMagnitude = 9, int thickness = 1, int line_type = 8, int shift = 0);
	};
}


#endif