#ifndef SEMANTIC_SLAM_GRID_PROCESSSOR_H
#define SEMANTIC_SLAM_GRID_PROCESSSOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <WebAPI.h>
#include <ConcurrentMap.h>
#include <ConcurrentSet.h>
#include <SLAM.h>

namespace SemanticSLAM {
	class GridProcessor {
	public:
		static void GridTest(EdgeSLAM::SLAM* SLAM, std::string user, int id, const cv::Mat& _img, const cv::Mat& _T, const cv::Mat& _invK);
		static void GridTest2(EdgeSLAM::SLAM* SLAM, std::string user, int id, const cv::Mat& _img, const cv::Mat& _T, const cv::Mat& _invK, int objID, cv::Point2f pt1, cv::Point2f pt4);
	private:

	};
}

#endif