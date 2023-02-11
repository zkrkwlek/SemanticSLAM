#ifndef COORDINATE_INTEGRATION_H
#define COORDINATE_INTEGRATION_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <SLAM.h>
#include <WebAPI.h>
#include <ConcurrentMap.h>

namespace SemanticSLAM {
	class CoordinateIntegration {
	public:
		CoordinateIntegration();
		virtual ~CoordinateIntegration();
	public:
		static void Process(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void DownloadImage(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void DownloadPose(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void TestImageReturn(EdgeSLAM::SLAM* SLAM, std::string user, int id);
	public:
		static ConcurrentMap<int, cv::Mat> IMGs, Ps;
		static ConcurrentMap<int, cv::Mat> DeviceMaps;
	};
}


#endif