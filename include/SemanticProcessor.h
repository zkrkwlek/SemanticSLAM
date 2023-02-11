#ifndef SEMANTIC_SLAM_H
#define SEMANTIC_SLAM_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <SLAM.h>
#include <WebAPI.h>
#include <ConcurrentMap.h>

namespace SemanticSLAM {

	enum class StructureLabel {
		WALL = 1,
		FLOOR = 4,
		CEIL = 6
	};

	class SemanticLabel;
	class SemanticProcessor {
	public:
		SemanticProcessor();
		virtual ~SemanticProcessor();
	public:
		static void Init();
		static void DenseOpticalFlow(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void MultiViewStereo(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void ObjectDetection(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void Segmentation(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void DownloadSuperPoint(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void ShareSemanticInfo(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void MatchingSuperPoint(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void LabelMapPoint(EdgeSLAM::SLAM* SLAM, std::string user, int id, const cv::Mat& label);
	
	public:
		static ConcurrentMap<int, std::vector<cv::Point2f>> SuperPoints;
		static ConcurrentMap<int, cv::Mat> SemanticLabelImage;
		static ConcurrentMap<int, SemanticLabel*> SemanticLabels;
		static std::vector<cv::Vec3b> SemanticColors;
		static std::vector<std::string> vecStrSemanticLabels, vecStrObjectLabels;
	private:
		static std::string strLabel, strYoloObjectLabel;
	};
}


#endif