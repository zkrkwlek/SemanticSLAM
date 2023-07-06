#ifndef SEMANTIC_SLAM_H
#define SEMANTIC_SLAM_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <SLAM.h>
#include <WebAPI.h>
#include <ConcurrentMap.h>

namespace EdgeSLAM {
	class ObjectNode;
	class ObjectBoundingBox;
	class KeyFrame;
}

namespace SemanticSLAM {

	enum class MovingObjectLabel {
		PERSON = 1,
		CHAIR = 57
	};

	enum class StructureLabel {
		WALL = 1,
		FLOOR = 4,
		CEIL = 6
	};
	
	class ObjectLabel;
	class SemanticLabel;
	class SemanticProcessor {
	public:
		SemanticProcessor();
		virtual ~SemanticProcessor();
	public:
		/// <summary>
		/// 다이나믹 오브젝트 슬램
		/// </summary>
		static void ObjectMapping(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void ObjectTracking(EdgeSLAM::SLAM* SLAM, std::string user, int id);

		static void Init();
		static void DenseOpticalFlow(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void MultiViewStereo(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void ObjectUpdate(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void ObjectDetection(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void SimpleRecon(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void Segmentation(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void DownloadSuperPoint(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void ShareSemanticInfo(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void MatchingSuperPoint(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void LabelMapPoint(EdgeSLAM::SLAM* SLAM, std::string user, int id, const cv::Mat& label);
		static void SendLocalMap(EdgeSLAM::SLAM* SLAM, std::string user, int id);
	public:
		static ConcurrentMap<int, std::vector<cv::Point2f>> SuperPoints;
		static ConcurrentMap<int, cv::Mat> SemanticLabelImage;
		static ConcurrentMap<int, ObjectLabel*> ObjectLabels;
		static ConcurrentMap<int, SemanticLabel*> SemanticLabels;
		static std::vector<cv::Vec3b> SemanticColors;
		static std::vector<std::string> vecStrSemanticLabels, vecStrObjectLabels;

		static ConcurrentMap<EdgeSLAM::ObjectNode*, std::set<EdgeSLAM::KeyFrame*>> GraphObjectKeyFrame;
		static ConcurrentMap<EdgeSLAM::KeyFrame*, std::set<EdgeSLAM::ObjectNode*>> GraphKeyFrameObject;

		//디텍션에서 바운딩박스 기록
		static ConcurrentMap<EdgeSLAM::KeyFrame*, std::set<EdgeSLAM::ObjectBoundingBox*>> GraphKeyFrameObjectBB;
	private:
		static std::string strLabel, strYoloObjectLabel;
	};
}


#endif