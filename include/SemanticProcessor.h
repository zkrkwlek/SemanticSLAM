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
	class ObjectMapPoint;
	class Map;
	class SLAM;
}

namespace SemanticSLAM {

	enum class MovingObjectLabel {
		PERSON = 1,
		CHAIR = 57,
		REFRIGERATOR = 73,
		HANDBAG = 27,
		SUITCASE = 29,
		LAPTOP = 64,
		KEYBOARD = 67
	};

	enum class StructureLabel {
		WALL = 1,
		FLOOR = 4,
		CEIL = 6,
		CHAIR = 20,
		BUILDING = 2,//바닥 벽처럼 인식
		EARTH = 14,
		TABLE = 16
	};
	static ConcurrentSet<int> ObjectWhiteList; //맵 생성 리스트
	static ConcurrentMap<int, int> ObjectCandidateList; //노이즈 고려해서 체크하는 리스트, A이면 B가 되게. 수트이면 의자 고려
	static ConcurrentSet<int> LabelWhiteList;

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
		static int ObjectDynamicTracking(EdgeSLAM::SLAM* SLAM, std::string user, int id, EdgeSLAM::ObjectBoundingBox* pNewBB, std::set<EdgeSLAM::ObjectBoundingBox*> setNeighObjectBBs, cv::Mat& P, const cv::Mat& K, const cv::Mat& D);
		static void CheckDynamicObject(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void ObjectPreprocessing(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void ObjectMapping(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void ObjectTracking(EdgeSLAM::SLAM* SLAM, std::string user, int id, const cv::Mat& img);
		static void ObjectMapGeneration(EdgeSLAM::SLAM* SLAM, std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs, std::set<EdgeSLAM::ObjectBoundingBox*> spNewBBs, std::set<EdgeSLAM::ObjectBoundingBox*> setNeighObjectBBs, EdgeSLAM::Map* MAP);
		static void CreateObjectMapPoint(EdgeSLAM::KeyFrame* pKF1, EdgeSLAM::KeyFrame* pKF2, EdgeSLAM::ObjectBoundingBox* pBB1, EdgeSLAM::ObjectBoundingBox* pBB2, float minThresh, float maxThresh, EdgeSLAM::Map* pMap, EdgeSLAM::ObjectNode* pObjMap);
		static void MapPointCulling(EdgeSLAM::ObjectNode* map, unsigned long int nCurrentKFid);
		static void CreateBoundingBox(EdgeSLAM::SLAM* SLAM, std::string user, int id, EdgeSLAM::KeyFrame* pTargetKF, cv::Mat labeled);

		static void Init(EdgeSLAM::SLAM* _SLAM);
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
		static ConcurrentMap<int, int> GlobalObjectCount;
		static ConcurrentMap<int, int> GlobalLabelCount;
		//디텍션에서 바운딩박스 기록
		static ConcurrentMap<EdgeSLAM::KeyFrame*, std::set<EdgeSLAM::ObjectBoundingBox*>> GraphKeyFrameObjectBB;
		static ConcurrentMap<int, std::set<EdgeSLAM::ObjectBoundingBox*>> GraphFrameObjectBB;
	private:
		static std::string strLabel, strYoloObjectLabel;
		static EdgeSLAM::SLAM* SLAM;
	};
}

#endif