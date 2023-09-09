#ifndef SEMANTIC_SLAM_OBJECT_SEARCH_POINTS_H
#define SEMANTIC_SLAM_OBJECT_SEARCH_POINTS_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <WebAPI.h>
#include <ConcurrentMap.h>
#include <ConcurrentVector.h>
#include <ConcurrentSet.h>
namespace EdgeSLAM {
	class FeatureTracker;
	class ObjectNode;
	class ObjectBoundingBox;
	class MapPoint;
	class Frame;
}
namespace SemanticSLAM {
	class ObjectSearchPoints {
	public:
		static EdgeSLAM::FeatureTracker* Matcher;
		static const int HISTO_LENGTH;
	public:
		static int SearchBoxByBoW(EdgeSLAM::ObjectBoundingBox* pBB1, EdgeSLAM::ObjectBoundingBox* pBB2, std::vector<EdgeSLAM::MapPoint*>& vpMapPointMatches, float thMinDesc, float thMatchRatio, bool bCheckOri = true);
		static int SearchObjectMapByProjection(std::vector<std::pair<int, int>>& matches, EdgeSLAM::Frame* F, const std::vector<EdgeSLAM::MapPoint*>& vpLocalMapPoints, const std::set<EdgeSLAM::MapPoint*>& sAlreadyFound, cv::Mat P, cv::Mat origin, const float th, const int ORBdist, bool bCheckOri = true);
		static int SearchObjectMapByProjection(std::vector<std::pair<int, int>>& matches, EdgeSLAM::Frame* F, const std::vector<EdgeSLAM::MapPoint*>& vpLocalMapPoints, const std::set<EdgeSLAM::MapPoint*>& sAlreadyFound, cv::Mat Pcw, cv::Mat Pwo, cv::Mat origin, const float th, const int ORBdist, bool bCheckOri = true);
		static int SearchFrameByProjection(EdgeSLAM::Frame* pNewFrame, EdgeSLAM::ObjectBoundingBox* pKeyBox, const std::set<EdgeSLAM::MapPoint*>& sAlreadyFound, cv::Mat P, const float th, const int ORBdist, bool bCheckOri = true);
		static int SearchBoxByProjection(EdgeSLAM::ObjectBoundingBox* pNewBox, EdgeSLAM::ObjectBoundingBox* pKeyBox, const std::set<EdgeSLAM::MapPoint*>& sAlreadyFound, cv::Mat P, const float th, const int ORBdist, bool bCheckOri = true);
		static int SearchObject(const cv::Mat& f1, const cv::Mat& f2, std::vector<std::pair<int, int>>& matches, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri);
		static int SearchObjectNodeAndBox(EdgeSLAM::ObjectNode* pNode, EdgeSLAM::ObjectBoundingBox* pBox, std::vector<std::pair<int, int>>& matches, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri);
		static int SearchObjectBoxAndBoxForTriangulation(EdgeSLAM::ObjectBoundingBox* pBox1, EdgeSLAM::ObjectBoundingBox* pBox2, std::vector<std::pair<int, int>>& matches, const cv::Mat& F12, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri);
		static int SearchObjectBoxAndBoxForTracking(EdgeSLAM::ObjectBoundingBox* pBox1, EdgeSLAM::ObjectBoundingBox* pBox2, std::vector<std::pair<int, int>>& matches, float thMinDesc, float thProjection);
		static void ComputeThreeMaxima(std::vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3);
		//static int SearchObject(ObjectNode* obj, Frame* curr, std::vector<std::pair<int, int>>& matches, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri);

	};
}

#endif