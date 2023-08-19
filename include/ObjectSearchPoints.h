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
}
namespace SemanticSLAM {
	class ObjectSearchPoints {
	public:
		static EdgeSLAM::FeatureTracker* Matcher;
		static const int HISTO_LENGTH;
	public:
		static int SearchObject(const cv::Mat& f1, const cv::Mat& f2, std::vector<std::pair<int, int>>& matches, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri);
		static int SearchObjectNodeAndBox(EdgeSLAM::ObjectNode* pNode, EdgeSLAM::ObjectBoundingBox* pBox, std::vector<std::pair<int, int>>& matches, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri);
		static int SearchObjectBoxAndBoxForTriangulation(EdgeSLAM::ObjectBoundingBox* pBox1, EdgeSLAM::ObjectBoundingBox* pBox2, std::vector<std::pair<int, int>>& matches, const cv::Mat& F12, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri);
		static int SearchObjectBoxAndBoxForTracking(EdgeSLAM::ObjectBoundingBox* pBox1, EdgeSLAM::ObjectBoundingBox* pBox2, std::vector<std::pair<int, int>>& matches, float thMinDesc, float thProjection);
		//static int SearchObject(ObjectNode* obj, Frame* curr, std::vector<std::pair<int, int>>& matches, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri);

	};
}

#endif