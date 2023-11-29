#ifndef SEMANTIC_SLAM_GRID_PROCESSSOR_H
#define SEMANTIC_SLAM_GRID_PROCESSSOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <WebAPI.h>
#include <ConcurrentMap.h>
#include <ConcurrentSet.h>
#include <SLAM.h>

namespace EdgeSLAM {
	class KeyFrame;
}

namespace SemanticSLAM {
	class Plane;
	class Content;
	class Grid {
	public:
		Grid();
		Grid(int x, int y, int z, float gsize);
		virtual ~Grid();
	public:
		cv::Mat pos;
		int mnID;
		Plane* Floor;
		ConcurrentSet<EdgeSLAM::KeyFrame*> ConnectedKFs;
		ConcurrentSet<Content*> ConnectedVOs;
	private:
		
	};

	class GridProcessor {
	public:
		static void GridTest(EdgeSLAM::SLAM* SLAM, std::string user, int id, const cv::Mat& _img, const cv::Mat& _T, const cv::Mat& _invK);
		static void GridTest2(EdgeSLAM::SLAM* SLAM, std::string user, int id, const cv::Mat& _img, const cv::Mat& _T, const cv::Mat& _invK, int objID, cv::Point2f pt1, cv::Point2f pt4);


		static void ConvertIndex(cv::Mat X, int& xidx, int& yidx, int& zidx);
		static void CalcGridWithKF(EdgeSLAM::SLAM* SLAM, EdgeSLAM::KeyFrame* pKF);
		static void CalcGrid(EdgeSLAM::SLAM* SLAM, std::string user, int id, cv::Mat label);
		static Grid* GetGrid(cv::Mat X);
		static Grid* SearchGrid(int xidx, int yidx, int zidx);
		static std::vector<cv::Point2f> ProjectdGrid(int x, int y, int z, float gsize, cv::Mat K, cv::Mat R, cv::Mat t);
	private:
		static cv::Point2f ProjectCorner(int x, int y, int z, float gsize, cv::Mat K, cv::Mat R, cv::Mat t);
	public:
		static std::atomic<int> nGridID;
		static ConcurrentMap<EdgeSLAM::KeyFrame*, std::set<Grid*>> GlobalKeyFrameNGrids;
	
	private:
		static float GridSize;
		static ConcurrentMap<int, ConcurrentMap<int, ConcurrentMap<int, Grid*>*>*> GlobalGrids;
	};
}

#endif