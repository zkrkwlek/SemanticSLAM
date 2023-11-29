#ifndef SEMANTIC_SLAM_GRID_CELL_H
#define SEMANTIC_SLAM_GRID_CELL_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <ConcurrentMap.h>
#include <ConcurrentSet.h>
#include <atomic>

namespace EdgeSLAM {
	class KeyFrame;
}

namespace SemanticSLAM {
	class Plane;
	class GridFrame;
	class Label {
	public:
		Label();
		Label(int n);
		virtual ~Label();
		void Update(int label);
		int GetLabel();
		cv::Mat GetLabels();
		int Count(int label);

	private:
		int mnLabel;
		int mnCount;
		cv::Mat matLabels;
		std::mutex mMutexObject;
	};
	//평면들 정보도 알고 있으면 좋음
	
	class GridCell {
	public:
		GridCell();
		virtual ~GridCell();
	public:
		void AddObservation(GridFrame* pGF, int idx);
		void EraseObservation(GridFrame* pGF);
		void SetBadFlag();
		//bool isBad();
	public:
		std::atomic<bool> mbBad;
		cv::Point2f pt;
		ConcurrentMap<GridFrame*, int> mapObservation;
		Label *mpObject, *mpSegLabel;
	private:

	};

	class GridFrame {
	public:
		GridFrame();
		GridFrame(int row, int col);
		virtual ~GridFrame();
	public:
		void Copy(GridFrame* p);
		std::vector<std::vector<GridCell*>> mGrid; //row, col
	private:
	};
}


#endif