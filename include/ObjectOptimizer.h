#ifndef SEMANTIC_SLAM_OBJECT_OPTIMIZER_H
#define SEMANTIC_SLAM_OBJECT_OPTIMIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <atomic>
#include <LoopClosingTypes.h>
/*
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/linear_solver_dense.h"
#include "g2o/types/types_seven_dof_expmap.h"
*/

namespace EdgeSLAM {
	class MapPoint;
	class Frame;
	class KeyFrame;
	class Map;
	class SLAM;
	class ObjectNode;
	class ObjectBoundingBox;
}
namespace SemanticSLAM{

	class ObjectOptimizer {
	public:
		void static ObjectMapAdjustment(EdgeSLAM::ObjectNode* pObjMap);
		int static ObjectPoseOptimization(EdgeSLAM::ObjectNode* pObj, EdgeSLAM::ObjectBoundingBox* pBox, std::vector<std::pair<int, int>> vecMatches);
		int static ObjectPoseOptimization(EdgeSLAM::ObjectBoundingBox* pBox, cv::Mat& P);

	};
}

#endif