#ifndef SEMANTIC_LABEL_H
#define SEMANTIC_LABEL_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <SLAM.h>
#include <ConcurrentMap.h>

namespace SemanticSLAM {

	class SemanticLabel {
	public:
		SemanticLabel();
		virtual ~SemanticLabel();

	public:
		ConcurrentMap<int, int> LabelCount;
	private:

	};
}


#endif

