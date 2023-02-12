#ifndef MARKER_PROCESSOR_H
#define MARKER_PROCESSOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <WebAPI.h>
#include <SLAM.h>
#include <KeyFrame.h>
#include <Utils.h>
#include <ArucoMarker.h>
#include <Marker.h>

namespace SemanticSLAM {

	class MarkerProcessor {
	public:
		MarkerProcessor();
		virtual ~MarkerProcessor();
	public:
		static void MarkerTest(EdgeSLAM::SLAM* SLAM, std::string user, int id, float len, float inc);
		static void DynamicObjectRegTest(EdgeSLAM::SLAM* SLAM, std::string user, int id, float len, float inc);
		static void DynamicObjectVisTest(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void CalculateInconsistency(EdgeSLAM::SLAM* SLAM, std::string user, int id, float len, float inc);
		static void SaveInconsistency();

		static void MarkerEventDetect(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void MarkerRegistration(EdgeSLAM::SLAM* SLAM, std::string user, int id, float len, float inc);

		//static std::vector<EdgeSLAM::KeyFrame*> MarkerGraphTraverse(int startID, int endID);
	public:
		//static ConcurrentMap<int, EdgeSLAM::KeyFrame*> MapMarkerKFs;
		static ConcurrentMap<int, std::set< EdgeSLAM::KeyFrame*>> MapMarkerKFs;
	};
}


#endif#pragma once
