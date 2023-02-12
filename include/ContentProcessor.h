#ifndef SEMANTIC_SLAM_CONTENT_PROCESSSOR_H
#define SEMANTIC_SLAM_CONTENT_PROCESSSOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <SLAM.h>
#include <WebAPI.h>
#include <ConcurrentMap.h>
#include <ConcurrentSet.h>
#include <KeyFrame.h>

namespace SemanticSLAM {

	class Content {
	public:
		Content();
		Content(cv::Mat _X, std::string _src, int _modelID);
		virtual~Content();
	public:
		int mnID, mnNextID, mnMarkerID;
		cv::Mat pos;
		cv::Mat endPos;
		cv::Mat dir;
		cv::Mat attribute;
		bool mbMoving;
		int mnContentModelID;
		std::string src;
	};
	class ContentProcessor {
	public:
		static void SaveLatency();
		static void TestTouch(std::string user, int id);
		static void UpdateLatency(std::string keyword, std::string user, int id);
		/////////
		static void ShareContent(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void ContentProcess(EdgeSLAM::SLAM* system, std::string user, int id, std::string kewword, int mid);
		static int ContentRegistration(EdgeSLAM::SLAM* SLAM, EdgeSLAM::KeyFrame* pKF, std::string user, cv::Mat data, int mid);
		static int MarkerContentRegistration(EdgeSLAM::SLAM* SLAM, EdgeSLAM::KeyFrame* pKF, std::string user, cv::Mat data, int id);
		static int PathContentRegistration(EdgeSLAM::SLAM* SLAM, int sid, int eid, std::string user, cv::Mat data, int mid);
		static Content* GetContent(int id);
		static void MovingObjectSync(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void UpdateProcess(EdgeSLAM::SLAM* system, std::string user, int id);
		/////////TEST
		static void DirectTest(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void IndirectTest(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void IndirectSend(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static ConcurrentMap<int, std::string> IndirectData;

		///////////////Anchor
		static void SetAnchor(EdgeSLAM::SLAM* system, std::string user, int id, std::string kewword, int mid);
		static void ShareAnchor(EdgeSLAM::SLAM* SLAM, std::string user, int id);
	public:
		static std::atomic<int> nContentID;
		static ConcurrentMap<int, Content*> AllContentMap;
		static ConcurrentMap<int, Content*> MarkerContentMap;
		static ConcurrentMap<EdgeSLAM::KeyFrame*, std::map<int, Content*>> ContentMap;
		static ConcurrentMap<int, cv::Mat> MapArucoMarkerPos;


		static ConcurrentMap<EdgeSLAM::KeyFrame*, std::map<int, std::string>> AnchorIDs;
	private:
		
	};
}


#endif