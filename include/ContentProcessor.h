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
#include <Node.h>

namespace SemanticSLAM {
	class Path {
	public:
		float speed;
		float length;
		bool bMove;
		std::chrono::high_resolution_clock::time_point start;

		cv::Mat s, e;
		void Init(cv::Mat start, cv::Mat end) {
			auto diffMat = end - start;
			length = sqrt(diffMat.dot(diffMat));
			speed = 0.2f;
			s = start;
			e = end;
		}
		void MoveStart() {
			start = std::chrono::high_resolution_clock::now();
			bMove=true;
		}
		cv::Mat Move() {
			if (bMove) {
				auto curr = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(curr - start).count();
				float t_path = duration / 1000.0;

				float distCovered = t_path*speed;
				float fractionOfJourney = distCovered / length;

				auto pos =  s + (e - s)*fractionOfJourney;

				if (fractionOfJourney > 0.99f) {
					MoveEnd();
				}
				return pos;
			}
		}
		void MoveEnd() {
			bMove = false;
		}
	};
	class Content : public EdgeSLAM::Node{
	public:
		Content();
		Content(const cv::Mat& _X, std::string _src, int _modelID, long long ts);
		virtual~Content();
	public:
		int mnID, mnNextID, mnMarkerID;
		std::atomic<long long> mnLastUpdatedTime; // 처음 생성시 마지막으로 갱신시
		cv::Mat data;
		/*cv::Mat pos;
		cv::Mat endPos;
		cv::Mat dir;
		cv::Mat attribute;*/
		bool mbMoving;
		int mnContentModelID;
		std::string src;
		Path* mpPath;
	};
	class ContentProcessor {
	public:
		static void SaveLatency();
		static void TestTouch(std::string user, int id);
		static void UpdateLatency(std::string keyword, std::string user, int id);
		/////////
		static void ResetContent(EdgeSLAM::SLAM* SLAM);
		static void ShareContent(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void ContentProcess(EdgeSLAM::SLAM* system, std::string user, int id, std::string kewword, int mid);
		static int ContentRegistration(EdgeSLAM::SLAM* SLAM, EdgeSLAM::KeyFrame* pKF, std::string user, const cv::Mat& data, int mid);
		static void DrawContentProcess(EdgeSLAM::SLAM* system, std::string user, int id, std::string kewword, int mid);
		static void DrawContentRegistration(EdgeSLAM::SLAM* SLAM, EdgeSLAM::KeyFrame* pKF, std::string user, cv::Mat data, int mid);
		static int MarkerContentRegistration(EdgeSLAM::SLAM* SLAM, EdgeSLAM::KeyFrame* pKF, std::string user, cv::Mat data, int id);
		static int PathContentRegistration(EdgeSLAM::SLAM* SLAM, int sid, int eid, std::string user, cv::Mat data, int mid);
		static Content* GetContent(int id);
		static void MovingObjectSync(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void ManageMovingObj(EdgeSLAM::SLAM* SLAM, int id);
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