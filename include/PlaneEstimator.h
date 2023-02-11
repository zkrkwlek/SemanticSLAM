#ifndef SEMANTIC_SLAM_PLANE_ESTIMATOR_H
#define SEMANTIC_SLAM_PLANE_ESTIMATOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <SLAM.h>
#include <WebAPI.h>
#include <ConcurrentMap.h>
#include <ConcurrentSet.h>
#include <KeyFrame.h>
#include <MapPoint.h>

namespace SemanticSLAM {
	class Plane;
	enum class PlaneCorrelation {
		NO_RELATION,
		IDENTICAL,
		PARALLEL,
		ORTHOGONAL,
	};

	enum class PlaneType {
		FLOOR = 1,
		CEIL = 2,
		WALL = 3, //ī�޶� �������� �¿�յ� ǥ��
		WALL_2 = 4,
		WALL_3 = 5,
		WALL_4 = 6
	};
	enum class PlaneStatus {
		NOT_INITIALIZED=0,
		INITIALIZED=1,
	};

	class LocalStructModel {
	public:
		std::map<int, EdgeSLAM::KeyFrame*, std::less<int>> mapKFs;
		std::map<PlaneType, int> mapPlanes; //plane�� id�� ���
	};

	class PlaneEstRes {
	public:
		PlaneEstRes();
		virtual ~PlaneEstRes();
	public:
		void SetData(); //������ ������ ��Ʈ������ ����.
		void AddData(EdgeSLAM::MapPoint* pMP);
	public:
		bool bres;
		cv::Mat param;
		cv::Mat normal;
		float dist;
		cv::Mat data;
		std::vector<EdgeSLAM::MapPoint*> vecMPs;
		std::set<EdgeSLAM::MapPoint*> setMPs;
	};

	class Plane {
	public:
		Plane();
		virtual ~Plane();
	public:
		int mnID;
		cv::Mat param;
		cv::Mat normal;
		float dist;
		PlaneType type; //���� Ÿ�Գ��� ������ �ǵ����ϱ�.
		PlaneStatus status;
		uchar count; //N���� �ʱ�ȭ ���� ���� ������ �Ǿ�� ��.
		ConcurrentSet<EdgeSLAM::KeyFrame*> mKFObs; //local map�� kf�ʹ� �ٸ�.
		ConcurrentMap<PlaneType, std::set<Plane*>> mPlaneObs; //���� ��� ������ ������. ��� � ����. �ٴ�, õ��, ���� ���� 1��, �������� ������
		ConcurrentSet<EdgeSLAM::MapPoint*> mSetMPs;
		ConcurrentSet<int> mSetIdxs;
		bool label;
		cv::Mat line;

	private:
	};

	class PlaneEstimator {
	public:
		PlaneEstimator();
		virtual ~PlaneEstimator();
	public:
		static void PlaneEstimation(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		static void UpdateLocalMapPlanes(EdgeSLAM::SLAM* SLAM, std::string user, int id);
		//static void EstimateLocalMapPlanes(EdgeSLAM::SLAM* system, std::string user, int id);
	public:
		static bool PlaneInitialization(PlaneEstRes* src, PlaneEstRes* inlier, PlaneEstRes* outlier, int ransac_trial, float thresh_distance, float thresh_ratio);
		static bool PlaneInitialization(const cv::Mat& src, cv::Mat& res, cv::Mat& matInliers, cv::Mat& matOutliers, int ransac_trial, float thresh_distance, float thresh_ratio);
		static cv::Mat CalcPlaneRotationMatrix(cv::Mat normal);
		static cv::Mat CalcFlukerLine(cv::Mat P1, cv::Mat P2);
		static cv::Mat LineProjection(cv::Mat R, cv::Mat t, cv::Mat Lw1, cv::Mat K, float& m);
		static cv::Point2f GetLinePoint(float val, cv::Mat mLine, bool opt);
	private:
		static bool LineFunction(cv::Mat data, cv::Mat line) {
			cv::Mat proj = data*line;
			int a1 = cv::countNonZero((proj) > 0.0);
			int a2 = cv::countNonZero((proj) < 0.0);
			if (a1 > a2)
				return true;
			return false;
		}
	private:
		static float HoughBinAngle;
		static int HistSize;
		static bool calcUnitNormalVector(cv::Mat& X);
	public:
		static cv::Point calcSphericalCoordinate(cv::Mat normal);
		static int ConvertSphericalToIndex(cv::Point norm);
		static cv::Point ConvertIndexToSphericalNorm(int idx);
	public:

		static ConcurrentMap<EdgeSLAM::KeyFrame*, std::set<Plane*>> mPlaneConnections;
		static ConcurrentMap<int, int> GlobalNormalCount;
		
		static std::atomic<int> nPlaneID;
		static ConcurrentMap<int, Plane*> GlobalMapPlanes;
		//static ConcurrentMap<EdgeSLAM::KeyFrame*, LocalStructModel*> LocalKeyFrameModel;
		static PlaneEstRes *FloorAllData, *CeilAllData;
	};
}


#endif