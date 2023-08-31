#include "DynamicObjectMap.h"
#include "KalmanFilter.h"

namespace SemanticSLAM {
	DynamicObjectMap::DynamicObjectMap() {
		_P = cv::Mat::eye(4, 4, CV_32FC1);
		int nStates = 18;            // the number of states
		int nMeasurements = 6;       // the number of measured states
		int nInputs = 0;             // the number of control actions
		double dt = 0.125;           // time between measurements (1/FPS)
		mpKalmanFilter = new KalmanFilter(nStates, nMeasurements, nInputs, dt);
	}
	DynamicObjectMap::~DynamicObjectMap() {
		delete mpKalmanFilter;
	}
	void DynamicObjectMap::SetPose(cv::Mat __P){
		std::unique_lock<std::mutex> lock(mMutexPose);
		_P = __P.clone();
	}
	cv::Mat DynamicObjectMap::GetPose() {
		std::unique_lock<std::mutex> lock(mMutexPose);
		return _P.clone();
	}
}

