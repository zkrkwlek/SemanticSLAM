#include <PlaneEstimator.h>
#include <random>
#include <Utils.h>
#include <User.h>
#include <Camera.h>
#include <SemanticProcessor.h>
#include <SemanticLabel.h>
#include <LabelInfo.h>

namespace SemanticSLAM {

	std::atomic<int> PlaneEstimator::nPlaneID = 0;
	ConcurrentMap<int, Plane*> PlaneEstimator::GlobalMapPlanes;
	ConcurrentMap<int, int> PlaneEstimator::GlobalNormalCount;
	ConcurrentMap<int, PlaneEstRes*> PlaneEstimator::GlobalNormalMPs;

	Plane* PlaneEstimator::GlobalFloor = new Plane();
	Plane* PlaneEstimator::GlobalCeil = new Plane();

	Plane::Plane():Node(),status(PlaneStatus::NOT_INITIALIZED), count(0), nScore(0), mnID(++PlaneEstimator::nPlaneID) {
		
	}
	Plane::~Plane(){
	
	}
	PlaneEstRes::PlaneEstRes():nData(0),nScore(0),bres(false), data(cv::Mat::zeros(0, 3, CV_32FC1)) {}//inlier(cv::Mat::zeros(0,3,CV_32FC1)), outlier(cv::Mat::zeros(0, 3, CV_32FC1))
	PlaneEstRes::~PlaneEstRes(){
		//inlier.release();
		//outlier.release();
		data.release();
		std::vector<EdgeSLAM::MapPoint*>().swap(vecMPs);
		std::set<EdgeSLAM::MapPoint*>().swap(setMPs);
	}
	void PlaneEstRes::UpdateData() {
		data = cv::Mat::zeros(0, 3, CV_32FC1);
		for (int i = 0; i < vecMPs.size(); i++) {

		}
	}
	void PlaneEstRes::ConvertData() {
		cv::Mat mat;
		cv::Mat ones = cv::Mat::ones(data.rows, 1, CV_32FC1);
		cv::hconcat(data, ones, mat);
		data = mat.clone();
	}

	void PlaneEstRes::ConvertWallData(cv::Mat Rsp) {
		cv::Mat temp = data*Rsp;
		cv::Mat a = temp.col(0);
		cv::Mat b = temp.col(2);
		cv::hconcat(a, b, data);
	}
	////벽 추정 용 변환 데이터 함수 추가하기

	void PlaneEstRes::SetData() {
		data = cv::Mat::zeros(0, 3, CV_32FC1);
		for (int i = 0; i < vecMPs.size(); i++) {
			auto pMPi = vecMPs[i];
			if (!pMPi)
				continue;
			if (pMPi->isBad() || pMPi->Observations() == 0)
				continue;
			data.push_back(pMPi->GetWorldPos().t());
		}
		nData = data.rows;
	}
	void PlaneEstRes::AddData(EdgeSLAM::MapPoint* pMP) {
		if (!pMP)
			return;
		if (pMP->isBad() || pMP->Observations() == 0)
			return;
		if (!setMPs.count(pMP)) {
			vecMPs.push_back(pMP);
			setMPs.insert(pMP);

		}
	}

	void PlaneEstRes::AddData(EdgeSLAM::MapPoint* pMP, const cv::Mat& t, int inc) {
		if (!pMP)
			return;
		if (pMP->isBad() || pMP->Observations() == 0)
			return;
		if (!setMPs.count(pMP)) {
			//data.push_back(pMP->GetWorldPos().t());
			data.push_back(t);
			vecMPs.push_back(pMP);
			setMPs.insert(pMP);
			if (inc > 0)
				pMP->mnPlaneCount++;
			nData++;
		}
	}

	void PlaneEstimator::Init() {
		for (int i = 0; i < HistSize; i++) {
			for (int j = 0; j < HistSize; j++) {
				int idx = ConvertSphericalToIndex(cv::Point(i, j));
				GlobalNormalCount.Update(idx, 0);
				GlobalNormalMPs.Update(idx, new PlaneEstRes());
			}
		}
	}

	PlaneEstimator::PlaneEstimator() {
		
		//GlobalFloor = new Plane();
	}
	PlaneEstimator::~PlaneEstimator() {}

	ConcurrentMap<EdgeSLAM::KeyFrame*, std::set<Plane*>> PlaneEstimator::mPlaneConnections;
	PlaneEstRes* PlaneEstimator::FloorAllData = new PlaneEstRes();
	PlaneEstRes* PlaneEstimator::CeilAllData = new PlaneEstRes();

	float PlaneEstimator::HoughBinAngle = 30.0;
	int PlaneEstimator::HistSize = 360 / HoughBinAngle;

	std::vector<cv::Point> fVec, wVec, cVec;
	void PlaneEstimator::SaveNormal() {
		{
			std::stringstream ssfile1;
			ssfile1 << "../bin/normal/floor.txt";
			std::ofstream f1;
			f1.open(ssfile1.str().c_str());
			for (int i = 0; i < fVec.size(); i++) {
				f1 << fVec[i].x << "," << fVec[i].y << std::endl;
			}
			f1.close();
		}
		{
			std::stringstream ssfile1;
			ssfile1 << "../bin/normal/wall.txt";
			std::ofstream f1;
			f1.open(ssfile1.str().c_str());
			for (int i = 0; i < wVec.size(); i++) {
				f1 << wVec[i].x << "," << wVec[i].y << std::endl;
			}
			f1.close();
		}
		{
			std::stringstream ssfile1;
			ssfile1 << "../bin/normal/ceil.txt";
			std::ofstream f1;
			f1.open(ssfile1.str().c_str());
			for (int i = 0; i < cVec.size(); i++) {
				f1 << cVec[i].x << "," << cVec[i].y << std::endl;
			}
			f1.close();
		}

	}
	/*void PlaneEstimator::PlaneEstimation(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		std::cout << "Estimatino!!!!" << std::endl << std::endl << std::endl;
	}*/

	float Plane::Distacne(cv::Mat X) {
		float a = this->param.at<float>(3);
		return ((float)this->normal.dot(X))+a;
	}

	bool Plane::CalcPosition(cv::Mat& Xp, float x, float y, cv::Mat Kinv, cv::Mat Tinv, cv::Mat O) {
		cv::Mat x3D = (cv::Mat_<float>(3, 1) << x,y, 1.0);
		cv::Mat Xw = Kinv * x3D;
		Xw.push_back(cv::Mat::ones(1, 1, CV_32FC1)); //3x1->4x1
		Xw = Tinv * Xw; // 4x4 x 4 x 1
		Xw = Xw.rowRange(0, 3) / Xw.at<float>(3); // 4x1 -> 3x1
		cv::Mat dir = Xw - O; //3x1
		float dist = param.at<float>(3);
		float a = -normal.dot(dir);
		if (std::abs(a) < 0.000001)
			return false;
		float u = (normal.dot(O) + dist) / a;
		Xp = O + dir * u;
		return true;
	}

	void PlaneEstimator::UpdateLocalMapPlanes (EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		//std::cout << "UpdateLocalMapPlanes start" << std::endl;
		auto pUser = SLAM->GetUser(user);
		if (!pUser){
			//std::cout << "UpdateLocalMapPlanes end1" << std::endl;
			return;
		}
		auto pKF = pUser->mpRefKF;
		if (!pKF){
			//std::cout << "UpdateLocalMapPlanes end2" << std::endl;
			return;
		}
		pUser->mnUsed++;
		////로컬맵의 KF 얻기
		std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(20);
		vpLocalKFs.push_back(pKF);
		auto pMap = SLAM->GetMap(pUser->mapName);

		//로컬맵의 평면 구성
		cv::Mat R = pKF->GetRotation();
		cv::Mat t = pKF->GetTranslation();
		
		Plane* floor = nullptr;
		Plane* ceil = nullptr;

		if (GlobalFloor->nScore > 0)
			floor = GlobalFloor;
		if (GlobalCeil->nScore > 0)
			ceil = GlobalCeil;
		std::set<Plane*> tempWallPlanes;
		for (std::vector<EdgeSLAM::KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			EdgeSLAM::KeyFrame* pKFi = *itKF;
			if (mPlaneConnections.Count(pKFi)) {
				auto tempPlanes = mPlaneConnections.Get(pKFi);
				for (auto iter = tempPlanes.begin(), iend = tempPlanes.end(); iter != iend; iter++) {
					auto plane = *iter;
					if (plane->type != PlaneType::WALL) {
						continue;
					}
					if (!tempWallPlanes.count(plane))
						tempWallPlanes.insert(plane);
				}
			}
		}

		////if (LocalMapPlanes.count(PlaneType::FLOOR))
		////	floor = *(LocalMapPlanes[PlaneType::FLOOR].begin());
		////if (LocalMapPlanes.count(PlaneType::CEIL))
		////	ceil = *(LocalMapPlanes[PlaneType::CEIL].begin());
		////if (LocalMapPlanes.count(PlaneType::WALL))
		////	tempWallPlanes = LocalMapPlanes[PlaneType::WALL];

		////////////////////수정 중
		//////if (tempWallPlanes.size() > 0 && (floor || ceil)) {
		//////	cv::Mat data = cv::Mat::zeros(1, 1, CV_32FC1);
		//////	float N = tempWallPlanes.size();
		//////	if (floor) {
		//////		N++;
		//////		cv::Mat pid = cv::Mat::zeros(1, 1, CV_32FC1);
		//////		pid.at<float>(0) = 0.0;
		//////		data.push_back(pid);
		//////		data.push_back(floor->param);
		//////	}
		//////	if (ceil) {
		//////		N++;
		//////		cv::Mat pid = cv::Mat::zeros(1, 1, CV_32FC1);
		//////		pid.at<float>(0) = 1.0;
		//////		data.push_back(pid);
		//////		data.push_back(ceil->param);
		//////	}
		//////	
		//////	////정면벽도 추가하면 변경하기
		//////	////바닥인지, 천장인지에 따라서 레이블 변경되어야 할 수 있음.
		//////	for (auto iter = tempWallPlanes.begin(), iend = tempWallPlanes.end(); iter != iend; iter++) {
		//////		auto wall = *iter;

		//////		float m;
		//////		cv::Mat param;
		//////		if (floor)
		//////			param = floor->param;
		//////		else
		//////			param = ceil->param;
		//////		cv::Mat Lw = CalcFlukerLine(param, wall->param);
		//////		cv::Mat line = LineProjection(R, t, Lw, pUser->mpCamera->Kfluker, m);
		//////		if (line.at<float>(0) < 0.0)
		//////			line *= -1.0;
		//////		int wid = 2;
		//////		if (line.at<float>(1) < 0.0)
		//////			wid = 4;
		//////		cv::Mat pid = cv::Mat::zeros(1, 1, CV_32FC1);
		//////		pid.at<float>(0) = wid;
		//////		data.push_back(pid);
		//////		data.push_back(wall->param);
		//////	}

		//////	data.at<float>(0) = N;
		//////	cv::Mat temp = cv::Mat::zeros(1000 - data.rows, 1, CV_32FC1);
		//////	data.push_back(temp);
		//////	{
		//////		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		//////		std::stringstream ss;
		//////		ss << "/Store?keyword=PlaneLine&id=" << id << "&src=" << user;
		//////		std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
		//////		auto res = mpAPI->Send(ss.str(), data.data, data.rows * sizeof(float));
		//////		std::chrono::high_resolution_clock::time_point e = std::chrono::high_resolution_clock::now();
		//////		delete mpAPI;
		//////	}
		//////}
		////////////////////수정 중

		//오직 바닥 또는 천장만 존재하는 경우 일단 테스트 용으로
		if (floor || ceil) {
			cv::Mat data = cv::Mat::zeros(1, 1, CV_32FC1);
			float N = 0;
			if (floor) {
				N++;
				cv::Mat pid = cv::Mat::zeros(1, 1, CV_32FC1);
				pid.at<float>(0) = 0.0;
				data.push_back(pid);
				data.push_back(floor->param);
				//std::cout <<"SEND PLANE LINE TEST = "<< N << " " << floor->param.t() << std::endl;
			}
			if (ceil) {
				N++;
				cv::Mat pid = cv::Mat::zeros(1, 1, CV_32FC1);
				pid.at<float>(0) = 1.0;
				data.push_back(pid);
				data.push_back(ceil->param);
			}
			for (auto iter = tempWallPlanes.begin(), iend = tempWallPlanes.end(); iter != iend; iter++) {
				auto plane = *iter;
				N++;
				cv::Mat pid = cv::Mat::zeros(1, 1, CV_32FC1);
				pid.at<float>(0) = (float)plane->mnID;
				data.push_back(pid);
				data.push_back(plane->param);
			}

			data.at<float>(0) = N;
			cv::Mat temp = cv::Mat::zeros(1000 - data.rows, 1, CV_32FC1);
			data.push_back(temp);
			{
				WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
				std::stringstream ss;
				ss << "/Store?keyword=PlaneLine&id=" << id << "&src=" << user;
				std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
				auto res = mpAPI->Send(ss.str(), data.data, data.rows * sizeof(float));
				std::chrono::high_resolution_clock::time_point e = std::chrono::high_resolution_clock::now();
				delete mpAPI;
			}
			//std::cout << "send plane line" << std::endl;
		}
		//오직 바닥 또는 천장만 존재하는 경우 일단 테스트 용으로
		//std::cout << "UpdateLocalMapPlanes end" << std::endl;
		pUser->mnUsed--;
		////라인 계산 후 보내는 경우
		//if (tempWallPlanes.size() > 0 && (floor || ceil)) {
		//	cv::Mat data = cv::Mat::zeros(1, 1, CV_32FC1);
		//	data.at<float>(0) = tempWallPlanes.size();

		//	if (floor) {
		//		for (auto iter = tempWallPlanes.begin(), iend = tempWallPlanes.end(); iter != iend; iter++) {
		//			auto wall = *iter;
		//			float m;
		//			cv::Mat  aram, wall->param);
		//			cv::Mat line = LineProjection(R, t, Lw, pUser->mpCamera->Kfluker, m);
		//			if (line.at<float>(0) < 0.0)
		//				line *= -1.0;

		//			//1 2 3 4 5 6 레이블 정보도 같이 넘기기
		//			//바닥과 벽, 천장과 벽, 벽은 수직벽인지 확인해야 함.
		//			//벽의 ID도 돌려 받아야 함.
		//			cv::Mat mL = cv::Mat::ones(1, 1, CV_32FC1);
		//			if (line.at<float>(1) < 0.0)
		//				mL += 1.0;
		//			//std::cout << "LABEL = " << mL <<line.t()<< std::endl;
		//			data.push_back(mL);
		//			data.push_back(Lw);
		//		}
		//	}
		//	if (ceil) {
		//		for (auto iter = tempWallPlanes.begin(), iend = tempWallPlanes.end(); iter != iend; iter++) {
		//			auto wall = *iter;
		//			float m;
		//			cv::Mat Lw = CalcFlukerLine(ceil->param, wall->param);
		//			cv::Mat line = LineProjection(R, t, Lw, pUser->mpCamera->Kfluker, m);
		//			if (line.at<float>(0) < 0.0)
		//				line *= -1.0;

		//			//1 2 3 4 5 6 레이블 정보도 같이 넘기기
		//			//바닥과 벽, 천장과 벽, 벽은 수직벽인지 확인해야 함.
		//			//벽의 ID도 돌려 받아야 함.
		//			cv::Mat mL = cv::Mat::ones(1, 1, CV_32FC1)*3;
		//			if (line.at<float>(1) < 0.0)
		//				mL += 1.0;
		//			//std::cout << "LABEL = " << mL <<line.t()<< std::endl;
		//			data.push_back(mL);
		//			data.push_back(Lw);
		//		}
		//	}

		////데이터 전송
		//	cv::Mat temp = cv::Mat::zeros(1000 - data.rows, 1, CV_32FC1);
		//	data.push_back(temp);
		//	{
		//		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		//		std::stringstream ss;
		//		ss << "/Store?keyword=PlaneLine&id=" << id << "&src=" << user;
		//		std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
		//		auto res = mpAPI->Send(ss.str(), data.data, data.rows * sizeof(float));
		//		std::chrono::high_resolution_clock::time_point e = std::chrono::high_resolution_clock::now();
		//		delete mpAPI;
		//	}
		//}

	}

	void PlaneEstimator::PlaneEstimation(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		//pUser->mnUsed++;
		//기기가 생성한 키프레임을 체크
		if (!pUser->KeyFrames.Count(id)){
			//pUser->mnUsed--;
			return;
		}
		auto pKF = pUser->KeyFrames.Get(id);
		if (!pKF)
			return;
		pUser->mnDebugPlane++;
		//키프레임의 자세 획득
		cv::Mat R = pKF->GetRotation();
		cv::Mat t = pKF->GetTranslation();

		//키프레임의 연결된 로컬 맵 획득
		std::vector<EdgeSLAM::MapPoint*> vpLocalMPs;
		auto spLocalKFs = pUser->mSetLocalKeyFrames.Get();
		auto vpLocalKFs = std::vector<EdgeSLAM::KeyFrame*>(spLocalKFs.begin(), spLocalKFs.end());

		/////이 이후로 실제 평면 쓸일이 없음.


		/////로컬 맵에서 맵포인트 모으기
		std::set<EdgeSLAM::MapPoint*> spMPs;
		Plane* maxFloor = nullptr;
		int maxFloorScore = 0;

		for (std::vector<EdgeSLAM::KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			EdgeSLAM::KeyFrame* pKFi = *itKF;
			if (!pKFi)
				continue;
			const std::vector<EdgeSLAM::MapPoint*> vpMPs = pKFi->GetMapPointMatches();

			for (std::vector<EdgeSLAM::MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
			{
				EdgeSLAM::MapPoint* pMPi = *itMP;
				if (!pMPi || pMPi->isBad() || spMPs.count(pMPi))
					continue;
				vpLocalMPs.push_back(pMPi);
				spMPs.insert(pMPi);
			}

			////평면 연결
			if (mPlaneConnections.Count(pKFi)) {
				auto setPlanes = mPlaneConnections.Get(pKFi);
				for (auto pter = setPlanes.begin(), pend = setPlanes.end(); pter != pend; pter++) {
					auto plane = *pter;
					if (plane->type == PlaneType::FLOOR) {
						if (plane->nScore > maxFloorScore) {
							maxFloorScore = plane->nScore;
							maxFloor = plane;
						}
					}
				}//for plane
			}
		}

		////로컬 맵의 맵포인트가 일정량이 넘어야 수행함.
		if (vpLocalMPs.size() < 100) {
			std::cout << "Plane::not enough map points in local map" << std::endl;
			//pUser->mnUsed--;
		
			pUser->mnDebugPlane--;
			return;
		}

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		PlaneEstRes* WallLocalData = new PlaneEstRes();

		//최소 N번 이상 레이블 된 맵포인트들을 모음.
		//한번은 우연하게 모일 수 있기 때문에
		//또한 그중에서 가장 높은 것에 할당함.
		int THRESH = 2;
		for (auto iter = vpLocalMPs.begin(); iter != vpLocalMPs.end(); iter++) {
			auto pMPi = *iter;// ->first;
			if (!pMPi || pMPi->isBad())
				continue;
			if (!pMPi->mpSemanticLabel)
				continue;
			auto pLabel = pMPi->mpSemanticLabel;
			
			if (pLabel->LabelCount.Count((int)StructureLabel::FLOOR) && pLabel->LabelCount.Get((int)StructureLabel::FLOOR) > THRESH) {
				FloorAllData->AddData(pMPi);
			}
			if (pLabel->LabelCount.Count((int)StructureLabel::WALL) && pLabel->LabelCount.Get((int)StructureLabel::WALL) > THRESH) {
				WallLocalData->AddData(pMPi);
			}
			if (pLabel->LabelCount.Count((int)StructureLabel::CEIL) && pLabel->LabelCount.Get((int)StructureLabel::CEIL) > THRESH) {
				CeilAllData->AddData(pMPi);
			}

		}//for
		FloorAllData->SetData();
		CeilAllData->SetData();
		WallLocalData->SetData();

		{
			
			auto mapNormalCount = GlobalNormalCount.Get();
			std::set<Plane*> sConnectedPlanes;
			////평면 추정
			//Plane* floor = nullptr;
			if (FloorAllData->data.rows > 50) {

				PlaneEstRes* fInlier = new PlaneEstRes();
				PlaneEstRes* fOutlier = new PlaneEstRes();
				bool bres = PlaneInitialization(FloorAllData, fInlier, fOutlier, 1500, 0.01, 0.2);
				if (bres) {
					 
					Plane* floor = new Plane();
					floor->type = PlaneType::FLOOR;
					floor->param = fInlier->param.clone();
					floor->normal = fInlier->param.rowRange(0, 3);
					floor->dist = cv::norm(floor->normal);
					floor->nScore = fInlier->data.rows;
					floor->count = 1;
					//sConnectedPlanes.insert(floor);
					auto pt = calcSphericalCoordinate(floor->normal.rowRange(0, 3));
					fVec.push_back(pt);
					int idx = ConvertSphericalToIndex(pt);
					auto c = GlobalNormalCount.Get(idx);
					GlobalNormalCount.Update(idx, ++c);

					if (floor->nScore > GlobalFloor->nScore) {
						delete GlobalFloor;
						GlobalFloor = floor;
					}
					else {
						GlobalFloor->count++;
					}

					/*{
						std::map<int, cv::Mat> mapDatas3;
						auto vecMPs = fInlier->vecMPs;
						for (size_t i = 0, iend = vecMPs.size(); i < iend; i++) {
							auto pMP = vecMPs[i];
							if (pMP->isBad())
								continue;
							if (pMP->mnPlaneCount > 10)
								mapDatas3[pMP->mnId] = pMP->GetWorldPos();
						}
						SLAM->TemporalDatas2.Update("GBAFloorOutlier", mapDatas3);
					}
					{
						std::map<int, cv::Mat> mapDatas4;
						auto vecMPs = fOutlier->vecMPs;
						for (size_t i = 0, iend = vecMPs.size(); i < iend; i++) {
							auto pMP = vecMPs[i];
							if (pMP->isBad())
								continue;
							if (pMP->mnPlaneCount > 10)
								mapDatas4[pMP->mnId] = pMP->GetWorldPos();
						}
						SLAM->TemporalDatas2.Update("GBAFloor", mapDatas4);
					}
					*/

					/*if (maxFloor) {
						if (maxFloor->nScore > floor->nScore) {
							floor = maxFloor;
							floor->count++;
						}
					}*/

					//std::cout << "Floor=" << GlobalFloor->param.t() <<"="<<pt<<"="<<c<<"=="<< FloorAllData->data.rows <<","<<fInlier->data.rows<<"="<< GlobalFloor->count<< std::endl;
				}

				delete fInlier;
				delete fOutlier;
			}
			if (CeilAllData->data.rows > 50) {
				PlaneEstRes* cInlier = new PlaneEstRes();
				PlaneEstRes* cOutlier = new PlaneEstRes();
				bool bres = PlaneInitialization(CeilAllData, cInlier, cOutlier, 1500, 0.01, 0.2);
				if (bres) {
					Plane* p = new Plane();
					p->type = PlaneType::CEIL;
					p->param = cInlier->param.clone();
					p->normal = p->param.rowRange(0, 3);
					p->dist = cv::norm(p->normal);
					p->nScore = cInlier->data.rows;
					p->count = 1;

					auto pt = calcSphericalCoordinate(p->normal.rowRange(0, 3));
					cVec.push_back(pt);

					if (p->nScore > GlobalCeil->nScore) {
						delete GlobalCeil;
						GlobalCeil = p;
					}
					else {
						GlobalCeil->count++;
					}

					//sConnectedPlanes.insert(p);
				}
			}
			if (WallLocalData->data.rows > 50 && GlobalFloor->nScore > 0) {
				cv::Mat Rsp = CalcPlaneRotationMatrix(GlobalFloor->param).clone();
				cv::Mat Rps = Rsp.t();

				PlaneEstRes* tempIn = new PlaneEstRes();
				PlaneEstRes* tempOut = new PlaneEstRes();

				PlaneEstRes* tempWallData = WallLocalData;
				/*cv::Mat temp = tempWallData->data*Rsp;
				cv::Mat a = temp.col(0);
				cv::Mat b = temp.col(2);
				cv::hconcat(a, b, tempWallData->data);*/
				tempWallData->ConvertWallData(Rsp);
				bool bres = PlaneInitialization(tempWallData, tempIn, tempOut, 1500, 0.01, 0.2);
				if (bres) {
					cv::Mat param = cv::Mat::zeros(4, 1, CV_32FC1);
					param.at<float>(0) = tempIn->param.at<float>(0);
					param.at<float>(2) = tempIn->param.at<float>(1);
					param.at<float>(3) = tempIn->param.at<float>(2);

					Plane* p = new Plane();
					p->type = PlaneType::WALL;
					p->param = param.clone();
					p->normal = p->param.rowRange(0, 3);
					p->dist = cv::norm(p->normal);
					p->nScore = tempIn->data.rows;
					p->count = 1;


					auto pt = calcSphericalCoordinate(p->normal.rowRange(0, 3));
					wVec.push_back(pt);
					int idx = ConvertSphericalToIndex(pt);
					auto c = GlobalNormalCount.Get(idx);
					GlobalNormalCount.Update(idx, ++c);
					{
						auto pRes = GlobalNormalMPs.Get(idx);
						auto vecMPs = tempIn->vecMPs;
						for (size_t i = 0, iend = vecMPs.size(); i < iend; i++) {
							auto pMP = vecMPs[i];
							if (pMP->isBad())
								continue;
							pRes->AddData(pMP);
						}
						pRes->nData = pRes->vecMPs.size();
						//std::cout << "WALL = " << p->param.t() << pt << "=" << c <<"="<<pRes->data.rows<< std::endl;
					}

					/*{
						std::map<int, cv::Mat> mapDatas4;
						auto vecMPs = tempIn->vecMPs;
						for (size_t i = 0, iend = vecMPs.size(); i < iend; i++) {
							auto pMP = vecMPs[i];
							if (pMP->isBad())
								continue;
							if (pMP->mnPlaneCount > 10)
								mapDatas4[pMP->mnId] = pMP->GetWorldPos();
						}
						SLAM->TemporalDatas2.Update("GBAFloorOutlier", mapDatas4);
						
					}*/
					
					//std::cout << "FLOOR = " << floor->param.t() << std::endl;
					//sConnectedPlanes.insert(p);
				}
			}
			//GlobalNormalCount.
			//mPlaneConnections.Update(pKF, sConnectedPlanes);
			////평면 정보 갱신
			
			///////바닥 성능 

			pUser->mnDebugPlane--;
			//pUser->mnUsed--;
			//SLAM->VisualizeImage(res, 3);
			return;
		}
	}
	cv::Point PlaneEstimator::calcSphericalCoordinate(cv::Mat normal) {
		cv::Mat a = normal.clone();

		cv::Mat b = normal.clone(); //0 y z -> x 0 z
		b.at<float>(1) = 0.0;

		cv::Mat c = cv::Mat::zeros(3, 1, CV_32FC1);
		c.at<float>(2) = -1.0; // 0 0 1

		float len_a = sqrt(a.dot(a));
		float len_b = sqrt(b.dot(b));
		float len_c = sqrt(c.dot(c));

		float azi = b.dot(c) / (len_b*len_c) * 180.0 / CV_PI;
		if (azi < 0.0)
			azi += 360.0;
		if (azi >= 360.0)
			azi -= 360.0;
		float ele = b.dot(a) / (len_a*len_b) * 180.0 / CV_PI;
		if (ele < 0.0)
			ele += 360.0;
		if (ele >= 360.0)
			ele -= 360.0;

		int x = (int)azi / HoughBinAngle;
		int y = (int)ele / HoughBinAngle;

		return cv::Point(x, y);
	}
	int PlaneEstimator::ConvertSphericalToIndex(cv::Point norm) {
		return norm.y*HistSize + norm.x;
	}
	cv::Point PlaneEstimator::ConvertIndexToSphericalNorm(int idx) {
		int x = idx % HistSize;
		int y = idx / HistSize;
		return cv::Point(x, y);
	}
	bool PlaneEstimator::calcUnitNormalVector(cv::Mat& X) {
		float sum = 0.0;
		int nDim = X.rows - 1;
		for (size_t i = 0; i < nDim; i++) {
			sum += (X.at<float>(i)*X.at<float>(i));
		}
		sum = sqrt(sum);
		if (sum != 0) {
			X /= sum;
			return true;
		}
		return false;
	}

	bool PlaneEstimator::PlaneInitialization(PlaneEstRes* src, PlaneEstRes* inlier, PlaneEstRes* outlier, int ransac_trial, float thresh_distance, float thresh_ratio) {
		
		int max_num_inlier = 0;
		cv::Mat best_plane_param;
		cv::Mat param, paramStatus;

		//초기 매트릭스 생성
		if (src->data.rows < 30)
			return false;
		cv::Mat ones = cv::Mat::ones(src->data.rows, 1, CV_32FC1);
		cv::Mat mMat;
		cv::hconcat(src->data, ones, mMat);

		int nDim = mMat.cols;
		int nDim2 = mMat.cols - 1;
		
		std::random_device rn;
		std::mt19937_64 rnd(rn());
		std::uniform_int_distribution<int> range(0, mMat.rows - 1);

		for (int n = 0; n < ransac_trial; n++) {

			cv::Mat arandomPts = cv::Mat::zeros(0, nDim, CV_32FC1);
			//select pts
			for (int k = 0; k < nDim2; k++) {
				int randomNum = range(rnd);
				cv::Mat temp = mMat.row(randomNum).clone();
				arandomPts.push_back(temp);
			}//select

			 //SVD
			cv::Mat X = EstimatePlaneParam(arandomPts, nDim2);
			//cv::Mat w, u, vt;
			//cv::SVD::compute(arandomPts, w, u, vt, cv::SVD::FULL_UV);
			//X = vt.row(nDim2).clone();
			//cv::transpose(X, X);

			//if (!calcUnitNormalVector(X)) {
			//	//std::cout << "PE::RANSAC_FITTING::UNIT Vector error" << std::endl;
			//}

			//cv::Mat checkResidual = abs(mMat*X) < thresh_distance;
			cv::Mat checkResidual = CalculateResidual(mMat, X, thresh_distance);

			int temp_inlier = cv::countNonZero(checkResidual);
			if (max_num_inlier < temp_inlier) {
				max_num_inlier = temp_inlier;
				param = X.clone();
				paramStatus = checkResidual.clone();
			}
		}//trial

		float planeRatio = ((float)max_num_inlier / mMat.rows);

		if (planeRatio > thresh_ratio) {

			/*for (int i = 0; i < src->data.rows; i++) {
				int checkIdx = paramStatus.at<uchar>(i);

				if (checkIdx == 0) {
					outlier->AddData(src->vecMPs[i]);
				}
				else {
					inlier->AddData(src->vecMPs[i],1);
				}
			}*/

			//한번 최적화하고 다시 나누기
			UpdateInlier(src, mMat, param, paramStatus, inlier, outlier,1);
			//param = EstimatePlaneParam(inlier->data, nDim2);
			inlier->param = param.clone();
			
			//std::cout << "1 = " << mMat.rows << " " << paramStatus.rows << std::endl;
			//PlaneEstRes* temp = new PlaneEstRes();
			//UpdateInlier(src, mMat, param, paramStatus, temp, outlier);
			//
			////temp->ConvertData();
			//cv::Mat param2 = EstimatePlaneParam(temp->data, nDim2);
			//cv::Mat residual = CalculateResidual(temp->data, param2, thresh_distance);
			//std::cout << "2 = " << temp->data.rows << " " << residual.rows << std::endl;
			//UpdateInlier(temp, temp->data, param2, residual, inlier, outlier,1);
			//std::cout << "test = " << param2.t()<<"   " <<inlier->data.rows<<", "<<outlier->data.rows<<"=="<< param2.type()<< std::endl;
			//delete temp;

			//inlier->param = param2.clone();
			return true;
		}
		else
		{
			//std::cout << "failed" << std::endl;
			return false;
		}
	}

	cv::Mat PlaneEstimator::EstimatePlaneParam(const cv::Mat& data, int dim) {
		//SVD
		cv::Mat w, u, vt;
		//std::cout << data.size() << ", " << dim << std::endl;
		cv::SVD::compute(data, w, u, vt, cv::SVD::FULL_UV);
		cv::Mat X = vt.row(dim).clone();
		//std::cout <<"asdf="<< X.t() << std::endl;
		cv::transpose(X, X);

		if (!calcUnitNormalVector(X)) {
			//std::cout << "PE::RANSAC_FITTING::UNIT Vector error" << std::endl;
		}
		return X.clone();
	}

	cv::Mat PlaneEstimator::CalculateResidual(const cv::Mat& data, const cv::Mat& param, float thresh) {
		cv::Mat checkResidual = abs(data*param) < thresh;
		return checkResidual.clone();
		////checkResidual = checkResidual / 255;
		//int temp_inlier = cv::countNonZero(checkResidual);

		//if (max_num_inlier < temp_inlier) {
		//	max_num_inlier = temp_inlier;
		//	param = X.clone();
		//	paramStatus = checkResidual.clone();
		//}
	}

	void PlaneEstimator::UpdateInlier(PlaneEstRes* src, const cv::Mat& data, const cv::Mat& param, const cv::Mat& residual, PlaneEstRes* inlier, PlaneEstRes* outlier,int inc) {
		for (int i = 0; i < residual.rows; i++) {
			int checkIdx = residual.at<uchar>(i);

			if (checkIdx == 0) {
				outlier->AddData(src->vecMPs[i], data.row(i));
			}
			else {
				inlier->AddData(src->vecMPs[i], data.row(i), inc);
			}
		}
	}


	bool PlaneEstimator::PlaneInitialization(const cv::Mat& src, cv::Mat& res, cv::Mat& matInliers, cv::Mat& matOutliers, int ransac_trial, float thresh_distance, float thresh_ratio) {

		//N
		//N-1
		//RANSAC
		int max_num_inlier = 0;
		cv::Mat best_plane_param;
		cv::Mat inlier;

		cv::Mat param, paramStatus;

		//초기 매트릭스 생성
		if (src.rows < 30)
			return false;
		cv::Mat ones = cv::Mat::ones(src.rows, 1, CV_32FC1);
		cv::Mat mMat;
		cv::hconcat(src, ones, mMat);

		std::random_device rn;
		std::mt19937_64 rnd(rn());
		std::uniform_int_distribution<int> range(0, mMat.rows - 1);

		for (int n = 0; n < ransac_trial; n++) {

			cv::Mat arandomPts = cv::Mat::zeros(0, 4, CV_32FC1);
			//select pts
			for (int k = 0; k < 3; k++) {
				int randomNum = range(rnd);
				cv::Mat temp = mMat.row(randomNum).clone();
				arandomPts.push_back(temp);
			}//select

			 //SVD
			cv::Mat X = EstimatePlaneParam(arandomPts, 3);;
			//cv::Mat w, u, vt;
			//cv::SVD::compute(arandomPts, w, u, vt, cv::SVD::FULL_UV);
			//X = vt.row(3).clone();
			//cv::transpose(X, X);

			//if (!calcUnitNormalVector(X)) {
			//	//std::cout << "PE::RANSAC_FITTING::UNIT Vector error" << std::endl;
			//}
			
			cv::Mat checkResidual = abs(mMat*X) < thresh_distance;
			//checkResidual = checkResidual / 255;
			int temp_inlier = cv::countNonZero(checkResidual);

			if (max_num_inlier < temp_inlier) {
				max_num_inlier = temp_inlier;
				param = X.clone();
				paramStatus = checkResidual.clone();
			}
		}//trial

		float planeRatio = ((float)max_num_inlier / mMat.rows);

		if (planeRatio > thresh_ratio) {

			for (int i = 0; i < src.rows; i++) {
				int checkIdx = paramStatus.at<uchar>(i);

				cv::Mat temp = src.row(i).clone();
				if (checkIdx == 0) {
					matOutliers.push_back(temp);
				}
				else {
					matInliers.push_back(temp);
				}
			}
			res = param.clone();
			return true;
		}
		else
		{
			//std::cout << "failed" << std::endl;
			return false;
		}
	}
	cv::Mat PlaneEstimator::CalcPlaneRotationMatrix(cv::Mat normal) {
		//euler zxy
		cv::Mat Nidealfloor = cv::Mat::zeros(3, 1, CV_32FC1);

		Nidealfloor.at<float>(1) = -1.0;
		float nx = normal.at<float>(0);
		float ny = normal.at<float>(1);
		float nz = normal.at<float>(2);

		float d1 = atan2(nx, -ny);
		float d2 = atan2(-nz, sqrt(nx*nx + ny*ny));
		cv::Mat R = Utils::RotationMatrixFromEulerAngles(d1, d2, 0.0, "ZXY");

		return R;
	}
	cv::Mat PlaneEstimator::CalcFlukerLine(cv::Mat P1, cv::Mat P2) {
		cv::Mat PLw1, Lw1, NLw1;
		PLw1 = P1*P2.t() - P2*P1.t();
		Lw1 = cv::Mat::zeros(6, 1, CV_32FC1);
		Lw1.at<float>(3) = PLw1.at<float>(2, 1);
		Lw1.at<float>(4) = PLw1.at<float>(0, 2);
		Lw1.at<float>(5) = PLw1.at<float>(1, 0);
		NLw1 = PLw1.col(3).rowRange(0, 3);
		NLw1.copyTo(Lw1.rowRange(0, 3));

		return Lw1;
	}
	cv::Mat PlaneEstimator::LineProjection(cv::Mat R, cv::Mat t, cv::Mat Lw1, cv::Mat K, float& m) {
		cv::Mat T2 = cv::Mat::zeros(6, 6, CV_32FC1);
		R.copyTo(T2.rowRange(0, 3).colRange(0, 3));
		R.copyTo(T2.rowRange(3, 6).colRange(3, 6));
		cv::Mat tempSkew = cv::Mat::zeros(3, 3, CV_32FC1);
		tempSkew.at<float>(0, 1) = -t.at<float>(2);
		tempSkew.at<float>(1, 0) = t.at<float>(2);
		tempSkew.at<float>(0, 2) = t.at<float>(1);
		tempSkew.at<float>(2, 0) = -t.at<float>(1);
		tempSkew.at<float>(1, 2) = -t.at<float>(0);
		tempSkew.at<float>(2, 1) = t.at<float>(0);
		tempSkew *= R;
		tempSkew.copyTo(T2.rowRange(0, 3).colRange(3, 6));
		cv::Mat Lc = T2*Lw1;
		cv::Mat Nc = Lc.rowRange(0, 3);
		cv::Mat res = K*Nc;
		float a = res.at<float>(0);
		float b = res.at<float>(1);
		float d = sqrt(a*a + b*b);
		res /= d;
		/*if (res.at<float>(0) < 0)
		res *= -1;
		if (res.at<float>(0) != 0)
		m = res.at<float>(1) / res.at<float>(0);
		else
		m = 9999.0;*/
		return res.clone();
	}
	cv::Point2f PlaneEstimator::GetLinePoint(float val, cv::Mat mLine, bool opt) {
		float x, y;
		if (opt) {
			x = 0.0;
			y = val;
			if (mLine.at<float>(0) != 0)
				x = (-mLine.at<float>(2) - mLine.at<float>(1)*y) / mLine.at<float>(0);
		}
		else {
			y = 0.0;
			x = val;
			if (mLine.at<float>(1) != 0)
				y = (-mLine.at<float>(2) - mLine.at<float>(0)*x) / mLine.at<float>(1);
		}

		return cv::Point2f(x, y);
	}
}