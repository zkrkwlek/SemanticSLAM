#include <PlaneEstimator.h>
#include <random>
#include <Utils.h>
#include <User.h>
#include <Camera.h>
#include <SemanticProcessor.h>
#include <SemanticLabel.h>

namespace SemanticSLAM {

	std::atomic<int> PlaneEstimator::nPlaneID = 0;
	ConcurrentMap<int, Plane*> PlaneEstimator::GlobalMapPlanes;
	ConcurrentMap<int, int> PlaneEstimator::GlobalNormalCount;

	Plane::Plane():status(PlaneStatus::NOT_INITIALIZED), count(0), mnID(++PlaneEstimator::nPlaneID) {
		
	}
	Plane::~Plane(){
	
	}
	PlaneEstRes::PlaneEstRes():bres(false), data(cv::Mat::zeros(0, 3, CV_32FC1)) {}//inlier(cv::Mat::zeros(0,3,CV_32FC1)), outlier(cv::Mat::zeros(0, 3, CV_32FC1))
	PlaneEstRes::~PlaneEstRes(){
		//inlier.release();
		//outlier.release();
		data.release();
		std::vector<EdgeSLAM::MapPoint*>().swap(vecMPs);
		std::set<EdgeSLAM::MapPoint*>().swap(setMPs);
	}
	void PlaneEstRes::SetData() {
		data = cv::Mat::zeros(0, 3, CV_32FC1);
		for (int i = 0; i < vecMPs.size(); i++) {
			auto pMPi = vecMPs[i];
			data.push_back(pMPi->GetWorldPos().t());
		}
	}
	void PlaneEstRes::AddData(EdgeSLAM::MapPoint* pMP) {
		if (!setMPs.count(pMP)) {
			data.push_back(pMP->GetWorldPos().t());
			vecMPs.push_back(pMP);
			setMPs.insert(pMP);
		}
	}


	PlaneEstimator::PlaneEstimator() {
		for (int i = 0; i < HistSize; i++) {
			for (int j = 0; j < HistSize; j++) {
				int idx = ConvertSphericalToIndex(cv::Point(i, j));
				GlobalNormalCount.Update(idx, 0);
			}
		}
	}
	PlaneEstimator::~PlaneEstimator() {}

	ConcurrentMap<EdgeSLAM::KeyFrame*, std::set<Plane*>> PlaneEstimator::mPlaneConnections;
	PlaneEstRes* PlaneEstimator::FloorAllData = new PlaneEstRes();
	PlaneEstRes* PlaneEstimator::CeilAllData = new PlaneEstRes();

	float PlaneEstimator::HoughBinAngle = 30.0;
	int PlaneEstimator::HistSize = 360 / HoughBinAngle;

	/*void PlaneEstimator::PlaneEstimation(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		std::cout << "Estimatino!!!!" << std::endl << std::endl << std::endl;
	}*/

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
		std::set<Plane*> tempWallPlanes;
		std::map<PlaneType, std::set<Plane*>> LocalMapPlanes;
		for (std::vector<EdgeSLAM::KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			EdgeSLAM::KeyFrame* pKFi = *itKF;
			if (mPlaneConnections.Count(pKFi)) {
				auto tempPlanes = mPlaneConnections.Get(pKFi);
				for (auto iter = tempPlanes.begin(), iend = tempPlanes.end(); iter != iend; iter++) {
					auto plane = *iter;
					LocalMapPlanes[plane->type].insert(plane);
				}
			}
		}

		if (LocalMapPlanes.count(PlaneType::FLOOR))
			floor = *(LocalMapPlanes[PlaneType::FLOOR].begin());
		if (LocalMapPlanes.count(PlaneType::CEIL))
			ceil = *(LocalMapPlanes[PlaneType::CEIL].begin());
		if (LocalMapPlanes.count(PlaneType::WALL))
			tempWallPlanes = LocalMapPlanes[PlaneType::WALL];

		////////////////수정 중
		//if (tempWallPlanes.size() > 0 && (floor || ceil)) {
		//	cv::Mat data = cv::Mat::zeros(1, 1, CV_32FC1);
		//	float N = tempWallPlanes.size();
		//	if (floor) {
		//		N++;
		//		cv::Mat pid = cv::Mat::zeros(1, 1, CV_32FC1);
		//		pid.at<float>(0) = 0.0;
		//		data.push_back(pid);
		//		data.push_back(floor->param);
		//	}
		//	if (ceil) {
		//		N++;
		//		cv::Mat pid = cv::Mat::zeros(1, 1, CV_32FC1);
		//		pid.at<float>(0) = 1.0;
		//		data.push_back(pid);
		//		data.push_back(ceil->param);
		//	}
		//	
		//	////정면벽도 추가하면 변경하기
		//	////바닥인지, 천장인지에 따라서 레이블 변경되어야 할 수 있음.
		//	for (auto iter = tempWallPlanes.begin(), iend = tempWallPlanes.end(); iter != iend; iter++) {
		//		auto wall = *iter;

		//		float m;
		//		cv::Mat param;
		//		if (floor)
		//			param = floor->param;
		//		else
		//			param = ceil->param;
		//		cv::Mat Lw = CalcFlukerLine(param, wall->param);
		//		cv::Mat line = LineProjection(R, t, Lw, pUser->mpCamera->Kfluker, m);
		//		if (line.at<float>(0) < 0.0)
		//			line *= -1.0;
		//		int wid = 2;
		//		if (line.at<float>(1) < 0.0)
		//			wid = 4;
		//		cv::Mat pid = cv::Mat::zeros(1, 1, CV_32FC1);
		//		pid.at<float>(0) = wid;
		//		data.push_back(pid);
		//		data.push_back(wall->param);
		//	}

		//	data.at<float>(0) = N;
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
		////////////////수정 중

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
		pUser->mnUsed++;
		//기기가 생성한 키프레임을 체크
		if (!pUser->KeyFrames.Count(id)){
			pUser->mnUsed--;
			return;
		}
		auto pKF = pUser->KeyFrames.Get(id);
		auto pMap = SLAM->GetMap(pUser->mapName);

		//해당 키프레임의 세그멘테이션이 될 때까지 대기. 키프레임은 무조건 세그멘테이션이 되야 함.
		//세그멘테이션 후에 동작하도록 변경하는게 나을듯
		while (!SemanticProcessor::SemanticLabelImage.Count(id)) {
			continue;
		}
		auto label = SemanticProcessor::SemanticLabelImage.Get(id);

		//키프레임의 자세 획득
		cv::Mat R = pKF->GetRotation();
		cv::Mat t = pKF->GetTranslation();

		//키프레임의 연결된 로컬 맵 획득
		std::vector<EdgeSLAM::MapPoint*> vpLocalMPs;
		//std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(20);
		//vpLocalKFs.push_back(pKF);
		auto spLocalKFs = pUser->mSetLocalKeyFrames.Get();
		auto vpLocalKFs = std::vector<EdgeSLAM::KeyFrame*>(spLocalKFs.begin(), spLocalKFs.end());

		//로컬 모델 생성 = 이전 키프레임들로부터 일단 평면 모델 가져오기
		{
			LocalStructModel* pLocalModel = new LocalStructModel();
			std::map<int, Plane*> tempWallPlanes;
			Plane* floor = nullptr;
			{
				//for (std::vector<EdgeSLAM::KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
				//{
				//	auto pKFi = *itKF;
				//	pLocalModel->mapKFs[pKFi->mnId] = pKFi;

				//	if (!LocalKeyFrameModel.Count(pKFi))
				//		continue;
				//	auto pTempModel = LocalKeyFrameModel.Get(pKFi);
				//	for (auto iter = pTempModel->mapPlanes.begin(), iend = pTempModel->mapPlanes.end(); iter != iend; iter++) {
				//		auto ptype = iter->first;
				//		auto id = iter->second;
				//		auto plane = GlobalMapPlanes.Get(id);
				//		if (ptype == PlaneType::FLOOR || ptype == PlaneType::CEIL) {
				//			pLocalModel->mapPlanes[ptype] = id;
				//			if (ptype == PlaneType::FLOOR)
				//				floor = plane;
				//		}
				//		else {
				//			tempWallPlanes[id] = plane;
				//		}
				//	}
				//	//현재 키프레임과 방향 비교
				//	//std::cout<<"Dir = "<<pKFi->mnId<<"="<< calcSphericalCoordinate(pKFi->GetRotation().row(2).t())<<std::endl;
				//}

				/*if (floor) {
				for (auto iter = tempWallPlanes.begin(), iend = tempWallPlanes.end(); iter != iend; iter++) {
				auto wall = iter->second;
				float m;
				cv::Mat Lw = CalcFlukerLine(floor->param, wall->param);
				cv::Mat line = LineProjection(R, t, Lw, pUser->mpCamera->Kfluker, m);
				if (line.at<float>(0) < 0.0)
				line *= -1.0;
				wall->label = LineFunction(floor2D, line);

				}
				}*/
			}
		}
		
		
		//현재 키프레임 추가
		
		//if (pLocalModel->mapKFs.size() > 0) {
		//	auto prevKF =  pLocalModel->mapKFs.begin()->second;
		//	auto prevDir = prevKF->GetRotation().row(2);//calcSphericalCoordinate(prevKF->GetRotation().row(2).t());
		//	auto currDir = pKF->GetRotation().row(2);// calcSphericalCoordinate(pKF->GetRotation().row(2).t());
		//	std::cout <<"ANGLE = "<< cv::norm(prevDir) << " " << cv::norm(currDir) << " " << acos(currDir.dot(prevDir) / (cv::norm(prevDir)*cv::norm(currDir)))*180.0 / CV_PI<<std::endl;
		//	std::cout << "Dir = " << pKF->mnId <<", "<<prevKF->mnId<< "=" <<  " == "<< prevKF->GetRotation().row(2) <<"  "<< pKF->GetRotation().row(2) << std::endl;
		//}
		//pLocalModel->mapKFs[pKF->mnId] = pKF;
		
		
		/////로컬 맵에서 맵포인트 모으기
		std::set<EdgeSLAM::MapPoint*> spMPs;
		for (std::vector<EdgeSLAM::KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			EdgeSLAM::KeyFrame* pKFi = *itKF;
			const std::vector<EdgeSLAM::MapPoint*> vpMPs = pKFi->GetMapPointMatches();

			for (std::vector<EdgeSLAM::MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
			{
				EdgeSLAM::MapPoint* pMPi = *itMP;
				if (!pMPi || pMPi->isBad() || spMPs.count(pMPi))
					continue;
				vpLocalMPs.push_back(pMPi);
				spMPs.insert(pMPi);
			}
		}

		////로컬 맵의 맵포인트가 일정량이 넘어야 수행함.
		if (vpLocalMPs.size() < 100) {
			std::cout << "Plane::not enough map points in local map" << std::endl;
			pUser->mnUsed--;
			return;
		}

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		/*std::set<EdgeSLAM::MapPoint*> spFloorMPs, spWallMPs, spCeilMPs;
		std::vector<EdgeSLAM::MapPoint*> vpFloorMPs, vpWallMPs, vpCeilMPs;

		cv::Mat matFloorData = cv::Mat::zeros(0, 3, CV_32FC1);
		cv::Mat matWallData = cv::Mat::zeros(0, 3, CV_32FC1);
		cv::Mat matCeilData = cv::Mat::zeros(0, 3, CV_32FC1);*/
		PlaneEstRes* WallLocalData = new PlaneEstRes();

		////레이블링 된 맵 포인트 데이터
		std::vector<cv::Mat> localWallDatas;
		std::map<int, cv::Mat> labelDatas;
		if (SLAM->TemporalDatas2.Count("label"))
			labelDatas = SLAM->TemporalDatas2.Get("label");
		std::map<int, cv::Mat> mapDatas;
		if (SLAM->TemporalDatas2.Count("map"))
			mapDatas = SLAM->TemporalDatas2.Get("map");

		//최소 N번 이상 레이블 된 맵포인트들을 모음.
		//한번은 우연하게 모일 수 있기 때문에
		//또한 그중에서 가장 높은 것에 할당함.
		int THRESH = 2;
		for (auto iter = vpLocalMPs.begin(); iter != vpLocalMPs.end(); iter++) {
			auto pMPi = *iter;// ->first;
			if (!pMPi || pMPi->isBad())
				continue;
			if (!SemanticProcessor::SemanticLabels.Count(pMPi->mnId))
				continue;
			auto pLabel = SemanticProcessor::SemanticLabels.Get(pMPi->mnId);
			int n1 = 0;
			int n2 = 0;
			int n3 = 0;
			if (pLabel->LabelCount.Count((int)StructureLabel::FLOOR))
				n1 = pLabel->LabelCount.Get((int)StructureLabel::FLOOR);
			if (pLabel->LabelCount.Count((int)StructureLabel::WALL))
				n2 = pLabel->LabelCount.Get((int)StructureLabel::WALL);
			if (pLabel->LabelCount.Count((int)StructureLabel::CEIL))
				n3 = pLabel->LabelCount.Get((int)StructureLabel::CEIL);
			
			if (n1 > n2) {
				mapDatas[pMPi->mnId] = pMPi->GetWorldPos();
				labelDatas[pMPi->mnId] = cv::Mat::ones(1, 1, CV_8UC1)*(int)StructureLabel::FLOOR;
			}
			else if (n2 > n1) {
				mapDatas[pMPi->mnId] = pMPi->GetWorldPos();
				labelDatas[pMPi->mnId] = cv::Mat::ones(1, 1, CV_8UC1)*(int)StructureLabel::WALL;
			}

			if (pLabel->LabelCount.Count((int)StructureLabel::FLOOR) && pLabel->LabelCount.Get((int)StructureLabel::FLOOR) > THRESH) {
				FloorAllData->AddData(pMPi);
			}
			if (pLabel->LabelCount.Count((int)StructureLabel::WALL) && pLabel->LabelCount.Get((int)StructureLabel::WALL) > THRESH) {
				WallLocalData->AddData(pMPi);
				localWallDatas.push_back(pMPi->GetWorldPos());
			}
			if (pLabel->LabelCount.Count((int)StructureLabel::CEIL) && pLabel->LabelCount.Get((int)StructureLabel::CEIL) > THRESH) {
				CeilAllData->AddData(pMPi);
			}

		}//for
		//SLAM->TemporalDatas.Update("wall", localWallDatas);
		if (mapDatas.size() > 0) {
			SLAM->TemporalDatas2.Update("map", mapDatas);
			SLAM->TemporalDatas2.Update("label", labelDatas);
		}
		
		{

			auto mapNormalCount = GlobalNormalCount.Get();
			std::set<Plane*> sConnectedPlanes;
			////평면 추정
			Plane* floor = nullptr;
			if (FloorAllData->data.rows > 50) {
				PlaneEstRes* fInlier = new PlaneEstRes();
				PlaneEstRes* fOutlier = new PlaneEstRes();
				bool bres = PlaneInitialization(FloorAllData, fInlier, fOutlier, 1500, 0.02, 0.2);
				if (bres) {
					floor = new Plane();
					floor->type = PlaneType::FLOOR;
					floor->param = FloorAllData->param.clone();
					floor->normal = floor->param.rowRange(0, 3);
					floor->dist = cv::norm(floor->normal);
					sConnectedPlanes.insert(floor);
					auto pt = calcSphericalCoordinate(floor->normal.rowRange(0, 3));
					int idx = ConvertSphericalToIndex(pt);
					auto c = GlobalNormalCount.Get(idx);
					GlobalNormalCount.Update(idx, ++c);
					std::cout << "Floor Param =" << floor->param.t() <<" = "<<pt<<" = "<<c<< std::endl;
				}
			}
			if (CeilAllData->data.rows > 50) {
				PlaneEstRes* cInlier = new PlaneEstRes();
				PlaneEstRes* cOutlier = new PlaneEstRes();
				bool bres = PlaneInitialization(CeilAllData, cInlier, cOutlier, 1500, 0.02, 0.2);
				if (bres) {
					Plane* p = new Plane();
					p->type = PlaneType::CEIL;
					p->param = CeilAllData->param.clone();
					p->normal = p->param.rowRange(0, 3);
					p->dist = cv::norm(p->normal);
					sConnectedPlanes.insert(p);
				}
			}
			if (WallLocalData->data.rows > 50 && floor) {
				cv::Mat Rsp = CalcPlaneRotationMatrix(floor->param).clone();
				cv::Mat Rps = Rsp.t();

				PlaneEstRes* tempIn = new PlaneEstRes;
				PlaneEstRes* tempOut = new PlaneEstRes;

				PlaneEstRes* tempWallData = WallLocalData;
				cv::Mat temp = tempWallData->data*Rsp;
				cv::Mat a = temp.col(0);
				cv::Mat b = temp.col(2);
				cv::hconcat(a, b, tempWallData->data);
				bool bres = PlaneInitialization(tempWallData, tempIn, tempOut, 1500, 0.02, 0.2);
				if (bres) {
					cv::Mat param = cv::Mat::zeros(4, 1, CV_32FC1);
					param.at<float>(0) = tempWallData->param.at<float>(0);
					param.at<float>(2) = tempWallData->param.at<float>(1);
					param.at<float>(3) = tempWallData->param.at<float>(2);

					Plane* p = new Plane();
					p->type = PlaneType::WALL;
					p->param = param.clone();
					p->normal = p->param.rowRange(0, 3);
					p->dist = cv::norm(p->normal);

					//std::cout <<"WALL = "<< tempWallData->param.t() << std::endl;
					//std::cout << "FLOOR = " << floor->param.t() << std::endl;
					sConnectedPlanes.insert(p);
				}
			}
			//GlobalNormalCount.
			mPlaneConnections.Update(pKF, sConnectedPlanes);
			////평면 정보 갱신
			
			///////바닥 성능 


			pUser->mnUsed--;
			//SLAM->VisualizeImage(res, 3);
			return;
		}

		/////////
		////local map의 키프레임 set 생성
		////로컬 맵의 키프레임들이 가지고 있는 평면으로부터 모델을 생성.
		std::map<PlaneType, std::set<Plane*>> LocalMapPlanes;
		for (std::vector<EdgeSLAM::KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			EdgeSLAM::KeyFrame* pKFi = *itKF;
			if (mPlaneConnections.Count(pKFi)) {
				auto tempPlanes = mPlaneConnections.Get(pKFi);
				for (auto iter = tempPlanes.begin(), iend = tempPlanes.end(); iter != iend; iter++) {
					auto plane = *iter;
					LocalMapPlanes[plane->type].insert(plane);
				}
			}
		}
		/////////
		
		////평면 추정    
		//키프레임 묶음
		//해당 포인트 얻기
		//추정
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		
		////바닥 추정 & 연결
		cv::Mat encoded = pUser->ImageDatas.Get(id);
		cv::Mat res = cv::imdecode(encoded, cv::IMREAD_COLOR);
		std::set<Plane*> sConnectedPlanes;
		Plane* floor = nullptr;
		if (FloorAllData->data.rows > 50){
			PlaneEstRes* fInlier = new PlaneEstRes();
			PlaneEstRes* fOutlier = new PlaneEstRes();
			bool bres = PlaneInitialization(FloorAllData, fInlier, fOutlier, 1500, 0.02, 0.2);
			
			if (bres) {
				if (LocalMapPlanes.count(PlaneType::FLOOR)) {
					floor = *(LocalMapPlanes[PlaneType::FLOOR].begin());
					floor->count++;
					if (floor->status == PlaneStatus::NOT_INITIALIZED && floor->count > 5) {
						floor->status = PlaneStatus::INITIALIZED;
					}
					//std::cout << "FLOOR::Association" <<(int)floor->count<<" "<<floor->param.t()<< std::endl;

				}
				else {
					floor = new Plane();
					floor->type = PlaneType::FLOOR;
					std::cout << "FLOOR::Initialization" << std::endl;
				}
				////inlier mp 추가
				floor->param = FloorAllData->param.clone();
				floor->normal = floor->param.rowRange(0, 3);
				floor->dist = cv::norm(floor->normal);
				sConnectedPlanes.insert(floor);
			}
			delete fInlier;
			delete fOutlier;
		}
		////바닥 추정 & 연결

		////천장 추정 & 연결
		Plane* ceil = nullptr;
		if (CeilAllData->data.rows > 30) {
			PlaneEstRes* cInlier = new PlaneEstRes();
			PlaneEstRes* cOutlier = new PlaneEstRes();
			bool bres = PlaneInitialization(CeilAllData, cInlier, cOutlier, 1500, 0.02, 0.2);
			if (bres) {

				if (LocalMapPlanes.count(PlaneType::CEIL)) {
					ceil = *(LocalMapPlanes[PlaneType::CEIL].begin());
					ceil->count++;
					if (ceil->status == PlaneStatus::NOT_INITIALIZED && ceil->count > 5) {
						ceil->status = PlaneStatus::INITIALIZED;
					}
					//std::cout << "CEIL::Association" << (int)ceil->count << " " << ceil->param.t() << std::endl;
				}
				else {
					ceil = new Plane();
					ceil->type = PlaneType::CEIL;
					std::cout << "CEIL::Initialization" << std::endl;
				}
				////inlier mp 추가
				ceil->param = CeilAllData->param.clone()*-1.0;
				ceil->normal = ceil->param.rowRange(0, 3);
				ceil->dist = cv::norm(ceil->normal);
				sConnectedPlanes.insert(ceil);
			}
		}
		////천장 추정 & 연결

		////연속 벽 추정
		cv::Mat floor2D = cv::Mat::zeros(0, 3, CV_32FC1);
		for (int x = 0; x < label.cols; x += 10) {
			for (int y = 0; y < label.rows; y += 10) {
				int l = label.at<uchar>(y, x) + 1;
				if (l == (int)StructureLabel::FLOOR) {
					cv::Mat tmep = (cv::Mat_<float>(1, 3) << x, y, 1.0);
					floor2D.push_back(tmep);
					//cv::circle(res, cv::Point2f(x,y), 3, cv::Scalar(255, 0, 255), -1);
				}
			}
		}

		//기존 벽 추정
		std::set<Plane*> tempWallPlanes;
		////벽 테스트
		if (floor && LocalMapPlanes.count(PlaneType::WALL)) {
			tempWallPlanes = LocalMapPlanes[PlaneType::WALL];
			for (auto iter = tempWallPlanes.begin(), iend = tempWallPlanes.end(); iter != iend; iter++) {
				auto wall = *iter;
				float m;
				cv::Mat Lw = CalcFlukerLine(floor->param, wall->param);
				wall->line = LineProjection(R, t, Lw, pUser->mpCamera->Kfluker, m);
				if (wall->line.at<float>(0) < 0.0)
					wall->line *= -1.0;
				wall->label = LineFunction(floor2D, wall->line);
			}
		}
		int trial = 0;
		//벽 연결 & 추가 과정
		PlaneEstRes* tempWallData = WallLocalData;
		while (floor && trial < 3 && WallLocalData->data.rows > 50) {

			cv::Mat Rsp = CalcPlaneRotationMatrix(floor->param).clone();
			cv::Mat Rps = Rsp.t();

			PlaneEstRes* tempIn = new PlaneEstRes;
			PlaneEstRes* tempOut = new PlaneEstRes;

			cv::Mat temp = tempWallData->data*Rsp;
			cv::Mat a = temp.col(0);
			cv::Mat b = temp.col(2);
			cv::hconcat(a, b, tempWallData->data);
			bool bres = PlaneInitialization(tempWallData, tempIn, tempOut, 1500, 0.02, 0.2);
			if (!bres)
				break;
			cv::Mat tempParam22 = cv::Mat::zeros(4, 1, CV_32FC1);
			tempParam22.at<float>(0) = tempWallData->param.at<float>(0);
			tempParam22.at<float>(2) = tempWallData->param.at<float>(1);
			tempParam22.at<float>(3) = tempWallData->param.at<float>(2);
			cv::Mat tempParam = Rsp*tempParam22.rowRange(0, 3);
			tempParam.copyTo(tempParam22.rowRange(0, 3));
			tempWallData->param = tempParam22.clone();
			tempWallData->normal = tempWallData->param.rowRange(0, 3);
			tempWallData->dist = cv::norm(tempWallData->normal);
			////line projection
			float m;
			cv::Mat Lw = CalcFlukerLine(floor->param, tempWallData->param);
			cv::Mat line = LineProjection(R, t, Lw, pUser->mpCamera->Kfluker, m);
			if (line.at<float>(0) < 0.0)
				line *= -1.0;
			bool label = LineFunction(floor2D, line);

			////local wall과 비교
			std::map<PlaneCorrelation, int> mapCountCorr;
			std::map<PlaneCorrelation, Plane*> mapPlaneCorr;
			for (auto iter = tempWallPlanes.begin(), iend = tempWallPlanes.end(); iter != iend; iter++) {
				auto twall = *iter;

				////코사인 시밀러리티 계산
				const float cosSim = tempWallData->normal.dot(twall->normal) / (twall->dist*tempWallData->dist);
				PlaneCorrelation cor = PlaneCorrelation::NO_RELATION;
				if (cosSim > 0.99) {
					if (twall->label == label) {
						cor = PlaneCorrelation::IDENTICAL;
					}
					else {
						cor = PlaneCorrelation::PARALLEL;
					}
				}
				else if (cosSim < 0.01) {
					cor = PlaneCorrelation::ORTHOGONAL;
				}
				mapPlaneCorr[cor] = twall; //중복은 여기서는 날라감.
				mapCountCorr[cor]++;
			}

			////IDENTICAL이면 연결.
			if (mapCountCorr[PlaneCorrelation::IDENTICAL] > 0) {
				auto wall = mapPlaneCorr[PlaneCorrelation::IDENTICAL];
				wall->count++;
				if (wall->status == PlaneStatus::NOT_INITIALIZED && wall->count > 5) {
					wall->status = PlaneStatus::INITIALIZED;
				}
				wall->param = tempWallData->param.clone();
				wall->normal = tempWallData->normal.clone();
				wall->dist = tempWallData->dist;
				sConnectedPlanes.insert(wall);
				
				//std::cout << "WALL::Association " << (int)wall->count << std::endl;
				////////////////
				/*{
					cv::Mat Okf = pKF->GetCameraCenter();
					cv::Mat Xaxis1 = R.row(0);
					cv::Mat Xaxis2 = -R.row(0);
					std::cout << Okf.t() << " " << Xaxis1 << std::endl;
					cv::Mat X1 = cv::Mat::zeros(3, 1, CV_32FC1);
					cv::Mat X2 = cv::Mat::zeros(3, 1, CV_32FC1);
					
					float d = (float)wall->normal.dot(Okf) + wall->dist;
					float u1 = abs(Xaxis1.dot(wall->normal.t()));
					float d1 = -1.0;
					float d2 = -1.0;
					if (u1 < 0.99) {
						X1 = Okf + Xaxis1.t()*(d / u1);
						cv::Mat proj = pUser->mpCamera->K*(R*X1+t);
						cv::Point2f pt(proj.at<float>(0) / proj.at<float>(2), proj.at<float>(1) / proj.at<float>(2));
						cv::circle(res, pt, 4, cv::Scalar(255, 0, 0), -1);
						std::cout << proj.t()<<" "<<pt << std::endl;
					}
					double u2 = abs(Xaxis2.dot(wall->normal.t()));
					if (u1 < 0.99) {
						X2 = Okf + Xaxis2.t()*(d / u2);
						cv::Mat proj = pUser->mpCamera->K*(R*X2 + t);
						cv::Point2f pt(proj.at<float>(0) / proj.at<float>(2), proj.at<float>(1) / proj.at<float>(2));
						cv::circle(res, pt, 4, cv::Scalar(0, 0, 255), -1);
						std::cout << proj.t() << " " << pt << std::endl;
					}
					std::cout << "Test = " << u1<<" "<<u2 << std::endl;
				}*/
				///////////////

			}
			else if (mapCountCorr[PlaneCorrelation::PARALLEL] > 0)
			{
				std::cout << "WALL::PARALLEL and init" << std::endl;
				auto wall = new Plane();
				wall->type = PlaneType::WALL;
				wall->param = tempWallData->param.clone();
				wall->normal = tempWallData->normal.clone();
				wall->dist = tempWallData->dist;
				sConnectedPlanes.insert(wall);
				tempWallPlanes.insert(wall);
			}
			else if (mapCountCorr[PlaneCorrelation::PARALLEL] == 0 && mapCountCorr[PlaneCorrelation::IDENTICAL] == 0) {
				std::cout << "WALL::INIT" << std::endl;
				auto wall = new Plane();
				wall->type = PlaneType::WALL;
				wall->param = tempWallData->param.clone();
				wall->normal = tempWallData->normal.clone();
				wall->dist = tempWallData->dist;
				sConnectedPlanes.insert(wall);
				tempWallPlanes.insert(wall);
			}
			else if (mapCountCorr[PlaneCorrelation::ORTHOGONAL]>0) {//mapCountCorr[PlaneCorrelation::PARALLEL] == 0 && 
				//std::cout << "WALL::ORTHOGONAL" << std::endl;

				/*auto wall = new Plane();
				wall->type = PlaneType::WALL;
				wall->param = resWall.param.clone();
				wall->normal = resWall.normal.clone();
				wall->dist = resWall.dist;
				sConnectedPlanes.insert(wall);
				tempWallPlanes.insert(wall);*/

				//////////////////
				//{
				//	auto wall = new Plane();
				//	wall->type = PlaneType::WALL;
				//	wall->param = tempWallData->param.clone();
				//	wall->normal = tempWallData->normal.clone();
				//	wall->dist = tempWallData->dist;
				//	
				//	cv::Mat Okf = pKF->GetCameraCenter();
				//	cv::Mat Xaxis1 = R.row(0);
				//	cv::Mat Xaxis2 = -R.row(0);
				//	std::cout << Okf.t() << " " << Xaxis1 << std::endl;
				//	cv::Mat X1 = cv::Mat::zeros(3, 1, CV_32FC1);
				//	cv::Mat X2 = cv::Mat::zeros(3, 1, CV_32FC1);

				//	float d = (float)wall->normal.dot(Okf) + wall->dist;
				//	float u1 = abs(Xaxis1.dot(wall->normal.t()));
				//	float d1 = -1.0;
				//	float d2 = -1.0;
				//	if (u1 < 0.99) {
				//		X1 = Okf + Xaxis1.t()*(d / u1);
				//		cv::Mat proj = pUser->mpCamera->K*(R*X1 + t);
				//		cv::Point2f pt(proj.at<float>(0) / proj.at<float>(2), proj.at<float>(1) / proj.at<float>(2));
				//		cv::circle(res, pt, 4, cv::Scalar(255, 0, 0), -1);
				//		std::cout << proj.t() << " " << pt << std::endl;
				//	}
				//	double u2 = abs(Xaxis2.dot(wall->normal.t()));
				//	if (u1 < 0.99) {
				//		X2 = Okf + Xaxis2.t()*(d / u2);
				//		cv::Mat proj = pUser->mpCamera->K*(R*X2 + t);
				//		cv::Point2f pt(proj.at<float>(0) / proj.at<float>(2), proj.at<float>(1) / proj.at<float>(2));
				//		cv::circle(res, pt, 4, cv::Scalar(0, 0, 255), -1);
				//		std::cout << proj.t() << " " << pt << std::endl;
				//	}
				//	std::cout << "Test = " << u1 << " " << u2 << std::endl;
				//}
				/////////////////

			}
			delete tempWallData;
			delete tempIn;
			//데이터 갱신
			tempWallData = tempOut;
			trial++;
			//std::cout << mapCountCorr[PlaneCorrelation::IDENTICAL] << " " << mapCountCorr[PlaneCorrelation::PARALLEL] << " " << mapCountCorr[PlaneCorrelation::ORTHOGONAL] << "::" << tempWallPlanes.size() << "     ?????????" << std::endl;
		}
		
		//벽 시각화
		for (auto iter = tempWallPlanes.begin(), iend = tempWallPlanes.end(); iter != iend; iter++) {
			auto twall = *iter;

			////line projection
			if (floor) {
				float m;
				cv::Mat Lw = CalcFlukerLine(floor->param, twall->param);
				cv::Mat line = LineProjection(R, t, Lw, pUser->mpCamera->Kfluker, m);
				if (line.at<float>(0) < 0.0)
					line *= -1.0;
				bool label = LineFunction(floor2D, line);

				auto pt11 = GetLinePoint(0.0, line, false);
				auto pt12 = GetLinePoint(pKF->mpCamera->mnWidth, line, false);
				if (label) {
					cv::line(res, pt11, pt12, cv::Scalar(255, 0, 0), 2);
				}
				else {
					cv::line(res, pt11, pt12, cv::Scalar(0, 0, 255), 2);
				}
			}
			if (ceil) {
				float m;
				cv::Mat Lw = CalcFlukerLine(ceil->param, twall->param);
				cv::Mat line = LineProjection(R, t, Lw, pUser->mpCamera->Kfluker, m);
				if (line.at<float>(0) < 0.0)
					line *= -1.0;
				auto pt11 = GetLinePoint(0.0, line, false);
				auto pt12 = GetLinePoint(pKF->mpCamera->mnWidth, line, false);
				cv::line(res, pt11, pt12, cv::Scalar(0, 255, 0), 2);
			}
		}

		/*for (std::vector<EdgeSLAM::KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			EdgeSLAM::KeyFrame* pKFi = *itKF;
			if (mPlaneConnections.Count(pKFi)) {
				auto tempPlanes = mPlaneConnections.Get(pKFi);
				if(floor)
					tempPlanes.insert(floor);
				if(ceil)
					tempPlanes.insert(ceil);
				for (auto iter = tempWallPlanes.begin(), iend = tempWallPlanes.end(); iter != iend; iter++) {
					auto wall = *iter;
					tempPlanes.insert(wall);
				}
				mPlaneConnections.Update(pKFi, tempPlanes);
			}
		}*/

		////연속 벽 추정

		
		/*resWall2.bres = false;
		if (matWallData.rows > 50) {
			resWall1.bres = PlaneInitialization(matWallData, resWall1.param, resWall1.inlier, resWall1.outlier, 1500, 0.02, 0.2);
			if (resWall1.bres) {
				resWall1.normal = resWall1.param.rowRange(0, 3);
				resWall1.dist = cv::norm(resWall1.normal);
			}
		}
		if (resWall1.bres && resWall1.outlier.rows > 50) {
			resWall2.bres = PlaneInitialization(resWall1.outlier, resWall2.param, resWall2.inlier, resWall2.outlier, 1500, 0.02, 0.2);
			if (resWall2.bres) {
				resWall2.normal = resWall2.param.rowRange(0, 3);
				resWall2.dist = cv::norm(resWall2.normal);
			}
		}*/

		//if (floor && (resWall1.bres || resWall2.bres)) {

		//	////바닥 세그멘테이션 영역 획득
		//	
		//	
		//	
		//	if(resWall1.bres){
		//		Plane* wall = nullptr;
		//		if (!bAlreadyWallDetection) {
		//			bAlreadyWallDetection = true;
		//			wall = new Plane();
		//			wall->type = PlaneType::WALL;
		//			std::cout << "WALL::Initialization1" << std::endl;
		//			wall->param = resWall1.param.clone();
		//			sConnectedPlanes.insert(wall);
		//		}
		//		else {
		//			////LINE PROJECTION
		//			float m;
		//			cv::Mat Lw = CalcFlukerLine(floor->param, resWall1.param);
		//			cv::Mat line = LineProjection(R, t, Lw, pUser->mpCamera->Kfluker, m);
		//			if (line.at<float>(0) < 0.0)
		//				line *= -1.0;
		//			bool label = LineFunction(floor2D, line);

		//			////로컬 맵의 벽과 비교
		//			std::map<Plane*, PlaneCorrelation> mapPlaneCorr;
		//			std::map<PlaneCorrelation, int> mapCountCorr;
		//			std::map<PlaneCorrelation, Plane*> mapPlaneCorr2;
		//			
		//			for (auto iter = tempWallPlanes.begin(), iend = tempWallPlanes.end(); iter != iend; iter++) {
		//				auto twall = *iter;
		//				
		//				////코사인 시밀러리티 계산
		//				cv::Mat normal = twall->param.rowRange(0, 3);
		//				double d1 = cv::norm(normal);
		//				const float cosSim = normal.dot(resWall1.normal) / (d1*resWall1.dist);
		//				PlaneCorrelation cor = PlaneCorrelation::NO_RELATION;
		//				if (cosSim > 0.99) {
		//					if (twall->label == label) {
		//						cor = PlaneCorrelation::IDENTICAL;
		//					}else{
		//						cor = PlaneCorrelation::PARALLEL;
		//					}
		//				}
		//				else if(cosSim < 0.01){
		//					cor = PlaneCorrelation::ORTHOGONAL;
		//				}
		//				/*else{
		//					mapPlaneCorr[twall] = PlaneCorrelation::NO_RELATION;
		//				}*/
		//				mapPlaneCorr2[cor] = twall; //중복은 여기서는 날라감.
		//				mapPlaneCorr[twall] = cor;
		//				mapCountCorr[cor]++;
		//			}

		//			////IDENTICAL이면 연결.
		//			if (mapCountCorr[PlaneCorrelation::IDENTICAL] > 0) {
		//				auto wall = mapPlaneCorr2[PlaneCorrelation::IDENTICAL];
		//				wall->count++;
		//				if (wall->status == PlaneStatus::NOT_INITIALIZED && wall->count > 5) {
		//					wall->status = PlaneStatus::INITIALIZED;
		//				}
		//				sConnectedPlanes.insert(wall);

		//				auto pt11 = GetLinePoint(0.0, line, false);
		//				auto pt12 = GetLinePoint(pKF->mpCamera->mnWidth, line, false);
		//				if (label) {
		//					cv::line(res, pt11, pt12, cv::Scalar(255, 0, 0), 2);
		//				}
		//				else {
		//					cv::line(res, pt11, pt12, cv::Scalar(0, 0, 255), 2);
		//				}
		//				std::cout << "WALL::Association " << (int)wall->count << std::endl;
		//			}
		//			else if (mapCountCorr[PlaneCorrelation::PARALLEL]==0 && mapCountCorr[PlaneCorrelation::ORTHOGONAL]>0) {
		//				std::cout << "WALL::ORTHOGONAL and init" << std::endl;
		//				wall = new Plane();
		//				wall->type = PlaneType::WALL;
		//				wall->param = resWall1.param.clone();
		//				sConnectedPlanes.insert(wall);
		//			}
		//			else {
		//				std::cout << "?????????" << std::endl;
		//			}
		//		}
		//	}
		//}
		mPlaneConnections.Update(pKF, sConnectedPlanes);
		pUser->mnUsed--;
		////association
		
		

		std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - start).count();
		float t_test1 = du_test1 / 1000.0;

		auto du_test2 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		float t_test2 = du_test1 / 1000.0;

		auto du_test3 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - start).count();
		float t_test3 = du_test1 / 1000.0;


		/*std::vector<cv::Mat> wallDatas;
		if (SLAM->TemporalDatas.Count("wall"))
			wallDatas = SLAM->TemporalDatas.Get("wall");
		std::vector<cv::Mat> floorDatas;
		if (SLAM->TemporalDatas.Count("floor"))
			floorDatas = SLAM->TemporalDatas.Get("floor");*/

		//if (bFloor) {
		//	std::cout << "Floor = " << param1.t() << std::endl;
		//	for (int i = 0, iend = inliers1.rows; i <iend; i++) {
		//		floorDatas.push_back(inliers1.row(i));
		//	}
		//	SLAM->TemporalDatas.Update("floor", floorDatas);
		//	auto pt = calcSphericalCoordinate(param1.rowRange(0, 3));
		//	std::ofstream output("../bin/hough_floor.txt", std::ios::app);
		//	output << pt.x << " " << pt.y << ";" << std::endl;
		//	output.close();
		//}

		//if (bWall1) {
		//	std::cout << "WALL1 = " << param2.t() << std::endl;
		//	for (int i = 0, iend = inliers2.rows; i <iend; i++) {
		//		wallDatas.push_back(inliers2.row(i));
		//	}
		//	auto pt = calcSphericalCoordinate(param2.rowRange(0, 3));
		//	std::ofstream output("../bin/hough.txt", std::ios::app);
		//	output << pt.x << " " << pt.y << ";" << std::endl;
		//	output.close();
		//}
		//if (bWall2) {
		//	std::cout << "WALL2 = " << param3.t() << std::endl;
		//	for (int i = 0, iend = inliers3.rows; i <iend; i++) {
		//		wallDatas.push_back(inliers3.row(i));
		//	}
		//	auto pt = calcSphericalCoordinate(param3.rowRange(0, 3));
		//	std::ofstream output("../bin/hough.txt", std::ios::app);
		//	output << pt.x << " " << pt.y << ";" << std::endl;
		//	output.close();
		//}
		//
		//if (bWall1 || bWall2) {
		//	SLAM->TemporalDatas.Update("wall", wallDatas);
		//}
		//
		//
		//
		//if (bFloor && bWall1) {
		//	float m1;
		//	cv::Mat line1;
		//	cv::Mat Lw1 = CalcFlukerLine(param1, param2);
		//	line1 = LineProjection(R, t, Lw1, pUser->mpCamera->Kfluker, m1);
		//	if (line1.at<float>(0) < 0.0)
		//		line1 *= -1.0;
		//	
		//	auto pt11 = GetLinePoint(0.0, line1, false);
		//	auto pt12 = GetLinePoint(pKF->mpCamera->mnWidth, line1, false);
		//	
		//	
		//	//////라인 레이블링
		//	//cv::Mat proj = floor2D*line1;
		//	//int a1 = cv::countNonZero((proj) < 0.0);
		//	//int a2 = cv::countNonZero((proj) > 0.0);
		//	//if (a1 > a2) {
		//	//	cv::line(res, pt11, pt12, cv::Scalar(255, 0, 0), 2);
		//	//}
		//	//else {
		//	//	cv::line(res, pt11, pt12, cv::Scalar(0, 0, 255), 2);
		//	//}
		//	//////라인 레이블링
		//	//std::cout << "L1 = " <<a1<<" "<<a2<< line1.t() << " " << param2.t() << std::endl;
		//}
		//if (bFloor && bWall2) {
		//	float m1;
		//	cv::Mat line1;
		//	cv::Mat Lw1 = CalcFlukerLine(param1, param3);
		//	line1 = LineProjection(R, t, Lw1, pUser->mpCamera->Kfluker, m1);
		//	if (line1.at<float>(0) < 0.0)
		//		line1 *= -1.0;

		//	/*m1 = -line1.at<float>(0) / line1.at<float>(1);
		//	bool bSlopeOpt1 = abs(m1) > 1.0;
		//	float val1;
		//	if (bSlopeOpt1)
		//		val1 = pKF->mpCamera->mnHeight;
		//	else
		//		val1 = pKF->mpCamera->mnWidth;
		//	
		//	auto pt11 = GetLinePoint(0.0, line1, bSlopeOpt1);
		//	auto pt12 = GetLinePoint(val1, line1, bSlopeOpt1);*/
		//	
		//	auto pt11 = GetLinePoint(0.0, line1, false);
		//	auto pt12 = GetLinePoint(pKF->mpCamera->mnWidth, line1, false);

		//	/*cv::Mat proj = floor2D*line1;
		//	int a1 = cv::countNonZero((proj) < 0.0);
		//	int a2 = cv::countNonZero((proj) > 0.0);
		//	if (a1 > a2) {
		//		cv::line(res, pt11, pt12, cv::Scalar(255, 0, 0), 2);
		//	}
		//	else {
		//		cv::line(res, pt11, pt12, cv::Scalar(0, 0, 255), 2);
		//	}*/
		//	//std::cout << "L2 = " << a1 << " " << a2 << line1.t() << " " << param3.t() << std::endl;
		//}
		
		SLAM->VisualizeImage(res, 3);

		//{
		//	/////save image
		//	std::stringstream sss;
		//	sss << "../bin/img/" << user << "/Track/" << id << "_plane.jpg";
		//	//sss << "../../bin/img/" << id << "_plane.jpg";
		//	cv::imwrite(sss.str(), res);
		//	/////save image
		//}

		////std::cout << "Plane = " << "=" << t_test1 << ", " << t_test2 << ", " << t_test3 << "=" << std::endl;
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
			cv::Mat X;
			cv::Mat w, u, vt;
			cv::SVD::compute(arandomPts, w, u, vt, cv::SVD::FULL_UV);
			X = vt.row(nDim2).clone();
			cv::transpose(X, X);

			if (!calcUnitNormalVector(X)) {
				//std::cout << "PE::RANSAC_FITTING::UNIT Vector error" << std::endl;
			}
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

			for (int i = 0; i < src->data.rows; i++) {
				int checkIdx = paramStatus.at<uchar>(i);

				if (checkIdx == 0) {
					outlier->AddData(src->vecMPs[i]);
				}
				else {
					inlier->AddData(src->vecMPs[i]);
				}
			}
			src->param = param.clone();
			return true;
		}
		else
		{
			//std::cout << "failed" << std::endl;
			return false;
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
			cv::Mat X;
			cv::Mat w, u, vt;
			cv::SVD::compute(arandomPts, w, u, vt, cv::SVD::FULL_UV);
			X = vt.row(3).clone();
			cv::transpose(X, X);

			if (!calcUnitNormalVector(X)) {
				//std::cout << "PE::RANSAC_FITTING::UNIT Vector error" << std::endl;
			}
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