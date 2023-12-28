#include "ContentProcessor.h"
#include <Utils.h>
#include <User.h>
#include "MarkerProcessor.h"
#include "SemanticProcessor.h"
#include "GridProcessor.h"
#include "Node.h"
#include <ThreadPool.h>

namespace SemanticSLAM {
	std::atomic<int> ContentProcessor::nContentID = 0;

	bool ContentProcessor::mbSaveLatency;

	ConcurrentMap<EdgeSLAM::KeyFrame*, std::map<int, Content*>> ContentProcessor::ContentMap;
	ConcurrentMap<int, Content*> ContentProcessor::AllContentMap;
	ConcurrentMap<int, Content*> ContentProcessor::MarkerContentMap;
	ConcurrentMap<int, cv::Mat> ContentProcessor::MapArucoMarkerPos;
	ConcurrentMap<int, std::string> ContentProcessor::IndirectData;
	ConcurrentMap<EdgeSLAM::KeyFrame*, std::map<int, std::string>> ContentProcessor::AnchorIDs;

	//임시로 마커 아이디에 다른 타입이 들어감. 테스트 다시 해야 함. 드로우용 컨텐츠 생성하기 위해서
	Content::Content():Node(){}
	Content::Content(const cv::Mat& _X, std::string _src, int _modelID, long long ts):Node(),mnID(++ContentProcessor::nContentID), mnNextID(0), mnContentModelID(_modelID), mnMarkerID(_modelID), src(_src), mbMoving(false), mpPath(nullptr), mpGrid(nullptr)
	,mnLastUpdatedTime(ts)
	{
		float len = _X.at<float>(0);
		data = _X.rowRange(0, len).clone();
		data.at<float>(1) = (float)mnID;
	}
	Content::~Content(){
		data.release();
		/*pos.release();
		dir.release();
		endPos.release();*/
	}

	////////////////실험용 코드
	//레퍼 레이턴시 기록 & 저장
	//터치 레이턴시 기록 & 저장 & 리턴
	ConcurrentVector<float> LatencyReference; //주기적
	ConcurrentVector<float> LatencyInteraction; //즉각적

	void ContentProcessor::SaveLatency() {
		std::stringstream ssPath;
		ssPath << "../bin/trajectory/";

		{
			std::stringstream ss;
			ss << ssPath.str() << "LatencyReference.txt";
			std::ofstream f;
			f.open(ss.str().c_str());

			auto vecInconsistency = LatencyReference.get();
			for (int i = 0; i < vecInconsistency.size(); i++) {
				float len = vecInconsistency[i];
				f << len << std::endl;
			}
			f.close();
		}
		{
			std::stringstream ss;
			ss << ssPath.str() << "LatencyInteraction.txt";
			std::ofstream f;
			f.open(ss.str().c_str());

			auto vecInconsistency = LatencyInteraction.get();
			for (int i = 0; i < vecInconsistency.size(); i++) {
				float len = vecInconsistency[i];
				f << len << std::endl;
			}
			f.close();
		}

	}
	void ContentProcessor::TestTouch(std::string user, int id) {
		WebAPI API("143.248.6.143", 35005);
		cv::Mat data = cv::Mat::zeros(1000, 1, CV_32FC1);
		std::stringstream ss;
		ss << "/Store?keyword=TouchB&id=" << id << "&src=" << user;
		auto res = API.Send(ss.str(), data.data, data.rows * sizeof(float));
	}
	void ContentProcessor::UpdateLatency(std::string keyword, std::string user, int id) {
		std::stringstream ss;
		ss << "/Load?keyword="<<keyword<< "&id=" << id << "&src=" << user;
		WebAPI API("143.248.6.143", 35005);
		auto res = API.Send(ss.str(), "");
		int n2 = res.size();

		cv::Mat fdata = cv::Mat::zeros(1000, 1, CV_32FC1);
		std::memcpy(fdata.data, res.data(), res.size());
		float len = fdata.at<float>(0);
		if (keyword == "TestRef") {
			LatencyReference.push_back(len);
		}
		else if (keyword == "TestInter") {
			LatencyInteraction.push_back(len);
		}
	}
	////////////////실험용 코드
	void ContentProcessor::DirectTest(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		IndirectData.Update(id, user);
		cv::Mat data = cv::Mat::ones(1000, 1, CV_32FC1);
		{
			WebAPI mpAPI("143.248.6.143", 35005);
			std::stringstream ss;
			ss << "/Store?keyword=dr&id=" << id << "&src=" << user;
			auto res = mpAPI.Send(ss.str(), data.data, data.rows * sizeof(float));
		}
		pUser->mnUsed--;
	}
	void ContentProcessor::IndirectSend(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		auto data = IndirectData.Get();
		cv::Mat tdata = cv::Mat::ones(1000, 1, CV_32FC1);
		for (auto iter = data.begin(); iter != data.end(); iter++) {
			int id2 = iter->first;
			std::string src = iter->second;
			WebAPI mpAPI("143.248.6.143", 35005);
			std::stringstream ss;
			ss << "/Store?keyword=ir&id=" << id2 << "&src=" << src;
			auto res = mpAPI.Send(ss.str(), tdata.data, tdata.rows * sizeof(float));
		}
		IndirectData.Clear();
		pUser->mnUsed--;
	}
	void ContentProcessor::IndirectTest(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		//IndirectData.
		IndirectData.Update(id, user);
		pUser->mnUsed--;
	}
	void ContentProcessor::GenerateGraphDataForSync(EdgeSLAM::SLAM* SLAM, EdgeSLAM::User* pUser, cv::Mat& totaldata, int id, long long ts) {
		
		totaldata = cv::Mat::zeros(2, 1, CV_32FC1);
		totaldata.at<float>(0) = 2.0; //data size
		totaldata.at<float>(1) = 3.0; //parsing id
		
		auto pKF = pUser->mpRefKF;
		if (pKF) {
			//사전 작업
			bool bGridCommu = pUser->mbCommuTest;
			int nKFs = pUser->mnContentKFs;
			std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(nKFs);
			vpLocalKFs.push_back(pKF);

			//관련 데이터
			std::set<Grid*> setGrids;
			std::set<Content*> spContents;
			std::set<EdgeSLAM::Node*> spNodes;

			if (bGridCommu) {

				std::vector<std::vector<cv::Point2f>> vecProjectedCorners;

				for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
					auto pKFi = *iter;
					if (GridProcessor::GlobalKeyFrameNGrids.Count(pKFi)) {
						auto spGrids = GridProcessor::GlobalKeyFrameNGrids.Get(pKFi);
						for (auto jter = spGrids.begin(), jend = spGrids.end(); jter != jend; jter++) {
							auto pTempGrid = *jter;
							if (setGrids.count(pTempGrid))
								continue;
							setGrids.insert(pTempGrid);
						}//for jter
					}//if
				}//iter
				for (auto iter = setGrids.begin(), iend = setGrids.end(); iter != iend; iter++) {
					auto pGrid = *iter;
					auto spVOs = pGrid->ConnectedVOs.Get();
					for (auto jter = spVOs.begin(), jend = spVOs.end(); jter != jend; jter++) {
						auto pContent = *jter;
						if (spContents.count(pContent)) {
							continue;
						}
						spContents.insert(pContent);
					}
				}
			}
			else {
				for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
					auto pKFi = *iter;
					std::map<int, Content*> mapContents;
					if (ContentMap.Count(pKFi)) {
						mapContents = ContentMap.Get(pKFi);
						for (auto jter = mapContents.begin(), jend = mapContents.end(); jter != jend; jter++) {
							auto pContent = jter->second;
							if (!spContents.count(pContent))
								spContents.insert(pContent);
						}
					}
				}
			}

			//실험일 때 전체 데이터 보내기
			bool bSyncTypeTest = pUser->mbVOSyncTest;
			if (bSyncTypeTest) {
				auto vecAllVOs = AllContentMap.Get();
				for (int i = 0, iend = vecAllVOs.size(); i < iend; i++) {
					auto pContent = vecAllVOs[i];
					if (pContent) {
						if (!spContents.count(pContent))
							spContents.insert(pContent);
					}
				}
			}

			//마커 정보 현재는 이용 안함
			auto pMarkerObjects = MarkerContentMap.Get();
			for (auto iter = pMarkerObjects.begin(); iter != pMarkerObjects.end(); iter++) {
				auto pContent = iter->second;
				if (!spContents.count(pContent))
					spContents.insert(pContent);
			}

			//연결 데이터, 관련 정보

			cv::Mat assodata = cv::Mat::zeros(1, 1, CV_32FC1); //사이즈, id
			cv::Mat posdata = cv::Mat::zeros(1, 1, CV_32FC1);  //사이즈, id, info
			int nConnect = 0;
			int nContent = 0;

			//전송 정보
			auto mapSynchedVOs = pUser->mapLastSyncedVOs.Get();
			auto mapSendedVOs = pUser->mapLastSendedVOs.Get();

			//cv::Mat data = cv::Mat::zeros(1, 1, CV_32FC1);
			//data.at<float>(0) = spContents.size();
			for (auto iter = spContents.begin(), iend = spContents.end(); iter != iend; iter++) {
				auto pContent = *iter;
				int contentID = pContent->mnID;
				//동기화 선별 과정
				bool bTrialSync = false;
				long long lastUpdated = pContent->mnLastUpdatedTime;
				if (mapSynchedVOs.count(contentID)) {
					auto lastSynced = mapSynchedVOs[contentID];
					if (lastUpdated > lastSynced) {
						bTrialSync = true;
					}
				}
				else {
					bTrialSync = true;
				}

				cv::Mat tempAsso = cv::Mat::zeros(1, 1, CV_32FC1);
				tempAsso.at<float>(0) = (float)contentID;
				assodata.push_back(tempAsso);
				nConnect++;

				if (bTrialSync) {
					pUser->mapLastSyncedVOs.Update(contentID, ts);
					posdata.push_back(pContent->data);
					nContent++;
				}

			}

			/*
			파싱 정보 = 3
			전체 정보
			연결 정보 = 개수, 아이디들
			데이터 정보 = 개수, 개별 정보
			*/
			assodata.at<float>(0) = (float)nConnect;
			posdata.at<float>(0) = (float)nContent;

			totaldata.at<float>(0) = (float)(assodata.rows + posdata.rows + 2);
			totaldata.push_back(assodata);
			totaldata.push_back(posdata);
		}
	}
	void ContentProcessor::ShareContent(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		//std::cout << "ShareContent start" << std::endl;
		auto pUser = SLAM->GetUser(user);
		if (!pUser){
			//std::cout << "ShareContent end1" << std::endl;
			return;
		}
		pUser->mnUsed++;
		auto pKF = pUser->mpRefKF;
		if (!pKF) {
			//std::cout << "ShareContent end2" << std::endl;
			pUser->mnUsed--;
			return;
		}

		//타임스탬프
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		//사전 작업
		bool bGridCommu = pUser->mbCommuTest;
		int nKFs = pUser->mnContentKFs;
		pUser->mnDebugAR++;
		std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(nKFs);
		vpLocalKFs.push_back(pKF);

		//관련 데이터
		std::set<Grid*> setGrids;
		std::set<Content*> spContents;
		std::set<EdgeSLAM::Node*> spNodes;

		
		if (bGridCommu) {

			std::vector<std::vector<cv::Point2f>> vecProjectedCorners;

			for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
				auto pKFi = *iter;
				if (GridProcessor::GlobalKeyFrameNGrids.Count(pKFi)) {
					auto spGrids = GridProcessor::GlobalKeyFrameNGrids.Get(pKFi);
					for (auto jter = spGrids.begin(), jend = spGrids.end(); jter != jend; jter++) {
						auto pTempGrid = *jter;
						if (setGrids.count(pTempGrid))
							continue;
						setGrids.insert(pTempGrid);
					}//for jter
				}//if
			}//iter
			for (auto iter = setGrids.begin(), iend = setGrids.end(); iter != iend; iter++) {
				auto pGrid = *iter;
				auto spVOs = pGrid->ConnectedVOs.Get();
				for (auto jter = spVOs.begin(), jend = spVOs.end(); jter != jend; jter++) {
					auto pContent = *jter;
					if(spContents.count(pContent)){
						continue;
					}
					spContents.insert(pContent);
				}
			}
		}
		else {
			for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
				auto pKFi = *iter;
				std::map<int, Content*> mapContents;
				if (ContentMap.Count(pKFi)) {
					mapContents = ContentMap.Get(pKFi);
					for (auto jter = mapContents.begin(), jend = mapContents.end(); jter != jend; jter++) {
						auto pContent = jter->second;
						if (!spContents.count(pContent))
							spContents.insert(pContent);
					}
				}
			}
		}
		

		auto pMarkerObjects = MarkerContentMap.Get();
		for (auto iter = pMarkerObjects.begin(); iter != pMarkerObjects.end(); iter++) {
			auto pContent = iter->second;
			if (!spContents.count(pContent))
				spContents.insert(pContent);
		}
		 
		//연결 데이터, 관련 정보
		cv::Mat totaldata = cv::Mat::zeros(2, 1, CV_32FC1);
		cv::Mat assodata = cv::Mat::zeros(1, 1, CV_32FC1); //사이즈, id
		cv::Mat posdata = cv::Mat::zeros(1, 1, CV_32FC1);  //사이즈, id, info
		int nConnect = 0;
		int nContent = 0;

		//전송 정보
		auto mapSynchedVOs = pUser->mapLastSyncedVOs.Get();
		auto mapSendedVOs  = pUser->mapLastSendedVOs.Get();

		//cv::Mat data = cv::Mat::zeros(1, 1, CV_32FC1);
		//data.at<float>(0) = spContents.size();
		for (auto iter = spContents.begin(), iend = spContents.end(); iter != iend; iter++) {
			auto pContent = *iter;
			int contentID = pContent->mnID;
			//동기화 선별 과정
			bool bTrialSync = false;
			long long lastUpdated = pContent->mnLastUpdatedTime;
			if (mapSynchedVOs.count(contentID)) {
				auto lastSynced = mapSynchedVOs[contentID];
				if (lastUpdated > lastSynced) {
					bTrialSync = true;
				}
			}
			else {
				bTrialSync = true;
			}

			cv::Mat tempAsso = cv::Mat::zeros(1, 1, CV_32FC1);
			tempAsso.at<float>(0) = (float)contentID;
			assodata.push_back(tempAsso);
			nConnect++;

			if (bTrialSync) {
				pUser->mapLastSyncedVOs.Update(contentID, ts);
				posdata.push_back(pContent->data);
				nContent++;
			}

		}
		
		/*
		파싱 정보 = 3
		전체 정보
		연결 정보 = 개수, 아이디들
		데이터 정보 = 개수, 개별 정보
		*/
		assodata.at<float>(0) = (float)nConnect;
		posdata.at<float>(0) = (float)nContent;

		totaldata.at<float>(0) = 3.0;
		totaldata.at<float>(1) = (float)(assodata.rows+posdata.rows + 1);
		totaldata.push_back(assodata);
		totaldata.push_back(posdata);

		if (totaldata.rows < 500) {
			cv::Mat temp = cv::Mat::zeros(1000 - totaldata.rows, 1, CV_32FC1);
			totaldata.push_back(temp);
		}
		
		{
			WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
			std::stringstream ss;
			ss << "/Store?keyword=LocalContent&id=" << id << "&src=" << user;
			std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
			auto res = mpAPI->Send(ss.str(), totaldata.data, totaldata.rows * sizeof(float));
			std::chrono::high_resolution_clock::time_point e = std::chrono::high_resolution_clock::now();
			delete mpAPI;
		}

		//시각화
		/*{
			cv::Mat encoded = pUser->ImageDatas.Get(id);
			cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);
			cv::Mat P = pUser->GetPose();

			cv::Mat R = P.rowRange(0, 3).colRange(0, 3);
			cv::Mat t = P.col(3).rowRange(0, 3);
			cv::Mat K = pUser->GetCameraMatrix();
			for (auto iter = spContents.begin(), iend = spContents.end(); iter != iend; iter++) {
				auto pContent = *iter;
				cv::Mat proj = K*(R*pContent->pos + t);
				float depth = proj.at<float>(2);
				if (depth > 0) {
					cv::Point2f pt(proj.at<float>(0) / depth, proj.at<float>(1) / depth);
					cv::circle(img, pt, 3, cv::Scalar(255, 0, 0), -1);
				}
			}
			SLAM->VisualizeImage(img, pUser->GetVisID());
		}*/
		//std::cout << "ShareContent end" << std::endl;
		pUser->mnDebugAR--;
		pUser->mnUsed--;
		//std::cout << "commu test = " << nConnect << " " << nContent << std::endl;
	}

	void ContentProcessor::DrawContentProcess(EdgeSLAM::SLAM* SLAM, std::string user, int id, std::string kewword, int mid) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;
		auto pKF = pUser->mpRefKF;
		if (!pKF) {
			pUser->mnUsed--;
			return;
		}
		pUser->mnDebugAR++;
		//auto pMap = SLAM->GetMap(pUser->mapName);
		std::stringstream ss;
		ss << "/Load?keyword=" << kewword << "&id=" << id << "&src=" << user;

		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");
		int n2 = res.size();
		
		cv::Mat fdata = cv::Mat(n2/4, 1, CV_32FC1, (void*)res.data());
		//std::memcpy(fdata.data, res.data(), res.size());
		DrawContentRegistration(SLAM, pKF, user, fdata, mid);
		pUser->mnUsed--;
	}
	void ContentProcessor::DirectCommuTest(EdgeSLAM::SLAM* SLAM, EdgeSLAM::User* USER, cv::Mat& data, int id, double ts, int N) {
		std::string name = USER->userName;
		long long du_upload =  Utils::SendData("dr", name, data, id, ts);
		
		if (mbSaveLatency)
		{
			std::stringstream ss;
			// "num,source,method,type,id,quality,size,latency\n";
			ss << N << "," << name << "," << "direct,upload," << id << "," << data.rows * sizeof(float) << "," << du_upload << std::endl;
			SLAM->EvaluationVirtualObjectLatency.push_back(ss.str());
		}
	}
	void ContentProcessor::ContentProcess(EdgeSLAM::SLAM* SLAM, std::string user, int id, std::string kewword, double ts, int mid) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		auto pKF = pUser->mpRefKF;
		if (!pKF) {
			return;
		}
		pUser->mnUsed++;
		pUser->mnDebugAR++;
		
		//t1 = down start
		//t2 = down end
		//t3 = registration end
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		std::stringstream ss;
		ss << "/Load?keyword="<< kewword << "&id=" << id << "&src=" << user;
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		int n2 = res.size();
		cv::Mat fdata = cv::Mat(n2 / 4, 1, CV_32FC1, (void*)res.data());
		
		std::chrono::high_resolution_clock::time_point t3;
		if (pUser->mbVOSyncTest) {
			int newVOID = SimulRegistration(SLAM, user, fdata, mid);
			t3 = std::chrono::high_resolution_clock::now();
			auto vecAllUsers = SLAM->GetAllUsersInMap(pUser->mapName);
			//바로 모든 기기에 전송해야 함. 
			//전송 기록 측정
			//std::cout << "content data = " << id << std::endl;
			for (int i = 0, iend = vecAllUsers.size(); i < iend; i++) {
				auto tempUser = vecAllUsers[i];
				if (!tempUser)
					continue;
				tempUser->mnUsed++;
				SLAM->pool->EnqueueJob(DirectCommuTest, SLAM, tempUser, fdata, id, ts, iend);
				tempUser->mnUsed--;
			}
			//std::cout << "content data end" << std::endl;
		}
		else {
			int nContentKFs = 100;//pUser->mnContentKFs;
			int newVOID = ContentRegistration(SLAM, pKF, user, fdata, mid, nContentKFs);
			t3 = std::chrono::high_resolution_clock::now();
		}

		if(mbSaveLatency)
		{
			auto vecAllUsers = SLAM->GetAllUsersInMap(pUser->mapName);
			int N = vecAllUsers.size();
			auto du_down = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
			//auto du_registration = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();
			std::stringstream ss;
			// "num,source,method,type,id,quality,size,latency\n";
			ss << N << "," << user << "," << "vo,download," << id << "," << res.size() << "," << du_down << std::endl;
			//ss << N << "," << user << "," << "vo," << id << "," << "download" << "," << res.size() << "," << du_down <<","<<du_registration <<","<<t1.time_since_epoch().count() << std::endl;
			SLAM->EvaluationVirtualObjectLatency.push_back(ss.str());
		}
		
		pUser->mnDebugAR--;
		pUser->mnUsed--;
	}
	Content* ContentProcessor::GetContent(int id) {
		if (AllContentMap.Count(id))
			return AllContentMap.Get(id);
		return nullptr;
	}
	void ContentProcessor::ResetContent(EdgeSLAM::SLAM* SLAM) {
		auto mapContents = AllContentMap.Get();
		std::set<Grid*> spGrids;
		for (auto iter = mapContents.begin(), iend = mapContents.end(); iter != iend; iter++) {
			auto pContent = iter->second;
			if (!pContent)
				continue;
			auto pGrid = pContent->mpGrid;
			if (!pGrid)
				continue;
			if (spGrids.count(pGrid))
				continue;
			spGrids.insert(pGrid);
		}
		for (auto iter = spGrids.begin(), iend = spGrids.end(); iter != iend; iter++) {
			auto pGrid = *iter;
			pGrid->ConnectedVOs.Clear();
		}

		AllContentMap.Clear();
		ContentMap.Clear();
		MarkerContentMap.Clear();
		MapArucoMarkerPos.Clear();
		AnchorIDs.Clear();
		{
			std::map<int, cv::Mat> mapDatas;
			if (SLAM->TemporalDatas2.Count("path")) {
				mapDatas = SLAM->TemporalDatas2.Get("path");
				mapDatas.clear();
				SLAM->TemporalDatas2.Update("path", mapDatas);

			}
		}
		{
			std::map<int, cv::Mat> mapDatas;
			if (SLAM->TemporalDatas2.Count("pathpos")) {
				mapDatas = SLAM->TemporalDatas2.Get("pathpos");
				mapDatas.clear();
				SLAM->TemporalDatas2.Update("pathpos", mapDatas);
			}
		}
		{
			std::map<int, cv::Mat> mapDatas;
			if (SLAM->TemporalDatas2.Count("content")) {
				mapDatas = SLAM->TemporalDatas2.Get("content");
				mapDatas.clear();
				SLAM->TemporalDatas2.Update("content", mapDatas);
			}
		}

		MarkerProcessor::MapMarkerKFs.Clear();
		MarkerProcessor::MapMarkerPos.Clear();
	}
	void ContentProcessor::DrawContentRegistration(EdgeSLAM::SLAM* SLAM, EdgeSLAM::KeyFrame* pKF, std::string user, cv::Mat data, int mid) {

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		auto pNewContent = new Content(data, user, mid, ts);
		AllContentMap.Update(pNewContent->mnID, pNewContent);
		
		//KF
		if(false)
		{
			std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(100);
			vpLocalKFs.push_back(pKF);
			for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
				auto pKFi = *iter;
				std::map<int, Content*> mapContents;
				if (ContentMap.Count(pKFi))
					mapContents = ContentMap.Get(pKFi);
				mapContents[pNewContent->mnID] = pNewContent;
				ContentMap.Update(pKFi, mapContents);
			}
		}
		
		//GRID
		//그리드
		{
			/*std::map<int, cv::Mat> mapDatas;
			if (SLAM->TemporalDatas2.Count("content"))
				mapDatas = SLAM->TemporalDatas2.Get("content");*/
			cv::Mat X = cv::Mat::zeros(3, 1, CV_32FC1);
			X.at<float>(0) = data.at<float>(4);
			X.at<float>(1) = -data.at<float>(5);
			X.at<float>(2) = data.at<float>(6);
			pNewContent->mpGrid = GridProcessor::GetGrid(X);
			pNewContent->mpGrid->ConnectedVOs.Update(pNewContent);
			/*mapDatas[pNewContent->mnID] = X;
			SLAM->TemporalDatas2.Update("content", mapDatas);*/
		}

		/*{
			std::map<int, cv::Mat> mapDatas;
			if (SLAM->TemporalDatas2.Count("drawcontent"))
				mapDatas = SLAM->TemporalDatas2.Get("drawcontent");
			cv::Mat x = cv::Mat::zeros(2, 1, CV_32FC1);
			x.at<float>(0) = data.at<float>(7);
			x.at<float>(1) = data.at<float>(8);
			mapDatas[pNewContent->mnID] = x;
			SLAM->TemporalDatas2.Update("drawcontent", mapDatas);
		}*/
		
	}
	int ContentProcessor::SimulRegistration(EdgeSLAM::SLAM* SLAM, std::string user, const cv::Mat& data, int mid) {
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();
		auto pNewContent = new Content(data, user, mid, ts);
		AllContentMap.Update(pNewContent->mnID, pNewContent);
		return pNewContent->mnID;
	}
	int ContentProcessor::ContentRegistration(EdgeSLAM::SLAM* SLAM, EdgeSLAM::KeyFrame* pKF, std::string user, const cv::Mat& data, int mid, int nKF) {
		
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		auto pNewContent = new Content(data, user, mid, ts);
		//std::cout <<"Add = " << data.t() << std::endl;
		AllContentMap.Update(pNewContent->mnID, pNewContent);
		
		//키프레임
		{
			std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(nKF);
			vpLocalKFs.push_back(pKF);
			for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
				auto pKFi = *iter;
				std::map<int, Content*> mapContents;
				if (ContentMap.Count(pKFi))
					mapContents = ContentMap.Get(pKFi);
				mapContents[pNewContent->mnID] = pNewContent;
				ContentMap.Update(pKFi, mapContents);
			}
		}
		//그리드
		{
			std::map<int, cv::Mat> mapDatas;
			if (SLAM->TemporalDatas2.Count("content"))
				mapDatas = SLAM->TemporalDatas2.Get("content");
			cv::Mat X = cv::Mat::zeros(3, 1, CV_32FC1);
			X.at<float>(0) = data.at<float>(3);
			X.at<float>(1) = -data.at<float>(4);
			X.at<float>(2) = data.at<float>(5);
			mapDatas[pNewContent->mnID] = X;
			//std::cout << "add = " << X.t() << std::endl;
			SLAM->TemporalDatas2.Update("content", mapDatas);
			pNewContent->mpGrid = GridProcessor::GetGrid(X);
			pNewContent->mpGrid->ConnectedVOs.Update(pNewContent);
		}
		
		//std::cout << "temp content" << data.at<float>(0) << " " << data.at<float>(1) << " " << data.at<float>(5) << " " << data.at<float>(7) << " || " << data.at<float>(6) << " " << data.at<float>(8) << std::endl;
		//std::cout << "temp content POS = " << data.at<float>(2) << " " << data.at<float>(3) << " " << data.at<float>(4) << " " << data.at<float>(5) << " " << data.at<float>(6) << std::endl;
		return pNewContent->mnID;
	}

	int ContentProcessor::MarkerContentRegistration(EdgeSLAM::SLAM* SLAM, EdgeSLAM::KeyFrame* pKF, std::string user, cv::Mat data, int id) {
		
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		auto pNewContent = new Content(data, user,id, ts);
		MarkerContentMap.Update(pNewContent->mnID, pNewContent);
		//pNewContent->mnMarkerID = id;
		MapArucoMarkerPos.Update(id, data);

		/*std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(20);
		vpLocalKFs.push_back(pKF);
		for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
			auto pKFi = *iter;
			std::map<int, Content*> mapContents;
			if (ContentMap.Count(pKFi))
				mapContents = ContentMap.Get(pKFi);
			mapContents[pNewContent->mnID] = pNewContent;
			ContentMap.Update(pKFi, mapContents);
		}*/

		return pNewContent->mnID;
	}
	int ContentProcessor::PathContentRegistration(EdgeSLAM::SLAM* SLAM, int sid, int eid, std::string user, cv::Mat data, int mid) {

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		auto pUser = SLAM->GetUser(user);
		auto pNewContent = new Content(data, user, mid, ts);
		//cv::Mat X = data.rowRange(0, 3).clone();
		/*pNewContent->attribute.at<float>(0, 0) = 1.0;
		pNewContent->endPos = data.rowRange(3,6).clone();
		pNewContent->mpPath = new Path();*/

		AllContentMap.Update(pNewContent->mnID, pNewContent);
		//두 지점 사이의 키프레임을 추가하도록 하기
		
		std::map<int, cv::Mat> mapDatas;
		if (SLAM->TemporalDatas2.Count("pathpos"))
			mapDatas = SLAM->TemporalDatas2.Get("pathpos");

		//auto vpLocalKFs = pUser->mSetLocalKeyFrames.Get();
		////std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(20);
		////vpLocalKFs.push_back(pKF);
		std::set<EdgeSLAM::KeyFrame*> spStartKFs = MarkerProcessor::MapMarkerKFs.Get(sid);
		std::set<EdgeSLAM::KeyFrame*> spEndKFs = MarkerProcessor::MapMarkerKFs.Get(eid);

		//마지막 키프레임의 인접 키프레임 추가하기
		std::set<EdgeSLAM::KeyFrame*> spKFs;
		{
			auto spLocalKFs = spStartKFs;
			for (auto iter = spLocalKFs.begin(), iend = spLocalKFs.end(); iter != iend; iter++) {
				//인접 얻기
				auto pKFi = *iter;
				if (spKFs.count(pKFi))
					continue;
				spKFs.insert(pKFi);
				mapDatas[pKFi->mnId] = pKFi->GetCameraCenter();
			}
			spLocalKFs = spEndKFs;
			for (auto iter = spLocalKFs.begin(), iend = spLocalKFs.end(); iter != iend; iter++) {
				//인접 얻기
				auto pKFi = *iter;
				if (spKFs.count(pKFi))
					continue;
				spKFs.insert(pKFi);
				mapDatas[pKFi->mnId] = pKFi->GetCameraCenter();
			}
		}
		
		for (auto iter = spKFs.begin(), iend = spKFs.end(); iter != iend; iter++) {
			//인접 얻기
			auto pKFi = *iter;
			if (!pKFi)
				continue;
			std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKFi->GetBestCovisibilityKeyFrames(30);
			//추가하기
			for (auto jter = vpLocalKFs.begin(), jend = vpLocalKFs.end(); jter != jend; jter++) {
				auto pKFj = *jter;
				if (spKFs.count(pKFj))
					continue;
				spKFs.insert(pKFj);
				mapDatas[pKFj->mnId] = pKFj->GetCameraCenter();
			}
			//iter가 pKFend이면 멈추기
			/*if (pKFi == pKFend)
				break;*/
		}
		SLAM->TemporalDatas2.Update("pathpos", mapDatas);
		std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = std::vector<EdgeSLAM::KeyFrame*>(spKFs.begin(), spKFs.end());

		for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
			auto pKFi = *iter;
			std::map<int, Content*> mapContents;
			if (ContentMap.Count(pKFi))
				mapContents = ContentMap.Get(pKFi);
			mapContents[pNewContent->mnID] = pNewContent;
			ContentMap.Update(pKFi, mapContents);
		}

		//pUser->mnUsed--;
		return pNewContent->mnID;
	}
	void ContentProcessor::ManageMovingObj(EdgeSLAM::SLAM* SLAM, int id) {
		auto C = AllContentMap.Get(id);
		auto path = C->mpPath;

		std::map<int, cv::Mat> mapDatas;
		if (path) {
			//path->Init(C->pos, C->endPos);
			//path->MoveStart();
			//while (path->bMove) {
			//	auto pos = path->Move();
			//	//객체 등록
			//	mapDatas[id] = pos;
			//	SLAM->TemporalDatas2.Update("MovingObject", mapDatas);
			//}
		}
		mapDatas.clear();
		SLAM->TemporalDatas2.Update("MovingObject", mapDatas);
	}
	void ContentProcessor::MovingObjectSync(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		if (AllContentMap.Count(id)) {
			
			cv::Mat data = cv::Mat::zeros(1000, 1, CV_32FC1);
			data.at<float>(0) = id;
			{
				WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
				std::stringstream ss;
				ss << "/Store?keyword=VO.MOVE&id=" << id << "&src=" << user;
				std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
				auto res = mpAPI->Send(ss.str(), data.data, data.rows * sizeof(float));
				std::chrono::high_resolution_clock::time_point e = std::chrono::high_resolution_clock::now();
				delete mpAPI;
			}
			SLAM->pool->EnqueueJob(ManageMovingObj, SLAM, id);
		}
		else {
			std::cout << "MovingObjectSync=error" << std::endl;
		}
	}

	void ContentProcessor::UpdateProcess(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		if (AllContentMap.Count(id)) {
			
			auto pContent = AllContentMap.Get(id);

			std::stringstream ss;
			ss << "/Load?keyword=VO.MANIPULATE" << "&id=" << id << "&src=" << user;
			WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
			auto res = mpAPI->Send(ss.str(), "");
			int n2 = res.size();
			cv::Mat fdata = cv::Mat(n2 / 4, 1, CV_32FC1, (void*)res.data());
			/*cv::Mat fdata = cv::Mat::zeros(n2 / 4, 1, CV_32FC1);
			std::memcpy(fdata.data, res.data(), res.size());*/
			float len = fdata.at<float>(0);
			pContent->data = fdata.rowRange(0, len).clone();
			//std::cout << fdata.rowRange(13, 17).t() << std::endl;
			{
				std::map<int, cv::Mat> mapDatas;
				if (SLAM->TemporalDatas2.Count("content"))
					mapDatas = SLAM->TemporalDatas2.Get("content");
				cv::Mat X = cv::Mat::zeros(3, 1, CV_32FC1);
				X.at<float>(0) = fdata.at<float>(3);
				X.at<float>(1) = -fdata.at<float>(4);
				X.at<float>(2) = fdata.at<float>(5);
				mapDatas[pContent->mnID] = X;
				SLAM->TemporalDatas2.Update("content", mapDatas);
				//std::cout << "upadte = " <<pContent->mnID<<" " << X.t() << std::endl;

				//GridProcessor::ConnectVO(X, id);
				//그리드가 다르면 변경해야 함.
				auto pPrevGrid = pContent->mpGrid;
				if (pPrevGrid) {
					auto pCurrGrid = GridProcessor::GetGrid(X);
					std::cout << pCurrGrid->pos.t() << std::endl; //삭제 해야 함. 왜 안되는지는 모름
					if (pPrevGrid != pCurrGrid) {
						/*if (!pCurrGrid)
							std::cout << "grid error ?????????" << std::endl;
						std::cout << "grid test = " << pPrevGrid->pos.t()<<" " << pCurrGrid->pos.t() << std::endl;
						*/
						pPrevGrid->ConnectedVOs.Erase(pContent);
						pCurrGrid->ConnectedVOs.Update(pContent);
						pContent->mpGrid = pCurrGrid;
					}
				}
			}

			pContent->mnLastUpdatedTime = ts;

			/*cv::Mat fdata = cv::Mat::zeros(1000, 1, CV_32FC1);
			std::memcpy(fdata.data, res.data(), res.size());
			cv::Mat X = cv::Mat::zeros(3, 1, CV_32FC1);
			X.at<float>(0) = fdata.at<float>(2);
			X.at<float>(1) = fdata.at<float>(3);
			X.at<float>(2) = fdata.at<float>(4);
			pContent->pos = X.clone();*/
			//가상 객체 그래프 변경도 되어야 함.

		}
		else {
			std::cout << "error???????????" << std::endl;
		}

	}

	/////Anchor 관련 
	void ContentProcessor::SetAnchor(EdgeSLAM::SLAM* SLAM, std::string user, int id, std::string kewword, int mid) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;
		auto pKF = pUser->mpRefKF;
		if (!pKF) {
			pUser->mnUsed--;
			return;
		}
		//std::cout << "set anchor start = " <<mid<< std::endl;
		auto pMap = SLAM->GetMap(pUser->mapName);

		std::stringstream ss;
		ss << "/Load?keyword=" << kewword << "&id=" << id << "&src=" << user;
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");

		std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(20);
		vpLocalKFs.push_back(pKF);
		for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
			auto pKFi = *iter;
			std::map<int, std::string> mapAnchors;
			if (AnchorIDs.Count(pKFi))
				mapAnchors = AnchorIDs.Get(pKFi);
			mapAnchors[id] = res;
			AnchorIDs.Update(pKFi, mapAnchors);
		}
		//std::cout << "set anchor end = " <<mid<< std::endl;
		pUser->mnUsed--;
	}
	void ContentProcessor::ShareAnchor(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		//std::cout << "ShareContent start" << std::endl;
		auto pUser = SLAM->GetUser(user);
		if (!pUser) {
			//std::cout << "ShareContent end1" << std::endl;
			return;
		}
		pUser->mnUsed++;
		auto pKF = pUser->mpRefKF;
		if (!pKF) {
			//std::cout << "ShareContent end2" << std::endl;
			pUser->mnUsed--;
			return;
		}
		std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(20);
		vpLocalKFs.push_back(pKF);
		auto pMap = SLAM->GetMap(pUser->mapName);

		std::map<int, std::string> mapContents;
		for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
			auto pKFi = *iter;
			
			if (AnchorIDs.Count(pKFi)) {
				mapContents = AnchorIDs.Get(pKFi);
				for (auto jter = mapContents.begin(), jend = mapContents.end(); jter != jend; jter++) {
					//auto pContent = jter->second;
					//if (!setIDs.count(pContent))
						//setIDs.insert(pContent);
					mapContents[jter->first] = jter->second;
					//nContent++;
				}
			}
		}
		
		if (mapContents.size() == 1) {
			std::string str = "";
			int mid = -1;
			for (auto iter = mapContents.begin(), iend = mapContents.end(); iter != iend; iter++) {
				str = iter->second;
				mid = iter->first;
			}
			{
				WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
				std::stringstream ss;
				ss << "/Store?keyword=GetCloudAnchor&id=" << mid << "&src=" << user;
				std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
				auto res = mpAPI->Send(ss.str(), str);
				std::chrono::high_resolution_clock::time_point e = std::chrono::high_resolution_clock::now();
				delete mpAPI;
			}
		}

		pUser->mnUsed--;

	}
	/////Anchor 관련 
}