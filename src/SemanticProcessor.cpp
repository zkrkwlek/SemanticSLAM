#include <SemanticProcessor.h>
#include <random>
#include <Utils.h>
#include <User.h>
#include <Camera.h>
#include <Map.h>
#include <KeyFrame.h>
#include <Frame.h>
#include <MapPoint.h>
#include <FeatureTracker.h>

#include <SemanticLabel.h>
#include <ObjectOptimizer.h>
#include <PlaneEstimator.h>
#include <GridProcessor.h>
#include <GridCell.h>
#include <ObjectFrame.h>
#include <ObjectSearchPoints.h>
#include <Optimizer.h>
#include <SLAM.h>

#include <DynamicTrackingProcessor.h>

namespace SemanticSLAM {
	SemanticProcessor::SemanticProcessor() {}
	SemanticProcessor::~SemanticProcessor() {}

	std::string SemanticProcessor::strLabel = "wall,building,sky,floor,tree,ceiling,road,bed,windowpane,grass,cabinet,sidewalk,person,earth,door,table,mountain,plant,curtain,chair,car,water,painting,sofa,shelf,house,sea,mirror,rug,field,armchair,seat,fence,desk,rock,wardrobe,lamp,bathtub,railing,cushion,base,box,column,signboard,chest of drawers,counter,sand,sink,skyscraper,fireplace,refrigerator,grandstand,path,stairs,runway,case,pool table,pillow,screen door,stairway,river,bridge,bookcase,blind,coffee table,toilet,flower,book,hill,bench,countertop,stove,palm,kitchen island,computer,swivel chair,boat,bar,arcade machine,hovel,bus,towel,light,truck,tower,chandelier,awning,streetlight,booth,television,airplane,dirt track,apparel,pole,land,bannister,escalator,ottoman,bottle,buffet,poster,stage,van,ship,fountain,conveyer belt,canopy,washer,plaything,swimming pool,stool,barrel,basket,waterfall,tent,bag,minibike,cradle,oven,ball,food,step,tank,trade name,microwave,pot,animal,bicycle,lake,dishwasher,screen,blanket,sculpture,hood,sconce,vase,traffic light,tray,ashcan,fan,pier,crt screen,plate,monitor,bulletin board,shower,radiator,glass,clock,flag";
	std::string SemanticProcessor::strYoloObjectLabel = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush";
	
	EdgeSLAM::SLAM* SemanticProcessor::SLAM;
	EdgeSLAM::FeatureTracker* ObjectSearchPoints::Matcher;

	//ConcurrentSet<int> ObjectWhiteList, LabelWhiteList;

	ConcurrentMap<int, std::vector<cv::Point2f>> SemanticProcessor::SuperPoints;
	ConcurrentMap<int, cv::Mat> SemanticProcessor::SemanticLabelImage;
	ConcurrentMap<int, ObjectLabel*> SemanticProcessor::ObjectLabels;
	ConcurrentMap<int, SemanticLabel*> SemanticProcessor::SemanticLabels;
	std::vector<std::string> SemanticProcessor::vecStrSemanticLabels;
	std::vector<std::string> SemanticProcessor::vecStrObjectLabels;
	std::vector<cv::Vec3b> SemanticProcessor::SemanticColors;

	////오브젝트 포즈 그래프
	//오브젝트-맵포인트
	//맵포인트-오브젝트
	//오브젝트-키프레임
	//키프레임-오브젝트
	//ConcurrentMap<EdgeSLAM::MapPoint*, std::set<EdgeSLAM::ObjectBoundingBox*>> EdgeSLAM::SLAM::GraphMapPointAndBoundingBox;

	ConcurrentMap<EdgeSLAM::ObjectNode*, std::set<EdgeSLAM::KeyFrame*>> SemanticProcessor::GraphObjectKeyFrame;
	ConcurrentMap<EdgeSLAM::KeyFrame*, std::set<EdgeSLAM::ObjectNode*>> SemanticProcessor::GraphKeyFrameObject;
	ConcurrentMap<EdgeSLAM::KeyFrame*, std::set<EdgeSLAM::ObjectBoundingBox*>> SemanticProcessor::GraphKeyFrameObjectBB;
	ConcurrentMap<int, std::set<EdgeSLAM::ObjectBoundingBox*>> SemanticProcessor::GraphFrameObjectBB;

	ConcurrentMap<int, int> SemanticProcessor::GlobalObjectCount;
	ConcurrentMap<int, int> SemanticProcessor::GlobalLabelCount;

	void SemanticProcessor::Init(EdgeSLAM::SLAM* _SLAM) {
		cv::Mat colormap = cv::Mat::zeros(256, 3, CV_8UC1);
		cv::Mat ind = cv::Mat::zeros(256, 1, CV_8UC1);
		for (int i = 1; i < ind.rows; i++) {
			ind.at<uchar>(i) = i;
		}

		for (int i = 7; i >= 0; i--) {
			for (int j = 0; j < 3; j++) {
				cv::Mat tempCol = colormap.col(j);
				int a = pow(2, j);
				int b = pow(2, i);
				cv::Mat temp = ((ind / a) & 1) * b;
				tempCol |= temp;
				tempCol.copyTo(colormap.col(j));
			}
			ind /= 8;
		}

		for (int i = 0; i < colormap.rows; i++) {
			cv::Vec3b color = cv::Vec3b(colormap.at<uchar>(i, 0), colormap.at<uchar>(i, 1), colormap.at<uchar>(i, 2));
			SemanticColors.push_back(color);
		}

		ObjectWhiteList.Update((int)MovingObjectLabel::CHAIR);
		//ObjectWhiteList.Update((int)MovingObjectLabel::PERSON);
		ObjectCandidateList.Update((int)MovingObjectLabel::SUITCASE, (int)MovingObjectLabel::CHAIR);
		ObjectCandidateList.Update((int)MovingObjectLabel::HANDBAG, (int)MovingObjectLabel::CHAIR);
		LabelWhiteList.Update((int)StructureLabel::CHAIR);

		vecStrSemanticLabels = Utils::Split(strLabel, ",");
		vecStrObjectLabels = Utils::Split(strYoloObjectLabel, ",");
		SLAM = _SLAM;
		ObjectSearchPoints::Matcher = SLAM->mpFeatureTracker;
		DynamicTrackingProcessor::Init();
	}

	void SemanticProcessor::SendLocalMap(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		auto pKF = pUser->mpRefKF;
		if (!pKF)
			return;
		pUser->mnUsed++;

		auto spLocalKFs = pUser->mSetLocalKeyFrames.Get();
		auto vpLocalKFs = std::vector<EdgeSLAM::KeyFrame*>(spLocalKFs.begin(), spLocalKFs.end());
		std::set<EdgeSLAM::MapPoint*> spMPs;
		std::vector<EdgeSLAM::MapPoint*> vpLocalMPs;

		cv::Mat pts = cv::Mat::zeros(0, 1, CV_32FC1);
		cv::Mat desc = cv::Mat::zeros(0,1, CV_8UC1);
		int nInput = 0;

		for (std::vector<EdgeSLAM::KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			EdgeSLAM::KeyFrame* pKFi = *itKF;
			if (!pKFi)
				continue;
			const std::vector<EdgeSLAM::MapPoint*> vpMPs = pKFi->GetMapPointMatches();

			int nInputTemp = 0;
			for (std::vector<EdgeSLAM::MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
			{
				EdgeSLAM::MapPoint* pMPi = *itMP;
				if (!pMPi || pMPi->isBad() || spMPs.count(pMPi))
					continue;
				vpLocalMPs.push_back(pMPi);
				spMPs.insert(pMPi);
				pts.push_back(pMPi->GetWorldPos());
				desc.push_back(pMPi->GetDescriptor().t());

				if (nInput == 0) {
					nInputTemp += 52;
				}
			}
			if (nInput == 0)
				nInput = nInputTemp;
		}
		pUser->mnUsed--;

		cv::Mat converted_desc = cv::Mat::zeros(vpLocalMPs.size() * 8, 1, CV_32FC1);
		std::memcpy(converted_desc.data, desc.data, desc.rows);
		pts.push_back(converted_desc);
		int nOutput = pts.rows * 4;
		//std::cout << "3" <<" "<<vpLocalMPs.size()<<" "<<pts.rows<<" "<<desc.rows<<" "<<converted_desc.rows << std::endl;
		
		//입력 = id + 2d + 3d + desc = 4 + 8 + 12 + 32
		//출력 = id + desc + 3d(여기도 원래 아이디가 포함되어야함)

		{
			std::stringstream ssfile1;
			ssfile1 << "../bin/normal/base.txt";
			std::ofstream f1;
			f1.open(ssfile1.str().c_str(), std::ios_base::out | std::ios_base::app);
			f1 << id << " " << nInput << " " << nOutput << std::endl;
			f1.close();
		}

		{
			WebAPI API("143.248.6.143", 35005);
			std::stringstream ss;
			ss << "/Store?keyword=UpdatedLocalMap&id=" << id << "&src=" << user;
			//std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
			auto res = API.Send(ss.str(), pts.data, pts.rows * sizeof(float));
			//std::chrono::high_resolution_clock::time_point e = std::chrono::high_resolution_clock::now();
		}
	}

	void SemanticProcessor::LabelMapPoint(EdgeSLAM::SLAM* SLAM, std::string user, int id, const cv::Mat& labeled) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		if (!pUser->KeyFrames.Count(id))
			return;
		/*if (!pUser->ImageDatas.Count(id))
		{
			return;
		}*/
		pUser->mnDebugLabel++;
		//pUser->mnUsed++;
		auto pKF = pUser->KeyFrames.Get(id);
		//cv::Mat encoded = pUser->ImageDatas.Get(id);
		//cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);

		
		//std::cout << "LabelMapPoint = " << spNewBBs.size() << std::endl;

		for (int i = 0, iend = pKF->N; i < iend; i++) {
			auto pMPi = pKF->mvpMapPoints.get(i);
			if (!pMPi || pMPi->isBad())
				continue;
			auto pt = pKF->mvKeys[i].pt;
			SemanticLabel* pLabel = nullptr;
			if (!SemanticLabels.Count(pMPi->mnId)) {
				pLabel = new SemanticLabel();
				SemanticLabels.Update(pMPi->mnId, pLabel);
			}
			else {
				pLabel = SemanticLabels.Get(pMPi->mnId);
			}
			
			int label = labeled.at<uchar>(pt) + 1;
			int c = 0;
			if (pLabel->LabelCount.Count(label))
				c = pLabel->LabelCount.Get(label);
			pLabel->LabelCount.Update(label, c + 1);

			int n1 = 0;
			int n2 = 0;
			int n3 = 0;
			if (pLabel->LabelCount.Count((int)StructureLabel::FLOOR))
				n1 = pLabel->LabelCount.Get((int)StructureLabel::FLOOR);
			if (pLabel->LabelCount.Count((int)StructureLabel::WALL))
				n2 = pLabel->LabelCount.Get((int)StructureLabel::WALL);
			if (pLabel->LabelCount.Count((int)StructureLabel::CEIL))
				n3 = pLabel->LabelCount.Get((int)StructureLabel::CEIL);

			auto val = std::max(std::max(n1, n2), n3);

			if (val == n1) {
				pMPi->mnLabelID = (int)StructureLabel::FLOOR;
				//mapDatas[pMPi->mnId] = pMPi->GetWorldPos();
				//labelDatas[pMPi->mnId] = cv::Mat::ones(1, 1, CV_8UC1)*(int)StructureLabel::FLOOR;
			}
			if (val == n2) {
				pMPi->mnLabelID = (int)StructureLabel::WALL;
				//mapDatas[pMPi->mnId] = pMPi->GetWorldPos();
				//labelDatas[pMPi->mnId] = cv::Mat::ones(1, 1, CV_8UC1)*(int)StructureLabel::WALL;
			}
			if (val == n3) {
				pMPi->mnLabelID = (int)StructureLabel::CEIL;
				//mapDatas[pMPi->mnId] = pMPi->GetWorldPos();
				//labelDatas[pMPi->mnId] = cv::Mat::ones(1, 1, CV_8UC1)*(int)StructureLabel::CEIL;
			}

			//cv::circle(img, pt, 3, SemanticColors[label], -1);
		}
		SemanticLabelImage.Update(id, labeled);
		//EstimateLocalMapPlanes(SLAM, user, id);
		pUser->mnDebugLabel--;
		
		SLAM->pool->EnqueueJob(SemanticProcessor::CreateBoundingBox, SLAM, user, id, pKF, labeled);
		//여기서 바운딩 박스 관련 업데이트가 필요하다


		//{
		//	/////save image
		//	std::stringstream sss;
		//	sss << "../bin/img/" << user << "/Track/" << id << "_label.jpg";
		//	cv::imwrite(sss.str(), img);
		//	/////save image
		//}

		//pUser->mnUsed--;
		
	}

	void SemanticProcessor::CreateBoundingBox(EdgeSLAM::SLAM* SLAM, std::string user, int id, EdgeSLAM::KeyFrame* pTargetKF, cv::Mat labeled) {
		////객체 레이블 포인트 테스트
		//MP도 추가할 예정
		std::set<EdgeSLAM::ObjectBoundingBox*> spNewBBs;
		if (!GraphKeyFrameObjectBB.Count(pTargetKF)) {
			return;
		}
		spNewBBs = GraphKeyFrameObjectBB.Get(pTargetKF);
		std::set<int> testLabelID;
		std::map<int, int> LabelCountTest = GlobalLabelCount.Get();
		std::map < EdgeSLAM::ObjectBoundingBox*, std::set<int>> LabelCountTest2;
		
		for (auto oter = spNewBBs.begin(), oend = spNewBBs.end(); oter != oend; oter++) {
			auto pBBox = *oter;
			LabelCountTest2[pBBox] = std::set<int>();
		}

		for (int k = 0, kend = pTargetKF->N; k < kend; k++) {
			
			auto pt = pTargetKF->mvKeys[k].pt;
			int label = labeled.at<uchar>(pt) + 1;
			if (label == (int)StructureLabel::FLOOR)
				continue;
			if (label == (int)StructureLabel::WALL)
				continue;
			if (label == (int)StructureLabel::CEIL) 
				continue;
			if (label == (int)StructureLabel::BUILDING)
				continue;
			LabelCountTest[label]++;

			for (auto oter = spNewBBs.begin(), oend = spNewBBs.end(); oter != oend; oter++) {
				auto pBBox = *oter;

				if (!pBBox->rect.contains(pt))
					continue;
				
				if (!LabelCountTest2[pBBox].count(label))	
					LabelCountTest2[pBBox].insert(label);
				/*if (label != 20)
					continue;*/

				auto pMP = pTargetKF->GetMapPoint(k);
				if (!pMP || pMP->isBad()) {
					pMP = nullptr;
				}
				cv::Mat row = pTargetKF->mDescriptors.row(k);
				pBBox->mvIDXs.push_back(k);
				pBBox->mapIDXs[k] = pBBox->mvKeys.size();
				pBBox->mvKeys.push_back(pTargetKF->mvKeysUn[k]);
				pBBox->desc.push_back(row.clone());
				pBBox->AddMapPoint(pMP);
			}
			
		}

		//label test
		/*for (auto iter = LabelCountTest2.begin(), iend = LabelCountTest2.end(); iter != iend; iter++) {
			auto pBBox = iter->first;
			auto setLabels = iter->second;
			std::string objName = vecStrObjectLabels[pBBox->label - 1];
			int objCount = GlobalObjectCount.Get(pBBox->label);
			std::cout << "BBOX = " << objName <<", id = "<<pBBox->id << "== label = " << pBBox->label << " " << objCount << std::endl;
			for (auto lter = setLabels.begin(), lend = setLabels.end(); lter != lend; lter++) {
				auto label = *lter;
				auto count = LabelCountTest[label];
				std::cout << "Label Count = " <<objName<<" = "<< vecStrSemanticLabels[label - 1] << " " << label <<" "<<count << std::endl;
			}
		}
		for (auto iter = LabelCountTest.begin(), iend = LabelCountTest.end(); iter != iend; iter++) {
			auto label = iter->first;
			auto count = iter->second;
			GlobalLabelCount.Update(label, count);
		}*/
		//label test

		//일정 이하의 키포인트가 있는 박스는 삭제
		std::map<EdgeSLAM::ObjectBoundingBox*, bool> TempBBs;
		for (auto oter = spNewBBs.begin(), oend = spNewBBs.end(); oter != oend; oter++) {
			auto pBBox = *oter;
			pBBox->N = pBBox->desc.rows;
			/*bool b = false;
			if ()
				b = true;*/
			TempBBs[pBBox] = pBBox->N < 20;
		}
		for (auto oter = TempBBs.begin(), oend = TempBBs.end(); oter != oend; oter++) {
			auto b = oter->second;
			auto pBox = oter->first;
			if (b)
				spNewBBs.erase(pBox);
			else
				pBox->ComputeBow(SLAM->mpDBoWVoc);
			//else {
			//	//이미지 얻고
			//	//포즈로 테스트
			//	cv::Mat Pose;
			//	int nRes = EdgeSLAM::Optimizer::ObjectPoseOptimization(pBox, Pose);
			//	if (nRes > 15) {
			//		//이미지로 확인하기
			//		auto pUser = SLAM->GetUser(user);
			//		if (!pUser)
			//			return;
			//		if (!pUser->ImageDatas.Count(id)) {
			//			return;
			//		}
			//		pUser->mnUsed++;
			//		cv::Mat K = pUser->GetCameraMatrix();
			//		cv::Mat encoded = pUser->ImageDatas.Get(id);
			//		std::string mapName = pUser->mapName;
			//		pUser->mnUsed--;

			//		auto vecMPs = pBox->mvpMapPoints.get();
			//		float Na = 0;
			//		cv::Mat avgPos = cv::Mat::zeros(3, 1, CV_32FC1);
			//		
			//		for (int i = 0; i < vecMPs.size(); i++) {

			//			auto pMP = vecMPs[i];
			//			if (!pMP || pMP->isBad())
			//				continue;
			//			Na++;
			//			avgPos += pMP->GetWorldPos();
			//		}
			//		avgPos /= Na;

			//		cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);
			//		cv::Mat Rco = Pose.rowRange(0, 3).colRange(0, 3);
			//		cv::Mat tco = Pose.rowRange(0, 3).col(3);
			//		cv::Mat temp = Rco * avgPos + tco;
			//		temp = K * temp;
			//		float depth = temp.at<float>(2);
			//		cv::Point2f pt(temp.at<float>(0) / depth, temp.at<float>(1) / depth);
			//		cv::circle(img, pt, 50, cv::Scalar(255, 0, 255), 3);
			//		SLAM->VisualizeImage(mapName, img, 3);
			//	}
			//}
		}
		GraphKeyFrameObjectBB.Update(pTargetKF, spNewBBs);
		SLAM->pool->EnqueueJob(SemanticProcessor::CheckDynamicObject, SLAM, user, id);
	}

	void SemanticProcessor::DownloadSuperPoint(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		
		/*auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;*/

		std::stringstream ss;
		ss << "/Load?keyword=Keypoints" << "&id=" << id << "&src=" << user; 
		WebAPI API("143.248.6.143", 35005);
		auto res = API.Send(ss.str(), "");
		int n2 = res.size() / 8;

		cv::Mat temp = cv::Mat::zeros(n2, 2, CV_32FC1);
		std::memcpy(temp.data, res.data(), res.size());

		/*if (!pUser->ImageDatas.Count(id)) {
			pUser->mnUsed--;
			return;
		}
		cv::Mat encoded = pUser->ImageDatas.Get(id);
		cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);*/

		std::vector<cv::Point2f> mvPts;
		for (int i = 0; i < n2; i++) {
			cv::Point2f pt(temp.at<float>(2 * i), temp.at<float>(2 * i + 1));
			mvPts.push_back(pt);
			//cv::circle(img, pt, 3, cv::Scalar(255, 0, 255), -1);
		}
		SuperPoints.Update(id, mvPts);
		//SLAM->VisualizeImage(img, 1);

		//int N = mvPts.size();
		//cv::Mat mat(N, 2, CV_32F);

		//for (int i = 0; i<N; i++)
		//{
		//	mat.at<float>(i, 0) = mvPts[i].x;
		//	mat.at<float>(i, 1) = mvPts[i].y;
		//}

		//// Undistort points
		//mat = mat.reshape(2);
		//cv::undistortPoints(mat, mat, pUser->GetCameraM, pUser->D, cv::Mat(), pUser->K);
		//mat = mat.reshape(1);
		//for (int i = 0; i<N; i++)
		//{
		//	mvPts[i].x = mat.at<float>(i, 0);
		//	mvPts[i].y = mat.at<float>(i, 1);
		//}

		//pUser->mnUsed--;
	}
	void SemanticProcessor::ShareSemanticInfo(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		//현재 프레임과 수퍼포인트 매칭
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

		//KeyFrame
		int kfID = pKF->mnFrameId;
		if (!pUser->ImageDatas.Count(kfID)) {
			pUser->mnUsed--;
			return;
		}
		cv::Mat encoded = pUser->ImageDatas.Get(kfID);
		if (!SuperPoints.Count(kfID)) {
			pUser->mnUsed--;
			return;
		}
		auto cornersA = SuperPoints.Get(kfID);
		int N = cornersA.size();
		cv::Mat data = cv::Mat::zeros(2 * N + 1,1, CV_32FC1);
		int idx = 0;
		data.at<float>(idx++) = (float)N;
		for (int i = 0; i < N; i++) {
			auto pt = cornersA[i];
			data.at<float>(idx++) = pt.x;
			data.at<float>(idx++) = pt.y;
		}
		//이미지와 코너 정보 전송
		//처음 인덱스는 포인트 수, N, 2*N, 나머지는 바이트로 읽기

		size_t len = sizeof(float)*(data.rows) + sizeof(char)*encoded.rows;
		unsigned char* vdata = (unsigned char*)malloc(len);
		memcpy(vdata, data.data, sizeof(float)*(data.rows));
		memcpy(vdata + sizeof(float)*(data.rows), encoded.data, sizeof(char)*encoded.rows);

		WebAPI API("143.248.6.143", 35005);
		std::stringstream ss;
		pUser->mnUsed--;
		ss << "/Store?keyword=ShareSemanticInfo&id=" << id << "&src=" << user;// << "&type2=" << user->userName;
		auto res = API.Send(ss.str(), vdata, len);
		free(vdata);
	}
	void SemanticProcessor::MatchingSuperPoint(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		//현재 프레임과 수퍼포인트 매칭
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

		//KeyFrame
		int kfID = pKF->mnFrameId;
		if (!pUser->ImageDatas.Count(kfID)) {
			pUser->mnUsed--;
			return;
		}
		if (!pUser->ImageDatas.Count(id)) {
			pUser->mnUsed--;
			return;
		}
		
		if (!SuperPoints.Count(kfID)) {
			pUser->mnUsed--;
			return;
		}
		cv::Mat encodedKF = pUser->ImageDatas.Get(kfID);
		cv::Mat imgKF = cv::imdecode(encodedKF, cv::IMREAD_COLOR);

		//Frame
		cv::Mat encoded = pUser->ImageDatas.Get(id);
		cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);

		auto cornersA = SuperPoints.Get(kfID);

		//OpticalFlow Matching
		int win_size = 10;
		std::vector<cv::Point2f> cornersB;
		std::vector<uchar> features_found;
		//std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		cv::calcOpticalFlowPyrLK(
			imgKF,                         // Previous image
			img,                         // Next image
			cornersA,                     // Previous set of corners (from imgA)
			cornersB,                     // Next set of corners (from imgB)
			features_found,               // Output vector, each is 1 for tracked
			cv::noArray(),                // Output vector, lists errors (optional)
			cv::Size(win_size * 2 + 1, win_size * 2 + 1),  // Search window size
			5,                            // Maximum pyramid level to construct
			cv::TermCriteria(
				cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
				20,                         // Maximum number of iterations
				0.3                         // Minimum change per iteration
			)
		);

		for (int i = 0; i < static_cast<int>(cornersA.size()); ++i) {
			if (!features_found[i]) {
				continue;
			}
			line(
				img,                        // Draw onto this image
				cornersA[i],                 // Starting here
				cornersB[i],                 // Ending here
				cv::Scalar(255, 255, 0),       // This color
				3,                           // This many pixels wide
				cv::LINE_AA                  // Draw line in this style
			);
		}
		//std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		//추가로 너무 긴 라인 제거하기

		if(pUser->GetVisID()==0)
			SLAM->VisualizeImage(pUser->mapName, img, 3);
		/*auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		std::cout << "Opticalflow processing time = " << t_test1 << std::endl;*/
		pUser->mnUsed--;
	}
	void SemanticProcessor::SimpleRecon(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		if (pUser->GetVisID() != 0)
			return;
		pUser->mnUsed++;
		pUser->mnDebugSeg++;
		std::stringstream ss;
		ss << "/Load?keyword=Recon" << "&id=" << id << "&src=" << user;
		WebAPI API("143.248.6.143", 35005);
		auto res = API.Send(ss.str(), "");
		int n2 = res.size();

		cv::Mat temp = cv::Mat::zeros(n2, 1, CV_8UC1);
		std::memcpy(temp.data, res.data(), res.size());
		cv::Mat depth = cv::imdecode(temp, cv::IMREAD_UNCHANGED);

		int w = depth.cols;
		int h = depth.rows;

		SLAM->VisualizeImage(pUser->mapName, depth, 0);

		pUser->mnDebugSeg--;
		pUser->mnUsed--;
		//{
		//	/////save image
		//	std::stringstream sss;
		//	sss << "../bin/img/" << user << "/Track/" << id << "_seg.jpg";
		//	cv::imwrite(sss.str(), segcolor);
		//	/////save image
		//}

	}
	void SemanticProcessor::Segmentation(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;
		pUser->mnDebugSeg++;
		std::stringstream ss;
		ss << "/Load?keyword=Segmentation" << "&id=" << id << "&src=" << user;
		WebAPI API("143.248.6.143", 35005);
		auto res = API.Send(ss.str(), "");
		int n2 = res.size();

		cv::Mat temp = cv::Mat::zeros(n2, 1, CV_8UC1);
		std::memcpy(temp.data, res.data(), res.size());
		cv::Mat labeled = cv::imdecode(temp, cv::IMREAD_GRAYSCALE);

		int w = labeled.cols;
		int h = labeled.rows;

		//int oriw = pUser->mpCamera->mnWidth;
		//int orih = pUser->mpCamera->mnHeight;

		//float sw = ((float)w) / oriw; //scaled
		//float sh = ((float)h) / orih;

		cv::Mat segcolor = cv::Mat::zeros(h, w, CV_8UC3);
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int label = labeled.at<uchar>(y, x) + 1;
				segcolor.at<cv::Vec3b>(y, x) = SemanticColors[label];
			}
		}

		cv::Mat segcolor2 = cv::Mat::zeros(h, w, CV_8UC3);
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int label = labeled.at<uchar>(y, x) + 1;
				if(label == (int)StructureLabel::CHAIR || label ==(int)StructureLabel::EARTH || label == (int)StructureLabel::TABLE)
					segcolor2.at<cv::Vec3b>(y, x) = SemanticColors[label];
			}
		}

		if (pUser->GetVisID() == 0) {
			SLAM->VisualizeImage(pUser->mapName, segcolor, 1);
			SLAM->VisualizeImage(pUser->mapName, segcolor2, 2);
		}
		LabelMapPoint(SLAM, user, id, labeled);
		
		PlaneEstimator::PlaneEstimation(SLAM, user, id);
		pUser->mnDebugSeg--;
		pUser->mnUsed--;
		
		//{
		//	/////save image
		//	std::stringstream sss;
		//	sss << "../bin/img/" << user << "/Track/" << id << "_seg.jpg";
		//	cv::imwrite(sss.str(), segcolor);
		//	/////save image 
		//}
		
	}
	//바운딩 박스와 바운딩 박스를 연결해야 함.
	void SemanticProcessor::ObjectUpdate(EdgeSLAM::SLAM* SLAM, std::string user, int id){
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		if (!pUser->ImageDatas.Count(id))
		{
			return;
		}
		if (!pUser->KeyFrames.Count(id)) {
			return;
		}
		auto pKF = pUser->KeyFrames.Get(id);
		if (!pKF) {
			return;
		}
		cv::Mat encoded = pUser->ImageDatas.Get(id);
		cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);
		//cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
		if (img.empty())
		{
			return;
		}
		
		std::set<EdgeSLAM::ObjectBoundingBox*> spNewBBs;
		if (!GraphKeyFrameObjectBB.Count(pKF)) {
			return;
		}
		spNewBBs = GraphKeyFrameObjectBB.Get(pKF);

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		int seg_th = 1;
		int ttt = 0;
		for (auto iter = spNewBBs.begin(), iend = spNewBBs.end(); iter != iend; iter++) 
		{
			//auto pBB = *iter;
			//cv::rectangle(img, pBB->rect, cv::Scalar(255, 255, 0), 2);
			//for (int j = 0, jend = pBB->mvIDXs.size(); j < jend; j++) {
			//	int idx = pBB->mvIDXs[j];
			//	auto pMPj = pKF->mvpMapPoints.get(idx);

			//	cv::circle(img, pBB->mvKeys[j].pt, 3, cv::Scalar(255, 255, 0), 2);

			//	if (!pMPj || pMPj->isBad())
			//		continue;
			//	//if (pMPi->mnObjectID > 0){
			//	//	continue;
			//	//}
			//	
			//	SemanticLabel* pStaticLabel = nullptr;
			//	if (!SemanticLabels.Count(pMPj->mnId)) {
			//		continue;
			//	}
			//	pStaticLabel = SemanticLabels.Get(pMPj->mnId);
			//	if (pStaticLabel->LabelCount.Count((int)StructureLabel::FLOOR) && pStaticLabel->LabelCount.Get((int)StructureLabel::FLOOR) > seg_th) {
			//		continue;
			//	}
			//	if (pStaticLabel->LabelCount.Count((int)StructureLabel::WALL) && pStaticLabel->LabelCount.Get((int)StructureLabel::WALL) > seg_th) {
			//		continue;
			//	}
			//	if (pStaticLabel->LabelCount.Count((int)StructureLabel::CEIL) && pStaticLabel->LabelCount.Get((int)StructureLabel::CEIL) > seg_th) {
			//		continue;
			//	}
			//	pBB->vecMPs[j] = pMPj;
			//	ttt++;
			//}
			//for (int j = 0, jend = pBB->vecMPs.size(); j < jend; j++) {
			//	auto pMPj = pBB->vecMPs[j];
			//	if (pMPj && !pMPj->isBad()) {
			//		cv::circle(img, pBB->mvKeys[j].pt, 3, cv::Scalar(0, 0, 255), 2, -1);
			//	}
			//}
		}
		SLAM->VisualizeImage(pUser->mapName, img, 2);
		 
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		std::cout << "Object update processing time = " << t_test1 << " " <<ttt<< std::endl;

		/*pUser->mnUsed++;
		pUser->mnDebugSeg++;

		cv::Mat T = pUser->PoseDatas.Get(id);
		cv::Mat invK = pUser->GetCameraInverseMatrix();
		int nVisID = pUser->GetVisID();

		pUser->mnDebugSeg--;
		pUser->mnUsed--;
				
		int nTest = 0;
		for (int i = 0, iend = pKF->N; i < iend; i++) {
			auto pMPi = pKF->mvpMapPoints.get(i);
			if (!pMPi || pMPi->isBad())
				continue;
			if (pMPi->mnObjectID == (int)MovingObjectLabel::CHAIR) {
				nTest++;
			}
		}
		std::cout << std::endl << std::endl << "atest = " << nTest <<" "<<pKF->mvbOutliers.size()<<" "<<pKF->N<< std::endl;*/
		
		//std::vector<int> vnIndexes;
		//
		//for (int i = 0, iend = pKF->N; i < iend; i++) {
		//	auto pMPi = pKF->mvpMapPoints.get(i);
		//	if (!pMPi || pMPi->isBad())
		//		continue;
		//	//if (pMPi->mnObjectID > 0){
		//	//	continue;
		//	//}
		//	auto pt = pKF->mvKeys[i].pt;
		//	
		//	SemanticLabel* pStaticLabel = nullptr;
		//	if (!SemanticLabels.Count(pMPi->mnId)) {
		//		continue;
		//	}
		//	pStaticLabel = SemanticLabels.Get(pMPi->mnId);
		//	if (pStaticLabel->LabelCount.Count((int)StructureLabel::FLOOR) && pStaticLabel->LabelCount.Get((int)StructureLabel::FLOOR) > seg_th) {
		//		continue;
		//	}
		//	if (pStaticLabel->LabelCount.Count((int)StructureLabel::WALL) && pStaticLabel->LabelCount.Get((int)StructureLabel::WALL) > seg_th) {
		//		continue;
		//	}
		//	if (pStaticLabel->LabelCount.Count((int)StructureLabel::CEIL) && pStaticLabel->LabelCount.Get((int)StructureLabel::CEIL) > seg_th) {
		//		continue;
		//	}

		//	vnIndexes.push_back(i);
		//	//int label = mask.at<uchar>(pt);
		//	//if (label != (int)MovingObjectLabel::CHAIR) {
		//	//	continue;
		//	//}
		//	//pMPi->mnObjectID = label;

		//	//ObjectLabel* pLabel = nullptr;
		//	//if (!ObjectLabels.Count(pMPi->mnId)) { 
		//	//	pLabel = new ObjectLabel();
		//	//	ObjectLabels.Update(pMPi->mnId, pLabel);
		//	//}
		//	//else {
		//	//	pLabel = ObjectLabels.Get(pMPi->mnId);
		//	//}
		//	////객체값으로 갱신
		//	//int c = 0;
		//	//if (pLabel->LabelCount.Count(label))
		//	//	c = pLabel->LabelCount.Get(label);
		//	//pLabel->LabelCount.Update(label, ++c);
		//	
		//}

		////바운딩 박스와 맵포인트 연결하기
		//for (auto iter = spNewBBs.begin(), iend = spNewBBs.end(); iter != iend; iter++) {
		//	auto pBBox = *iter;
		//	int label = pBBox->label;
		//	if (label != (int)MovingObjectLabel::CHAIR) {
		//		continue;
		//	}
		//	for (int i = 0, iend = vnIndexes.size(); i < iend; i++) {
		//		int idx = vnIndexes[i];
		//		auto pMPi = pKF->mvpMapPoints.get(idx);
		//		if (!pMPi || pMPi->isBad())
		//			continue;
		//		auto pt = pKF->mvKeys[idx].pt;
		//		if (pBBox->rect.contains(pt)) {
		//			pMPi->mspBBs.Update(pBBox);
		//			pMPi->mnObjectID = pBBox->label;
		//			pBBox->vecMPs.push_back(pMPi);
		//		}
		//	}
		//	//std::cout << "Object " << vecStrObjectLabels[pBBox->label-1] << " " << pBBox->vecMPs.size() << std::endl;
		//	//rectangle(mask, pBBox->rect, cv::Scalar(pBBox->label), -1);
		//}

		//////테스트
		//std::set<EdgeSLAM::ObjectBoundingBox*> spBBs;
		//std::set<EdgeSLAM::ObjectNode*> spONs;

		////MP와 연결 된 이전 BB와 ON 찾기
		////여기의 BB는 지금 생성한 것. ON을 생성하거나 이전 노드와 연결이 필요함.
		//for (auto iter = spNewBBs.begin(), iend = spNewBBs.end(); iter != iend; iter++) {
		//	auto pBBox = *iter;
		//	int label = pBBox->label;
		//	if (label != (int)MovingObjectLabel::CHAIR) {
		//		continue;
		//	}
		//	
		//	for (int i = 0; i < pBBox->vecMPs.size(); i++) {
		//		auto pMPi = pBBox->vecMPs[i];
		//		auto spTempBBs = pMPi->mspBBs.Get();
		//		for (auto iter = spTempBBs.begin(), iend = spTempBBs.end(); iter != iend; iter++) {
		//			auto pBB = *iter;
		//			if (!spBBs.count(pBB)) {
		//				spBBs.insert(pBB);
		//			}
		//		}
		//		if (pMPi->mpObjectNode) {
		//			if (!spONs.count(pMPi->mpObjectNode)) {
		//				spONs.insert(pMPi->mpObjectNode);
		//			}
		//		}
		//	}
		//}
		//for (auto iter = spBBs.begin(), iend = spBBs.end(); iter != iend; iter++) {
		//	auto pBBox = *iter;
		//	if (pBBox->mpNode && !spONs.count(pBBox->mpNode)) {
		//		spONs.insert(pBBox->mpNode);
		//	}
		//}
		//////맵포인트와 바운딩 박스 카운트
		//std::map<EdgeSLAM::MapPoint*, int> mapMPCount;
		//std::vector<EdgeSLAM::MapPoint*> vecMPs;
		//if (spBBs.size() > 2) {
		//	for (auto iter = spBBs.begin(), iend = spBBs.end(); iter != iend; iter++) {
		//		auto pBB = *iter;
		//		for (int j = 0; j < pBB->vecMPs.size(); j++) {
		//			auto pMPj = pBB->vecMPs[j];
		//			if (pMPj && !pMPj->isBad()) {
		//				mapMPCount[pMPj]++;
		//			}
		//		}
		//	}
		//	for (auto iter = mapMPCount.begin(), iend = mapMPCount.end(); iter != iend; iter++) {
		//		auto pMPi = iter->first;
		//		auto count = iter->second;
		//		if (count > 2) {
		//			vecMPs.push_back(pMPi);
		//		}
		//	}
		//}

		//////오브젝트 노드와 새로운 바운딩 박스와 맵포인트 연결, 키프레임도 연결
		//if (spONs.size() > 0) {

		//	auto pObjNode = *spONs.begin();

		//	for (auto iter = spNewBBs.begin(), iend = spNewBBs.end(); iter != iend; iter++) {
		//		auto pBB = *iter;
		//		if (pBB->label == pObjNode->mnLabel) {
		//			pBB->mpNode = pObjNode;

		//			for (int j = 0, jend = pBB->vecMPs.size(); j < jend; j++) {
		//				auto pMPj = pBB->vecMPs[j];
		//				if (!pMPj || pMPj->isBad() || pMPj->mpObjectNode)
		//					continue;
		//				if (!pObjNode->mvpMPs.Count(pMPj)) {
		//					pObjNode->mvpMPs.Update(pMPj);
		//					pMPj->mpObjectNode = pObjNode;
		//				}
		//			}
		//		}
		//	}
		//	//디스크립터 갱신
		//	auto spMPs = pObjNode->mvpMPs.Get();
		//	pObjNode->ClearDescriptor();
		//	for (auto iter = spMPs.begin(), iend = spMPs.end(); iter != iend; iter++) {
		//		auto pMPi = *iter;
		//		pObjNode->AddDescriptor(pMPi->GetDescriptor());
		//	}
		//	pObjNode->ComputeBow(SLAM->mpDBoWVoc);
		//	GraphKeyFrameObject.Update(pKF, spONs);
		//	std::cout << "OBJECT NODE UPDATE = " << spMPs.size() << " " << pObjNode->GetDescriptor().rows << std::endl;
		//	//객체->키프레임도 연결하기

		//	{
		//		std::map<int, cv::Mat> contentDatas;
		//		auto spMPs = pObjNode->mvpMPs.Get();
		//		for (auto iter = spMPs.begin(), iend = spMPs.end(); iter != iend; iter++) {
		//			auto pMPi = *iter;
		//			if (!pMPi || pMPi->isBad())
		//				continue;
		//			contentDatas[pMPi->mnId] = pMPi->GetWorldPos();
		//		}
		//		SLAM->TemporalDatas2.Update("objnode", contentDatas);
		//	}
		//}
		//
		//////바운딩 박스로부터 오브젝트 노드 생성
		//if (spONs.size() == 0 && spBBs.size() > 2 && vecMPs.size() > 50) {
		//	std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(10);
		//	std::set<EdgeSLAM::KeyFrame*> tempKFs = std::set<EdgeSLAM::KeyFrame*>(vpLocalKFs.begin(), vpLocalKFs.end());
		//	EdgeSLAM::ObjectNode* pObjNode = new EdgeSLAM::ObjectNode();
		//	pObjNode->mnLabel = (int)MovingObjectLabel::CHAIR;

		//	for (auto iter = spBBs.begin(), iend = spBBs.end(); iter != iend; iter++) {
		//		auto pBB = *iter;
		//		pBB->mpNode = pObjNode;
		//	}

		//	for (int i = 0; i < vecMPs.size(); i++) {
		//		auto pMPi = vecMPs[i];
		//		if (!pMPi || pMPi->isBad())
		//			continue;
		//		pMPi->mpObjectNode = pObjNode;
		//		pObjNode->AddDescriptor(pMPi->GetDescriptor());
		//	}
		//	pObjNode->ComputeBow(SLAM->mpDBoWVoc);

		//	GraphObjectKeyFrame.Update(pObjNode, tempKFs);
		//	std::set<EdgeSLAM::ObjectNode*> tempObjs;
		//	tempObjs.insert(pObjNode);
		//	for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
		//		auto pKFi = *iter;
		//		GraphKeyFrameObject.Update(pKFi, tempObjs);
		//	}
		//}

		////for (int i = 0; i < vnIndexes.size(); i++) {
		////	int idx = vnIndexes[i];
		////	auto pMPi = pKF->mvpMapPoints.get(idx);
		////	auto spTempBBs = pMPi->mspBBs.Get();
		////	for (auto iter = spTempBBs.begin(), iend = spTempBBs.end(); iter != iend; iter++) {
		////		auto pBB = *iter;
		////		if (!spBBs.count(pBB)) {
		////			spBBs.insert(pBB);
		////		}
		////	}
		////	///*if (SLAM->GraphMapPointAndBoundingBox.Count(pMPi)) {
		////	//	auto spTempBBs = SLAM->GraphMapPointAndBoundingBox.Get(pMPi);
		////	//	for(auto iter = spTempBBs.begin(), iend = spTempBBs.end(); iter != iend; iter++){
		////	//		auto pBB = *iter;
		////	//		if (!spBBs.count(pBB)) {
		////	//			spBBs.insert(pBB);
		////	//		}
		////	//	}
		////	//}
		////	//if (SLAM->GraphMapPointAndObjectNode.Count(pMPi)) {
		////	//	auto spTempONs = SLAM->GraphMapPointAndObjectNode.Get(pMPi);
		////	//	for (auto iter = spTempONs.begin(), iend = spTempONs.end(); iter != iend; iter++) {
		////	//		auto pON = *iter;
		////	//		if (!spONs.count(pON)) {
		////	//			spONs.insert(pON);
		////	//		}
		////	//	}
		////	//}*/
		////}

		//

		////객체 노드와 바운딩박스 연결
		//
		//

		//
		//std::cout << "Test test test = " << " == " << spBBs.size() << " " << spONs.size() << std::endl;

		//return;
		//////키프레임 노드로부터 연결 된 오브젝트 찾기
		//////객체 트래킹에서 연결 된 객체는 미리 알려주는것이 좋음.
		//std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(10);
		//std::set<EdgeSLAM::ObjectNode*> setObjectNodes;
		//for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
		//	auto pKFi = *iter;
		//	std::set<EdgeSLAM::ObjectNode*> setTempNodes;
		//	if (GraphKeyFrameObject.Count(pKFi)) {
		//		setTempNodes = GraphKeyFrameObject.Get(pKFi);
		//		for (auto jter = setTempNodes.begin(), jend = setTempNodes.end(); jter != jend; jter++) {
		//			auto pContent = *jter;
		//			if (!setObjectNodes.count(pContent))
		//				setObjectNodes.insert(pContent);
		//		}
		//	}
		//}

		//////생성 및 갱신
		////생성 조건 1) 연결 된 오브젝트가 한개도 없을 경우
		////          2) 바운딩박스가 오브젝트 후보군 중 하나도 연결되지 않을 경우
		////바운딩 박스와 오브젝트 노드 연결 구현이 필요함.
		//if (setObjectNodes.size() == 0) {
		//	
		//	std::cout << "Object Generation Test" << std::endl;
		//	std::set<EdgeSLAM::KeyFrame*> tempKFs = std::set<EdgeSLAM::KeyFrame*>(vpLocalKFs.begin(), vpLocalKFs.end());
		//	
		//	std::set<EdgeSLAM::ObjectNode*> tempObjs;
		//	for (auto iter = spNewBBs.begin(), iend = spNewBBs.end(); iter != iend; iter++) {
		//		auto pBBox = *iter;
		//		int label = pBBox->label;
		//		if (label != (int)MovingObjectLabel::CHAIR) {
		//			continue;
		//		}
		//		EdgeSLAM::ObjectNode* pObjNode = new EdgeSLAM::ObjectNode();
		//		GraphObjectKeyFrame.Update(pObjNode, tempKFs);
		//		tempObjs.insert(pObjNode);
		//		pObjNode->mnLabel = pBBox->label;
		//		for (int i = 0; i < pBBox->vecMPs.size(); i++) {
		//			auto pMPi = pBBox->vecMPs[i];
		//			pObjNode->AddDescriptor(pMPi->GetDescriptor());
		//		}
		//		pObjNode->ComputeBow(SLAM->mpDBoWVoc);
		//	}
		//	
		//	for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
		//		auto pKFi = *iter;
		//		GraphKeyFrameObject.Update(pKFi, tempObjs);
		//	}
		//}

	}
	void SemanticProcessor::ObjectPreprocessing(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		auto pKF = pUser->mpRefKF;
		if (!pKF)
			return;

	}

	void SemanticProcessor::ObjectMapping(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
	
		//일단 바운딩 박스와는 연관이 없긴한데 여기 시작할때쯤의 박스는 다 디스크립터가 있음.
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		if (!pUser->KeyFrames.Count(id))
			return;
		auto pKF = pUser->KeyFrames.Get(id);
		/*if (!pKF)
			return;*/
		std::set<EdgeSLAM::ObjectBoundingBox*> spNewBBs;
		if (!GraphKeyFrameObjectBB.Count(pKF)) {
			return;
		}
		spNewBBs = GraphKeyFrameObjectBB.Get(pKF);
		if (spNewBBs.size() == 0)
			return;

		/*cv::Mat encoded = pUser->ImageDatas.Get(id);
		cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);
		if (img.empty())
		{
			return;
		}*/

		auto pMap = pUser->mpMap;
		if (!pMap) {
			std::cout << "map???????" << std::endl << std::endl << std::endl << std::endl;
			return;
		}

		//객체 생성
		bool bObjectMapGeneration = false;
		if (!GraphKeyFrameObject.Count(pKF)) {
			bObjectMapGeneration = true;
		}

		////키프레임 박스 테스트
		std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(10);
		std::set<EdgeSLAM::ObjectBoundingBox*> setNeighObjectBBs;
		int nobj = 0;
		for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
			auto pKFi = *iter;
			std::set<EdgeSLAM::ObjectBoundingBox*> setTempBBs;
			if (GraphKeyFrameObjectBB.Count(pKFi)) {
				setTempBBs = GraphKeyFrameObjectBB.Get(pKFi);
				for (auto jter = setTempBBs.begin(), jend = setTempBBs.end(); jter != jend; jter++) {
					auto pContent = *jter;
					if (pContent->label != (int)MovingObjectLabel::CHAIR)
						continue;
					if (!setNeighObjectBBs.count(pContent)){
						setNeighObjectBBs.insert(pContent);
						if (pContent->mpNode)
							nobj++;
					}
				}
			}
		}
		if (setNeighObjectBBs.size() == 0)
			return;

		//매칭 관련 정보
		auto thMaxDesc = SLAM->mpFeatureTracker->max_descriptor_distance;
		auto thMinDesc = SLAM->mpFeatureTracker->min_descriptor_distance;
		
		//처리
		std::cout << "Mapping = " << spNewBBs.size() << " " << setNeighObjectBBs.size() << std::endl;
		for (auto oter = spNewBBs.begin(), oend = spNewBBs.end(); oter != oend; oter++) {
			auto pBBox = *oter;
			if (pBBox->label != (int)MovingObjectLabel::CHAIR)
				continue;
			//노드 생성
			if (nobj == 0) {
				pBBox->mpNode = new EdgeSLAM::ObjectNode();
			}

			std::set<EdgeSLAM::ObjectNode*> spNodes;
			//std::cout <<"ID = "<<pBBox->id << " || OBJ = " << vecStrObjectLabels[pBBox->label - 1] << std::endl;

			for (auto bter = setNeighObjectBBs.begin(), bend = setNeighObjectBBs.end(); bter != bend; bter++) {
				auto pTempBox = *bter;
				std::chrono::high_resolution_clock::time_point astart = std::chrono::high_resolution_clock::now();

				if (bObjectMapGeneration) {
					auto pKF1 = pBBox->mpKF;
					auto pKF2 = pTempBox->mpKF;
					if (pKF1 && pKF2) {
						EdgeSLAM::ObjectNode* pObjNode = nullptr;
						//CreateObjectMapPoint(pKF1, pKF2, pBBox, pTempBox, thMinDesc, thMaxDesc, pMap, pObjNode);
					}
				}
				else {
					std::vector<std::pair<int, int>> matches;
					int n = 0;// ObjectSearchPoints::SearchObject(pBBox->desc, pTempBox->desc, matches, thMaxDesc, thMinDesc, 0.8, false);

					if (n > 10) {
						if (pTempBox->mpNode) {
							pBBox->mpNode = pTempBox->mpNode;
						}
						else if (pBBox->mpNode) {
							pTempBox->mpNode = pBBox->mpNode;
						}
					}

					/*for (int i = 0; i < matches.size(); i++) {
						int idx = matches[i].first;
						auto pt = pBBox->mvKeys[idx].pt;
						cv::circle(img, pt, 5, SemanticColors[pBBox->label], -1);
					}*/

					std::stringstream ss;
					if (pBBox->mpNode)
						ss << " box node = " << pBBox->mpNode->mnId;
					if (pTempBox->mpNode)
						ss << " temp node = " << pTempBox->mpNode->mnId;


					std::chrono::high_resolution_clock::time_point aend = std::chrono::high_resolution_clock::now();
					auto du_a2 = std::chrono::duration_cast<std::chrono::milliseconds>(aend - astart).count();
					float t_test1 = du_a2 / 1000.0;
					std::cout << "test == id = " << pBBox->id << "," << pTempBox->id << " || match = " << n << " || " << vecStrObjectLabels[pBBox->label - 1] << ", " << vecStrObjectLabels[pTempBox->label - 1] << " = " << pBBox->desc.rows << "," << pTempBox->desc.rows << " " << du_a2 << std::endl;
					std::cout << "node test = " << ss.str() << std::endl;
				}
			}
		}

		//SLAM->VisualizeImage(pUser->mapName, img, 3);
		
	}
	void SemanticProcessor::ObjectTracking(EdgeSLAM::SLAM* SLAM, std::string user, int id, const cv::Mat& img) {
		
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		auto pRefKF = pUser->mpRefKF;
		if (!pRefKF){
			std::cout << "wo kf = " << id << std::endl;
			return;
		}

		//Object node set
		std::set<EdgeSLAM::ObjectNode*> spObjNodes;

		//neighbor kf
		auto vNeighKFs = pRefKF->GetBestCovisibilityKeyFrames(20);
		vNeighKFs.push_back(pRefKF);
		for (auto iter = vNeighKFs.begin(), iend = vNeighKFs.end(); iter != iend; iter++) {
			auto pKFi = *iter;
			std::set<EdgeSLAM::ObjectNode*> tempNodes;
			if (GraphKeyFrameObject.Count(pKFi)) {
				tempNodes = GraphKeyFrameObject.Get(pKFi);
				for (auto jter = tempNodes.begin(), jend = tempNodes.end(); jter != jend; jter++) {
					auto pNode = *jter;
					if (!spObjNodes.count(pNode)) {
						spObjNodes.insert(pNode);
					}
				}//jter
			}//iter
		}

		/*if (!GraphKeyFrameObject.Count(pRefKF)){
			std::cout << "wo object" << std::endl;
			return;
		}*/

		/*auto spObjNodes = GraphKeyFrameObject.Get(pRefKF);
		if (spObjNodes.size() == 0){
			std::cout << "?????" << std::endl;
			return;
		}
		std::cout << "ttt = 1" << std::endl;*/

		std::set<EdgeSLAM::ObjectBoundingBox*> spNewBBs;
		if (!GraphFrameObjectBB.Count(id)) {
			return;
		}
		spNewBBs = GraphFrameObjectBB.Get(id);

		//std::cout << "222222222222222222222222" << std::endl;

		//////키프레임 박스 테스트
		//std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pRefKF->GetBestCovisibilityKeyFrames(10);
		std::set<EdgeSLAM::ObjectBoundingBox*> setNeighObjectBBs;
		int nobj = 0;
		for (auto iter = vNeighKFs.begin(), iend = vNeighKFs.end(); iter != iend; iter++) {
			auto pKFi = *iter;
			std::set<EdgeSLAM::ObjectBoundingBox*> setTempBBs;
			if (GraphKeyFrameObjectBB.Count(pKFi)) {
				setTempBBs = GraphKeyFrameObjectBB.Get(pKFi);
				for (auto jter = setTempBBs.begin(), jend = setTempBBs.end(); jter != jend; jter++) {
					auto pContent = *jter;
					if (pContent->label != (int)MovingObjectLabel::CHAIR)
						continue;
					if (!setNeighObjectBBs.count(pContent)) {
						setNeighObjectBBs.insert(pContent);
						if (pContent->mpNode)
							nobj++;
					}
				}
			}
		}
		//if (setNeighObjectBBs.size() == 0)
		//	return;
		pUser->mnUsed++;
		cv::Mat P;
		if (!pUser->PoseDatas.Count(id)) {
			std::cout << "Not estimated camera pose with Image " << id << std::endl;
			pUser->mnUsed--;
			return;
		}
		P = pUser->PoseDatas.Get(id);
		if (P.empty()) {
			std::cout << "Localization Failed = " << id << std::endl;
			pUser->mnUsed--;
			return;
		}

		cv::Mat Rcw = P.rowRange(0, 3).colRange(0, 3);
		cv::Mat tcw = P.rowRange(0, 3).col(3);

		cv::Mat Rwc = Rcw.t();
		cv::Mat twc = -Rwc * tcw;

		auto mapName = pUser->mapName;
		pUser->mnUsed--;
		pUser->mnDebugSeg--;
		pUser->mnDebugLabel--;
		//bbox 얻기

		//평면
		Plane* Floor = nullptr;
		/*std::set<Plane*> Planes;
		for (auto iter = vNeighKFs.begin(), iend = vNeighKFs.end(); iter != iend; iter++) {
			auto pKFi = *iter;
			if (PlaneEstimator::mPlaneConnections.Count(pKFi)) {
				auto tempPlanes = PlaneEstimator::mPlaneConnections.Get(pKFi);
				for (auto tter = tempPlanes.begin(), tend = tempPlanes.end(); tter != tend; tter++) {
					auto plane = *tter;
					if (plane->type == PlaneType::FLOOR) {
						Floor = plane;
						break;
					}
				}
			}
		}
		*/
		if (PlaneEstimator::GlobalFloor)
		{
			Floor = PlaneEstimator::GlobalFloor;
		}

		auto thMaxDesc = SLAM->mpFeatureTracker->max_descriptor_distance;
		auto thMinDesc = SLAM->mpFeatureTracker->min_descriptor_distance;
		auto cam = pUser->mpCamera;
		 
		std::set<EdgeSLAM::ObjectNode*> spRefKeyFrameNodes;
		std::map<int, cv::Mat> ttt;

		cv::Mat k = cam->K;
		cv::Mat d = cam->D;
		cv::Mat Kdouble;
		k.convertTo(Kdouble, CV_64FC1);
		//d.convertTo(d, CV_64FC1);
		
		if (spObjNodes.size() == 0){
			//리로컬 때만
			std::cout << "without Object" << std::endl;
			return;
		}

		auto mapObjectCount = GlobalObjectCount.Get();
		//std::cout << "Object Tracking start " << " = " << setNeighObjectBBs.size() << " " << spNewBBs.size() << std::endl;
		for (auto nter = spNewBBs.begin(), nend = spNewBBs.end(); nter != nend; nter++) {
			auto pNewBox = *nter;
			cv::Point2f pt(pNewBox->rect.x+pNewBox->rect.width/2, pNewBox->rect.y + 20);
			
			//cv::putText(img, vecStrObjectLabels[pNewBox->label-1], pt, 2, 1.2, cv::Scalar(255, 255, 255));
			if (pNewBox->label != (int)MovingObjectLabel::CHAIR && pNewBox->label != (int)MovingObjectLabel::PERSON){
				std::cout << "Not chair =" << vecStrObjectLabels[pNewBox->label-1] <<" = "<<pNewBox->label <<","<< mapObjectCount[pNewBox->label]<< std::endl;
				cv::rectangle(img, pNewBox->rect, cv::Scalar(255, 255, 0), 2);
				continue;
			}
			std::chrono::high_resolution_clock::time_point astart = std::chrono::high_resolution_clock::now();
			
			
			
			/*for (auto iter = setNeighObjectBBs.begin(), iend = setNeighObjectBBs.end(); iter != iend; iter++) {
				auto pNeighBox = *iter;
				DynamicTrackingProcessor::MatchTest(pNewBox, pNeighBox, img.clone(),cv::Mat(),k);
			}*/
			cv::Mat P = cv::Mat::eye(4,4,CV_32FC1);
			//int nRes = DynamicTrackingProcessor::MatchTest(pNewBox, setNeighObjectBBs, img, Kdouble, P);
			auto pObject = *spObjNodes.begin();

			int oid = pObject->mnId;
			pUser->mnUsed++;
			EdgeSLAM::ObjectTrackingResult* pTracking = nullptr;
			if (pUser->mapObjectTrackingResult.Count(oid)) {
				pTracking = pUser->mapObjectTrackingResult.Get(oid);
			}
			else {
				pTracking = new EdgeSLAM::ObjectTrackingResult(pObject, pObject->mnLabel, user);
				pUser->mapObjectTrackingResult.Update(oid, pTracking);
			}
			pUser->mnUsed--;
			if (pTracking->mState == EdgeSLAM::ObjectTrackingState::Success){
				continue;
			}
			
			///tracking test
			/*pUser->mnUsed++;
			if (pUser->mapObjectTrackingResult.Count(pObject->mnId)) {
				auto pTrackRes = pUser->mapObjectTrackingResult.Get(pObject->mnId);
				if (pTrackRes->mState == EdgeSLAM::ObjectTrackingState::Success && pTrackRes->mpLastFrame) {
					DynamicTrackingProcessor::ObjectTracking(SLAM, mapName, pNewBox, pObject, pTrackRes, img, Kdouble, P);
				}
			}
			pUser->mnUsed--;*/
			///

			int nRes = DynamicTrackingProcessor::ObjectRelocalization(pNewBox, pObject, img, Kdouble, P);
			//DynamicTrackingProcessor::MatchTestByFrame(pNewBox->mpF, setNeighObjectBBs, img, Kdouble, P);
			
			//int nMatch = ObjectDynamicTracking(SLAM, user, id, pNewBox, setNeighObjectBBs, P, Kdouble, d);
			std::chrono::high_resolution_clock::time_point aend = std::chrono::high_resolution_clock::now();
			auto du_a2 = std::chrono::duration_cast<std::chrono::milliseconds>(aend - astart).count();
			float t_test1 = du_a2 / 1000.0;
			std::cout << "Dynamic Tracking Processing time= " << " " << t_test1 <<" "<< nRes <<" "<< spObjNodes .size()<<"==" << pObject->mnId << std::endl;
			
			EdgeSLAM::ObjectTrackingState state = EdgeSLAM::ObjectTrackingState::Failed;
			if (nRes > 10 ) {
				state = EdgeSLAM::ObjectTrackingState::Success;
				pTracking->mnLastSuccessFrameId = id;
				pTracking->Pose = P.clone();

				EdgeSLAM::ObjectTrackingFrame* pTrackFrame = new EdgeSLAM::ObjectTrackingFrame();
				for (int i = 0; i < pNewBox->N; i++) {
					auto pMPi = pNewBox->mvpMapPoints.get(i);
					if (!pMPi || pMPi->isBad())
						continue;
					auto pt = pNewBox->mvKeys[i].pt;
					pTrackFrame->frame = img.clone();
					pTrackFrame->mvImagePoints.push_back(pt);
					pTrackFrame->mvpMapPoints.push_back(pMPi);
				}
				if (pTracking->mpLastFrame)
					delete pTracking->mpLastFrame;
				pTracking->mpLastFrame = pTrackFrame;
			}
			pTracking->mState = state;
			pTracking->mnLastSuccessFrameId = id;

			//visualization
			{
				float Na = 0;
				cv::Mat avgPos = cv::Mat::zeros(3, 1, CV_32FC1);
				cv::Mat R2 = P.rowRange(0, 3).colRange(0, 3);
				cv::Mat t2 = P.rowRange(0, 3).col(3);
				for (int i = 0; i < pNewBox->N; i++) {

					auto pMPi = pNewBox->mvpMapPoints.get(i);
					if (!pMPi || pMPi->isBad())
						continue;

					cv::Mat Xw = pMPi->GetWorldPos();
					cv::Mat proj = k * (R2 * Xw + t2);
					float depth = proj.at<float>(2);
					cv::Point2f objPt(proj.at<float>(0) / depth, proj.at<float>(1) / depth);

					auto imgPt = pNewBox->mvKeys[i].pt;
					cv::circle(img, imgPt, 2, cv::Scalar(0, 0, 255), -1);
					cv::circle(img, objPt, 2, cv::Scalar(255, 0, 0), -1);

					/*proj = k * (R * Xw + t);
					depth = proj.at<float>(2);
					cv::Point2f objPt2 = cv::Point2f(proj.at<float>(0) / depth, proj.at<float>(1) / depth);
					cv::circle(img, objPt2, 3, cv::Scalar(0, 255, 0), -1);*/

					cv::line(img, imgPt, objPt, cv::Scalar(255, 255, 0), 5);
					avgPos += Xw;
					Na++;
				}
				auto pFrame = pNewBox->mpF;
				for (int i = 0; i < pFrame->N; i++) {
					auto pMPi = pFrame->mvpMapPoints[i];
					if (!pMPi || pMPi->isBad())
						continue;
					auto imgPt = pFrame->mvKeysUn[i].pt;
					cv::circle(img, imgPt, 2, cv::Scalar(0, 0, 255), -1);
				}
				{
					cv::Mat origin = cv::Mat::zeros(3, 1, CV_32FC1);
					//avgPos /= Na;
					cv::Mat OBJc = t2;
					cv::Mat proj = k * OBJc;
					float depth = proj.at<float>(2);
					cv::Point2f pt(proj.at<float>(0) / depth, proj.at<float>(1) / depth);
					cv::circle(img, pt, 80, cv::Scalar(255, 255, 0), 3);

					//world에서 표현
					cv::Mat OBJw = Rwc * OBJc + twc;
					std::map<int, cv::Mat> a;
					a[0] = OBJw;
					//SLAM->TemporalDatas2.Update("dynamic", a);
				}
			}

			//std::cout << "relocal test = " << nMatch << std::endl;
			continue;
			int max_match = 0;
			EdgeSLAM::ObjectBoundingBox* pMaxBox = nullptr;
			std::vector<std::pair<int, int>> vMaxMatchedIndices;
			std::set<EdgeSLAM::MapPoint*> setMapFounds;
			int nTotal = 0;
			float Na = 0;
			cv::Mat avgPos = cv::Mat::zeros(3, 1, CV_32FC1);
			cv::Mat P2 = cv::Mat::eye(4, 4, CV_32FC1);
			for (auto iter = setNeighObjectBBs.begin(), iend = setNeighObjectBBs.end(); iter != iend; iter++) {
				auto pNeighBox = *iter;
				
				std::vector<EdgeSLAM::MapPoint*> vpMapPointMatches;
				int nMatch = ObjectSearchPoints::SearchBoxByBoW(pNeighBox, pNewBox, vpMapPointMatches, 50, 0.8);
				
				std::vector<cv::Point2f> imagePoints;
				std::vector<cv::Point3f> objectPoints;
				for (int i = 0, iend = vpMapPointMatches.size(); i < iend; i++) {
					auto pMPi = vpMapPointMatches[i];
					if (!pMPi || pMPi->isBad())
						continue;
					if (Floor) {
						float dist = Floor->Distacne(pMPi->GetWorldPos());
						if (abs(dist) < 0.1)
							continue;
					}
					if (setMapFounds.count(pMPi)) {
						continue;
					}
					cv::Point2f pt = pNewBox->mvKeys[i].pt;
					cv::Mat Xo = pMPi->GetWorldPos();
					cv::Point3f pt2(Xo.at<float>(0), Xo.at<float>(1), Xo.at<float>(2));
					imagePoints.push_back(pt);
					objectPoints.push_back(pt2);
					pNewBox->mvpMapPoints.update(i, pMPi);
					setMapFounds.insert(pMPi);
					nTotal++;
				}
				
				cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
				cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
				cv::Mat R = cv::Mat::eye(3, 3, CV_32FC1);
				cv::Mat t = cv::Mat::zeros(3, 1, CV_32FC1);
				/*bool bPnP = cv::solvePnPRansac(objectPoints, imagePoints, k, d, rvec, tvec);
				cv::Rodrigues(rvec, R);
				R.convertTo(R, CV_32FC1);
				tvec.convertTo(t, CV_32FC1);*/

				/*for (int i = 0, iend = imagePoints.size(); i < iend; i++) {
					cv::Mat Xw(objectPoints[i], CV_32FC1);
					cv::Mat proj = k*(R * Xw + t);
					float depth = proj.at<float>(2);
					cv::Point2f pt(proj.at<float>(0) / depth, proj.at<float>(1) / depth);
					cv::line(img, pt, imagePoints[i], cv::Scalar(255, 255, 0), 1);
					cv::circle(img, pt, 5,cv::Scalar(0,0,255), -1);
					cv::circle(img, imagePoints[i], 5,cv::Scalar(255, 0, 0), -1);
				}*/

				
				R.copyTo(P.rowRange(0, 3).colRange(0, 3));
				t.copyTo(P.col(3).rowRange(0, 3));
				//ObjectPoseInitialization
				int nOpt = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P);
				
				for (int i = 0; i < pNewBox->N; i++) {

					auto pMPi = pNewBox->mvpMapPoints.get(i);
					if (!pMPi || pMPi->isBad())
						continue;
					if (pNewBox->mvbOutliers[i])
					{
						pNewBox->mvpMapPoints.update(i, nullptr);
						pNewBox->mvbOutliers[i] = false;
						if (pMPi && setMapFounds.count(pMPi)) {
							setMapFounds.erase(pMPi);
						}
						nTotal--;
					}
				}

				if (nOpt > 10)
					break;
				//std::cout << "Optimization test = " << nOpt << std::endl;
				if (max_match < nMatch){
					max_match = nMatch;
					pMaxBox = pNeighBox;
					//vMaxMatchedIndices = vMatchedIndices;
				}
			}
			
			std::cout << "Tracking = " << pNewBox->N << " " << nTotal <<" "<<max_match << std::endl;
			continue;

			if (nTotal > 10) {
				float Na = 0;
				cv::Mat avgPos = cv::Mat::zeros(3, 1, CV_32FC1);
				auto vecMPs = pNewBox->mvpMapPoints.get();
				for (int i = 0, iend = vecMPs.size(); i < iend; i++) {
					auto pMP = vecMPs[i];
					if (!pMP || pMP->isBad())
						continue;
					avgPos += pMP->GetWorldPos();
					Na++;
					cv::circle(img, pNewBox->mvKeys[i].pt, 3, cv::Scalar(255, 0, 0), -1);
				}
				avgPos /= Na;

				//solve pnp test
				std::vector<cv::Point2f> imagePoints;
				std::vector<cv::Point3f> objectPoints;

				std::map<int, cv::Mat> dynamicOBJs;

				for (int i = 0, iend = pNewBox->N; i < iend; i++) {
					//std::cout << "0" << std::endl;
					auto pMP = vecMPs[i];
					auto pt = pNewBox->mvKeys[i].pt;

					if (!pMP || pMP->isBad())
						continue;
					cv::Mat Xo = pMP->GetWorldPos();
					cv::Point3f pt2(Xo.at<float>(0), Xo.at<float>(1), Xo.at<float>(2));
					//std::cout << Xo.t() <<" "<<pt<<" "<<k<<" "<<d<< std::endl;
					imagePoints.push_back(pt);
					objectPoints.push_back(pt2);
					dynamicOBJs[i] = Xo;
				}
				{
					cv::Mat P = cv::Mat::eye(4, 4, CV_32FC1);
					int nOpt = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P);
				}
				cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
				cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
				bool bPnP = cv::solvePnPRansac(objectPoints, imagePoints, k, d, rvec, tvec);
				//cv::solvePnPRefineLM(objectPoints, imagePoints, k, d, rvec, tvec);
				if (bPnP) {
					//cv::Mat R, t;
					//cv::Rodrigues(rvec, R);
					//R.convertTo(R, CV_32FC1);
					//tvec.convertTo(t, CV_32FC1);
					//cv::Mat P = cv::Mat::eye(4, 4, CV_32FC1);
					//R.copyTo(P.rowRange(0, 3).colRange(0, 3));
					//t.copyTo(P.col(3).rowRange(0, 3));
					//int nOpt = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P);
					//if (nOpt > 10) {

					//	cv::Mat Rco = P.rowRange(0, 3).colRange(0, 3);
					//	cv::Mat tco = P.rowRange(0, 3).col(3);
					//	cv::Mat temp = Rco * avgPos + tco;
					//	temp = k * temp;
					//	float depth = temp.at<float>(2);
					//	cv::Point2f pt(temp.at<float>(0) / depth, temp.at<float>(1) / depth);
					//	cv::circle(img, pt, 50, cv::Scalar(255, 0, 255), 3);
					//	//SLAM->VisualizeImage(mapName, img, 3);
					//}
					{
						cv::Mat Rco, tco;
						cv::Rodrigues(rvec, Rco);
						Rco.convertTo(Rco, CV_32FC1);
						tvec.convertTo(tco, CV_32FC1);
						cv::Mat OBJc = Rco * avgPos + tco;
						cv::Mat proj = k * OBJc;
						float depth = proj.at<float>(2);
						cv::Point2f pt(proj.at<float>(0) / depth, proj.at<float>(1) / depth);
						cv::circle(img, pt, 80, cv::Scalar(255, 255, 0), 3);

						//world에서 표현
						cv::Mat OBJw = Rwc* OBJc + twc;
						std::map<int, cv::Mat> a;
						a[0] = OBJw;
						SLAM->TemporalDatas2.Update("dynamic", a);
						SLAM->TemporalDatas2.Update("dynamic2", dynamicOBJs);
					}
				}
				else
					std::cout << "Fail Object Initial Pose Estimation" << std::endl;
				//std::cout <<"PNP = "<<bPnP<< rvec << " " << tvec << " " << rvec.type() << std::endl;
				/*std::vector<cv::Point2f> projectedPoints;

				cv::projectPoints(objectPoints, rvec, tvec, k, d, projectedPoints);
				std::cout << "22222" << std::endl;
				for (int i = 0; i < projectedPoints.size(); i++) {
					auto pt = projectedPoints[i];
					auto pt2 = imagePoints[i];
					std::cout << pt << " " << pt2 << std::endl;
					cv::circle(img, pt, 5, cv::Scalar(255,0,255), -1);
				}*/

				continue;

				//tracking test
				cv::Mat Pose;
				int nOpt = ObjectOptimizer::ObjectPoseOptimization(pNewBox, Pose);
				if (nOpt > 10) {

					cv::Mat Rco = Pose.rowRange(0, 3).colRange(0, 3);
					cv::Mat tco = Pose.rowRange(0, 3).col(3);
					cv::Mat temp = Rco * avgPos + tco;
					temp = k * temp;
					float depth = temp.at<float>(2);
					cv::Point2f pt(temp.at<float>(0) / depth, temp.at<float>(1) / depth);
					cv::circle(img, pt, 50, cv::Scalar(255, 0, 255), 3);
					//SLAM->VisualizeImage(mapName, img, 3);
				}
			}
			else
				std::cout << "Not enough matching points" << std::endl;
			//if (max_match > 10) {
			//	float Na = 0;
			//	cv::Mat avgPos = cv::Mat::zeros(3, 1, CV_32FC1);
			//	/*auto vecMPs = pMaxBox->mvpMapPoints.get();
			//	for (int i = 0, iend = vecMPs.size(); i < iend; i++) {
			//		auto pMP = vecMPs[i];
			//		if (!pMP || pMP->isBad())
			//			continue;
			//	}*/

			//	for (int i = 0, iend = vMaxMatchedIndices.size(); i < iend; i++) {
			//		int idx1 = vMaxMatchedIndices[i].first;
			//		int idx2 = vMaxMatchedIndices[i].second;
			//		auto pMPi = pMaxBox->mvpMapPoints.get(idx1);
			//		if (!pMPi || pMPi->isBad())
			//			continue;
			//		cv::circle(img, pNewBox->mvKeys[idx2].pt, 3, cv::Scalar(255, 0, 0), -1);
			//		pNewBox->mvpMapPoints.update(idx2, pMPi);
			//		Na++;
			//		avgPos += pMPi->GetWorldPos();
			//	}
			//	avgPos /= Na;

			//	cv::Mat Pose;
			//	int nOpt = EdgeSLAM::Optimizer::ObjectPoseOptimization(pNewBox, Pose);
			//	if (nOpt > 10) {

			//		cv::Mat Rco = Pose.rowRange(0, 3).colRange(0, 3);
			//		cv::Mat tco = Pose.rowRange(0, 3).col(3);
			//		cv::Mat temp = Rco * avgPos + tco;
			//		temp = k * temp;
			//		float depth = temp.at<float>(2);
			//		cv::Point2f pt(temp.at<float>(0) / depth, temp.at<float>(1) / depth);
			//		cv::circle(img, pt, 50, cv::Scalar(255, 0, 255), 3);
			//		//SLAM->VisualizeImage(mapName, img, 3);
			//	}
			//}
			
			
			//std::cout << "processing time = " << t_test1 << " " << max_match << std::endl;

			//for (auto iter = spObjNodes.begin(), iend = spObjNodes.end(); iter != iend; iter++) {
			//	auto pObjNode = *iter;

			//	auto spBBs = pObjNode->mspBBs.Get();
			//	std::chrono::high_resolution_clock::time_point astart = std::chrono::high_resolution_clock::now();
			//	int max_match = 0;
			//	EdgeSLAM::KeyFrame* pMaxKF = nullptr;
			//	for (auto bter = spBBs.begin(), bend = spBBs.end(); bter != bend; bter++) {
			//		auto pOldBox = *bter;
			//		std::vector<std::pair<int, int>> vMatchedIndices;
			//		int nMatch = EdgeSLAM::SearchPoints::SearchObjectBoxAndBoxForTracking(pOldBox, pNewBox, vMatchedIndices, thMinDesc, 0.85);
			//		//std::cout << "box match test = " << nMatch << std::endl;
			//		if (max_match < nMatch){
			//			max_match = nMatch;
			//			pMaxKF = pOldBox->mpKF;
			//		}
			//	}
			//	//auto vNeighKFs = pMaxKF->GetBestCovisibilityKeyFrames(10);
			//	//for (auto kter = vNeighKFs.begin(), kend = vNeighKFs.end(); kter != kend; kter++) {
			//	//	//auto pKFBox = *kter;
			//	//	std::vector<std::pair<int, int>> vMatchedIndices;
			//	//	//int nMatch = EdgeSLAM::SearchPoints::SearchObjectBoxAndBoxForTracking(pKFBox, pNewBox, vMatchedIndices, thMinDesc, 0.85);
			//	//}

			//	std::chrono::high_resolution_clock::time_point aend = std::chrono::high_resolution_clock::now();
			//	auto du_a2 = std::chrono::duration_cast<std::chrono::milliseconds>(aend - astart).count();
			//	float t_test1 = du_a2 / 1000.0;
			//	
			//	////std::cout << "Tracking Matching start" << std::endl;
			//	std::vector<std::pair<int, int>> vMatchedIndices;
			//	int nMatch = EdgeSLAM::SearchPoints::SearchObjectNodeAndBox(pObjNode, pNewBox, vMatchedIndices, thMaxDesc, thMaxDesc, 0.85, false);
			//	std::cout << "processing time = " << t_test1 << " " << max_match <<", "<<nMatch << std::endl;
			//	
			//	auto mvpOPs = pObjNode->mspMPs.ConvertVector();
			//	for (int i = 0, iend = vMatchedIndices.size(); i < iend; i++) {
			//		int idx1 = vMatchedIndices[i].first;
			//		int idx2 = vMatchedIndices[i].second;
			//		auto pMP = mvpOPs[idx1];
			//		if (!pMP || pMP->isBad())
			//			continue;
			//		pNewBox->mvpObjectPoints.update(idx2, pMP);
			//	}

			//	int nOpt = 0;
			//	if (nMatch > 15) {
			//		nOpt = EdgeSLAM::Optimizer::ObjectPoseOptimization(pObjNode, pNewBox, vMatchedIndices);
			//		
			//		if (nOpt > 15) {
			//			
			//			cv::Mat objPose = pObjNode->GetObjectPose();
			//			cv::Mat tco = objPose.rowRange(0, 3).col(3);
			//			cv::Mat temp = k * tco;
			//			float depth = temp.at<float>(2);
			//			cv::Point2f pt(temp.at<float>(0) / depth, temp.at<float>(1) / depth);
			//			cv::circle(img, pt, 50, cv::Scalar(255, 0, 255),3);
			//			std::cout << "AAAA = " <<pt<<"    " << pObjNode->GetObjectPose() << std::endl << std::endl;
			//		}

			//		//std::vector<cv::Point2f> imagePoints;
			//		//std::vector<cv::Point3f> objectPoints;

			//		//for (int i = 0; i < vMatchedIndices.size(); i++) {
			//		//	//std::cout << "0" << std::endl;
			//		//	int idx1 = vMatchedIndices[i].first;
			//		//	int idx2 = vMatchedIndices[i].second;
			//		//	auto pt = pNewBox->mvKeys[idx2].pt;

			//		//	auto pOP = mvpOPs[idx1];
			//		//	if (pOP && !pOP->isBad()) {
			//		//		cv::Mat Xo = pOP->GetObjectPos();
			//		//		cv::Point3f pt2(Xo.at<float>(0), Xo.at<float>(1), Xo.at<float>(2));
			//		//		//std::cout << Xo.t() <<" "<<pt<<" "<<k<<" "<<d<< std::endl;
			//		//		imagePoints.push_back(pt);
			//		//		objectPoints.push_back(pt2);
			//		//	}
			//		//}
			//		//std::cout << "11111" << std::endl;
			//		//std::cout << k.type() << " " << d.type() << std::endl;
			//		//cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
			//		//cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
			//		//cv::solvePnP(objectPoints, imagePoints, k, d, rvec, tvec);
			//		//cv::solvePnPRefineLM(objectPoints, imagePoints, k, d, rvec, tvec);
			//		//std::vector<cv::Point2f> projectedPoints;
			//		//std::cout << "aaaaa "  << std::endl;
			//		//cv::projectPoints(objectPoints, rvec, tvec, k, d, projectedPoints);
			//		//std::cout << "22222" << std::endl;
			//		//for (int i = 0; i < projectedPoints.size(); i++) {
			//		//	auto pt = projectedPoints[i];
			//		//	auto pt2 = imagePoints[i];
			//		//	std::cout << pt << " " << pt2 << std::endl;
			//		//	cv::circle(img, pt, 5, cv::Scalar(255,0,255), -1);
			//		//}


			//	}

			//	//std::cout << "Tracking Matching test = " << vecStrObjectLabels[pNewBox->label - 1] <<" " << pObjNode->mnId << "  && " << pNewBox->id << " = " << pNewBox->N << " == " << pObjNode->mspMPs.Size() << " == " << nMatch <<", "<<nOpt << std::endl;
			//	//for (int i = 0; i < vMatchedIndices.size(); i++) {
			//	//	//std::cout << "0" << std::endl;
			//	//	int idx1 = vMatchedIndices[i].first;
			//	//	int idx2 = vMatchedIndices[i].second;
			//	//	if (pNewBox->mvbOutliers[idx2])
			//	//		continue;
			//	//	auto pt = pNewBox->mvKeys[idx2].pt;
			//	//	cv::circle(img, pt, 5, SemanticColors[pNewBox->label], -1);
			//	//	 
			//	//	auto pOP = mvpOPs[idx1];
			//	//	if (pOP && !pOP->isBad()) {
			//	//		ttt[pOP->mnId] = pOP->GetWorldPos();
			//	//	}
			//	//}
		
			//	//if (nMatch > 20 && !spRefKeyFrameNodes.count(pObjNode)) {
			//	//	spRefKeyFrameNodes.insert(pObjNode);
			//	//}

			//	//{
			//	//	SLAM->TemporalDatas2.Update("objnode", ttt);
			//	//}
			//}
		}
		GraphKeyFrameObject.Update(pRefKF, spObjNodes);

		SLAM->VisualizeImage(mapName, img, 3);
		 
		//EdgeSLAM::Frame frame(img, cam, id);

		//for (auto oter = spNewBBs.begin(), oend = spNewBBs.end(); oter != oend; oter++) {
		//	auto pBBox = *oter;
		//	if (pBBox->label != (int)MovingObjectLabel::CHAIR)
		//		continue;

		//	for (auto bter = setNeighObjectBBs.begin(), bend = setNeighObjectBBs.end(); bter != bend; bter++) {
		//		auto pTempBox = *bter;
		//		std::chrono::high_resolution_clock::time_point astart = std::chrono::high_resolution_clock::now();
		//		std::vector<std::pair<int, int>> matches;
		//		int n = EdgeSLAM::SearchPoints::SearchObject(pBBox->desc, pTempBox->desc, matches, thMaxDesc, thMinDesc, 0.8, false);
		//		
		//		for (int i = 0; i < matches.size(); i++) {
		//			int idx = matches[i].first;
		//			auto pt = pBBox->fmvKeys[idx].pt;
		//			cv::circle(img, pt, 5, SemanticColors[pBBox->label], -1);
		//		}

		//		std::stringstream ss;
		//		if (pBBox->mpNode)
		//			ss << " box node = " << pBBox->mpNode->mnId;
		//		if (pTempBox->mpNode)
		//			ss << " temp node = " << pTempBox->mpNode->mnId;

		//		std::chrono::high_resolution_clock::time_point aend = std::chrono::high_resolution_clock::now();
		//		auto du_a2 = std::chrono::duration_cast<std::chrono::milliseconds>(aend - astart).count();
		//		float t_test1 = du_a2 / 1000.0;
		//		//std::cout << "tracking test == id = " << pBBox->id << "," << pTempBox->id << " || match = " << n << " || " << vecStrObjectLabels[pBBox->label - 1] << ", " << vecStrObjectLabels[pTempBox->label - 1] << " = " << pBBox->desc.rows << "," << pTempBox->desc.rows << " " << du_a2 << std::endl;
		//		//std::cout << "tracking node test = " << ss.str() << std::endl;
		//	}
		//}
		
	}

	

	//추후에는 오브젝트 디텍션만 하는 용도로. 여기의 코드가 객체 갱신으로 옮기기
	void SemanticProcessor::ObjectDetection(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		auto bMapping = pUser->mbMapping;
		
		std::stringstream ss;
		ss << "/Load?keyword=ObjectDetection" << "&id=" << id << "&src=" << user;
		WebAPI API("143.248.6.143", 35005);
		auto res = API.Send(ss.str(), "");

		int n2 = res.size();
		int n = n2 / 24;

		cv::Mat data = cv::Mat::zeros(n, 6, CV_32FC1);
		std::memcpy(data.data, res.data(), res.size());

		std::set<EdgeSLAM::ObjectBoundingBox*> spBBoxes;

		auto mapObjectCount = GlobalObjectCount.Get();
		if (bMapping) {
			//매핑용
			auto pKF = pUser->KeyFrames.Get(id);
			if (!pKF) {
				return;
			}
			
			for (int j = 0; j < n; j++) {
				int label = (int)data.at<float>(j, 0)+1;
				bool bC1 = ObjectWhiteList.Count(label);
				bool bC2 = ObjectCandidateList.Count(label);
				
				if (!bC1 && !bC2)
					continue;
				if(!bC1 && bC2){
					label = ObjectCandidateList.Get(label);
				}
				float conf = data.at<float>(j, 1);

				cv::Point2f left(data.at<float>(j, 2), data.at<float>(j, 3));
				cv::Point2f right(data.at<float>(j, 4), data.at<float>(j, 5));

				//사람이 0이기 때문에 +1을 함.
				auto pBBox = new EdgeSLAM::ObjectBoundingBox(pKF, label, conf, left, right);
				spBBoxes.insert(pBBox);
				mapObjectCount[pBBox->label]++;

			}

			//merge

			if (n > 0) {
				GraphKeyFrameObjectBB.Update(pKF, spBBoxes);
			}
		}
		else {
			//트래킹용
			cv::Mat img = cv::Mat();
			/*if (pUser->ImageDatas.Count(id)) {
				cv::Mat encoded = pUser->ImageDatas.Get(id);
				img = cv::imdecode(encoded, cv::IMREAD_COLOR);
			}*/
			if (img.empty())
			{
				//std::cout << "return wo image" << std::endl;
				//return;

				WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
				std::stringstream ss;
				ss << "/Load?keyword=Image" << "&id=" << id << "&src=" << user;
				auto res = mpAPI->Send(ss.str(), "");
				int n2 = res.size();
				cv::Mat temp = cv::Mat(n2, 1, CV_8UC1, (void*)res.data());
				img = cv::imdecode(temp, cv::IMREAD_COLOR);
			}
			
			auto cam = pUser->mpCamera;
			EdgeSLAM::Frame* frame = new EdgeSLAM::Frame(img, cam, id);
			frame->reset_map_points();
			for (int j = 0; j < n; j++) {
				int label = (int)data.at<float>(j, 0)+1;

				bool bC1 = ObjectWhiteList.Count(label);
				bool bC2 = ObjectCandidateList.Count(label);

				if (!bC1 && !bC2)
					continue;
				if (!bC1 && bC2) {
					label = ObjectCandidateList.Get(label);
				}

				float conf = data.at<float>(j, 1);

				cv::Point2f left(data.at<float>(j, 2), data.at<float>(j, 3));
				cv::Point2f right(data.at<float>(j, 4), data.at<float>(j, 5));

				//사람이 0이기 때문에 +1을 함.
				auto pBBox = new EdgeSLAM::ObjectBoundingBox(frame, label, conf, left, right);
				spBBoxes.insert(pBBox);
				mapObjectCount[pBBox->label]++;
			}
			
			for (auto oter = spBBoxes.begin(), oend = spBBoxes.end(); oter != oend; oter++) {
				auto pBBox = *oter;
				for (int k = 0, kend = frame->N; k < kend; k++) {
					auto pt = frame->mvKeys[k].pt;
					if (!pBBox->rect.contains(pt))
						continue;
					
					cv::Mat row = frame->mDescriptors.row(k);
					pBBox->mvIDXs.push_back(k);
					pBBox->mapIDXs[k] = pBBox->mvKeys.size();
					pBBox->mvKeys.push_back(frame->mvKeysUn[k]);
					pBBox->mvbOutliers.push_back(false);
					pBBox->mvpObjectPoints.push_back(nullptr);
					pBBox->mvpMapPoints.push_back(nullptr);
					pBBox->desc.push_back(row.clone());
				}
				pBBox->N = pBBox->desc.rows;
				pBBox->ComputeBow(SLAM->mpDBoWVoc);
			}
			if (n > 0) {
				GraphFrameObjectBB.Update(id, spBBoxes);
				ObjectTracking(SLAM, user, id, img);
				//SLAM->pool->EnqueueJob(SemanticProcessor::ObjectTracking, SLAM, user, id, img);
			}
		}
		///object detection count test
		for (auto iter = mapObjectCount.begin(), iend = mapObjectCount.end(); iter != iend; iter++) {
			auto label = iter->first;
			auto count = iter->second;
			/*if (count > 10)
			{
				std::cout << "object = " << vecStrObjectLabels[label - 1] << " " << label << " " << count << std::endl;
			}*/
			GlobalObjectCount.Update(label, count);
		}
		////바운딩 박스 & 마스킹
		return;
	}

	//오브젝트와 바운딩 박스의 연결 과정.
	//슬램 좌표계에서 오브젝트와 트래킹
	//트래킹이 안되면 오브젝트가 움직인다는 뜻
	//트래킹이 되면 오브젝트가 움직이지 않는 상태 - 오브젝트 매핑을 수행함.
	void SemanticProcessor::CheckDynamicObject(EdgeSLAM::SLAM* SLAM, std::string user, int id) {

		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;

		if (!pUser->KeyFrames.Count(id))
			return;
		auto pKF = pUser->KeyFrames.Get(id);
		/*if (!pKF)
			return;*/
		std::set<EdgeSLAM::ObjectBoundingBox*> spNewBBs;
		if (!GraphKeyFrameObjectBB.Count(pKF)) {
			return;
		}
		spNewBBs = GraphKeyFrameObjectBB.Get(pKF);
		if (spNewBBs.size() == 0)
			return;
		////키프레임 박스 테스트
		std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(20);
		std::set<EdgeSLAM::ObjectBoundingBox*> setNeighObjectBBs;
		std::set<EdgeSLAM::ObjectNode*> spNodes;
		int nobj = 0;
		for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
			auto pKFi = *iter;
			std::set<EdgeSLAM::ObjectBoundingBox*> setTempBBs;
			if (GraphKeyFrameObjectBB.Count(pKFi)) {
				setTempBBs = GraphKeyFrameObjectBB.Get(pKFi);
				for (auto jter = setTempBBs.begin(), jend = setTempBBs.end(); jter != jend; jter++) {
					auto pContent = *jter;
					if (pContent->label != (int)MovingObjectLabel::CHAIR)
						continue;
					if (!setNeighObjectBBs.count(pContent)) {
						setNeighObjectBBs.insert(pContent);
						/*if (pContent->mpNode)
							nobj++;*/
					}
				}
			}
			std::set<EdgeSLAM::ObjectNode*> setTempNodes;
			if (GraphKeyFrameObject.Count(pKFi)) {
				setTempNodes = GraphKeyFrameObject.Get(pKFi);
				for (auto jter = setTempNodes.begin(), jend = setTempNodes.end(); jter != jend; jter++) {
					auto pObjNode = *jter;
					if (!spNodes.count(pObjNode)) {
						spNodes.insert(pObjNode);
						/*if (pContent->mpNode)
							nobj++;*/
					}
				}
			}
		}
		for (auto mit = setNeighObjectBBs.begin(), mend = setNeighObjectBBs.end(); mit != mend; mit++) {
			auto pBox = *mit;
			auto pObj = pBox->mpNode;
			if (!pObj)
				continue;
			if (!spNodes.count(pObj)) {
				spNodes.insert(pObj);
			}
		}
		if (setNeighObjectBBs.size() == 0)
			return;
		pUser->mnUsed++;
		int w = pUser->mpCamera->mnWidth;
		int h = pUser->mpCamera->mnHeight;
		auto pMap = pUser->mpMap;
		pUser->mnUsed--;

		std::cout << "Object Matching Test " << spNodes.size() << "=" << spNewBBs.size() << std::endl;

		if (spNodes.size() > 0) {
			//spNodes = GraphKeyFrameObject.Get(pKF);
			//객체와 오브젝트 매칭
			// 
			//트래킹과 오브젝트 체크
			//맵포인트 추가 : 박스와 박스 매칭
			//최적화
			
			auto thMaxDesc = SLAM->mpFeatureTracker->max_descriptor_distance;
			auto thMinDesc = SLAM->mpFeatureTracker->min_descriptor_distance;
			auto pObject = *spNodes.begin();
			for (auto nter = spNewBBs.begin(), nend = spNewBBs.end(); nter != nend; nter++) {
				auto pNewBox = *nter;
				if (pNewBox->mvKeys.size() < 20)
					continue;
				pNewBox->UpdateConnections(pObject);
				//연결 되었으면 생성하도록 변경하기
				std::cout << "Visibility = " << pObject->mspBBs .Size()<<" " << pNewBox->GetBestCovisibilityBoxes(20).size() << std::endl;
			}
			////추가 매핑
			//for (auto nter = spNewBBs.begin(), nend = spNewBBs.end(); nter != nend; nter++) {
			//	auto pNewBox = *nter;
			//	if (pNewBox->mvKeys.size() < 20)
			//		continue;
			//	auto pKF1 = pNewBox->mpKF;
			//	
			//	for (auto iter = setNeighObjectBBs.begin(), iend = setNeighObjectBBs.end(); iter != iend; iter++) {
			//		auto pNeighBox = *iter;

			//		////F매칭 테스트
			//		//std::vector<std::pair<int, int>> vMatchedIndices;
			//		//int nMatch = ObjectSearchPoints::SearchObjectBoxAndBoxForTracking(pNeighBox, pNewBox, vMatchedIndices, thMaxDesc, 0.9);
			//		//std::vector<cv::Point2f> pts1, pts2;
			//		//std::vector<uchar> inliers;
			//		//cv::Mat R, t, TempMap;
			//		//for (int i = 0, iend = vMatchedIndices.size(); i < iend; i++) {
			//		//	int idx1 = vMatchedIndices[i].first;
			//		//	int idx2 = vMatchedIndices[i].second;

			//		//	auto pt1 = pNeighBox->mvKeys[idx1].pt;
			//		//	auto pt2 = pNeighBox->mvKeys[idx1].pt;

			//		//	pts1.push_back(pt1);
			//		//	pts2.push_back(pt2);

			//		//}
			//		//int nF = Utils::RecoverPose(pts1, pts2, inliers, pNewBox->K, R, t, TempMap);
			//		//std::cout << "F matching test111 = " << nF <<" "<<nMatch << std::endl;

			//		//삼각화 테스트
			//		//auto pNeighBox = *iter;
			//		auto pKF2 = pNeighBox->mpKF;
			//		if (pKF1 && pKF2 && (pKF1->mnId != pKF2->mnId)) {
			//			cv::Mat K = pKF1->K.clone();
			//			cv::Mat R1 = pKF1->GetRotation();
			//			cv::Mat t1 = pKF1->GetTranslation();
			//			cv::Mat R2 = pKF2->GetRotation();
			//			cv::Mat t2 = pKF2->GetTranslation();
			//			cv::Mat F12 = Utils::ComputeF12(R1, t1, R2, t2, K, K);
			//			std::vector<std::pair<int, int>> vMatchedIndices;
			//			int nMatch = ObjectSearchPoints::SearchObjectBoxAndBoxForTriangulation(pNeighBox, pNewBox, vMatchedIndices, F12, thMaxDesc, thMinDesc, 0.8, false);
			//			std::cout << "F matching test222 = " << nMatch << " " << pNewBox->N << " " << pNeighBox->N << std::endl;
			//		}
			//		
			//	}
			//	break;
			//	for (auto iter = spNodes.begin(), iend = spNodes.end(); iter != iend; iter++) {
			//		auto pObjNode = *iter;
			//		std::vector<std::pair<int, int>> vMatchedIndices;
			//		
			//		//매칭과 최적화가 필요함.

			//		int nMatch = ObjectSearchPoints::SearchObjectNodeAndBox(pObjNode, pNewBox, vMatchedIndices, thMaxDesc, thMaxDesc, 0.85, false);

			//		auto mvpMPs = pObjNode->mspMPs.ConvertVector();
			//		//auto spObjBBs = pObjNode->mspBBs.Get();
			//		int nTemp = 0;
			//		for (int i = 0, iend = vMatchedIndices.size(); i < iend; i++) {
			//			int idx1 = vMatchedIndices[i].first;
			//			int idx2 = vMatchedIndices[i].second;
			//			auto pMP = mvpMPs[idx1];
			//			if (!pMP || pMP->isBad())
			//				continue;
			//			pNewBox->mvpObjectPoints.update(idx2, pMP);
			//			pMP->AddObservation(pNewBox, idx2);
			//			pMP->ComputeDistinctiveDescriptors();
			//		}
			//		pNewBox->UpdateConnections();

			//		std::cout << "Matching test = " << mvpMPs.size() << " || " <<pNewBox->mvKeys.size()<<" "<<pNewBox->GetBestCovisibilityBoxes(20).size() << " || " << nMatch << " || " << nTemp << std::endl;
			//	}
			//}
			////추가 매핑
			

			////bow matching test
			cv::Mat matched_img = cv::Mat::zeros(2*h, w, CV_8UC3);
			cv::Rect upper(0, 0, w, h);
			cv::Rect lower(0, h,w,h);
			
			//오브젝트 맵포인트 컬링
			for (auto iter = spNodes.begin(), iend = spNodes.end(); iter != iend; iter++) {
				auto pObjNode = *iter;
				MapPointCulling(pObjNode, pKF->mnId);
			}
			//여기서 맵 포인트 다시 만들기
			//매칭 테스트
			for (auto nter = spNewBBs.begin(), nend = spNewBBs.end(); nter != nend; nter++) {
				auto pNewBox = *nter;
				if (pNewBox->mvKeys.size() < 20)
					continue;

				for (auto iter = spNodes.begin(), iend = spNodes.end(); iter != iend; iter++) {
					auto pObjNode = *iter;
					auto spObjBBs = pObjNode->mspBBs.Get();

					for (auto oter = spObjBBs.begin(), oend = spObjBBs.end(); oter != oend; oter++) {
						auto pTempBB = *oter;
						auto pTempKF = pTempBB->mpKF;
						if (pKF && pTempKF) {
							
							//return number of created mps
							CreateObjectMapPoint(pKF, pTempKF, pNewBox, pTempBB, thMinDesc, thMaxDesc, pMap, pObjNode);
							//일정값 이상일 때 추가
						}
					}
					pObjNode->UpdateOrigin();
					//pObjNode->UpdateObjectPos();
					//pObjNode->mspBBs.Update(pNewBox);
				}
				
			}
			/*for (auto iter = spNodes.begin(), iend = spNodes.end(); iter != iend; iter++) {
				auto pObjNode = *iter;
				ObjectOptimizer::ObjectMapAdjustment(pObjNode);
			}*/
			
			//퓨즈 및 최적화
		}
		else {
			//객체 생성
			//박스와 박스 매칭
			vpLocalKFs.push_back(pKF);
			std::cout << "Object Map Generation~~" << std::endl;
			ObjectMapGeneration(SLAM, vpLocalKFs, spNewBBs, setNeighObjectBBs, pMap);
			
			//최적화
		}

		//인접한 키프레임으로부터 오브젝트 맵 획득

		//인접한 키프레임으로부터 박수 집합 획득

		//바운딩 박스로부터 오브젝트 맵 획득
	}
	 
	void SemanticProcessor::MapPointCulling(EdgeSLAM::ObjectNode* map, unsigned long int nCurrentKFid)
	{
		// Check Recent Added MapPoints
		std::list<EdgeSLAM::ObjectMapPoint*>::iterator lit = map->mlpNewMPs.begin();
		//const unsigned long int nCurrentKFid = targetKF->mnId;

		const int cnThObs = 2;

		int N = 0;
		//while (lit != map->mlpNewMPs.end())
		//{
		//	EdgeSLAM::ObjectMapPoint* pMP = *lit;
		//	if (pMP->isBad())
		//	{
		//		lit = map->mlpNewMPs.erase(lit);
		//	}
		//	else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs)
		//	{
		//		pMP->SetBadFlag();
		//		lit = map->mlpNewMPs.erase(lit);
		//		//N++;
		//	}
		//	else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
		//		lit = map->mlpNewMPs.erase(lit);
		//	else
		//		lit++;
		//}

		/*int N2 = 0;
		auto mvpMPs = map->mspMPs.ConvertVector();
		for (int i = 0; i < mvpMPs.size(); i++) {
			if (mvpMPs[i]->Observations() > 2)
				N2++;
		}
		std::cout << "Object Map Test = " << mvpMPs.size() << " " << N2 << " = " << N << std::endl;*/
	}

	//이코드도 나중에 수정하기
	//박스 생성했을 때 해당 키프레임과 연결 된 오브젝트 맵이 하나도 없을 때
	void SemanticProcessor::ObjectMapGeneration(EdgeSLAM::SLAM* SLAM, std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs, std::set<EdgeSLAM::ObjectBoundingBox*> spNewBBs, std::set<EdgeSLAM::ObjectBoundingBox*> setNeighObjectBBs, EdgeSLAM::Map* MAP){
	
		//매칭 관련 정보
		auto thMaxDesc = SLAM->mpFeatureTracker->max_descriptor_distance;
		auto thMinDesc = SLAM->mpFeatureTracker->min_descriptor_distance;

		std::set<EdgeSLAM::ObjectNode*> spNodes;

		//처리
		for (auto oter = spNewBBs.begin(), oend = spNewBBs.end(); oter != oend; oter++) {
			auto pBBox = *oter;
			if (pBBox->label != (int)MovingObjectLabel::CHAIR)
				continue;
			auto pNewObjectMap = new EdgeSLAM::ObjectNode();
			pNewObjectMap->mspBBs.Update(pBBox);
			for (auto bter = setNeighObjectBBs.begin(), bend = setNeighObjectBBs.end(); bter != bend; bter++) {
				auto pTempBox = *bter;
				std::chrono::high_resolution_clock::time_point astart = std::chrono::high_resolution_clock::now();

				auto pKF1 = pBBox->mpKF;
				auto pKF2 = pTempBox->mpKF;
				if (pKF1 && pKF2) {
					CreateObjectMapPoint(pKF1, pKF2, pBBox, pTempBox, thMinDesc, thMaxDesc, MAP, pNewObjectMap);
					//일정값 이상일 때 추가
					pNewObjectMap->mspBBs.Update(pTempBox);
				}
			}
			pNewObjectMap->UpdateOrigin();
			//pNewObjectMap->UpdateObjectPos();
			spNodes.insert(pNewObjectMap);
			SLAM->GlobalObjectMap.Update(pNewObjectMap);
			//ObjectOptimizer::ObjectMapAdjustment(pNewObjectMap);
			pBBox->UpdateConnections();
		}
		for (auto kter = vpLocalKFs.begin(), kend = vpLocalKFs.end(); kter != kend; kter++) {
			auto pKFi = *kter;
			GraphKeyFrameObject.Update(pKFi, spNodes);
		}
		
	}

	//void SemanticProcessor::ObjectMapGeneration(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
	//	//일단 바운딩 박스와는 연관이 없긴한데 여기 시작할때쯤의 박스는 다 디스크립터가 있음.
	//	auto pUser = SLAM->GetUser(user);
	//	if (!pUser)
	//		return;
	//	if (!pUser->KeyFrames.Count(id))
	//		return;
	//	auto pKF = pUser->KeyFrames.Get(id);
	//	/*if (!pKF)
	//		return;*/
	//	std::set<EdgeSLAM::ObjectBoundingBox*> spNewBBs;
	//	if (!GraphKeyFrameObjectBB.Count(pKF)) {
	//		return;
	//	}
	//	spNewBBs = GraphKeyFrameObjectBB.Get(pKF);
	//	if (spNewBBs.size() == 0)
	//		return;
	//	auto pMap = pUser->mpMap;
	//	if (!pMap) {
	//		std::cout << "map???????" << std::endl << std::endl << std::endl << std::endl;
	//		return;
	//	}

	//	////키프레임 박스 테스트
	//	std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(10);
	//	std::set<EdgeSLAM::ObjectBoundingBox*> setNeighObjectBBs;
	//	int nobj = 0;
	//	for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
	//		auto pKFi = *iter;
	//		std::set<EdgeSLAM::ObjectBoundingBox*> setTempBBs;
	//		if (GraphKeyFrameObjectBB.Count(pKFi)) {
	//			setTempBBs = GraphKeyFrameObjectBB.Get(pKFi);
	//			for (auto jter = setTempBBs.begin(), jend = setTempBBs.end(); jter != jend; jter++) {
	//				auto pContent = *jter;
	//				if (pContent->label != (int)MovingObjectLabel::CHAIR)
	//					continue;
	//				if (!setNeighObjectBBs.count(pContent)) {
	//					setNeighObjectBBs.insert(pContent);
	//					/*if (pContent->mpNode)
	//						nobj++;*/
	//				}
	//			}
	//		}
	//	}
	//	if (setNeighObjectBBs.size() == 0)
	//		return;

	//	//매칭 관련 정보
	//	auto thMaxDesc = SLAM->mpFeatureTracker->max_descriptor_distance;
	//	auto thMinDesc = SLAM->mpFeatureTracker->min_descriptor_distance;

	//	std::set<EdgeSLAM::ObjectNode*> spNodes;

	//	//처리
	//	for (auto oter = spNewBBs.begin(), oend = spNewBBs.end(); oter != oend; oter++) {
	//		auto pBBox = *oter;
	//		if (pBBox->label != (int)MovingObjectLabel::CHAIR)
	//			continue;
	//		auto pNewObjectMap = new EdgeSLAM::ObjectNode();

	//		for (auto bter = setNeighObjectBBs.begin(), bend = setNeighObjectBBs.end(); bter != bend; bter++) {
	//			auto pTempBox = *bter;
	//			std::chrono::high_resolution_clock::time_point astart = std::chrono::high_resolution_clock::now();

	//			auto pKF1 = pBBox->mpKF;
	//			auto pKF2 = pTempBox->mpKF;
	//			if (pKF1 && pKF2) {
	//				CreateObjectMapPoint(pKF1, pKF2, pBBox, pTempBox, thMinDesc, thMaxDesc, pMap, pBBox->mpNode);
	//			}
	//		}

	//		spNodes.insert(pNewObjectMap);
	//	}
	//	GraphKeyFrameObject.Update(pKF, spNodes);
	//	//SLAM->VisualizeImage(pUser->mapName, img, 3);
	//}

	void SemanticProcessor::CreateObjectMapPoint(EdgeSLAM::KeyFrame* pKF1, EdgeSLAM::KeyFrame* pKF2, EdgeSLAM::ObjectBoundingBox* pBB1, EdgeSLAM::ObjectBoundingBox* pBB2, float minThresh, float maxThresh, EdgeSLAM::Map* pMap, EdgeSLAM::ObjectNode* pObjMap) {
	
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		const float& fx1 = pKF1->fx;
		const float& fy1 = pKF1->fy;
		const float& cx1 = pKF1->cx;
		const float& cy1 = pKF1->cy;
		const float& invfx1 = pKF1->invfx;
		const float& invfy1 = pKF1->invfy;

		const float& fx2 = pKF2->fx;
		const float& fy2 = pKF2->fy;
		const float& cx2 = pKF2->cx;
		const float& cy2 = pKF2->cy;
		const float& invfx2 = pKF2->invfx;
		const float& invfy2 = pKF2->invfy;

		cv::Mat Rcw1 = pKF1->GetRotation();
		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat tcw1 = pKF1->GetTranslation();
		cv::Mat Tcw1(3, 4, CV_32F);
		Rcw1.copyTo(Tcw1.colRange(0, 3));
		tcw1.copyTo(Tcw1.col(3));

		cv::Mat Rcw2 = pKF2->GetRotation();
		cv::Mat Rwc2 = Rcw2.t();
		cv::Mat tcw2 = pKF2->GetTranslation();
		cv::Mat Tcw2(3, 4, CV_32F);
		Rcw2.copyTo(Tcw2.colRange(0, 3));
		tcw2.copyTo(Tcw2.col(3));

		cv::Mat Ow1 = pKF1->GetCameraCenter();
		cv::Mat Ow2 = pKF2->GetCameraCenter();

		cv::Mat K  = pKF1->K.clone();
		cv::Mat R1 = pKF1->GetRotation();
		cv::Mat t1 = pKF1->GetTranslation();
		cv::Mat R2 = pKF2->GetRotation();
		cv::Mat t2 = pKF2->GetTranslation();
		cv::Mat F12 = Utils::ComputeF12(R1, t1, R2, t2, K, K);

		// Triangulate each match
		/*std::vector<std::pair<size_t, size_t> > vMatchedIndices;
		int nMatch = SearchPoints::SearchForTriangulation(pRefKeyframe, pCurKeyframe, F12, vMatchedIndices);*/
		
		std::vector<std::pair<int, int>> vMatchedIndices;
		int nMatch = ObjectSearchPoints::SearchObjectBoxAndBoxForTriangulation(pBB1, pBB2, vMatchedIndices, F12, maxThresh, minThresh, 0.8, false);
		int nMap = 0;
		for (int ikp = 0; ikp < nMatch; ikp++)

		{
			const int& idx1 = vMatchedIndices[ikp].first;
			const int& idx2 = vMatchedIndices[ikp].second;

			const cv::KeyPoint& kp1 = pBB1->mvKeys[idx1];
			const cv::KeyPoint& kp2 = pBB2->mvKeys[idx2];

			// Check parallax between rays
			cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
			cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

			cv::Mat ray1 = Rwc1 * xn1;
			cv::Mat ray2 = Rwc2 * xn2;
			const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

			cv::Mat x3D;
			if (cosParallaxRays > 0 && cosParallaxRays < 0.9998)
			{
				// Linear Triangulation Method
				cv::Mat A(4, 4, CV_32F);
				A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
				A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
				A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
				A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

				cv::Mat w, u, vt;
				cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

				x3D = vt.row(3).t();

				if (x3D.at<float>(3) == 0)
					continue;

				// Euclidean coordinates
				x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

			}
			else
				continue; //No stereo and very low parallax

			cv::Mat x3Dt = x3D.t();

			//Check triangulation in front of cameras
			float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
			if (z1 <= 0)
				continue;

			float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
			if (z2 <= 0)
				continue;

			//Check reprojection error in first keyframe
			const float& sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];
			const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
			const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
			const float invz1 = 1.0 / z1;

			float u1 = fx1 * x1 * invz1 + cx1;
			float v1 = fy1 * y1 * invz1 + cy1;
			float errX1 = u1 - kp1.pt.x;
			float errY1 = v1 - kp1.pt.y;
			float err1 = errX1 * errX1 + errY1 * errY1;
			/*if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
				continue;*/

				//Check reprojection error in second keyframe
			const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
			const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
			const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
			const float invz2 = 1.0 / z2;
			float u2 = fx2 * x2 * invz2 + cx2;
			float v2 = fy2 * y2 * invz2 + cy2;
			float errX2 = u2 - kp2.pt.x;
			float errY2 = v2 - kp2.pt.y;
			float err2 = (errX2 * errX2 + errY2 * errY2);
			/*if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
				continue;*/
				//std::cout << err1 << " " << err2 << std::endl;

			if (err1 > 4.0 || err2 > 4.0)
				continue;

			//Check scale consistency
			cv::Mat normal1 = x3D - Ow1;
			float dist1 = cv::norm(normal1);

			cv::Mat normal2 = x3D - Ow2;
			float dist2 = cv::norm(normal2);

			if (dist1 == 0 || dist2 == 0)
				continue;

			//// Triangulation is succesfull
			EdgeSLAM::MapPoint* pMP = new EdgeSLAM::MapPoint(x3D, pKF2, pMap, ts);
			pMP->mnObjectID = 100;

			pBB1->AddMapPoint(pMP, idx1);
			pBB2->AddMapPoint(pMP, idx2);

			int kfidx1 = pBB1->mvIDXs[idx1];
			int kfidx2 = pBB2->mvIDXs[idx2];

			pKF1->AddMapPoint(pMP, kfidx1);
			pKF2->AddMapPoint(pMP, kfidx2);
			//pKF1->mvbOutliers[kfidx1] = false; //check error
			//pKF2->mvbOutliers[kfidx2] = false;
			pMP->AddObservation(pKF1, kfidx1);
			pMP->AddObservation(pKF2, kfidx2);
			pMP->ComputeDistinctiveDescriptors();
			pMP->UpdateNormalAndDepth();
			pMap->mlpNewMPs.push_back(pMP);
			pMap->AddMapPoint(pMP);
			pObjMap->mspMPs.Update(pMP);
			////Create Object Map Points
			//EdgeSLAM::ObjectMapPoint* pMP = new EdgeSLAM::ObjectMapPoint(x3D, pBB1, pKF2, pMap, ts);
			////((EdgeSLAM::MapPoint*)pMP)->mnObjectID = 200;
			//pBB1->mvpObjectPoints.update(idx1, pMP);
			//pBB2->mvpObjectPoints.update(idx2, pMP);
			//pMP->AddObservation(pBB1, idx1);
			//pMP->AddObservation(pBB2, idx2);
			//pMP->ComputeDistinctiveDescriptors();
			//pObjMap->mlpNewMPs.push_back(pMP);
			//pMP->mpObjectMap = pObjMap;
			//pObjMap->mspOPs.Update(pMP);CreateObjectMapPoint
			nMap++;
		}//for
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_frame = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_frame = du_frame / 1000.0;
		//std::cout << "MapPoint Test = " << nMap <<", "<<nMatch<<" ==" << t_frame << std::endl;
	}

	int SemanticProcessor::ObjectDynamicTracking(EdgeSLAM::SLAM* SLAM, std::string user, int id, EdgeSLAM::ObjectBoundingBox* pNewBB, std::set<EdgeSLAM::ObjectBoundingBox*> setNeighObjectBBs, cv::Mat& P, const cv::Mat& K, const cv::Mat& D) {

		const int nBBs = setNeighObjectBBs.size();
		std::vector<std::vector<EdgeSLAM::MapPoint*> > vvpMapPointMatches;
		vvpMapPointMatches.resize(nBBs);

		std::vector<EdgeSLAM::ObjectBoundingBox*> vpCandidateBBs(setNeighObjectBBs.begin(), setNeighObjectBBs.end());

		std::vector<bool> vbDiscarded;
		vbDiscarded.resize(nBBs);

		int nCandidates = 0;

		auto thMaxDesc = SLAM->mpFeatureTracker->max_descriptor_distance;
		auto thMinDesc = SLAM->mpFeatureTracker->min_descriptor_distance;

		int nInitialMatchThresh = 5;
		int nBoxMatchThresh = 5;
		int nSuccessTracking = 15;

		for (int i = 0; i < nBBs; i++)
		{
			auto pBB = vpCandidateBBs[i];
			int nmatches = ObjectSearchPoints::SearchBoxByBoW(pBB, pNewBB, vvpMapPointMatches[i], 60, 0.75);
			if (nmatches < nInitialMatchThresh)
			{
				vbDiscarded[i] = true;
				continue;
			}
			else {
				nCandidates++;
			}
		}

		bool bMatch = false;
		int nGood = 0;
		while (nCandidates > 0 && !bMatch)
		{
			for (int i = 0; i < nBBs; i++)
			{
				if (vbDiscarded[i])
					continue;
				std::vector<bool> vbInliers;
				int nInliers;
				bool bNoMore;

				if (vvpMapPointMatches[i].size() < nBoxMatchThresh) {
					vbDiscarded[i] = true;
					nCandidates--;
					continue;
				}

				P = cv::Mat::eye(4, 4, CV_32FC1);
				std::set<EdgeSLAM::MapPoint*> sFound;

				for (size_t j = 0, jend = vvpMapPointMatches[i].size(); j < jend; j++)
				{
					auto pMP = vvpMapPointMatches[i][j];

					if (pMP && !pMP->isBad())
					{
						pNewBB->mvpMapPoints.update(j, pMP);
						sFound.insert(pMP);
					}
				}
				cv::Mat inliers;
				ObjectOptimizer::ObjectPoseInitialization(pNewBB, K, D, P, 1000,8.0,0.9,cv::SOLVEPNP_EPNP, inliers);
				nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBB, P);
				
				if (nGood < nBoxMatchThresh) {
					vbDiscarded[i] = true;
					nCandidates--;
					continue;
				}
				for (int io = 0; io < pNewBB->N; io++)
					if (pNewBB->mvbOutliers[io])
						pNewBB->mvpMapPoints.update(io, nullptr);

				if (nGood < nSuccessTracking)
				{
					int nadditional = ObjectSearchPoints::SearchBoxByProjection(pNewBB, vpCandidateBBs[i], sFound, P, 10, 100);
					//std::cout << "nadditional = " << nadditional << std::endl;
					if (nadditional + nGood >= nSuccessTracking)
					{
						nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBB, P);

						// If many inliers but still not enough, search by projection again in a narrower window
						// the camera has been already optimized with many points
						if (nGood > nBoxMatchThresh && nGood < nSuccessTracking)
						{
							sFound.clear();
							for (int ip = 0; ip < pNewBB->N; ip++){
								if (pNewBB->mvbOutliers[ip])
									continue;
								sFound.insert(pNewBB->mvpMapPoints.get(ip));
							}
							nadditional = ObjectSearchPoints::SearchBoxByProjection(pNewBB, vpCandidateBBs[i], sFound, P, 3, 64);
							// Final optimization
							if (nGood + nadditional >= nSuccessTracking)
							{
								nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBB, P);
								for (int io = 0; io < pNewBB->N; io++) {
									if (pNewBB->mvbOutliers[io])
									{
										pNewBB->mvpMapPoints.update(io, nullptr);
									}
										continue;
								}
							}
						}
					}
				}
				if (nGood >= nSuccessTracking)
				{
					bMatch = true;
					break;
				}
				else {
					vbDiscarded[i] = true;
					nCandidates--;
				}

				continue;
			}
		}
		return nGood;
	}

	void SemanticProcessor::DenseOpticalFlow(EdgeSLAM::SLAM* SLAM, std::string user, int id) {

		GridFrame* pG = new GridFrame();
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 10; j++) {
				auto pCell = new GridCell();
				pCell->AddObservation(pG, i * 10 + j);
				pG->mGrid[i][j] = pCell;
			}
		}
		GridFrame* pNewG = new GridFrame();
		pNewG->Copy(pG);
		delete pNewG;
		delete pG;

		auto pUser = SLAM->GetUser(user);
		if (!pUser->mbMapping)
			return;
		WebAPI API("143.248.6.143", 35005);
		int id1 = pUser->mnPrevFrameID;
		int id2 = pUser->mnCurrFrameID;

		if (id1 < 0 || id2 < 0)
			return;

		int scale = 16;
		cv::Mat img1, img2;
		cv::Mat gray1, gray2;
		cv::Mat P1, P2;
		cv::Mat R1, t1, R2, t2;
		std::vector<cv::KeyPoint> vecKP1, vecKP2;
		cv::Mat X3D1, X3D2;
		std::map<int, int> match1, match2;
		cv::Mat labeled1;
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		{
			cv::Mat rimg;
			std::stringstream ss;
			ss << "/Load?keyword=Image" << "&id=" << id1 << "&src=" << user;
			auto res = API.Send(ss.str(), "");
			int N = res.size();
			cv::Mat temp = cv::Mat::zeros(N, 1, CV_8UC1);
			std::memcpy(temp.data, res.data(), res.size());
			img1 = cv::imdecode(temp, cv::IMREAD_COLOR);
			cv::resize(img1, rimg, img1.size() / scale);
			cv::cvtColor(rimg, gray1, cv::COLOR_BGR2GRAY);//COLOR_BGR2GRAY
		}
		{
			cv::Mat rimg;
			std::stringstream ss;
			ss << "/Load?keyword=Image" << "&id=" << id2 << "&src=" << user;
			auto res = API.Send(ss.str(), "");
			int N = res.size();
			cv::Mat temp = cv::Mat::zeros(N, 1, CV_8UC1);
			std::memcpy(temp.data, res.data(), res.size());
			img2 = cv::imdecode(temp, cv::IMREAD_COLOR);
			cv::resize(img2, rimg, img2.size() / scale);
			cv::cvtColor(rimg, gray2, cv::COLOR_BGR2GRAY);//COLOR_BGR2GRAY
		}
		
		std::vector<cv::Rect> vecRects1;
		{
			std::stringstream ss;
			ss << "/Load?keyword=Segmentation" << "&id=" << id1 << "&src=" << user;
			auto res = API.Send(ss.str(), "");
			int N = res.size();
			/*
			cv::Mat data = cv::Mat::zeros(N/24, 6, CV_32FC1);
			std::memcpy(data.data, res.data(), res.size());

			for (int j = 0; j < data.rows; j++) {
				int label = (int)data.at<float>(j, 0);
				float conf = data.at<float>(j, 1);
				
				cv::Point2f left(data.at<float>(j, 2), data.at<float>(j, 3));
				cv::Point2f right(data.at<float>(j, 4), data.at<float>(j, 5));

				rectangle(img1, left, right, cv::Scalar(255, 255, 255));
				cv::Rect rect = cv::Rect(left, right);
				vecRects1.push_back(rect);
			}*/

			cv::Mat temp = cv::Mat::zeros(N, 1, CV_8UC1);
			std::memcpy(temp.data, res.data(), res.size());
			cv::Mat labeled = cv::imdecode(temp, cv::IMREAD_GRAYSCALE);
			if (labeled.rows == 0 || labeled.cols == 0)
				return;
			labeled1 = labeled.clone();
			int w = labeled.cols;
			int h = labeled.rows;

			//int oriw = pUser->mpCamera->mnWidth;
			//int orih = pUser->mpCamera->mnHeight;

			//float sw = ((float)w) / oriw; //scaled
			//float sh = ((float)h) / orih;

			/*cv::Mat segcolor = cv::Mat::zeros(h, w, CV_8UC3);
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					int label = labeled.at<uchar>(y, x) + 1;
					segcolor.at<cv::Vec3b>(y, x) = SemanticColors[label];
				}
			}*/

		}

		//{
		//	std::stringstream ss;
		//	ss << "/Load?keyword=ObjectDetection" << "&id=" << id2 << "&src=" << user;
		//	auto res = API.Send(ss.str(), "");
		//	int N = res.size();

		//	cv::Mat data = cv::Mat::zeros(N / 24, 6, CV_32FC1);
		//	std::memcpy(data.data, res.data(), res.size());

		//	for (int j = 0; j < data.rows; j++) {
		//		int label = (int)data.at<float>(j, 0);
		//		float conf = data.at<float>(j, 1);

		//		cv::Point2f left(data.at<float>(j, 2), data.at<float>(j, 3));
		//		cv::Point2f right(data.at<float>(j, 4), data.at<float>(j, 5));

		//		rectangle(img2, left, right, cv::Scalar(255, 255, 255));
		//		//cv::Rect rect = cv::Rect(left, right);
		//	}
		//}

		cv::Mat flow;
		cv::calcOpticalFlowFarneback(gray1, gray2, flow, 0.5, 3, 15, 3, 5, 1.1, 0);

		/*for (int r = 0; r < vecRects1.size(); r++) {
			cv::Rect rect = vecRects1[r];

			for (int x = rect.x, xend = rect.x+rect.width; x < xend; x += scale) {
				for (int y = rect.y, yend = rect.y + rect.height; y < yend; y += scale) {
					int nx = x / scale;
					int ny = y / scale;
					if (nx <= 0 || ny <= 0 || nx >= flow.cols || ny >= flow.rows )
						continue;

					float fx = flow.at<cv::Vec2f>(ny, nx).val[0] * scale;
					float fy = flow.at<cv::Vec2f>(ny, nx).val[1] * scale;

					cv::Point2f pt1(x, y);
					cv::Point2f pt2(x + fx, y + fy);
					cv::line(img1, pt1, pt2, cv::Scalar(255, 0, 0), 2);
					cv::circle(img2, pt2, 3, cv::Scalar(255, 0, 255), -1);
				}
			}
		}*/

		for (int x = 1; x < flow.cols; x++) {
			for (int y = 1; y < flow.rows; y++) {
				int nx = x * scale;
				int ny = y * scale;

				if (nx >= img1.cols || ny >= img1.rows)
					continue;

				float fx = flow.at<cv::Vec2f>(y, x).val[0]* scale;
				float fy = flow.at<cv::Vec2f>(y, x).val[1]* scale;

				cv::Point2f pt1(nx, ny);
				cv::Point2f pt2(nx + fx, ny + fy);
				cv::line(img1, pt1, pt2, cv::Scalar(255, 0, 0), 2);

				if (pt2.x <= 0 || pt2.y <= 0 || pt2.x >= img1.cols || pt2.y >= img1.rows)
					continue;
				int label = labeled1.at<uchar>(ny, nx) + 1;
				if (label <= 0)
				{
					continue;
				}
				cv::circle(img2, pt2, 5, SemanticColors[label], -1);
			}
		}

		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		
		/*SLAM->VisualizeImage(img1, 2);
		SLAM->VisualizeImage(img2, 3);*/
		//std::cout << "Desne = " << t_test1 << std::endl;

	}

	void SemanticProcessor::MultiViewStereo(EdgeSLAM::SLAM* SLAM, std::string user, int id)
	{
		auto pUser = SLAM->GetUser(user);
		if (!pUser->mbMapping)
			return;

		WebAPI API("143.248.6.143", 35005);
		int id1 = pUser->mnPrevFrameID;
		int id2 = pUser->mnCurrFrameID;

		cv::Mat img1, img2;
		cv::Mat P1, P2;
		cv::Mat R1, t1, R2, t2;
		std::vector<cv::KeyPoint> vecKP1, vecKP2;
		cv::Mat X3D1, X3D2;
		std::map<int, int> match1, match2;

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		{
			std::stringstream ss;
			ss << "/Load?keyword=ReferenceFrame" << "&id=" << id1 << "&src=" << user;
			auto res = API.Send(ss.str(), "");
			int N = res.size();

			cv::Mat data = cv::Mat::zeros(N / 4, 1, CV_32FC1);
			std::memcpy(data.data, res.data(), res.size());
			cv::Mat tempT = cv::Mat::eye(4, 4, CV_32FC1);
			int nIDX = 1;
			tempT.at<float>(0, 0) = data.at<float>(nIDX++);
			tempT.at<float>(0, 1) = data.at<float>(nIDX++);
			tempT.at<float>(0, 2) = data.at<float>(nIDX++);
			tempT.at<float>(1, 0) = data.at<float>(nIDX++);
			tempT.at<float>(1, 1) = data.at<float>(nIDX++);
			tempT.at<float>(1, 2) = data.at<float>(nIDX++);
			tempT.at<float>(2, 0) = data.at<float>(nIDX++);
			tempT.at<float>(2, 1) = data.at<float>(nIDX++);
			tempT.at<float>(2, 2) = data.at<float>(nIDX++);
			tempT.at<float>(0, 3) = data.at<float>(nIDX++);
			tempT.at<float>(1, 3) = data.at<float>(nIDX++);
			tempT.at<float>(2, 3) = data.at<float>(nIDX++);
						
			N = data.at<float>(0);
			std::vector<cv::KeyPoint> vecKP;
			cv::Mat X3D = cv::Mat::zeros(0, 3, CV_32FC1);
			for (int i = 0; i < N; i++) {
				cv::KeyPoint kp;
				kp.pt.x = data.at<float>(nIDX++);
				kp.pt.y = data.at<float>(nIDX++);
				kp.octave = (int)data.at<float>(nIDX++);
				kp.angle = data.at<float>(nIDX++);
				int id = (int)data.at<float>(nIDX++);
				float x = data.at<float>(nIDX++);
				float y = data.at<float>(nIDX++);
				float z = data.at<float>(nIDX++);

				vecKP.push_back(kp);
				cv::Mat X = (cv::Mat_<float>(1,3) << x, y, z);
				X3D.push_back(X.clone());
				match1.insert(std::make_pair(id, i));
			}
			P1 = tempT.clone();
			R1 = tempT.rowRange(0, 3).colRange(0, 3).clone();
			t1 = tempT.rowRange(0, 3).col(3).clone();
			vecKP1 = vecKP;
			X3D1 = X3D.clone();
		}

		{
			std::stringstream ss;
			ss << "/Load?keyword=ReferenceFrame" << "&id=" << id2 << "&src=" << user;
			auto res = API.Send(ss.str(), "");
			int N = res.size();

			cv::Mat data = cv::Mat::zeros(N / 4, 1, CV_32FC1);
			std::memcpy(data.data, res.data(), res.size());
			cv::Mat tempT = cv::Mat::eye(4, 4, CV_32FC1);
			int nIDX = 1;
			tempT.at<float>(0, 0) = data.at<float>(nIDX++);
			tempT.at<float>(0, 1) = data.at<float>(nIDX++);
			tempT.at<float>(0, 2) = data.at<float>(nIDX++);
			tempT.at<float>(1, 0) = data.at<float>(nIDX++);
			tempT.at<float>(1, 1) = data.at<float>(nIDX++);
			tempT.at<float>(1, 2) = data.at<float>(nIDX++);
			tempT.at<float>(2, 0) = data.at<float>(nIDX++);
			tempT.at<float>(2, 1) = data.at<float>(nIDX++);
			tempT.at<float>(2, 2) = data.at<float>(nIDX++);
			tempT.at<float>(0, 3) = data.at<float>(nIDX++);
			tempT.at<float>(1, 3) = data.at<float>(nIDX++);
			tempT.at<float>(2, 3) = data.at<float>(nIDX++);

			N = data.at<float>(0);
			std::vector<cv::KeyPoint> vecKP;
			cv::Mat X3D = cv::Mat::zeros(0, 3, CV_32FC1);
			for (int i = 0; i < N; i++) {
				cv::KeyPoint kp;
				kp.pt.x = data.at<float>(nIDX++);
				kp.pt.y = data.at<float>(nIDX++);
				kp.octave = (int)data.at<float>(nIDX++);
				kp.angle = data.at<float>(nIDX++);
				int id = (int)data.at<float>(nIDX++);
				float x = data.at<float>(nIDX++);
				float y = data.at<float>(nIDX++);
				float z = data.at<float>(nIDX++);

				vecKP.push_back(kp);
				cv::Mat X = (cv::Mat_<float>(1, 3) << x, y, z);
				X3D.push_back(X.clone());
				match2.insert(std::make_pair(id, i));
			}
			P2 = tempT.clone();
			R2 = tempT.rowRange(0, 3).colRange(0, 3).clone();
			t2 = tempT.rowRange(0, 3).col(3).clone();
			vecKP2 = vecKP;
			X3D2 = X3D.clone();
		}
		{
			std::stringstream ss;
			ss << "/Load?keyword=Image" << "&id=" << id1 << "&src=" << user;
			auto res = API.Send(ss.str(), "");
			int N = res.size();
			cv::Mat temp = cv::Mat::zeros(N, 1, CV_8UC1);
			std::memcpy(temp.data, res.data(), res.size());
			img1= cv::imdecode(temp, cv::IMREAD_COLOR);
		}
		{
			std::stringstream ss;
			ss << "/Load?keyword=Image" << "&id=" << id2 << "&src=" << user;
			auto res = API.Send(ss.str(), "");
			int N = res.size();
			cv::Mat temp = cv::Mat::zeros(N, 1, CV_8UC1);
			std::memcpy(temp.data, res.data(), res.size());
			img2 = cv::imdecode(temp, cv::IMREAD_COLOR);
		}

		std::vector<cv::Point2f> vec1, vec2;
		for (auto iter = match1.begin(), iend = match1.end(); iter != iend; iter++) {
			int id = iter->first;
			int idx1 = iter->second;
			if (match2.count(id)) {
				int idx2 = match2[id];
				vec1.push_back(vecKP1[idx1].pt);
				vec2.push_back(vecKP2[idx2].pt);
			}
		}
		std::cout << vecKP1.size() << " " << vec1.size() << std::endl;

		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		
		cv::Mat K = pUser->GetCameraMatrix();
		cv::Mat D = pUser->mpCamera->D.clone();
		cv::Mat F12 = Utils::ComputeF12(R1, t1, R2, t2,K, K);

		cv::Mat H1, H2;
		cv::Mat rectified1, rectified2;

		/*cv::Mat uimg1, uimg2;
		cv::undistort(img1, uimg1, K, D);
		cv::undistort(img2, uimg2, K, D);

		cv::Ptr<cv::StereoSGBM> ptr = cv::StereoSGBM::create(-128, 256, 11, 8 * 121, 32 * 121, 0, 0, 5, 200, 2);
		cv::Mat res;

		ptr->compute(uimg1, uimg2, res);

		cv::normalize(res, res, 255, 0, cv::NORM_MINMAX, CV_8UC1);
		cv::cvtColor(res, res, cv::COLOR_GRAY2BGR);
		SLAM->VisualizeImage(res, 3);*/

		bool bRectify = cv::stereoRectifyUncalibrated(vec1, vec2, F12, img1.size(), H1, H2);
		if (bRectify) {
			cv::warpPerspective(img1, rectified1, H1, img1.size());
			cv::warpPerspective(img1, rectified2, H2, img2.size());
			/*SLAM->VisualizeImage(rectified1, 2);
			SLAM->VisualizeImage(rectified2, 3);*/

			/*cv::Ptr<cv::StereoSGBM> ptr = cv::StereoSGBM::create(-128, 256, 11, 8 * 121, 32 * 121, 0, 0, 5, 200, 2);
			cv::Mat res;

			std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
			ptr->compute(rectified1, rectified2, res);

			std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
			auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			float t_test1 = du_test1 / 1000.0;
			std::cout << "SGBM = " << t_test1 << std::endl;
			cv::normalize(res, res, 255, 0, cv::NORM_MINMAX, CV_8UC1);
			cv::cvtColor(res, res, cv::COLOR_GRAY2BGR);
			SLAM->VisualizeImage(res, 3);*/

		}
	}
}


