#include <SemanticProcessor.h>
#include <random>
#include <Utils.h>
#include <User.h>
#include <Camera.h>
#include <KeyFrame.h>
#include <Frame.h>
#include <MapPoint.h>
#include <FeatureTracker.h>

#include <SemanticLabel.h>
#include <PlaneEstimator.h>
#include <GridProcessor.h>
#include <GridCell.h>
#include <ObjectFrame.h>
#include <SearchPoints.h>
#include <SLAM.h>

namespace SemanticSLAM {
	SemanticProcessor::SemanticProcessor() {}
	SemanticProcessor::~SemanticProcessor() {}

	std::string SemanticProcessor::strLabel = "wall,building,sky,floor,tree,ceiling,road,bed,windowpane,grass,cabinet,sidewalk,person,earth,door,table,mountain,plant,curtain,chair,car,water,painting,sofa,shelf,house,sea,mirror,rug,field,armchair,seat,fence,desk,rock,wardrobe,lamp,bathtub,railing,cushion,base,box,column,signboard,chest of drawers,counter,sand,sink,skyscraper,fireplace,refrigerator,grandstand,path,stairs,runway,case,pool table,pillow,screen door,stairway,river,bridge,bookcase,blind,coffee table,toilet,flower,book,hill,bench,countertop,stove,palm,kitchen island,computer,swivel chair,boat,bar,arcade machine,hovel,bus,towel,light,truck,tower,chandelier,awning,streetlight,booth,television,airplane,dirt track,apparel,pole,land,bannister,escalator,ottoman,bottle,buffet,poster,stage,van,ship,fountain,conveyer belt,canopy,washer,plaything,swimming pool,stool,barrel,basket,waterfall,tent,bag,minibike,cradle,oven,ball,food,step,tank,trade name,microwave,pot,animal,bicycle,lake,dishwasher,screen,blanket,sculpture,hood,sconce,vase,traffic light,tray,ashcan,fan,pier,crt screen,plate,monitor,bulletin board,shower,radiator,glass,clock,flag";
	std::string SemanticProcessor::strYoloObjectLabel = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush";
	
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

	void SemanticProcessor::Init() {
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

		vecStrSemanticLabels = Utils::Split(strLabel, ",");
		vecStrObjectLabels = Utils::Split(strYoloObjectLabel, ",");

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

		////객체 레이블 포인트 테스트
		//MP도 추가할 예정
		std::set<EdgeSLAM::ObjectBoundingBox*> spNewBBs;
		if (!GraphKeyFrameObjectBB.Count(pKF)) {
			return;
		}
		spNewBBs = GraphKeyFrameObjectBB.Get(pKF);
		std::set<int> testLabelID;
		for (auto oter = spNewBBs.begin(), oend = spNewBBs.end(); oter != oend; oter++) {
			auto pBBox = *oter;
			for (int k = 0, kend = pKF->N; k < kend; k++) {
				auto pt = pKF->mvKeys[k].pt;
				int label = labeled.at<uchar>(pt) + 1;
				if (!pBBox->rect.contains(pt))
					continue;
				if (label == (int)StructureLabel::FLOOR)
					continue;
				if (label == (int)StructureLabel::WALL)
					continue;
				if (label == (int)StructureLabel::CEIL)
					continue;
				testLabelID.insert(label);
				cv::Mat row = pKF->mDescriptors.row(k);
				//auto pMPk = pKF->mvpMapPoints.get(k);
				pBBox->mvIDXs.push_back(k);
				pBBox->mvKeys.push_back(pKF->mvKeys[k]);
				pBBox->mvbOutliers.push_back(false);
				pBBox->vecMPs.push_back(nullptr);
				pBBox->desc.push_back(row.clone());
			}
			//std::cout << "LabelMapPoint Label = " <<pBBox->label<<" "<< pBBox->desc.rows << std::endl;
		}
		
		SLAM->pool->EnqueueJob(SemanticProcessor::ObjectMapping, SLAM, user, id);
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
		if(pUser->GetVisID()==0)
			SLAM->VisualizeImage(pUser->mapName, segcolor, 1);
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
			auto pBB = *iter;
			cv::rectangle(img, pBB->rect, cv::Scalar(255, 255, 0), 2);
			for (int j = 0, jend = pBB->mvIDXs.size(); j < jend; j++) {
				int idx = pBB->mvIDXs[j];
				auto pMPj = pKF->mvpMapPoints.get(idx);

				cv::circle(img, pBB->mvKeys[j].pt, 3, cv::Scalar(255, 255, 0), 2);

				if (!pMPj || pMPj->isBad())
					continue;
				//if (pMPi->mnObjectID > 0){
				//	continue;
				//}
				
				SemanticLabel* pStaticLabel = nullptr;
				if (!SemanticLabels.Count(pMPj->mnId)) {
					continue;
				}
				pStaticLabel = SemanticLabels.Get(pMPj->mnId);
				if (pStaticLabel->LabelCount.Count((int)StructureLabel::FLOOR) && pStaticLabel->LabelCount.Get((int)StructureLabel::FLOOR) > seg_th) {
					continue;
				}
				if (pStaticLabel->LabelCount.Count((int)StructureLabel::WALL) && pStaticLabel->LabelCount.Get((int)StructureLabel::WALL) > seg_th) {
					continue;
				}
				if (pStaticLabel->LabelCount.Count((int)StructureLabel::CEIL) && pStaticLabel->LabelCount.Get((int)StructureLabel::CEIL) > seg_th) {
					continue;
				}
				pBB->vecMPs[j] = pMPj;
				ttt++;
			}
			for (int j = 0, jend = pBB->vecMPs.size(); j < jend; j++) {
				auto pMPj = pBB->vecMPs[j];
				if (pMPj && !pMPj->isBad()) {
					cv::circle(img, pBB->mvKeys[j].pt, 3, cv::Scalar(0, 0, 255), 2, -1);
				}
			}
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

		/*pUser->mnUsed++;
		pUser->mnDebugSeg++;
		pUser->mnUsed--;
		pUser->mnDebugSeg--;*/

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
				std::vector<std::pair<int, int>> matches;
				int n = EdgeSLAM::SearchPoints::SearchObject(pBBox->desc, pTempBox->desc, matches, thMaxDesc, thMinDesc, 0.8, false);

				if (n > 10) {
					if(pTempBox->mpNode){
						pBBox->mpNode = pTempBox->mpNode;
					}
					else if(pBBox->mpNode){
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

		//SLAM->VisualizeImage(pUser->mapName, img, 3);
		
	}
	void SemanticProcessor::ObjectTracking(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		/*auto pKF = pUser->mpRefKF;
		if (!pKF)
			return;*/
		if (!pUser->KeyFrames.Count(id))
			return;
		auto pKF = pUser->KeyFrames.Get(id);

		cv::Mat encoded = pUser->ImageDatas.Get(id);
		cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);
		if (img.empty())
		{
			return;
		}

		std::set<EdgeSLAM::ObjectBoundingBox*> spNewBBs;
		if (!GraphKeyFrameObjectBB.Count(pKF)) {
			return;
		}
		spNewBBs = GraphKeyFrameObjectBB.Get(pKF);

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
					if (!setNeighObjectBBs.count(pContent)) {
						setNeighObjectBBs.insert(pContent);
						if (pContent->mpNode)
							nobj++;
					}
				}
			}
		}
		if (setNeighObjectBBs.size() == 0)
			return;

		pUser->mnUsed++;
		pUser->mnDebugSeg++;

		auto mapName = pUser->mapName;

		pUser->mnUsed--;
		pUser->mnDebugSeg--;
		//bbox 얻기

		auto thMaxDesc = SLAM->mpFeatureTracker->max_descriptor_distance;
		auto thMinDesc = SLAM->mpFeatureTracker->min_descriptor_distance;
		auto cam = pUser->mpCamera;
		EdgeSLAM::Frame frame(img, cam, id);

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
				std::vector<std::pair<int, int>> matches;
				int n = EdgeSLAM::SearchPoints::SearchObject(pBBox->desc, pTempBox->desc, matches, thMaxDesc, thMinDesc, 0.8, false);
				
				for (int i = 0; i < matches.size(); i++) {
					int idx = matches[i].first;
					auto pt = pBBox->mvKeys[idx].pt;
					cv::circle(img, pt, 5, SemanticColors[pBBox->label], -1);
				}

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
		SLAM->VisualizeImage(pUser->mapName, img, 3);

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		int n = 0;

		if (spNewBBs.size() > 0) {

			for (auto oter = spNewBBs.begin(), oend = spNewBBs.end(); oter != oend; oter++) {
				auto pBBox = *oter;
				std::cout << "OBJ = " << vecStrObjectLabels[pBBox->label - 1] << std::endl;
			}

			auto pBBox = *spNewBBs.begin();
			std::cout << "Object tracking desc = " << pBBox->desc.rows << std::endl;
			
			
			std::vector<std::pair<int, int>> matches;
			n = EdgeSLAM::SearchPoints::SearchObject(pBBox->desc, frame.mDescriptors, matches, thMaxDesc, thMinDesc, 0.8, false);

			for (int i = 0; i < matches.size(); i++) {
				int idx = matches[i].second;
				auto pt = frame.mvKeys[idx].pt;
				cv::circle(img, pt, 5, SemanticColors[pBBox->label], -1);
			}
			SLAM->VisualizeImage(pUser->mapName, img, 3);

			////키프레임 박스 테스트
			std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(5);
			std::set<EdgeSLAM::ObjectBoundingBox*> setObjectBBs;
			for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
				auto pKFi = *iter;
				std::set<EdgeSLAM::ObjectBoundingBox*> setTempBBs;
				if (GraphKeyFrameObjectBB.Count(pKFi)) {
					setTempBBs = GraphKeyFrameObjectBB.Get(pKFi);
					for (auto jter = setTempBBs.begin(), jend = setTempBBs.end(); jter != jend; jter++) {
						auto pContent = *jter;
						if (!setObjectBBs.count(pContent))
							setObjectBBs.insert(pContent);
					}
				}
			}
			//BB 매칭 테스트
			std::chrono::high_resolution_clock::time_point bstart = std::chrono::high_resolution_clock::now();
			for (auto bter = setObjectBBs.begin(), bend = setObjectBBs.end(); bter != bend; bter++) {
				auto pTempBox = *bter;
				std::chrono::high_resolution_clock::time_point astart = std::chrono::high_resolution_clock::now();
				n = EdgeSLAM::SearchPoints::SearchObject(pBBox->desc, pTempBox->desc, matches, thMaxDesc, thMinDesc, 0.8, false);
				std::chrono::high_resolution_clock::time_point aend = std::chrono::high_resolution_clock::now();
				auto du_a2 = std::chrono::duration_cast<std::chrono::milliseconds>(aend - astart).count();
				float t_test1 = du_a2 / 1000.0;
				std::cout << "test == id = "<<pBBox->id<<","<< pTempBox->id << " || match = " << n << " || " << vecStrObjectLabels[pBBox->label - 1] <<", "<< vecStrObjectLabels[pTempBox->label - 1] << " = " << pBBox->desc.rows << "," << pTempBox->desc.rows << " " << du_a2 << std::endl;
			}
			std::chrono::high_resolution_clock::time_point bend = std::chrono::high_resolution_clock::now();
			auto du_b2 = std::chrono::duration_cast<std::chrono::milliseconds>(bend - bstart).count();
			float t_test1 = du_b2 / 1000.0;
			t_test1 /= setObjectBBs.size();
			std::cout << "bb matching time = " << t_test1 << std::endl;
		}

		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		std::cout << "Object Tracking processing time = " << t_test1 << " " << n << std::endl;

		////사용자와 가까운 키프레임
		////키프레임 집합
		////연결 된 오브젝트 집합
		//std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(10);
		//vpLocalKFs.push_back(pKF);
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
		////std::cout << "Object Tracking= " << setObjectNodes.size() << std::endl;
		//if (setObjectNodes.size() > 0) {

		//	auto thMaxDesc = SLAM->mpFeatureTracker->max_descriptor_distance;
		//	auto thMinDesc = SLAM->mpFeatureTracker->min_descriptor_distance;

		//	auto cam = pUser->mpCamera;
		//	EdgeSLAM::Frame frame(img, cam, id);
		//	frame.ComputeBoW();
		//	//std::cout << "Object Node tracking = " << setObjectNodes.size() << " " << frame.mDescriptors.rows << std::endl;
		//	
		//	for (auto iter = setObjectNodes.begin(), iend = setObjectNodes.end(); iter != iend; iter++) {
		//		
		//		auto pObjNode = *iter;
		//		std::cout << "Object Tracking = " << vecStrObjectLabels[pObjNode->mnLabel - 1] <<" "<< std::endl;
		//		std::vector<std::pair<int, int>> matches;
		//		EdgeSLAM::SearchPoints::SearchObject(pObjNode, &frame, matches, thMaxDesc, thMinDesc, 0.8, false);
		//		int label = pObjNode->mnLabel;
		//		////시각화
		//		for (int i = 0; i < matches.size(); i++) {
		//			int idx = matches[i].second;
		//			auto pt = frame.mvKeys[idx].pt;
		//			cv::circle(img, pt, 5, SemanticColors[label], -1);
		//		}
		//	}
		//	SLAM->VisualizeImage(pUser->mapName, img, 3);
		//}
		
		

		//실제 트래킹 되는 오브젝트

		//피쳐 검출 & bow 변환
		
		//매칭 && 존재 여부만 체크

		pUser->mnUsed--;
		pUser->mnDebugSeg--;
	}

	//추후에는 오브젝트 디텍션만 하는 용도로. 여기의 코드가 객체 갱신으로 옮기기
	void SemanticProcessor::ObjectDetection(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		/*if (!pUser->ImageDatas.Count(id))
		{
			return;
		}*/
		auto pKF = pUser->KeyFrames.Get(id);
		if (!pKF){
			return;
		}

		std::stringstream ss;
		ss << "/Load?keyword=ObjectDetection" << "&id=" << id << "&src=" << user;
		WebAPI API("143.248.6.143", 35005);
		auto res = API.Send(ss.str(), "");

		int n2 = res.size();
		int n = n2 / 24;

		cv::Mat data = cv::Mat::zeros(n, 6, CV_32FC1);
		std::memcpy(data.data, res.data(), res.size());
		
		////바운딩 박스 & 마스킹
		std::set<EdgeSLAM::ObjectBoundingBox*> spBBoxes;
		for (int j = 0; j < n; j++) {
			int label = (int)data.at<float>(j, 0);
			float conf = data.at<float>(j, 1);

			cv::Point2f left(data.at<float>(j, 2), data.at<float>(j, 3));
			cv::Point2f right(data.at<float>(j, 4), data.at<float>(j, 5));

			//사람이 0이기 때문에 +1을 함.
			auto pBBox = new EdgeSLAM::ObjectBoundingBox(label+1, conf, left, right);
			pBBox->mpKF = pKF;
			spBBoxes.insert(pBBox);
		}
		//std::cout << "AAAAAAAAAAA = " << spBBoxes.size() <<" "<<n << std::endl;
		if (n > 0) {
			GraphKeyFrameObjectBB.Update(pKF, spBBoxes);
		}
		
		return;

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


