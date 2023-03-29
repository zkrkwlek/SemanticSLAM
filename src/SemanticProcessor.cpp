#include <SemanticProcessor.h>
#include <random>
#include <Utils.h>
#include <User.h>
#include <Camera.h>
#include <KeyFrame.h>
#include <MapPoint.h>

#include <SemanticLabel.h>
#include <PlaneEstimator.h>
#include<GridCell.h>

namespace SemanticSLAM {
	SemanticProcessor::SemanticProcessor() {}
	SemanticProcessor::~SemanticProcessor() {}


	
	std::string SemanticProcessor::strLabel = "wall,building,sky,floor,tree,ceiling,road,bed,windowpane,grass,cabinet,sidewalk,person,earth,door,table,mountain,plant,curtain,chair,car,water,painting,sofa,shelf,house,sea,mirror,rug,field,armchair,seat,fence,desk,rock,wardrobe,lamp,bathtub,railing,cushion,base,box,column,signboard,chest of drawers,counter,sand,sink,skyscraper,fireplace,refrigerator,grandstand,path,stairs,runway,case,pool table,pillow,screen door,stairway,river,bridge,bookcase,blind,coffee table,toilet,flower,book,hill,bench,countertop,stove,palm,kitchen island,computer,swivel chair,boat,bar,arcade machine,hovel,bus,towel,light,truck,tower,chandelier,awning,streetlight,booth,television,airplane,dirt track,apparel,pole,land,bannister,escalator,ottoman,bottle,buffet,poster,stage,van,ship,fountain,conveyer belt,canopy,washer,plaything,swimming pool,stool,barrel,basket,waterfall,tent,bag,minibike,cradle,oven,ball,food,step,tank,trade name,microwave,pot,animal,bicycle,lake,dishwasher,screen,blanket,sculpture,hood,sconce,vase,traffic light,tray,ashcan,fan,pier,crt screen,plate,monitor,bulletin board,shower,radiator,glass,clock,flag";
	std::string SemanticProcessor::strYoloObjectLabel = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush";
	
	
	ConcurrentMap<int, std::vector<cv::Point2f>> SemanticProcessor::SuperPoints;
	ConcurrentMap<int, cv::Mat> SemanticProcessor::SemanticLabelImage;
	ConcurrentMap<int, SemanticLabel*> SemanticProcessor::SemanticLabels;
	std::vector<std::string> SemanticProcessor::vecStrSemanticLabels;
	std::vector<std::string> SemanticProcessor::vecStrObjectLabels;
	std::vector<cv::Vec3b> SemanticProcessor::SemanticColors;

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
		if (!pUser->ImageDatas.Count(id))
		{
			return;
		}
		pUser->mnDebugLabel++;
		//pUser->mnUsed++;
		auto pKF = pUser->KeyFrames.Get(id);
		cv::Mat encoded = pUser->ImageDatas.Get(id);
		//cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);

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

		//{
		//	/////save image
		//	std::stringstream sss;
		//	sss << "../bin/img/" << user << "/Track/" << id << "_label.jpg";
		//	cv::imwrite(sss.str(), img);
		//	/////save image
		//}

		//pUser->mnUsed--;
		pUser->mnDebugLabel--;
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
			SLAM->VisualizeImage(img, 3);
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

		SLAM->VisualizeImage(depth, 0);

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
			SLAM->VisualizeImage(segcolor, 1);
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
	void SemanticProcessor::ObjectDetection(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		if (!pUser->ImageDatas.Count(id))
		{
			return;
		}
		pUser->mnUsed++;
		pUser->mnDebugSeg++;

		cv::Mat encoded = pUser->ImageDatas.Get(id);
		int nVisID = pUser->GetVisID();

		pUser->mnDebugSeg--;
		pUser->mnUsed--;

		std::stringstream ss;
		ss << "/Load?keyword=ObjectDetection" << "&id=" << id << "&src=" << user;
		WebAPI API("143.248.6.143", 35005);
		auto res = API.Send(ss.str(), "");

		int n2 = res.size();
		int n = n2 / 24;

		cv::Mat data = cv::Mat::zeros(n, 6, CV_32FC1);
		std::memcpy(data.data, res.data(), res.size());
		cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);

		for (int j = 0; j < n; j++) {
			int label = (int)data.at<float>(j, 0);
			float conf = data.at<float>(j, 1);
			if (conf < 0.6)
				continue;
			std::stringstream ss;
			ss << vecStrObjectLabels[label] << "(" << conf << ")";
			cv::Point2f left(data.at<float>(j, 2), data.at<float>(j, 3));
			cv::Point2f right(data.at<float>(j, 4), data.at<float>(j, 5));

			rectangle(img, left, right, cv::Scalar(255, 255, 255));
			cv::putText(img, ss.str(), cv::Point(left.x, left.y - 6), 1, 1.5, cv::Scalar::all(0));
			//std::cout <<label<<"="<< left << " " << right << std::endl;
		}

		if (nVisID == 0)
			SLAM->VisualizeImage(img, 2);
		//std::cout << "4" << std::endl;
		
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
		
		SLAM->VisualizeImage(img1, 2);
		SLAM->VisualizeImage(img2, 3);
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
			SLAM->VisualizeImage(rectified1, 2);
			SLAM->VisualizeImage(rectified2, 3);

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


