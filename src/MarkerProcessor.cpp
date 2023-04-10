#include <MarkerProcessor.h>
#include <Windows.h>
#include <SLAM.h>

#include <MapPoint.h>
#include <User.h>
#include <Camera.h>

#include <PlaneEstimator.h>
#include <ContentProcessor.h>
//#include <SemanticProcessor.h>
//#include <SemanticLabel.h>


void distortPoint(const cv::Point2f& xy, cv::Point2f& uv, const cv::Mat &M, const cv::Mat &d)
{
	cv::Mat K;
	cv::Mat D;
	M.convertTo(K, CV_64FC1);
	d.convertTo(D, CV_64FC1);
	std::cout << K << " " << D << std::endl;
	cv::Mat tmp;
	cv::Mat xyz = (cv::Mat_<double>(3, 1) << (double)xy.x, (double)xy.y, 1.0);
	cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::projectPoints(xyz, rvec, tvec, K, D, tmp);
	uv.x = (float)tmp.at<double>(0);
	uv.y = (float)tmp.at<double>(1);
	std::cout << "distort back = " << uv << std::endl;
}

namespace SemanticSLAM {
	
	ConcurrentMap<int, bool>	MapDynamicMarker;
	ConcurrentMap<int, cv::Mat> MarkerProcessor::MapMarkerPos;
	////Ű���������� ����, �������� ���� ��� ��
	ConcurrentMap<int, std::set<EdgeSLAM::KeyFrame*>> MarkerProcessor::MapMarkerKFs;
	//ConcurrentMap<int, EdgeSLAM::KeyFrame*> MarkerProcessor::MapMarkerKFs;
	
	////��� ���� ����
	ConcurrentMap<int, SemanticSLAM::Content*> MapContents;
	ConcurrentMap<int, std::chrono::high_resolution_clock::time_point> MapMarkerStart;
	
	MarkerProcessor::MarkerProcessor() {}
	MarkerProcessor::~MarkerProcessor() {}
	
	int nStartID = 100;
	int startID = nStartID;
	int endID = 98;
	bool bDynamicMove = false;
	int nDetectedMarker = 0;
	float totalTime = 5.0;


	ConcurrentVector<float> VectorInconsistency;

	void MarkerProcessor::SaveInconsistency() {

		std::stringstream ssPath;
		ssPath << "../bin/trajectory/";

		std::stringstream ss;
		ss << ssPath.str() << "inconsistency.txt";
		std::ofstream f;
		f.open(ss.str().c_str());

		auto vecInconsistency = VectorInconsistency.get();
		for (int i = 0; i < vecInconsistency.size(); i++) {
			float len = vecInconsistency[i];
			f << len << std::endl;
		}
		f.close();
	}
	void MarkerProcessor::MarkerEventDetect(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;

		pUser->QueueNotiMsg.Update(id, "marker");

		pUser->mnUsed--;
	}
	void MarkerProcessor::MarkerRegistrationAA(EdgeSLAM::SLAM* SLAM, std::string user, int id, float len, float inc) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;
		if (!pUser->QueueNotiMsg.Count(id)) {
			pUser->mnUsed--;
			return;
		}
		pUser->QueueNotiMsg.Erase(id);

		//////��Ŀ����
		WebAPI API("143.248.6.143", 35005);
		std::stringstream ss;
		ss << "/Load?keyword=MarkerResults" << "&id=" << id << "&src=" << user;
		auto res3 = API.Send(ss.str(), "");

		cv::Mat tempdata = cv::Mat::zeros(res3.size(), 1, CV_8UC1);
		std::memcpy(tempdata.data, res3.data(), res3.size());
		float* fdata = (float*)tempdata.data;
		int numMarker = (int)fdata[0];
		//////��Ŀ����

		auto pKF = pUser->mpRefKF;
		if (pKF)
		{
			std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(20);
			vpLocalKFs.push_back(pKF);

			///���� �������� ��� �ν��� ������ ���
			
			if (pUser->PoseDatas.Count(id)) {

				////�̹��� ������ ȹ��
				cv::Mat encoded = pUser->ImageDatas.Get(id);
				cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);

				////���� �������� ���� ȹ��
				cv::Mat K = pUser->GetCameraMatrix();
				cv::Mat Kinv = pUser->GetCameraInverseMatrix();
				cv::Mat T = pUser->PoseDatas.Get(id);
				cv::Mat Rslam = T.rowRange(0, 3).colRange(0, 3);
				cv::Mat tslam = T.rowRange(0, 3).col(3);
				cv::Mat Tslaminv = cv::Mat::eye(4, 4, CV_32FC1); //pKF->GetPoseInverse();
				cv::Mat Rinv = Rslam.t();
				cv::Mat tinv = -Rinv*tslam;
				Rinv.copyTo(Tslaminv.rowRange(0, 3).colRange(0, 3));
				tinv.copyTo(Tslaminv.col(3).rowRange(0, 3));
				cv::Mat Ow = tinv.clone();// pUser->GetPosition();

				////��� ������ ȹ��
				Plane* floor = nullptr;
				std::set<Plane*> tempWallPlanes;
				std::map<PlaneType, std::set<Plane*>> LocalMapPlanes;
				for (std::vector<EdgeSLAM::KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
				{
					EdgeSLAM::KeyFrame* pKFi = *itKF;
					if (PlaneEstimator::mPlaneConnections.Count(pKFi)) {
						auto tempPlanes = PlaneEstimator::mPlaneConnections.Get(pKFi);
						for (auto iter = tempPlanes.begin(), iend = tempPlanes.end(); iter != iend; iter++) {
							auto plane = *iter;
							LocalMapPlanes[plane->type].insert(plane);
							if (plane->type == PlaneType::FLOOR) {
								floor = plane;
								break;
							}
						}
					}
					if (!floor)
						break;
				}

				if (floor) {

					for (int i = 0; i < numMarker; i++) {
						int markerID = fdata[3 * i + 1];
						float x = fdata[3 * i + 2];
						float y = fdata[3 * i + 3];
						
						cv::Point2f corner(x, y);
						////��Ŀ�� ��ϵǾ� ���� ���� ��쿡 �߰�
						if (!ContentProcessor::MapArucoMarkerPos.Count(markerID)) {
							cv::Mat Xw;
							bool bX3D = Utils::CreateWorldPoint(corner, Kinv, Tslaminv, Ow, floor->param, Xw);
							int cid = ContentProcessor::MarkerContentRegistration(SLAM, pKF, user, Xw, markerID);
						}
						
						

					}//for
					
				}else
						std::cout << "not found floor" << std::endl;

				////��Ŀ ���� ����
				//for (int i = 0; i < numMarker; i++) {
				//	int markerID = fdata[3 * i + 1];
				//	float x = fdata[3 * i + 2];
				//	float y = fdata[3 * i + 3];

				//	cv::Point2f corner(x, y);
				//	if (ContentProcessor::MapArucoMarkerPos.Count(markerID)) {
				//		float depth;
				//		cv::Point2f xy;
				//		cv::Mat Xw = ContentProcessor::MapArucoMarkerPos.Get(markerID);
				//		bool bProj = Utils::ProjectPoint(xy, depth, Xw, K, Rslam, tslam);
				//		if (bProj) {
				//			cv::circle(img, xy, 5, cv::Scalar(0, 255, 255), -1);
				//			cv::circle(img, corner, 3, cv::Scalar(255, 255, 0), -1);
				//			std::cout << "projection test " << markerID << "=" << corner << " " << xy << std::endl;

				//			cv::Point2f diff = xy - corner;
				//			float dist = sqrt(diff.dot(diff));

				//			{
				//				cv::Mat data = cv::Mat::zeros(2, 1, CV_32FC1);
				//				data.at<float>(0) = markerID;
				//				data.at<float>(1) = dist;
				//				std::stringstream ss;
				//				ss << "/Store?keyword=MarkerDist&id=" << id << "&src=" << user;
				//				WebAPI API("143.248.6.143", 35005);
				//				auto res = API.Send(ss.str(), data.data, data.rows * sizeof(float));
				//			}
				//		}
				//		else {
				//			std::cout << "projection fail " << depth << ", " << xy << std::endl;
				//		}
				//	}//if find marker
				//}//for traverse marker
				////��Ŀ ���� ��
				if(pUser->GetVisID()==0)
					SLAM->VisualizeImage(img, 3);
			}
			else
				std::cout << "not found pose data" << std::endl;
		}
		else
			std::cout << "Not found Keyframe" << std::endl;
		
		//cv::Mat K = pUser->GetCameraMatrix();
		//cv::Mat D = pUser->GetDistortionMatrix();

		//cv::Mat encoded = pUser->ImageDatas.Get(id);
		//cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);

		//if (!pUser->PoseDatas.Count(id)) {
		//	pUser->mnUsed--;
		//	return;
		//}

		//cv::Mat res; //�̹��� ����
		//std::vector<Marker*> vecMarkers;
		//img.copyTo(res);
		//ArucoMarker::MarkerDetection(img, K, D, vecMarkers, len, inc);
		//if (vecMarkers.size() == 0) {
		//	pUser->mnUsed--;
		//	return;
		//}

		//cv::Mat T = pUser->PoseDatas.Get(id);
		//cv::Mat Rslam = T.rowRange(0, 3).colRange(0, 3);
		//cv::Mat tslam = T.rowRange(0, 3).col(3);
		//auto pMarker = vecMarkers[0];
		//if (!MapMarkerPos.Count(pMarker->mnId))
		//{
		//	cv::Mat X = pMarker->t.clone();
		//	cv::Mat proj = K*X;
		//	float depth = proj.at<float>(2);
		//	float px = proj.at<float>(0) / depth;
		//	float py = proj.at<float>(1) / depth;
		//	cv::circle(res, cv::Point2f(px, py), 3, cv::Scalar(0, 0, 255), -1);
		//	
		//	cv::Mat Rinv = Rslam.t();
		//	cv::Mat tinv = -Rinv*tslam;
		//	X = Rinv*X + tinv;
		//	MapMarkerPos.Update(pMarker->mnId, X);
		//}
		//else {
		//	cv::Point2f pt = pMarker->vecCorners[0];
		//	cv::Mat Xw = MapMarkerPos.Get(pMarker->mnId);
		//	cv::Point2f xy;
		//	float depth;
		//	Utils::ProjectPoint(xy, depth, Xw, K, Rslam, tslam);
		//	cv::circle(res, pt, 5, cv::Scalar(255, 0, 0), -1);
		//	cv::circle(res, xy, 3, cv::Scalar(0, 0, 255), -1);
		//	std::cout << xy << std::endl;
		//}
		//SLAM->VisualizeImage(res, 2);
		//SLAM->VisualizeImage(res, 3);
		////if (pMarker->mnId == 100 && MapMarkerPos.Count(100)) {
		////	//���ø��� ��� ����
		////	cv::Mat T = pUser->PoseDatas.Get(id);
		////	cv::Mat Rslam = T.rowRange(0, 3).colRange(0, 3);
		////	cv::Mat tslam = T.rowRange(0, 3).col(3);

		////	float depth;
		////	cv::Point2f xy;
		////	cv::Mat Xw = MapMarkerPos.Get(100);
		////	Utils::ProjectPoint(xy, depth, Xw, K, Rslam, tslam);

		////	cv::Point2f pt = pMarker->vecCorners[0];
		////	auto pt2 = pt - xy;
		////	float len = sqrt(pt2.dot(pt2));
		////	VectorInconsistency.push_back(len);

		////	cv::circle(img, pt, 3, cv::Scalar(255, 0, 0), -1);
		////	cv::circle(img, xy, 3, cv::Scalar(0, 0, 255), -1);
		////	SLAM->VisualizeImage(img, 2);
		////}
		pUser->mnUsed--;
	}

	void MarkerProcessor::CalculateInconsistency(EdgeSLAM::SLAM* SLAM, std::string user, int id, float len, float inc) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;

		cv::Mat K = pUser->GetCameraMatrix();
		cv::Mat D = pUser->GetDistortionMatrix();

		cv::Mat encoded = pUser->ImageDatas.Get(id);
		cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);

		cv::Mat res;
		std::vector<Marker*> vecMarkers;
		img.copyTo(res);
		ArucoMarker::MarkerDetection(img, K, D, vecMarkers, len, inc);
		if (vecMarkers.size() == 0) {
			pUser->mnUsed--;
			return;
		}
		auto pMarker = vecMarkers[0];
		if (pMarker->mnId == 100 && MapMarkerPos.Count(100)) {
			//���ø��� ��� ����
			cv::Mat T = pUser->PoseDatas.Get(id);
			cv::Mat Rslam = T.rowRange(0, 3).colRange(0, 3);
			cv::Mat tslam = T.rowRange(0, 3).col(3);

			float depth;
			cv::Point2f xy;
			cv::Mat Xw = MapMarkerPos.Get(100);
			Utils::ProjectPoint(xy, depth, Xw, K, Rslam, tslam);

			cv::Point2f pt = pMarker->vecCorners[0];
			auto pt2 = pt - xy;
			float len = sqrt(pt2.dot(pt2));
			VectorInconsistency.push_back(len);

			cv::circle(img, pt, 3, cv::Scalar(255, 0, 0), -1);
			cv::circle(img, xy, 3, cv::Scalar(0, 0, 255), -1);
			SLAM->VisualizeImage(img, 2);
		}
		pUser->mnUsed--;
	}

	void MarkerProcessor::DynamicObjectVisTest(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;

		cv::Mat K = pUser->GetCameraMatrix();
		cv::Mat D = pUser->GetDistortionMatrix();

		cv::Mat encoded = pUser->ImageDatas.Get(id);
		cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);
		
		//���۷��� Ű�������� ��ã���� �н�
		auto pKF = pUser->mpRefKF;
		if (!pKF) {
			pUser->mnUsed--;
			return;
		}

		//////Ʈ���Ű� �ʿ���.
		if (!bDynamicMove && nDetectedMarker == 3) {

			cv::Mat res;
			std::vector<Marker*> vecMarkers;
			img.copyTo(res);
			ArucoMarker::MarkerDetection(img, K, D, vecMarkers, 0.018, 0.07);
			if (vecMarkers.size() == 0) {
				pUser->mnUsed--;
				return;
			}
			auto pMarker = vecMarkers[0];
			if(pMarker->mnId == nStartID)
				bDynamicMove = true;
			/*bool bTrigger = true;
			for (int i = startID; i <= endID; i--) {
				if (!MapMarkerPos.Count(startID))
				{
					bTrigger = false;
					break;
				}
			}
			if (bTrigger)
				bDynamicMove = true;*/
		}
		

		////��Ŀ �ν� ���ϸ� �н�
		
		
		
		//if (pMarker->mnId != startID) {
		//	pUser->mnUsed--;
		//	return;
		//}

		
		
		int nextID = startID-1;
		if (bDynamicMove && MapMarkerPos.Count(startID) && MapMarkerPos.Count(nextID)){

			//���ø��� ��� ����
			cv::Mat T = pUser->PoseDatas.Get(id);
			cv::Mat Rslam = T.rowRange(0, 3).colRange(0, 3);
			cv::Mat tslam = T.rowRange(0, 3).col(3);

			if (!MapMarkerStart.Count(startID)) {
				//MapDynamicMarker.Update(100, true);
				MapMarkerStart.Update(startID, std::chrono::high_resolution_clock::now());
			}

			////Ʈ���Ÿ� ����
			//Ű������ ID
			auto tstart = MapMarkerStart.Get(startID);
			auto tcurr = std::chrono::high_resolution_clock::now();
			auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(tcurr - tstart).count();
			float t_test1 = du_test1 / 1000.0;

			cv::Mat Xs = MapMarkerPos.Get(startID);
			cv::Mat Xe   = MapMarkerPos.Get(nextID);

			/*{
				auto ePos = MapContents.Get(nextID)->pos;
				MapContents.Get(startID)->endPos = ePos.clone();
				int cid = MapContents.Get(startID)->mnID;
			}*/
			
			cv::Mat dir = (Xe - Xs)/totalTime;
			/*float mag = dir.dot(dir)*20.0;
			dir /= mag;*/
			cv::Mat Xt = Xs + dir*t_test1;
			float depth;
			cv::Point2f xy1, xy2, xy3;
			Utils::ProjectPoint(xy1, depth, Xs, K, Rslam, tslam);
			Utils::ProjectPoint(xy2, depth, Xe, K, Rslam, tslam);
			Utils::ProjectPoint(xy3, depth, Xt, K, Rslam, tslam);
			cv::line(img, xy1, xy2, cv::Scalar(255, 0, 0), 2);
			cv::circle(img, xy3, 5, cv::Scalar(0, 0, 255), -1);
			SLAM->VisualizeImage(img, 2);

			////���� ��η� ����
			if (t_test1 > totalTime) {
				MapMarkerStart.Erase(startID);
				startID--;
			}
		}
		////���� ��ü �ʱ�ȭ
		if (startID == endID) {
			startID = nStartID;
			bDynamicMove = false;
		}
		pUser->mnUsed --;
	}
	void MarkerProcessor::DynamicObjectRegTest(EdgeSLAM::SLAM* SLAM, std::string user, int id, float len, float inc) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;

		cv::Mat K = pUser->GetCameraMatrix();
		cv::Mat D = pUser->GetDistortionMatrix();

		cv::Mat encoded = pUser->ImageDatas.Get(id);
		cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);
		cv::Mat res;
		img.copyTo(res);

		std::vector<Marker*> vecMarkers;
		//��Ŀ �ν� ���ϸ� �н�
		ArucoMarker::MarkerDetection(img, K, D, vecMarkers, len, inc);
		if (vecMarkers.size() == 0) {
			pUser->mnUsed--;
			return;
		}
		auto pMarker = vecMarkers[0];

		//���۷��� Ű�������� ��ã���� �н�
		auto pKF = pUser->mpRefKF;
		if (!pKF) {
			pUser->mnUsed--;
			return;
		}
		////���ø��� KF ���
		std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(20);
		vpLocalKFs.push_back(pKF);
		auto pMap = SLAM->GetMap(pUser->mapName);

		//���ø��� ��� ����
		cv::Mat T = pUser->PoseDatas.Get(id);
		cv::Mat Rslam = T.rowRange(0, 3).colRange(0, 3);
		cv::Mat tslam = T.rowRange(0, 3).col(3);
		/*cv::Mat Tslam = cv::Mat::eye(4, 4, CV_32FC1);
		Rslam.copyTo(Tslam.rowRange(0, 3).colRange(0, 3));
		tslam.copyTo(Tslam.col(3).rowRange(0, 3));*/

		cv::Mat Kinv = pUser->GetCameraInverseMatrix();
		cv::Mat Tslaminv = cv::Mat::eye(4, 4, CV_32FC1); //pKF->GetPoseInverse();
		cv::Mat Rinv = Rslam.t();
		cv::Mat tinv = -Rinv*tslam;
		Rinv.copyTo(Tslaminv.rowRange(0, 3).colRange(0, 3));
		tinv.copyTo(Tslaminv.col(3).rowRange(0, 3));
		cv::Mat Ow = tinv.clone();// pUser->GetPosition();


		Plane* floor = nullptr;
		Plane* ceil = nullptr;
		std::set<Plane*> tempWallPlanes;
		std::map<PlaneType, std::set<Plane*>> LocalMapPlanes;

		for (std::vector<EdgeSLAM::KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			EdgeSLAM::KeyFrame* pKFi = *itKF;
			if (PlaneEstimator::mPlaneConnections.Count(pKFi)) {
				auto tempPlanes = PlaneEstimator::mPlaneConnections.Get(pKFi);
				for (auto iter = tempPlanes.begin(), iend = tempPlanes.end(); iter != iend; iter++) {
					auto plane = *iter;
					LocalMapPlanes[plane->type].insert(plane);
					if (plane->type == PlaneType::FLOOR) {
						floor = plane;
						break;
					}
				}
			}
			if (!floor)
				break;
		}
		if (!floor) {
			pUser->mnUsed--;
			return;
		}

		if (pMarker->mnId > 97) {

			if (!MapMarkerPos.Count(pMarker->mnId)) {
				auto pt = pMarker->vecCorners[0];
				cv::Mat Xw;
				float depth;
				cv::Point2f xy;
				bool bX3D = Utils::CreateWorldPoint(pt, Kinv, Tslaminv, Ow, floor->param, Xw);
				bool bProj = bX3D && Utils::ProjectPoint(xy, depth, Xw, K, Rslam, tslam);
				//std::cout << xy << std::endl;
				//cv::circle(res, xy, 5, cv::Scalar(0, 0, 255), -1);
				//SLAM->VisualizeImage(res, 2);
				MapMarkerPos.Update(pMarker->mnId, Xw);
				nDetectedMarker++;

				std::cout << "MARKER DETECTED = " <<pMarker->mnId<< std::endl << std::endl << std::endl << std::endl;
				{
					//api�� ����
					cv::Mat data = cv::Mat::zeros(1000, 1, CV_32FC1);
					data.at<float>(0) = pt.x;
					data.at<float>(1) = pt.y;
					data.at<float>(2) = Xw.at<float>(0);
					data.at<float>(3) = Xw.at<float>(1);
					data.at<float>(4) = Xw.at<float>(2);
					data.at<float>(5) = 100.0; //�ϴ� ���̳��� ǥ��
					data.at<float>(6) = (float)pMarker->mnId; // ��Ŀ ���̵� ���� �ѱ��

					int cid = ContentProcessor::ContentRegistration(SLAM, pKF, user, data,0);
					MapContents.Update(pMarker->mnId, ContentProcessor::GetContent(cid));
					//std::stringstream ss;
					//ss << "/Store?keyword=" << "ContentGeneration" << "&id=" << id << "&src=" << "MARKER" << "&ts=" << std::fixed << std::setprecision(6) << 0.0;
					////std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();
					//WebAPI api("143.248.6.143", 35005);
					//api.Send(ss.str(), (const unsigned char*)data.data, sizeof(float) * 1000);
				}
			}
			////��� �߰�
			if (MapContents.Count(pMarker->mnId) && MapContents.Count(pMarker->mnId - 1)) {
				auto pContent = MapContents.Get(pMarker->mnId);
				auto pNextContent = MapContents.Get(pMarker->mnId - 1);
				auto ePos = pNextContent->pos;

				pContent->attribute.at<float>(0, 0) = 1.0;
				pContent->endPos = ePos.clone();
				pContent->mnNextID = pNextContent->mnID;
			}
			
		}
		pUser->mnUsed--;
	}

	void MarkerProcessor::MarkerCreation(EdgeSLAM::SLAM* SLAM, std::string user, int id){
		//if (vecMarkers.size() > 0)
		//{
		//	//��Ŀ ��� ������ 3���� ���� ���
		//	if (!pUser->PoseDatas.Count(id)) {
		//		pUser->mnUsed--;
		//		return;
		//	}
		//	auto T = pUser->PoseDatas.Get(id);
		//	cv::Mat Rslam = T.rowRange(0, 3).colRange(0, 3);
		//	cv::Mat tslam = T.rowRange(0, 3).col(3);
		//	cv::Mat Tslaminv = cv::Mat::eye(4, 4, CV_32FC1); //pKF->GetPoseInverse();
		//	cv::Mat Rinv = Rslam.t();
		//	cv::Mat tinv = -Rinv * tslam;
		//	Rinv.copyTo(Tslaminv.rowRange(0, 3).colRange(0, 3));
		//	tinv.copyTo(Tslaminv.col(3).rowRange(0, 3));
		//	cv::Mat Ow = tinv.clone();// pUser->GetPosition();

		//	auto pMarker = vecMarkers[0];
		//	
		//	////��� �׽�Ʈ
		//	{
		//		////���ø��� KF ���
		//		std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(50);
		//		vpLocalKFs.push_back(pKF);
		//		auto pMap = SLAM->GetMap(pUser->mapName);

		//		Plane* floor = nullptr;
		//		Plane* ceil = nullptr;
		//		std::set<Plane*> tempWallPlanes;
		//		std::map<PlaneType, std::set<Plane*>> LocalMapPlanes;

		//		for (std::vector<EdgeSLAM::KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		//		{
		//			EdgeSLAM::KeyFrame* pKFi = *itKF;
		//			if (PlaneEstimator::mPlaneConnections.Count(pKFi)) {
		//				auto tempPlanes = PlaneEstimator::mPlaneConnections.Get(pKFi);
		//				for (auto iter = tempPlanes.begin(), iend = tempPlanes.end(); iter != iend; iter++) {
		//					auto plane = *iter;
		//					LocalMapPlanes[plane->type].insert(plane);
		//					if (plane->type == PlaneType::FLOOR) {
		//						auto norm = PlaneEstimator::calcSphericalCoordinate(plane->normal);
		//						int idx = PlaneEstimator::ConvertSphericalToIndex(norm);
		//						int count = PlaneEstimator::GlobalNormalCount.Get(idx);
		//						if (count > 80) {
		//							floor = plane;
		//							break;
		//						}
		//					}
		//				}
		//			}
		//			if (floor)
		//				break;
		//		}
		//		if (PlaneEstimator::GlobalFloor->nScore > 0)
		//			floor = PlaneEstimator::GlobalFloor;
		//		if (floor) {
		//			////���ý��Ͻ� üũ
		//			cv::Mat Kinv = pUser->GetCameraInverseMatrix();
		//			//cv::Mat Tslaminv = pKF->GetPoseInverse();
		//			//cv::Mat Ow = pUser->GetPosition();

		//			cv::Mat param = floor->param.clone();
		//			cv::Mat normal = floor->normal.clone();

		//			////��Ŀ�� ������� ����
		//			for (int i = 0; i < vecMarkers.size(); i++) {
		//				auto pMarker = vecMarkers[i];
		//				auto pt = pMarker->vecCorners[0];

		//				cv::Point2f ptun;
		//				Utils::undistortPoint(pt, ptun, K, D);

		//				//��� ������ ����
		//				cv::Mat x3D = (cv::Mat_<float>(3, 1) << ptun.x, ptun.y, 1.0);
		//				cv::Mat Xw = Kinv * x3D;
		//				Xw.push_back(cv::Mat::ones(1, 1, CV_32FC1)); //3x1->4x1
		//				Xw = Tslaminv * Xw; // 4x4 x 4 x 1
		//				Xw = Xw.rowRange(0, 3) / Xw.at<float>(3); // 4x1 -> 3x1
		//				cv::Mat dir = Xw - Ow; //3x1
		//				float dist = param.at<float>(3);
		//				float a = -normal.dot(dir);
		//				if (std::abs(a) < 0.000001)
		//					continue;
		//				float u = (normal.dot(Ow) + dist) / a;
		//				cv::Mat Xplane = Ow + dir * u;
		//				//��� ������ ����

		//				cv::Point2f xy;
		//				float depth;
		//				Utils::ProjectPoint(xy, depth, Xplane, K, Rslam, tslam);
		//				cv::circle(res, xy, 7, cv::Scalar(125, 125, 125), -1);

		//				{
		//					if (!MapMarkerPos.Count(pMarker->mnId))
		//					{
		//						cv::Mat R, rvec;
		//						cv::Rodrigues(pMarker->rvec, R);
		//						cv::Rodrigues(Rslam.t() * R, rvec);

		//						Xplane.push_back(rvec);
		//						MapMarkerPos.Update(pMarker->mnId, Xplane);

		//						////���� �ð�ȭ�� ��Ŀ ������ �߰�
		//						std::map<int, cv::Mat> mapDatas;
		//						if (SLAM->TemporalDatas2.Count("marker"))
		//							mapDatas = SLAM->TemporalDatas2.Get("marker");
		//						cv::Mat X = cv::Mat::zeros(3, 1, CV_32FC1);
		//						X.at<float>(0) = Xplane.at<float>(0);
		//						X.at<float>(1) = Xplane.at<float>(1);
		//						X.at<float>(2) = Xplane.at<float>(2);
		//						mapDatas[pMarker->mnId] = X;
		//						SLAM->TemporalDatas2.Update("marker", mapDatas);
		//						////���� �ð�ȭ�� ��Ŀ ������ �߰�

		//						////Ű������ �����ϱ�.
		//						//�ϴ� �����ϰ� �н��� �����Ǹ� �н� ������ �� �� �ֵ��� �ϱ�.
		//						{
		//							auto spLocalKFs = pUser->mSetLocalKeyFrames.Get();
		//							if (!spLocalKFs.count(pKF))
		//								spLocalKFs.insert(pKF);
		//							MapMarkerKFs.Update(pMarker->mnId, spLocalKFs);
		//							//MapMarkerKFs.Update(pMarker->mnId, pKF);

		//						}
		//						////Ű������ �����ϱ�.

		//						////�н� �׽�Ʈ
		//						int prevID = pMarker->mnId - 1;
		//						int nextID = pMarker->mnId + 1;
		//						int pathid = -1;
		//						int pathEndID = -1;
		//						cv::Mat pathData = cv::Mat::zeros(0, 1, CV_32FC1);
		//						bool bPath = false;
		//						if (MapMarkerPos.Count(prevID)) {
		//							pathid = prevID;
		//							pathEndID = pMarker->mnId;
		//							cv::Mat Xp = MapMarkerPos.Get(prevID).rowRange(0, 3);
		//							pathData.push_back(Xp);
		//							pathData.push_back(X);
		//							bPath = true;
		//						}
		//						if (MapMarkerPos.Count(nextID)) {
		//							pathid = pMarker->mnId;
		//							pathEndID = nextID;
		//							cv::Mat Xn = MapMarkerPos.Get(nextID).rowRange(0, 3);
		//							pathData.push_back(X);
		//							pathData.push_back(Xn);
		//							bPath = true;
		//						}
		//						//�ð�ȭ�� �߰�
		//						if (bPath)
		//						{
		//							std::map<int, cv::Mat> mapDatas;
		//							if (SLAM->TemporalDatas2.Count("path"))
		//								mapDatas = SLAM->TemporalDatas2.Get("path");
		//							mapDatas[pathid] = pathData;
		//							SLAM->TemporalDatas2.Update("path", mapDatas);
		//							//MarkerGraphTraverse(pathid, pathEndID);
		//							ContentProcessor::PathContentRegistration(SLAM, pathid, pathEndID, user, pathData, 0);
		//						}
		//						////�н� �׽�Ʈ
		//					}
		//					else {

		//						cv::Mat Xw = MapMarkerPos.Get(pMarker->mnId);
		//						cv::Point2f xy;
		//						float depth;
		//						Utils::ProjectPoint(xy, depth, Xw.rowRange(0, 3), K, Rslam, tslam);

		//						//cv::drawFrameAxes(res, K, D, rvec,tvec, 0.1);
		//						//cv::drawFrameAxes(res, K, D, pMarker->rvec, pMarker->t, 0.1);
		//						std::cout << xy << " " << pMarker->vecCorners[0] << std::endl;
		//						cv::circle(res, xy, 3, cv::Scalar(0, 0, 255), -1);
		//					}
		//				}

		//				{
		//					//������ ����
		//					cv::Mat corners = cv::Mat::zeros(8, 1, CV_32FC1);
		//					corners.at<float>(0) = pMarker->vecCorners[0].x;
		//					corners.at<float>(1) = pMarker->vecCorners[0].y;
		//					corners.at<float>(2) = pMarker->vecCorners[1].x;
		//					corners.at<float>(3) = pMarker->vecCorners[1].y;
		//					corners.at<float>(4) = pMarker->vecCorners[2].x;
		//					corners.at<float>(5) = pMarker->vecCorners[2].y;
		//					corners.at<float>(6) = pMarker->vecCorners[3].x;
		//					corners.at<float>(7) = pMarker->vecCorners[3].y;

		//					//������ : 2 + 3 + 3 + 8 = 16
		//					//1 : id
		//					//2 : depth
		//					//3-5 : t
		//					//6-8 : rvec
		//					//9-16: corner
		//					cv::Mat data = cv::Mat(2, 1, CV_32FC1);
		//					data.at<float>(0) = (float)id;
		//					data.at<float>(1) = u;
		//					cv::Mat Xw = MapMarkerPos.Get(pMarker->mnId);
		//					data.push_back(Xw);
		//					//data.push_back(pMarker->rvec);
		//					data.push_back(corners);

		//					std::stringstream ss;
		//					ss << "/Store?keyword=ArUcoMarkerDetection2" << "&id=" << pMarker->mnId << "&src=" << user;
		//					auto res = API.Send(ss.str(), data.data, data.rows * sizeof(float));
		//				}
		//			}//for marker
		//		}
		//		else {
		//			std::cout << "not found plane param" << std::endl;
		//		}//if floor
		//	}
		//	///��� �׽�Ʈ
		//}
	}

	void MarkerProcessor::MarkerRegistration(EdgeSLAM::SLAM* SLAM, std::string keyword, std::string user, int mid) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;
		
		WebAPI API("143.248.6.143", 35005);
		if (!MapMarkerPos.Count(mid)) {
			//auto vpLocalKFs = pUser->mSetLocalKeyFrames.Get();

			std::stringstream ss;
			ss << "/Load?keyword=" << keyword << "&id=" << mid << "&src=" << user;
			
			auto res = API.Send(ss.str(), "");
			int n2 = res.size();

			cv::Mat fdata = cv::Mat::zeros(6, 1, CV_32FC1);
			std::memcpy(fdata.data, res.data(), res.size());

			MapMarkerPos.Update(mid, fdata);
		}
		else {
			cv::Mat Xw = MapMarkerPos.Get(mid);
			std::stringstream ss;
			ss << "/Store?keyword=VO.MARKER.CREATED" << "&id=" << mid<< "&src=" << user;
			auto res = API.Send(ss.str(), Xw.data, Xw.rows * sizeof(float));
		}
		pUser->mnUsed--;
	}

	void MarkerProcessor::MarkerTest(EdgeSLAM::SLAM* SLAM, std::string user, int id, float len, float inc) {
		//std::cout << "marker test 1 " << std::endl;
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;
		
		cv::Mat K = pUser->GetCameraMatrix();
		cv::Mat D = pUser->GetDistortionMatrix();

		if (!pUser->ImageDatas.Count(id)) {
			pUser->mnUsed--;
			return;
		}

		cv::Mat encoded = pUser->ImageDatas.Get(id);
		cv::Mat img = cv::imdecode(encoded, cv::IMREAD_COLOR);
		cv::Mat res;
		img.copyTo(res);
		
		std::vector<Marker*> vecMarkers;
		ArucoMarker::MarkerDetection(img, K, D, vecMarkers, len, inc);
		if (vecMarkers.size() == 0) {
			pUser->mnUsed--;
			return;
		}

		//WebAPI API("143.248.6.143", 35005);
		//for (int i = 0; i < vecMarkers.size(); i++) {
		//	auto pMarker = vecMarkers[i];
		//	auto pt = pMarker->vecCorners[0];
		//	//cv::Mat X = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
		//	//cv::Mat resa;
		//	//cv::projectPoints(X, pMarker->rvec, pMarker->t, K, D, resa);
		//	//cv::Point2f pt3((float)resa.at<double>(0), (float)resa.at<double>(1));
		//	cv::circle(res, pt, 3, cv::Scalar(0, 255, 0), -1);

		//	cv::Mat corners = cv::Mat::zeros(8, 1, CV_32FC1);
		//	corners.at<float>(0) = pMarker->vecCorners[0].x;
		//	corners.at<float>(1) = pMarker->vecCorners[0].y;
		//	corners.at<float>(2) = pMarker->vecCorners[1].x;
		//	corners.at<float>(3) = pMarker->vecCorners[1].y;
		//	corners.at<float>(4) = pMarker->vecCorners[2].x;
		//	corners.at<float>(5) = pMarker->vecCorners[2].y;
		//	corners.at<float>(6) = pMarker->vecCorners[3].x;
		//	corners.at<float>(7) = pMarker->vecCorners[3].y;

		//	cv::Mat data = cv::Mat(1, 1, CV_32FC1);
		//	data.at<float>(0) = (float)id;
		//	data.push_back(pMarker->rvec);
		//	data.push_back(pMarker->t);
		//	data.push_back(corners);

		//	std::stringstream ss;
		//	ss << "/Store?keyword=ArUcoMarkerDetection" << "&id=" << pMarker->mnId << "&src=" << user;
		//	auto res = API.Send(ss.str(), data.data, data.rows * sizeof(float));
		//}
		//��Ŀ ���� ����

		if(pUser->GetVisID()==0)
			SLAM->VisualizeImage(res, 2);

		pUser->mnUsed--;
		return;
	}

	//std::vector<EdgeSLAM::KeyFrame*> MarkerProcessor::MarkerGraphTraverse(int startID, int endID) {
	//	//auto 
	//	auto pKFstart = MapMarkerKFs.Get(startID);
	//	auto pKFend   = MapMarkerKFs.Get(endID);

	//	//������ Ű�������� ���� Ű������ �߰��ϱ�
	//	std::set<EdgeSLAM::KeyFrame*> spKFs;
	//	for (auto iter = spKFs.begin(), iend = spKFs.end(); iter != iend; iter++) {
	//		//���� ���
	//		auto pKF = *iter;
	//		if (!pKF)
	//			break;
	//		std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(20);
	//		//�߰��ϱ�
	//		for (auto jter = vpLocalKFs.begin(), jend = vpLocalKFs.end(); jter != jend; jter++) {
	//			auto pKFj = *jter;
	//			if (spKFs.count(pKFj))
	//				continue;
	//		}
	//		if (pKF == pKFend)
	//			break;
	//		//iter�� pKFend�̸� ���߱�
	//	}
	//	return std::vector<EdgeSLAM::KeyFrame*>(spKFs.begin(),spKFs.end());
	//}
}