#include <CoordinateIntegration.h>
#include <Utils.h>
#include <User.h>
#include <Camera.h>
#include <PlaneEstimator.h>
#include <ContentProcessor.h>
#include <MarkerProcessor.h>

namespace SemanticSLAM {

	ConcurrentMap<int, cv::Mat> CoordinateIntegration::DeviceMaps;
	ConcurrentMap<int, cv::Mat> CoordinateIntegration::IMGs;
	ConcurrentMap<int, cv::Mat> CoordinateIntegration::Ps;
CoordinateIntegration
	::CoordinateIntegration() {
	}
	CoordinateIntegration::~CoordinateIntegration() {
	}
	void CoordinateIntegration::TestImageReturn(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;

		WebAPI API("143.248.6.143", 35005);
		std::stringstream ss;
		ss << "/Load?keyword=TestImage" << "&id=" << id << "&src=" << user;
		auto res = API.Send(ss.str(), "");
		int n2 = res.size();
		cv::Mat temp = cv::Mat::zeros(n2, 1, CV_8UC1);
		std::memcpy(temp.data, res.data(), res.size());
		cv::Mat img = cv::imdecode(temp, cv::IMREAD_COLOR);

		//	IMGs.Update(id, img);

		SLAM->VisualizeImage(img, 0);

		cv::Mat data = cv::Mat::ones(1000, 1, CV_32FC1);
		{
			WebAPI mpAPI("143.248.6.143", 35005);
			std::stringstream ss;
			ss << "/Store?keyword=TestImageReturn&id=" << id << "&src=" << user;
			auto res = mpAPI.Send(ss.str(), data.data, data.rows * sizeof(float));
		}

		pUser->mnUsed--;
	}
	void CoordinateIntegration::DownloadPose(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;

		WebAPI API("143.248.6.143", 35005);
		std::stringstream ss;
		ss << "/Load?keyword=DevicePose" << "&id=" << id << "&src=" << user;
		cv::Mat P = cv::Mat::zeros(4, 3, CV_32FC1);
		auto res2 = API.Send(ss.str(), "");
		std::memcpy(P.data, res2.data(), res2.size());
		
		cv::Mat R = P.rowRange(0, 3).colRange(0, 3);
		cv::Mat t = P.row(3); 
		t = t.t();

		cv::Mat P2 = cv::Mat::zeros(3, 4, CV_32FC1);
		cv::Mat R2 = R.t();
		cv::Mat t2 = -R.t()*t;

		R2.copyTo(P2.rowRange(0, 3).colRange(0, 3));
		t2.copyTo(P2.col(3));
		//유니티와 opencv 좌표계 자체에 대한 변환
		cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
		//T.at<float>(0, 0) = -1.0;
		//T.at<float>(1, 1) = -1.0;
		//T.at<float>(2, 2) = -1.0;
		P2 = P2*T;
		//유니티와 opencv 좌표계 자체에 대한 변환
		Ps.Update(id, P2);
		
		pUser->SetDevicePose(P2);
		cv::Mat P3 = pUser->GetPose();
		cv::Mat R3 = P3.rowRange(0, 3).colRange(0, 3);
		cv::Mat Q1 = Utils::Rot2Quat(R2);
		cv::Mat Q2 = Utils::Rot2Quat(R3);
		std::cout << "Q = " << Q1.t() << "\n" << Q2.t() << std::endl;
		
		//std::cout << P2 << std::endl;

		//while (!Ps.Count(id)) continue;
		while (!IMGs.Count(id)) continue;
		if (!DeviceMaps.Count(0)) {
			pUser->mnUsed--;
			return;
		}

		////cv::Mat P = Ps.Get(id);
		cv::Mat img = IMGs.Get(id);
		cv::Mat MPs = DeviceMaps.Get(0);
		int nMP = MPs.cols;
		cv::Mat K = pUser->GetCameraMatrix();
		cv::Mat proj = K*P2*MPs;
		for (int i = 0; i < nMP; i++) {
			float d = proj.at<float>(2, i);
			cv::Point2f pt(proj.at <float>(0, i) / d, proj.at <float>(1, i) / d);
			//cv::Point2f pt(proj.at <float>(0, i) / d, -pUser->mpCamera->mnHeight + proj.at <float>(1, i) / d);
			//cv::Point2f pt(proj.at <float>(0, i) / d, - proj.at <float>(1, i) / d);
			//cv::Point2f pt(proj.at <float>(0, i) / d, pUser->mpCam era->mnHeight - proj.at <float>(1, i) / d);
			//std::cout <<proj.col(i).t()<<" "<< proj.at <float>(0, i) / d<<" "<< proj.at <float>(1, i) <<" "<< pUser->mpCamera->mnHeight<< std::endl;
			cv::circle(img, pt, 3, cv::Scalar(255, 0, 255), -1);
		}
		SLAM->VisualizeImage(img, 2);
		pUser->mnUsed--;
	}
	void CoordinateIntegration::DownloadImage(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;

		WebAPI API("143.248.6.143", 35005);
		std::stringstream ss;
		ss << "/Load?keyword=Image" << "&id=" << id << "&src=" << user;
		auto res = API.Send(ss.str(), "");
		int n2 = res.size();
		cv::Mat temp = cv::Mat::zeros(n2, 1, CV_8UC1);
		std::memcpy(temp.data, res.data(), res.size());
		cv::Mat img = cv::imdecode(temp, cv::IMREAD_COLOR);
		
		IMGs.Update(id, img);

		pUser->mnUsed--;
	}

	void CoordinateIntegration::Process(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		//이미지랑
		//포즈랑 받아야 함

		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;
		
		WebAPI API("143.248.6.143", 35005);
		std::stringstream ss;
		ss << "/Load?keyword=ARFoundationMPs" << "&id=" << id << "&src=" << user;
		auto res3 = API.Send(ss.str(), "");
		
		cv::Mat totalData = cv::Mat::zeros(res3.size(), 1, CV_8UC1);
		std::memcpy(totalData.data, res3.data(), res3.size());
		float* fdata = (float*)totalData.data;
		int nImageIdx = (int)fdata[0];

		///////이미지
		cv::Mat imgData = cv::Mat::zeros(res3.size() - nImageIdx, 1, CV_8UC1);
		std::memcpy(imgData.data, totalData.data+nImageIdx, res3.size() - nImageIdx);
		cv::Mat img = cv::imdecode(imgData, cv::IMREAD_COLOR);
		//cv::flip(img, img, 0);
		
		//////포즈
		int nInitDataSize = 4;
		int n = (nImageIdx- nInitDataSize) / 12;
		cv::Mat tempData = cv::Mat::zeros(n, 3, CV_32FC1);
		std::memcpy(tempData.data, totalData.data + nInitDataSize, nImageIdx- nInitDataSize);
		cv::Mat Pdown = tempData.rowRange(0, 4);
		cv::Mat R = Pdown.rowRange(0, 3).colRange(0, 3);
		cv::Mat t = Pdown.row(3);
		t = t.t();

		cv::Mat Pinv = cv::Mat::zeros(3, 4, CV_32FC1);
		cv::Mat Rinv = R.t();
		cv::Mat tinv = -R.t()*t;
		Rinv.copyTo(Pinv.rowRange(0, 3).colRange(0, 3));
		tinv.copyTo(Pinv.col(3));

		//cv::Mat E1 = cv::Mat::eye(3, 3, CV_32FC1);
		//cv::Mat E2 = cv::Mat::eye(4, 4, CV_32FC1);
		//E1.at<float>(1, 1) = -1.0;
		//E2.at<float>(1, 1) = -1.0;
		////T.at<float>(0, 0) = -1.0;
		////T.at<float>(1, 1) = -1.0;
		////T.at<float>(2, 2) = -1.0;
		//Pinv = E1*Pinv*E2;
		
		/*tinv.at<float>(0) *= -1.0f;
		tinv.at<float>(1) *= -1.0f;*/

		//////맵포인트
		int nMP = n - 4;
		cv::Mat MPs = tempData.rowRange(4, tempData.rows);//cv::Mat::zeros(nMP, 3, CV_32FC1);//std::memcpy(MPs.data, res3.data(), res3.size());
		/*cv::Mat a = cv::Mat::ones(nMP, 1, CV_32FC1);
		cv::hconcat(MPs, a, MPs);*/
		MPs = MPs.t();

		////시각화
		cv::Mat K = pUser->GetCameraMatrix();
		//cv::Mat proj = K*Pinv*MPs;
		//for (int i = 0; i < nMP; i++) {
		//	float d = proj.at<float>(2, i);
		//	cv::Point2f pt(proj.at <float>(0, i) / d, proj.at <float>(1, i) / d);
		//	//cv::Point2f pt(proj.at <float>(0, i) / d, pUser->mpCamera->mnHeight -proj.at <float>(1, i) / d);
		//	//std::cout <<proj.col(i).t()<<" "<< proj.at <float>(0, i) / d<<" "<< proj.at <float>(1, i) <<" "<< pUser->mpCamera->mnHeight<< std::endl;
		//	cv::circle(img, pt, 3, cv::Scalar(255, 0, 255), -1);
		//}

		////////테스트용 대응쌍 저장
		//std::stringstream ssPath;
		//ssPath << "../bin/data/correspondences/";
		//{
		//	//K
		//	std::stringstream ss;
		//	ss << ssPath.str()<<"intrinsic.txt";
		//	std::ofstream f;
		//	f.open(ss.str().c_str());
		//	f << K.at<float>(0, 0) << " " << K.at<float>(1, 1) << " " << K.at<float>(0, 2) << " " << K.at<float>(1, 2);
		//	f.close();
		//}
		//{
		//	//Pose
		//	std::stringstream ss;
		//	ss << ssPath.str() << id<<"_pose.txt";
		//	std::ofstream f;
		//	f.open(ss.str().c_str());
		//	f << R.at<float>(0, 0) << " " << R.at<float>(0, 1) << " " << R.at<float>(0, 2)<<" ";
		//	f << R.at<float>(1, 0) << " " << R.at<float>(1, 1) << " " << R.at<float>(1, 2) << " ";
		//	f << R.at<float>(2, 0) << " " << R.at<float>(2, 1) << " " << R.at<float>(2, 2) << " ";
		//	f << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2);
		//	f.close();
		//}
		////대응쌍
		//std::stringstream ssc;
		//ssc << ssPath.str() << id << "_corres.txt";
		//std::ofstream f;
		//f.open(ssc.str().c_str());
		///////////////////////////
		cv::Mat proj = K*MPs;
		for (int i = 0; i < nMP; i++) {
			float d = proj.at<float>(2, i);
			cv::Point2f pt(proj.at <float>(0, i) / d, proj.at <float>(1, i) / d);
			//cv::Point2f pt(proj.at <float>(0, i) / d, pUser->mpCamera->mnHeight -proj.at <float>(1, i) / d);
			//std::cout <<proj.col(i).t()<<" "<< proj.at <float>(0, i) / d<<" "<< proj.at <float>(1, i) <<" "<< pUser->mpCamera->mnHeight<< std::endl;
			cv::circle(img, pt, 3, cv::Scalar(255, 0, 255), -1);
			////MP 저장
			////f << pt.x << " " << pt.y << " " << MPs.at<float>(0, i) << " " << MPs.at<float>(1, i) << " " << MPs.at<float>(2, i) << std::endl;
		}
		//f.close();
		
		//////////////////////////
		////////////////////////
		SLAM->VisualizeImage(img, 2);
		//////////////////////////////


		//int n = res3.size() / 12; //4줄은 포즈임.
		//cv::Mat data = cv::Mat::zeros(n, 3, CV_32FC1);
		//std::memcpy(data.data, res3.data(), res3.size());
		//		
		//
		//
		

		//cv::Mat Pinv = cv::Mat::zeros(3, 4, CV_32FC1);
		//cv::Mat Rinv = R.t();
		//cv::Mat tinv = -R.t()*t;

		//Rinv.copyTo(Pinv.rowRange(0, 3).colRange(0, 3));
		//tinv.copyTo(Pinv.col(3));
		////유니티와 opencv 좌표계 자체에 대한 변환
		//cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
		////T.at<float>(0, 0) = -1.0;
		////T.at<float>(1, 1) = -1.0;
		////T.at<float>(2, 2) = -1.0;
		//Pinv = Pinv*T;

		//
		//cv::Mat MPs = data.rowRange(4, data.rows);//cv::Mat::zeros(nMP, 3, CV_32FC1);//std::memcpy(MPs.data, res3.data(), res3.size());
		//cv::Mat a = cv::Mat::ones(nMP, 1, CV_32FC1);
		//cv::hconcat(MPs, a, MPs);
		//MPs = MPs.t();
		//DeviceMaps.Update(0, MPs);
		//
		//cv::Mat T2 = cv::Mat::eye(4, 4, CV_32FC1);
		////T2.at<float>(0, 0) = -1.0;
		////T2.at<float>(1, 1) = -1.0;
		////T2.at<float>(2, 2) = -1.0;
		//MPs = T2*MPs;

		//while (!IMGs.Count(id)) continue;
		//
		//cv::Mat img = IMGs.Get(id);
		//cv::flip(img, img, 0);

		////std::cout << R << t << nMP << std::endl;

		///*std::map<int, cv::Mat> ARFoundationMPs;
		//if (SLAM->TemporalDatas2.Count("ARFoundationMPs")){
		//	ARFoundationMPs = SLAM->TemporalDatas2.Get("ARFoundationMPs");
		//	auto NewMPs = ARFoundationMPs[0];
		//	cv::hconcat(MPs,NewMPs, MPs);
		//	ARFoundationMPs[0] = MPs;
		//}
		//ARFoundationMPs[0] = MPs;
		//SLAM->TemporalDatas2.Update("ARFoundationMPs", ARFoundationMPs);*/




		////while (!Ps.Count(id)) continue;
		////while (!IMGs.Count(id)) continue;

		////cv::Mat P = Ps.Get(id);
		////cv::Mat img = IMGs.Get(id);
		////
		//cv::Mat K = pUser->GetCameraMatrix();
		//cv::Mat proj = K*Pinv*MPs;
		//for (int i = 0; i < nMP; i++) {
		//	float d = proj.at<float>(2, i);
		//	cv::Point2f pt(proj.at <float>(0, i) / d, proj.at <float>(1, i) / d);
		//	//cv::Point2f pt(proj.at <float>(0, i) / d, pUser->mpCamera->mnHeight -proj.at <float>(1, i) / d);
		//	//std::cout <<proj.col(i).t()<<" "<< proj.at <float>(0, i) / d<<" "<< proj.at <float>(1, i) <<" "<< pUser->mpCamera->mnHeight<< std::endl;
		//	cv::circle(img, pt, 3, cv::Scalar(255, 0, 255), -1);
		//}
		////std::cout << "end " <<nMP<< std::endl;
		//SLAM->VisualizeImage(img, 2);

		//std::cout << "좌표계 통합 " << P << std::endl << (int)data.at<float>(0) << std::endl;
		////데이터 생성
		
		////시각화


		pUser->mnUsed--;
	}
}