#include "GridProcessor.h"
#include "Utils.h"
#include <User.h>
#include <KeyFrame.h>
#include <LabelInfo.h>
#include <SemanticProcessor.h>
#include <PlaneEstimator.h>

namespace SemanticSLAM {
	float GridProcessor::GridSize = 0.1;
	std::atomic<int> GridProcessor::nGridID = 0;
	ConcurrentMap<int, ConcurrentMap<int, ConcurrentMap<int, Grid*>*>*> GridProcessor::GlobalGrids;
	ConcurrentMap<EdgeSLAM::KeyFrame*, std::set<Grid*>> GridProcessor::GlobalKeyFrameNGrids;

	Grid::Grid():mnID(++GridProcessor::nGridID),Floor(nullptr){}
	Grid::Grid(int x, int y, int z, float gsize):Grid(){
		float gx = x * gsize;
		float gy = y * gsize;
		float gz = z * gsize;
		pos = (cv::Mat_<float>(3, 1) <<  gx, gy, gz);
	}
	Grid::~Grid() {}

	void GridProcessor::ConvertIndex(cv::Mat X, int& xidx, int& yidx, int& zidx) {
		float x = X.at<float>(0);
		float y = X.at<float>(1);
		float z = X.at<float>(2);

		xidx = (int)(x / GridSize);
		yidx = (int)(y / GridSize);
		zidx = (int)(z / GridSize);
	}

	Grid* GridProcessor::SearchGrid(int xidx, int yidx, int zidx) {
		Grid* g = nullptr;
		if (GlobalGrids.Count(xidx)) {
			auto a = GlobalGrids.Get(xidx);
			if (a->Count(yidx)) {
				auto b = a->Get(yidx);
				if (b->Count(zidx)) {
					g = b->Get(zidx);
				}
				else {
					g = new Grid(xidx, yidx, zidx, GridSize);
					b->Update(zidx, g);
				}
			}
			else {
				auto b = new ConcurrentMap<int, Grid*>();
				g = new Grid(xidx, yidx, zidx, GridSize);
				b->Update(zidx, g);
				a->Update(yidx, b);
			}
		}
		else {
			//나머지도 없다는 것임.
			auto a = new ConcurrentMap<int, ConcurrentMap<int, SemanticSLAM::Grid*>*>();
			auto b = new ConcurrentMap<int, Grid*>();
			g = new Grid(xidx, yidx, zidx, GridSize);
			b->Update(zidx, g);
			a->Update(yidx, b);
			GlobalGrids.Update(xidx, a);
		}
		return g;
	}

	std::vector<cv::Point2f> GridProcessor::ProjectdGrid(int x, int y, int z, float gsize, cv::Mat K, cv::Mat R, cv::Mat t) {
		std::vector<cv::Point2f> res;
		res.push_back(ProjectCorner(x,   y,   z, gsize, K, R, t));
		res.push_back(ProjectCorner(x,   y, z+1, gsize, K, R, t));
		res.push_back(ProjectCorner(x+1, y,   z, gsize, K, R, t));
		res.push_back(ProjectCorner(x+1, y, z+1, gsize, K, R, t));
		return res;
	}

	cv::Point2f GridProcessor::ProjectCorner(int x, int y, int z, float gsize, cv::Mat K, cv::Mat R, cv::Mat t) {
		float gx = x * gsize;
		float gy = y * gsize;
		float gz = z * gsize;
		cv::Mat pos = (cv::Mat_<float>(3, 1) << gx, gy, gz);
		cv::Mat Xcam = R * pos + t;
		cv::Mat Ximg = K * Xcam;
		float depth = Ximg.at<float>(2);
		cv::Point2f pt(Ximg.at<float>(0) / depth, Ximg.at<float>(1) / depth);
		return pt;
	}

	void GridProcessor::CalcGridWithKF(EdgeSLAM::SLAM* SLAM, EdgeSLAM::KeyFrame* pKF) {
		//평면 회전 후 동작
		int inc = 5;
		Plane* Floor = nullptr;
		if (PlaneEstimator::GlobalFloor)
		{
			Floor = PlaneEstimator::GlobalFloor;
			if (Floor->nScore <= 0)
				return;
		}
		
		if (!SemanticProcessor::GraphKFNLabel.Count(pKF))
			return;
		cv::Mat labeled = SemanticProcessor::GraphKFNLabel.Get(pKF);
		//std::cout << "????????" << std::endl;
		cv::Mat Tinv = pKF->GetPoseInverse();
		cv::Mat Kinv = pKF->K.inv();
		cv::Mat O = pKF->GetCameraCenter();

		cv::Mat K = pKF->K.clone();
		cv::Mat R = pKF->GetRotation();
		cv::Mat t = pKF->GetTranslation();

		std::map<int, cv::Mat> mapDatas;
		if (SLAM->TemporalDatas2.Count("dynamic2")) {
			mapDatas = SLAM->TemporalDatas2.Get("dynamic2");
		}

		std::set<Grid*> setGrids;

		std::vector<std::vector<cv::Point2f>> vecProjectedCorners;
		for (int y = 0; y < labeled.rows; y += inc) {
			for (int x = 0; x < labeled.cols; x += inc) {
				int label = labeled.at<uchar>(y, x) + 1;
				if (label == (int)StructureLabel::FLOOR) {
					//Grid 만들기
					//포즈와 평면으로 3차원 복원
					//3차원 위치에서 그리드 선택을 위한 인덱스화
					//그리드 안에 현재 프레임 연결
					cv::Mat X;
					int xidx, yidx, zidx;
					if (Floor->CalcPosition(X, x, y, Kinv, Tinv, O)) {
						ConvertIndex(X, xidx, yidx, zidx);
						//Search Grid

						auto pGrid = SearchGrid(xidx, yidx, zidx);
						if (!pGrid)
							continue;
						
						//그리드 시각화용
						auto corners = ProjectdGrid(xidx, yidx, zidx, GridSize, K, R, t);
						vecProjectedCorners.push_back(corners);
						//그리드 시각화용
						if (setGrids.count(pGrid))
							continue;
						setGrids.insert(pGrid);
						if (!pGrid->ConnectedKFs.Count(pKF)) {
							pGrid->ConnectedKFs.Update(pKF);
							mapDatas[pGrid->mnID] = X;
						}
						
						
						//std::cout << X << " " << xidx << " " << yidx << " " << zidx << " " << pGrid->ConnectedKFs.Size() << std::endl;
					}

				}//label
			}//for
		}//for

		//그리드 시각화
		//cv::Mat gridImage = cv::Mat::zeros(labeled.size(), CV_8UC3);
		//for (int i = 0, iend = vecProjectedCorners.size(); i < iend; i++) {
		//	auto pt1 = vecProjectedCorners[i][0];
		//	auto pt2 = vecProjectedCorners[i][1];
		//	auto pt3 = vecProjectedCorners[i][2];
		//	auto pt4 = vecProjectedCorners[i][3];

		//	cv::line(gridImage, pt1, pt2, cv::Scalar(255, 255, 0));
		//	cv::line(gridImage, pt1, pt3, cv::Scalar(255, 255, 0));
		//	cv::line(gridImage, pt4, pt2, cv::Scalar(255, 255, 0));
		//	cv::line(gridImage, pt4, pt3, cv::Scalar(255, 255, 0));
		//}
		//cv::imshow("asdfasdfasdf", gridImage);
		//cv::waitKey(0);
		////그리드 시각화
		GlobalKeyFrameNGrids.Update(pKF, setGrids);
		SLAM->TemporalDatas2.Update("dynamic2", mapDatas);

	}
	void GridProcessor::CalcGrid(EdgeSLAM::SLAM* SLAM, std::string user, int id, cv::Mat labeled) {

		int inc = 5;
		Plane* Floor = nullptr;
		if (PlaneEstimator::GlobalFloor)
		{
			Floor = PlaneEstimator::GlobalFloor;
			if (Floor->nScore <= 0)
				return;
		}
		else {
			return;
		}
		
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		if (!pUser->KeyFrames.Count(id)) {
			return;
		}
		auto pKF = pUser->KeyFrames.Get(id);
		if (!pKF)
			return;

		cv::Mat Tinv = pKF->GetPoseInverse();
		cv::Mat Kinv = pKF->K.inv();
		cv::Mat O = pKF->GetCameraCenter();

		std::map<int, cv::Mat> mapDatas;
		if (SLAM->TemporalDatas2.Count("dynamic2")) {
			mapDatas = SLAM->TemporalDatas2.Get("dynamic2");
		}

		std::set<Grid*> setGrids;
		for (int y = 0; y < labeled.rows; y+=inc) {
			for (int x = 0; x < labeled.cols; x+=inc) {
				int label = labeled.at<uchar>(y, x) + 1;
				if (label == (int)StructureLabel::FLOOR) {
					//Grid 만들기
					//포즈와 평면으로 3차원 복원
					//3차원 위치에서 그리드 선택을 위한 인덱스화
					//그리드 안에 현재 프레임 연결
					cv::Mat X;
					int xidx, yidx, zidx;
					if (Floor->CalcPosition(X, x, y, Kinv, Tinv, O)) {
						ConvertIndex(X, xidx, yidx, zidx);
						//Search Grid

						auto pGrid = SearchGrid(xidx, yidx, zidx);
						if (!pGrid)
							continue;
						if (setGrids.count(pGrid))
							continue;
						setGrids.insert(pGrid);
						if (!pGrid->ConnectedKFs.Count(pKF)){
							pGrid->ConnectedKFs.Update(pKF);
							mapDatas[pGrid->mnID] = X;
						}
						//std::cout << X << " " << xidx << " " << yidx << " " << zidx << " " << pGrid->ConnectedKFs.Size() << std::endl;
					}
					
				}//label
			}//for
		}//for
		GlobalKeyFrameNGrids.Update(pKF, setGrids);
		SLAM->TemporalDatas2.Update("dynamic2", mapDatas);
		
	}

	Grid* GridProcessor::GetGrid(cv::Mat X){
		int xidx, yidx, zidx;
		ConvertIndex(X, xidx, yidx, zidx);
		//std::cout << "grid " << xidx << " " << yidx << " " <<zidx<< std::endl;
		auto pGrid = SearchGrid(xidx, yidx, zidx);
		return pGrid;
	}

	void GridProcessor::GridTest(EdgeSLAM::SLAM* SLAM, std::string user, int id, const cv::Mat& _img, const cv::Mat& _T, const cv::Mat& _invK) {
		cv::Mat img = _img.clone();
		cv::Mat T = _T.clone();
		cv::Mat invK = _invK.clone();
		//for(int x = )

		cv::Mat P1 = invK * (cv::Mat_<float>(3,1) << 0, 0,1);
		cv::Mat P2 = invK * (cv::Mat_<float>(3,1) << img.cols, 0, 1);
		cv::Mat P3 = invK * (cv::Mat_<float>(3,1) << 0, img.rows, 1);
		cv::Mat P4 = invK * (cv::Mat_<float>(3,1) << img.cols, img.rows, 1);
		cv::Mat P5 = 3.0 * invK * (cv::Mat_<float>(3,1) << 0, 0, 1);
		cv::Mat P6 = 3.0 * invK * (cv::Mat_<float>(3,1) << img.cols, 0, 1);
		cv::Mat P7 = 3.0 * invK * (cv::Mat_<float>(3,1) << 0, img.rows, 1);
		cv::Mat P8 = 3.0 * invK * (cv::Mat_<float>(3,1) << img.cols, img.rows, 1);

		{
			cv::Mat R = T.colRange(0, 3).rowRange(0, 3);
			cv::Mat t = T.rowRange(0, 3).col(3);
			
			cv::Mat Rinv = R.t();
			cv::Mat tinv = -Rinv * t;
			
			P1 = Rinv * P1 + tinv;
			P2 = Rinv * P2 + tinv;
			P3 = Rinv * P3 + tinv;
			P4 = Rinv * P4 + tinv;
			P5 = Rinv * P5 + tinv;
			P6 = Rinv * P6 + tinv;
			P7 = Rinv * P7 + tinv;
			P8 = Rinv * P8 + tinv;

			std::map<int, cv::Mat> mapDatas;
			mapDatas[0] = P1;
			mapDatas[1] = P2;
			mapDatas[2] = P3;
			mapDatas[3] = P4;
			mapDatas[4] = P5;
			mapDatas[5] = P6;
			mapDatas[6] = P7;
			mapDatas[7] = P8;
			SLAM->TemporalDatas2.Update("view", mapDatas);

			/*std::cout << "4" << std::endl;
			if (SLAM->TemporalDatas2.Count("view")) {
				mapDatas = SLAM->TemporalDatas2.Get("view");
				mapDatas.clear();
			}
			std::cout << "5" << std::endl;*/
		}

		/*cv::Mat V1 = P2 - P1; 
		V1 /= sqrt(V1.dot(V1));

		cv::Mat V2 = P3 - P1;
		V2 /= sqrt(V2.dot(V2));*/

	}

	int testID = 0;
	void GridProcessor::GridTest2(EdgeSLAM::SLAM* SLAM, std::string user, int id, const cv::Mat& _img, const cv::Mat& _T, const cv::Mat& _invK, int objID, cv::Point2f pt1, cv::Point2f pt4) {
		cv::Mat img = _img.clone();
		cv::Mat T = _T.clone();
		cv::Mat invK = _invK.clone();
		
		float lx = pt1.x;
		float ly = pt1.y;
		float rx = pt4.x;
		float ry = pt4.y;

		cv::Mat P5 = 3.0 * invK * (cv::Mat_<float>(3, 1) << lx, ly, 1);
		cv::Mat P6 = 3.0 * invK * (cv::Mat_<float>(3, 1) << rx,ly, 1);
		cv::Mat P7 = 3.0 * invK * (cv::Mat_<float>(3, 1) << lx,ry, 1);
		cv::Mat P8 = 3.0 * invK * (cv::Mat_<float>(3, 1) << rx,ry, 1);

		{
			cv::Mat R = T.colRange(0, 3).rowRange(0, 3);
			cv::Mat t = T.rowRange(0, 3).col(3);

			cv::Mat Rinv = R.t();
			cv::Mat tinv = -Rinv * t; //cam center

			P5 = Rinv * P5 + tinv;
			P6 = Rinv * P6 + tinv;
			P7 = Rinv * P7 + tinv;
			P8 = Rinv * P8 + tinv;

			cv::Mat temp = cv::Mat::zeros(0, 3, CV_32FC1);
			temp.push_back(tinv.t());
			temp.push_back(P5.t());
			temp.push_back(P6.t());
			temp.push_back(P7.t());
			temp.push_back(P8.t());

			std::map<int, cv::Mat> mapDatas;
			if (SLAM->TemporalDatas2.Count("view2")) {
				mapDatas = SLAM->TemporalDatas2.Get("view2");
			}
			mapDatas[++testID] = temp;
			SLAM->TemporalDatas2.Update("view2", mapDatas);

			/*std::cout << "4" << std::endl;
			
			std::cout << "5" << std::endl;*/
		}

		/*cv::Mat V1 = P2 - P1;
		V1 /= sqrt(V1.dot(V1));

		cv::Mat V2 = P3 - P1;
		V2 /= sqrt(V2.dot(V2));*/

	}
}  