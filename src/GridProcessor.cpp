#include "GridProcessor.h"
#include "Utils.h"
#include <User.h>
#include <KeyFrame.h>

namespace SemanticSLAM {
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