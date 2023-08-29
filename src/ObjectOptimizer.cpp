#include <ObjectOptimizer.h>
#include <SLAM.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <Map.h>
#include <MapPoint.h>
#include <Converter.h>
#include <LoopCloser.h>
#include <ObjectFrame.h>

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/linear_solver_dense.h"
#include "g2o/types/types_seven_dof_expmap.h"

namespace SemanticSLAM {
	int ObjectOptimizer::ObjectPoseInitialization(EdgeSLAM::ObjectBoundingBox* pBox, const cv::Mat& K, const cv::Mat& D, cv::Mat& P, int count, float err, float confidence, int method, cv::Mat& inliers) {
		cv::Mat _K = K.clone();
		cv::Mat _D = D.clone();
		//_K.convertTo(_K, CV_64FC1);
		//_D.convertTo(_D, CV_64FC1);
		std::vector<cv::Point2d> imagePoints;
		std::vector<cv::Point3d> objectPoints;
		cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
		cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
		for (int i = 0; i < pBox->N; i++) {
			auto pMPi = pBox->mvpMapPoints.get(i);
			if (!pMPi || pMPi->isBad() || pBox->mvbOutliers[i])
				continue;
			auto imgPt = pBox->mvKeys[i].pt;
			cv::Mat Xo = pMPi->GetWorldPos();
			cv::Point3d objPt(Xo.at<float>(0), Xo.at<float>(1), Xo.at<float>(2));
			imagePoints.push_back(imgPt);
			objectPoints.push_back(objPt);
		}
		bool bPnP = cv::solvePnPRansac(objectPoints, imagePoints, K, D, rvec, tvec, false, count, err, confidence, inliers, method);
		cv::Mat R, t;
		cv::Rodrigues(rvec, R);
		R.convertTo(R, CV_32FC1);
		tvec.convertTo(t, CV_32FC1);
		R.copyTo(P.rowRange(0, 3).colRange(0, 3));
		t.copyTo(P.col(3).rowRange(0, 3));

	}
	int ObjectOptimizer::ObjectPoseOptimization(EdgeSLAM::Frame* pFrame, cv::Mat& P) {
		g2o::SparseOptimizer optimizer;
		g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

		linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

		g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		int nInitialCorrespondences = 0;

		// Set Frame vertex
		P = cv::Mat::eye(4, 4, CV_32FC1);
		g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(EdgeSLAM::Converter::toSE3Quat(P));
		vSE3->setId(0);
		vSE3->setFixed(false);
		optimizer.addVertex(vSE3);

		// Set MapPoint vertices
		int N = pFrame->N;

		std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
		std::vector<size_t> vnIndexEdgeMono;
		vpEdgesMono.reserve(N);
		vnIndexEdgeMono.reserve(N);


		float th = 10.0;//5.991;
		float deltaMono = sqrt(th);

		auto vecMPs = pFrame->mvpMapPoints;
		float Na = 0;
		cv::Mat avgPos = cv::Mat::zeros(3, 1, CV_32FC1);
		{
			/*for (int i = 0; i < N; i++) {

				auto pMP = vecMPs[i];
				if (!pMP || pMP->isBad())
					continue;
				Na++;
				avgPos += pMP->GetWorldPos();
			}
			avgPos /= Na;*/

			//std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);
			for (int i = 0; i < N; i++) {

				auto pMP = vecMPs[i];
				if (!pMP || pMP->isBad())
					continue;
				const cv::KeyPoint& kp = pFrame->mvKeysUn[i];
				nInitialCorrespondences++;
				pFrame->mvbOutliers[i] = false;

				Eigen::Matrix<double, 2, 1> obs;
				obs << kp.pt.x, kp.pt.y;

				g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
				e->setMeasurement(obs);
				float invSigma2 = pFrame->mvInvLevelSigma2[kp.octave];
				e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);
				rk->setDelta(deltaMono);

				e->fx = pFrame->fx;
				e->fy = pFrame->fy;
				e->cx = pFrame->cx;
				e->cy = pFrame->cy;
				cv::Mat Xw = pMP->GetWorldPos();
				e->Xw[0] = Xw.at<float>(0);
				e->Xw[1] = Xw.at<float>(1);
				e->Xw[2] = Xw.at<float>(2);

				optimizer.addEdge(e);
				vpEdgesMono.push_back(e);
				vnIndexEdgeMono.push_back(i);
			}
		}

		if (nInitialCorrespondences < 3) {
			//delete linearSolver;
			//delete vSE3;
			return 0;
		}

		float chi2Mono[4] = { th,th,th,th };
		int its[4] = { 10,10,10,10 };

		int nBad = 0;
		for (size_t it = 0; it < 4; it++)
		{

			vSE3->setEstimate(EdgeSLAM::Converter::toSE3Quat(P));
			optimizer.initializeOptimization(0);

			/*g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
			g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
			cv::Mat pose = EdgeSLAM::Converter::toCvMat(SE3quat_recov);*/

			optimizer.optimize(its[it]);

			nBad = 0;
			for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
			{
				g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

				size_t idx = vnIndexEdgeMono[i];

				if (pFrame->mvbOutliers[idx])
				{
					e->computeError();
				}

				float chi2 = e->chi2();

				if (chi2 > chi2Mono[it])
				{
					pFrame->mvbOutliers[idx] = true;
					e->setLevel(1);
					nBad++;
				}
				else
				{
					pFrame->mvbOutliers[idx] = false;
					e->setLevel(0);
				}

				if (it == 2)
					e->setRobustKernel(0);
			}
			if (nBad == nInitialCorrespondences)
				break;
			//std::cout << "Object pOse opti test = " << nBad << " " << nInitialCorrespondences << " " << optimizer.edges().size() << std::endl;
			if (optimizer.edges().size() < 10)
				break;
		}

		// Recover optimized pose and return number of inliers
		g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
		g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
		cv::Mat pose = EdgeSLAM::Converter::toCvMat(SE3quat_recov);
		P = pose.clone();
		//std::cout << "Test obj pose = " << pose <<avgPos.t()<< std::endl;

		return nInitialCorrespondences - nBad;
	}
	
	int ObjectOptimizer::ObjectPoseOptimization(EdgeSLAM::ObjectBoundingBox* pBox, cv::Mat& P, cv::Mat origin) {
		g2o::SparseOptimizer optimizer;
		g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

		linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

		g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		int nInitialCorrespondences = 0;

		// Set Frame vertex
		//P = cv::Mat::eye(4, 4, CV_32FC1);
		g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(EdgeSLAM::Converter::toSE3Quat(P));
		vSE3->setId(0);
		vSE3->setFixed(false);
		optimizer.addVertex(vSE3);

		// Set MapPoint vertices
		int N = pBox->N;

		std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
		std::vector<size_t> vnIndexEdgeMono;
		vpEdgesMono.reserve(N);
		vnIndexEdgeMono.reserve(N);

		float th = 10.0;//5.991;
		float deltaMono = sqrt(th);

		auto vecMPs = pBox->mvpMapPoints.get();
		float Na = 0;
		cv::Mat avgPos = cv::Mat::zeros(3, 1, CV_32FC1);
		{
			/*for (int i = 0; i < N; i++) {

				auto pMP = vecMPs[i];
				if (!pMP || pMP->isBad())
					continue;
				Na++;
				avgPos += pMP->GetWorldPos();
			}
			avgPos /= Na;*/

			//std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);
			for (int i = 0; i < N; i++) {

				auto pMP = vecMPs[i];
				if (!pMP || pMP->isBad())
					continue;
				const cv::KeyPoint& kp = pBox->mvKeys[i];
				nInitialCorrespondences++;
				pBox->mvbOutliers[i] = false;

				Eigen::Matrix<double, 2, 1> obs;
				obs << kp.pt.x, kp.pt.y;

				g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
				e->setMeasurement(obs);
				float invSigma2 = pBox->mvInvLevelSigma2[kp.octave];
				e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);
				rk->setDelta(deltaMono);

				e->fx = pBox->fx;
				e->fy = pBox->fy;
				e->cx = pBox->cx;
				e->cy = pBox->cy;
				cv::Mat Xw = pMP->GetWorldPos()-origin;
				e->Xw[0] = Xw.at<float>(0);
				e->Xw[1] = Xw.at<float>(1);
				e->Xw[2] = Xw.at<float>(2);

				optimizer.addEdge(e);
				vpEdgesMono.push_back(e);
				vnIndexEdgeMono.push_back(i);
			}
		}

		if (nInitialCorrespondences < 3) {
			//delete linearSolver;
			//delete vSE3;
			return 0;
		}

		float chi2Mono[4] = { th,th,th,th };
		int its[4] = { 10,10,10,10 };

		int nBad = 0;
		for (size_t it = 0; it < 4; it++)
		{

			vSE3->setEstimate(EdgeSLAM::Converter::toSE3Quat(P));
			optimizer.initializeOptimization(0);

			/*g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
			g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
			cv::Mat pose = EdgeSLAM::Converter::toCvMat(SE3quat_recov);*/

			optimizer.optimize(its[it]);

			nBad = 0;
			for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
			{
				g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

				size_t idx = vnIndexEdgeMono[i];

				if (pBox->mvbOutliers[idx])
				{
					e->computeError();
				}

				float chi2 = e->chi2();

				if (chi2 > chi2Mono[it])
				{
					pBox->mvbOutliers[idx] = true;
					e->setLevel(1);
					nBad++;
				}
				else
				{
					pBox->mvbOutliers[idx] = false;
					e->setLevel(0);
				}

				if (it == 2)
					e->setRobustKernel(0);
			}
			if (nBad == nInitialCorrespondences)
				break;
			//std::cout << "Object pOse opti test = " << nBad << " " << nInitialCorrespondences << " " << optimizer.edges().size() << std::endl;
			if (optimizer.edges().size() < 10)
				break;
		}

		// Recover optimized pose and return number of inliers
		g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
		g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
		cv::Mat pose = EdgeSLAM::Converter::toCvMat(SE3quat_recov);
		P = pose.clone();
		//std::cout << "Test obj pose = " << pose <<avgPos.t()<< std::endl;

		return nInitialCorrespondences - nBad;
	}
	
	void ObjectOptimizer::ObjectMapAdjustment(EdgeSLAM::ObjectNode* pObjMap) {
		auto spOPs = pObjMap->mspOPs.Get();
		auto spBBs = pObjMap->mspBBs.Get();
		std::set<EdgeSLAM::KeyFrame*> spKFs;
		std::list<EdgeSLAM::ObjectMapPoint*> lLocalObjectPoints = std::list<EdgeSLAM::ObjectMapPoint*>(spOPs.begin(), spOPs.end());
		std::list<EdgeSLAM::KeyFrame*> lLocalKeyFrames;
		for (auto iter = spBBs.begin(), iend = spBBs.end(); iter != iend; iter++) {
			auto pBB = *iter;
			auto pKF = pBB->mpKF;
			if (!spKFs.count(pKF))
				spKFs.insert(pKF);
		}
		std::list<EdgeSLAM::KeyFrame*> lFixedCameras = std::list<EdgeSLAM::KeyFrame*>(spKFs.begin(), spKFs.end());

		g2o::SparseOptimizer optimizer;
		g2o::BlockSolver_6_3::LinearSolverType* linearSolver;

		linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

		g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		/*if (pbStopFlag)
			optimizer.setForceStopFlag(pbStopFlag);*/

		unsigned long maxKFid = 0;

		// Set Local KeyFrame vertices
		for (std::list<EdgeSLAM::KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
		{
			EdgeSLAM::KeyFrame* pKFi = *lit;
			g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
			vSE3->setEstimate(EdgeSLAM::Converter::toSE3Quat(pKFi->GetPose()));
			vSE3->setId(pKFi->mnId);
			vSE3->setFixed(pKFi->mnId == 0);
			optimizer.addVertex(vSE3);
			if (pKFi->mnId > maxKFid)
				maxKFid = pKFi->mnId;
		}

		// Set Fixed KeyFrame vertices
		for (std::list<EdgeSLAM::KeyFrame*>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
		{
			EdgeSLAM::KeyFrame* pKFi = *lit;
			g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
			vSE3->setEstimate(EdgeSLAM::Converter::toSE3Quat(pKFi->GetPose()));
			vSE3->setId(pKFi->mnId);
			vSE3->setFixed(true);
			optimizer.addVertex(vSE3);
			if (pKFi->mnId > maxKFid)
				maxKFid = pKFi->mnId;
		}

		// Set MapPoint vertices
		const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalObjectPoints.size();

		std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
		vpEdgesMono.reserve(nExpectedSize);

		std::vector<EdgeSLAM::ObjectBoundingBox*> vpEdgeBoxMono;
		vpEdgeBoxMono.reserve(nExpectedSize);

		std::vector<EdgeSLAM::ObjectMapPoint*> vpMapPointEdgeMono;
		vpMapPointEdgeMono.reserve(nExpectedSize);

		const float thHuberMono = sqrt(5.991);

		for (std::list<EdgeSLAM::ObjectMapPoint*>::iterator lit = lLocalObjectPoints.begin(), lend = lLocalObjectPoints.end(); lit != lend; lit++)
		{
			//EdgeSLAM::ObjectMapPoint* pMP = *lit;
			//g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
			//vPoint->setEstimate(EdgeSLAM::Converter::toVector3d(pMP->GetWorldPos()));
			//int id = pMP->mnId + maxKFid + 1;
			//vPoint->setId(id);
			//vPoint->setMarginalized(true);
			//optimizer.addVertex(vPoint);

			//const std::map<EdgeSLAM::ObjectBoundingBox*, size_t> observations = pMP->GetObservations();

			////Set edges
			//for (std::map<EdgeSLAM::ObjectBoundingBox*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			//{
			//	auto pBBi = mit->first;
			//	auto pKFi = pBBi->mpKF;
			//	if (!pKFi->isBad())
			//	{
			//		const cv::KeyPoint& kpUn = pBBi->mvKeys[mit->second];

			//		Eigen::Matrix<double, 2, 1> obs;
			//		obs << kpUn.pt.x, kpUn.pt.y;

			//		g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

			//		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
			//		e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
			//		e->setMeasurement(obs);
			//		const float& invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
			//		e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

			//		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			//		e->setRobustKernel(rk);
			//		rk->setDelta(thHuberMono);

			//		e->fx = pKFi->fx;
			//		e->fy = pKFi->fy;
			//		e->cx = pKFi->cx;
			//		e->cy = pKFi->cy;

			//		optimizer.addEdge(e);
			//		vpEdgesMono.push_back(e);
			//		vpEdgeBoxMono.push_back(pBBi);
			//		vpMapPointEdgeMono.push_back(pMP);
			//	}
			//}
		}

		/*if (pbStopFlag)
			if (*pbStopFlag)
				return;*/

		optimizer.initializeOptimization();
		optimizer.optimize(5);

		bool bDoMore = true;

		/*if (pbStopFlag)
			if (*pbStopFlag)
				bDoMore = false;*/

		//if (bDoMore)
		//{

		//	// Check inlier observations
		//	for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
		//	{
		//		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
		//		EdgeSLAM::ObjectMapPoint* pMP = vpMapPointEdgeMono[i];

		//		if (pMP->isBad())
		//			continue;

		//		if (e->chi2() > 5.991 || !e->isDepthPositive())
		//		{
		//			e->setLevel(1);
		//		}

		//		e->setRobustKernel(0);
		//	}
		//	// Optimize again without the outliers
		//	optimizer.initializeOptimization(0);
		//	optimizer.optimize(10);

		//}

		//std::vector<std::pair<EdgeSLAM::ObjectBoundingBox*, EdgeSLAM::ObjectMapPoint*> > vToErase;
		//vToErase.reserve(vpEdgesMono.size());

		//// Check inlier observations       
		//for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
		//{
		//	g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
		//	EdgeSLAM::ObjectMapPoint* pOP = vpMapPointEdgeMono[i];

		//	if (pOP->isBad())
		//		continue;

		//	if (e->chi2() > 5.991 || !e->isDepthPositive())
		//	{
		//		EdgeSLAM::ObjectBoundingBox* pBBi = vpEdgeBoxMono[i];
		//		vToErase.push_back(std::make_pair(pBBi, pOP));
		//	}
		//}

		//// Get Map Mutex
		////std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

		//if (!vToErase.empty())
		//{
		//	for (size_t i = 0; i < vToErase.size(); i++)
		//	{
		//		auto pKFi = vToErase[i].first;
		//		auto pMPi = vToErase[i].second;
		//		pKFi->EraseObjectPointMatch(pMPi);
		//		pMPi->EraseObservation(pKFi);
		//	}
		//}

		// Recover optimized data

		//Keyframes
		/*for (std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
		{
			KeyFrame* pKF = *lit;
			g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
			g2o::SE3Quat SE3quat = vSE3->estimate();
			pKF->SetPose(Converter::toCvMat(SE3quat));
		}*/

		//Points
		//for (std::list<EdgeSLAM::ObjectMapPoint*>::iterator lit = lLocalObjectPoints.begin(), lend = lLocalObjectPoints.end(); lit != lend; lit++)
		//{
		//	EdgeSLAM::MapPoint* pMP = *lit;
		//	g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
		//	pMP->SetWorldPos(EdgeSLAM::Converter::toCvMat(vPoint->estimate()));
		//	pMP->UpdateNormalAndDepth();
		//	//pMP->mnLastUpdatedTime = ts;
		//}

	}

}