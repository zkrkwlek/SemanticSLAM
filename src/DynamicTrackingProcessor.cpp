#include <DynamicTrackingProcessor.h>
#include <Frame.h>
#include <ObjectFrame.h>
#include <ObjectOptimizer.h>
#include <ObjectSearchPoints.h>
#include <Visualizer.h>
#include <User.h>
#include <PlaneEstimator.h>
#include <KalmanFilter.h>
#include <Confidence.h>
#include <Camera.h>
#include <SemanticProcessor.h>
#include <FeatureTracker.h>
#include <Map.h>

namespace SemanticSLAM {

    RobustMatcher DynamicTrackingProcessor::rmatcher;
    PnPProblem DynamicTrackingProcessor::pnp_detection;
    PnPProblem DynamicTrackingProcessor::pnp_detection_est;
    bool DynamicTrackingProcessor::mbFastMatch;

    void DynamicTrackingProcessor::Init() {
        // Robust Matcher parameters
        int numKeyPoints = 2000;      // number of detected keypoints
        float ratioTest = 0.85f;      // ratio test
        mbFastMatch = true;       // fastRobustMatch() or robustMatch()

        // RANSAC parameters
        int iterationsCount = 500;      // number of Ransac iterations.
        float reprojectionError = 6.0;  // maximum allowed distance to consider it an inlier.
        double confidence = 0.99;       // ransac successful confidence.

        // Kalman Filter parameters
        int minInliersKalman = 10;    // Kalman threshold updating

        // PnP parameters
        /*cv::SOLVEPNP_EPNP;
        cv::SOLVEPNP_P3P;
        cv::SOLVEPNP_DLS;*/
        int pnpMethod = cv::SOLVEPNP_ITERATIVE;
        std::string featureName = "ORB";
        bool useFLANN = false;

        cv::Ptr<cv::FeatureDetector> detector, descriptor;
        createFeatures(featureName, numKeyPoints, detector, descriptor);
        rmatcher.setFeatureDetector(detector);                                      // set feature detector
        rmatcher.setDescriptorExtractor(descriptor);                                // set descriptor extractor
        rmatcher.setDescriptorMatcher(createMatcher(featureName, useFLANN));        // set matcher
        rmatcher.setRatio(ratioTest); // set ratio test parameter


        //int nStates = 18;            // the number of states
        //int nMeasurements = 6;       // the number of measured states
        //int nInputs = 0;             // the number of control actions
        //double dt = 0.125;           // time between measurements (1/FPS)
        //initKalmanFilter(KFilter, nStates, nMeasurements, nInputs, dt);
    }

    void DynamicTrackingProcessor::ObjectMapping(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
        auto pUser = SLAM->GetUser(user);
        if (!pUser)
            return;

        if (!pUser->KeyFrames.Count(id)) {
            return;
        }
        pUser->mnUsed++;
        auto pKF = pUser->KeyFrames.Get(id);
        int w = pUser->mpCamera->mnWidth;
        int h = pUser->mpCamera->mnHeight;
        auto pMap = pUser->mpMap;
        pUser->mnUsed--;
        std::set<EdgeSLAM::ObjectBoundingBox*> spNewBBs;
        if (!SemanticProcessor::GraphKeyFrameObjectBB.Count(pKF)) {
            return;
        }
        spNewBBs = SemanticProcessor::GraphKeyFrameObjectBB.Get(pKF);
        if (spNewBBs.size() == 0) {
            return;
        }

        //키프레임으로부터 연결
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        ////키프레임 박스 테스트
        std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs = pKF->GetBestCovisibilityKeyFrames(20);
        std::set<EdgeSLAM::ObjectBoundingBox*> setNeighObjectBBs;
        std::set<EdgeSLAM::ObjectNode*> spNodes;
        int nobj = 0;
        for (auto iter = vpLocalKFs.begin(), iend = vpLocalKFs.end(); iter != iend; iter++) {
            auto pKFi = *iter;
            std::set<EdgeSLAM::ObjectBoundingBox*> setTempBBs;
            if (SemanticProcessor::GraphKeyFrameObjectBB.Count(pKFi)) {
                setTempBBs = SemanticProcessor::GraphKeyFrameObjectBB.Get(pKFi);
                for (auto jter = setTempBBs.begin(), jend = setTempBBs.end(); jter != jend; jter++) {
                    auto pContent = *jter;

                    if (!setNeighObjectBBs.count(pContent)) {
                        setNeighObjectBBs.insert(pContent);
                        /*if (pContent->mpNode)
                            nobj++;*/
                    }
                }
            }
            std::set<EdgeSLAM::ObjectNode*> setTempNodes;
            if (SemanticProcessor::GraphKeyFrameObject.Count(pKFi)) {
                setTempNodes = SemanticProcessor::GraphKeyFrameObject.Get(pKFi);
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
        if (setNeighObjectBBs.size() == 0) {
            return;
        }

        std::cout << "TEST=ObjectMapping" << std::endl;

        //현재 박스를 오브젝트의 박스와 연결'
        //이전 포인트 체크
        //박스 연결 후 오브젝트 포인트 생성
        //포인트 생성 후 BA로 최적화
        //오브젝트가 없으면 오브젝트 맵 생성
        if (spNodes.size() > 0) {

        }
        else {
            vpLocalKFs.push_back(pKF);
            std::cout << "Object Map Generation~~" << std::endl;
            ObjectMapGeneration(SLAM, vpLocalKFs, spNewBBs, setNeighObjectBBs, pMap);
        }
    }
    void DynamicTrackingProcessor::ObjectMapGeneration(EdgeSLAM::SLAM* SLAM, std::vector<EdgeSLAM::KeyFrame*> vpLocalKFs, std::set<EdgeSLAM::ObjectBoundingBox*> spNewBBs, std::set<EdgeSLAM::ObjectBoundingBox*> setNeighObjectBBs, EdgeSLAM::Map* MAP) {
        //일단 오브젝트 맵을 여기서 생성하고
        //그 맵으로부터 트래킹 한번 해보고
        //좌표계 찾은 후 그 좌표계로부터 에피폴라 매칭 해보는거

        auto thMaxDesc = SLAM->mpFeatureTracker->max_descriptor_distance;
        auto thMinDesc = SLAM->mpFeatureTracker->min_descriptor_distance;

        for (auto oter = spNewBBs.begin(), oend = spNewBBs.end(); oter != oend; oter++) {
            auto pBBox = *oter;

            auto pNewObjectMap = new EdgeSLAM::ObjectNode();
            pNewObjectMap->mspBBs.Update(pBBox);
            for (auto bter = setNeighObjectBBs.begin(), bend = setNeighObjectBBs.end(); bter != bend; bter++) {
                auto pTempBox = *bter;
                std::chrono::high_resolution_clock::time_point astart = std::chrono::high_resolution_clock::now();

                auto pKF1 = pBBox->mpKF;
                auto pKF2 = pTempBox->mpKF;

                

                if (pKF1 && pKF2) {

                    CreateObjectMapPoint(pKF1, pKF2, pBBox, pTempBox, thMinDesc, thMaxDesc, MAP, pNewObjectMap);

                    //std::cout << "TEST=ObjectMapGeneration=Overlap=" << Noverlap << "||" << pBBox->N << " " << nMatch << "=  " << nRes1 << " " << nRes2 << std::endl;
                    //포즈 찾고 매칭 테스트

                }
            }
        }

    }

    void DynamicTrackingProcessor::CreateObjectMapPoint(EdgeSLAM::KeyFrame* pKF1, EdgeSLAM::KeyFrame* pKF2, EdgeSLAM::ObjectBoundingBox* pBB1, EdgeSLAM::ObjectBoundingBox* pBB2, float minThresh, float maxThresh, EdgeSLAM::Map* pMap, EdgeSLAM::ObjectNode* pObjMap)
    {
        
        int Noverlap = 0;

        std::vector<cv::Point2f> imagePoints1, imagePoints2;
        std::vector<cv::Point3f> objectPoints;
        std::set<EdgeSLAM::MapPoint*> sFound;
        std::vector<EdgeSLAM::MapPoint*> vpMPs;
        cv::Mat avg3D = cv::Mat::zeros(3, 1, CV_32FC1);

        for (int i = 0, iend = pBB1->N; i < iend; i++) {
            auto pMPi = pBB1->mvpMapPoints.get(i);
            if (!pMPi || pMPi->isBad())
                continue;
            auto pOPi = pMPi->mpObjectPoint;
            if (!pOPi)
                continue;
            int idx = pOPi->GetIndexInKeyFrame(pBB2);
            if (idx < 0)
                continue;
            Noverlap++;
            vpMPs.push_back(pMPi);
            avg3D += pMPi->GetWorldPos();
            imagePoints1.push_back(pBB1->mvKeys[i].pt);
            imagePoints2.push_back(pBB2->mvKeys[idx].pt);
        }
        if (Noverlap < 20)
            return;

        avg3D /= imagePoints1.size();
        for (int i = 0, iend = imagePoints1.size(); i < iend; i++) {
            cv::Mat temp = (vpMPs[i]->GetWorldPos() - avg3D);
            cv::Point3f objPt(temp);
            objectPoints.push_back(objPt);
        }

        int iterationsCount = 1000;      // number of Ransac iterations.
        float reprojectionError = 8.0;  // maximum allowed distance to consider it an inlier.
        double confidence = 0.9;       // ransac successful confidence.
        int minInliersKalman = 15;    // Kalman threshold updating
        int pnpMethod = cv::SOLVEPNP_EPNP;
        cv::Mat K1 = pKF1->K.clone();
        cv::Mat K2 = pKF2->K.clone();
        K1.convertTo(K1, CV_64FC1);
        K2.convertTo(K2, CV_64FC1);
        cv::Mat rvec1 = cv::Mat::zeros(3, 1, CV_64FC1);
        cv::Mat tvec1 = cv::Mat::zeros(3, 1, CV_64FC1);
        cv::Mat rvec2 = cv::Mat::zeros(3, 1, CV_64FC1);
        cv::Mat tvec2 = cv::Mat::zeros(3, 1, CV_64FC1);
        cv::Mat inliers_idx;
        // -- Step 3: Estimate the pose using RANSAC approach
        pnp_detection.estimatePoseRANSAC(objectPoints, imagePoints1,
            K1, rvec1, tvec1, pnpMethod, inliers_idx,
            iterationsCount, reprojectionError, confidence);
        int n1 = inliers_idx.rows;
        pnp_detection.estimatePoseRANSAC(objectPoints, imagePoints2,
            K2, rvec2, tvec2, pnpMethod, inliers_idx,
            iterationsCount, reprojectionError, confidence);
        int n2 = inliers_idx.rows;

        cv::Mat Rco1 = cv::Mat::eye(3, 3, CV_64FC1);
        cv::Mat Rco2 = cv::Mat::eye(3, 3, CV_64FC1);
        cv::Mat Tco1 = cv::Mat::eye(4, 4, CV_32FC1);
        cv::Mat Tco2 = cv::Mat::eye(4, 4, CV_32FC1);
        cv::Mat tco1, tco2;
        cv::Rodrigues(rvec1, Rco1);
        Rco1.convertTo(Rco1, CV_32FC1);
        tvec1.convertTo(tco1, CV_32FC1);
        Rco1.copyTo(Tco1.rowRange(0, 3).colRange(0, 3));
        tco1.copyTo(Tco1.rowRange(0, 3).col(3));
        
        cv::Rodrigues(rvec2, Rco2);
        Rco2.convertTo(Rco2, CV_32FC1);
        tvec2.convertTo(tco2, CV_32FC1);
        Rco2.copyTo(Tco2.rowRange(0, 3).colRange(0, 3));
        tco2.copyTo(Tco2.rowRange(0, 3).col(3));
        
        std::vector<bool> outliers1(imagePoints1.size(), false);
        int nRes1 = ObjectOptimizer::ObjectPoseOptimization(imagePoints1, objectPoints, outliers1, Tco1, pKF1->fx, pKF1->fy, pKF1->cx, pKF1->cy);
        std::vector<bool> outliers2(imagePoints2.size(), false);
        int nRes2 = ObjectOptimizer::ObjectPoseOptimization(imagePoints2, objectPoints, outliers2, Tco2, pKF2->fx, pKF2->fy, pKF2->cx, pKF2->cy);

        //Calculate F
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
        
        cv::Mat F12 = Utils::ComputeF12(Rco1, tco1, Rco2, tco2, pKF1->K, pKF2->K);
        std::vector<std::pair<int, int>> vMatchedIndices;
        F12.convertTo(F12, CV_32FC1);
        int nMatch = ObjectSearchPoints::SearchObjectBoxAndBoxForTriangulation(pBB1, pBB2, vMatchedIndices, F12, minThresh, maxThresh, 0.8, false);
        int nMap = 0;
        int nTemp1 = 0;
        int nTemp2 = 0;
        int nTemp3 = 0;

        cv::Mat Twc = pKF1->GetPoseInverse();
        cv::Mat Two = Twc * Tco1;
        cv::Mat Rwo = Two.rowRange(0, 3).colRange(0, 3);
        cv::Mat two = Two.rowRange(0, 3).col(3);
        for (int ikp = 0; ikp < nMatch; ikp++)

        {
            const int& idx1 = vMatchedIndices[ikp].first;
            const int& idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint& kp1 = pBB1->mvKeys[idx1];
            const cv::KeyPoint& kp2 = pBB2->mvKeys[idx2];

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

            cv::Mat ray1 = Rco1 * xn1;
            cv::Mat ray2 = Rco2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

            cv::Mat x3D;
            if (cosParallaxRays > 0 && cosParallaxRays < 0.9998)
            {
                // Linear Triangulation Method
                cv::Mat A(4, 4, CV_32F);
                A.row(0) = xn1.at<float>(0) * Tco1.row(2) - Tco1.row(0);
                A.row(1) = xn1.at<float>(1) * Tco1.row(2) - Tco1.row(1);
                A.row(2) = xn2.at<float>(0) * Tco2.row(2) - Tco2.row(0);
                A.row(3) = xn2.at<float>(1) * Tco2.row(2) - Tco2.row(1);

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

            nTemp1++;
            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rco1.row(2).dot(x3Dt) + tco1.at<float>(2);
            if (z1 <= 0)
                continue;

            float z2 = Rco2.row(2).dot(x3Dt) + tco2.at<float>(2);
            if (z2 <= 0)
                continue;
            nTemp2++;
            //Check reprojection error in first keyframe
            const float& sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];
            const float x1 = Rco1.row(0).dot(x3Dt) + tco1.at<float>(0);
            const float y1 = Rco1.row(1).dot(x3Dt) + tco1.at<float>(1);
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
            const float x2 = Rco2.row(0).dot(x3Dt) + tco2.at<float>(0);
            const float y2 = Rco2.row(1).dot(x3Dt) + tco2.at<float>(1);
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
            nTemp3++;
            
            //Check scale consistency
            cv::Mat normal1 = x3D;// -Ow1;11111111111111111111111
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D;// -Ow2;
            float dist2 = cv::norm(normal2);

            if (dist1 == 0 || dist2 == 0)
                continue;

            
            //// Triangulation is succesfull
            EdgeSLAM::MapPoint* pMP = new EdgeSLAM::MapPoint(Rwo*x3D+two, pKF2, pMap, ts);
            pMP->mnObjectID = 100;

            pBB1->AddMapPoint(pMP, idx1);
            pBB2->AddMapPoint(pMP, idx2);

            int kfidx1 = pBB1->mvIDXs[idx1];
            int kfidx2 = pBB2->mvIDXs[idx2];

            //이 맵은 일단 등록할려면 Tcw를 알아야 함.

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
        }

        std::cout << "map generation = " << nMap <<" "<<nTemp1 <<" "<<nTemp2<<" "<<nTemp3 << " " << nMatch << std::endl;
    }

    void DynamicTrackingProcessor::ObjectTracking(ThreadPool::ThreadPool* POOL, EdgeSLAM::SLAM* SLAM, std::string user, EdgeSLAM::Frame* frame, const cv::Mat& _img, int id) {
        auto pUser = SLAM->GetUser(user);
        if (!pUser)
            return;
        cv::Mat img = _img.clone();
        pUser->mnUsed++;
        auto vecObjectTrackingRes = pUser->mapObjectTrackingResult.Get();
        cv::Mat _K = pUser->GetCameraMatrix();
        cv::Mat Pcw = pUser->GetPose();
        //pUser->mpKalmanFilter->Predict(Pcw);
        pUser->mnUsed--;
        cv::Mat K;
        _K.convertTo(K, CV_64FC1);
        //thread pool tracking

        int nTrackObjTest = 0;
        for (auto iter = vecObjectTrackingRes.begin(), iend = vecObjectTrackingRes.end(); iter != iend; iter++) {
            auto pTrackRes = iter->second;
            if (pTrackRes->mState == EdgeSLAM::ObjectTrackingState::Success) {
                nTrackObjTest++;
            }
        }

        if (vecObjectTrackingRes.size() == 0 || nTrackObjTest == 0) {
            //relocal과 디텍션 요청하기
            WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
            std::stringstream ss;
            ss << "/Store?keyword=RequestObjectDetection&id=" << id << "&src=" << user;
            auto res = mpAPI->Send(ss.str(), "");
            delete mpAPI;
        }

        for (auto iter = vecObjectTrackingRes.begin(), iend = vecObjectTrackingRes.end(); iter != iend; iter++) {
            auto pTrackRes = iter->second;
            if(pTrackRes->mState == EdgeSLAM::ObjectTrackingState::Success)
                POOL->EnqueueJob(DynamicTrackingProcessor::ObjectTracking2, SLAM, pTrackRes, frame, img, id, Pcw, K);
            
            //ObjectTracking(SLAM, mapName, pTrackRes, img, id, K)
        }
    }

    void DynamicTrackingProcessor::UpdateConfidence(EdgeSLAM::ObjectTrackingFrame* pTrackFrame, cv::Mat Pcw, cv::Mat Pco, cv::Mat Pwo, cv::Mat Ow, cv::Mat K) {
        Plane* Floor = nullptr;
        if (PlaneEstimator::GlobalFloor)
        {
            Floor = PlaneEstimator::GlobalFloor;
        }

        cv::Mat Pwc = Pcw.inv();

        cv::Mat Rco = Pco.rowRange(0, 3).colRange(0, 3);
        cv::Mat tco = Pco.rowRange(0, 3).col(3);

        cv::Mat Rwo = Pwo.rowRange(0, 3).colRange(0, 3);
        cv::Mat two = Pwo.rowRange(0, 3).col(3);

        cv::Mat Rcw = Pcw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tcw = Pcw.rowRange(0, 3).col(3);

        //cv::Mat P
        for (int i = 0, N = pTrackFrame->mvImagePoints.size(); i < N; i++) {
            auto pMPi = pTrackFrame->mvpMapPoints[i];
            if (!pMPi || pMPi->isBad())
                continue;
            if (!pMPi->mpConfidence) {
                pMPi->mpConfidence = new Confidence();
            }
            
            cv::Mat Xo = pMPi->GetWorldPos() - Ow;
            cv::Mat Xw = pMPi->GetWorldPos();

            cv::Mat Xw2 = Rwo * Xo + two;
            ////cv::Mat d1 = Xw2 - pMPi->GetWorldPos();
            //
            cv::Mat p1 = K* (Rco * Xo + tco);
            cv::Mat p2 = K * (Rcw * Xw + tcw);

            float d1 = p1.at<float>(2);
            float d2 = p2.at<float>(2);

            cv::Point2f pt1(p1.at<float>(0) / d1, p1.at<float>(1) / d1);    //오브젝트
            cv::Point2f pt2(p2.at<float>(0) / d2, p2.at<float>(1) / d2);    //슬램

            pMPi->mpConfidence->CalcConfidence(pt2, pTrackFrame->mvImagePoints[i]);
            if (Floor) {
                float dist3 = Floor->Distacne(pMPi->GetWorldPos());
                pMPi->mpConfidence->CalcConfidence(abs(dist3));
            }
            //auto diff1 = pt1 - pt2;
            //auto diff2 = Xw - Xw2;

            ////std::cout <<pt1<<" "<<pt2 <<" "<<pTrackFrame->mvImagePoints[i] << diff1.dot(diff1) << " " << diff2.dot(diff2) << std::endl;
            //if (Floor) {
            //    float dist3 = Floor->Distacne(pMPi->GetWorldPos());
            //    std::cout << sqrt(diff1.dot(diff1)) << " " << sqrt(diff2.dot(diff2)) <<" "<<dist3 << std::endl;
            //}
           
        }

    }

    void DynamicTrackingProcessor::UpdateKalmanFilter(EdgeSLAM::ObjectNode* pObject, int nPnP, cv::Mat _Pcw, cv::Mat& _Pco, cv::Mat& Pwo) {

        cv::Mat Pcw, Pco;
        _Pcw.convertTo(Pcw, CV_64FC1);
        _Pco.convertTo(Pco, CV_64FC1);

        cv::Mat Rcw = Pcw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tcw = Pcw.rowRange(0, 3).col(3);
        cv::Mat Rwc = Rcw.t();
        cv::Mat twc = -Rwc * tcw;
        cv::Mat Rco = Pco.rowRange(0, 3).colRange(0, 3);
        cv::Mat tco = Pco.rowRange(0, 3).col(3);

        cv::Mat Rwo = Rwc * Rco;
        cv::Mat two = Rwc * tco + twc;

        //// -- Step 5: Kalman Filter
        //// GOOD MEASUREMENT
        if (nPnP >= 15)
        {
            // fill the measurements vector

            pObject->mpKalmanFilter->fillMeasurements(two, Rwo);
        }

        pObject->mpKalmanFilter->updateKalmanFilter(two, Rwo);
        //객체->슬램 좌표계
        Pwo = cv::Mat::eye(4, 4, CV_64FC1);
        Rwo.copyTo(Pwo.rowRange(0, 3).colRange(0, 3));
        two.copyTo(Pwo.rowRange(0, 3).col(3));
        Pwo.convertTo(Pwo, CV_32FC1);

        //객체->카메라 좌표계
        //Pcw*Pwo
        _Pco = _Pcw * Pwo;
        /*Rco = Rcw * Rwo;
        tco = Rcw * two + tcw;
        _Pco = cv::Mat::eye(4, 4, CV_64FC1);
        Rwo.copyTo(_Pco.rowRange(0, 3).colRange(0, 3));
        two.copyTo(_Pco.rowRange(0, 3).col(3));*/
        _Pco.convertTo(_Pco, CV_32FC1);
        pObject->SetWorldPose(Pwo);
    }

    int nTrack = 0;
    float totalTime3 = 0.0f;

    int DynamicTrackingProcessor::ObjectTracking2(EdgeSLAM::SLAM* SLAM, EdgeSLAM::ObjectTrackingResult* pTrackRes, EdgeSLAM::Frame* frame, const cv::Mat& newframe, int fid, const cv::Mat& Pcw, const cv::Mat& K) {
        
        std::chrono::high_resolution_clock::time_point t_track_start = std::chrono::high_resolution_clock::now();

        auto pTrackFrame = pTrackRes->mpLastFrame;
        auto pObject = pTrackRes->mpObject;
        //일단 옵티컬 플로우로 테스트부터
        
        //cv::Mat Pco = pTrackRes->Pose.clone(); //오브젝트 좌표계에서 카메라 포즈임
        cv::Mat Pwo = pObject->GetWorldPose();
        cv::Mat Pco = Pcw * Pwo;

        int win_size = 10;
        std::vector<cv::Point2f> cornersB;
        std::vector<uchar> features_found;
        cv::Mat tempFrame = newframe.clone();
        //std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        cv::calcOpticalFlowPyrLK(
            pTrackFrame->frame,                         // Previous image
            tempFrame,                         // Next image
            pTrackFrame->mvImagePoints,                     // Previous set of corners (from imgA)
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

        std::vector<cv::Point2f> imagePoints;
        std::vector<cv::Point3f> objectPoints;
        std::vector<EdgeSLAM::MapPoint*> mapPoints;
        std::set<EdgeSLAM::MapPoint*> sFound;
        int nGood = 0;
        for (int i = 0; i < static_cast<int>(pTrackFrame->mvImagePoints.size()); ++i) {
            if (!features_found[i]) {
                continue;
            }
            
            line(
                tempFrame,                        // Draw onto this image
                pTrackFrame->mvImagePoints[i],                 // Starting here
                cornersB[i],                 // Ending here
                cv::Scalar(255, 255, 0),       // This color
                3,                           // This many pixels wide
                cv::LINE_AA                  // Draw line in this style
            );

            auto pMPi = pTrackFrame->mvpMapPoints[i];
            if (!pMPi || pMPi->isBad())
                continue;
            nGood++;
            cv::Mat temp = pMPi->GetWorldPos() - pObject->GetOrigin();
            cv::Point2f imgPt = cornersB[i];
            cv::Point3f objPt(temp);
            imagePoints.push_back(imgPt);
            objectPoints.push_back(objPt);
            mapPoints.push_back(pMPi);
            sFound.insert(pMPi);
        }
        /*{
            cv::Mat Pwo = DynaObjMap->GetPose();
            std::set<EdgeSLAM::MapPoint*> sFound;
            std::vector<std::pair<int, int>> matches;
            int nAdditional = ObjectSearchPoints::SearchObjectMapByProjection(matches, frame, pTrackFrame->mvpMapPoints, sFound, Pcw, Pwo, pObject->GetOrigin(), 10, 100, false);
            std::cout << "match test = " << nGood << " " << nAdditional << std::endl;
        }*/

        int nRes = 0;
        cv::Mat inliers_idx;
        if (nGood > 6) {
            // RANSAC parameters
            int iterationsCount = 1000;      // number of Ransac iterations.
            float reprojectionError = 8.0;  // maximum allowed distance to consider it an inlier.
            double confidence = 0.9;       // ransac successful confidence.

            // Kalman Filter parameters
            int minInliersKalman = 15;    // Kalman threshold updating
            int pnpMethod = cv::SOLVEPNP_EPNP;
            
            std::vector<cv::Point2f> list_points2d_inliers;

            cv::Mat Rco = Pco.rowRange(0, 3).colRange(0, 3);
            cv::Mat tco = Pco.rowRange(0, 3).col(3);
            cv::Mat rvec;// = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Mat tvec;// = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Rodrigues(Rco, rvec);
            rvec.convertTo(rvec, CV_64FC1);
            tco.convertTo(tvec, CV_64FC1);

            pnp_detection.estimatePoseRANSAC(objectPoints, imagePoints,
                K, rvec, tvec, pnpMethod, inliers_idx,
                iterationsCount, reprojectionError, confidence, true);

            cv::Rodrigues(rvec, Rco);
            Rco.copyTo(Pco.rowRange(0, 3).colRange(0, 3));
            tvec.copyTo(Pco.rowRange(0, 3).col(3));
            Pco.convertTo(Pco, CV_32FC1);

            //프로젝션 매칭
            Plane* Floor = nullptr;
            if (PlaneEstimator::GlobalFloor)
            {
                Floor = PlaneEstimator::GlobalFloor;
            }
            EdgeSLAM::ObjectLocalMap* LocalObjectMap = new EdgeSLAM::ObjectLocalMap(mapPoints);
            std::vector<std::pair<int, int>> matches;
            int nAdditional = ObjectSearchPoints::SearchObjectMapByProjection(matches, frame, LocalObjectMap->mvpLocalMapPoints, sFound, Pco, pObject->GetOrigin(), 10, 100, false);
            //최적화
            for (int i = 0, N = matches.size(); i < N; i++) {
                int idx1 = matches[i].first;
                int idx2 = matches[i].second;

                auto pMPi = LocalObjectMap->mvpLocalMapPoints[idx1];
                if (!pMPi || pMPi->isBad())
                    continue;

                if (Floor) {
                    float dist = Floor->Distacne(pMPi->GetWorldPos());
                    if (abs(dist) < 0.1)
                        continue;
                }

                cv::Mat temp = pMPi->GetWorldPos() - pObject->GetOrigin();
                cv::Point3f objPt(temp);
                auto pt = frame->mvKeys[idx2].pt;
                imagePoints.push_back(pt);
                objectPoints.push_back(objPt);
				mapPoints.push_back(pMPi);
            }
            
            std::vector<bool> outliers(imagePoints.size(), false);
            nRes = ObjectOptimizer::ObjectPoseOptimization(imagePoints, objectPoints, outliers, Pco, frame->fx, frame->fy, frame->cx, frame->cy);
            //std::cout << "Local Map Test = " << nAdditional <<" "<< imagePoints.size()<<"=" << " " << nRes << " ==" << LocalObjectMap->mvpLocalBoxes.size() << " " << LocalObjectMap->mvpLocalMapPoints.size() << std::endl;

            //nRes = inliers_idx.rows;
            if (nRes > 15) {
                pTrackRes->Pose = Pco.clone();
                pTrackRes->mState = EdgeSLAM::ObjectTrackingState::Success;
                pTrackRes->mnLastSuccessFrameId = fid;
                auto pTrackFrame = pTrackRes->mpLastFrame;
                pTrackFrame->mvImagePoints.clear();
                pTrackFrame->mvpMapPoints.clear();
                pTrackFrame->mvImagePoints.reserve(nRes);
                pTrackFrame->mvpMapPoints.reserve(nRes);
                
                for (int i = 0, N = imagePoints.size(); i < N; i++) {
                    auto pMPi = mapPoints[i];
                    if (!pMPi || pMPi->isBad() || outliers[i])
                        continue;
                    cv::Point2f pt = imagePoints[i];
                    pTrackFrame->mvImagePoints.push_back(pt);
                    pTrackFrame->mvpMapPoints.push_back(pMPi);
                }
                pTrackFrame->frame = newframe.clone();
            }
            else {
                pTrackRes->mState = EdgeSLAM::ObjectTrackingState::Failed;
            }
        }
        
        //update
        pTrackRes->mnLastTrackFrameId = fid;
        
        std::chrono::high_resolution_clock::time_point t_track_end = std::chrono::high_resolution_clock::now();
        auto du_track = std::chrono::duration_cast<std::chrono::milliseconds>(t_track_end - t_track_start).count();
        nTrack++;
        totalTime3 += (du_track / 1000.0);
        //std::cout << "tracking avg = " << totalTime3 / nTrack << std::endl;

    }

    int DynamicTrackingProcessor::ObjectRelocalization(EdgeSLAM::ObjectBoundingBox* pNewBox, EdgeSLAM::ObjectNode* pObject, EdgeSLAM::ObjectTrackingResult* pTrackRes, const cv::Mat& newframe, const cv::Mat& K, cv::Mat& P) {

        std::set<EdgeSLAM::ObjectBoundingBox*> setNeighBoxes = pObject->mspBBs.Get();
        const int nBBs = setNeighBoxes.size();
        int nCandidates = nBBs;
        std::vector<std::vector<EdgeSLAM::MapPoint*> > vvpMapPointMatches(nBBs);

        std::vector<EdgeSLAM::ObjectBoundingBox*> vpCandidateBBs(setNeighBoxes.begin(), setNeighBoxes.end());
        std::vector<bool> vbDiscarded(nBBs, false);

        int nBoxMatchThresh = 5;
        int nSuccessTracking = 15;

        auto pFrame = pNewBox->mpF;
        if (!pFrame)
            std::cout << "frame is not exist??????" << std::endl;

        int nGood = 0;
        std::set<EdgeSLAM::MapPoint*> sFound;
        for (auto iter = setNeighBoxes.begin(), iend = setNeighBoxes.end(); iter != iend; iter++) {
            auto pNeighBox = *iter;
            std::vector<cv::DMatch> good_matches;
            rmatcher.robustMatch(newframe, cv::Mat(), pNewBox->mvKeys, pNewBox->desc, pNeighBox->mvKeys, pNeighBox->desc, good_matches);
            std::cout << good_matches.size() <<" "<< pNewBox->mvKeys.size()<<" "<<pNeighBox->mvKeys.size() << std::endl;
            for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index)
            {
                int newIdx = good_matches[match_index].queryIdx;
                int neighIdx = good_matches[match_index].trainIdx;

                auto pMPi = pNeighBox->mvpMapPoints.get(neighIdx);
                if (sFound.count(pMPi))
                    continue;
                if (!pMPi || pMPi->isBad())
                    continue;
                sFound.insert(pMPi);
                pNewBox->mvpMapPoints.update(newIdx, pMPi);
                nGood++;
            }
        }
        std::cout << "relocal test = " << nCandidates<<" "<< nGood << std::endl;

        P = cv::Mat::eye(4, 4, CV_32FC1);
        bool bRes = false;
        if (nGood >= 4) {

            cv::Mat frame_vis = newframe;

            // RANSAC parameters
            int iterationsCount = 1000;      // number of Ransac iterations.
            float reprojectionError = 8.0;  // maximum allowed distance to consider it an inlier.
            double confidence = 0.9;       // ransac successful confidence.

            // Kalman Filter parameters
            int minInliersKalman = 15;    // Kalman threshold updating
            int pnpMethod = cv::SOLVEPNP_EPNP;
            cv::Mat inliers_idx;
            std::vector<cv::Point2f> list_points2d_inliers;

            //int nMeasurements = 6;
            //cv::Mat measurements(nMeasurements, 1, CV_64FC1); measurements.setTo(cv::Scalar(0));
            bool good_measurement = false;

            std::vector<cv::Point2f> imagePoints;
            std::vector<cv::Point3f> objectPoints;
            std::set<EdgeSLAM::MapPoint*> sFound;
            std::vector<EdgeSLAM::MapPoint*> vpMPs;

            cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
            
            for (int i = 0; i < pNewBox->N; i++) {
                auto pMPi = pNewBox->mvpMapPoints.get(i);
                if (!pMPi || pMPi->isBad())
                    continue;
                auto imgPt = pNewBox->mvKeys[i].pt;
                cv::Mat temp = pMPi->GetWorldPos() - pObject->GetOrigin();
                cv::Point3f objPt(temp);
                imagePoints.push_back(imgPt);
                objectPoints.push_back(objPt);

                sFound.insert(pMPi);
                vpMPs.push_back(pMPi);
            }
            
            // -- Step 3: Estimate the pose using RANSAC approach
            pnp_detection.estimatePoseRANSAC(objectPoints, imagePoints,
                K, rvec, tvec, pnpMethod, inliers_idx,
                iterationsCount, reprojectionError, confidence);

            cv::Mat R;
            cv::Rodrigues(rvec, R);

            R.copyTo(P.rowRange(0, 3).colRange(0, 3));
            tvec.copyTo(P.rowRange(0, 3).col(3));
            P.convertTo(P, CV_32FC1);
            std::vector<std::pair<int, int>> matches;
            EdgeSLAM::ObjectLocalMap* LocalObjectMap = new EdgeSLAM::ObjectLocalMap(vpMPs);
            int nAdditional = ObjectSearchPoints::SearchObjectMapByProjection(matches, pFrame, LocalObjectMap->mvpLocalMapPoints, sFound, P, pObject->GetOrigin(), 10, 100, false);
            
            for (int i = 0, N = matches.size(); i < N; i++) {
                int idx1 = matches[i].first;
                int idx2 = matches[i].second;

                auto pMPi = LocalObjectMap->mvpLocalMapPoints[idx1];
                if (!pMPi || pMPi->isBad())
                    continue;

                cv::Mat temp = pMPi->GetWorldPos() - pObject->GetOrigin();
                cv::Point3f objPt(temp);
                auto pt = pFrame->mvKeys[idx2].pt;
                imagePoints.push_back(pt);
                objectPoints.push_back(objPt);
                vpMPs.push_back(pMPi);
            }

            std::vector<bool> outliers(imagePoints.size(), false);
            int nRes = ObjectOptimizer::ObjectPoseOptimization(imagePoints, objectPoints, outliers, P, pFrame->fx, pFrame->fy, pFrame->cx, pFrame->cy);
            
            if (nRes > 10) {
                pTrackRes->Pose = P.clone();
                pTrackRes->mState = EdgeSLAM::ObjectTrackingState::Success;
                pTrackRes->mnLastSuccessFrameId = pFrame->mnFrameID;
                pTrackRes->Pose = P.clone();
                auto pTrackFrame = pTrackRes->mpLastFrame;
                if (!pTrackFrame) {
                    pTrackFrame = new EdgeSLAM::ObjectTrackingFrame();
                    pTrackRes->mpLastFrame = pTrackFrame;
                }
                pTrackFrame->mvImagePoints.clear();
                pTrackFrame->mvpMapPoints.clear();
                pTrackFrame->mvImagePoints.reserve(nRes);
                pTrackFrame->mvpMapPoints.reserve(nRes);

                for (int i = 0, N = imagePoints.size(); i < N; i++) {
                    auto pMPi = vpMPs[i];
                    if (!pMPi || pMPi->isBad() || outliers[i])
                        continue;
                    cv::Point2f pt = imagePoints[i];
                    pTrackFrame->mvImagePoints.push_back(pt);
                    pTrackFrame->mvpMapPoints.push_back(pMPi);
                }
                pTrackFrame->frame = newframe.clone();
            }
            else {
                pTrackRes->mState = EdgeSLAM::ObjectTrackingState::Failed;
            }
            pTrackRes->mnLastTrackFrameId = pFrame->mnFrameID;
            std::cout << "matching test = " << nAdditional <<" "<<nRes << std::endl;

            // -- Step 4: Catch the inliers keypoints to draw
            for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index)
            {
                int n = inliers_idx.at<int>(inliers_index);         // i-inlier
                cv::Point2f point2d = imagePoints[n];     // i-inlier point 2D
                list_points2d_inliers.push_back(point2d);           // add i-inlier to list
            }

            
            cv::Mat D = cv::Mat::zeros(4, 1, CV_64FC1);

            //nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P, pObject->GetOrigin());
            //std::cout << "Last Test = " << nGood << " " << inliers_idx.rows << std::endl;
            return nRes;

        }
        return 0;
    }
    
    void DynamicTrackingProcessor::createFeatures(const std::string& featureName, int numKeypoints, cv::Ptr<cv::Feature2D>& detector, cv::Ptr<cv::Feature2D>& descriptor)
    {
        if (featureName == "ORB")
        {
            detector = cv::ORB::create(numKeypoints);
            descriptor = cv::ORB::create(numKeypoints);
        }
        else if (featureName == "KAZE")
        {
            detector = cv::KAZE::create();
            descriptor = cv::KAZE::create();
        }
        else if (featureName == "AKAZE")
        {
            detector = cv::AKAZE::create();
            descriptor = cv::AKAZE::create();
        }
        else if (featureName == "BRISK")
        {
            detector = cv::BRISK::create();
            descriptor = cv::BRISK::create();
        }
        else if (featureName == "SIFT")
        {
            detector = cv::SIFT::create();
            descriptor = cv::SIFT::create();
        }
        else if (featureName == "SURF")
        {
#if defined (OPENCV_ENABLE_NONFREE) && defined (HAVE_OPENCV_XFEATURES2D)
            detector = cv::xfeatures2d::SURF::create(100, 4, 3, true);   //extended=true
            descriptor = cv::xfeatures2d::SURF::create(100, 4, 3, true); //extended=true
#else
            std::cout << "xfeatures2d module is not available or nonfree is not enabled." << std::endl;
            std::cout << "Default to ORB." << std::endl;
            detector = cv::ORB::create(numKeypoints);
            descriptor = cv::ORB::create(numKeypoints);
#endif
        }
        else if (featureName == "BINBOOST")
        {
#if defined (HAVE_OPENCV_XFEATURES2D)
            detector = cv::KAZE::create();
            descriptor = cv::xfeatures2d::BoostDesc::create();
#else
            std::cout << "xfeatures2d module is not available." << std::endl;
            std::cout << "Default to ORB." << std::endl;
            detector = cv::ORB::create(numKeypoints);
            descriptor = cv::ORB::create(numKeypoints);
#endif
        }
        else if (featureName == "VGG")
        {
#if defined (HAVE_OPENCV_XFEATURES2D)
            detector = cv::KAZE::create();
            descriptor = cv::xfeatures2d::VGG::create();
#else
            std::cout << "xfeatures2d module is not available." << std::endl;
            std::cout << "Default to ORB." << std::endl;
            detector = cv::ORB::create(numKeypoints);
            descriptor = cv::ORB::create(numKeypoints);
#endif
        }
    }

    cv::Ptr<cv::DescriptorMatcher> DynamicTrackingProcessor::createMatcher(const std::string& featureName, bool useFLANN)
    {
        if (featureName == "ORB" || featureName == "BRISK" || featureName == "AKAZE" || featureName == "BINBOOST")
        {
            if (useFLANN)
            {
                cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1); // instantiate LSH index parameters
                cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);       // instantiate flann search parameters
                return cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
            }
            else
            {
                return cv::DescriptorMatcher::create("BruteForce-Hamming");
            }

        }
        else
        {
            if (useFLANN)
            {
                return cv::DescriptorMatcher::create("FlannBased");
            }
            else
            {
                return cv::DescriptorMatcher::create("BruteForce");
            }
        }
    }

    ///draw function
    // For text
    const int fontFace = cv::FONT_ITALIC;
    const double fontScale = 0.75;
    const int thickness_font = 2;

    // For circles
    const int lineType = 8;
    const int radius = 4;

    void DynamicTrackingProcessor::draw2DPoints(cv::Mat image, std::vector<cv::Point2f>& list_points, cv::Scalar color)
    {
        for (size_t i = 0; i < list_points.size(); i++)
        {
            cv::Point2f point_2d = list_points[i];

            // Draw Selected points
            cv::circle(image, point_2d, radius, color, -1, lineType);
        }
    }

    cv::Mat ReprojectPoints(const cv::Mat& R, const cv::Mat& t, const cv::Mat& K, const cv::Mat& X) {
        cv::Mat porj = R* X + t;
    }

    void DynamicTrackingProcessor::drawBoundingBox(cv::Mat& img, const cv::Mat& Pco, const cv::Mat& K, float radx, float rady, float radz) {
        cv::Mat R = Pco.rowRange(0, 3).colRange(0, 3);
        cv::Mat t = Pco.rowRange(0, 3).col(3);

        std::vector<cv::Point2f> points;
        /*points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f( radius, -radius, -radius)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f(-radius, -radius, -radius)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f( radius,  radius, -radius)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f(-radius,  radius, -radius)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f( radius, -radius,  radius)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f(-radius, -radius,  radius)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f( radius,  radius,  radius)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f(-radius,  radius,  radius)), K, R, t));*/
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f( radx, -rady, -radz)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f(-radx, -rady, -radz)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f( radx,  rady, -radz)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f(-radx,  rady, -radz)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f( radx, -rady,  radz)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f(-radx, -rady,  radz)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f( radx,  rady,  radz)), K, R, t));
        points.push_back(Utils::ProjectPoint(cv::Mat(cv::Point3f(-radx,  rady,  radz)), K, R, t));

        cv::Scalar color(255, 255, 255);
        cv::line(img, points[0], points[1], color, 2);
        cv::line(img, points[0], points[2], color, 2);
        cv::line(img, points[3], points[1], color, 2);
        cv::line(img, points[3], points[2], color, 2);

        cv::line(img, points[4], points[5], color, 2);
        cv::line(img, points[4], points[6], color, 2);
        cv::line(img, points[7], points[5], color, 2);
        cv::line(img, points[7], points[6], color, 2);

        cv::line(img, points[0], points[4], color, 2);
        cv::line(img, points[1], points[5], color, 2);
        cv::line(img, points[2], points[6], color, 2);
        cv::line(img, points[3], points[7], color, 2);
    }

    void DynamicTrackingProcessor::draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f>& list_points2d) {
        cv::Scalar red(0, 0, 255);
        cv::Scalar green(0, 255, 0);
        cv::Scalar blue(255, 0, 0);
        cv::Scalar black(0, 0, 0);

        cv::Point2i origin = list_points2d[0];
        cv::Point2i pointX = list_points2d[1];
        cv::Point2i pointY = list_points2d[2];
        cv::Point2i pointZ = list_points2d[3];

        drawArrow(image, origin, pointX, red, 9, 2);
        drawArrow(image, origin, pointY, green, 9, 2);
        drawArrow(image, origin, pointZ, blue, 9, 2);
        cv::circle(image, origin, radius / 2, black, -1, lineType);
    }
    void DynamicTrackingProcessor::drawArrow(cv::Mat image, cv::Point2i p, cv::Point2i q, cv::Scalar color, int arrowMagnitude, int thickness, int line_type, int shift)
    {
        //Draw the principle line
        cv::line(image, p, q, color, thickness, line_type, shift);
        const double PI = CV_PI;
        //compute the angle alpha
        double angle = atan2((double)p.y - q.y, (double)p.x - q.x);
        //compute the coordinates of the first segment
        p.x = (int)(q.x + arrowMagnitude * cos(angle + PI / 4));
        p.y = (int)(q.y + arrowMagnitude * sin(angle + PI / 4));
        //Draw the first segment
        cv::line(image, p, q, color, thickness, line_type, shift);
        //compute the coordinates of the second segment
        p.x = (int)(q.x + arrowMagnitude * cos(angle - PI / 4));
        p.y = (int)(q.y + arrowMagnitude * sin(angle - PI / 4));
        //Draw the second segment
        cv::line(image, p, q, color, thickness, line_type, shift);
    }
}