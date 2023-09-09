#include <DynamicTrackingProcessor.h>
#include <Frame.h>
#include <ObjectFrame.h>
#include <ObjectOptimizer.h>
#include <ObjectSearchPoints.h>
#include <Visualizer.h>
#include <User.h>
#include <PlaneEstimator.h>
#include <KalmanFilter.h>

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
        for (auto iter = vecObjectTrackingRes.begin(), iend = vecObjectTrackingRes.end(); iter != iend; iter++) {
            auto pTrackRes = iter->second;
            if(pTrackRes->mState == EdgeSLAM::ObjectTrackingState::Success)
                POOL->EnqueueJob(DynamicTrackingProcessor::ObjectTracking2, SLAM, pTrackRes, frame, img, id, Pcw, K);
            //ObjectTracking(SLAM, mapName, pTrackRes, img, id, K)
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

    int DynamicTrackingProcessor::ObjectTracking2(EdgeSLAM::SLAM* SLAM, EdgeSLAM::ObjectTrackingResult* pTrackRes, EdgeSLAM::Frame* frame, const cv::Mat& newframe, int fid, const cv::Mat& Pcw, const cv::Mat& K) {
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