#include <DynamicTrackingProcessor.h>
#include <Frame.h>
#include <ObjectFrame.h>
#include <ObjectOptimizer.h>
#include <ObjectSearchPoints.h>
#include <Visualizer.h>

namespace SemanticSLAM {

    RobustMatcher DynamicTrackingProcessor::rmatcher;
    cv::KalmanFilter DynamicTrackingProcessor::KFilter;
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


        int nStates = 18;            // the number of states
        int nMeasurements = 6;       // the number of measured states
        int nInputs = 0;             // the number of control actions
        double dt = 0.125;           // time between measurements (1/FPS)
        initKalmanFilter(KFilter, nStates, nMeasurements, nInputs, dt);
    }
    
    int DynamicTrackingProcessor::ObjectTracking(EdgeSLAM::SLAM* SLAM, std::string name, EdgeSLAM::ObjectBoundingBox* pNewBox, EdgeSLAM::ObjectNode* pObject, EdgeSLAM::ObjectTrackingResult* pTrackRes, const cv::Mat& newframe, const cv::Mat& K, cv::Mat& P) {
        auto pTrackFrame = pTrackRes->mpLastFrame;
        //일단 옵티컬 플로우로 테스트부터

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

        }

        if (nGood > 6) {
            // RANSAC parameters
            int iterationsCount = 1000;      // number of Ransac iterations.
            float reprojectionError = 8.0;  // maximum allowed distance to consider it an inlier.
            double confidence = 0.9;       // ransac successful confidence.

            // Kalman Filter parameters
            int minInliersKalman = 15;    // Kalman threshold updating
            int pnpMethod = cv::SOLVEPNP_EPNP;
            cv::Mat inliers_idx;
            std::vector<cv::Point2f> list_points2d_inliers;

            int nMeasurements = 6;
            cv::Mat measurements(nMeasurements, 1, CV_64FC1); measurements.setTo(cv::Scalar(0));
            bool good_measurement = false;

            cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

            // -- Step 3: Estimate the pose using RANSAC approach
            pnp_detection.estimatePoseRANSAC(objectPoints, imagePoints,
                K, pnpMethod, inliers_idx,
                iterationsCount, reprojectionError, confidence);

            // -- Step 4: Catch the inliers keypoints to draw
            for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index)
            {
                int n = inliers_idx.at<int>(inliers_index);         // i-inlier
                cv::Point2f point2d = imagePoints[n];     // i-inlier point 2D
                list_points2d_inliers.push_back(point2d);           // add i-inlier to list
            }

            // Draw inliers points 2D
            //draw2DPoints(frame_vis, list_points2d_inliers, cv::Scalar(255, 255, 0));

            // -- Step 5: Kalman Filter
            // GOOD MEASUREMENT
            if (inliers_idx.rows >= minInliersKalman)
            {
                // Get the measured translation
                cv::Mat translation_measured = pnp_detection.get_t_matrix();

                // Get the measured rotation
                cv::Mat rotation_measured = pnp_detection.get_R_matrix();

                // fill the measurements vector
                fillMeasurements(measurements, translation_measured, rotation_measured);
                good_measurement = true;
            }

            // update the Kalman filter with good measurements, otherwise with previous valid measurements
            cv::Mat translation_estimated(3, 1, CV_64FC1);
            cv::Mat rotation_estimated(3, 3, CV_64FC1);
            updateKalmanFilter(KFilter, measurements,
                translation_estimated, rotation_estimated);

            // -- Step 6: Set estimated projection matrix
            pnp_detection_est.set_P_matrix(rotation_estimated, translation_estimated);

            if (good_measurement)
            {
                P = pnp_detection_est.get_P_matrix();
            }
            else {
                P = pnp_detection.get_P_matrix();
            }
            P.convertTo(P, CV_32FC1);
            cv::Mat D = cv::Mat::zeros(4, 1, CV_64FC1);
        
            float l = 2;
            std::vector<cv::Point2f> pose_points2d;
            pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, 0, 0), K));  // axis center
            pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(l, 0, 0), K));  // axis x
            pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, l, 0), K));  // axis y
            pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, 0, l), K));  // axis z
            draw3DCoordinateAxes(tempFrame, pose_points2d);

        }
        

        SLAM->VisualizeImage(name, tempFrame, 2);
    }

    int DynamicTrackingProcessor::ObjectRelocalization(EdgeSLAM::ObjectBoundingBox* pNewBox, EdgeSLAM::ObjectNode* pObject, const cv::Mat& newframe, const cv::Mat& K, cv::Mat& P) {

        std::set<EdgeSLAM::ObjectBoundingBox*> setNeighBoxes = pObject->mspBBs.Get();
        const int nBBs = setNeighBoxes.size();
        int nCandidates = nBBs;
        std::vector<std::vector<EdgeSLAM::MapPoint*> > vvpMapPointMatches(nBBs);

        std::vector<EdgeSLAM::ObjectBoundingBox*> vpCandidateBBs(setNeighBoxes.begin(), setNeighBoxes.end());
        std::vector<bool> vbDiscarded(nBBs, false);

        int nBoxMatchThresh = 5;
        int nSuccessTracking = 15;

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

            int nMeasurements = 6;
            cv::Mat measurements(nMeasurements, 1, CV_64FC1); measurements.setTo(cv::Scalar(0));
            bool good_measurement = false;

            std::vector<cv::Point2f> imagePoints;
            std::vector<cv::Point3f> objectPoints;
            cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Point3f avg(0, 0, 0);
            for (int i = 0; i < pNewBox->N; i++) {
                auto pMPi = pNewBox->mvpMapPoints.get(i);
                if (!pMPi || pMPi->isBad())
                    continue;
                auto imgPt = pNewBox->mvKeys[i].pt;
                //cv::Mat Xo = pMPi->GetWorldPos();
                cv::Point3f objPt(pMPi->GetWorldPos());
                imagePoints.push_back(imgPt);
                objectPoints.push_back(objPt);
                avg += objPt;
            }
            avg.x /= objectPoints.size();
            avg.y /= objectPoints.size();
            avg.z /= objectPoints.size();
            for (int i = 0, iend = objectPoints.size(); i < iend; i++)
                objectPoints[i] -= cv::Point3f(pObject->GetOrigin());

            // -- Step 3: Estimate the pose using RANSAC approach
            pnp_detection.estimatePoseRANSAC(objectPoints, imagePoints,
                K, pnpMethod, inliers_idx,
                iterationsCount, reprojectionError, confidence);

            // -- Step 4: Catch the inliers keypoints to draw
            for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index)
            {
                int n = inliers_idx.at<int>(inliers_index);         // i-inlier
                cv::Point2f point2d = imagePoints[n];     // i-inlier point 2D
                list_points2d_inliers.push_back(point2d);           // add i-inlier to list
            }

            // Draw inliers points 2D
            //draw2DPoints(frame_vis, list_points2d_inliers, cv::Scalar(255, 255, 0));

            // -- Step 5: Kalman Filter
            // GOOD MEASUREMENT
            if (inliers_idx.rows >= minInliersKalman)
            {
                // Get the measured translation
                cv::Mat translation_measured = pnp_detection.get_t_matrix();

                // Get the measured rotation
                cv::Mat rotation_measured = pnp_detection.get_R_matrix();

                // fill the measurements vector
                fillMeasurements(measurements, translation_measured, rotation_measured);
                good_measurement = true;
            }

            // update the Kalman filter with good measurements, otherwise with previous valid measurements
            cv::Mat translation_estimated(3, 1, CV_64FC1);
            cv::Mat rotation_estimated(3, 3, CV_64FC1);
            updateKalmanFilter(KFilter, measurements,
                translation_estimated, rotation_estimated);

            // -- Step 6: Set estimated projection matrix
            pnp_detection_est.set_P_matrix(rotation_estimated, translation_estimated);

            if (good_measurement)
            {
                P = pnp_detection_est.get_P_matrix();
            }
            else {
                P = pnp_detection.get_P_matrix();
            }
            P.convertTo(P, CV_32FC1);
            cv::Mat D = cv::Mat::zeros(4, 1, CV_64FC1);

            //nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P, pObject->GetOrigin());
            //std::cout << "Last Test = " << nGood << " " << inliers_idx.rows << std::endl;
            return inliers_idx.rows;

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

                    std::set<EdgeSLAM::MapPoint*> sFound;
                    cv::Mat inliers_idx;
                    std::vector<cv::Point2f> list_points2d_inliers;
                    //ObjectOptimizer::ObjectPoseInitialization(pNewBox, K, D, P, iterationsCount, reprojectionError, confidence, pnpMethod, inliers_idx);
                    nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P);

                    if (nGood < nBoxMatchThresh) {
                        vbDiscarded[i] = true;
                        nCandidates--;
                        continue;
                    }
                    for (int io = 0; io < pNewBox->N; io++)
                        if (pNewBox->mvbOutliers[io])
                            pNewBox->mvpMapPoints.update(io, nullptr);
                        else {
                            auto pMPi = pNewBox->mvpMapPoints.get(io);
                            if (pMPi && !pMPi->isBad())
                                sFound.insert(pMPi);
                        }
                    if (nGood < nSuccessTracking)
                    {
                        int nadditional = ObjectSearchPoints::SearchBoxByProjection(pNewBox, vpCandidateBBs[i], sFound, P, 10, 100);
                        std::cout << "nadditional = " << nadditional << std::endl;
                        if (nadditional + nGood >= nSuccessTracking)
                        {
                            nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > nBoxMatchThresh && nGood < nSuccessTracking)
                            {
                                sFound.clear();
                                for (int ip = 0; ip < pNewBox->N; ip++) {
                                    if (pNewBox->mvbOutliers[ip])
                                        continue;
                                    sFound.insert(pNewBox->mvpMapPoints.get(ip));
                                }
                                nadditional = ObjectSearchPoints::SearchBoxByProjection(pNewBox, vpCandidateBBs[i], sFound, P, 3, 64);
                                // Final optimization
                                if (nGood + nadditional >= nSuccessTracking)
                                {
                                    nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P);
                                    for (int io = 0; io < pNewBox->N; io++) {
                                        if (pNewBox->mvbOutliers[io])
                                        {
                                            pNewBox->mvpMapPoints.update(io, nullptr);
                                        }
                                        continue;
                                    }
                                }
                            }
                        }
                    }//inf ntracking

                    if (nGood >= nSuccessTracking)
                    {
                        good_measurement = true;
                        bRes = true;
                        bMatch = true;
                        break;
                    }
                }//for neighbor boxes
            }//while

            // -- Step X: Draw pose and coordinate frame
            float l = 5;
            bool displayFilteredPose = false;
            std::vector<cv::Point2f> pose_points2d;
            if (good_measurement) {
                pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, 0, 0), K));  // axis center
                pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(l, 0, 0), K));  // axis x
                pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, l, 0), K));  // axis y
                pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, 0, l), K));  // axis z
            }
            else {
                pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(0, 0, 0), K));  // axis center
                pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(l, 0, 0), K));  // axis x
                pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(0, l, 0), K));  // axis y
                pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(0, 0, l), K));  // axis z
            }
            draw3DCoordinateAxes(frame_vis, pose_points2d);           // draw axes
        }
        return 0;
    }

    void DynamicTrackingProcessor::PoseRelocalization(EdgeSLAM::ObjectBoundingBox* pNewBox, std::set<EdgeSLAM::ObjectBoundingBox*> setNeighBoxes, const cv::Mat& newframe, const cv::Mat& K, cv::Mat& P) {

        const int nBBs = setNeighBoxes.size();
        std::vector<std::vector<EdgeSLAM::MapPoint*> > vvpMapPointMatches(nBBs);

        std::vector<EdgeSLAM::ObjectBoundingBox*> vpCandidateBBs(setNeighBoxes.begin(), setNeighBoxes.end());
        std::vector<bool> vbDiscarded;
        vbDiscarded.resize(nBBs);

        int nCandidates = 0;

        auto thMaxDesc = 100;
        auto thMinDesc = 50;

        int nInitialMatchThresh = 5;
        int nBoxMatchThresh = 5;
        int nSuccessTracking = 15;

        // RANSAC parameters
        int iterationsCount = 1000;      // number of Ransac iterations.
        float reprojectionError = 8.0;  // maximum allowed distance to consider it an inlier.
        double confidence = 0.9;       // ransac successful confidence.

        // Kalman Filter parameters
        int minInliersKalman = 15;    // Kalman threshold updating
        int pnpMethod = cv::SOLVEPNP_EPNP;
        
        int nMeasurements = 6;
        cv::Mat measurements(nMeasurements, 1, CV_64FC1); measurements.setTo(cv::Scalar(0));
        bool good_measurement = false;

        for (int i = 0; i < nBBs; i++)
        {
            auto pNeighBox = vpCandidateBBs[i];
            
            std::vector<cv::DMatch> good_matches;
            std::vector<EdgeSLAM::MapPoint*> tempMPs(pNewBox->N);
            rmatcher.robustMatch(newframe, cv::Mat(), pNewBox->mvKeys, pNewBox->desc, pNeighBox->mvKeys, pNeighBox->desc, good_matches);

            int nMatch = 0;
            for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index)
            {
                int newIdx = good_matches[match_index].queryIdx;
                int neighIdx = good_matches[match_index].trainIdx;

                auto pMPi = pNeighBox->mvpMapPoints.get(neighIdx);
                                
                if (!pMPi || pMPi->isBad())
                    continue;
                tempMPs[newIdx] = pMPi;
                nMatch++;
            }
            vvpMapPointMatches[i] = tempMPs;
            if (nMatch < nInitialMatchThresh) {
                vbDiscarded[i] = true;
            }
            else
                nCandidates++;
        }
        std::cout << "Initial Matching = " << nCandidates << std::endl;

        bool bMatch = false;
        int nGood = 0;
        P = cv::Mat::eye(4, 4, CV_32FC1);
        cv::Mat D = cv::Mat::zeros(4, 1, CV_64FC1);
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

                std::set<EdgeSLAM::MapPoint*> sFound;
                for (size_t j = 0, jend = vvpMapPointMatches[i].size(); j < jend; j++)
                {
                    auto pMP = vvpMapPointMatches[i][j];
                    if (pMP && !pMP->isBad())
                    {
                        pNewBox->mvpMapPoints.update(j, pMP);
                        sFound.insert(pMP);
                    }
                }
                cv::Mat inliers_idx;
                std::vector<cv::Point2f> list_points2d_inliers;
                ObjectOptimizer::ObjectPoseInitialization(pNewBox, K, D, P, iterationsCount, reprojectionError, confidence, pnpMethod, inliers_idx);
                nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P);

                if (nGood < nBoxMatchThresh) {
                    vbDiscarded[i] = true;
                    nCandidates--;
                    continue;
                }
                for (int io = 0; io < pNewBox->N; io++)
                    if (pNewBox->mvbOutliers[io])
                        pNewBox->mvpMapPoints.update(io, nullptr);
                if (nGood < nSuccessTracking)
                {
                    int nadditional = ObjectSearchPoints::SearchBoxByProjection(pNewBox, vpCandidateBBs[i], sFound, P, 10, 100);
                    //std::cout << "nadditional = " << nadditional << std::endl;
                    if (nadditional + nGood >= nSuccessTracking)
                    {
                        nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > nBoxMatchThresh && nGood < nSuccessTracking)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < pNewBox->N; ip++) {
                                if (pNewBox->mvbOutliers[ip])
                                    continue;
                                sFound.insert(pNewBox->mvpMapPoints.get(ip));
                            }
                            nadditional = ObjectSearchPoints::SearchBoxByProjection(pNewBox, vpCandidateBBs[i], sFound, P, 3, 64);
                            // Final optimization
                            if (nGood + nadditional >= nSuccessTracking)
                            {
                                nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P);
                                for (int io = 0; io < pNewBox->N; io++) {
                                    if (pNewBox->mvbOutliers[io])
                                    {
                                        pNewBox->mvpMapPoints.update(io, nullptr);
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

            }//for
        }//while

    }
    
    void DynamicTrackingProcessor::MatchTestByFrame(EdgeSLAM::Frame* pNewFrame, std::set<EdgeSLAM::ObjectBoundingBox*> setNeighBoxes, const cv::Mat& newframe, const cv::Mat& K, cv::Mat& P) {
    
        const int nBBs = setNeighBoxes.size();
        int nCandidates = nBBs;
        std::vector<std::vector<EdgeSLAM::MapPoint*> > vvpMapPointMatches(nBBs);

        std::vector<EdgeSLAM::ObjectBoundingBox*> vpCandidateBBs(setNeighBoxes.begin(), setNeighBoxes.end());
        std::vector<bool> vbDiscarded(nBBs, false);

        int nBoxMatchThresh = 5;
        int nSuccessTracking = 15;
        int nGood = 0;
        std::set<EdgeSLAM::MapPoint*> sFound;
        for (auto iter = setNeighBoxes.begin(), iend = setNeighBoxes.end(); iter != iend; iter++) {
            auto pNeighBox = *iter;
            std::vector<cv::DMatch> good_matches;
            rmatcher.robustMatch(newframe, cv::Mat(), pNewFrame->mvKeysUn, pNewFrame->mDescriptors, pNeighBox->mvKeys, pNeighBox->desc, good_matches);
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
                pNewFrame->mvpMapPoints[newIdx] = pMPi;
                nGood++;
            }
        }
        //P = cv::Mat::eye(4, 4, CV_32FC1);
        if (nGood >= 4) {

            //cv::Mat frame_vis = newframe;
            // RANSAC parameters
            int iterationsCount = 1000;      // number of Ransac iterations.
            float reprojectionError = 8.0;  // maximum allowed distance to consider it an inlier.
            double confidence = 0.9;       // ransac successful confidence.

            // Kalman Filter parameters
            int minInliersKalman = 15;    // Kalman threshold updating
            int pnpMethod = cv::SOLVEPNP_EPNP;
            cv::Mat inliers_idx;
            std::vector<cv::Point2f> list_points2d_inliers;

            int nMeasurements = 6;
            cv::Mat measurements(nMeasurements, 1, CV_64FC1); measurements.setTo(cv::Scalar(0));
            bool good_measurement = false;
            
            std::vector<cv::Point2f> imagePoints;
            std::vector<cv::Point3f> objectPoints;
            cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Point3f avg(0, 0, 0);
            for (int i = 0; i < pNewFrame->N; i++) {
                auto pMPi = pNewFrame->mvpMapPoints[i];
                if (!pMPi || pMPi->isBad())
                    continue;
                auto imgPt = pNewFrame->mvKeysUn[i].pt;
                //cv::Mat Xo = pMPi->GetWorldPos();
                cv::Point3f objPt(pMPi->GetWorldPos());
                imagePoints.push_back(imgPt);
                objectPoints.push_back(objPt);
                avg += objPt;
            }
            avg.x /= objectPoints.size();
            avg.y /= objectPoints.size();
            avg.z /= objectPoints.size();
            for (int i = 0, iend = objectPoints.size(); i < iend; i++)
                objectPoints[i] -= avg;
            
            // -- Step 3: Estimate the pose using RANSAC approach
            pnp_detection.estimatePoseRANSAC(objectPoints, imagePoints,
                K, pnpMethod, inliers_idx,
                iterationsCount, reprojectionError, confidence);

            // -- Step 4: Catch the inliers keypoints to draw
            for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index)
            {
                int n = inliers_idx.at<int>(inliers_index);         // i-inlier
                cv::Point2f point2d = imagePoints[n];     // i-inlier point 2D
                list_points2d_inliers.push_back(point2d);           // add i-inlier to list
            }

            // Draw inliers points 2D
            //draw2DPoints(frame_vis, list_points2d_inliers, cv::Scalar(255, 255, 0));

            // -- Step 5: Kalman Filter
            // GOOD MEASUREMENT
            if (inliers_idx.rows >= minInliersKalman)
            {
                // Get the measured translation
                cv::Mat translation_measured = pnp_detection.get_t_matrix();

                // Get the measured rotation
                cv::Mat rotation_measured = pnp_detection.get_R_matrix();

                // fill the measurements vector
                fillMeasurements(measurements, translation_measured, rotation_measured);
                good_measurement = true;
            }

            // update the Kalman filter with good measurements, otherwise with previous valid measurements
            cv::Mat translation_estimated(3, 1, CV_64FC1);
            cv::Mat rotation_estimated(3, 3, CV_64FC1);
            updateKalmanFilter(KFilter, measurements,
                translation_estimated, rotation_estimated);

            // -- Step 6: Set estimated projection matrix
            pnp_detection_est.set_P_matrix(rotation_estimated, translation_estimated);
            
            if (good_measurement)
            {
                P = pnp_detection_est.get_P_matrix();
            }
            else {
                P = pnp_detection.get_P_matrix();
            }
            
            P.convertTo(P, CV_32FC1);
            std::cout << P << std::endl;
            cv::Mat D = cv::Mat::zeros(4, 1, CV_64FC1);
            
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

                    std::set<EdgeSLAM::MapPoint*> sFound;
                    cv::Mat inliers_idx;
                    std::vector<cv::Point2f> list_points2d_inliers;
                    //ObjectOptimizer::ObjectPoseInitialization(pNewBox, K, D, P, iterationsCount, reprojectionError, confidence, pnpMethod, inliers_idx);
                    nGood = ObjectOptimizer::ObjectPoseOptimization(pNewFrame, P);
                    
                    if (nGood < nBoxMatchThresh) {
                        vbDiscarded[i] = true;
                        nCandidates--;
                        continue;
                    }
                    for (int io = 0; io < pNewFrame->N; io++)
                        if (pNewFrame->mvbOutliers[io])
                            pNewFrame->mvpMapPoints[io] = nullptr;
                        else {
                            auto pMPi = pNewFrame->mvpMapPoints[io];
                            if (pMPi && !pMPi->isBad())
                                sFound.insert(pMPi);
                        }
                    
                    if (nGood < nSuccessTracking)
                    {
                        int nadditional = ObjectSearchPoints::SearchFrameByProjection(pNewFrame, vpCandidateBBs[i], sFound, P, 10, 100);
                        
                        //std::cout << "nadditional = " << nadditional << std::endl;
                        if (nadditional + nGood >= nSuccessTracking)
                        {
                            nGood = ObjectOptimizer::ObjectPoseOptimization(pNewFrame, P);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > nBoxMatchThresh && nGood < nSuccessTracking)
                            {
                                sFound.clear();
                                for (int ip = 0; ip < pNewFrame->N; ip++) {
                                    if (pNewFrame->mvbOutliers[ip])
                                        continue;
                                    sFound.insert(pNewFrame->mvpMapPoints[ip]);
                                }
                                nadditional = ObjectSearchPoints::SearchFrameByProjection(pNewFrame, vpCandidateBBs[i], sFound, P, 3, 64);
                                // Final optimization
                                if (nGood + nadditional >= nSuccessTracking)
                                {
                                    nGood = ObjectOptimizer::ObjectPoseOptimization(pNewFrame, P);
                                    for (int io = 0; io < pNewFrame->N; io++) {
                                        if (pNewFrame->mvbOutliers[io])
                                        {
                                            pNewFrame->mvpMapPoints[io] = nullptr;
                                        }
                                        continue;
                                    }
                                 }
                            }
                        }
                    }//inf ntracking
                    
                    if (nGood >= nSuccessTracking)
                    {
                        good_measurement = true;

                        bMatch = true;
                        break;
                    }
                }//for neighbor boxes
            }//while

            // -- Step X: Draw pose and coordinate frame
            //float l = 5;
            //bool displayFilteredPose = false;
            //std::vector<cv::Point2f> pose_points2d;
            //if (good_measurement) {
            //    pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, 0, 0), K));  // axis center
            //    pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(l, 0, 0), K));  // axis x
            //    pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, l, 0), K));  // axis y
            //    pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, 0, l), K));  // axis z
            //}
            //else {
            //    pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(0, 0, 0), K));  // axis center
            //    pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(l, 0, 0), K));  // axis x
            //    pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(0, l, 0), K));  // axis y
            //    pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(0, 0, l), K));  // axis z
            //}
            //draw3DCoordinateAxes(frame_vis, pose_points2d);           // draw axes
        }
    }

    int DynamicTrackingProcessor::MatchTest(EdgeSLAM::ObjectBoundingBox* pNewBox, std::set<EdgeSLAM::ObjectBoundingBox*> setNeighBoxes, const cv::Mat& newframe, const cv::Mat& K, cv::Mat& P) {
        
        const int nBBs = setNeighBoxes.size();
        int nCandidates = nBBs;
        std::vector<std::vector<EdgeSLAM::MapPoint*> > vvpMapPointMatches(nBBs);

        std::vector<EdgeSLAM::ObjectBoundingBox*> vpCandidateBBs(setNeighBoxes.begin(), setNeighBoxes.end());
        std::vector<bool> vbDiscarded(nBBs, false);
        
        int nBoxMatchThresh = 5;
        int nSuccessTracking = 15;

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
                    continue ;
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

            int nMeasurements = 6;
            cv::Mat measurements(nMeasurements, 1, CV_64FC1); measurements.setTo(cv::Scalar(0));
            bool good_measurement = false;

            std::vector<cv::Point2f> imagePoints;
            std::vector<cv::Point3f> objectPoints;
            cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Point3f avg(0, 0, 0);
            for (int i = 0; i < pNewBox->N; i++) {
                auto pMPi = pNewBox->mvpMapPoints.get(i);
                if (!pMPi || pMPi->isBad())
                    continue;
                auto imgPt = pNewBox->mvKeys[i].pt;
                //cv::Mat Xo = pMPi->GetWorldPos();
                cv::Point3f objPt(pMPi->GetWorldPos());
                imagePoints.push_back(imgPt);
                objectPoints.push_back(objPt);
                avg += objPt;
            }
            avg.x /= objectPoints.size();
            avg.y /= objectPoints.size();
            avg.z /= objectPoints.size();
            for (int i = 0, iend = objectPoints.size(); i < iend; i++)
                objectPoints[i] -= avg;

            // -- Step 3: Estimate the pose using RANSAC approach
            pnp_detection.estimatePoseRANSAC(objectPoints, imagePoints,
                K, pnpMethod, inliers_idx,
                iterationsCount, reprojectionError, confidence);
            
            // -- Step 4: Catch the inliers keypoints to draw
            for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index)
            {
                int n = inliers_idx.at<int>(inliers_index);         // i-inlier
                cv::Point2f point2d = imagePoints[n];     // i-inlier point 2D
                list_points2d_inliers.push_back(point2d);           // add i-inlier to list
            }

            // Draw inliers points 2D
            draw2DPoints(frame_vis, list_points2d_inliers, cv::Scalar(255, 255, 0));
            
            // -- Step 5: Kalman Filter
            // GOOD MEASUREMENT
            if (inliers_idx.rows >= minInliersKalman)
            {
                // Get the measured translation
                cv::Mat translation_measured = pnp_detection.get_t_matrix();

                // Get the measured rotation
                cv::Mat rotation_measured = pnp_detection.get_R_matrix();

                // fill the measurements vector
                fillMeasurements(measurements, translation_measured, rotation_measured);
                good_measurement = true;
            }

            // update the Kalman filter with good measurements, otherwise with previous valid measurements
            cv::Mat translation_estimated(3, 1, CV_64FC1);
            cv::Mat rotation_estimated(3, 3, CV_64FC1);
            updateKalmanFilter(KFilter, measurements,
                translation_estimated, rotation_estimated);

            // -- Step 6: Set estimated projection matrix
            pnp_detection_est.set_P_matrix(rotation_estimated, translation_estimated);

            if (good_measurement)
            {
                P = pnp_detection_est.get_P_matrix();
            }
            else {
                P = pnp_detection.get_P_matrix();
            }
            P.convertTo(P, CV_32FC1);
            cv::Mat D = cv::Mat::zeros(4, 1, CV_64FC1);

            return inliers_idx.rows;

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

                    std::set<EdgeSLAM::MapPoint*> sFound;
                    cv::Mat inliers_idx;
                    std::vector<cv::Point2f> list_points2d_inliers;
                    //ObjectOptimizer::ObjectPoseInitialization(pNewBox, K, D, P, iterationsCount, reprojectionError, confidence, pnpMethod, inliers_idx);
                    nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P);

                    if (nGood < nBoxMatchThresh) {
                        vbDiscarded[i] = true;
                        nCandidates--;
                        continue;
                    }
                    for (int io = 0; io < pNewBox->N; io++)
                        if (pNewBox->mvbOutliers[io])
                            pNewBox->mvpMapPoints.update(io, nullptr);
                        else {
                            auto pMPi = pNewBox->mvpMapPoints.get(io);
                            if (pMPi && !pMPi->isBad())
                                sFound.insert(pMPi);
                        }
                    if (nGood < nSuccessTracking)
                    {
                        int nadditional = ObjectSearchPoints::SearchBoxByProjection(pNewBox, vpCandidateBBs[i], sFound, P, 10, 100);
                        std::cout << "nadditional = " << nadditional << std::endl;
                        if (nadditional + nGood >= nSuccessTracking)
                        {
                            nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > nBoxMatchThresh && nGood < nSuccessTracking)
                            {
                                sFound.clear();
                                for (int ip = 0; ip < pNewBox->N; ip++) {
                                    if (pNewBox->mvbOutliers[ip])
                                        continue;
                                    sFound.insert(pNewBox->mvpMapPoints.get(ip));
                                }
                                nadditional = ObjectSearchPoints::SearchBoxByProjection(pNewBox, vpCandidateBBs[i], sFound, P, 3, 64);
                                // Final optimization
                                if (nGood + nadditional >= nSuccessTracking)
                                {
                                    nGood = ObjectOptimizer::ObjectPoseOptimization(pNewBox, P);
                                    for (int io = 0; io < pNewBox->N; io++) {
                                        if (pNewBox->mvbOutliers[io])
                                        {
                                            pNewBox->mvpMapPoints.update(io, nullptr);
                                        }
                                        continue;
                                    }
                                }
                            }
                        }
                    }//inf ntracking

                    if (nGood >= nSuccessTracking)
                    {
                        good_measurement = true;
                        bRes = true;
                        bMatch = true;
                        break;
                    }
                }//for neighbor boxes
            }//while

            // -- Step X: Draw pose and coordinate frame
            float l = 5;
            bool displayFilteredPose = false;
            std::vector<cv::Point2f> pose_points2d;
            if (good_measurement) {
                pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, 0, 0), K));  // axis center
                pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(l, 0, 0), K));  // axis x
                pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, l, 0), K));  // axis y
                pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, 0, l), K));  // axis z
            }
            else {
                pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(0, 0, 0), K));  // axis center
                pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(l, 0, 0), K));  // axis x
                pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(0, l, 0), K));  // axis y
                pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(0, 0, l), K));  // axis z
            }
            draw3DCoordinateAxes(frame_vis, pose_points2d);           // draw axes
        }
        return 0;
    }
    
    void DynamicTrackingProcessor::MatchTest(EdgeSLAM::ObjectBoundingBox* pNewBox, EdgeSLAM::ObjectBoundingBox* pNeighBox, const cv::Mat& newframe, const cv::Mat& neighframe, const cv::Mat& K) {
        std::vector<cv::DMatch> good_matches;
        rmatcher.robustMatch(newframe, neighframe, pNewBox->mvKeys, pNewBox->desc, pNeighBox->mvKeys, pNeighBox->desc, good_matches);

        /*cv::Mat frame_matching = rmatcher.getImageMatching();
        if (!frame_matching.empty())
        {
            imshow("Keypoints matching", frame_matching);
        }*/

        std::vector<cv::Point3f> list_points3d_model_match; // container for the model 3D coordinates found in the scene
        std::vector<cv::Point2f> list_points2d_scene_match; // container for the model 2D coordinates found in the scene

        int nGood = 0;
        cv::Point3f avg(0, 0, 0);
        for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index)
        {
            int newIdx = good_matches[match_index].queryIdx;
            int neighIdx = good_matches[match_index].trainIdx;

            auto pMPi = pNeighBox->mvpMapPoints.get(neighIdx);
            if (!pMPi || pMPi->isBad())
                continue;
            auto pt = pNewBox->mvKeys[newIdx].pt;
            nGood++;
            cv::Point3f point3d_model(pMPi->GetWorldPos());
            cv::Point2f point2d_scene = pt;
            list_points3d_model_match.push_back(point3d_model);         // add 3D point
            list_points2d_scene_match.push_back(point2d_scene);         // add 2D point
            avg += point3d_model;
        }

        //std::cout << "dyna test = " << nGood <<" "<< good_matches.size() << std::endl;

        // RANSAC parameters
        int iterationsCount = 500;      // number of Ransac iterations.
        float reprojectionError = 6.0;  // maximum allowed distance to consider it an inlier.
        double confidence = 0.99;       // ransac successful confidence.

        // Kalman Filter parameters
        int minInliersKalman = 30;    // Kalman threshold updating
        int pnpMethod = cv::SOLVEPNP_ITERATIVE;
        cv::Mat inliers_idx;
        std::vector<cv::Point2f> list_points2d_inliers;

        int nMeasurements = 6;
        cv::Mat measurements(nMeasurements, 1, CV_64FC1); measurements.setTo(cv::Scalar(0));
        bool good_measurement = false;

        cv::Mat frame_vis = newframe.clone();
        if (nGood >= 4) // OpenCV requires solvePnPRANSAC to minimally have 4 set of points
        {
            // -- Step 3: Estimate the pose using RANSAC approach
            pnp_detection.estimatePoseRANSAC(list_points3d_model_match, list_points2d_scene_match,
                K, pnpMethod, inliers_idx,
                iterationsCount, reprojectionError, confidence);
            std::cout << "inlier test = " << inliers_idx.rows << std::endl;
            // -- Step 4: Catch the inliers keypoints to draw
            for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index)
            {
                int n = inliers_idx.at<int>(inliers_index);         // i-inlier
                cv::Point2f point2d = list_points2d_scene_match[n];     // i-inlier point 2D
                list_points2d_inliers.push_back(point2d);           // add i-inlier to list
            }

            // Draw inliers points 2D
            draw2DPoints(frame_vis, list_points2d_inliers, cv::Scalar(255,255,0));

            // -- Step 5: Kalman Filter
            // GOOD MEASUREMENT
            if (inliers_idx.rows >= minInliersKalman)
            {
                // Get the measured translation
                cv::Mat translation_measured = pnp_detection.get_t_matrix();

                // Get the measured rotation
                cv::Mat rotation_measured = pnp_detection.get_R_matrix();

                // fill the measurements vector
                fillMeasurements(measurements, translation_measured, rotation_measured);
                good_measurement = true;
            }

            // update the Kalman filter with good measurements, otherwise with previous valid measurements
            cv::Mat translation_estimated(3, 1, CV_64FC1);
            cv::Mat rotation_estimated(3, 3, CV_64FC1);
            updateKalmanFilter(KFilter, measurements,
                translation_estimated, rotation_estimated);

            // -- Step 6: Set estimated projection matrix
            pnp_detection_est.set_P_matrix(rotation_estimated, translation_estimated);
            std::cout << pnp_detection.get_P_matrix() << " " << pnp_detection_est.get_P_matrix() << std::endl;
            // -- Step X: Draw pose and coordinate frame
            float l = 5;
            bool displayFilteredPose = false;
            std::vector<cv::Point2f> pose_points2d;
            if (!good_measurement || displayFilteredPose)
            {
                pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, 0, 0),K));  // axis center
                pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(l, 0, 0),K));  // axis x
                pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, l, 0),K));  // axis y
                pose_points2d.push_back(pnp_detection_est.backproject3DPoint(cv::Point3f(0, 0, l),K));  // axis z
                draw3DCoordinateAxes(frame_vis, pose_points2d);           // draw axes
            }
            else
            {
                pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(0, 0, 0), K));  // axis center
                pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(l, 0, 0), K));  // axis x
                pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(0, l, 0), K));  // axis y
                pose_points2d.push_back(pnp_detection.backproject3DPoint(cv::Point3f(0, 0, l), K));  // axis z
                draw3DCoordinateAxes(frame_vis, pose_points2d);           // draw axes
            }
        }
        //cv::imshow("REAL TIME DEMO", frame_vis); cv::waitKey(30);
    }

    void DynamicTrackingProcessor::initKalmanFilter(cv::KalmanFilter& KF, int nStates, int nMeasurements, int nInputs, double dt)
    {
        KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter

        setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));       // set process noise
        setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-2));   // set measurement noise
        setIdentity(KF.errorCovPost, cv::Scalar::all(1));             // error covariance

        /** DYNAMIC MODEL **/

        //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
        //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
        //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
        //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
        //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
        //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
        //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
        //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
        //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
        //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
        //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]

        // position
        KF.transitionMatrix.at<double>(0, 3) = dt;
        KF.transitionMatrix.at<double>(1, 4) = dt;
        KF.transitionMatrix.at<double>(2, 5) = dt;
        KF.transitionMatrix.at<double>(3, 6) = dt;
        KF.transitionMatrix.at<double>(4, 7) = dt;
        KF.transitionMatrix.at<double>(5, 8) = dt;
        KF.transitionMatrix.at<double>(0, 6) = 0.5 * pow(dt, 2);
        KF.transitionMatrix.at<double>(1, 7) = 0.5 * pow(dt, 2);
        KF.transitionMatrix.at<double>(2, 8) = 0.5 * pow(dt, 2);

        // orientation
        KF.transitionMatrix.at<double>(9, 12) = dt;
        KF.transitionMatrix.at<double>(10, 13) = dt;
        KF.transitionMatrix.at<double>(11, 14) = dt;
        KF.transitionMatrix.at<double>(12, 15) = dt;
        KF.transitionMatrix.at<double>(13, 16) = dt;
        KF.transitionMatrix.at<double>(14, 17) = dt;
        KF.transitionMatrix.at<double>(9, 15) = 0.5 * pow(dt, 2);
        KF.transitionMatrix.at<double>(10, 16) = 0.5 * pow(dt, 2);
        KF.transitionMatrix.at<double>(11, 17) = 0.5 * pow(dt, 2);


        /** MEASUREMENT MODEL **/

        //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
        //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
        //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]

        KF.measurementMatrix.at<double>(0, 0) = 1;  // x
        KF.measurementMatrix.at<double>(1, 1) = 1;  // y
        KF.measurementMatrix.at<double>(2, 2) = 1;  // z
        KF.measurementMatrix.at<double>(3, 9) = 1;  // roll
        KF.measurementMatrix.at<double>(4, 10) = 1; // pitch
        KF.measurementMatrix.at<double>(5, 11) = 1; // yaw
    }

    /**********************************************************************************************************/
    void DynamicTrackingProcessor::updateKalmanFilter(cv::KalmanFilter& KF, cv::Mat& measurement, cv::Mat& translation_estimated, cv::Mat& rotation_estimated)
    {
        // First predict, to update the internal statePre variable
        cv::Mat prediction = KF.predict();

        // The "correct" phase that is going to use the predicted value and our measurement
        cv::Mat estimated = KF.correct(measurement);

        // Estimated translation
        translation_estimated.at<double>(0) = estimated.at<double>(0);
        translation_estimated.at<double>(1) = estimated.at<double>(1);
        translation_estimated.at<double>(2) = estimated.at<double>(2);

        // Estimated euler angles
        cv::Mat eulers_estimated(3, 1, CV_64F);
        eulers_estimated.at<double>(0) = estimated.at<double>(9);
        eulers_estimated.at<double>(1) = estimated.at<double>(10);
        eulers_estimated.at<double>(2) = estimated.at<double>(11);

        // Convert estimated quaternion to rotation matrix
        rotation_estimated = euler2rot(eulers_estimated);
    }

    /**********************************************************************************************************/
    void DynamicTrackingProcessor::fillMeasurements(cv::Mat& measurements, const cv::Mat& translation_measured, const cv::Mat& rotation_measured)
    {
        // Convert rotation matrix to euler angles
        cv::Mat measured_eulers(3, 1, CV_64F);
        measured_eulers = rot2euler(rotation_measured);

        // Set measurement to predict
        measurements.at<double>(0) = translation_measured.at<double>(0); // x
        measurements.at<double>(1) = translation_measured.at<double>(1); // y
        measurements.at<double>(2) = translation_measured.at<double>(2); // z
        measurements.at<double>(3) = measured_eulers.at<double>(0);      // roll
        measurements.at<double>(4) = measured_eulers.at<double>(1);      // pitch
        measurements.at<double>(5) = measured_eulers.at<double>(2);      // yaw
    }

    cv::Mat DynamicTrackingProcessor::rot2euler(const cv::Mat& rotationMatrix)
    {
        cv::Mat euler(3, 1, CV_64F);

        double m00 = rotationMatrix.at<double>(0, 0);
        double m02 = rotationMatrix.at<double>(0, 2);
        double m10 = rotationMatrix.at<double>(1, 0);
        double m11 = rotationMatrix.at<double>(1, 1);
        double m12 = rotationMatrix.at<double>(1, 2);
        double m20 = rotationMatrix.at<double>(2, 0);
        double m22 = rotationMatrix.at<double>(2, 2);

        double bank, attitude, heading;

        // Assuming the angles are in radians.
        if (m10 > 0.998) { // singularity at north pole
            bank = 0;
            attitude = CV_PI / 2;
            heading = atan2(m02, m22);
        }
        else if (m10 < -0.998) { // singularity at south pole
            bank = 0;
            attitude = -CV_PI / 2;
            heading = atan2(m02, m22);
        }
        else
        {
            bank = atan2(-m12, m11);
            attitude = asin(m10);
            heading = atan2(-m20, m00);
        }

        euler.at<double>(0) = bank;
        euler.at<double>(1) = attitude;
        euler.at<double>(2) = heading;

        return euler;
    }

    cv::Mat DynamicTrackingProcessor::euler2rot(const cv::Mat& euler)
    {
        cv::Mat rotationMatrix(3, 3, CV_64F);

        double bank = euler.at<double>(0);
        double attitude = euler.at<double>(1);
        double heading = euler.at<double>(2);

        // Assuming the angles are in radians.
        double ch = cos(heading);
        double sh = sin(heading);
        double ca = cos(attitude);
        double sa = sin(attitude);
        double cb = cos(bank);
        double sb = sin(bank);

        double m00, m01, m02, m10, m11, m12, m20, m21, m22;

        m00 = ch * ca;
        m01 = sh * sb - ch * sa * cb;
        m02 = ch * sa * sb + sh * cb;
        m10 = sa;
        m11 = ca * cb;
        m12 = -ca * sb;
        m20 = -sh * ca;
        m21 = sh * sa * cb + ch * sb;
        m22 = -sh * sa * sb + ch * cb;

        rotationMatrix.at<double>(0, 0) = m00;
        rotationMatrix.at<double>(0, 1) = m01;
        rotationMatrix.at<double>(0, 2) = m02;
        rotationMatrix.at<double>(1, 0) = m10;
        rotationMatrix.at<double>(1, 1) = m11;
        rotationMatrix.at<double>(1, 2) = m12;
        rotationMatrix.at<double>(2, 0) = m20;
        rotationMatrix.at<double>(2, 1) = m21;
        rotationMatrix.at<double>(2, 2) = m22;

        return rotationMatrix;
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