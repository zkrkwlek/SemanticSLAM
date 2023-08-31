
#ifndef DYNAMIC_OBJECT_MAP_H
#define DYNAMIC_OBJECT_MAP_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

class KalmanFilter;

namespace SemanticSLAM {
    class DynamicObjectMap
    {
    public:
        DynamicObjectMap();
        virtual ~DynamicObjectMap();

        KalmanFilter* mpKalmanFilter;

        void SetPose(cv::Mat _P);
        cv::Mat GetPose();

    private:
        std::mutex mMutexPose;
        cv::Mat _P;
    };
}



#endif /* PNPPROBLEM_H_ */
