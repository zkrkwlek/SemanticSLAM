#include <ObjectSearchPoints.h>
#include <SLAM.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <FeatureTracker.h>
#include <ObjectFrame.h>
#include <Utils.h>

namespace SemanticSLAM {
	const int ObjectSearchPoints::HISTO_LENGTH = 30;
	int ObjectSearchPoints::SearchObjectNodeAndBox(EdgeSLAM::ObjectNode* pNode, EdgeSLAM::ObjectBoundingBox* pBox, std::vector<std::pair<int, int>>& matches, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri) {
		int nmatches = 0;
		std::vector<bool> bMatches = std::vector<bool>(pBox->N, false);

		//auto mspMPs = pNode->mspMPs.Get();
		//auto mvpMPs = std::vector<EdgeSLAM::ObjectMapPoint*>(mspMPs.begin(), mspMPs.end());
		auto mvpMPs = pNode->mspMPs.ConvertVector();

		int Ntmp = 0;

		for (int i = 0, iend = mvpMPs.size(); i < iend; i++) {
			int idxObj = i;
			auto pMPi = mvpMPs[i];

			if (!pMPi || pMPi->isBad())
				continue;
			const cv::Mat& dObj = pMPi->GetDescriptor();

			int bestDist1 = 256;
			int bestIdxF = -1;
			int bestDist2 = 256;

			for (int j = 0, jend = pBox->N; j < jend; j++) {
				int idxFrame = j;
				if (bMatches[idxFrame])
					continue;
				auto pMPj = pBox->mvpObjectPoints.get(idxFrame);
				if (pMPj && !pMPj->isBad())
					continue;

				const cv::Mat& dFrame = pBox->desc.row(idxFrame);

				const int dist = (int)Matcher->DescriptorDistance(dObj, dFrame);
				//std::cout <<"a "<< dist << std::endl;
				if (dist < bestDist1)
				{
					bestDist2 = bestDist1;
					bestDist1 = dist;
					bestIdxF = idxFrame;
				}
				else if (dist < bestDist2)
				{
					bestDist2 = dist;
				}
			}//for frame

			if (bestDist1 <= thMinDesc)
			{
				Ntmp++;
				if (static_cast<float>(bestDist1) < thProjection * static_cast<float>(bestDist2))
				{
					bMatches[bestIdxF] = true;
					matches.push_back(std::make_pair((int)idxObj, (int)bestIdxF));
					nmatches++;
				}
			}
			//std::cout << "matching test = " << thMinDesc << "||" << Ntmp << " " << nmatches << std::endl;
		}
		return nmatches;
	}
	//박스1은 노드, 박스2는 새로운 박스
	int ObjectSearchPoints::SearchObjectBoxAndBoxForTracking(EdgeSLAM::ObjectBoundingBox* pBox1, EdgeSLAM::ObjectBoundingBox* pBox2, std::vector<std::pair<int, int>>& matches, float thMinDesc, float thProjection) {
		int nmatches = 0;
		std::vector<bool> bMatches = std::vector<bool>(pBox2->N, false);

		for (int i = 0, iend = pBox1->N; i < iend; i++) {
			int idxObj = i;
			auto pMPi = pBox1->mvpMapPoints.get(idxObj);
			if (!pMPi || pMPi->isBad())
				continue;
			//해결해야 하는 상황이 될 듯
			if (pBox2->mspMapPoints.Count(pMPi))
				continue;
			//const cv::Mat& dObj = pBox1->desc.row(idxObj);
			const cv::Mat& dObj = pMPi->GetDescriptor();

			int bestDist1 = 256;
			int bestIdxF = -1;
			int bestDist2 = 256;

			for (int j = 0, jend = pBox2->N; j < jend; j++) {
				int idxFrame = j;

				auto pMPj = pBox2->mvpMapPoints.get(idxFrame);
				if (pMPj && !pMPj->isBad())
					continue;
				if (bMatches[idxFrame])
					continue;
				const cv::Mat& dFrame = pBox2->desc.row(idxFrame);

				const int dist = (int)Matcher->DescriptorDistance(dObj, dFrame);
				//std::cout <<"a "<< dist << std::endl;
				if (dist < bestDist1)
				{
					bestDist2 = bestDist1;
					bestDist1 = dist;
					bestIdxF = idxFrame;
				}
				else if (dist < bestDist2)
				{
					bestDist2 = dist;
				}
			}//for frame

			if (bestDist1 <= thMinDesc)
			{
				if (static_cast<float>(bestDist1) < thProjection * static_cast<float>(bestDist2))
				{
					bMatches[bestIdxF] = true;
					matches.push_back(std::make_pair((int)idxObj, (int)bestIdxF));
					nmatches++;
				}
			}

		}//for obj

		return nmatches;
	}
	int ObjectSearchPoints::SearchObjectBoxAndBoxForTriangulation(EdgeSLAM::ObjectBoundingBox* pBox1, EdgeSLAM::ObjectBoundingBox* pBox2, std::vector<std::pair<int, int>>& matches, const cv::Mat& F12, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri) {

		//Compute epipole in second image
		auto pKF1 = pBox1->mpKF;
		auto pKF2 = pBox2->mpKF;

		cv::Mat Cw = pKF1->GetCameraCenter();
		cv::Mat R2w = pKF2->GetRotation();
		cv::Mat t2w = pKF2->GetTranslation();
		cv::Mat C2 = R2w * Cw + t2w;
		const float invz = 1.0f / C2.at<float>(2);
		const float ex = pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;
		const float ey = pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;

		int nmatches = 0;
		std::vector<bool> bMatches = std::vector<bool>(pBox2->N, false);

		for (int i = 0, iend = pBox1->N; i < iend; i++) {
			int idxObj = i;
			auto pMPi = pBox1->mvpObjectPoints.get(idxObj);
			if (pMPi && !pMPi->isBad())
				continue;
			const cv::Mat& dObj = pBox1->desc.row(idxObj);

			int bestDist1 = 256;
			int bestIdxF = -1;
			int bestDist2 = 256;

			for (int j = 0, jend = pBox2->N; j < jend; j++) {
				int idxFrame = j;

				auto pMPj = pBox2->mvpObjectPoints.get(idxFrame);
				if (pMPj && !pMPj->isBad())
					continue;
				if (bMatches[idxFrame])
					continue;
				const cv::Mat& dFrame = pBox2->desc.row(idxFrame);

				const int dist = (int)Matcher->DescriptorDistance(dObj, dFrame);
				//std::cout <<"a "<< dist << std::endl;
				if (dist < bestDist1)
				{
					bestDist2 = bestDist1;
					bestDist1 = dist;
					bestIdxF = idxFrame;
				}
				else if (dist < bestDist2)
				{
					bestDist2 = dist;
				}

				const cv::KeyPoint& kp2 = pBox2->mvKeys[idxFrame];

				{
					const float distex = ex - kp2.pt.x;
					const float distey = ey - kp2.pt.y;
					if (distex * distex + distey * distey < 100 * pKF2->mvScaleFactors[kp2.octave])
						continue;
				}
				const cv::KeyPoint& kp1 = pBox1->mvKeys[idxObj];
				if (Utils::CheckDistEpipolarLine(kp1, kp2, F12, pKF2->mvLevelSigma2))
				{
					bestIdxF = idxFrame;
					bestDist1 = dist;
				}

			}//for frame

			/*if (bestDist1 <= thMinDesc)
			{
				if (static_cast<float>(bestDist1) < thProjection * static_cast<float>(bestDist2))
				{
					bMatches[bestIdxF] = true;
					matches.push_back(std::make_pair((int)idxObj, (int)bestIdxF));
					nmatches++;
				}
			}*/

			if (bestIdxF >= 0)
			{
				matches.push_back(std::make_pair((int)idxObj, (int)bestIdxF));
				nmatches++;

				/*if (bCheckOri)
				{
					float rot = kp1.angle - kp2.angle;
					if (rot < 0.0)
						rot += 360.0f;
					int bin = round(rot * factor);
					if (bin == HISTO_LENGTH)
						bin = 0;
					assert(bin >= 0 && bin < HISTO_LENGTH);
					rotHist[bin].push_back(idx1);
				}*/
			}
		}//for obj

		return nmatches;
	}
	int ObjectSearchPoints::SearchObject(const cv::Mat& f1, const cv::Mat& f2, std::vector<std::pair<int, int>>& matches, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri) {

		int nmatches = 0;
		std::vector<bool> bMatches = std::vector<bool>(f2.rows, false);

		for (int i = 0, iend = f1.rows; i < iend; i++) {
			int idxObj = i;
			const cv::Mat& dObj = f1.row(idxObj);

			int bestDist1 = 256;
			int bestIdxF = -1;
			int bestDist2 = 256;

			for (int j = 0, jend = f2.rows; j < jend; j++) {
				int idxFrame = j;
				if (bMatches[idxFrame])
					continue;
				const cv::Mat& dFrame = f2.row(idxFrame);

				const int dist = (int)Matcher->DescriptorDistance(dObj, dFrame);
				//std::cout <<"a "<< dist << std::endl;
				if (dist < bestDist1)
				{
					bestDist2 = bestDist1;
					bestDist1 = dist;
					bestIdxF = idxFrame;
				}
				else if (dist < bestDist2)
				{
					bestDist2 = dist;
				}
			}//for frame

			if (bestDist1 <= thMinDesc)
			{
				if (static_cast<float>(bestDist1) < thProjection * static_cast<float>(bestDist2))
				{
					bMatches[bestIdxF] = true;
					matches.push_back(std::make_pair((int)idxObj, (int)bestIdxF));
					nmatches++;
				}
			}

		}//for obj

		return nmatches;
	}
	//int ObjectSearchPoints::SearchObject(ObjectNode* obj, Frame* curr, std::vector<std::pair<int, int>>& matches, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri) {
	//	int nmatches = 0;

	//	// Rotation Histogram (to check rotation consistency)
	//	std::vector<int> rotHist[HISTO_LENGTH];
	//	const float factor = 1.0f / HISTO_LENGTH;

	//	std::vector<bool> bMatches = std::vector<bool>(curr->N, false);
	//	DBoW3::FeatureVector::const_iterator KFit = obj->mFeatVec.begin();
	//	DBoW3::FeatureVector::const_iterator KFend = obj->mFeatVec.end();
	//	DBoW3::FeatureVector::const_iterator Fit = curr->mFeatVec.begin();
	//	DBoW3::FeatureVector::const_iterator Fend = curr->mFeatVec.end();

	//	auto objDesc = obj->GetDescriptor();

	//	while (KFit != KFend && Fit != Fend)
	//	{
	//		if (KFit->first == Fit->first)
	//		{
	//			const std::vector<unsigned int> vIndicesKF = KFit->second;
	//			const std::vector<unsigned int> vIndicesF = Fit->second;

	//			for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
	//			{
	//				const unsigned int realIdxKF = vIndicesKF[iKF];

	//				const cv::Mat& dKF = objDesc.row(realIdxKF);

	//				int bestDist1 = 256;
	//				int bestIdxF = -1;
	//				int bestDist2 = 256;

	//				for (size_t iF = 0; iF < vIndicesF.size(); iF++)
	//				{
	//					const unsigned int realIdxF = vIndicesF[iF];

	//					if (bMatches[realIdxF])
	//						continue;

	//					const cv::Mat& dF = curr->mDescriptors.row(realIdxF);

	//					const int dist = (int)Matcher->DescriptorDistance(dKF, dF);
	//					//std::cout <<"a "<< dist << std::endl;
	//					if (dist < bestDist1)
	//					{
	//						bestDist2 = bestDist1;
	//						bestDist1 = dist;
	//						bestIdxF = realIdxF;
	//					}
	//					else if (dist < bestDist2)
	//					{
	//						bestDist2 = dist;
	//					}
	//				}

	//				if (bestDist1 <= thMinDesc)
	//				{
	//					if (static_cast<float>(bestDist1) < thProjection * static_cast<float>(bestDist2))
	//					{
	//						bMatches[bestIdxF] = true;
	//						matches.push_back(std::make_pair((int)realIdxKF, (int)bestIdxF));
	//						nmatches++;
	//					}
	//				}

	//			}

	//			KFit++;
	//			Fit++;
	//		}
	//		else if (KFit->first < Fit->first)
	//		{
	//			KFit = obj->mFeatVec.lower_bound(Fit->first);
	//		}
	//		else
	//		{
	//			Fit = curr->mFeatVec.lower_bound(KFit->first);
	//		}
	//	}
	//	std::cout << "obj match = " << nmatches << std::endl;

	//	/*for (int i = 0; i < obj.rows; i++)
	//	{

	//		int bestDist = 256;
	//		int bestIdx2 = -1;
	//		const cv::Mat &d1 = obj.row(i);
	//		for (size_t j = 0, jend = curr->N; j < jend; j++) {
	//			const cv::Mat &d2 = curr->mDescriptors.row(j);

	//			const int dist = (int)curr->matcher->DescriptorDistance(d1, d2);

	//			if (dist < bestDist)
	//			{
	//				bestDist = dist;
	//				bestIdx2 = j;
	//			}
	//		}
	//		if (bestDist <= thMaxDesc)
	//		{
	//			nmatches++;
	//			matches.push_back(std::make_pair(i, bestIdx2));
	//		}
	//	}*/
	//	/*for (int i = 0; i < HISTO_LENGTH; i++)
	//		std::vector<int>().swap(rotHist[i]);*/
	//	return nmatches;
	//}

}