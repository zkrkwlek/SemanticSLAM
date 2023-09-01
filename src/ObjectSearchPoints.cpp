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

	int ObjectSearchPoints::SearchBoxByBoW(EdgeSLAM::ObjectBoundingBox* pBB1, EdgeSLAM::ObjectBoundingBox* pBB2, std::vector<EdgeSLAM::MapPoint*>& vpMapPointMatches, float thMinDesc, float thMatchRatio, bool bCheckOri)
	{
		const auto vpMapPointsKF = pBB1->mvpMapPoints.get();
		vpMapPointMatches = std::vector<EdgeSLAM::MapPoint*>(pBB2->N, static_cast<EdgeSLAM::MapPoint*>(nullptr));
		const DBoW3::FeatureVector& vFeatVecKF = pBB1->mFeatVec;
		int nmatches = 0;

		std::vector<int> rotHist[HISTO_LENGTH];
		const float factor = 1.0f / HISTO_LENGTH;

		DBoW3::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
		DBoW3::FeatureVector::const_iterator Fit = pBB2->mFeatVec.begin();
		DBoW3::FeatureVector::const_iterator KFend = vFeatVecKF.end();
		DBoW3::FeatureVector::const_iterator Fend = pBB2->mFeatVec.end();

		while (KFit != KFend && Fit != Fend)
		{
			if (KFit->first == Fit->first)
			{
				const std::vector<unsigned int> vIndicesKF = KFit->second;
				const std::vector<unsigned int> vIndicesF = Fit->second;

				for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
				{
					const unsigned int realIdxKF = vIndicesKF[iKF];

					EdgeSLAM::MapPoint* pMP = vpMapPointsKF[realIdxKF];

					if (!pMP || pMP->isBad())
						continue;

					const cv::Mat& dKF = pBB1->desc.row(realIdxKF);

					int bestDist1 = 256;
					int bestIdxF = -1;
					int bestDist2 = 256;

					for (size_t iF = 0; iF < vIndicesF.size(); iF++)
					{
						const unsigned int realIdxF = vIndicesF[iF];

						if (vpMapPointMatches[realIdxF])
							continue;

						const cv::Mat& dF = pBB2->desc.row(realIdxF);

						const int dist = (int)Matcher->DescriptorDistance(dKF, dF);

						if (dist < bestDist1)
						{
							bestDist2 = bestDist1;
							bestDist1 = dist;
							bestIdxF = realIdxF;
						}
						else if (dist < bestDist2)
						{
							bestDist2 = dist;
						}
					}

					if (bestDist1 <= thMinDesc)
					{
						if (static_cast<float>(bestDist1) < thMatchRatio * static_cast<float>(bestDist2))
						{
							vpMapPointMatches[bestIdxF] = pMP;

							const cv::KeyPoint& kp = pBB1->mvKeys[realIdxKF];

							if (bCheckOri)
							{
								float rot = kp.angle - pBB2->mvKeys[bestIdxF].angle;
								if (rot < 0.0)
									rot += 360.0f;
								int bin = round(rot * factor);
								if (bin == HISTO_LENGTH)
									bin = 0;
								assert(bin >= 0 && bin < HISTO_LENGTH);
								rotHist[bin].push_back(bestIdxF);
							}
							nmatches++;
						}
					}

				}

				KFit++;
				Fit++;
			}
			else if (KFit->first < Fit->first)
			{
				KFit = vFeatVecKF.lower_bound(Fit->first);
			}
			else
			{
				Fit = pBB2->mFeatVec.lower_bound(KFit->first);
			}
		}

		if (bCheckOri)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;

			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i < HISTO_LENGTH; i++)
			{
				if (i == ind1 || i == ind2 || i == ind3)
					continue;
				for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
				{
					vpMapPointMatches[rotHist[i][j]] = static_cast<EdgeSLAM::MapPoint*>(nullptr);
					nmatches--;
				}
			}
		}

		return nmatches;
	}

	int ObjectSearchPoints::SearchObjectMapByProjection(std::vector<std::pair<int, int>>& matches, EdgeSLAM::Frame* pF, const std::vector<EdgeSLAM::MapPoint*>& vpLocalMapPoints, const std::set<EdgeSLAM::MapPoint*>& sAlreadyFound, cv::Mat P, cv::Mat origin, const float th, const int ORBdist, bool bCheckOri) {
		
		std::vector<bool> bMatches = std::vector<bool>(pF->N, false);
		cv::Mat Rcw = P.rowRange(0, 3).colRange(0, 3);
		cv::Mat tcw = P.rowRange(0, 3).col(3);
		cv::Mat Ow = -Rcw.t() * tcw;

		std::vector<int> rotHist[HISTO_LENGTH];
		const float factor = 1.0f / HISTO_LENGTH;

		float fx = pF->fx;
		float fy = pF->fy;
		float cx = pF->cx;
		float cy = pF->cy;
		float nMinX = pF->mnMinX;
		float nMinY = pF->mnMinY;
		float nMaxX = pF->mnMaxX;
		float nMaxY = pF->mnMaxY;

		int nmatches = 0;
		for (size_t i = 0, iend = vpLocalMapPoints.size(); i < iend; i++)
		{
			auto pMP = vpLocalMapPoints[i];

			if (pMP)
			{
				if (!pMP->isBad() && !sAlreadyFound.count(pMP))
				{
					//Project
					cv::Mat x3Dw = pMP->GetWorldPos()-origin;
					cv::Mat x3Dc = Rcw * x3Dw + tcw;

					const float xc = x3Dc.at<float>(0);
					const float yc = x3Dc.at<float>(1);
					const float invzc = 1.0 / x3Dc.at<float>(2);

					const float u = fx * xc * invzc + cx;
					const float v = fy * yc * invzc + cy;

					if (u<nMinX || u>nMaxX)
						continue;
					if (v<nMinY || v>nMaxY)
						continue;

					// Compute predicted scale level
					cv::Mat PO = x3Dw - Ow;
					float dist3D = cv::norm(PO);

					const float maxDistance = pMP->GetMaxDistanceInvariance();
					const float minDistance = pMP->GetMinDistanceInvariance();

					// Depth must be inside the scale pyramid of the image
					if (dist3D<minDistance || dist3D>maxDistance)
						continue;

					int nPredictedLevel = pMP->PredictScale(dist3D, pF);

					// Search in a window
					const float radius = th * pF->mvScaleFactors[nPredictedLevel];

					const auto vIndices2 = pF->GetFeaturesInArea(u, v, radius, nPredictedLevel - 1, nPredictedLevel + 1);

					if (vIndices2.empty())
						continue;

					const cv::Mat dMP = pMP->GetDescriptor();

					int bestDist = 256;
					int bestIdx2 = -1;

					for (auto vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
					{
						const size_t i2 = *vit;
						int newIDX = i2;

						if (bMatches[newIDX])
							continue;

						const cv::Mat& d = pF->mDescriptors.row(newIDX);

						const int dist = (int)Matcher->DescriptorDistance(dMP, d);

						if (dist < bestDist)
						{
							bestDist = dist;
							bestIdx2 = newIDX;
						}
					}

					if (bestDist <= ORBdist)
					{
						bMatches[bestIdx2] = true;
						matches.push_back(std::make_pair((int)i, (int)bestIdx2));
						nmatches++;

						/*if (bCheckOri)
						{
							float rot = pKeyBox->mvKeys[i].angle - pF->mvKeysUn[bestIdx2].angle;
							if (rot < 0.0)
								rot += 360.0f;
							int bin = round(rot * factor);
							if (bin == HISTO_LENGTH)
								bin = 0;
							assert(bin >= 0 && bin < HISTO_LENGTH);
							rotHist[bin].push_back(bestIdx2);
						}*/
					}

				}
			}
		}

		if (bCheckOri)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;

			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i < HISTO_LENGTH; i++)
			{
				if (i != ind1 && i != ind2 && i != ind3)
				{
					for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
					{
						pF->mvpMapPoints[rotHist[i][j]] = nullptr;
						nmatches--;
					}
				}
			}
		}
		return nmatches;
	}

	int ObjectSearchPoints::SearchFrameByProjection(EdgeSLAM::Frame* pF, EdgeSLAM::ObjectBoundingBox* pKeyBox, const std::set<EdgeSLAM::MapPoint*>& sAlreadyFound, cv::Mat P, const float th, const int ORBdist, bool bCheckOri)
	{
		cv::Mat Rcw = P.rowRange(0, 3).colRange(0, 3);
		cv::Mat tcw = P.rowRange(0, 3).col(3);
		cv::Mat Ow = -Rcw.t() * tcw;

		std::vector<int> rotHist[HISTO_LENGTH];
		const float factor = 1.0f / HISTO_LENGTH;

		const auto vpMPs = pKeyBox->mvpMapPoints.get();
				
		float fx = pF->fx;
		float fy = pF->fy;
		float cx = pF->cx;
		float cy = pF->cy;
		float nMinX = pF->mnMinX;
		float nMinY = pF->mnMinY;
		float nMaxX = pF->mnMaxX;
		float nMaxY = pF->mnMaxY;

		int nmatches = 0;
		for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
		{
			auto pMP = vpMPs[i];

			if (pMP)
			{
				if (!pMP->isBad() && !sAlreadyFound.count(pMP))
				{
					//Project
					cv::Mat x3Dw = pMP->GetWorldPos();
					cv::Mat x3Dc = Rcw * x3Dw + tcw;

					const float xc = x3Dc.at<float>(0);
					const float yc = x3Dc.at<float>(1);
					const float invzc = 1.0 / x3Dc.at<float>(2);

					const float u = fx * xc * invzc + cx;
					const float v = fy * yc * invzc + cy;

					if (u<nMinX || u>nMaxX)
						continue;
					if (v<nMinY || v>nMaxY)
						continue;

					// Compute predicted scale level
					cv::Mat PO = x3Dw - Ow;
					float dist3D = cv::norm(PO);

					const float maxDistance = pMP->GetMaxDistanceInvariance();
					const float minDistance = pMP->GetMinDistanceInvariance();

					// Depth must be inside the scale pyramid of the image
					if (dist3D<minDistance || dist3D>maxDistance)
						continue;

					int nPredictedLevel = pMP->PredictScale(dist3D, pF);
					if (nPredictedLevel >= pF->mvScaleFactors.size()) {
						std::cout << "Error = Predicted Level " << nPredictedLevel << " " << pF->mvScaleFactors.size() << std::endl;
						continue;
					}
					// Search in a window
					const float radius = th * pF->mvScaleFactors[nPredictedLevel];

					const auto vIndices2 = pF->GetFeaturesInArea(u, v, radius, nPredictedLevel - 1, nPredictedLevel + 1);

					if (vIndices2.empty())
						continue;

					const cv::Mat dMP = pMP->GetDescriptor();

					int bestDist = 256;
					int bestIdx2 = -1;

					for (auto vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
					{
						const size_t i2 = *vit;
						int newIDX = i2;

						if (pF->mvpMapPoints[newIDX])
							continue;

						const cv::Mat& d = pF->mDescriptors.row(newIDX);

						const int dist = (int)Matcher->DescriptorDistance(dMP, d);

						if (dist < bestDist)
						{
							bestDist = dist;
							bestIdx2 = newIDX;
						}
					}

					if (bestDist <= ORBdist)
					{
						pF->mvpMapPoints[bestIdx2] =pMP;
						nmatches++;

						if (bCheckOri)
						{
							float rot = pKeyBox->mvKeys[i].angle - pF->mvKeysUn[bestIdx2].angle;
							if (rot < 0.0)
								rot += 360.0f;
							int bin = round(rot * factor);
							if (bin == HISTO_LENGTH)
								bin = 0;
							assert(bin >= 0 && bin < HISTO_LENGTH);
							rotHist[bin].push_back(bestIdx2);
						}
					}

				}
			}
		}

		if (bCheckOri)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;

			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i < HISTO_LENGTH; i++)
			{
				if (i != ind1 && i != ind2 && i != ind3)
				{
					for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
					{
						pF->mvpMapPoints[rotHist[i][j]] = nullptr;
						nmatches--;
					}
				}
			}
		}
		return nmatches;
	}

	int ObjectSearchPoints::SearchBoxByProjection(EdgeSLAM::ObjectBoundingBox* pNewBox, EdgeSLAM::ObjectBoundingBox* pKeyBox, const std::set<EdgeSLAM::MapPoint*>& sAlreadyFound, cv::Mat P, const float th, const int ORBdist, bool bCheckOri)
	{
		cv::Mat Rcw = P.rowRange(0, 3).colRange(0, 3);
		cv::Mat tcw = P.rowRange(0, 3).col(3);
		cv::Mat Ow = -Rcw.t() * tcw;

		std::vector<int> rotHist[HISTO_LENGTH];
		const float factor = 1.0f / HISTO_LENGTH;

		const auto vpMPs = pKeyBox->mvpMapPoints.get();
			
		auto pF = pNewBox->mpF;
		if (!pF)
			return -1;
		
		float fx = pF->fx;
		float fy = pF->fy;
		float cx = pF->cx;
		float cy = pF->cy;
		float nMinX = pF->mnMinX;
		float nMinY = pF->mnMinY;
		float nMaxX = pF->mnMaxX;
		float nMaxY = pF->mnMaxY;

		int nmatches = 0;
		for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
		{
			auto pMP = vpMPs[i];

			if (pMP)
			{
				if (!pMP->isBad() && !sAlreadyFound.count(pMP))
				{
					//Project
					cv::Mat x3Dw = pMP->GetWorldPos();
					cv::Mat x3Dc = Rcw * x3Dw + tcw;

					const float xc = x3Dc.at<float>(0);
					const float yc = x3Dc.at<float>(1);
					const float invzc = 1.0 / x3Dc.at<float>(2);

					const float u = fx * xc * invzc + cx;
					const float v = fy * yc * invzc + cy;

					if (u<nMinX || u>nMaxX)
						continue;
					if (v<nMinY || v>nMaxY)
						continue;

					// Compute predicted scale level
					cv::Mat PO = x3Dw - Ow;
					float dist3D = cv::norm(PO);

					const float maxDistance = pMP->GetMaxDistanceInvariance();
					const float minDistance = pMP->GetMinDistanceInvariance();

					// Depth must be inside the scale pyramid of the image
					if (dist3D<minDistance || dist3D>maxDistance)
						continue;

					int nPredictedLevel = pMP->PredictScale(dist3D, pF);

					// Search in a window
					const float radius = th * pF->mvScaleFactors[nPredictedLevel];

					const auto vIndices2 = pF->GetFeaturesInArea(u, v, radius, nPredictedLevel - 1, nPredictedLevel + 1);

					if (vIndices2.empty())
						continue;

					const cv::Mat dMP = pMP->GetDescriptor();

					int bestDist = 256;
					int bestIdx2 = -1;

					for (auto vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
					{
						const size_t i2 = *vit;
						if (!pNewBox->mapIDXs.count(i2))
							continue;
						int newIDX = pNewBox->mapIDXs[i2];

						if (pNewBox->mvpMapPoints.get(newIDX))
							continue;

						const cv::Mat& d = pNewBox->desc.row(newIDX);

						const int dist = (int)Matcher->DescriptorDistance(dMP, d);

						if (dist < bestDist)
						{
							bestDist = dist;
							bestIdx2 = newIDX;
						}
					}

					if (bestDist <= ORBdist)
					{
						pNewBox->mvpMapPoints.update(bestIdx2, pMP);
						nmatches++;

						if (bCheckOri)
						{
							float rot = pKeyBox->mvKeys[i].angle - pNewBox->mvKeys[bestIdx2].angle;
							if (rot < 0.0)
								rot += 360.0f;
							int bin = round(rot * factor);
							if (bin == HISTO_LENGTH)
								bin = 0;
							assert(bin >= 0 && bin < HISTO_LENGTH);
							rotHist[bin].push_back(bestIdx2);
						}
					}

				}
			}
		}

		if (bCheckOri)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;

			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i < HISTO_LENGTH; i++)
			{
				if (i != ind1 && i != ind2 && i != ind3)
				{
					for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
					{
						pNewBox->mvpMapPoints.update(rotHist[i][j], nullptr);
						nmatches--;
					}
				}
			}
		}
		return nmatches;
	}

	int ObjectSearchPoints::SearchObjectNodeAndBox(EdgeSLAM::ObjectNode* pNode, EdgeSLAM::ObjectBoundingBox* pBox, std::vector<std::pair<int, int>>& matches, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri) {
		int nmatches = 0;
		std::vector<bool> bMatches = std::vector<bool>(pBox->N, false);

		//auto mspMPs = pNode->mspMPs.Get();
		//auto mvpMPs = std::vector<EdgeSLAM::ObjectMapPoint*>(mspMPs.begin(), mspMPs.end());
		auto mvpMPs = pNode->mspOPs.ConvertVector();

		int Ntmp = 0;

		//for (int i = 0, iend = mvpMPs.size(); i < iend; i++) {
		//	int idxObj = i;
		//	auto pMPi = mvpMPs[i];

		//	if (!pMPi || pMPi->isBad())
		//		continue;
		//	const cv::Mat& dObj = pMPi->GetDescriptor();

		//	int bestDist1 = 256;
		//	int bestIdxF = -1;
		//	int bestDist2 = 256;

		//	for (int j = 0, jend = pBox->N; j < jend; j++) {
		//		int idxFrame = j;
		//		if (bMatches[idxFrame])
		//			continue;
		//		auto pMPj = pBox->mvpObjectPoints.get(idxFrame);
		//		if (pMPj && !pMPj->isBad())
		//			continue;

		//		const cv::Mat& dFrame = pBox->desc.row(idxFrame);

		//		const int dist = (int)Matcher->DescriptorDistance(dObj, dFrame);
		//		//std::cout <<"a "<< dist << std::endl;
		//		if (dist < bestDist1)
		//		{
		//			bestDist2 = bestDist1;
		//			bestDist1 = dist;
		//			bestIdxF = idxFrame;
		//		}
		//		else if (dist < bestDist2)
		//		{
		//			bestDist2 = dist;
		//		}
		//	}//for frame

		//	if (bestDist1 <= thMinDesc)
		//	{
		//		Ntmp++;
		//		if (static_cast<float>(bestDist1) < thProjection * static_cast<float>(bestDist2))
		//		{
		//			bMatches[bestIdxF] = true;
		//			matches.push_back(std::make_pair((int)idxObj, (int)bestIdxF));
		//			nmatches++;
		//		}
		//	}
		//	//std::cout << "matching test = " << thMinDesc << "||" << Ntmp << " " << nmatches << std::endl;
		//}
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
					//해결해야 하는 상황이 될 듯
					if (pBox2->mspMapPoints.Count(pMPi))
						continue;
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
			auto pMPi = pBox1->mvpMapPoints.get(idxObj);
			if (pMPi && !pMPi->isBad())
				continue;
			const cv::Mat& dObj = pBox1->desc.row(idxObj);

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
				bMatches[bestIdxF] = true;
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

	void ObjectSearchPoints::ComputeThreeMaxima(std::vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3)
	{
		int max1 = 0;
		int max2 = 0;
		int max3 = 0;

		for (int i = 0; i < L; i++)
		{
			const int s = histo[i].size();
			if (s > max1)
			{
				max3 = max2;
				max2 = max1;
				max1 = s;
				ind3 = ind2;
				ind2 = ind1;
				ind1 = i;
			}
			else if (s > max2)
			{
				max3 = max2;
				max2 = s;
				ind3 = ind2;
				ind2 = i;
			}
			else if (s > max3)
			{
				max3 = s;
				ind3 = i;
			}
		}

		if (max2 < 0.1f * (float)max1)
		{
			ind2 = -1;
			ind3 = -1;
		}
		else if (max3 < 0.1f * (float)max1)
		{
			ind3 = -1;
		}
	}

}