#include <GridCell.h>

namespace SemanticSLAM {
	
	Label::Label() :mnLabel(0), mnCount(0) {
		matLabels = cv::Mat::zeros(200, 1, CV_16UC1);
	}
	Label::Label(int n) : mnLabel(0), mnCount(0) {
		matLabels = cv::Mat::zeros(n, 1, CV_16UC1);
	}
	Label::~Label() {
		matLabels.release();
	}
	void Label::Update(int nLabel) {

		std::unique_lock<std::mutex> lock(mMutexObject);
		matLabels.at<ushort>(nLabel)++;
		if (mnLabel == nLabel) {
			mnCount++;
		}
		else {
			int count = matLabels.at<ushort>(nLabel);
			if (count > mnCount) {
				double minVal;
				double maxVal;
				int minIdx, maxIdx;
				cv::minMaxIdx(matLabels, &minVal, &maxVal, &minIdx, &maxIdx, cv::Mat());
				mnLabel = maxIdx;
				mnCount = matLabels.at<ushort>(maxIdx);
			}
		}
	}
	int Label::GetLabel() {
		std::unique_lock<std::mutex> lock(mMutexObject);
		return mnLabel;
	}
	cv::Mat Label::GetLabels() {
		std::unique_lock<std::mutex> lock(mMutexObject);
		return matLabels.clone();
	}
	int Label::Count(int l) {
		std::unique_lock<std::mutex> lock(mMutexObject);
		return matLabels.at<ushort>(l);
	}
	
	GridCell::GridCell(){}
	GridCell::~GridCell(){
		delete mpObject;
		delete mpSegLabel;
		mapObservation.Release();
	}

	void GridCell::AddObservation(GridFrame* pGF, int idx){
		mapObservation.Update(pGF, idx);
	}
	void GridCell::EraseObservation(GridFrame* pGF){
		mapObservation.Erase(pGF);
		if (mapObservation.Size() == 0)
		{
			SetBadFlag();
		}
	}
	void GridCell::SetBadFlag(){
		mbBad = true;
	}
	/*bool GridCell::isBad(){
		return mbBad.load();
	}*/

	GridFrame::GridFrame(){
		mGrid = std::vector<std::vector<GridCell*>>(10);
		for (int i = 0, iend = mGrid.size(); i < iend; i++)
			mGrid[i] = std::vector<GridCell*>(10, nullptr);
	}
	GridFrame::GridFrame(int row, int col) {
		mGrid = std::vector<std::vector<GridCell*>>(row);
		for (int i = 0, iend = mGrid.size(); i < iend; i++)
			mGrid[i] = std::vector<GridCell*>(col, nullptr);
	}
	GridFrame::~GridFrame(){
		for (int i = 0, iend = mGrid.size(); i < iend; i++) {
			for (int j = 0, jend = mGrid[i].size(); j < jend; j++) {
				auto pGC = mGrid[i][j];
				if (!pGC)
					continue;
				pGC->EraseObservation(this);
				if (pGC->mbBad)
					delete pGC;
			}
			std::vector<GridCell*>().swap(mGrid[i]);
		}
		std::vector<std::vector<GridCell*>>().swap(mGrid);
	}
	void GridFrame::Copy(GridFrame* p) {
		
		for (int i = 0, iend = this->mGrid.size(); i < iend; i++) {
			for (int j = 0, jend = this->mGrid[i].size(); j < jend; j++) {
				auto pCell = p->mGrid[i][j];
				pCell->AddObservation(this, i*iend + j);
				this->mGrid[i][j] = pCell;
			}
		}
	}
}