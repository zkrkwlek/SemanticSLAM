#include <SemanticLabel.h>

namespace SemanticSLAM {
	SemanticLabel::SemanticLabel(){}
	SemanticLabel::~SemanticLabel() {
		LabelCount.Release();
	}
}
