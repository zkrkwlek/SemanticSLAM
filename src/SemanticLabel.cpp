#include <SemanticLabel.h>

namespace SemanticSLAM {
	ObjectLabel::ObjectLabel(){}
	ObjectLabel::~ObjectLabel() {
		LabelCount.Release();
	}
	SemanticLabel::SemanticLabel() {}
	SemanticLabel::~SemanticLabel() {
		LabelCount.Release();
	}
}
