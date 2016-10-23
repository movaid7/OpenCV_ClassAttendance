#pragma once

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <vector>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

const int DESIRED_FACE_WIDTH	= 120;
const int DESIRED_FACE_HEIGHT	= 120;
const int DESIRED_LEFT_EYE_Y	= 0.16;

class Detect
{
public:
	static void faceProcessing(Point leftEye, Point rightEye, Mat &gray, bool &faceProcessed);
};

