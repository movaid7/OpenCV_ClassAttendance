#include "Globals.h"


void Globals::detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth)
{
	// Search for many objects in the one image.
	int flags = CASCADE_SCALE_IMAGE;

	// Smallest object size.
	Size minFeatureSize = Size(20, 20);
	// How detailed should the search be. Must be larger than 1.0.
	float searchScaleFactor = 1.1f;
	// How much the detections should be filtered out. This should depend on how bad false detections are to your system.
	// minNeighbors=2 means lots of good+bad detections, and minNeighbors=6 means only good detections are given but some are missed.
	int minNeighbors = 4;

	// Perform Object or Face Detection, looking for many objects in the one image.
	detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
}

void Globals::detectObjectsCustom(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors,bool &eyeDetection)
{
	// If the input image is not grayscale, then convert the BGR or BGRA color image to grayscale.
	Mat gray;
	if (img.channels() == 3) {
		cvtColor(img, gray, CV_BGR2GRAY);
	}
	else if (img.channels() == 4) {
		cvtColor(img, gray, CV_BGRA2GRAY);
	}
	else {
		// Access the input image directly, since it is already grayscale.
		gray = img;
	}

	// Possibly shrink the image, to run much faster.
	Mat inputImg;
	float scale = img.cols / (float)scaledWidth;
	if (img.cols > scaledWidth) {
		// Shrink the image while keeping the same aspect ratio.
		int scaledHeight = cvRound(img.rows / scale);
		resize(gray, inputImg, Size(scaledWidth, scaledHeight));
	}
	else {
		// Access the input image directly, since it is already small.
		inputImg = gray;
	}

	// Standardize the brightness and contrast to improve dark images.
	Mat equalizedImg;
	equalizeHist(inputImg, equalizedImg);

	if (eyeDetection)			//IF DOING EYE DETECTION - UNDO EQUALISATION AND RESIZING
		equalizedImg = gray;

	// Detect objects in the small grayscale image.
	cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

	//TEST - DISPLAY MODIFIED IMAGE WITH 'BOXED' OBJECTS
	for (int i = 0; i < (int)objects.size(); i++) {
		Point pt1(objects[i].x, objects[i].y);
		Point pt2((objects[i].x + objects[i].height), (objects[i].y + objects[i].width));
		rectangle(equalizedImg, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}
	//imshow("TEST", equalizedImg);
	//waitKey(0);
	//destroyWindow("TEST");


	// Enlarge the results if the image was temporarily shrunk before detection.
	if (img.cols > scaledWidth && !eyeDetection) {
		for (int i = 0; i < (int)objects.size(); i++) {
			objects[i].x = cvRound(objects[i].x * scale);
			objects[i].y = cvRound(objects[i].y * scale);
			objects[i].width = cvRound(objects[i].width * scale);
			objects[i].height = cvRound(objects[i].height * scale);
		}
	}

	// Make sure the object is completely within the image, in case it was on a border.
	for (int i = 0; i < (int)objects.size(); i++) {
		if (objects[i].x < 0)
			objects[i].x = 0;
		if (objects[i].y < 0)
			objects[i].y = 0;
		if (objects[i].x + objects[i].width > img.cols)
			objects[i].x = img.cols - objects[i].width;
		if (objects[i].y + objects[i].height > img.rows)
			objects[i].y = img.rows - objects[i].height;
	}

	// Return with the detected face rectangles stored in "objects".
}

void Globals::detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth, bool &eyeDetection)
{
	// Only search for just 1 object (the biggest in the image).
	int flags = CASCADE_FIND_BIGGEST_OBJECT;// | CASCADE_DO_ROUGH_SEARCH;
											// Smallest object size.
	Size minFeatureSize = Size(20, 20);
	// How detailed should the search be. Must be larger than 1.0.
	float searchScaleFactor = 1.1f;
	// How much the detections should be filtered out. This should depend on how bad false detections are to your system.
	// minNeighbors=2 means lots of good+bad detections, and minNeighbors=6 means only good detections are given but some are missed.
	int minNeighbors = 4;

	// Perform Object or Face Detection, looking for just 1 object (the biggest in the image).
	vector<Rect> objects;
	detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors, eyeDetection);

	if (objects.size() > 0) {
		// Return the only detected object.
		largestObject = (Rect)objects.at(0);
	}
	else {
		// Return an invalid rect.
		largestObject = Rect(-1, -1, -1, -1);
	}
}

double Globals::getSimilarity(const Mat A, const Mat B)
{
	// Calculate the L2 relative error between the 2 images.
	double errorL2 = norm(A, B, CV_L2);
	// Scale the value since L2 is summed across all pixels.
	double similarity = errorL2 / (double)(A.rows * A.cols);
	return similarity;
}