#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay(Mat img);
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth);
void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth);
void faceProcessing(Point, Point, const Mat &img);

// Global variables	
const double EYE_SX = 0.10;
const double EYE_SY = 0.19;
const double EYE_SW = 0.40;
const double EYE_SH = 0.36;
const int DETECTION_WIDTH = 800;
bool eyeDetection = false;				//Will be used to skip histogram equalisation and resizing for eye detection

// Copy this file from opencv/data/haarscascades to target folder
string face_cascade_name = "c:/opencv-build/install/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
string eye1_cascade_name = "c:/opencv-build/install/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
string eye2_cascade_name = "c:/opencv-build/install/share/OpenCV/haarcascades/haarcascade_eye.xml";

CascadeClassifier faceDetector;
CascadeClassifier eyeDetector1;
CascadeClassifier eyeDetector2;

//int filenumber; // Number of file to be saved
//string filename;

// Function main
int main(void)
{
	//Load detectors
	try{
		faceDetector.load(face_cascade_name);
		eyeDetector1.load(eye1_cascade_name);
		eyeDetector2.load(eye2_cascade_name);
	}
	catch (cv::Exception e) {}	
	if (faceDetector.empty()){
		cerr << "ERROR: Couldn't load Face Detector (";
		cerr << face_cascade_name << ")!" << endl;
		exit(1);
	}
	else if (eyeDetector1.empty()||eyeDetector2.empty())
	{
		cerr << "ERROR: Couldn't load Eye Detector (";
		cerr << eye1_cascade_name << ")!" << endl;
		exit(1);
	}
	
	//Load image and call detect function
	IplImage* pic = cvLoadImage("Pictures/United4.jpg");
	Mat img(pic);
	if (!img.empty()){
		detectAndDisplay(img);
	}
	else{
		printf("(!)-- No captured frame --(!)\n\n"); 
	}
	
	system("pause");
	return 0;
}


void detectAndDisplay(Mat img)
{
	//Vector to hold face boxes
	vector<Rect> faces;
	detectManyObjects(img, faceDetector, faces, DETECTION_WIDTH);;

	//How many faces did multiscale detect? - [test parameter]
	printf("%d faces were detected\n\n", faces.size());
	
	//ADD RECTANGLES TO ORIGINAL IMAGE
	for (int i = 0; i < (int)faces.size(); i++)
	{
		Point pt1(faces[i].x, faces[i].y);
		Point pt2((faces[i].x + faces[i].height), (faces[i].y + faces[i].width));
		rectangle(img, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}

	//OUTPUT ORIGINAL PICTURE WITH RECTANGLES AROUND EACH FACE
	imshow("Detected", img);
	waitKey(0);
	destroyWindow("Detected");
	
	//------------------------------------------------------------------------------
	//----------------------------START OF EYE DETECTION----------------------------
	//------------------------------------------------------------------------------

	Mat face,res;
	Mat topLeftOfFace;
	Mat topRightOfFace;
	
	//EXTRACT AND SAVE FACES
	for (int i = 0; i < (int)faces.size(); i++)
	{
		face = img(faces[i]);
		resize(face, face, Size(512, 512), 0, 0, INTER_LINEAR); // This will be needed later while saving images
		int leftX = cvRound(face.cols * EYE_SX);
		int topY = cvRound(face.rows * EYE_SY);
		int widthX = cvRound(face.cols * EYE_SW);
		int heightY = cvRound(face.rows * EYE_SH);
		int rightX = cvRound(face.cols * (1.0 - EYE_SX - EYE_SW));
		topLeftOfFace = face(Rect(leftX, topY, widthX,	heightY));
		topRightOfFace = face(Rect(rightX, topY, widthX, heightY));

		//TEST
		//resize(topLeftOfFace, topLeftOfFace, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
		//resize(topRightOfFace, topRightOfFace, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images

		//TEST DISPLAY
	/*	imshow("TEST", topLeftOfFace);
		waitKey(0);
		imshow("TEST", topRightOfFace);
		waitKey(0);

		destroyWindow("TEST");*/

		//Write picture to file
		/*if (filenumber < 11)
		{
			filename = "";
			stringstream ssfn;
			ssfn << filenumber << ".png";
			filename = ssfn.str();
			imwrite(filename, face);
		}
		filenumber++;*/
		
		Rect leftEyeRect, rightEyeRect;
		Point leftEye, rightEye;
		eyeDetection = true;

		// Search the left region, then the right region using the 1st eye detector.
		detectLargestObject(topLeftOfFace, eyeDetector1, leftEyeRect, topLeftOfFace.cols);
		detectLargestObject(topRightOfFace, eyeDetector1, rightEyeRect, topRightOfFace.cols);

		// If the eye was not detected, try a different cascade classifier.
		if (leftEyeRect.width <= 0 && !eyeDetector2.empty()) {
			detectLargestObject(topLeftOfFace, eyeDetector2, leftEyeRect, topLeftOfFace.cols);
			if (leftEyeRect.width > 0)
			    cout << "2nd eye detector LEFT SUCCESS" << endl;
			else
			    cout << "2nd eye detector LEFT failed" << endl;
		}
		else
		    cout << "1st eye detector LEFT SUCCESS" << endl;

		// If the eye was not detected, try a different cascade classifier.
		if (rightEyeRect.width <= 0 && !eyeDetector2.empty()) {
			detectLargestObject(topRightOfFace, eyeDetector2, rightEyeRect, topRightOfFace.cols);
			if (rightEyeRect.width > 0)
			    cout << "2nd eye detector RIGHT SUCCESS" << endl;
			else
			    cout << "2nd eye detector RIGHT failed" << endl;
		}
		else
		    cout << "1st eye detector RIGHT SUCCESS" << endl;

		if (leftEyeRect.width > 0) {   // Check if the eye was detected.
			leftEyeRect.x += leftX;    // Adjust the left-eye rectangle because the face border was removed.
			leftEyeRect.y += topY;
			leftEye = Point(leftEyeRect.x + leftEyeRect.width / 2, leftEyeRect.y + leftEyeRect.height / 2);
		}
		else {
			leftEye = Point(-1, -1);    // Return an invalid point
		}

		if (rightEyeRect.width > 0) { // Check if the eye was detected.
			rightEyeRect.x += rightX; // Adjust the right-eye rectangle, since it starts on the right side of the image.
			rightEyeRect.y += topY;  // Adjust the right-eye rectangle because the face border was removed.
			rightEye = Point(rightEyeRect.x + rightEyeRect.width / 2, rightEyeRect.y + rightEyeRect.height / 2);
		}
		else {
			rightEye = Point(-1, -1);    // Return an invalid point
		}

		Mat gray;
		if (face.channels() == 3) {
			cvtColor(face, gray, CV_BGR2GRAY);
		}
		else if (face.channels() == 4) {
			cvtColor(face, gray, CV_BGRA2GRAY);
		}
		else {
			// Access the input image directly, since it is already grayscale.
			gray = face;
		}

		if (leftEye.x >= 0 && rightEye.x >= 0){
			faceProcessing(leftEye,rightEye, gray);
		}
		else{
			imshow("Eyes not found",face);
			waitKey(0);
			destroyWindow("Eyes not found");
		}
			
		eyeDetection = false;

		 //////Show each face/region
		 //printf("%d", leftEyeRect.size());
		 //imshow("Detected", topLeftOfFace);
		 //waitKey(0);
		 //destroyWindow("Detected");
	}


}

void detectObjectsCustom(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
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

	if(eyeDetection)			//IF DOING EYE DETECTION - UNDO EQUALISATION AND RESIZING
		equalizedImg=gray;

	// Detect objects in the small grayscale image.
	cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

	//TEST - DISPLAY MODIFIED IMAGE WITH 'BOXED' OBJECTS
	for (int i = 0; i < (int)objects.size(); i++){
		Point pt1(objects[i].x, objects[i].y); 
		Point pt2((objects[i].x + objects[i].height), (objects[i].y + objects[i].width));
		rectangle(equalizedImg, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}
	imshow("TEST", equalizedImg);
	waitKey(0);
	destroyWindow("TEST");


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

void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth)
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
	detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);

	if (objects.size() > 0) {
		// Return the only detected object.
		largestObject = (Rect)objects.at(0);
	}
	else {
		// Return an invalid rect.
		largestObject = Rect(-1, -1, -1, -1);
	}
}

void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth)
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


//------------------------------------------------------------------------------------
//-----------------------------------PRE-PROCESSING-----------------------------------
//------------------------------------------------------------------------------------

void faceProcessing(Point leftEye, Point rightEye, const Mat &gray)
{
	//---------------------------------------STAGE 1---------------------------------------

	// Get the center between the 2 eyes.
	Point2f eyesCenter;
	eyesCenter.x = (leftEye.x + rightEye.x) * 0.5f;
	eyesCenter.y = (leftEye.y + rightEye.y) * 0.5f;
	// Get the angle between the 2 eyes.
	double dy = (rightEye.y - leftEye.y);
	double dx = (rightEye.x - leftEye.x);
	double len = sqrt(dx*dx + dy*dy);
	// Convert Radians to Degrees.
	double angle = atan2(dy, dx) * 180.0 / CV_PI;
	// Hand measurements shown that the left eye center should
	// ideally be roughly at (0.16, 0.14) of a scaled face image.
	const double DESIRED_LEFT_EYE_X = 0.18;
	const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);
	// Get the amount we need to scale the image to be the desired
	// fixed size we want.

	const int DESIRED_FACE_WIDTH = 70;
	const int DESIRED_FACE_HEIGHT = 80;
	const int DESIRED_LEFT_EYE_Y = 0.16;

	double desiredLen = (DESIRED_RIGHT_EYE_X - 0.16);
	double scale = desiredLen * DESIRED_FACE_WIDTH / len;

	// Get the transformation matrix for the desired angle & size.
	Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
	// Shift the center of the eyes to be the desired center.
	double ex = DESIRED_FACE_WIDTH * 0.5f - eyesCenter.x;
	double ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter.y +15;
	rot_mat.at<double>(0, 2) += ex;
	rot_mat.at<double>(1, 2) += ey;
	// Transform the face image to the desired angle & size &
	// position! Also clear the transformed image background to a
	// default grey.
	Mat warped = Mat(DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH, CV_8U, Scalar(128));
	warpAffine(gray, warped, rot_mat, warped.size());

	//---------------------------------------STAGE 2---------------------------------------
	
	Mat faceImg = warped;
	int w = faceImg.cols;
	int h = faceImg.rows;
	Mat wholeFace;
	equalizeHist(faceImg, wholeFace);
	int midX = w / 2;
	Mat leftSide = faceImg(Rect(0, 0, midX, h));
	Mat rightSide = faceImg(Rect(midX, 0, w - midX, h));
	equalizeHist(leftSide, leftSide);
	equalizeHist(rightSide, rightSide);

	for (int y = 0; y<h; y++) {
		for (int x = 0; x<w; x++) {
			int v;
			if (x < w / 4) {
				// Left 25%: just use the left face.
				v = leftSide.at<uchar>(y, x);
			}
			else if (x < w * 2 / 4) {
				// Mid-left 25%: blend the left face & whole face.
				int lv = leftSide.at<uchar>(y, x);
				int wv = wholeFace.at<uchar>(y, x);
				// Blend more of the whole face as it moves
				// further right along the face.
				float f = (x - w * 1 / 4) / (float)(w / 4);
				v = cvRound((1.0f - f) * lv + (f)* wv);
			}
			else if (x < w * 3 / 4) {
				// Mid-right 25%: blend right face & whole face.
				int rv = rightSide.at<uchar>(y, x - midX);
				int wv = wholeFace.at<uchar>(y, x);
				// Blend more of the right-side face as it moves
				// further right along the face.
				float f = (x - w * 2 / 4) / (float)(w / 4);
				v = cvRound((1.0f - f) * wv + (f)* rv);
			}
			else {
				// Right 25%: just use the right face.
				v = rightSide.at<uchar>(y, x - midX);
			}
			faceImg.at<uchar>(y, x) = v;
		}// end x loop
	}//end y loop

	//---------------------------------------STAGE 3---------------------------------------

	Mat filtered = Mat(warped.size(), CV_8U);
	bilateralFilter(warped, filtered, 0, 20.0, 2.0);

	//---------------------------------------STAGE 4---------------------------------------


	//TEST
	imshow("TEST", filtered);
	waitKey(0);
	destroyWindow("TEST");
}