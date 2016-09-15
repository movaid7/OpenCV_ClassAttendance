#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <vector>
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay(Mat &img);
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth);
void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth);
void faceProcessing(Point, Point, Mat &img);
double getSimilarity(const Mat A, const Mat B);
template <typename T> string toString(T t);
Mat getImageFrom1DFloatMat(const Mat matrixRow, int height);


// Global variables	
const double EYE_SX = 0.10;
const double EYE_SY = 0.19;
const double EYE_SW = 0.40;
const double EYE_SH = 0.36;
const int DETECTION_WIDTH = 800;
const double UNKNOWN_PERSON_THRESHOLD = 0.5;
const int DESIRED_FACE_WIDTH = 70;
const int DESIRED_FACE_HEIGHT = 80;
const int DESIRED_LEFT_EYE_Y = 0.16;
double imageDiff;
int m_selectedPerson = 0;
bool eyeDetection = false;				//Will be used to skip histogram equalisation and resizing for eye detection
bool faceProcessed = false;

vector<Rect> faceRect;					//Value of boxes around detected faces
vector<Mat> preprocessedFaces;			//Vector to store preprocessed faces
vector<int> faceLabels;					//Vector to store facelabels of preprocessed faces

//Location of detectors
string face_cascade_name = "c:/opencv-build/install/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
string eye1_cascade_name = "c:/opencv-build/install/share/OpenCV/haarcascades/haarcascade_eye.xml";
string eye2_cascade_name = "c:/opencv-build/install/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier faceDetector;
CascadeClassifier eyeDetector1;
CascadeClassifier eyeDetector2;

//Unused atm - will be used to store pictures
//int filenumber;
//string filename;


//**************************************************************************************************
//										START OF MAIN
//**************************************************************************************************

int main(void)
{
	//-----------------Load detectors-----------------
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
	
	VideoCapture cam(0); //webcam
	Mat img;
	Mat old_prepreprocessedFace;
	Mat new_preprocessedFace;

	//-----------------Collecting faces-----------------

	// Will be used to check how long since the previous face was added.
	double old_time = (double)getTickCount();

	Mat getface;

	//IplImage* pic = cvLoadImage("Pictures/test2.jpg");				//for static images
	//Mat img(pic);

	for (;;)//will break when 10 faces are found
	{
		cam >> img;
		Mat displayedFrame;
		
		if (!img.empty()) {
			getface = img;
			img.copyTo(displayedFrame);
			faceProcessed = false;
			detectAndDisplay(getface);	//getface will return with the preprocessed face - passed by ref
		}
		else {
			printf("(!)-- No captured frame --(!)\n\n");
		}

		//get difference in time - since last pic
		double current_time = (double)getTickCount();
		double timeDiff_seconds = (current_time - old_time) / getTickFrequency();

		if (faceProcessed)		//if a face was processed, will be stored in new_preprocessedFace
		{
			new_preprocessedFace = getface;
			////TEST
			//imshow("averageFace", new_preprocessedFace);
			//waitKey(0);
			//destroyAllWindows();

			if (old_prepreprocessedFace.data)		//if there exists an old face
			{
				imageDiff = getSimilarity(new_preprocessedFace, old_prepreprocessedFace);
				printf("Difference = %lf\n\n", imageDiff); //test
			}
			else //if not, old face = current face
				old_prepreprocessedFace = new_preprocessedFace;
			
			if ((imageDiff > 0.3) && (timeDiff_seconds > 1.0)) {						//if image differs by 0.3 and 1 second has passed
				// Also add the mirror image to the training set.
				Mat mirroredFace;														
				flip(new_preprocessedFace, mirroredFace, 1);							
				// Add the face & mirrored face to the detected face lists.
				preprocessedFaces.push_back(new_preprocessedFace);			
				preprocessedFaces.push_back(mirroredFace);
				faceLabels.push_back(m_selectedPerson);
				faceLabels.push_back(m_selectedPerson);

				// Keep a copy of the processed face,
				// to compare on next iteration.
				old_prepreprocessedFace = new_preprocessedFace;
				old_time = current_time;

				// Get access to the face region-of-interest.
				Mat displayedFaceRegion = displayedFrame(faceRect.at(0));				//at(0) because only one face atm
				// Add some brightness to each pixel of the face region.				//unneccesary - just to show user face detected
				displayedFaceRegion += CV_RGB(90, 90, 90);

				//TEST
				imshow("IMAGE", displayedFaceRegion);
				waitKey(0);
				destroyWindow("IMAGE");
			}
		}
		if (faceLabels.size() > 12)														//once 6 faces have been processed, break for - aka stop collecting
			break; 
			
	}

		//------------------------Train model------------------------

		// Load the "contrib" module is dynamically at runtime.
		bool haveContribModule = initModule_contrib();
		if (!haveContribModule) {
			cerr << "ERROR: The 'contrib' module is needed for ";
			cerr << "FaceRecognizer but hasn't been loaded to OpenCV!";
			cerr << endl;
			exit(1);
		}

		string facerecAlgorithm = "FaceRecognizer.Eigenfaces";
		Ptr<FaceRecognizer> model;
		// Use OpenCV's new FaceRecognizer in the "contrib" module:
		model = Algorithm::create<FaceRecognizer>(facerecAlgorithm);
		if (model.empty()) {
			cerr << "ERROR: The FaceRecognizer [" << facerecAlgorithm;
			cerr << "] is not available in your version of OpenCV. ";
			cerr << "Please update to OpenCV v2.4.1 or newer." << endl;
			exit(1);
		}
		
		
		model->train(preprocessedFaces, faceLabels);
		printf("trained\n\n");
	

		
		Mat averageFace = model->get<Mat>("mean");										//Calculate average face
		// Convert a 1D float row matrix to a regular 8-bit image.
		averageFace = getImageFrom1DFloatMat(averageFace, DESIRED_FACE_HEIGHT);

		////TEST
		//resize(averageFace, averageFace, Size(256, 256), 0, 0, INTER_LINEAR);
		//imshow("averageFace", averageFace);
		//waitKey(0);
		//destroyAllWindows();
		
		// Get the eigenvectors
		Mat eigenvectors = model->get<Mat>("eigenvectors");
		
		// Show the best 20 (or less) eigenfaces
		for (int i = 0; i < min(20, eigenvectors.cols); i++) {
			// Create a continuous column vector from eigenvector #i.
			Mat eigenvector = eigenvectors.col(i).clone();
			Mat eigenface = getImageFrom1DFloatMat(eigenvector,DESIRED_FACE_HEIGHT);

			////TEST
			/*imshow(format("Eigenface%d", i), eigenface);
			waitKey(0);
			destroyAllWindows();*/
		}

		//------------------------Recognition------------------------

		int identity = model->predict(new_preprocessedFace);							//NEED TO ADD MODES AND PERFORM RECOGNITION WITH NEW IMAGES FROM CAMERA
		 
		// Get some required data from the FaceRecognizer model.
		eigenvectors = model->get<Mat>("eigenvectors");
		Mat averageFaceRow = model->get<Mat>("mean");
		// Project the input image onto the eigenspace.
		Mat projection = subspaceProject(eigenvectors, averageFaceRow,
			new_preprocessedFace.reshape(1, 1));
		// Generate the reconstructed face back from the eigenspace.
		Mat reconstructionRow = subspaceReconstruct(eigenvectors,averageFaceRow, projection);
		// Make it a rectangular shaped image instead of a single row.
		Mat reconstructionMat = reconstructionRow.reshape(1,DESIRED_FACE_HEIGHT);
		// Convert the floating-point pixels to regular 8-bit uchar.
		Mat reconstructedFace = Mat(reconstructionMat.size(), CV_8U);
		reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);

	/*	//TEST
		imshow("averageFace", reconstructedFace); 
		waitKey(0);
		destroyAllWindows();
		imshow("averageFace", new_preprocessedFace);
		waitKey(0);
		destroyAllWindows();
*/

		double similarity = getSimilarity(new_preprocessedFace, reconstructedFace);
		string outputStr;

		//if similarity is greater than threshold, person is not recognised
		if (similarity > UNKNOWN_PERSON_THRESHOLD) {
			outputStr = "Unknown";
		}
		else
		{
				int identity = model->predict(new_preprocessedFace);
				outputStr = toString(identity);		
		}
		cout << "Identity: " << outputStr << ". Similarity: " << similarity << endl;		//current identity value will be 0 since only 1 face trained and detected

	system("pause");
	return 0;
}



//**************************************************************************************************
//										FUNCTIONS BELOW
//**************************************************************************************************

void detectAndDisplay(Mat &input) 
{
	Mat img = input;
	//Vector to hold face boxes
	detectManyObjects(img, faceDetector, faceRect, DETECTION_WIDTH);;

	//How many faces did multiscale detect? - [test parameter]
	printf("%d faces were detected\n\n", faceRect.size());
	
	//ADD RECTANGLES TO ORIGINAL IMAGE
	for (int i = 0; i < (int)faceRect.size(); i++)
	{
		Point pt1(faceRect[i].x, faceRect[i].y);
		Point pt2((faceRect[i].x + faceRect[i].height), (faceRect[i].y + faceRect[i].width));
		rectangle(img, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}

	//OUTPUT ORIGINAL PICTURE WITH RECTANGLES AROUND EACH FACE
	//imshow("Detected", img);
	//waitKey(0);
	//destroyWindow("Detected");
	
	//------------------------------------------------------------------------------
	//----------------------------START OF EYE DETECTION----------------------------
	//------------------------------------------------------------------------------

	Mat face,res;
	Mat topLeftOfFace;
	Mat topRightOfFace;
	
	//EXTRACT AND SAVE FACES
	for (int i = 0; i < (int)faceRect.size(); i++)
	{
		face = img(faceRect[i]);
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

		//TEST
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
			//TEST - shows entire face of person whose eyes were not located
			//imshow("Eyes not found",face);
			//waitKey(0);
			//destroyWindow("Eyes not found");
		}

		eyeDetection = false;
		input = gray;

		 //TEST
		 //printf("%d", leftEyeRect.size());
		 //imshow("Detected", topLeftOfFace);
		 //waitKey(0);
		 //destroyWindow("Detected");
	}

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

double getSimilarity(const Mat A, const Mat B) {
	// Calculate the L2 relative error between the 2 images.
	double errorL2 = norm(A, B, CV_L2);
	// Scale the value since L2 is summed across all pixels.
	double similarity = errorL2 / (double)(A.rows * A.cols);
	return similarity;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Convert the matrix row or column (float matrix) to a
// rectangular 8-bit image that can be displayed or saved.
// Scales the values to be between 0 to 255.
Mat getImageFrom1DFloatMat(const Mat matrixRow, int height)
{
	// Make a rectangular shaped image instead of a single row.
	Mat rectangularMat = matrixRow.reshape(1, height);
	// Scale the values to be between 0 to 255 and store them
	// as a regular 8-bit uchar image.
	Mat dst;
	normalize(rectangularMat, dst, 0, 255, NORM_MINMAX,
		CV_8UC1);
	return dst;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename T> string toString(T t)
{
	ostringstream out;
	out << t;
	return out.str();
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//------------------------------------------------------------------------------------
//-----------------------------------PRE-PROCESSING-----------------------------------
//------------------------------------------------------------------------------------

void faceProcessing(Point leftEye, Point rightEye, Mat &gray)
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

	double desiredLen = (DESIRED_RIGHT_EYE_X - 0.16);
	double scale = desiredLen * DESIRED_FACE_WIDTH / len;

	// Get the transformation matrix for the desired angle & size.
	Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
	// Shift the center of the eyes to be the desired center.
	double ex = DESIRED_FACE_WIDTH * 0.5f - eyesCenter.x;
	double ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter.y + 15;
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

	// Draw a black-filled ellipse in the middle of the image.
	// First we initialize the mask image to white (255).
	Mat mask = Mat(warped.size(), CV_8UC1, Scalar(255));
	double dw = DESIRED_FACE_WIDTH;
	double dh = DESIRED_FACE_HEIGHT;
	Point faceCenter = Point(cvRound(dw * 0.5),
		cvRound(dh * 0.4));
	Size size = Size(cvRound(dw * 0.5), cvRound(dh * 0.8));
	ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(0), CV_FILLED);

	// Apply the elliptical mask on the face, to remove corners.
	// Sets corners to gray, without touching the inner face.
	filtered.setTo(Scalar(128), mask);

	faceProcessed = true;
	gray = filtered;

	//TEST
	//imshow("TEST", filtered);
	//waitKey(0);
	//destroyWindow("TEST");
}
