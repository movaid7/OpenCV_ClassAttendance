#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay(Mat img);

// Global variables	

// Copy this file from opencv/data/haarscascades to target folder
string face_cascade_name = "c:/opencv-build/install/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
CascadeClassifier faceDetector;

//int filenumber; // Number of file to be saved
//string filename;


// Function main
int main(void)
{
	//Load detector
	try {
		faceDetector.load(face_cascade_name);
	}
	catch (cv::Exception e) {}
	if (faceDetector.empty()) {
		cerr << "ERROR: Couldn't load Face Detector (";
		cerr << face_cascade_name << ")!" << endl;
		exit(1);
	}
	
	//Load image and call detect class
	IplImage* pic = cvLoadImage("test1.jpg");
	Mat img(pic);
	if (!img.empty())
	{
		detectAndDisplay(img);
	}
	else
	{
		printf("(!)-- No captured frame --(!)\n\n"); 
	}
	
	system("pause");
	return 0;
}


void detectAndDisplay(Mat img)
{
	Mat modified;		//Gets altered with each change
	Mat gray;			//Store greyscale image
	Mat smallImg;		//Store resized image
	Mat equalizedImg;	//Store equalized image

	//CONVERT TO GRAYSCALE
	if (img.channels() == 3) {
		cvtColor(img, gray, CV_BGR2GRAY);
	}
	else if (img.channels() == 4) {
		cvtColor(img, gray, CV_BGRA2GRAY);
	}
	else {
		// Access the grayscale input image directly.
		gray = img;
	}
	modified = gray;


	//SHRIK IMAGE FOR FASTER PROCESSING
	const int DETECTION_WIDTH = 800;
	
	float scale = modified.cols / (float)DETECTION_WIDTH;
	if (modified.cols > DETECTION_WIDTH) {
		// Shrink the image while keeping the same aspect ratio.
		int scaledHeight = cvRound(modified.rows / scale);
		resize(modified, smallImg, Size(DETECTION_WIDTH, scaledHeight));
	}
	else {
		// Access the input directly since it is already small.
		smallImg = modified;
	}
	modified = smallImg;


	//EQUALIZE HISTOGRAMS - IMPROVES CONTRAST AND BRIGHTNESS
	equalizeHist(modified, equalizedImg);
	modified = equalizedImg;


	//Parameters for detectMulti
	int flags = CASCADE_SCALE_IMAGE; // Search for many faces.
	Size minFeatureSize(20, 20); // Smallest face size.
	float searchScaleFactor = 1.1f; // How many sizes to search.
	int minNeighbors = 4; // Reliability vs many faces.
						  
	// Detect objects in the small grayscale image.
	//Detect faces and stores rectangles in vector
	vector<Rect> faces;
	faceDetector.detectMultiScale(modified, faces, searchScaleFactor, minNeighbors, flags, minFeatureSize);
	Rect roi_b;

	//How many faces did multiscale detect - [test parameter]
	printf("%d faces were detected\n\n", faces.size());

	// Enlarge the results if the image was temporarily shrunk.
	if (img.cols > DETECTION_WIDTH)
	{
		for (int i = 0; i < (int)faces.size(); i++) {
			faces[i].x = cvRound(faces[i].x * scale);
			faces[i].y = cvRound(faces[i].y * scale);
			faces[i].width = cvRound(faces[i].width * scale);
			faces[i].height = cvRound(faces[i].height * scale);
		}
	}
	// If the object is on a border, keep it in the image.
	for (int i = 0; i < (int)faces.size(); i++)
	{
		if (faces[i].x < 0)
			faces[i].x = 0;
		if (faces[i].y < 0)
			faces[i].y = 0;
		if (faces[i].x + faces[i].width > img.cols)
			faces[i].x = img.cols - faces[i].width;
		if (faces[i].y + faces[i].height > img.rows)
			faces[i].y = img.rows - faces[i].height;
	}

	//ADD RECTANGLES TO ORIGINAL IMAGE
	for (int i = 0; i < (int)faces.size(); i++)
	{
		Point pt1(faces[i].x, faces[i].y); // Display detected faces on main window - live stream from camera
		Point pt2((faces[i].x + faces[i].height), (faces[i].y + faces[i].width));
		rectangle(img, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}

	//OUTPUT ORIGINAL PICTURE WITH RECTANGLES AROUND EACH FACE
	imshow("Detected", img);
	waitKey(0);
	destroyWindow("Detected");

}