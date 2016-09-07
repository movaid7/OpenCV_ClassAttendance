#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay(Mat frame);

// Global variables
// Copy this file from opencv/data/haarscascades to target folder
string face_cascade_name = "c:/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
int filenumber; // Number of file to be saved
string filename;

// Function main
int main(void)
{
	VideoCapture capture(0);
	IplImage* img = cvLoadImage("united1.jpg");
	
	if (!capture.isOpened())  // check if we succeeded
		return -1;

	// Load the cascade
	if (!face_cascade.load(face_cascade_name))
	{
		printf("--(!)Error loading\n");
		return (-1);
	};

	// Read the video stream
	Mat frame(img);

	for (;;)
	{
		//capture >> frame;

		// Apply the classifier to the frame
		if (!frame.empty())
		{
			detectAndDisplay(frame);
		}
		else
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		int c = waitKey(10);

		if (27 == char(c))
		{
			break;
		}
	}

	return 0;
}

// Function detectAndDisplay
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	Mat crop;
	Mat res;
	Mat gray;
	string text;
	stringstream sstm;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	
	// Set Region of Interest
	cv::Rect roi_b;
	cv::Rect roi_c;

	size_t ic = 0; // ic is index of current element
	int ac = 0; // ac is area of current element

	size_t ib = 0; // ib is index of biggest element
	int ab = 0; // ab is area of biggest element

	for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)

	{
		roi_c.x = faces[ic].x;
		roi_c.y = faces[ic].y;
		roi_c.width = (faces[ic].width);
		roi_c.height = (faces[ic].height);

		ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

		roi_b.x = faces[ib].x;
		roi_b.y = faces[ib].y;
		roi_b.width = (faces[ib].width);
		roi_b.height = (faces[ib].height);

		ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

		//if (ac > ab)							//REMOVED THIS TO SAVE ALL FACES
		//{
			ib = ic;
			roi_b.x = faces[ib].x;
			roi_b.y = faces[ib].y;
			roi_b.width = (faces[ib].width);
			roi_b.height = (faces[ib].height);
		//}

		crop = frame(roi_b);
		resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
		cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

										   // Form a filename
		filename = "";
		stringstream ssfn;
		ssfn << filenumber << ".png";
		filename = ssfn.str();
		filenumber++;

		imwrite(filename, gray);

		Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}

	// Show image
	sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
	text = sstm.str();

	putText(frame, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	imshow("original", frame);

	if (!crop.empty())
	{
		imshow("detected", crop);
	}
	else
		destroyWindow("detected");
}
//
//void Detect(){
//
//
//	//STEP 3-SET THE SOURCE OF IMAGE
//	//fetch the frame captured by web cam
//	Image<Bgr, Byte> ImageFrame = capture.QueryFrame();
//
////FACE DETECTION PROCESS
//if (ImageFrame != null) // confirm that image is valid
//{
//	//STEP 4.1-convert the image to gray scale
//	Image<Gray, byte> grayframe = ImageFrame.Convert<Gray, byte>();
//
//	//STEP 4.2-Detect faces from the gray-scale image and store into an array of type 'var',i.e 'MCvAvgComp[]'
//	var faces = grayframe.DetectHaarCascade(haar, 1.4, 4,
//		HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
//		new Size(25, 25))[0];
//
//	//STEP 4.3-draw a green rectangle on each detected face in image
//	foreach(var face in faces)
//	{
//		ImageFrame.Draw(face.rect, new Bgr(Color.Green), 3);
//	}
//}
////STEP 4.4-Display the image
//CamImageBox.Image = ImageFrame; //CamImageBox is an EmguCV imageBox
//}
