#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <vector>
#include <iostream>
#include <stdio.h>
#include <string>
#include <time.h>  
#include "Globals.h"
#include "Detect.h"

using namespace std;
using namespace cv;

// Function Headers
template <typename T> string toString(T t);
void detectFaces(Mat &img);
void trainFace(int m_selectedPerson);
void recogniseFace(Mat &input);
double getSimilarity(const Mat A, const Mat B);

void quickDetect(Mat &src);
Mat getImageFrom1DFloatMat(const Mat matrixRow, int height);
Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace);

// Global variables	
const double EYE_SX = 0.10;
const double EYE_SY = 0.19;
const double EYE_SW = 0.40;
const double EYE_SH = 0.36;
const double UNKNOWN_PERSON_THRESHOLD = 0.5;
const int DETECTION_WIDTH = 800;
char mode = '0';
double imageDiff;
bool m_debug = true;
bool eyeDetection = false;				//Will be used to skip histogram equalisation and resizing for eye detection
bool faceProcessed = false;
bool facefound = false;
double min_face_size = 20;
double max_face_size = 300;
bool run;
int random;

VideoCapture cam(0); //webcam
Ptr<FaceRecognizer> model;
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

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event != EVENT_LBUTTONDOWN)
		return;

		if (x<145 && x>67 && y<450 && y>425)
		{
			destroyAllWindows();
			mode = '2';
		}
		else if (x<580 && x>420 && y<450 && y>425)
		{
			destroyAllWindows();
			mode = '3';
		}
		//cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
}

//**************************************************************************************************
//										START OF MAIN
//**************************************************************************************************

int main(void)
{
	srand(time(NULL)); //for random num gen - recognise will train model with new face when random num = 7
	//-----------------Load detectors-----------------
	try {
		faceDetector.load(face_cascade_name);
		eyeDetector1.load(eye1_cascade_name);
		eyeDetector2.load(eye2_cascade_name);
	}
	catch (cv::Exception e) {}
	if (faceDetector.empty()) {
		cerr << "ERROR: Couldn't load Face Detector (";
		cerr << face_cascade_name << ")!" << endl;
		exit(1);
	}
	else if (eyeDetector1.empty() || eyeDetector2.empty())
	{
		cerr << "ERROR: Couldn't load Eye Detector (";
		cerr << eye1_cascade_name << ")!" << endl;
		exit(1);
	}
	
	run = true;
	Mat img;
	int ID;

	if (cam.isOpened())
	{
		while (run == true)
		{
			switch (mode) //Menu
			{
			case '0':
			{
				destroyAllWindows();
				system("cls");
				printf("MAIN MENU\n");
				printf("Option 1 - Detect Faces\n");						//Just box faces and display
				printf("Option 2 - Train Faces\n");							//Ask user for ID and train
				printf("Option 3 - Recognise Faces\n");						//Attempt facial recognition

				printf("\nType out an option number and hit ENTER\n(S to exit)\n");
				printf("\nYour Choice: ");
				cin >> mode;
				break;
			}

			case '1':
			{
				system("cls");
				while (mode=='1')
				{
					cam >> img;
					quickDetect(img);
					namedWindow("Automated Attendence", 1);
					setMouseCallback("Automated Attendence", CallBackFunc, NULL);
					putText(img, "Train", Point(70, 450), CV_FONT_VECTOR0, 1.0, CV_RGB(255, 255, 255), 3.0);
					putText(img, "Train", Point(70, 450), CV_FONT_VECTOR0, 1.0, CV_RGB(0, 0, 0), 2.0);
					putText(img, "Recognise", Point(420, 450), CV_FONT_VECTOR0, 1.0, CV_RGB(255, 255, 255), 3.0);
					putText(img, "Recognise", Point(420, 450), CV_FONT_VECTOR0, 1.0, CV_RGB(0, 0, 0), 2.0);
					imshow("Automated Attendence", img);
					// press 's' to escape
					if (waitKey(1) == 's') { mode = '0'; destroyAllWindows();  break; };
				}
				break;
			}

			case '2':
			{
				system("cls");
				printf("Enter person's ID or -1 to stop training\n");
				cin >> ID;
				if (ID == -1)
					mode = '0';
				else
				{
					//Mat input = img;
					trainFace(ID);
					//imshow("Processed", input);
					int c = waitKey(10);
					if ((char)c == 's') { mode = '0'; destroyAllWindows(); }
				}

				break;
			}

			case '3':
			{
				int c = waitKey(10);
				cam >> img;
				if (!img.empty() && preprocessedFaces.size()>0 && preprocessedFaces.size() == faceLabels.size())
				{
					Mat input = img;
					//namedWindow("Webcam", 1);
					recogniseFace(input);
					imshow("Webcam", img);
				}
				else
				{
					system("cls");
					mode = '0';
					destroyAllWindows();
					if (preprocessedFaces.size() <= 0)
					{
						printf("There is no trained data to test against\n\n");
					}

					if (preprocessedFaces.size() != faceLabels.size())
					{
						printf("Data is corrupted\n\n");
					}
					system("pause");
					break;
				}
				
				if ((char)c == 's') { mode = '0'; destroyAllWindows(); }
				break;
			}
			case 's':
			{
				run = false;
				break;
			}
			case 'S':
			{
				run = false;
				break;
			}
			default:
			{
				system("cls");
				destroyAllWindows();
				mode = '0';
				printf("Invalid Entry! Please Retry\n");
				system("pause");
				break;
			}

			}
		}
	}
	return 0;
}


//**************************************************************************************************
//										FUNCTIONS BELOW
//**************************************************************************************************

void quickDetect(Mat &image)
{
	faceDetector.detectMultiScale(image, faceRect, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size), Size(max_face_size, max_face_size));
	printf("%d faces were detected\n\n", faceRect.size());

	// Draw on the detected faces
	for (int i = 0; i < (int)faceRect.size(); i++)
	{
		Point pt1(faceRect[i].x, faceRect[i].y);
		Point pt2((faceRect[i].x + faceRect[i].height), (faceRect[i].y + faceRect[i].width));
		rectangle(image, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}
	
	if ((int)faceRect.size()>0)
		facefound = true;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


void detectFaces(Mat &input)
{
	Mat img = input;
	//Vector to hold face boxes
	Globals::detectManyObjects(img, faceDetector, faceRect, DETECTION_WIDTH);;

	//How many faces did multiscale detect? - [test parameter]
	printf("%d faces were detected\n\n", faceRect.size());

	//ADD RECTANGLES TO ORIGINAL IMAGE
	for (int i = 0; i < (int)faceRect.size(); i++)
	{
		Point pt1(faceRect[i].x, faceRect[i].y);
		Point pt2((faceRect[i].x + faceRect[i].height), (faceRect[i].y + faceRect[i].width));
		rectangle(img, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}
	input = img;
	if ((int)faceRect.size()>0)
		facefound = true;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void detectEyes(Mat &input)
{
	Mat img = input;
	Mat topLeftOfFace;
	Mat topRightOfFace;
	Mat gray, face;
	face = img;
	/*for (int i = 0; i < (int)faceRect.size(); i++)
	{
		face = img(faceRect[i]);*/
		resize(face, face, Size(512, 512), 0, 0, INTER_LINEAR); // This will be needed later while saving images
		int leftX = cvRound(face.cols * EYE_SX);
		int topY = cvRound(face.rows * EYE_SY);
		int widthX = cvRound(face.cols * EYE_SW);
		int heightY = cvRound(face.rows * EYE_SH);
		int rightX = cvRound(face.cols * (1.0 - EYE_SX - EYE_SW));
		topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
		topRightOfFace = face(Rect(rightX, topY, widthX, heightY));


		//EYE DETECTION

		Rect leftEyeRect, rightEyeRect;
		Point leftEye, rightEye;
		eyeDetection = true;

		// Search the left region, then the right region using the 1st eye detector.
		Globals::detectLargestObject(topLeftOfFace, eyeDetector1, leftEyeRect, topLeftOfFace.cols,eyeDetection);
		Globals::detectLargestObject(topRightOfFace, eyeDetector1, rightEyeRect, topRightOfFace.cols, eyeDetection);

		// If the eye was not detected, try a different cascade classifier.
		if (leftEyeRect.width <= 0 && !eyeDetector2.empty()) {
			Globals::detectLargestObject(topLeftOfFace, eyeDetector2, leftEyeRect, topLeftOfFace.cols, eyeDetection);
			if (leftEyeRect.width > 0)
				cout << "2nd eye detector LEFT SUCCESS" << endl;
			else
				cout << "2nd eye detector LEFT failed" << endl;
		}
		else
			cout << "1st eye detector LEFT SUCCESS" << endl;

		// If the eye was not detected, try a different cascade classifier.
		if (rightEyeRect.width <= 0 && !eyeDetector2.empty()) {
			Globals::detectLargestObject(topRightOfFace, eyeDetector2, rightEyeRect, topRightOfFace.cols,eyeDetection);
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

		if (leftEye.x >= 0 && rightEye.x >= 0) {
			Detect::faceProcessing(leftEye, rightEye, gray,faceProcessed);
		}
		else {
			faceProcessed = false;
		}
	//}
	eyeDetection = false;
	input = gray;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void trainFace(int m_selectedPerson)
{
	Mat img;
	Mat input;
	Mat displayedFrame;
	Mat old_prepreprocessedFace;
	Mat new_preprocessedFace;
	int count = 0;
	double old_time = (double)getTickCount();

	for (;;)//will break when 10 faces are found
	{
		cam >> img;
		
		if (!img.empty()) {
			input = img;
			img.copyTo(displayedFrame);

			facefound = false;
			faceProcessed = false;

			detectFaces(input);				//Find box around face

			if (facefound)
			{
				for (int i = 0; i < (int)faceRect.size(); i++)
				{
					input = input(faceRect[i]);
					detectEyes(input);
				}
			}
		}
		else {
			printf("(!)-- No captured frame --(!)\n\n");
		}


		//test
		/*imshow("", input);
		waitKey(0);
		destroyAllWindows();*/

		//get difference in time - since last pic
		double current_time = (double)getTickCount();
		double timeDiff_seconds = (current_time - old_time) / getTickFrequency();

		if (faceProcessed)		//if a face was processed, will be stored in new_preprocessedFace
		{
			new_preprocessedFace = input;

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
				count++;
				// Keep a copy of the processed face,
				// to compare on next iteration.
				old_prepreprocessedFace = new_preprocessedFace;
				old_time = current_time;

				// Get access to the face region-of-interest.
				Mat displayedFaceRegion = displayedFrame(faceRect.at(0));				//at(0) because only one face atm
																						// Add some brightness to each pixel of the face region.				//unneccesary - just to show user face detected
				displayedFaceRegion += CV_RGB(90, 90, 90);

				////TEST
				//imshow("IMAGE", displayedFaceRegion);
				//waitKey(0);
				//destroyWindow("IMAGE");
			}
		}
		if (count>=6)														//once 6 faces have been processed, break for - aka stop collecting
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
	// Show the best 20 eigenfaces
	for (int i = 0; i < min(20, eigenvectors.cols); i++) {
		// Create a continuous column vector from eigenvector #i.
		Mat eigenvector = eigenvectors.col(i).clone();
		Mat eigenface = getImageFrom1DFloatMat(eigenvector, DESIRED_FACE_HEIGHT);

		//test
		/*imshow(format("Eigenface%d", i), eigenface);
		waitKey(0);
		destroyAllWindows();*/
	}
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

double getSimilarity(const Mat A, const Mat B) {
	// Calculate the L2 relative error between the 2 images.
	double errorL2 = norm(A, B, CV_L2);
	// Scale the value since L2 is summed across all pixels.
	double similarity = errorL2 / (double)(A.rows * A.cols);
	return similarity;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void recogniseFace(Mat &input)	//TODO: Need to get face FIRST & then call this function - improve modularity
{
	int identity = -1;
	Mat img = input;
	Mat face;
	quickDetect(img);

	for (int i = 0; i < (int)faceRect.size(); i++)
	{
		faceProcessed = false;
		face = img(faceRect[i]);
		if (facefound)
		{
			detectEyes(face);
		}
	
			
		if (faceProcessed)
		{
			Mat preprocessedFace = face;	
			// Generate a face approximation by back-projecting the eigenvectors & eigenvalues.
			Mat reconstructedFace;
			reconstructedFace = reconstructFace(model, preprocessedFace);

			if (m_debug)//test
				if (reconstructedFace.data)
					imshow("reconstructedFace", reconstructedFace);

			// Verify whether the reconstructed face looks like the preprocessed face, otherwise it is probably an unknown person.
			double similarity = getSimilarity(preprocessedFace, reconstructedFace);

			string outputStr;
			if (similarity < UNKNOWN_PERSON_THRESHOLD) {
				// Identify who the person is in the preprocessed face image.
				identity = model->predict(preprocessedFace);
				outputStr = toString(identity);

				//Will train randomly with new recognised face - Done so to improve run speed
				random = rand() % 10 + 1;
				//cout << random; testing
				if (random == 7)
				{
					preprocessedFaces.push_back(preprocessedFace);
					faceLabels.push_back(identity);
					model->train(preprocessedFaces, faceLabels);
				}
			}
			else {
				// Since the confidence is low, assume it is an unknown person.
				outputStr = "Unknown";
			}
			putText(input, outputStr, Point(faceRect[i].x, faceRect[i].y), CV_FONT_VECTOR0, 1.0, CV_RGB(0, 255, 0), 2.0);
			cout << "Identity: " << outputStr << ". Similarity: " << similarity << endl;

			//// Show the confidence rating for the recognition in the mid-top of the display.
			//int cx = (input.cols - DESIRED_FACE_WIDTH) / 2;
			//Point ptBottomRight = Point(cx - 5, 8 + DESIRED_FACE_HEIGHT);
			//Point ptTopLeft = Point(cx - 15, 8);
			//// Draw a gray line showing the threshold for an "unknown" person.
			//Point ptThreshold = Point(ptTopLeft.x, ptBottomRight.y - (1.0 - UNKNOWN_PERSON_THRESHOLD) * DESIRED_FACE_HEIGHT);
			//rectangle(img, ptThreshold, Point(ptBottomRight.x, ptThreshold.y), CV_RGB(200, 200, 200), 1, CV_AA);
			//// Crop the confidence rating between 0.0 to 1.0, to show in the bar.
			//double confidenceRatio = 1.0 - min(max(similarity, 0.0), 1.0);
			//Point ptConfidence = Point(ptTopLeft.x, ptBottomRight.y - confidenceRatio * DESIRED_FACE_HEIGHT);
			//// Show the light-blue confidence bar.
			//rectangle(img, ptConfidence, ptBottomRight, CV_RGB(0, 255, 255), CV_FILLED, CV_AA);
			//// Show the gray border of the bar.
			//rectangle(img, ptTopLeft, ptBottomRight, CV_RGB(200, 200, 200), 1, CV_AA);

			//test
			//imshow("", img);
			//waitKey(0);
			//destroyAllWindows();
		}
	}
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename T> string toString(T t)
{
	ostringstream out;
	out << t;
	return out.str();
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace)
{
	// Since we can only reconstruct the face for some types of FaceRecognizer models (ie: Eigenfaces or Fisherfaces),
	// we should surround the OpenCV calls by a try/catch block so we don't crash for other models.
	try {

		// Get some required data from the FaceRecognizer model.
		Mat eigenvectors = model->get<Mat>("eigenvectors");
		Mat averageFaceRow = model->get<Mat>("mean");

		int faceHeight = preprocessedFace.rows;

		// Project the input image onto the PCA subspace.
		Mat projection = subspaceProject(eigenvectors, averageFaceRow, preprocessedFace.reshape(1, 1));
		//printMatInfo(projection, "projection");

		// Generate the reconstructed face back from the PCA subspace.
		Mat reconstructionRow = subspaceReconstruct(eigenvectors, averageFaceRow, projection);
		//printMatInfo(reconstructionRow, "reconstructionRow");

		// Convert the float row matrix to a regular 8-bit image. Note that we
		// shouldn't use "getImageFrom1DFloatMat()" because we don't want to normalize
		// the data since it is already at the perfect scale.

		// Make it a rectangular shaped image instead of a single row.
		Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
		// Convert the floating-point pixels to regular 8-bit uchar pixels.
		Mat reconstructedFace = Mat(reconstructionMat.size(), CV_8U);
		reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
		//printMatInfo(reconstructedFace, "reconstructedFace");

		return reconstructedFace;

	}
	catch (cv::Exception e) {
		//cout << "WARNING: Missing FaceRecognizer properties." << endl;
		return Mat();
	}
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
