#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <time.h>  
#include "Globals.h"
#include "Detect.h"

// Function Headers
template <typename T> string toString(T t);
void detectFaces(Mat &img);
void detectEyes(Mat &input);
void trainFace(string name,bool nameFound);
void recogniseFace(Mat &input);
void quickDetect(Mat &src);
char easytolower(char in);

Mat getImageFrom1DFloatMat(const Mat matrixRow, int height);
Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace);
inline const char * const BoolToString(bool b);
String getMonth(int month);
void readFile(string fileName);
void getDailyRegister(string month, string day, int intMonth, int intDay);
Rect drawButton(Mat img, string text, Point coord, int minWidth = 0);
bool isPointInRect(const Point pt, const Rect rc);
Rect drawString(Mat img, string text, Point coord, Scalar color, float fontScale = 0.5f, int thickness = 1, int fontFace = CV_FONT_VECTOR0);

// Global variables	
const double EYE_SX = 0.10;
const double EYE_SY = 0.19;
const double EYE_SW = 0.40;
const double EYE_SH = 0.36;
const double UNKNOWN_PERSON_THRESHOLD = 3000;
const int DETECTION_WIDTH = 800;
double min_face_size = 20;
double max_face_size = 300;
const int classSize = 30;
const int border = 8;

double startTime;						//Used to det if person is late
double currentTime;
double timePassed;
char mode = '0';						//Mode that programming is running in
char option = '0';
bool m_debug = true;					//Temp variable for testing. needs to be applied for all testing cases
bool eyeDetection = false;				//Will be used to skip histogram equalisation and resizing for eye detection
bool faceProcessed = false;				//Has the face been processed?
bool facefound = false;					//Has a face been found?
int numFaces = 0;						//The number of faces currently stored in yml file

VideoCapture cam(0); //webcam
vector<Rect> faceRect;					//Value of boxes around detected faces
vector<Mat> preprocessedFaces;			//Vector to store preprocessed faces
vector<int> faceLabels;					//Vector to store facelabels of preprocessed faces
map<int, string> stringLabels;			//matches int labels to strings (Name of each person)
vector<bool> Present(classSize);		//Is person present
vector<bool> Late(classSize);			//Is person late
vector<double> MinsPassed(classSize);	//What time was person first spotted in lecture

//On-screen buttons
Rect button_recog;
Rect button_new;
Rect button_train;
Rect button_data;

//Location of detectors
string face_cascade_name = "c:/opencv-build/install/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
string eye1_cascade_name = "c:/opencv-build/install/share/OpenCV/haarcascades/haarcascade_eye.xml";
string eye2_cascade_name = "c:/opencv-build/install/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier faceDetector;
CascadeClassifier eyeDetector1;
CascadeClassifier eyeDetector2;

Ptr<FaceRecognizer> model1 = createEigenFaceRecognizer();

//------------------------------------MOUSE CLICKS------------------------------------
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event != EVENT_LBUTTONDOWN)
		return;

	Point pt = Point(x, y);

	if (isPointInRect(pt, button_recog)) {
		destroyAllWindows();
		mode = '1';
	}
	else if (isPointInRect(pt, button_new)) {
		destroyAllWindows();
		mode = '2';
	}
	else if (isPointInRect(pt, button_train)) {
		destroyAllWindows();
		mode = '4';
	}
	else if (isPointInRect(pt, button_data)) {
		destroyAllWindows();
		mode = '3';
	}
	return;
		//cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
}

//**************************************************************************************************
//										START OF MAIN
//**************************************************************************************************

int main(void)
{
	bool run;				//used to end program
	srand(time(NULL));		//for random num gen - recognise will train model with new face when random num = 7

	time_t t = time(0);   
	struct tm *now = localtime(&t); //current time

	//Get the time at program run - corresponds to lecture start	
	startTime = (double)getTickCount();

	//-----------------Load detectors-----------------
	try{
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
	//model = Algorithm::create<FaceRecognizer>(facerecAlgorithm);
	if (model1.empty()) {
		cerr << "ERROR: The FaceRecognizer [" << facerecAlgorithm;
		cerr << "] is not available in your version of OpenCV. ";
		cerr << "Please update to OpenCV v2.4.1 or newer." << endl;
		exit(1);
	}

	
	//-----------------Load trained data-----------------
	try
	{
		model1->load("trainedData.yml");
		FileStorage fs("retrainModel.yml", FileStorage::READ);
		fs["mats"] >> preprocessedFaces;
		fs["labels"] >> faceLabels;
		fs.release();

		//get largest int label value
		for (int i = 0; i < faceLabels.size(); i++)
		{
			if (faceLabels.at(i) > numFaces)
			{
				numFaces = faceLabels.at(i);
			}
		}
		int i = 0;
		for (i = 0; !model1->getLabelInfo(i).compare("") == 0;i++)
		{
			stringLabels.insert(pair<int, string>(i, model1->getLabelInfo(i)));
		}
	}
	catch (const std::exception&)
	{
		//File not found
		numFaces = 0;
	}

	//-----------------Main program loop starts-----------------
	run = true;
	Mat img;
	string ID;

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
				printf("MAIN MENU\n\n");
				printf("Option 1 - Recognise Faces\n");						//Just box faces and display
				printf("Option 2 - Add New Student\n");							//Ask user for ID and train
				printf("Option 3 - Get Register Data\n");							//Register and student info
				printf("Option 4 - Improve Model\n");							//Add new student
				printf("Option 5 - Detect Faces\n");						//Attempt facial recognition	
				printf("Option 9 - Wipe Memory\n");							//Wipe data
				printf("\nType out an option number and hit ENTER\n(or s to EXIT)\n");
				printf("\nYour Choice: ");
				cin >> mode;
				system("cls");
				break;
			}

			case '5':
			{
				system("cls");
				while (mode=='5')
				{
					cam >> img;
					detectFaces(img);
					namedWindow("Automated Attendence", 1);
					
					button_recog = drawButton(img, "Recognise Mode", Point(border+30,border));
					button_new = drawButton(img, " Add Student", Point(button_recog.x+button_recog.width, button_recog.y),button_recog.width);
					button_train = drawButton(img, "Train Face", Point(button_new.x+button_recog.width, button_new.y), button_recog.width);
					button_data = drawButton(img, "Get Data", Point(button_train.x+button_recog.width, button_train.y), button_recog.width);

					setMouseCallback("Automated Attendence", CallBackFunc, NULL);
					imshow("Automated Attendence", img);

					// press 's' to escape
					if (waitKey(1) == 's') { mode = '0'; destroyAllWindows();  break; };
				}
				break;
			}

			case '4':
			{
				system("cls");
				printf("Enter student number (or s to return)\n");
				cin >> ID;
				transform(ID.begin(), ID.end(), ID.begin(), ::easytolower);	//all to lower case

				if (ID == "s")		//s to stop training
					mode = '0';
				else
				{	
					int i = 0;
					bool nameFound = false;

					try
					{
						while (!model1->getLabelInfo(i).empty())
						{
							if (model1->getLabelInfo(i).compare(ID) == 0)
							{
								nameFound = true;	//Person exists in model
								break;
							}
							i++;
						}
					}
					catch (const std::exception&)
					{
						nameFound = false;
					}
					
					if (nameFound == false)
					{
						system("cls");
						cout << "This student does not exist\nPlease Retry or ADD the student\n" << endl;
						system("pause");
					}
					else
					{
						system("cls");
						cout << "\nPlease ensure that no other faces are within frame\n" << endl;
						system("pause");
						trainFace(ID, nameFound);
						system("cls");
						cout << "Training Successfully Completed!!" << endl;
						cout << "Do you wish to continue training (Y/N): ";
						cin >> mode;
						if (mode == 'Y' || mode == 'y')
							mode = '4';
						else
							mode = '0';
					}
					cin.clear();
					cin.ignore(256, '\n');
					break;
				}
				break;
			}

			case '1':
			{
				int c = waitKey(1);
				if (preprocessedFaces.size()>0)
				{
					cam >> img;
					recogniseFace(img);
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
				if ((char)c == 's') 
				{ 
					mode = '0'; 
					destroyAllWindows(); 

					//Write to register
					String date;
					date = toString(now->tm_mday) + "_" + getMonth((now->tm_mon));
					ofstream outFile("Register/"+date + ".txt");
					outFile << "ID" << "\t" << "Stu Num" << "\t\t" << "Present" << "\t" << "Time" << "\t" << "Late" << endl;
					int i;
					for (i = 0; !model1->getLabelInfo(i).compare("") == 0; i++)
					{
						outFile << i << "\t" << model1->getLabelInfo(i) << "\t" << BoolToString(Present[i]) << "\t" << (int)MinsPassed[i] << "\t" << BoolToString(Late[i]) << endl;
					}
					outFile.close();
					//Save trained data
					model1->save("trainedData.yml");
					FileStorage fs("retrainModel.yml", FileStorage::WRITE);
					fs << "mats" << preprocessedFaces << "labels" << faceLabels;
					fs.release();
				}
				break;
			}
			case '3':
			{
				switch (option) //Data Menu
				{
					case '0':
					{
						system("cls");
						printf("GET-DATA MENU\n\n");
						printf("Option 1 - View All Course Participants\n");
						printf("Option 2 - View Daily Register\n");
						printf("\nType out an option number and hit ENTER\n(or s to EXIT)\n");
						printf("\nYour Choice: ");
						cin >> option;
						system("cls");
						break;
					}
					case '1':
					{
						system("cls");
						readFile("Register/Students.txt");
						cout << endl;
						system("pause");
						option = '0';
						break;
					}
					case '2':
					{
						option = '0';
						system("cls");
						string month, day;
						int intMonth, intDay;
						printf("Enter month number (or s to return) e.g 1 for Jan\n");
						cin >> month;

						if (month == "s" || month == "S")
						{
							system("cls");
							option = '0';
							break;
						}
						printf("Enter date e.g 17\n");
						cin >> day;

						system("cls");

						try
						{
							intMonth = stoi(month);
							intMonth -= 1;
							intDay = stoi(day);
						}
						catch (const std::exception&)
						{
							cout << "Invalid Entry - Not an integer value\n";
							break;
						}
						getDailyRegister(month, day, intMonth, intDay);
						break;
					}
					case 's':
					{
						option = '0';
						mode = '0';
						break;
					}
					case 'S':
					{
						option = '0';
						mode = '0';
						break;
					}
					default:
					{
						system("cls");
						cin.clear();
						cin.ignore(256, '\n');
						destroyAllWindows();
						option = '0';
						printf("Invalid Entry! Please Retry\n");
						system("pause");
						break;
					}
				}

				//-------------------------------------------------------------------
				cin.clear();
				break;
			}
			case '2':
			{
				system("cls");
				string stdNum, name;
				printf("Enter Student Number (or s to return)\n");
				cin >> stdNum;
				if (stdNum == "s"  || stdNum == "S")
				{
					mode = '0';
					break;
				}
				cin.clear();
				cin.ignore(256, '\n');
				cout << "Enter Student's name and surname\n";
				getline(cin, name);

				int i = 0;
				bool nameFound = false;
				while (!model1->getLabelInfo(i).empty())
				{
					if (model1->getLabelInfo(i).compare(stdNum) == 0)
					{
						nameFound = true;
						break;
					}
					i++;
				}
				system("cls");
				cout << "Please ensure that no other faces are within frame\n" << endl;
				system("pause");
				system("cls");
				trainFace(stdNum, nameFound);
				destroyAllWindows();
				if (!nameFound)
				{
					ofstream outFile("Register/Students.txt", ofstream::app);
					outFile << numFaces << "\t" << model1->getLabelInfo(numFaces) << "\t" << name << "\t\t" << endl;
					outFile.close();
				}	
				system("cls");
				cout << "Student Successfully Added!!" << endl;
				cout << "Do you wish to add another student? (Y/N): ";
				cin >> mode;
				if (mode == 'Y' || mode == 'y')
					mode = '2';
				else
					mode = '0';
				break;
			}
			case '9':
			{
				system("cls");
				cout << "WARNING!!! DELETES ALL TRAINING DATA: " << endl;
				cout << "Are you sure? (Y/N): ";
				cin >> mode;
				if (mode == 'Y' || mode == 'y')
					cout << "Okay. But I warned you..." << endl;
				else
				{
					mode = '0';
					break;
				}
					
				
				try
				{
					faceLabels.clear();
					stringLabels.clear();
					preprocessedFaces.clear();
					Present.clear();
					Late.clear();
					MinsPassed.clear();
					model1 = createEigenFaceRecognizer();
					remove("trainedData.yml");
					remove("retrainModel.yml");
					numFaces = 0;
				}
				catch (const std::exception&)
				{
					cout << "Data failed to delete" << endl << endl;
					system("pause");
					mode = '0';
					break;
				}
				cout << "Successfully Wiped Data"<< endl << endl;
				system("pause");
				mode = '0';
				break;
			}
			case '#':	//DEBUGGING CASE
			{
				system("cls");
				cin.clear();
				cin.ignore(256, '\n');
				destroyAllWindows();

				mode = '0';
				//cout << numFaces << endl;
				//system("pause");
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
				cin.clear();
				cin.ignore(256, '\n');
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

void quickDetect(Mat &input)
{
	Mat image = input;
	faceDetector.detectMultiScale(image, faceRect, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size), Size(max_face_size, max_face_size));
	printf("%d faces were detected\n\n", faceRect.size());

	// Draw rectangles around the detected faces
	for (int i = 0; i < (int)faceRect.size(); i++)
	{
		Point pt1(faceRect[i].x, faceRect[i].y);
		Point pt2((faceRect[i].x + faceRect[i].height), (faceRect[i].y + faceRect[i].width));
		rectangle(image, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}
	input = image;
	if ((int)faceRect.size()>0)
		facefound = true;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


void detectFaces(Mat &input)
{
	Mat img = input;
	Globals::detectManyObjects(img, faceDetector, faceRect, DETECTION_WIDTH);;

	//How many faces did multiscale detect? - [test parameter]
	printf("%d faces were detected\n\n", faceRect.size());

	//ADD RECTANGLES TO ORIGINAL IMAGE
	for (int i = 0; i < (int)faceRect.size(); i++)
	{
		detectEyes(img(faceRect[0]));
		if (!faceProcessed)
			break;
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

void detectEyes(Mat &input) //Takes  in single face and is used to determine if eyes were found
{
	Mat img = input;
	Mat topLeftOfFace;
	Mat topRightOfFace;
	Mat gray, face;
	face = img;

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

void trainFace(string name, bool nameFound)
{
	int labelNum;
	bool faceSaved = false;

	if (nameFound == false)	//If name doesn't exist as label - this is a new face/person
	{
		while(!model1->getLabelInfo(numFaces).compare("") == 0)
			numFaces++;

		labelNum = numFaces;
	}
	else
	{
		for (int i = 0; i <= numFaces; i++)
		{
			if (model1->getLabelInfo(i) == name)	//if name does exist, get int label value to add data to correct person
			{
				labelNum = i;
				break;
			}
		}
	}
	
	double imageDiff = 0;
	Mat img;
	Mat input,face;
	Mat displayedFrame;
	Mat old_prepreprocessedFace;
	Mat new_preprocessedFace;
	int count = 0;
	double old_time = (double)getTickCount();

	for (;;)//will break when 10 faces are found
	{
		cam.release();
		VideoCapture cam(0);
		cam >> img;

		if (!img.empty()) {
			input = img;
			img.copyTo(displayedFrame);

			facefound = false;
			faceProcessed = false;

			detectFaces(input);				//Find box around face
			face = input;				//Will be written to file

			//Display detected face whilst training
			imshow("", input);
			if (waitKey(1) == 's') {};

			if (facefound && (int)faceRect.size() == 1)
			{
					input = input(faceRect[0]);
					detectEyes(input);
			}
		}
		else {
			printf("(!)-- No captured frame --(!)\n\n");
		}

		//get difference in time - since last pic
		double current_time = (double)getTickCount();
		double timeDiff_seconds = (current_time - old_time) / getTickFrequency();

		if (faceProcessed)		//if a face was processed, will be stored in new_preprocessedFace
		{
			//test - show processed face
			//imshow(" ", input);
			//if (waitKey(1) == 's') {};

			if (!faceSaved && !nameFound)
			{
				imwrite("Faces/" + name + ".jpg", face(faceRect[0]));
				faceSaved = true;
			}
				
			new_preprocessedFace = input;

			if (old_prepreprocessedFace.data)		//if there exists an old face
			{
				imageDiff = Globals::getSimilarity(new_preprocessedFace, old_prepreprocessedFace);
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
				faceLabels.push_back(labelNum);
				faceLabels.push_back(labelNum);
				count++;
				// Keep a copy of the processed face,
				// to compare on next iteration.
				old_prepreprocessedFace = new_preprocessedFace;
				old_time = current_time;

				string c = to_string(count);
				imwrite("Faces/" + name + "_" + c + ".jpg", new_preprocessedFace);

				//TEST - face region display
				// Get access to the face region-of-interest.
				//Mat displayedFaceRegion = displayedFrame(faceRect.at(0));				//at(0) because only one face atm
				//																		// Add some brightness to each pixel of the face region.				//unneccesary - just to show user face detected
				//displayedFaceRegion += CV_RGB(90, 90, 90);
				//imshow("IMAGE", displayedFaceRegion);
				//waitKey(0);
				//destroyWindow("IMAGE");
			}
		}
		if (count >= 10)														//once enough faces have been processed, break for loop - aka stop collecting faces
			break;
	}

	model1->train(preprocessedFaces, faceLabels);
	stringLabels.insert(pair<int, string>(labelNum, name));
	model1->setLabelsInfo(stringLabels);											//String name that corresponds to numFace value
	model1->save("trainedData.yml");												//Write to file
	printf("trained\n\n");
	destroyAllWindows();

	FileStorage fs("retrainModel.yml", FileStorage::WRITE);
	fs << "mats" << preprocessedFaces << "labels" <<faceLabels;
	fs.release();
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void recogniseFace(Mat &input)	//TODO: Need to get face FIRST & then call this function - improve modularity
{
	int random;
	int identity = -1;
	Mat img = input;
	Mat face;
	double conf;
	quickDetect(img);

	for (int i = 0; i < (int)faceRect.size(); i++)
	{
		identity = -1;
		conf = 9999;
		faceProcessed = false;
		face = img(faceRect[i]);
		detectEyes(face);
			
		if (faceProcessed)
		{
			string outputStr;
			identity = -1;
			conf = 9999;

			// Identify who the person is in the preprocessed face image.
			model1->predict(face,identity,conf);
			outputStr = model1->getLabelInfo(identity);

			if (conf < UNKNOWN_PERSON_THRESHOLD)	//Person found
			{
				//Will train randomly with new recognised face - Done so to improve run speed
				random = rand() % 5 + 1;
				if (random == 3 && conf > UNKNOWN_PERSON_THRESHOLD - 200)	//EXPERIMENTAL - Might slow down recognition
				{
					preprocessedFaces.push_back(face);
					faceLabels.push_back(identity);
					model1->train(preprocessedFaces, faceLabels);
				}

				//Update register info if not already marked present
				if (!Present[identity])
				{
					if (identity > 0)
					{
						Present[identity] = true;

						currentTime = (double)getTickCount();
						timePassed = ((currentTime - startTime) / getTickFrequency()) / 60;
						MinsPassed[identity] = timePassed;

						if (timePassed > 15)	//TIME SINCE LECTURE STARTED
							Late[identity] = true;
					}
				}
			}
	
			if(conf > UNKNOWN_PERSON_THRESHOLD || identity <0) // Since the confidence is low, assume it is an unknown person.
				outputStr = "Unknown";

			//Display person's name above their face
			putText(input, outputStr, Point(faceRect[i].x, faceRect[i].y-5), CV_FONT_VECTOR0, 1.0, CV_RGB(0, 255, 0), 2.0);

			if(m_debug)
				cout << "Identity: " << identity << "| Stu Num: " << outputStr << "| Confidence: " << conf << endl;
		}
	}
}

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


char easytolower(char in) {
	if (in <= 'Z' && in >= 'A')
		return in - ('Z' - 'z');
	return in;
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

inline const char * const BoolToString(bool b)
{
	return b ? "[x]" : "[ ]";
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

String getMonth(int month)
{
	switch (month)
	{
	case 0:
		return "Jan";
		break;
	case 1:
		return "Feb";
		break;
	case 2:
		return "Mar";
		break;
	case 3:
		return "Apr";
		break;
	case 4:
		return "May";
		break;
	case 5:
		return "June";
		break;
	case 6:
		return "July";
		break;
	case 7:
		return "Aug";
		break;
	case 8:
		return "Sep";
		break;
	case 9:
		return "Oct";
		break;
	case 10:
		return "Nov";
		break;
	case 11:
		return "Dec";
		break;
	default:
		return "NULL";
		break;
	}
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void readFile(string fileName)
{
	string id, stuNum, pres, mins, delay;
	try
	{
		ifstream myfile(fileName);
		string line;
		if (myfile.is_open())
		{
			while (!myfile.eof())
			{
				getline(myfile, line);
				cout << line << '\n';
			}
			myfile.close();
		}
		else
		{
			cout << "No data exists for this date" << endl;
		}
	}
	catch (const std::exception&)
	{
		cout << "No data exists for this date" << endl;
	}
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void getDailyRegister(string month, string day,int intMonth,int intDay)
{

	if (intMonth < 13 && intMonth >= 0 && intDay < 32 && intDay>0)
	{
		string fileName = "Register/" + day + "_" + getMonth(intMonth) + ".txt";
		readFile(fileName);
		system("pause");
		mode = '0';
	}
	else
	{
		cout << "Incorrect date provided\n";
		system("pause");
	}

	return;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bool isPointInRect(const Point pt, const Rect rc)
{
	if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
		if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
			return true;

	return false;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rect drawButton(Mat img, string text, Point coord, int minWidth)
{
	int B = border;
	Point textCoord = Point(coord.x + B, coord.y + B);
	// Get the bounding box around the text.
	Rect rcText = drawString(img, text, textCoord, CV_RGB(0, 0, 0));
	// Draw a filled rectangle around the text.
	Rect rcButton = Rect(rcText.x - B, rcText.y - B, rcText.width + 2 * B, rcText.height + 2 * B);
	// Set a minimum button width.
	if (rcButton.width < minWidth)
		rcButton.width = minWidth;
	// Make a semi-transparent white rectangle.
	Mat matButton = img(rcButton);
	matButton += CV_RGB(90, 90, 90);
	// Draw a non-transparent white border.
	rectangle(img, rcButton, CV_RGB(200, 200, 200), 1, CV_AA);
	
	// Draw the actual text that will be displayed, using anti-aliasing.
	drawString(img, text, textCoord, CV_RGB(0,0,0));
	return rcButton;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rect drawString(Mat img, string text, Point coord, Scalar color, float fontScale, int thickness, int fontFace)
{
	// Get the text size & baseline.
	int baseline = 0;
	Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
	baseline += thickness;

	// Adjust the coords for left/right-justified or top/bottom-justified.
	if (coord.y >= 0) {
		// Coordinates are for the top-left corner of the text from the top-left of the image, so move down by one row.
		coord.y += textSize.height;
	}
	else {
		// Coordinates are for the bottom-left corner of the text from the bottom-left of the image, so come up from the bottom.
		coord.y += img.rows - baseline + 1;
	}
	// Become right-justified if desired.
	if (coord.x < 0) {
		coord.x += img.cols - textSize.width + 1;
	}

	// Get the bounding box around the text.
	Rect boundingRect = Rect(coord.x, coord.y - textSize.height, textSize.width, baseline + textSize.height);

	// Draw anti-aliased text.
	putText(img, text, coord, fontFace, fontScale, color, thickness, CV_AA);

	// Let the user know how big their text is, in case they want to arrange things.
	return boundingRect;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
