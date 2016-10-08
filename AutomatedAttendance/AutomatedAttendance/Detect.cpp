#include "Detect.h"


void Detect::faceProcessing(Point leftEye, Point rightEye, Mat & gray, bool &faceProcessed)
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