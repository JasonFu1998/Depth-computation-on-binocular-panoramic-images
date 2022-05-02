/*************************************/
/* Stereo Matching and Range Finding */
/*************************************/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

const int imageWidth = 600;
const int imageHeight = 3000;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;                     // area after cropping
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;
Mat Rl, Rr, Pl, Pr, Q;              // calibration of rotation matrix R, projection matrix P, and reprojection matrix Q
Mat xyz;

Point origin;
Rect selection;
bool selectObject = false;


Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 32, 5);

/*
pre-calibrated camera parameters
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 952.72830, 0, 1983.30000,
	0, 1241.80000, -664.90840,
	0, 0, 1.0);

Mat distCoeffL = (Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);

Mat cameraMatrixR = (Mat_<double>(3, 3) << 952.72830, 0, 1983.30000,
	0, 1241.80000, -664.90840,
	0, 0, 1.0);

Mat distCoeffR = (Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);

Mat T = (Mat_<double>(3, 1) << -50.0, 0.0, 0.0);

Mat rec = (Mat_<double>(3, 1) << 0.0, 0.0, 0.0);

Mat R;

static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 16.0e4;
	FILE* fp = fopen(filename, "wt");
	printf("%d %d \n", mat.rows, mat.cols);
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
			
		}
	}
	fclose(fp);
}

/*Depth map coloration*/
void GenerateFalseMap(cv::Mat &src, cv::Mat &disp)
{
	// color map
	float max_val = 255.0f;
	float map[8][4] = { { 0,0,0,114 },{ 0,0,1,185 },{ 1,0,0,114 },{ 1,0,1,174 },
	{ 0,1,0,114 },{ 0,1,1,185 },{ 1,1,0,114 },{ 1,1,1,0 } };
	float sum = 0;
	for (int i = 0; i < 8; i++)
		sum += map[i][3];

	float weights[8]; // relative weights
	float cumsum[8];  // cumulative weights
	cumsum[0] = 0;
	for (int i = 0; i < 7; i++) {
		weights[i] = sum / map[i][3];
		cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
	}

	int height_ = src.rows;
	int width_ = src.cols;
	// for all pixels do
	for (int v = 0; v < height_; v++) {
		for (int u = 0; u < width_; u++) {

			// get normalized value
			float val = std::min(std::max(src.data[v*width_ + u] / max_val, 0.0f), 1.0f);

			// find bin
			int i;
			for (i = 0; i < 7; i++)
				if (val < cumsum[i + 1])
					break;

			// compute red/green/blue values
			float   w = 1.0 - (val - cumsum[i]) * weights[i];
			uchar r = (uchar)((w*map[i][0] + (1.0 - w)*map[i + 1][0]) * 255.0);
			uchar g = (uchar)((w*map[i][1] + (1.0 - w)*map[i + 1][1]) * 255.0);
			uchar b = (uchar)((w*map[i][2] + (1.0 - w)*map[i + 1][2]) * 255.0);
			// continuous RGB memory storage
			disp.data[v*width_ * 3 + 3 * u + 0] = b;
			disp.data[v*width_ * 3 + 3 * u + 1] = g;
			disp.data[v*width_ * 3 + 3 * u + 2] = r;
		}
	}
}

/*Stereo matching*/
void stereo_match(int, void*)
{
	sgbm->setPreFilterCap(15);
	int SADWindowSize =  5;       // set based on users' actual situation
	int NumOfDisparities = 32;    // set based on users' actual situation
	int UniquenessRatio = 10;     // set based on users' actual situation
	sgbm->setBlockSize(SADWindowSize);
	int cn = rectifyImageL.channels();

	sgbm->setP1(8 * cn * SADWindowSize * SADWindowSize);
	sgbm->setP2(32 * cn * SADWindowSize * SADWindowSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(NumOfDisparities);
	sgbm->setUniquenessRatio(UniquenessRatio);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(10);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setMode(StereoSGBM::MODE_SGBM);
    Mat disp, dispf, disp8;
	sgbm->compute(rectifyImageL, rectifyImageR, disp);
	// remove black matte
	Mat img1p, img2p;
	copyMakeBorder(rectifyImageL, img1p, 0, 0, NumOfDisparities, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(rectifyImageR, img2p, 0, 0, NumOfDisparities, 0, IPL_BORDER_REPLICATE);
	dispf = disp.colRange(NumOfDisparities, img2p.cols- NumOfDisparities);

	dispf.convertTo(disp8, CV_8U, 255 / (NumOfDisparities *16.));
	reprojectImageTo3D(dispf, xyz, Q, true);
	xyz = xyz * 16;
	imshow("disparity", disp8);
	Mat color(dispf.size(), CV_8UC3);
	GenerateFalseMap(disp8, color);
	imshow("disparity", color);
	saveXYZ("xyz.xls", xyz);
}

/*Callback of the mouse operations*/
static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);
	}

	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
		break;
	case EVENT_LBUTTONUP:
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			break;
	}
}

int main()
{
	/*Stereo rectification*/
	Rodrigues(rec, R);
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY, 0, imageSize, &validROIL, &validROIR);
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_16SC2, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_16SC2, mapRx, mapRy);

	rgbImageL = imread("left.jpg", CV_LOAD_IMAGE_COLOR);
	rgbImageR = imread("right.jpg", CV_LOAD_IMAGE_COLOR);
	
	/*The images of the left and right camera are coplanar and aligned after the remap*/
	remap(rgbImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
	remap(rgbImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

	// draw on one canvas
	Mat canvas;
	double sf;
	int w, h;
	sf = 700. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
	canvas.create(h, w * 2, CV_8UC3);

	// left image painted on canvas
	Mat canvasPart = canvas(Rect(w * 0, 0, w, h));
	resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     // scale the image to the same size as canvasPart
	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),
		cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));             // get the cropped area
    // rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);
	cout << "Painted ImageL" << endl;

	// right image painted on canvas
	canvasPart = canvas(Rect(w, 0, w, h));
	resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	// rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
	cout << "Painted ImageR" << endl;

	// draw the corresponding lines
	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
	imshow("rectified", canvas);

	namedWindow("disparity", CV_WINDOW_NORMAL);
	setMouseCallback("disparity", onMouse, 0);
	stereo_match(0, 0);

	waitKey(0);
	return 0;
}
