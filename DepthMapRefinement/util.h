#ifndef _UTIL_H_
#define _UTIL_H_
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/highgui.h>

#include <iostream>

using namespace cv;
using namespace std;

#include <stdarg.h>


double getPSNR(Mat& src1, Mat& src2);

void alphaBlend(const Mat& src1, const Mat& src2, double alpha, Mat& dest);
void alphaBlend(const Mat& src1, const Mat& src2, const Mat& alpha, Mat& dest);

void guiAlphaBlend(const Mat& src1, const Mat& src2);

class ConsoleImage
{
private:
	int count;
	std::vector<std::string> strings;

public:
	cv::Mat show;
	ConsoleImage(cv::Size size=Size(640,480));
	~ConsoleImage();
	void printData();
	void clear();
	void operator()(string src);
	void operator()(const char *format, ...);
	void operator()(cv::Scalar color, const char *format, ...);
};

enum
{
	TIME_NSEC=0,
	TIME_MSEC,
	TIME_SEC,
	TIME_MIN,
	TIME_HOUR
};

class CalcTime
{
	int64 pre;
	string mes;

	int timeMode;

	double cTime;
	bool _isShow;


public:

	void start();
	void setMode(int mode);//単位
	void setMessage(string src);
	void restart();//再計測開始
	double getTime();//時間を取得
	void show();//cout<< time
	void show(string message);//cout<< time

	CalcTime(string message="time ", int mode=TIME_MSEC ,bool isShow=true);
	~CalcTime();
};


class StereoEval
{
	void threshmap_init();
public:
	bool isInit;
	string message;
	Mat state_all;
	Mat state_nonocc;
	Mat state_disc;
	Mat ground_truth;
	Mat mask_all;
	Mat all_th;
	Mat mask_nonocc; 
	Mat nonocc_th;
	Mat mask_disc;
	Mat disc_th;
	double amp;
	double all;
	double nonocc;
	double disc;

	double allMSE;
	double nonoccMSE;
	double discMSE;

	
	void init(Mat& ground_truth_, Mat& mask_nonocc_, Mat& mask_all_, Mat& mask_disc_, double amp_);
	StereoEval();
	StereoEval(char* ground_truth_, char* mask_nonocc_, char* mask_all_, char* mask_disc_, double amp_);
	StereoEval(Mat& ground_truth_, Mat& mask_nonocc_, Mat& mask_all_, Mat& mask_disc_, double amp_);
	~StereoEval(){;}
	void  getBadPixel(Mat& src, double threshold=1.0, bool isPrint=true);
	void getMSE(Mat& src, bool isPrint=true);
	virtual void operator() (Mat& src, double threshold=1.0, bool isPrint=true, int disparity_scale=1);
	void compare (Mat& before, Mat& after, double threshold=1.0, bool isPrint=true);
};
#endif