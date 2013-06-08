#ifndef _FILTER_H_
#define _FILTER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include <opencv2/core/internal.hpp>
using namespace cv;
using namespace std;

//box filter extention for 32F32F SIMD 
void boxFilter2( InputArray _src, OutputArray _dst, int ddepth,
                Size ksize, Point anchor,
                bool normalize, int borderType=cv::BORDER_REPLICATE);

//max, min filter and blur remove filter by using min-max filter
void maxFilter(const Mat& src, Mat& dest, Size ksize, int borderType=cv::BORDER_REPLICATE);
void minFilter(const Mat& src, Mat& dest, Size ksize, int borderType=cv::BORDER_REPLICATE);
void blurRemoveMinMax(Mat& src, Mat& dest, const int r, const int threshold);

//depth map hole filling
void fillOcclusion(Mat& src, int invalidvalue);// for disparity map
void fillOcclusionDepth(Mat& src, int invalidvalue);//for depth map
void fillOcclusionDepth(Mat& depth, Mat& image, int invalidvalue, int threshold);
void jointColorDepthFillOcclusion(const Mat& src, const Mat& guide, Mat& dest, const Size ksize, double threshold);

//remove Streaking Noise in stereo DP matching and hole filling function
void removeStreakingNoise(Mat& src, Mat& dest, int th);

void jointNearestFilter(const Mat& src, const Mat& before, const Size ksize, Mat& dest);
void jointNearestFilterBF(const Mat& src, const Mat& before, const Size ksize, Mat& dest);

//rgb interleave function for bilateral filter
void splitBGRLineInterleave( const Mat& src, Mat& dest);

//bilateral filter functions
enum
{
BILATERAL_NORMAL = 0,
BILATERAL_SEPARABLE,
BILATERAL_ORDER2,//underconstruction
BILATERAL_ORDER2_SEPARABLE//underconstruction
};

void bilateralFilterBase( const Mat& src, Mat& dst, int d,
	double sigma_color, double sigma_space,int borderType=cv::BORDER_REPLICATE);
void bilateralWeightMapBase( const Mat& src, Mat& dst, int d,
	double sigma_color, double sigma_space,int borderType=cv::BORDER_REPLICATE);

void bilateralFilter(const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method=BILATERAL_NORMAL, int borderType=cv::BORDER_REPLICATE);
void weightedBilateralFilter(const Mat& src, Mat& weight, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method=BILATERAL_NORMAL, int borderType=cv::BORDER_REPLICATE);
void bilateralWeightMap(const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method=BILATERAL_NORMAL, int borderType=cv::BORDER_REPLICATE);

void jointBilateralFilterBase( const Mat& src,const Mat& joint, Mat& dst, int d,
	double sigma_color, double sigma_space,int borderType=4);//basic implimentation

void jointBilateralFilter(const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method=BILATERAL_NORMAL, int borderType=cv::BORDER_REPLICATE);
void weightedJointBilateralFilter(const Mat& src, Mat& weightMap,const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method=BILATERAL_NORMAL, int borderType=cv::BORDER_REPLICATE);

void trilateralWeightMapBase( const Mat& src, const Mat& guide, Mat& dst, int d,
	double sigma_color, double sigma_guide_color, double sigma_space,int borderType=cv::BORDER_REPLICATE);

void trilateralFilter( const Mat& src, const Mat& guide, Mat& dst, int d,
	double sigma_color, double sigma_guide_color, double sigma_space,int borderType=4);

void trilateralFilter( const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int method=BILATERAL_NORMAL, int borderType=cv::BORDER_REPLICATE);
void weightedTrilateralFilter( const Mat& src, Mat& weightMap, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space,int method=BILATERAL_NORMAL, int borderType=cv::BORDER_REPLICATE);
void trilateralWeightMap( const Mat& src, const Mat& guide, Mat& dst, Size kernelSize, double sigma_color, double sigma_guide_color, double sigma_space, int method=BILATERAL_NORMAL, int borderType=cv::BORDER_REPLICATE);


void weightedJointBilateralRefinement_8u( const vector<Mat>& srcVolume, Mat& weight, const Mat& guide, vector<Mat>& destVolume, Size kernelSize, double sigma_color, double sigma_space, int borderType );


//guided filter
void guidedFilter(const cv::Mat& src, const cv::Mat& guidance, cv::Mat& dest, const int radius,const float eps);
void guidedFilter(const cv::Mat& src, cv::Mat& dest, const int radius,const float eps);

void guidedFilterTBB(const Mat& src, Mat& dest, int radius,float eps, const int threadmax);
void guidedFilterTBB(const Mat& src,const Mat& guidance, Mat& dest, int radius,float eps, const int threadmax);
void guidedFilterBF(const Mat& src, Mat& guidance, Mat& dest, const int radius,const float eps);

#endif