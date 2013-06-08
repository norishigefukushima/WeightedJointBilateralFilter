#include "filter.h"
#include "util.h"
#include "viewSynthesis.h"
#include <opencv2/opencv.hpp>
#include <omp.h>

#ifdef _DEBUG
//#pragma comment(lib, "opencv_video245d.lib")
//#pragma comment(lib, "opencv_ts245d.lib")
//#pragma comment(lib, "opencv_stitching245d.lib")
//#pragma comment(lib, "opencv_photo245d.lib")
//#pragma comment(lib, "opencv_objdetect245d.lib")
//#pragma comment(lib, "opencv_ml245d.lib")
//#pragma comment(lib, "opencv_legacy245d.lib")
#pragma comment(lib, "opencv_imgproc245d.lib")
#pragma comment(lib, "opencv_highgui245d.lib")
//#pragma comment(lib, "opencv_haartraining_engine.lib")
//#pragma comment(lib, "opencv_gpu245d.lib")
//#pragma comment(lib, "opencv_flann245d.lib")
//#pragma comment(lib, "opencv_features2d245d.lib")
#pragma comment(lib, "opencv_core245d.lib")
#pragma comment(lib, "opencv_contrib245d.lib")
#pragma comment(lib, "opencv_calib3d245d.lib")
#else
//#pragma comment(lib, "opencv_video245.lib")
//#pragma comment(lib, "opencv_ts245.lib")
//#pragma comment(lib, "opencv_stitching245.lib")
//#pragma comment(lib, "opencv_photo245.lib")
//#pragma comment(lib, "opencv_objdetect245.lib")
//#pragma comment(lib, "opencv_ml245.lib")
//#pragma comment(lib, "opencv_legacy245.lib")
#pragma comment(lib, "opencv_imgproc245.lib")
#pragma comment(lib, "opencv_highgui245.lib")
//#pragma comment(lib, "opencv_haartraining_engine.lib")
//#pragma comment(lib, "opencv_gpu245.lib")
//#pragma comment(lib, "opencv_flann245.lib")
//#pragma comment(lib, "opencv_features2d245.lib")
#pragma comment(lib, "opencv_core245.lib")
#pragma comment(lib, "opencv_contrib245.lib")
#pragma comment(lib, "opencv_calib3d245.lib")
#endif

void jointNearestTest()
{
	string srcimagedir = "./dataset/kinect/meeting_small_1_1.png";
	Mat src8u = imread(srcimagedir,0);
	Mat src16u,src16s,src32f;
	string wname = "joint Nearest";
	namedWindow(wname);
	int amp = 0;createTrackbar("amp",wname,&amp,100);
	int r = 3;createTrackbar("r",wname,&r,20);

	int key=0;
	Mat show;
	while(key!='q')
	{
		src8u.convertTo(src16u,CV_16U,amp+1);
		src8u.convertTo(src16s,CV_16S,amp+1);
		src8u.convertTo(src32f,CV_32F,amp+1);
		cout<<"------------------"<<endl;
		cout<<"Joint Nearest Test: ";
		double minv,maxv; minMaxLoc(src16u,&minv,&maxv);cout<<format("range: %f %f\n\n",minv,maxv);

		int d = 2*r+1;

		{
			Mat dest1,dest2,blursrc;
			GaussianBlur(src8u,blursrc,Size(d,d),d);
			{
				CalcTime t("8u Base");
				jointNearestFilterBF(blursrc,src8u,Size(d,d),dest2);

			}
			{
				CalcTime t("8u SSE");
				jointNearestFilter(blursrc,src8u,Size(d,d),dest1);
			}
			cout<<"joint neaest 8u:"<<getPSNR(dest1,dest2)<<" dB"<<endl;
		}
		{
			Mat dest1,dest2,blursrc;
			GaussianBlur(src16u,blursrc,Size(d,d),d);
			{
				CalcTime t("16u Base");
				jointNearestFilterBF(blursrc,src16u,Size(d,d),dest2);
			}
			{
				CalcTime t("16u SSE");

				jointNearestFilter(blursrc,src16u,Size(d,d),dest1);
			}

			cout<<"joint neaest 16u:"<<getPSNR(dest1,dest2)<<" dB"<<endl;
		}
		{
			Mat dest1,dest2,blursrc;
			GaussianBlur(src16s,blursrc,Size(d,d),d);
			{
				CalcTime t("16s Base");
				jointNearestFilterBF(blursrc,src16s,Size(d,d),dest2);
			}
			{
				CalcTime t("16s SSE");
				jointNearestFilter(blursrc,src16s,Size(d,d),dest1);
			}


			cout<<"joint neaest 16s:"<<getPSNR(dest1,dest2)<<" dB"<<endl;
		}
		{
			Mat dest1,dest2,blursrc;
			GaussianBlur(src32f,blursrc,Size(d,d),d);
			{
				CalcTime t("32f Base");
				jointNearestFilterBF(blursrc,src32f,Size(d,d),dest2);
			}
			{
				CalcTime t("32f SSE");
				jointNearestFilter(blursrc,src32f,Size(d,d),dest1);
			}
			cout<<"joint neaest 32f:"<<getPSNR(dest1,dest2)<<" dB"<<endl;
			dest1.convertTo(show,CV_8U,(1.0/(amp+1.0)));
		}
		imshow(wname,show);
		key = waitKey(1);
	}
}

void depth162depth8Color(Mat& src, Mat& dest, double minv, double maxv)
{
	Mat depthmap8u;
	src-=(short)minv;
	src.convertTo(depthmap8u,CV_8U,255.0/(maxv-minv));
	applyColorMap(depthmap8u,dest,2);
}
//float input is up to 256
void testKinectRefinement()
{
	//string srcimagedir = "./dataset/kinect/desk_1_1.png";
	//string srcdepthdir = "./dataset/kinect/desk_1_1_depth.png";
	string srcimagedir = "./dataset/kinect/meeting_small_1_1.png";
	string srcdepthdir = "./dataset/kinect/meeting_small_1_1_depth.png";

	Mat srcImage = imread(srcimagedir);
	if(srcImage.empty())cout<<srcimagedir<<" is invalid dir\n";
	Mat srcImageGray; cvtColor(srcImage,srcImageGray,CV_BGR2GRAY);
	Mat srcImagef;srcImageGray.convertTo(srcImagef,CV_32F);

	Mat srcDepth = imread(srcdepthdir,-1);
	if(srcDepth.empty())cout<<srcdepthdir<<" is invalid dir\n";
	string wname = "Kinect Refinement";
	namedWindow(wname);

	int alpha = 0; createTrackbar("alpha",wname,&alpha,100);
	int sw = 8; createTrackbar("sw",wname,&sw,8);
	int r = 3; createTrackbar("r",wname,&r,20);
	int sigs = 30; createTrackbar("sig_s",wname,&sigs,200);
	int sigc = 50; createTrackbar("sig_c",wname,&sigc,255);
	int sigc2 = 50; createTrackbar("sig_c2",wname,&sigc2,255);

	int pr = 2; createTrackbar("pr",wname,&pr,20);

	Mat filledDepth = srcDepth.clone();
	fillOcclusionDepth(filledDepth,0);
	Mat tp;
	transpose(filledDepth,tp);
	fillOcclusionDepth(tp,0);
	transpose(tp,filledDepth);

	Mat filledDepthf; filledDepth.convertTo(filledDepthf,CV_32F);

	double minv,maxv;
	minMaxLoc(filledDepth,&minv,&maxv);

	int key = 0;
	Mat depthout,depthShow;
	cout<<"alpha: alpha blending src image and output depth map\n";
	cout<<"sw: switch output depth map\n";
	cout<<"sw=0: RAW depth map\n";
	cout<<"sw=1: occlusion filled depth map\n";
	cout<<"sw=2: filtered by Gauusian filter\n";
	cout<<"sw=3: filtered by bilateral filter\n";
	cout<<"sw=4: filtered by joint bilateral filter\n";
	cout<<"sw=5: filtered by guided filter\n";
	cout<<"sw=6: filtered by guided filter(src image only)\n";
	cout<<"sw=7: filtered by proposed filter\n";
	cout<<"sw=8: filtered by proposed filter fastest setting\n";
	cout<<"press 'v' for calling view synthesis function\n";

	bool isGray = false;
	while(key!='q')
	{
		//static int count=0;cout<<count++<<endl;

		double ss = sigs/10.0;
		double sc = sigc/10.0;
		double sc2 = sigc2;
		int d = 2*r+1;

		if(sw == 0)
		{
			srcDepth.copyTo(depthout);
		}
		else if(sw == 1)
		{
			filledDepth = srcDepth.clone();
			fillOcclusionDepth(filledDepth,srcImage,0,(int)sc*10);
			removeStreakingNoise(filledDepth,filledDepth,3);
			Mat tp;
			transpose(filledDepth,tp);
			fillOcclusionDepth(tp,0);
			transpose(tp,filledDepth);
			filledDepth.convertTo(filledDepthf,CV_32F);

			/*
			srcDepth.copyTo(filledDepth);
			for(int i=0;i<pr;i++)
			jointColorDepthFillOcclusion(filledDepth,srcImageGray,filledDepth,Size(d,d),sc*10);
			filledDepth.convertTo(depthout,CV_16U);
			filledDepth.convertTo(filledDepthf,CV_32F);
			*/

			/*srcDepth.copyTo(filledDepth);
			filledDepth.convertTo(filledDepthf,CV_32F);

			for(int i=0;i<pr;i++)
			{
			Mat w,filledDepth2;
			compare(filledDepthf,0,w,cv::CMP_NE);
			Mat wmap;w.convertTo(wmap,CV_32F);
			weightedJointBilateralFilter(filledDepthf,wmap,srcImagef,filledDepth2,Size(d,d),sc,ss);
			jointNearestFilter(filledDepth2,filledDepthf,Size(d,d),filledDepthf);
			}*/
			filledDepthf.convertTo(depthout,CV_16U);
		}
		else if(sw == 2)
		{
			CalcTime t("Gaussian ");
			Mat filteredDepth;
			GaussianBlur(filledDepth, filteredDepth,Size(d,d),ss);

			filteredDepth.convertTo(depthout,CV_16U);
		}
		else if(sw == 3)
		{
			CalcTime t("Bilateral ");
			Mat filteredDepthf = Mat::ones(srcDepth.size(),CV_32F);
			bilateralFilter(filledDepthf, filteredDepthf,Size(d,d),sc*10,ss);
			filteredDepthf.convertTo(depthout,CV_16U);
		}
		else if(sw == 4)
		{
			CalcTime t("Joint ");
			Mat filteredDepthf = Mat::ones(srcDepth.size(),CV_32F);
			jointBilateralFilter(filledDepthf,srcImagef,filteredDepthf,Size(d,d),sc,ss,0);
			filteredDepthf.convertTo(depthout,CV_16U);
		}
		else if(sw == 5)
		{
			CalcTime t("Guided src guide");
			Mat filteredDepthf = Mat::ones(srcDepth.size(),CV_32F);
			guidedFilterTBB(filledDepthf,srcImagef,filteredDepthf,d,(float)(sc*0.001),8);
			filteredDepthf.convertTo(depthout,CV_16U);
		}
		else if(sw == 6)
		{
			CalcTime t("Guided src only");
			Mat filteredDepthf = Mat::ones(srcDepth.size(),CV_32F);
			guidedFilterTBB(filledDepthf,filteredDepthf,d,(float)(sc*0.1),8);
			filteredDepthf.convertTo(depthout,CV_16U);
		}
		else if(sw == 7)
		{
			Mat weight;
			{
				CalcTime t("Prop. ");
				Mat filteredDepthf = Mat::ones(srcDepth.size(),CV_32F);

				//bilateralWeightMap(srcImagef,weight,Size(d,d),sc,ss);
				trilateralWeightMap(srcImagef,filledDepthf,weight,Size(d,d),sc,sc2,ss);
				weightedJointBilateralFilter(filledDepthf,weight,srcImagef,filteredDepthf,Size(d,d),sc,ss,0);

				filteredDepthf.convertTo(depthout,CV_16U);
				jointNearestFilter(depthout,filledDepth,Size(2*pr+1,2*pr+1),depthout);
			}

			double minv,maxv;
			minMaxLoc(weight,&minv,&maxv);
			Mat wmap;
			weight.convertTo(wmap,CV_8U,255.0/maxv);
			imshow("weightMap",wmap);
		}
		else if(sw == 8)
		{
			CalcTime t("Prop. fastest");
			Mat filteredDepthf = Mat::ones(srcDepth.size(),CV_32F);

			jointBilateralFilter(filledDepthf,srcImagef,filteredDepthf,Size(d,d),sc,ss,BILATERAL_SEPARABLE);
			filteredDepthf.convertTo(depthout,CV_16U);
			jointNearestFilter(depthout,filledDepth,Size(2*pr+1,2*pr+1),depthout);
		}

		if(key=='v')
		{
			static StereoViewSynthesis svs;
			svs.isPostFilter=0;
			Mat disp;

			int amp = 16;
			const int mindepth = 4000;
			Mat temp;
			divide((float)mindepth*amp,depthout,temp,1.0,CV_32F);
			temp.convertTo(disp,CV_16U);
			svs.preview(srcImage,disp,0,4);
		}
		if(key=='g') isGray = isGray ? false:true;

		if(key == 'a')
		{
			// show all data
			imshow("srcImage",srcImage);
			imshow("filledDepth",srcImage);
		}

		depth162depth8Color(depthout,depthShow,minv,maxv);

		if(!isGray)
		{
			addWeighted(srcImage,alpha/100.0,depthShow,1.0-alpha/100.0,0.0,depthShow);
		}
		else
		{
			Mat temp;
			cvtColor(srcImageGray,temp,CV_GRAY2BGR);
			addWeighted(temp,alpha/100.0,depthShow,1.0-alpha/100.0,0.0,depthShow);
		}
		imshow(wname,depthShow);
		key = waitKey(1);
	}
}


void testStereoRefinementEval()
{
	string wname = "refinement";
	namedWindow(wname);

	char* dir="dataset/Middlebury/";
	char* sequence="teddy";
	const int DISPARITY_MAX = 16*3;
	const int amp=4;
	int disparityMin = 15;
	int disparityRange = DISPARITY_MAX;
	char name[128];

	sprintf(name,"%s%s/%s.png",dir,sequence,"groundtruth");
	Mat gt = imread(name,0);

	sprintf(name,"%s%s/%s.png",dir,sequence,"all");
	Mat all=imread(name,0);
	sprintf(name,"%s%s/%s.png",dir,sequence,"disc");
	Mat disc=imread(name,0);
	sprintf(name,"%s%s/%s.png",dir,sequence,"nonocc");
	Mat nonocc=imread(name,0);
	StereoEval eval(gt,nonocc,all,disc,amp);

	Mat src_ = imread("dataset/img_stereo/sgm.png",0);
	fillOcclusion(src_, 0);
	//removeStreakingNoise(src_,src_,8);
	Mat srcg = imread("dataset/img_stereo/sgm.png",0);
	//Mat guide = imread("dataset/img_stereo/dp.png",0);

	Mat guide_ = imread("dataset/img_stereo/teddy.png",0);
	Mat guidec = imread("dataset/img_stereo/teddy.png");
	Mat src,guide;
	int ress = 1;
	resize(src_,src,Size(src_.cols*ress,src_.rows*ress));
	resize(guide_,guide,Size(src_.cols*ress,src_.rows*ress));
	//resize(src_,src,Size(1920,1080));
	//resize(guide_,guide,Size(1920,1080));

	imshow("src",src);imshow("guide",guide);

	vector<Mat> result(20);
	for(int i=0;i<20;i++)
	{
		result[i]=Mat::zeros(src.size(),CV_8U);
	}

	int sw = 0;createTrackbar("sw",wname,&sw,20-1);
	int r = 11;createTrackbar("r",wname,&r,20);

	int sigS = 70;createTrackbar("s",wname,&sigS,200);
	int sigC = 100;createTrackbar("c",wname,&sigC,1500);
	int sigWS = 100;createTrackbar("WS",wname,&sigWS,1500);
	int sigWC = 100;createTrackbar("WC",wname,&sigWC,1500);

	/*
	int minmaxr = 1;createTrackbar("minmaxr",wname,&minmaxr,25);
	int minmaxth = 4;createTrackbar("minmaxth",wname,&minmaxth,255);
	*/
	int pr = 1;createTrackbar("pr",wname,&pr,20);

	//int res = 1;createTrackbar("res",wname,&res,16);
	//int th = 3;createTrackbar("th",wname,&th,8);

	int key=0;
	int count=0;

	int bmethod=0;
	ConsoleImage ci;
	while(key!='q')
	{
		ci.clear();
		Mat weight = Mat::ones(src.size(),CV_32F);

		int d = 2*r+1;
		double swc=sigWC/10.0;
		double sws=sigWS/10.0;
		double sc=sigC/10.0;
		double ss=sigS/10.0;

		int index=0;
		src.copyTo(result[index++]);
		medianBlur(src,result[index++],3);
		{
			CalcTime t("bilateral SSE4");			
			bilateralFilter(src,result[index++],Size(2*r+1,2*r+1),sc,ss,bmethod,4);
		}
		{
			CalcTime t("weighted bilateral SSE4");			
			bilateralWeightMap(src,weight,Size(2*r+1,2*r+1),swc,sws,bmethod);
			weightedBilateralFilter(src,weight,result[index++],Size(2*r+1,2*r+1),sc,ss,bmethod,4);
		}
		{
			CalcTime t("joint bilateral SSE4 gray");			
			jointBilateralFilter(src,guide,result[index++],Size(2*r+1,2*r+1),sc,ss,bmethod,4);
		}
		{
			CalcTime t("joint bilateral SSE4 color");			
			jointBilateralFilter(src,guidec,result[index++],Size(2*r+1,2*r+1),sc,ss,bmethod,4);
		}
		{
			CalcTime t("weighted joint bilateral SSE4 gray");			
			bilateralWeightMap(src,weight,Size(2*r+1,2*r+1),swc,sws,bmethod);
			weightedJointBilateralFilter(src,weight,guide,result[index++],Size(2*r+1,2*r+1),sc,ss,bmethod,4);
		}
		{
			CalcTime t("weighted joint bilateral SSE4 color");			
			bilateralWeightMap(src,weight,Size(2*r+1,2*r+1),swc,sws,bmethod);
			weightedJointBilateralFilter(src,weight,guidec,result[index++],Size(2*r+1,2*r+1),sc,ss,bmethod,4);
		}
		{
			CalcTime t("blur remove + weighted joint bilateral SSE4 gray");			
			bilateralWeightMap(src,weight,Size(2*r+1,2*r+1),swc,sws,bmethod);
			weightedJointBilateralFilter(src,weight,guide,result[index++],Size(2*r+1,2*r+1),sc,ss,bmethod,4);
			jointNearestFilter(result[index-1],src,Size(2*pr+1,2*pr+1),result[index-1]);
		}
		{
			CalcTime t("blur remove + weighted joint bilateral SSE4 color");			

			bilateralWeightMap(src,weight,Size(2*r+1,2*r+1),swc,sws,bmethod);
			weightedJointBilateralFilter(src,weight,guidec,result[index++],Size(2*r+1,2*r+1),sc,ss,bmethod,4);
			jointNearestFilter(result[index-1],src,Size(2*pr+1,2*pr+1),result[index-1]);
		}
		{
			CalcTime t("blur remove + weighted joint bilateral SSE4 color");			
			Mat bfil;
			bilateralFilter(src,bfil,Size(2*r+1,2*r+1),5.0,10.0,bmethod,4);
			bilateralWeightMap(bfil,weight,Size(2*r+1,2*r+1),swc,sws,bmethod);
			weightedJointBilateralFilter(bfil,weight,guidec,result[index++],Size(2*r+1,2*r+1),sc,ss,bmethod,4);
			jointNearestFilter(result[index-1],bfil,Size(2*pr+1,2*pr+1),result[index-1]);
		}

		Mat wmap;
		weight.convertTo(wmap,CV_8U,255.0/(d*d));
		imshow("w",wmap);

		for(int i=0;i<index;i++)
		{
			eval(result[i]);
			ci(format("%02d: ",i)+eval.message);
		}

		imshow("console",ci.show);
		Mat show;
		applyColorMap(result[sw],show,2);
		imshow(wname,show);

		key = waitKey(100);
	}
	//	return 0;
}

void testTrilateral()
{
	string wname = "Trilateral";
	namedWindow(wname);

	Mat src = imread("dataset/img_stereo/dp.png",0);
	fillOcclusion(src, 0);
	//removeStreakingNoise(src_,src_,8);
	Mat srcg = imread("dataset/img_stereo/dp.png",0);
	//Mat guide = imread("dataset/img_stereo/dp.png",0);

	Mat guide = imread("dataset/img_stereo/teddy.png",0);
	Mat guidec = imread("dataset/img_stereo/teddy.png");

	imshow("src",src);imshow("guide",guide);

	Mat dest  = Mat::zeros(src.size(),src.type());
	Mat dest2 = Mat::zeros(src.size(),src.type());
	Mat dest3 = Mat::zeros(src.size(),src.type());

	//int sw = 0;createTrackbar("sw",wname,&sw,2);
	int r = 7;createTrackbar("r",wname,&r,20);

	int sigC = 100;createTrackbar("c",wname,&sigC,1500);
	int sigC2 = 100;createTrackbar("c2",wname,&sigC2,1500);
	int sigS = 70;createTrackbar("s",wname,&sigS,200);

	int key=0;

	Mat srcf;src.convertTo(srcf,CV_32F);
	Mat guidecf;guidec.convertTo(guidecf,CV_32F);
	Mat guidef;guide.convertTo(guidef,CV_32F);

	Mat destBaseGrayGray8u = Mat::zeros(src.size(),CV_8U);
	Mat destBaseGrayGray32f = Mat::zeros(src.size(),CV_32F);
	Mat destBaseColorColor8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destBaseColorColor32f = Mat::zeros(src.size(),CV_32FC3);
	Mat destBaseGrayColor8u = Mat::zeros(src.size(),CV_8U);
	Mat destBaseGrayColor32f = Mat::zeros(src.size(),CV_32F);
	Mat destBaseColorGray8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destBaseColorGray32f = Mat::zeros(src.size(),CV_32FC3);

	Mat destSSE4GrayGray8u = Mat::zeros(src.size(),CV_8U);
	Mat destSSE4GrayGray32f = Mat::zeros(src.size(),CV_32F);
	Mat destSSE4ColorColor8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destSSE4ColorColor32f = Mat::zeros(src.size(),CV_32FC3);
	Mat destSSE4GrayColor8u = Mat::zeros(src.size(),CV_8U);
	Mat destSSE4GrayColor32f = Mat::zeros(src.size(),CV_32F);
	Mat destSSE4ColorGray8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destSSE4ColorGray32f = Mat::zeros(src.size(),CV_32FC3);

	Mat destSSE4GrayGraySP8u = Mat::zeros(src.size(),CV_8U);
	Mat destSSE4GrayGraySP32f = Mat::zeros(src.size(),CV_32F);
	Mat destSSE4ColorColorSP8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destSSE4ColorColorSP32f = Mat::zeros(src.size(),CV_32FC3);
	Mat destSSE4GrayColorSP8u = Mat::zeros(src.size(),CV_8U);
	Mat destSSE4GrayColorSP32f = Mat::zeros(src.size(),CV_32F);
	Mat destSSE4ColorGraySP8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destSSE4ColorGraySP32f = Mat::zeros(src.size(),CV_32FC3);


	{
		double sc=sigC/10.0;
		double sc2=sigC2/10.0;
		double ss=sigS/10.0;
		int d = 2*r+1;
		cout<<"======================================="<<endl;
		cout<<"0. Trilateral filter check"<<endl;
		{
			Mat dst1,dst2;
			trilateralFilter(src,src,dst1,d,sc,10000000000000000.0,ss,cv::BORDER_REPLICATE);
			bilateralFilter(src,dst2,d,sc,ss,cv::BORDER_REPLICATE);
			cout<<"Trilateral check gray8u  :"<<getPSNR(dst1,dst2)<<" dB"<<endl;

			trilateralFilter(src,guide,dst1,d,10000000000000000.0,sc,ss,cv::BORDER_REPLICATE);
			jointBilateralFilterBase(src,guide,dst2,d,sc,ss,cv::BORDER_REPLICATE);
			cout<<"Trilateral check gray8u  :"<<getPSNR(dst1,dst2)<<" dB"<<endl;
		}
		{
			Mat dst1,dst2;
			trilateralFilter(guidec,guidec,dst1,d,sc,10000000000000000.0,ss,cv::BORDER_REPLICATE);
			bilateralFilter(guidec,dst2,d,sc,ss,cv::BORDER_REPLICATE);
			cout<<"Trilateral check color8u :"<<getPSNR(dst1,dst2)<<" dB"<<endl;

			trilateralFilter(guidec,guidec,dst1,d,10000000000000000.0,sc,ss,cv::BORDER_REPLICATE);
			jointBilateralFilterBase(guidec,guidec,dst2,d,sc,ss,cv::BORDER_REPLICATE);
			cout<<"Trilateral check color8u :"<<getPSNR(dst1,dst2)<<" dB"<<endl;
		}

		{
			Mat dst1,dst2;
			trilateralFilter(srcf,srcf,dst1,d,sc,10000000000000000.0,ss,cv::BORDER_REPLICATE);
			bilateralFilter(srcf,dst2,d,sc,ss,cv::BORDER_REPLICATE);
			cout<<"Trilateral check gray32f :"<<getPSNR(dst1,dst2)<<" dB"<<endl;

			trilateralFilter(srcf,guidef,dst1,d,10000000000000000.0,sc,ss,cv::BORDER_REPLICATE);
			jointBilateralFilterBase(srcf,guidef,dst2,d,sc,ss,cv::BORDER_REPLICATE);
			cout<<"Trilateral check gray32f  :"<<getPSNR(dst1,dst2)<<" dB"<<endl;
		}
		{
			Mat dst1,dst2;
			trilateralFilter(guidecf,guidecf,dst1,d,sc,10000000000000000.0,ss,cv::BORDER_REPLICATE);
			bilateralFilter(guidecf,dst2,d,sc,ss,cv::BORDER_REPLICATE);
			cout<<"Trilateral check color32f:"<<getPSNR(dst1,dst2)<<" dB"<<endl;

			trilateralFilter(guidecf,guidecf,dst1,d,10000000000000000.0,sc,ss,cv::BORDER_REPLICATE);
			jointBilateralFilterBase(guidecf,guidecf,dst2,d,sc,ss,cv::BORDER_REPLICATE);
			cout<<"Trilateral check color32f:"<<getPSNR(dst1,dst2)<<" dB"<<endl;
		}
	}

	while(key!='q')
	{
		Mat weight = Mat::ones(src.size(),CV_32F);
		int d = 2*r+1;
		double sc=sigC/10.0;
		double sc2=sigC2/10.0;
		double ss=sigS/10.0;

		cout<<"======================================="<<endl;
		cout<<"1. Trilateral filter Base test"<<endl;
		{
			CalcTime t("Base trilateral gray gray  8u");
			trilateralFilter(src,guide,destBaseGrayGray8u,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base trilateral color color  8u");
			trilateralFilter(guidec,guidec,destBaseColorColor8u,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base trilateral gray color  8u");
			trilateralFilter(src,guidec,destBaseGrayColor8u,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base trilateral color gray 8u");
			trilateralFilter(guidec,guide,destBaseColorGray8u,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base trilateral gray gray  32f");
			trilateralFilter(srcf,guidef,destBaseGrayGray32f,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base trilateral color color  32f");
			trilateralFilter(guidecf,guidecf,destBaseColorColor32f,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base trilateral gray color  32f");
			trilateralFilter(srcf,guidecf,destBaseGrayColor32f,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("Base trilateral color gray  32f");
			trilateralFilter(guidecf,guidef,destBaseColorGray32f,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}

		cout<<"======================================="<<endl;
		cout<<"2. trilateral filter SSE Normal test"<<endl;
		{
			CalcTime t("SSE4 trilateral gray gray  8u");
			trilateralFilter(src,guide,destSSE4GrayGray8u,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral color color  8u");
			trilateralFilter(guidec,guidec,destSSE4ColorColor8u,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 trilateral gray color  8u");
			trilateralFilter(src,guidec,destSSE4GrayColor8u,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral color gray 8u");
			trilateralFilter(guidec,guide,destSSE4ColorGray8u,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral gray gray  32f");
			trilateralFilter(srcf,guidef,destSSE4GrayGray32f,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 trilateral color color  32f");
			trilateralFilter(guidecf,guidecf,destSSE4ColorColor32f,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral gray color  32f");
			trilateralFilter(srcf,guidecf,destSSE4GrayColor32f,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral color gray  32f");
			trilateralFilter(guidecf,guidef,destSSE4ColorGray32f,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		cout<<"gray gray   8u: SSE4 normal:"<<getPSNR(destBaseGrayGray8u,destSSE4GrayGray8u)<<" dB"<<endl;
		cout<<"color color 8u: SSE4 normal:"<<getPSNR(destBaseColorColor8u,destSSE4ColorColor8u)<<" dB"<<endl;
		cout<<"gray color  8u: SSE4 normal:"<<getPSNR(destBaseGrayColor8u,destSSE4GrayColor8u)<<" dB"<<endl;
		cout<<"color gray  8u: SSE4 normal:"<<getPSNR(destBaseColorGray8u,destSSE4ColorGray8u)<<" dB"<<endl;

		cout<<"gray gray   32f: SSE4 normal:"<<getPSNR(destBaseGrayGray32f,destSSE4GrayGray32f)<<" dB"<<endl;
		cout<<"color color 32f: SSE4 normal:"<<getPSNR(destBaseColorColor32f,destSSE4ColorColor32f)<<" dB"<<endl;
		cout<<"gray color  32f: SSE4 normal:"<<getPSNR(destBaseGrayColor32f,destSSE4GrayColor32f)<<" dB"<<endl;
		cout<<"color gray  32f: SSE4 normal:"<<getPSNR(destBaseColorGray32f,destSSE4ColorGray32f)<<" dB"<<endl;

		cout<<"======================================="<<endl;
		cout<<"3. Trilateral filter SSE SP test"<<endl;
		{
			CalcTime t("SSE4 trilateral gray gray  8u");
			trilateralFilter(src,guide,destSSE4GrayGray8u,Size(d,d),sc,sc2,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral color color  8u");
			trilateralFilter(guidec,guidec,destSSE4ColorColor8u,Size(d,d),sc,sc2,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral gray color  8u");
			trilateralFilter(src,guidec,destSSE4GrayColor8u,Size(d,d),sc,sc2,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral gray 8u");
			trilateralFilter(guidec,guide,destSSE4ColorGray8u,Size(d,d),sc,sc2,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral gray gray  32f");
			trilateralFilter(srcf,guidef,destSSE4GrayGray32f,Size(d,d),sc,sc2,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral color color  32f");
			trilateralFilter(guidecf,guidecf,destSSE4ColorColor32f,Size(d,d),sc,sc2,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral gray color  32f");
			trilateralFilter(srcf,guidecf,destSSE4GrayColor32f,Size(d,d),sc,sc2,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral color gray  32f");
			trilateralFilter(guidecf,guidef,destSSE4ColorGray32f,Size(d,d),sc,sc2,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		cout<<"gray gray   8u: SSE4 SP :"<<getPSNR(destBaseGrayGray8u,destSSE4GrayGray8u)<<" dB"<<endl;
		cout<<"color color 8u: SSE4 SP :"<<getPSNR(destBaseColorColor8u,destSSE4ColorColor8u)<<" dB"<<endl;
		cout<<"gray color  8u: SSE4 SP :"<<getPSNR(destBaseGrayColor8u,destSSE4GrayColor8u)<<" dB"<<endl;
		cout<<"color gray  8u: SSE4 SP :"<<getPSNR(destBaseColorGray8u,destSSE4ColorGray8u)<<" dB"<<endl;

		cout<<"gray gray   32f: SSE4 SP:"<<getPSNR(destBaseGrayGray32f,destSSE4GrayGray32f)<<" dB"<<endl;
		cout<<"color color 32f: SSE4 SP:"<<getPSNR(destBaseColorColor32f,destSSE4ColorColor32f)<<" dB"<<endl;
		cout<<"gray color  32f: SSE4 SP:"<<getPSNR(destBaseGrayColor32f,destSSE4GrayColor32f)<<" dB"<<endl;
		cout<<"color gray  32f: SSE4 SP:"<<getPSNR(destBaseColorGray32f,destSSE4ColorGray32f)<<" dB"<<endl;

		cout<<"======================================="<<endl;
		cout<<"4. Weighted trilateral filter SSE Normal test"<<endl;
		{
			CalcTime t("SSE4 Trilateral gray gray  8u");
			weightedTrilateralFilter(src,weight,guide,destSSE4GrayGray8u,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral  color color  8u");
			weightedTrilateralFilter(guidec,weight,guidec,destSSE4ColorColor8u,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral  gray color  8u");
			weightedTrilateralFilter(src,weight,guidec,destSSE4GrayColor8u,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral  color gray 8u");
			weightedTrilateralFilter(guidec,weight,guide,destSSE4ColorGray8u,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 Trilateral gray gray  32f");
			weightedTrilateralFilter(srcf,weight,guidef,destSSE4GrayGray32f,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 Trilateral color color  32f");
			weightedTrilateralFilter(guidecf,weight,guidecf,destSSE4ColorColor32f,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 Trilateral gray color  32f");
			weightedTrilateralFilter(srcf,weight,guidecf,destSSE4GrayColor32f,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 Trilateral color gray  32f");
			weightedTrilateralFilter(guidecf,weight,guidef,destSSE4ColorGray32f,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		cout<<"gray gray   8u: SSE4 normal:"<<getPSNR(destBaseGrayGray8u,destSSE4GrayGray8u)<<" dB"<<endl;
		cout<<"color color 8u: SSE4 normal:"<<getPSNR(destBaseColorColor8u,destSSE4ColorColor8u)<<" dB"<<endl;
		cout<<"gray color  8u: SSE4 normal:"<<getPSNR(destBaseGrayColor8u,destSSE4GrayColor8u)<<" dB"<<endl;
		cout<<"color gray  8u: SSE4 normal:"<<getPSNR(destBaseColorGray8u,destSSE4ColorGray8u)<<" dB"<<endl;

		cout<<"gray gray   32f: SSE4 normal:"<<getPSNR(destBaseGrayGray32f,destSSE4GrayGray32f)<<" dB"<<endl;
		cout<<"color color 32f: SSE4 normal:"<<getPSNR(destBaseColorColor32f,destSSE4ColorColor32f)<<" dB"<<endl;
		cout<<"gray color  32f: SSE4 normal:"<<getPSNR(destBaseGrayColor32f,destSSE4GrayColor32f)<<" dB"<<endl;
		cout<<"color gray  32f: SSE4 normal:"<<getPSNR(destBaseColorGray32f,destSSE4ColorGray32f)<<" dB"<<endl;


		cout<<"======================================="<<endl;
		cout<<"5. Weighted Trilateral filter SSE SP test"<<endl;
		{
			CalcTime t("SSE4 trilateral gray gray  8u");
			weightedTrilateralFilter(src,weight,guide,destSSE4GrayGray8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral color color  8u");
			weightedTrilateralFilter(guidec,weight,guidec,destSSE4ColorColor8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral gray color  8u");
			weightedTrilateralFilter(src,weight,guidec,destSSE4GrayColor8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral color gray 8u");
			weightedTrilateralFilter(guidec,weight,guide,destSSE4ColorGray8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral gray gray  32f");
			weightedTrilateralFilter(srcf,weight,guidef,destSSE4GrayGray32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral color color  32f");
			weightedTrilateralFilter(guidecf,weight,guidecf,destSSE4ColorColor32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral gray color  32f");
			weightedTrilateralFilter(srcf,weight,guidecf,destSSE4GrayColor32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 trilateral color gray  32f");
			weightedJointBilateralFilter(guidecf,weight,guidef,destSSE4ColorGray32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}

		cout<<"gray gray   8u: SSE4 normal:"<<getPSNR(destBaseGrayGray8u,destSSE4GrayGray8u)<<" dB"<<endl;
		cout<<"color color 8u: SSE4 normal:"<<getPSNR(destBaseColorColor8u,destSSE4ColorColor8u)<<" dB"<<endl;
		cout<<"gray color  8u: SSE4 normal:"<<getPSNR(destBaseGrayColor8u,destSSE4GrayColor8u)<<" dB"<<endl;
		cout<<"color gray  8u: SSE4 normal:"<<getPSNR(destBaseColorGray8u,destSSE4ColorGray8u)<<" dB"<<endl;

		cout<<"gray gray   32f: SSE4 normal:"<<getPSNR(destBaseGrayGray32f,destSSE4GrayGray32f)<<" dB"<<endl;
		cout<<"color color 32f: SSE4 normal:"<<getPSNR(destBaseColorColor32f,destSSE4ColorColor32f)<<" dB"<<endl;
		cout<<"gray color  32f: SSE4 normal:"<<getPSNR(destBaseGrayColor32f,destSSE4GrayColor32f)<<" dB"<<endl;
		cout<<"color gray  32f: SSE4 normal:"<<getPSNR(destBaseColorGray32f,destSSE4ColorGray32f)<<" dB"<<endl;


		cout<<"======================================="<<endl;
		cout<<"6. Trilateral Weighted Map test"<<endl;

		Mat wgg;
		Mat wcc;
		Mat wgc;
		Mat wcg;
		Mat wggf;
		Mat wccf;
		Mat wgcf;
		Mat wcgf;
		{
			CalcTime t("Base Trilateral gray gray  8u");
			cout<<format("s: %d g%d\n",src.channels(),guide.channels());
			trilateralWeightMapBase(src, guide,wgg,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base Trilateral color color  8u");
			cout<<format("s: %d g%d\n",guidec.channels(),guidec.channels());
			trilateralWeightMapBase(guidec, guidec,wcc,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base Trilateral gray color  8u");
			cout<<format("s: %d g%d\n",src.channels(),guidec.channels());
			trilateralWeightMapBase(src, guidec,wgc,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base Trilateral color gray  8u");
			cout<<format("s: %d g%d\n",guidec.channels(),src.channels());
			trilateralWeightMapBase(guidec, src,wcg,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base Trilateral gray gray  32f");
			cout<<format("s: %d g%d\n",srcf.channels(),guidef.channels());
			trilateralWeightMapBase(srcf, guidef,wggf,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base Trilateral color color  32f");
			cout<<format("s: %d g%d\n",guidecf.channels(),guidecf.channels());
			trilateralWeightMapBase(guidecf, guidecf,wccf,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base Trilateral gray color  32f");
			cout<<format("s: %d g%d\n",srcf.channels(),guidecf.channels());
			trilateralWeightMapBase(srcf, guidecf,wgcf,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base Trilateral color gray  32f");
			cout<<format("s: %d g%d\n",guidecf.channels(),srcf.channels());
			trilateralWeightMapBase(guidecf, srcf,wcgf,d,sc,sc2,ss,cv::BORDER_REPLICATE);
		}
		Mat ssewgg;
		Mat ssewcc;
		Mat ssewgc;
		Mat ssewcg;
		Mat ssewggf;
		Mat ssewccf;
		Mat ssewgcf;
		Mat ssewcgf;
		{
			CalcTime t("SSE Trilateral gray gray  8u");
			cout<<format("s: %d g%d\n",src.channels(),guide.channels());
			trilateralWeightMap(src, guide,ssewgg,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE Trilateral color color  8u");
			cout<<format("s: %d g%d\n",guidec.channels(),guidec.channels());
			trilateralWeightMap(guidec, guidec,ssewcc,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE Trilateral gray color  8u");
			cout<<format("s: %d g%d\n",src.channels(),guidec.channels());
			trilateralWeightMap(src, guidec,ssewgc,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE Trilateral color gray  8u");
			cout<<format("s: %d g%d\n",guidec.channels(),src.channels());
			trilateralWeightMap(guidec, src,ssewcg,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE Trilateral gray gray  32f");
			cout<<format("s: %d g%d\n",srcf.channels(),guidef.channels());
			trilateralWeightMap(srcf, guidef,ssewggf,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE Trilateral color color  32f");
			cout<<format("s: %d g%d\n",guidecf.channels(),guidecf.channels());
			trilateralWeightMap(guidecf, guidecf,ssewccf,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE Trilateral gray color  32f");
			cout<<format("s: %d g%d\n",srcf.channels(),guidecf.channels());
			trilateralWeightMap(srcf, guidecf,ssewgcf,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE Trilateral color gray  32f");
			cout<<format("s: %d g%d\n",guidecf.channels(),srcf.channels());
			trilateralWeightMap(guidecf, srcf,ssewcgf,Size(d,d),sc,sc2,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}

		cout<<"gray gray    8u: SSE4 normal:"<<getPSNR(wgg,ssewgg)<<" dB"<<endl;
		cout<<"color color  8u: SSE4 normal:"<<getPSNR(wcc,ssewcc)<<" dB"<<endl;
		cout<<"gray color   8u: SSE4 normal:"<<getPSNR(wgc,ssewgc)<<" dB"<<endl;
		cout<<"color gray   8u: SSE4 normal:"<<getPSNR(wcg,ssewcg)<<" dB"<<endl;
		cout<<"gray gray   32f: SSE4 normal:"<<getPSNR(wggf,ssewggf)<<" dB"<<endl;
		cout<<"color color 32f: SSE4 normal:"<<getPSNR(wccf,ssewccf)<<" dB"<<endl;
		cout<<"gray color  32f: SSE4 normal:"<<getPSNR(wgcf,ssewgcf)<<" dB"<<endl;
		cout<<"color gray  32f: SSE4 normal:"<<getPSNR(wcgf,ssewcgf)<<" dB"<<endl;

		//Mat wmapshow;
		//wcgf.convertTo(wmapshow,CV_8U,255.0/(d*d));
		//imshow("w1",wmapshow);
		//ssewcgf.convertTo(wmapshow,CV_8U,255.0/(d*d));
		//imshow("w2",wmapshow);




		Mat show;

		//destBaseGrayGray32f.convertTo(show,CV_8UC1);;imshow("opencv",show);
		//destSSE4GrayGray32f.convertTo(show,CV_8UC1);imshow("sse",show);
		//destBaseColorColor32f.convertTo(show,CV_8UC3);imshow("opencv",show);
		//destSSE4ColorColor32f.convertTo(show,CV_8UC3);imshow("sse",show);
		//destBaseGrayColor32f.convertTo(show,CV_8UC1);imshow("opencv",show);
		//destSSE4GrayColor32f.convertTo(show,CV_8UC1);imshow("sse",show);
		//destBaseColorGray32f.convertTo(show,CV_8UC3);imshow("opencv",show);
		//destSSE4ColorGray32f.convertTo(show,CV_8UC3);imshow("sse",show);

		Mat wmap;
		weight.convertTo(wmap,CV_8U,255.0/(d*d));
		imshow("w",wmap);
		//cout<<"SSE4:"<< getPSNR(dest2,dest)<<" dB"<<endl;;
		//cout<<"SSE4:"<< getPSNR(dest2,dest3)<<" dB"<<endl;;

		/*cout<<"src           :";eval(src);
		cout<<"fast        :";eval(dest);
		cout<<"normal      :";eval(dest2);*/
		//cout<<"weightedBlend :";eval(dest3);

		destBaseGrayGray8u.copyTo(dest);

		imshow(wname,dest);

		key = waitKey(1);
	}
}

void testJointBilateral()
{
	string wname = "joint bilateral";
	namedWindow(wname);

	Mat src = imread("dataset/img_stereo/dp.png",0);
	fillOcclusion(src, 0);
	//removeStreakingNoise(src_,src_,8);
	Mat srcg = imread("dataset/img_stereo/dp.png",0);
	//Mat guide = imread("dataset/img_stereo/dp.png",0);

	Mat guide = imread("dataset/img_stereo/teddy.png",0);
	Mat guidec = imread("dataset/img_stereo/teddy.png");

	imshow("src",src);imshow("guide",guide);

	Mat dest  = Mat::zeros(src.size(),src.type());
	Mat dest2 = Mat::zeros(src.size(),src.type());
	Mat dest3 = Mat::zeros(src.size(),src.type());

	int sw = 0;createTrackbar("sw",wname,&sw,2);
	int r = 7;createTrackbar("r",wname,&r,20);
	int sigC = 100;createTrackbar("c",wname,&sigC,1500);
	int sigS = 70;createTrackbar("s",wname,&sigS,200);

	int key=0;
	int count=0;

	Mat srcf;src.convertTo(srcf,CV_32F);
	Mat guidecf;guidec.convertTo(guidecf,CV_32F);
	Mat guidef;guide.convertTo(guidef,CV_32F);

	Mat destBaseGrayGray8u = Mat::zeros(src.size(),CV_8U);
	Mat destBaseGrayGray32f = Mat::zeros(src.size(),CV_32F);
	Mat destBaseColorColor8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destBaseColorColor32f = Mat::zeros(src.size(),CV_32FC3);
	Mat destBaseGrayColor8u = Mat::zeros(src.size(),CV_8U);
	Mat destBaseGrayColor32f = Mat::zeros(src.size(),CV_32F);
	Mat destBaseColorGray8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destBaseColorGray32f = Mat::zeros(src.size(),CV_32FC3);

	Mat destSSE4GrayGray8u = Mat::zeros(src.size(),CV_8U);
	Mat destSSE4GrayGray32f = Mat::zeros(src.size(),CV_32F);
	Mat destSSE4ColorColor8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destSSE4ColorColor32f = Mat::zeros(src.size(),CV_32FC3);
	Mat destSSE4GrayColor8u = Mat::zeros(src.size(),CV_8U);
	Mat destSSE4GrayColor32f = Mat::zeros(src.size(),CV_32F);
	Mat destSSE4ColorGray8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destSSE4ColorGray32f = Mat::zeros(src.size(),CV_32FC3);

	Mat destSSE4GrayGraySP8u = Mat::zeros(src.size(),CV_8U);
	Mat destSSE4GrayGraySP32f = Mat::zeros(src.size(),CV_32F);
	Mat destSSE4ColorColorSP8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destSSE4ColorColorSP32f = Mat::zeros(src.size(),CV_32FC3);
	Mat destSSE4GrayColorSP8u = Mat::zeros(src.size(),CV_8U);
	Mat destSSE4GrayColorSP32f = Mat::zeros(src.size(),CV_32F);
	Mat destSSE4ColorGraySP8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destSSE4ColorGraySP32f = Mat::zeros(src.size(),CV_32FC3);

	{
		double sc=sigC/10.0;
		double ss=sigS/10.0;
		int d = 2*r+1;
		cout<<"======================================="<<endl;
		cout<<"0. Joint Bilateral filter check"<<endl;
		{
			Mat dst1,dst2;
			jointBilateralFilterBase(src,src,dst1,d,sc,ss,cv::BORDER_REPLICATE);
			bilateralFilter(src,dst2,d,sc,ss,cv::BORDER_REPLICATE);
			cout<<"joint check gray8u  :"<<getPSNR(dst1,dst2)<<" dB"<<endl;
		}
		{
			Mat dst1,dst2;
			jointBilateralFilterBase(guidec,guidec,dst1,d,sc,ss,cv::BORDER_REPLICATE);
			bilateralFilter(guidec,dst2,d,sc,ss,cv::BORDER_REPLICATE);
			cout<<"joint check color8u :"<<getPSNR(dst1,dst2)<<" dB"<<endl;
		}
		{
			Mat dst1,dst2;
			jointBilateralFilterBase(srcf,srcf,dst1,d,sc,ss,cv::BORDER_REPLICATE);
			bilateralFilter(srcf,dst2,d,sc,ss,cv::BORDER_REPLICATE);
			cout<<"joint check gray32f :"<<getPSNR(dst1,dst2)<<" dB"<<endl;
		}
		{
			Mat dst1,dst2;
			jointBilateralFilterBase(guidecf,guidecf,dst1,d,sc,ss,cv::BORDER_REPLICATE);
			bilateralFilter(guidecf,dst2,d,sc,ss,cv::BORDER_REPLICATE);
			cout<<"joint check color32f:"<<getPSNR(dst1,dst2)<<" dB"<<endl;
		}
	}

	while(key!='q')
	{
		int d = 2*r+1;
		double sc=sigC/10.0;
		double ss=sigS/10.0;

		cout<<"======================================="<<endl;
		cout<<"1. Joint Bilateral filter Base test"<<endl;
		{
			CalcTime t("Base joint bilateral gray gray  8u");
			jointBilateralFilterBase(src,guide,destBaseGrayGray8u,d,sc,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base joint bilateral color color  8u");
			jointBilateralFilterBase(guidec,guidec,destBaseColorColor8u,d,sc,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base joint bilateral gray color  8u");
			jointBilateralFilterBase(src,guidec,destBaseGrayColor8u,d,sc,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base joint bilateral color gray 8u");
			jointBilateralFilterBase(guidec,guide,destBaseColorGray8u,d,sc,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base joint bilateral gray gray  32f");
			jointBilateralFilterBase(srcf,guidef,destBaseGrayGray32f,d,sc,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base joint bilateral color color  32f");
			jointBilateralFilterBase(guidecf,guidecf,destBaseColorColor32f,d,sc,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base joint bilateral gray color  32f");
			jointBilateralFilterBase(srcf,guidecf,destBaseGrayColor32f,d,sc,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base joint bilateral color gray  32f");
			jointBilateralFilterBase(guidecf,guidef,destBaseColorGray32f,d,sc,ss,cv::BORDER_REPLICATE);
		}

		cout<<"======================================="<<endl;
		cout<<"2. Joint Bilateral filter SSE test"<<endl;
		{
			CalcTime t("SSE4 joint bilateral gray gray  8u");
			jointBilateralFilter(src,guide,destSSE4GrayGray8u,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 joint bilateral color color  8u");
			jointBilateralFilter(guidec,guidec,destSSE4ColorColor8u,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 joint bilateral gray color  8u");
			jointBilateralFilter(src,guidec,destSSE4GrayColor8u,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 joint bilateral color gray 8u");
			jointBilateralFilter(guidec,guide,destSSE4ColorGray8u,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral gray gray  32f");
			jointBilateralFilter(srcf,guidef,destSSE4GrayGray32f,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral color color  32f");
			jointBilateralFilter(guidecf,guidecf,destSSE4ColorColor32f,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral gray color  32f");
			jointBilateralFilter(srcf,guidecf,destSSE4GrayColor32f,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral color gray  32f");
			jointBilateralFilter(guidecf,guidef,destSSE4ColorGray32f,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}

		cout<<"gray gray   8u: SSE4 normal:"<<getPSNR(destBaseGrayGray8u,destSSE4GrayGray8u)<<" dB"<<endl;
		cout<<"color color 8u: SSE4 normal:"<<getPSNR(destBaseColorColor8u,destSSE4ColorColor8u)<<" dB"<<endl;
		cout<<"gray color  8u: SSE4 normal:"<<getPSNR(destBaseGrayColor8u,destSSE4GrayColor8u)<<" dB"<<endl;
		cout<<"color gray  8u: SSE4 normal:"<<getPSNR(destBaseColorGray8u,destSSE4ColorGray8u)<<" dB"<<endl;

		cout<<"gray gray   32f: SSE4 normal:"<<getPSNR(destBaseGrayGray32f,destSSE4GrayGray32f)<<" dB"<<endl;
		cout<<"color color 32f: SSE4 normal:"<<getPSNR(destBaseColorColor32f,destSSE4ColorColor32f)<<" dB"<<endl;
		cout<<"gray color  32f: SSE4 normal:"<<getPSNR(destBaseGrayColor32f,destSSE4GrayColor32f)<<" dB"<<endl;
		cout<<"color gray  32f: SSE4 normal:"<<getPSNR(destBaseColorGray32f,destSSE4ColorGray32f)<<" dB"<<endl;


		cout<<"======================================="<<endl;
		cout<<"3. Joint Bilateral filter SSE SP test"<<endl;

		{
			CalcTime t("SSE4 joint bilateral gray gray  8u");

			jointBilateralFilter(src,guide,destSSE4GrayGray8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 joint bilateral color color  8u");
			jointBilateralFilter(guidec,guidec,destSSE4ColorColor8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 joint bilateral gray color  8u");
			jointBilateralFilter(src,guidec,destSSE4GrayColor8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 joint bilateral color gray 8u");
			jointBilateralFilter(guidec,guide,destSSE4ColorGray8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral gray gray  32f");
			jointBilateralFilter(srcf,guidef,destSSE4GrayGray32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral color color  32f");
			jointBilateralFilter(guidecf,guidecf,destSSE4ColorColor32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral gray color  32f");
			jointBilateralFilter(srcf,guidecf,destSSE4GrayColor32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral color gray  32f");
			jointBilateralFilter(guidecf,guidef,destSSE4ColorGray32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}

		cout<<"gray gray   8u: SSE4 SP :"<<getPSNR(destBaseGrayGray8u,destSSE4GrayGray8u)<<" dB"<<endl;
		cout<<"color color 8u: SSE4 SP :"<<getPSNR(destBaseColorColor8u,destSSE4ColorColor8u)<<" dB"<<endl;
		cout<<"gray color  8u: SSE4 SP :"<<getPSNR(destBaseGrayColor8u,destSSE4GrayColor8u)<<" dB"<<endl;
		cout<<"color gray  8u: SSE4 SP :"<<getPSNR(destBaseColorGray8u,destSSE4ColorGray8u)<<" dB"<<endl;

		cout<<"gray gray   32f: SSE4 SP:"<<getPSNR(destBaseGrayGray32f,destSSE4GrayGray32f)<<" dB"<<endl;
		cout<<"color color 32f: SSE4 SP:"<<getPSNR(destBaseColorColor32f,destSSE4ColorColor32f)<<" dB"<<endl;
		cout<<"gray color  32f: SSE4 SP:"<<getPSNR(destBaseGrayColor32f,destSSE4GrayColor32f)<<" dB"<<endl;
		cout<<"color gray  32f: SSE4 SP:"<<getPSNR(destBaseColorGray32f,destSSE4ColorGray32f)<<" dB"<<endl;


		cout<<"======================================="<<endl;
		cout<<"4. Weighted joint Bilateral filter SSE test"<<endl;
		Mat weight = Mat::ones(src.size(),CV_32F);
		{
			CalcTime t("SSE4 joint bilateral gray gray  8u");
			weightedJointBilateralFilter(src,weight,guide,destSSE4GrayGray8u,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 joint bilateral color color  8u");
			weightedJointBilateralFilter(guidec,weight,guidec,destSSE4ColorColor8u,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 joint bilateral gray color  8u");
			weightedJointBilateralFilter(src,weight,guidec,destSSE4GrayColor8u,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 joint bilateral color gray 8u");
			weightedJointBilateralFilter(guidec,weight,guide,destSSE4ColorGray8u,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral gray gray  32f");
			weightedJointBilateralFilter(srcf,weight,guidef,destSSE4GrayGray32f,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral color color  32f");
			weightedJointBilateralFilter(guidecf,weight,guidecf,destSSE4ColorColor32f,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral gray color  32f");
			weightedJointBilateralFilter(srcf,weight,guidecf,destSSE4GrayColor32f,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral color gray  32f");
			weightedJointBilateralFilter(guidecf,weight,guidef,destSSE4ColorGray32f,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}

		cout<<"gray gray   8u: SSE4 normal:"<<getPSNR(destBaseGrayGray8u,destSSE4GrayGray8u)<<" dB"<<endl;
		cout<<"color color 8u: SSE4 normal:"<<getPSNR(destBaseColorColor8u,destSSE4ColorColor8u)<<" dB"<<endl;
		cout<<"gray color  8u: SSE4 normal:"<<getPSNR(destBaseGrayColor8u,destSSE4GrayColor8u)<<" dB"<<endl;
		cout<<"color gray  8u: SSE4 normal:"<<getPSNR(destBaseColorGray8u,destSSE4ColorGray8u)<<" dB"<<endl;

		cout<<"gray gray   32f: SSE4 normal:"<<getPSNR(destBaseGrayGray32f,destSSE4GrayGray32f)<<" dB"<<endl;
		cout<<"color color 32f: SSE4 normal:"<<getPSNR(destBaseColorColor32f,destSSE4ColorColor32f)<<" dB"<<endl;
		cout<<"gray color  32f: SSE4 normal:"<<getPSNR(destBaseGrayColor32f,destSSE4GrayColor32f)<<" dB"<<endl;
		cout<<"color gray  32f: SSE4 normal:"<<getPSNR(destBaseColorGray32f,destSSE4ColorGray32f)<<" dB"<<endl;

		cout<<"======================================="<<endl;
		cout<<"5. Weighted joint Bilateral filter SSE SP test"<<endl;
		{
			CalcTime t("SSE4 joint bilateral gray gray  8u");
			weightedJointBilateralFilter(src,weight,guide,destSSE4GrayGray8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 joint bilateral color color  8u");
			weightedJointBilateralFilter(guidec,weight,guidec,destSSE4ColorColor8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 joint bilateral gray color  8u");
			weightedJointBilateralFilter(src,weight,guidec,destSSE4GrayColor8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}

		{
			CalcTime t("SSE4 joint bilateral color gray 8u");
			weightedJointBilateralFilter(guidec,weight,guide,destSSE4ColorGray8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral gray gray  32f");
			weightedJointBilateralFilter(srcf,weight,guidef,destSSE4GrayGray32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral color color  32f");
			weightedJointBilateralFilter(guidecf,weight,guidecf,destSSE4ColorColor32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral gray color  32f");
			weightedJointBilateralFilter(srcf,weight,guidecf,destSSE4GrayColor32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4 joint bilateral color gray  32f");
			weightedJointBilateralFilter(guidecf,weight,guidef,destSSE4ColorGray32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE,cv::BORDER_REPLICATE);
		}

		cout<<"gray gray   8u: SSE4 normal:"<<getPSNR(destBaseGrayGray8u,destSSE4GrayGray8u)<<" dB"<<endl;
		cout<<"color color 8u: SSE4 normal:"<<getPSNR(destBaseColorColor8u,destSSE4ColorColor8u)<<" dB"<<endl;
		cout<<"gray color  8u: SSE4 normal:"<<getPSNR(destBaseGrayColor8u,destSSE4GrayColor8u)<<" dB"<<endl;
		cout<<"color gray  8u: SSE4 normal:"<<getPSNR(destBaseColorGray8u,destSSE4ColorGray8u)<<" dB"<<endl;

		cout<<"gray gray   32f: SSE4 normal:"<<getPSNR(destBaseGrayGray32f,destSSE4GrayGray32f)<<" dB"<<endl;
		cout<<"color color 32f: SSE4 normal:"<<getPSNR(destBaseColorColor32f,destSSE4ColorColor32f)<<" dB"<<endl;
		cout<<"gray color  32f: SSE4 normal:"<<getPSNR(destBaseGrayColor32f,destSSE4GrayColor32f)<<" dB"<<endl;
		cout<<"color gray  32f: SSE4 normal:"<<getPSNR(destBaseColorGray32f,destSSE4ColorGray32f)<<" dB"<<endl;

		/*Mat show;
		destBaseColorColor32f.convertTo(show,CV_8UC3);
		imshow("opencv",show);
		destSSE4ColorColor32f.convertTo(show,CV_8UC3);
		imshow("sse",show);*/

		destBaseGrayGray8u.copyTo(dest);
		imshow(wname,dest);

		key = waitKey(1);
	}
}
void testBilateral()
{
	string wname = "bilateral";
	namedWindow(wname);

	Mat src = imread("dataset/img_stereo/dp.png",0);
	fillOcclusion(src, 0);
	//removeStreakingNoise(src_,src_,8);
	Mat srcg = imread("dataset/img_stereo/dp.png",0);
	//Mat guide = imread("dataset/img_stereo/dp.png",0);

	Mat guide = imread("dataset/img_stereo/teddy.png",0);
	Mat guidec = imread("dataset/img_stereo/teddy.png");

	imshow("src",src);imshow("guide",guide);

	Mat dest  = Mat::zeros(src.size(),src.type());
	Mat dest2 = Mat::zeros(src.size(),src.type());
	Mat dest3 = Mat::zeros(src.size(),src.type());

	//int sw = 0;createTrackbar("sw",wname,&sw,2);
	int r = 7;createTrackbar("r",wname,&r,20);
	int sigC = 100;createTrackbar("c",wname,&sigC,1500);
	int sigS = 70;createTrackbar("s",wname,&sigS,200);

	Mat srcf;src.convertTo(srcf,CV_32F);
	Mat guidecf;guidec.convertTo(guidecf,CV_32F);
	Mat destBaseGray8u = Mat::zeros(src.size(),CV_8U);
	Mat destBaseGray32f = Mat::zeros(src.size(),CV_32F);
	Mat destBaseColor8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destBaseColor32f = Mat::zeros(src.size(),CV_32FC3);

	Mat destSSE4Gray8u = Mat::zeros(src.size(),CV_8U);
	Mat destSSE4Gray32f = Mat::zeros(src.size(),CV_32F);
	Mat destSSE4Color8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destSSE4Color32f = Mat::zeros(src.size(),CV_32FC3);

	Mat destSSE4GraySP8u = Mat::zeros(src.size(),CV_8U);
	Mat destSSE4GraySP32f = Mat::zeros(src.size(),CV_32F);
	Mat destSSE4ColorSP8u = Mat::zeros(src.size(),CV_8UC3);
	Mat destSSE4ColorSP32f = Mat::zeros(src.size(),CV_32FC3);

	int key=0;
	while(key!='q')
	{
		int d = 2*r+1;
		double sc=sigC/10.0;
		double ss=sigS/10.0;

		cout<<"======================================="<<endl;
		cout<<"1. Bilateral filter Base test"<<endl;
		{
			CalcTime t("Base bilateral gray   8u");
			bilateralFilterBase(src,destBaseGray8u,d,sc,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base bilateral gray  32f");
			bilateralFilterBase(srcf,destBaseGray32f,d,sc,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base bilateral color  8u");
			bilateralFilterBase(guidec,destBaseColor8u,d,sc,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base bilateral color 32f");
			bilateralFilterBase(guidecf,destBaseColor32f,d,sc,ss,cv::BORDER_REPLICATE);
		}
		cout<<"======================================="<<endl;
		cout<<"2. Bilateral filter SSE test"<<endl;
		{
			CalcTime t("SSE4  bilateral gray   8u");
			bilateralFilter(src,destSSE4Gray8u,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4  bilateral gray  32f");
			bilateralFilter(srcf,destSSE4Gray32f,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4  bilateral color  8u");
			bilateralFilter(guidec,destSSE4Color8u,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4  bilateral color 32f");
			bilateralFilter(guidecf,destSSE4Color32f,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		cout<<"gray   8u: SSE4 normal:"<<getPSNR(destBaseGray8u,destSSE4Gray8u)<<" dB"<<endl;
		cout<<"gray  32f: SSE4 normal:"<<getPSNR(destBaseGray32f,destSSE4Gray32f)<<" dB"<<endl;
		cout<<"color  8u: SSE4 normal:"<<getPSNR(destBaseColor8u,destSSE4Color8u)<<" dB"<<endl;
		cout<<"color 32f: SSE4 normal:"<<getPSNR(destBaseColor32f,destSSE4Color32f)<<" dB"<<endl;

		cout<<"======================================="<<endl;
		cout<<"2'. Bilateral filter SSE Order2 test: exp(cx) -> 1.0 - 1/(c*c)*x*x"<<endl;
		{
			CalcTime t("SSE4  bilateral gray   8u");
			bilateralFilter(src,destSSE4Gray8u,Size(d,d),sc,ss,BILATERAL_ORDER2,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE4  bilateral color  8u");
			bilateralFilter(guidec,destSSE4Color8u,Size(d,d),sc,ss,BILATERAL_ORDER2,cv::BORDER_REPLICATE);
		}
		
		cout<<"gray   8u: SSE4 normal:"<<getPSNR(destBaseGray8u,destSSE4Gray8u)<<" dB"<<endl;
		//cout<<"gray  32f: SSE4 normal:"<<getPSNR(destBaseGray32f,destSSE4Gray32f)<<" dB"<<endl;
		cout<<"color  8u: SSE4 normal:"<<getPSNR(destBaseColor8u,destSSE4Color8u)<<" dB"<<endl;
		//cout<<"color 32f: SSE4 normal:"<<getPSNR(destBaseColor32f,destSSE4Color32f)<<" dB"<<endl;
		cout<<"32f function is not implimented yet\n";
		//Mat mask;compare(destBaseColor32f,destSSE4Color32f,mask,cv::CMP_EQ);imshow("mask",mask);

		cout<<"======================================="<<endl;
		cout<<"3. Bilateral filter SP test"<<endl;
		{
			CalcTime t("SSE4  bilateral gray  SP  8u");
			bilateralFilter(src,destSSE4GraySP8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE);
		}
		{
			CalcTime t("SSE4  bilateral gray  SP 32f");
			bilateralFilter(srcf,destSSE4GraySP32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE);
		}
		{
			CalcTime t("SSE4  bilateral color SP  8u");
			bilateralFilter(guidec,destSSE4ColorSP8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE);
		}
		{
			CalcTime t("SSE4  bilateral color SP 32f");
			bilateralFilter(guidecf,destSSE4ColorSP32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE);
		}
		cout<<"gray   8u: SSE4 normal:"<<getPSNR(destBaseGray8u,destSSE4GraySP8u)<<" dB"<<endl;
		cout<<"gray  32f: SSE4 normal:"<<getPSNR(destBaseGray32f,destSSE4GraySP32f)<<" dB"<<endl;
		cout<<"color  8u: SSE4 normal:"<<getPSNR(destBaseColor8u,destSSE4ColorSP8u)<<" dB"<<endl;
		cout<<"color 32f: SSE4 normal:"<<getPSNR(destBaseColor32f,destSSE4ColorSP32f)<<" dB"<<endl;

		cout<<"======================================="<<endl;
		cout<<"4. Weighted Bilateral filter test"<<endl;

		Mat weight = Mat::ones(src.size(),CV_32F);
		{
			CalcTime t("SSE4 Weighted bilateral gray   8u");
			weightedBilateralFilter(src,weight,destSSE4Gray8u,Size(d,d),sc,ss,BILATERAL_NORMAL);
		}
		{
			CalcTime t("SSE4 Weighted bilateral gray  32f");
			weightedBilateralFilter(srcf,weight,destSSE4Gray32f,Size(d,d),sc,ss,BILATERAL_NORMAL);
		}
		{
			CalcTime t("SSE4 Weighted bilateral color  8u");
			weightedBilateralFilter(guidec,weight,destSSE4Color8u,Size(d,d),sc,ss,BILATERAL_NORMAL);
		}
		{
			CalcTime t("SSE4 Weighted bilateral color 32f");
			weightedBilateralFilter(guidecf,weight,destSSE4Color32f,Size(d,d),sc,ss,BILATERAL_NORMAL);
		}
		cout<<"gray   8u: SSE4 normal:"<<getPSNR(destBaseGray8u,destSSE4Gray8u)<<" dB"<<endl;
		cout<<"gray  32f: SSE4 normal:"<<getPSNR(destBaseGray32f,destSSE4Gray32f)<<" dB"<<endl;
		cout<<"color  8u: SSE4 normal:"<<getPSNR(destBaseColor8u,destSSE4Color8u)<<" dB"<<endl;
		cout<<"color 32f: SSE4 normal:"<<getPSNR(destBaseColor32f,destSSE4Color32f)<<" dB"<<endl;

		cout<<"======================================="<<endl;
		cout<<"5. Weighted Bilateral filter test"<<endl;
		{
			CalcTime t("SSE4 Weighted bilateral gray  SP  8u");
			weightedBilateralFilter(src,weight,destSSE4GraySP8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE);
		}
		{
			CalcTime t("SSE4 Weighted bilateral gray  SP 32f");
			weightedBilateralFilter(srcf,weight,destSSE4GraySP32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE);
		}
		{
			CalcTime t("SSE4 Weighted bilateral color SP  8u");
			weightedBilateralFilter(guidec,weight,destSSE4ColorSP8u,Size(d,d),sc,ss,BILATERAL_SEPARABLE);
		}
		{
			CalcTime t("SSE4 Weighted bilateral color SP 32f");
			weightedBilateralFilter(guidecf,weight,destSSE4ColorSP32f,Size(d,d),sc,ss,BILATERAL_SEPARABLE);
		}
		cout<<"gray   8u: SSE4 normal:"<<getPSNR(destBaseGray8u,destSSE4GraySP8u)<<" dB"<<endl;
		cout<<"gray  32f: SSE4 normal:"<<getPSNR(destBaseGray32f,destSSE4GraySP32f)<<" dB"<<endl;
		cout<<"color  8u: SSE4 normal:"<<getPSNR(destBaseColor8u,destSSE4ColorSP8u)<<" dB"<<endl;
		cout<<"color 32f: SSE4 normal:"<<getPSNR(destBaseColor32f,destSSE4ColorSP32f)<<" dB"<<endl;

		cout<<"======================================="<<endl;
		cout<<"6. Bilateral Weight Map test"<<endl;
		Mat wgg;
		Mat wcc;
		Mat wggf;
		Mat wccf;
		{
			CalcTime t("Base Trilateral gray gray  8u");
			bilateralWeightMapBase(src, wgg,d,sc,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base Trilateral color color  8u");
			bilateralWeightMapBase(guidec, wcc,d,sc,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base Trilateral gray gray  32f");
			bilateralWeightMapBase(srcf, wggf,d,sc,ss,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("Base Trilateral color color  32f");
			bilateralWeightMapBase(guidecf, wccf,d,sc,ss,cv::BORDER_REPLICATE);
		}

		Mat ssewgg;
		Mat ssewcc;
		Mat ssewggf;
		Mat ssewccf;
		{
			CalcTime t("SSE bilateral gray   8u");
			bilateralWeightMap(src, ssewgg,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE bilateral color   8u");
			bilateralWeightMap(guidec, ssewcc,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE bilateral gray   32f");
			bilateralWeightMap(srcf, ssewggf,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}
		{
			CalcTime t("SSE bilateral color c  32f");
			bilateralWeightMap(guidecf, ssewccf,Size(d,d),sc,ss,BILATERAL_NORMAL,cv::BORDER_REPLICATE);
		}

		cout<<"gray   8u: SSE4 normal:"<<getPSNR(wgg,ssewgg)<<" dB"<<endl;
		cout<<"color  8u: SSE4 normal:"<<getPSNR(wcc,ssewcc)<<" dB"<<endl;
		cout<<"gray  32f: SSE4 normal:"<<getPSNR(wggf,ssewggf)<<" dB"<<endl;
		cout<<"color 32f: SSE4 normal:"<<getPSNR(wccf,ssewccf)<<" dB"<<endl;

		Mat wmap;
		ssewgg.convertTo(wmap,CV_8U,255.0/(d*d));
		imshow("w",wmap);

		Mat show;
		destBaseColor32f.convertTo(show,CV_8UC3);
		imshow(wname,show);
		key = waitKey(1);
	}
}
int main()
{
	//unit tests
	/*testBilateral();
	testJointBilateral();
	testTrilateral();
	jointNearestTest();*/

	testStereoRefinementEval();
	//testKinectRefinement();
}