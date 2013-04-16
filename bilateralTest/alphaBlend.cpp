#include "util.h"

void alphaBlend(Mat& src1, Mat& src2, double alpha,Mat& dest)
{
	int T;
	Mat s1,s2;
	if(src1.channels()<=src2.channels())T=src2.type();
	else T=src1.type();
	if(dest.empty())dest=Mat::zeros(src1.size(),T);
	if(src1.channels()==src2.channels())
	{
		s1=src1;
		s2=src2;
	}
	else if(src2.channels()==3)
	{
		cvtColor(src1,s1,CV_GRAY2BGR);
		s2=src2;
	}
	else
	{
		cvtColor(src2,s2,CV_GRAY2BGR);
		s1=src1;
	}
	cv::addWeighted(s1,alpha,s2,1.0-alpha,0.0,dest);
}

void alphaBlend(Mat& src1, Mat& src2, Mat& alpha,Mat& dest)
{
	int T;
	Mat s1,s2;
	if(src1.channels()<=src2.channels())T=src2.type();
	else T=src1.type();
	if(dest.empty())dest=Mat::zeros(src1.size(),T);
	if(src1.channels()==src2.channels())
	{
		s1=src1;
		s2=src2;
	}
	else if(src2.channels()==3)
	{
		cvtColor(src1,s1,CV_GRAY2BGR);
		s2=src2;
	}
	else
	{
		cvtColor(src2,s2,CV_GRAY2BGR);
		s1=src1;
	}
	Mat a;
	if(alpha.type()==CV_8U)
		alpha.convertTo(a,CV_32F,1.0/255.0);
	else if(alpha.type()==CV_32F || alpha.type()==CV_64F)
		alpha.convertTo(a,CV_32F);

	if(dest.channels()==3)
	{
		vector<Mat> ss1,ss2;
		vector<Mat> ss1f(3),ss2f(3);
		split(s1,ss1);
		split(s2,ss2);	
		for(int c=0;c<3;c++)
		{
			ss1[c].convertTo(ss1f[c],CV_32F);
			ss2[c].convertTo(ss2f[c],CV_32F);
		}
		{
			float* s1r = ss1f[0].ptr<float>(0);
			float* s2r = ss2f[0].ptr<float>(0);

			float* s1g = ss1f[1].ptr<float>(0);
			float* s2g = ss2f[1].ptr<float>(0);

			float* s1b = ss1f[2].ptr<float>(0);
			float* s2b = ss2f[2].ptr<float>(0);


			float* al = a.ptr<float>(0);
			const int size = src1.size().area()/4;

			const __m128 ones = _mm_set1_ps(1.0f);

			for(int i=size;i--;)
			{
				const __m128 msa = _mm_load_ps(al);
				const __m128 imsa = _mm_sub_ps(ones,msa);
				__m128 ms1 = _mm_load_ps(s1r);
				__m128 ms2 = _mm_load_ps(s2r);
				ms1 = _mm_mul_ps(ms1,msa);
				ms2 = _mm_mul_ps(ms2,imsa);
				ms1 = _mm_add_ps(ms1,ms2);
				_mm_store_ps(s1r,ms1);//store ss1f

				ms1 = _mm_load_ps(s1g);
				ms2 = _mm_load_ps(s2g);
				ms1 = _mm_mul_ps(ms1,msa);
				ms2 = _mm_mul_ps(ms2,imsa);
				ms1 = _mm_add_ps(ms1,ms2);
				_mm_store_ps(s1g,ms1);//store ss1f

				ms1 = _mm_load_ps(s1b);
				ms2 = _mm_load_ps(s2b);
				ms1 = _mm_mul_ps(ms1,msa);
				ms2 = _mm_mul_ps(ms2,imsa);
				ms1 = _mm_add_ps(ms1,ms2);
				_mm_store_ps(s1b,ms1);//store ss1f

				al+=4,s1r+=4,s2r+=4,s1g+=4,s2g+=4,s1b+=4,s2b+=4;
			}
			for(int c=0;c<3;c++)
			{
				ss1f[c].convertTo(ss1[c],CV_8U);
			}
			merge(ss1,dest);
		}
	}
	else if(dest.channels()==1)
	{
		Mat ss1f,ss2f;
		s1.convertTo(ss1f,CV_32F);
		s2.convertTo(ss2f,CV_32F);
		{
			float* s1r = ss1f.ptr<float>(0);
			float* s2r = ss2f.ptr<float>(0);
			float* al = a.ptr<float>(0);
			const int size = src1.size().area()/4;
			const int nn = src1.size().area() - size*4;
			const __m128 ones = _mm_set1_ps(1.0f);
			for(int i=size;i--;)
			{
				const __m128 msa = _mm_load_ps(al);
				const __m128 imsa = _mm_sub_ps(ones,msa);
				__m128 ms1 = _mm_load_ps(s1r);
				__m128 ms2 = _mm_load_ps(s2r);
				ms1 = _mm_mul_ps(ms1,msa);
				ms2 = _mm_mul_ps(ms2,imsa);
				ms1 = _mm_add_ps(ms1,ms2);
				_mm_store_ps(s1r,ms1);//store ss1f

				al+=4,s1r+=4,s2r+=4;
			}
			for(int i=nn;i--;)
			{
				*s1r = *al * *s1r + (1.0f-*al)* *s2r;
				al++,s1r++,s2r++;
			}
			ss1f.convertTo(dest,CV_8U);
		}
	}
}

void alphaBlendSSE_8u(Mat& src1, Mat& src2, Mat& alpha, Mat& dest)
{
	if(dest.empty())dest.create(src1.size(),CV_8U);

	const int imsize = (src1.size().area()/16);
	uchar* s1 = src1.data;
	uchar* s2 = src2.data;
	uchar* a = alpha.data;
	uchar* d = dest.data;

	const __m128i zero = _mm_setzero_si128();
	const __m128i amax = _mm_set1_epi8(char(255));
	int i=0;
	if(s1==d)
	{
		for(;i<imsize;++i)
		{
			__m128i ms1h = _mm_load_si128((__m128i*)(s1));
			__m128i ms2h = _mm_load_si128((__m128i*)(s2));
			__m128i mah = _mm_load_si128((__m128i*)(a));
			__m128i imah = _mm_sub_epi8(amax,mah);

			__m128i ms1l = _mm_unpacklo_epi8(ms1h, zero);
			ms1h = _mm_unpackhi_epi8(ms1h, zero);

			__m128i ms2l = _mm_unpacklo_epi8(ms2h, zero);
			ms2h = _mm_unpackhi_epi8(ms2h, zero);

			__m128i mal = _mm_unpacklo_epi8(mah, zero);
			mah = _mm_unpackhi_epi8(mah, zero);

			__m128i imal = _mm_unpacklo_epi8(imah, zero);
			imah = _mm_unpackhi_epi8(imah, zero);

			ms1l = _mm_mullo_epi16(ms1l,mal);
			ms2l = _mm_mullo_epi16(ms2l,imal);
			ms1l = _mm_add_epi16(ms1l,ms2l);
			//ms1l = _mm_srli_epi16(ms1l,8);
			ms1l = _mm_srai_epi16(ms1l,8);

			ms1h = _mm_mullo_epi16(ms1h,mah);
			ms2h = _mm_mullo_epi16(ms2h,imah);
			ms1h = _mm_add_epi16(ms1h,ms2h);
			//ms1h = _mm_srli_epi16(ms1h,8);
			ms1h = _mm_srai_epi16(ms1h,8);

			_mm_stream_si128((__m128i*)s1,_mm_packs_epi16(ms1l,ms1h));

			s1+=16;
			s2+=16;
			a+=16;
		}
	}
	else
	{
		for(;i<imsize;++i)
		{
			__m128i ms1h = _mm_load_si128((__m128i*)(s1));
			__m128i ms2h = _mm_load_si128((__m128i*)(s2));
			__m128i mah = _mm_load_si128((__m128i*)(a));
			__m128i imah = _mm_sub_epi8(amax,mah);

			__m128i ms1l = _mm_unpacklo_epi8(ms1h, zero);
			ms1h = _mm_unpackhi_epi8(ms1h, zero);

			__m128i ms2l = _mm_unpacklo_epi8(ms2h, zero);
			ms2h = _mm_unpackhi_epi8(ms2h, zero);

			__m128i mal = _mm_unpacklo_epi8(mah, zero);
			mah = _mm_unpackhi_epi8(mah, zero);

			__m128i imal = _mm_unpacklo_epi8(imah, zero);
			imah = _mm_unpackhi_epi8(imah, zero);

			ms1l = _mm_mullo_epi16(ms1l,mal);
			ms2l = _mm_mullo_epi16(ms2l,imal);
			ms1l = _mm_add_epi16(ms1l,ms2l);
			//ms1l = _mm_srli_epi16(ms1l,8);
			ms1l = _mm_srai_epi16(ms1l,8);

			ms1h = _mm_mullo_epi16(ms1h,mah);
			ms2h = _mm_mullo_epi16(ms2h,imah);
			ms1h = _mm_add_epi16(ms1h,ms2h);
			//ms1h = _mm_srli_epi16(ms1h,8);
			ms1h = _mm_srai_epi16(ms1h,8);

			_mm_store_si128((__m128i*)d,_mm_packs_epi16(ms1l,ms1h));

			s1+=16;
			s2+=16;
			a+=16;
			d+=16;
		}
	}

	{
		uchar* s1 = src1.data;
		uchar* s2 = src2.data;
		uchar* a = alpha.data;
		uchar* d = dest.data;
		for(int n=i*16;n<src1.size().area();n++)
		{
			d[n] = (a[n]*s1[n] + (255-a[n])*s2[n])>>8;
		}
	}
}
static void alphablend1(Mat& src1, Mat& src2,Mat& alpha, Mat& dest)
{
	if(dest.empty())dest.create(src1.size(),CV_8U);
	const int imsize = (src1.size().area());
	uchar* s1 = src1.data;
	uchar* s2 = src2.data;
	uchar* a = alpha.data;
	uchar* d = dest.data;
	const double div = 1.0/255;
	for(int i=0;i<imsize;i++)
	{
		d[i]=(uchar)((a[i]*s1[i]+(255-a[i])*s2[i])*div + 0.5);
	}
}
static void alphablend2(Mat& src1, Mat& src2,Mat& alpha, Mat& dest)
{
	if(dest.empty())dest.create(src1.size(),CV_8U);
	const int imsize = (src1.size().area());
	uchar* s1 = src1.data;
	uchar* s2 = src2.data;
	uchar* a = alpha.data;
	uchar* d = dest.data;
	const double div = 1.0/255;
	for(int i=0;i<imsize;i++)
	{
		d[i]=(a[i]*s1[i]+(255-a[i])*s2[i])>>8;
	}
}
static void alphaBtest(Mat& src1, Mat& src2)
{
//	ConsoleImage ci(Size(640,480));
	namedWindow("alphaB");
	int a=0;
	createTrackbar("a","alphaB",&a,255);
	int key = 0;
	Mat alpha(src1.size(),CV_8U);
	Mat s1,s2;
	if(src1.channels()==3)cvtColor(src1,s1,CV_BGR2GRAY);
	else s1 = src1;
	if(src2.channels()==3)cvtColor(src2,s2,CV_BGR2GRAY);
	else s2 = src2;

	Mat dest;
	Mat destbf;
	Mat destshift;

	int iter = 50;
	createTrackbar("iter","alphaB",&iter,200);
	while(key!='q')
	{
//		ci.clear();
		alpha.setTo(a);
		{
			CalcTime t("alpha sse");
			for(int i=0;i<iter;i++)
				alphaBlendSSE_8u(s1,s2,alpha,dest);
	//		ci("SSE %f ms", t.getTime());

		}
		{
			CalcTime t("alpha bf");
			for(int i=0;i<iter;i++)
				alphablend1(s1,s2,alpha,destbf);
	//		ci("BF %f ms", t.getTime());
		}
		{
			CalcTime t("alpha shift");
			for(int i=0;i<iter;i++)
				alphablend2(s1,s2,alpha,destshift);
			//alphaBlend(s1,s2,alpha,destshift);
	//		ci("SHIFT %f ms", t.getTime());
		}
//		ci("bf->sse:   %f dB",calcPSNR(dest,destbf));
//		ci("bf->shift  %f dB",calcPSNR(destshift,destbf));
	//	ci("shift->sse %f dB",calcPSNR(destshift,dest));
//		imshow("console",ci.show);
		imshow("alphaB",destbf);
		key = waitKey(1);
	}
}
void xcvAddWeighted(IplImage* src1, double alpha, IplImage* src2, double beta, double gamma, IplImage* dest, IplImage* mask)
{
	int a = cvRound(alpha*1024);
	int b = cvRound(beta*1024);
	int c = cvRound(gamma*1024);


	if(mask==NULL)
	{
		if(c==0)
		{
			//#pragma omp parallel for
			for(int j=0;j<src1->height;j++)
			{
				unsigned char* ss1=(unsigned char*)src1->imageData+src1->widthStep*j;
				unsigned char* ss2=(unsigned char*)src2->imageData+src2->widthStep*j;
				unsigned char* dd=(unsigned char*)dest->imageData+dest->widthStep*j;

				for(int i=0;i<src1->width;i++)
				{
					int v = a*(ss1[3*i]) + b*(ss2[3*i]);
					dd[3*i]=(unsigned char)(v>>10);
					v = a*(ss1[3*i+1]) + b*(ss2[3*i+1]);
					dd[3*i+1]=(unsigned char)(v>>10);
					v = a*(ss1[3*i+2]) + b*(ss2[3*i+2]);
					dd[3*i+2]=(unsigned char)(v>>10);
				}
			}	
		}
		else
		{
			//#pragma omp parallel for
			for(int j=0;j<src1->height;j++)
			{
				unsigned char* ss1=(unsigned char*)src1->imageData+src1->widthStep*j;
				unsigned char* ss2=(unsigned char*)src2->imageData+src2->widthStep*j;
				unsigned char* dd=(unsigned char*)dest->imageData+dest->widthStep*j;

				for(int i=0;i<src1->width;i++)
				{
					int v = a*(ss1[3*i]) + b*(ss2[3*i])+c;
					v=(v>>10);
					v=(v>255)?255:v;
					v=(v<0)?0:v;
					dd[3*i]=(unsigned char)v;

					v = a*(ss1[3*i+1]) + b*(ss2[3*i+1])+c;
					v=(v>>10);
					v=(v>255)?255:v;
					v=(v<0)?0:v;
					dd[3*i+1]=(unsigned char)v;

					v = a*(ss1[3*i+2]) + b*(ss2[3*i+2])+c;
					v=(v>>10);
					v=(v>255)?255:v;
					v=(v<0)?0:v;
					dd[3*i+2]=(unsigned char)v;
				}
			}	
		}
	}
	else
	{
		if(c==0)
		{
			//#pragma omp parallel for
			for(int j=0;j<src1->height;j++)
			{
				unsigned char* ss1=(unsigned char*)src1->imageData+src1->widthStep*j;
				unsigned char* ss2=(unsigned char*)src2->imageData+src2->widthStep*j;
				unsigned char* dd=(unsigned char*)dest->imageData+dest->widthStep*j;
				unsigned char* mm=(unsigned char*)mask->imageData+mask->widthStep*j;;

				for(int i=0;i<src1->width;i++)
				{
					if(mm[i]!=0)
					{
						int v = a*(ss1[3*i]) + b*(ss2[3*i]);
						dd[3*i]=(unsigned char)(v>>10);
						v = a*(ss1[3*i+1]) + b*(ss2[3*i+1]);
						dd[3*i+1]=(unsigned char)(v>>10);
						v = a*(ss1[3*i+2]) + b*(ss2[3*i+2]);
						dd[3*i+2]=(unsigned char)(v>>10);
					}
				}
			}	
		}
		else
		{
			//#pragma omp parallel for
			for(int j=0;j<src1->height;j++)
			{
				unsigned char* ss1=(unsigned char*)src1->imageData+src1->widthStep*j;
				unsigned char* ss2=(unsigned char*)src2->imageData+src2->widthStep*j;
				unsigned char* dd=(unsigned char*)dest->imageData+dest->widthStep*j;
				unsigned char* mm=(unsigned char*)mask->imageData+mask->widthStep*j;;

				for(int i=0;i<src1->width;i++)
				{
					if(mm[i]!=0)
					{
						int v = a*(ss1[3*i]) + b*(ss2[3*i])+c;
						v=(v>>10);
						v=(v>255)?255:v;
						v=(v<0)?0:v;
						dd[3*i]=(unsigned char)v;

						v = a*(ss1[3*i+1]) + b*(ss2[3*i+1])+c;
						v=(v>>10);
						v=(v>255)?255:v;
						v=(v<0)?0:v;
						dd[3*i+1]=(unsigned char)v;

						v = a*(ss1[3*i+2]) + b*(ss2[3*i+2])+c;
						v=(v>>10);
						v=(v>255)?255:v;
						v=(v<0)?0:v;
						dd[3*i+2]=(unsigned char)v;
					}
				}
			}	
		}
	}
}
void xcvGUIAlphaBlend(IplImage* _image, IplImage* _image2, IplImage* mask)
{
	cv::Ptr<IplImage> mask_;
	if(mask==NULL)
	{
		mask_ = cvCreateImage(cvGetSize(_image),8,1);
		cvSet(mask_,cvScalarAll(255));
	}
	else
	{
		mask_ = cvCreateImage(cvGetSize(_image),8,1);
		cvCopy(mask,mask_);
	}

	IplImage* _depth = _image2;
	char* winName = "Image alpha blend";
	IplImage* depth = cvCreateImage(cvGetSize(_image),8,3);
	IplImage* image = cvCreateImage(cvGetSize(_image),8,3);
	if(_depth->nChannels==1)
	{
		cvCvtColor(_depth,depth,CV_GRAY2BGR);
	}
	else
	{
		cvCopy(_depth,depth);
	}
	if(_image->nChannels==1)
	{
		cvCvtColor(_image,image,CV_GRAY2BGR);
	}
	else
	{
		cvCopy(_image,image);
	}
	cvNamedWindow(winName,CV_WINDOW_AUTOSIZE);

	int x = 50;
	cvCreateTrackbar("alpha",winName,&x,100,NULL);

	IplImage* render = cvCloneImage(image);

	int key = 0;
	bool flag = false;
	while(key!='q')
	{
		cvZero(render);
		xcvAddWeighted(image,1.0-x/100.0,depth,x/100.0,0,render,mask_);

		if(key =='?')
		{
			cout<<"*** Help message ***"<<endl;
			cout<<"f: "<<"Flip input image1 and 2"<<endl;
			cout<<"p: "<<"Print PSNR"<<endl;
			cout<<"s: "<<"Save blending image (blend.png)"<<endl;
			cout<<"q: "<<"Quit"<<endl;
			cout<<"********************"<<endl;
		}
		if(key =='p')
		{
			//cout<<"PSNR (Y): "<<xcvCalcPSNR(image,depth,0,CV_BGR2YUV)<<" [dB]"<<endl;
		}
		if(key =='s')
		{
			cvSaveImage("blend.png",render);
		}
		if(key =='f')
		{
			if(flag)
			{
				x=0;
				cvSetTrackbarPos("alpha",winName,x);
			}
			else
			{
				x=100;
				cvSetTrackbarPos("alpha",winName,x);
			}
			flag = (flag)?false:true;
		}
		/*if(key ==XCV_KEY_ARROW_LEFT)
		{
		x--;
		cvSetTrackbarPos("alpha",winName,x);
		}
		if(key ==XCV_KEY_ARROW_RIGHT)
		{
		x++;
		cvSetTrackbarPos("alpha",winName,x);
		}*/
		cvShowImage(winName,render);
		key = cvWaitKey(33);
	}

	cvReleaseImage(&render);
	cvReleaseImage(&depth);
	cvReleaseImage(&image);
	cvDestroyWindow(winName);
}
void guiAlphaBlend(cv::Mat& image1, cv::Mat& image2,cv::Mat& mask)
{
	if (mask.empty())
	{

		if(image1.type()==CV_16S&&image2.type()==CV_16S)
		{
			Mat a,b;
			image1.convertTo(a,CV_8U);
			image2.convertTo(b,CV_8U);
			xcvGUIAlphaBlend(&IplImage(a),&IplImage(b),NULL);		
		}
		else
		{
			xcvGUIAlphaBlend(&IplImage(image1),&IplImage(image2),NULL);	
		}
	}
	else
	{
		if(image1.type()==CV_16S&&image2.type()==CV_16S)
		{
			Mat a,b;
			image1.convertTo(a,CV_8U);
			image2.convertTo(b,CV_8U);
			xcvGUIAlphaBlend(&IplImage(a),&IplImage(b),&IplImage(mask));		
		}
		else
		{
			xcvGUIAlphaBlend(&IplImage(image1),&IplImage(image2),&IplImage(mask));		

		}

	}
}