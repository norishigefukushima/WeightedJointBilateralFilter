#include "filter.h"

		void blurRemoveMinMax(Mat& src, Mat& dest, const int r, const int threshold)
	{
		const Size ksize = Size(2*r+1,2*r+1);
		src.copyTo(dest);

		Mat xv;
		Mat nv;
		maxFilter(src,xv,ksize);
		minFilter(src,nv,ksize);

		uchar* n = nv.data;
		uchar* x = xv.data;
		uchar* s = src.data;
		uchar* d = dest.data;
		
		Mat mind;
		Mat maxd;
		Mat mask;
		absdiff(src,nv,mind);
		absdiff(src,xv,maxd);
		min(mind,maxd,mask);

		uchar* nd = mind.data;
		uchar* xd = maxd.data;
		uchar* mk = mask.data;
		
		for(int i=0;i<src.size().area();i++)
		{
			if(mk[i]>threshold)
			{
				if(nd[i]==mk[i])
				{
					d[i]=n[i];
				}
				else
				{
					d[i]=x[i];
				}
			}
		}
	}

		template<class T>
		static void maxFilter_sp(const Mat& src, Mat& dest,int width, const T maxval,int borderType)
		{
			if(src.channels()!=1)return;
			if(width==1){src.copyTo(dest);return;}
			if(dest.empty())dest=Mat::zeros(src.size(),src.type());

			Size size = src.size();

			Mat sim;
			int radiusx = width/2;
			copyMakeBorder( src, sim, 0, 0, radiusx, radiusx, borderType );

			const int st = width - 1;
	//#pragma omp parallel for
			for(int i = 0; i < size.height; i++ )
			{
				const T* sptr = sim.ptr<T>(i);
				T* dptr = dest.ptr<T>(i);
				
				T prev = maxval;
				for(int k = 0; k < width; k++ )
					prev = max(prev,sptr[+ k]);
				dptr[0] = prev;
				T ed = sptr[0];
				for(int j = 1; j < size.width; j++ )
				{
					if(prev<=sptr[j + st])
					{
						prev = sptr[j + st];
						dptr[j] = prev;	
					}
					else if(ed!=prev)
					{
						dptr[j] = prev;	
						ed = sptr[j];
					}
					else
					{
						T maxv=maxval;
						for(int k = 0; k < width; k++ )
						{
							maxv = max(maxv,sptr[j + k]);
						}
						dptr[j] = maxv;
						prev = maxv;
						ed = sptr[j];
					}
				}		
			}
		}
	template<class T>
	static void maxFilter_(const Mat& src, Mat& dest, Size ksize, T maxval,int borderType)
	{
		maxFilter_sp<T>(src,dest,ksize.width,maxval,borderType);
		Mat temp = dest.t();
		Mat temp2;
		maxFilter_sp<T>(temp,temp2,ksize.height,maxval,borderType);
		Mat(temp2.t()).copyTo(dest);
	}
	void maxFilter(const Mat& src, Mat& dest, Size ksize, int borderType)
	{
		if(src.type()==CV_8U)
		{
			maxFilter_<uchar>(src,dest,ksize,0,borderType);
		}
		if(src.type()==CV_16S)
		{
			maxFilter_<short>(src,dest,ksize,SHRT_MIN,borderType);
		}
		if(src.type()==CV_16U)
		{
			maxFilter_<ushort>(src,dest,ksize,0,borderType);
		}
		if(src.type()==CV_32F)
		{
			maxFilter_<float>(src,dest,ksize,FLT_MIN,borderType);
		}
	}
	
		template<class T>
		static void minFilter_sp(const Mat& src, Mat& dest,int width, const T maxval,int borderType)
		{
			if(src.channels()!=1)return;
			if(width==1){src.copyTo(dest);return;}
			if(dest.empty())dest=Mat::zeros(src.size(),src.type());

			Size size = src.size();

			Mat sim;
			int radiusx = width/2;
			copyMakeBorder( src, sim, 0, 0, radiusx, radiusx, borderType );

			const int st = width - 1;
			//#pragma omp parallel for
			for(int i = 0; i < size.height; i++ )
			{
				const T* sptr = sim.ptr<T>(i);
				T* dptr = dest.ptr<T>(i);
				
				T prev = maxval;
				for(int k = 0; k < width; k++ )
					prev = min(prev,sptr[+ k]);
				dptr[0] = prev;
				T ed = sptr[0];
				for(int j = 1; j < size.width; j++ )
				{
					if(prev>=sptr[j + st])
					{
						prev = sptr[j + st];
						dptr[j] = prev;	
					}
					else if(ed!=prev)
					{
						dptr[j] = prev;	
						ed = sptr[j];
					}
					else
					{
						T maxv=maxval;
						for(int k = 0; k < width; k++ )
						{
							maxv = min(maxv,sptr[j + k]);
						}
						dptr[j] = maxv;
						prev = maxv;
						ed = sptr[j];
					}
				}		
			}
		}
	template<class T>
	static void minFilter_(const Mat& src, Mat& dest, Size ksize, T maxval,int borderType)
	{
		minFilter_sp<T>(src,dest,ksize.width,maxval,borderType);
		Mat temp = dest.t();
		Mat temp2;
		minFilter_sp<T>(temp,temp2,ksize.height,maxval,borderType);
		Mat(temp2.t()).copyTo(dest);
	}
	void minFilter(const Mat& src, Mat& dest, Size ksize, int borderType)
	{
		if(src.type()==CV_8U)
		{
			minFilter_<uchar>(src,dest,ksize,255,borderType);
		}
		if(src.type()==CV_16S)
		{
			minFilter_<short>(src,dest,ksize,SHRT_MAX,borderType);
		}
		if(src.type()==CV_16U)
		{
			minFilter_<ushort>(src,dest,ksize,USHRT_MAX,borderType);
		}
		if(src.type()==CV_32F)
		{
			minFilter_<float>(src,dest,ksize,FLT_MAX,borderType);
		}
	}