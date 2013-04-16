#include "filter.h"

template <class T>
void jointColorDepthFillOcclusion_(const Mat& src, const Mat& guide, Mat& dest, const Size ksize, T threshold)
{
	if(dest.empty())dest.create(src.size(),src.type());
	Mat sim,gim;
	const int radiusw = ksize.width/2;
	const int radiush = ksize.height/2;;
	copyMakeBorder(src, sim, radiush, radiush, radiusw, radiusw,cv::BORDER_DEFAULT);
	copyMakeBorder(guide, gim, radiush, radiush, radiusw, radiusw,cv::BORDER_DEFAULT);

	vector<int> _space_ofs_before(ksize.area());
	int* space_ofs_before = &_space_ofs_before[0];

	int maxk=0;
	for(int i = -radiush; i <= radiush; i++ )
	{
		for(int j = -radiusw; j <= radiusw; j++ )
		{
			double r = std::sqrt((double)i*i + (double)j*j);
			if( r > radiusw )
				continue;

			space_ofs_before[maxk++] = (int)(i*sim.cols + j);
		}
	}
	const int steps = sim.cols;
	const int step = dest.cols;

	T* sptr = sim.ptr<T>(radiush);sptr+=radiusw;
	uchar* jptr = gim.ptr<uchar>(radiush);jptr+=radiusw;

	T* dst = dest.ptr<T>(0);

	T th2 = threshold*threshold;
	for(int i = 0; i < src.rows; i++ )
	{
		for(int j = 0; j < src.cols; j++ )
		{
			const T val0j = jptr[j];

			int minv=INT_MAX;
			T mind=0;
			if(sptr[j]==0)
			{
				for(int k = 0; k < maxk; k++ )
				{
					if(sptr[j + space_ofs_before[k]]==0) continue;

					const T valj = jptr[j + space_ofs_before[k]];
					int ab=(int)((valj-val0j)*(valj-val0j));
					if(ab<minv)
					{
						minv=ab;
						mind= sptr[j + space_ofs_before[k]];
					}
				}
				if(minv<th2)
				{
					dst[j]=mind;
				}
			}
		}
		sptr+=steps;
		jptr+=steps;
		dst+=step;
	}
}

void jointColorDepthFillOcclusion(const Mat& src, const Mat& guide, Mat& dest, const Size ksize, double threshold)
{
	{
		if(src.type()==CV_8U)
		{
			jointColorDepthFillOcclusion_<uchar>(src, guide, dest, ksize, (uchar)threshold);
		}
		else if(src.type()==CV_16S)
		{
			jointColorDepthFillOcclusion_<short>(src, guide, dest, ksize, (short)threshold);
		}
		else if(src.type()==CV_16U)
		{
			jointColorDepthFillOcclusion_<ushort>(src, guide, dest, ksize, (ushort)threshold);
		}
		else if(src.type()==CV_32F)
		{
			jointColorDepthFillOcclusion_<float>(src, guide, dest, ksize, (float)threshold);
		}
	}
}

template <class T>
static void fillOcclusionInv_(Mat& src, Mat& image, const T invalidvalue, const T minval, int threshold)
{
	int bb=1;
	const int MAX_LENGTH=(int)(src.cols*0.9);
	//#pragma omp parallel for
	for(int j=bb;j<src.rows-bb;j++)
	{
		T* s = src.ptr<T>(j);
		uchar* imap = image.ptr<uchar>(j);

		s[0]=minval;
		s[src.cols-1]=minval;
		for(int i=1;i<src.cols-1;i++)
		{
			if(s[i]<=invalidvalue)
			{
				int t=i;
				int emax=0;
				int argt;
				do
				{
					const int kstep = 2;
					int E = 
						max(max(
						abs(imap[3*(t-kstep)+0]-imap[3*(t+kstep)+0]),
						abs(imap[3*(t-kstep)+1]-imap[3*(t+kstep)+1])),
						abs(imap[3*(t-kstep)+2]-imap[3*(t+kstep)+2])
						);
					if(E>emax)
					{
						emax = E;
						argt = t;
					}
					t++;
					if(t>src.cols-2)break;
				}while(s[t]<=invalidvalue);

				if(emax>threshold && t-argt>2 && argt-i>2)
				{
					const T dl = s[i-1];
					const T	dr = s[t];
					if(t-i>MAX_LENGTH)
					{
						for(;i<t;i++)
						{
							s[i]=invalidvalue;
						}
					}
					else
					{
						//for(;i<t;i++)
						//{
						//	s[i]=invalidvalue;
						//}
						for(;i<argt;i++)
						{
							s[i]=dl;
						}
						for(;i<t;i++)
						{
							s[i]=dr;
						}
					}
				}
				else
				{
					const T dd = max(s[i-1],s[t]);
					if(t-i>MAX_LENGTH)
					{
						for(;i<t;i++)
						{
							s[i]=invalidvalue;
						}
					}
					else
					{
						for(;i<t;i++)
						{
							s[i]=dd;
						}
					}
				}
			}
		}
		s[0]=s[1];
		s[src.cols-1]=s[src.cols-2];
	}

	T* s1 = src.ptr<T>(0);
	T* s2 = src.ptr<T>(1);
	T* s3 = src.ptr<T>(src.rows-2);
	T* s4 = src.ptr<T>(src.rows-1);
	for(int i=0;i<src.cols;i++)
	{
		s1[i]=s2[i];
		s4[i]=s3[i];
	}
}
template <class T>
static void fillOcclusionInv_(Mat& src, const T invalidvalue, const T minval)
{
	int bb=1;
	const int MAX_LENGTH=(int)(src.cols*0.3);
	//#pragma omp parallel for
	for(int j=bb;j<src.rows-bb;j++)
	{
		T* s = src.ptr<T>(j);

		s[0]=minval;
		s[src.cols-1]=minval;
		for(int i=1;i<src.cols-1;i++)
		{
			if(s[i]<=invalidvalue)
			{
				int t=i;
				do
				{
					t++;
					if(t>src.cols-2)break;
				}while(s[t]<=invalidvalue);

				const T dd = max(s[i-1],s[t]);
				if(t-i>MAX_LENGTH)
				{
					for(;i<t;i++)
					{
						s[i]=invalidvalue;
					}
				}
				else
				{
					for(;i<t;i++)
					{
						s[i]=dd;
					}
				}
			}
		}
		s[0]=s[1];
		s[src.cols-1]=s[src.cols-2];
	}

	T* s1 = src.ptr<T>(0);
	T* s2 = src.ptr<T>(1);
	T* s3 = src.ptr<T>(src.rows-2);
	T* s4 = src.ptr<T>(src.rows-1);
	for(int i=0;i<src.cols;i++)
	{
		s1[i]=s2[i];
		s4[i]=s3[i];
	}
}
void fillOcclusionDepth(Mat& src, int invalidvalue)
{
	{
		if(src.type()==CV_8U)
		{
			fillOcclusionInv_<uchar>(src, (uchar)invalidvalue, 0);
		}
		else if(src.type()==CV_16S)
		{
			fillOcclusionInv_<short>(src, (short)invalidvalue, 0);
		}
		else if(src.type()==CV_16U)
		{
			fillOcclusionInv_<unsigned short>(src, (unsigned short)invalidvalue, 0);
		}
		else if(src.type()==CV_32F)
		{
			fillOcclusionInv_<float>(src, (float)invalidvalue,0);
		}
	}
}

void fillOcclusionDepth(Mat& depth, Mat& image, int invalidvalue, int threshold)
{
	if(depth.type()==CV_8U)
	{
		fillOcclusionInv_<uchar>(depth, image, (uchar)invalidvalue, 0, threshold);
	}
	else if(depth.type()==CV_16S)
	{
		fillOcclusionInv_<short>(depth, image, (short)invalidvalue, 0, threshold);
	}
	else if(depth.type()==CV_16U)
	{
		fillOcclusionInv_<unsigned short>(depth, image, (ushort)invalidvalue, 0, threshold);
	}
	else if(depth.type()==CV_32F)
	{
		fillOcclusionInv_<float>(depth, image, (float)invalidvalue, 0, threshold);
	}
}

template <class T>
static void fillOcclusion_(Mat& src, const T invalidvalue, const T maxval)
{
	int bb=0;
	const int MAX_LENGTH=(int)(src.cols*1.0-5);

	T* s = src.ptr<T>(0);
	const int step = src.cols;
	//Mat testim = Mat::zeros(src.size(),CV_8UC3);const int lineth = 30;
	//check right
	{
		T* s = src.ptr<T>(0);
		s+=(src.cols-1);
		const int step = src.cols;
		for(int j=0;j<src.rows;j++)
		{
			if(s[0]==invalidvalue)
				s[0] = s[-step];
			s+=step;
		}
	}

	//int table[500];
	const int leftmax=64;
	for(int j=0;j<src.rows;j++)
	{
		s[0]=maxval;//可能性のある最大値を入力
		s[src.cols-1]=maxval;//可能性のある最大値を入力
		//もし視差が0だったら値の入っている近傍のピクセル（エピポーラ線上）の最小値で埋める
		int i=1;
		// check left
		//if(j!=0)
		{
			for(;i<src.cols-1;i++)
			{
				if(s[i]<=invalidvalue)
				{
					int t=i;
					do
					{
						t++;
						if(t>leftmax)break;
					}while(s[t]<=invalidvalue);
					T dd = s[t];
					//table[j]=t;

					if(t>leftmax)
					{
						//for(int n=0;n<src.cols;n++)s[n]=invalidvalue;
						//memcpy(s,s-step,step*sizeof(T));
						i=1;break;	
					}
					else
					{
						double dsub=0.0;
						int n=1;
						for(;n<128;n++)
						{
							if( abs(s[t+n] - dd)> 1)break;
						}
						const int n1=n;
						T d2 = s[t+n];
						n++;
						for(;n<128;n++)
						{
							if( abs(s[t+n] - d2)> 0)break;
						}
						dsub = 2.0/(double)(n-n1);

						//dsub = 1.0/(double)(n1);


						//if(s[t+n]-s[t+n1]>0)dsub*=-1;

						//cout<<j<<": "<<dsub<<endl;
						for(;i<t+1;i++)
						{
							//s[i]=(0.33333333*s[i-step] + 0.66666667*(dd +dsub*(t-i))) ;
							s[i]=(T)(dd +dsub*(t-i)+0.5);
						}
					}
					break;
				}
			}
			//main
			for(;i<src.cols-1;i++)
			{
				if(s[i]<=invalidvalue)
				{
					if(s[i+1]>invalidvalue)
					{
						s[i]=min(s[i+1],s[i-1]);
						i++;continue;
					}

					int t=i;
					do
					{
						t++;
						if(t>src.cols-2)break;
					}while(s[t]<=invalidvalue);

					//if(t-i>lineth)line(testim,Point(i,j),Point(t,j),CV_RGB(255,0,0));

					T dd;
					//if(s[i-1]<=invalidvalue)dd=s[t];
					//else if(s[t]<=invalidvalue)dd=s[i-1];
					//else dd = min(s[i-1],s[t]);
					dd = min(s[i-1],s[t]);
					if(t-i>MAX_LENGTH)
					{
						//for(int n=0;n<src.cols;n++)s[n]=invalidvalue;
						memcpy(s,s-step,step*sizeof(T));
					}
					else
					{
						/*const int n=i-1;
						double dsub=0.0;
						if(abs(s[i-1]-s[t])<21)
						dsub=abs(s[n]-s[t])/(double)(t-n+1);

						if(s[i-1]==dd)
						{
						for(;i<t;i++)
						{
						s[i]=(dd + dsub*(i-n));
						}
						}
						else
						{
						for(;i<t;i++)
						{
						s[i]=(dd -dsub*(i-n));
						}
						}*/

						for(;i<t;i++)s[i]=dd;
					}
				}
			}
			s[0]=s[1];
			s[src.cols-1]=s[src.cols-2];
			s+=step;
		}
		//imshow("test",testim);
	}
}
void fillOcclusion(Mat& src, int invalidvalue)
{
	{
		if(src.type()==CV_8U)
		{
			fillOcclusion_<uchar>(src, (uchar)invalidvalue, UCHAR_MAX);
		}
		else if(src.type()==CV_16S)
		{
			fillOcclusion_<short>(src, (short)invalidvalue, SHRT_MAX);
		}
		else if(src.type()==CV_16U)
		{
			fillOcclusion_<unsigned short>(src, (unsigned short)invalidvalue, USHRT_MAX);
		}
		else if(src.type()==CV_32F)
		{
			fillOcclusion_<float>(src, (float)invalidvalue,FLT_MAX);
		}
	}
}

void removeStreakingNoise(Mat& src, Mat& dest, int th)
{
	Mat dst;
	src.copyTo(dst);
	int bb=1;
	int j;

	int t2 = th;
	const int step=2*src.cols;
	const int step2=3*src.cols;
	//#pragma omp parallel for
	if(src.type()==CV_8U)
	{
		for(j=bb;j<src.rows-bb;j++)
		{
			int i;
			uchar* d = dst.ptr(j);
			uchar* s = src.ptr(j-1);
			for(i=0;i<src.cols;i++)
			{
				if(abs(s[i]-s[step+i])<t2)
					d[i]=(s[i]+s[step+i])>>1;
			}
		}
	}
	if(src.type()==CV_16S)
	{
		const int istep = src.cols;
		short* d = dst.ptr<short>(bb);
		short* s = src.ptr<short>(bb-1);

		for(j=bb;j<src.rows-bb-1;j++)
		{
			int i;
			for(i=0;i<src.cols;i++)
			{
				if(abs(s[i]-s[step2+i])<t2)
				{
					short v=(s[i]+s[step2+i])>>1;
					if(abs(v-d[i])>t2)
					{
						d[i]=v;
						d[i+istep]=v;
					}
				}
				if(abs(s[i]-s[step+i])<t2)
				{
					short v=(s[i]+s[step+i])>>1;
					if(abs(v-d[i])>t2)
						d[i]=v;
				}
			}
			s+=istep;
			d+=istep;
		}
	}
		if(src.type()==CV_16U)
	{
		const int istep = src.cols;
		ushort* d = dst.ptr<ushort>(bb);
		ushort* s = src.ptr<ushort>(bb-1);

		for(j=bb;j<src.rows-bb-1;j++)
		{
			int i;
			for(i=0;i<src.cols;i++)
			{
				if(abs(s[i]-s[step2+i])<t2)
				{
					ushort v=(s[i]+s[step2+i])>>1;
					if(abs(v-d[i])>t2)
					{
						d[i]=v;
						d[i+istep]=v;
					}
				}
				if(abs(s[i]-s[step+i])<t2)
				{
					ushort v=(s[i]+s[step+i])>>1;
					if(abs(v-d[i])>t2)
						d[i]=v;
				}
			}
			s+=istep;
			d+=istep;
		}
	}

	dst.copyTo(dest);
}
void removeStreakingNoiseV(Mat& src, Mat& dest, int th)
{
	Mat dst;
	src.copyTo(dst);

	int bb=0;
	int j;

	int t2 = th;
	const int step=2*src.cols;
	const int step2=3*src.cols;
	//#pragma omp parallel for
	if(src.type()==CV_8U)
	{
		for(j=bb;j<src.rows-bb;j++)
		{
			int i;
			uchar* d = dst.ptr(j);
			uchar* s = src.ptr(j);
			for(i=1;i<src.cols-1;i++)
			{
				if(abs(s[i-1]-s[i+1])<t2)
					d[i]=(s[i-1]+s[i+1])>>1;
			}
		}
	}
	if(src.type()==CV_16S)
	{
		const int istep = src.cols;
		short* d = dst.ptr<short>(0);d++;
		short* s = src.ptr<short>(0);

		for(j=0;j<src.rows;j++)
		{
			int i;
			for(i=2;i<src.cols-2;i++)
			{
				if(abs(s[i]-s[3+i])<t2)
				{
					short v=(s[i]+s[3+i])>>1;
					if(abs(v-d[i])>t2)
					{
						d[i]=v;
						d[i+1]=v;
					}
				}
				if(abs(s[i]-s[2+i])<t2)
				{
					short v=(s[i]+s[2+i])>>1;
					if(abs(v-d[i])>t2)
						d[i]=v;
				}
			}
			s+=istep;
			d+=istep;
		}
	}
	if(src.type()==CV_16U)
	{
		const int istep = src.cols;
		ushort* d = dst.ptr<ushort>(0);d++;
		ushort* s = src.ptr<ushort>(0);

		for(j=0;j<src.rows;j++)
		{
			int i;
			for(i=2;i<src.cols-2;i++)
			{
				if(abs(s[i]-s[3+i])<t2)
				{
					ushort v=(s[i]+s[3+i])>>1;
					if(abs(v-d[i])>t2)
					{
						d[i]=v;
						d[i+1]=v;
					}
				}
				if(abs(s[i]-s[2+i])<t2)
				{
					ushort v=(s[i]+s[2+i])>>1;
					if(abs(v-d[i])>t2)
						d[i]=v;
				}
			}
			s+=istep;
			d+=istep;
		}
	}
	dst.copyTo(dest);
}