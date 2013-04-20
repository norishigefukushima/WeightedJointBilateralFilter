#include "viewSynthesis.h"
#include "filter.h"
#include "util.h"

static void getZeroMask(Mat& src, Mat& mask)
{
	const int size = src.size().area();
	if(mask.empty())
		mask = Mat::zeros(src.size(),src.type());
	else
		mask.setTo(0);
	uchar* s = src.data;
	uchar* m = mask.data;
	for(int i=0;i<size;i++)
	{
		if (s[0] ==0 &&s[1]==0&&s[2]==0)
		{
			m[0]=255;
		}
		m++,s+=3;
	}
}

template <class T>
static void shiftImInvNN_(Mat& srcim, Mat& srcdisp, Mat& destim, double amp, Mat& mask, int invalid = 0)
{		
	if(amp>0)
	{
		//#pragma omp parallel for
		for(int j=0;j<srcdisp.rows;j++)
		{
			uchar* sim = srcim.ptr<uchar>(j);
			uchar* dim = destim.ptr<uchar>(j);
			uchar* m = mask.ptr<uchar>(j);
			T* s = srcdisp.ptr<T>(j);

			for(int i=srcdisp.cols-1;i>=0;i--)
			{
				if(m[i]!=255)continue;

				const T disp = s[i];
				if(disp ==invalid)
				{
					dim[3*(i)+0]=0;
					dim[3*(i)+1]=0;
					dim[3*(i)+2]=0;
					continue;
				}

				const int dest = (int)(disp*amp);
				if(i-dest-1>=0 &&i-dest-1<srcdisp.cols-1)
				{
					dim[3*(i)+0]=sim[3*(i-dest)+0];
					dim[3*(i)+1]=sim[3*(i-dest)+1];
					dim[3*(i)+2]=sim[3*(i-dest)+2];		
				}
			}
		}
	}
	else if(amp<0)
	{

		//#pragma omp parallel for
		for(int j=0;j<srcdisp.rows;j++)
		{
			uchar* sim = srcim.ptr<uchar>(j);
			uchar* dim = destim.ptr<uchar>(j);
			uchar* m = mask.ptr<uchar>(j);

			T* s = srcdisp.ptr<T>(j);
			for(int i=0;i<srcdisp.cols;i++)
			{
				if(m[i]!=255)continue;
				const T disp = s[i];
				if(disp ==invalid)
				{
					dim[3*(i)+0]=0;
					dim[3*(i)+1]=0;
					dim[3*(i)+2]=0;
					continue;
				}

				const int dest = (int)(-amp*disp);//符号
				if(i+dest+1>=0 &&i+dest+1<srcdisp.cols-1)
				{
					dim[3*(i)+0]=sim[3*(i+dest)+0];
					dim[3*(i)+1]=sim[3*(i+dest)+1];
					dim[3*(i)+2]=sim[3*(i+dest)+2];
				}
			}
		}
	}
	else
	{
		//Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask
		srcim.copyTo(destim);
		mask.setTo(Scalar(255));
	}	
}
template <class T>
static void shiftImInvCubic_(Mat& srcim, Mat& srcdisp, Mat& destim, double amp, Mat& mask, int invalid = 0)
{		
	const double cubic = -1.0;
	const double c1 = cubic;
	const double c2 = -5.0*cubic;
	const double c3 = 8.0*cubic;
	const double c4 = -4.0*cubic;
	const double c5 = 2.0+cubic;
	const double c6 = -(cubic+3.0);
	if(amp>0)
	{
		//#pragma omp parallel for
		for(int j=0;j<srcdisp.rows;j++)
		{
			uchar* sim = srcim.ptr<uchar>(j);
			uchar* dim = destim.ptr<uchar>(j);
			uchar* m = mask.ptr<uchar>(j);
			T* s = srcdisp.ptr<T>(j);

			for(int i=srcdisp.cols-1;i>=0;i--)
			{
				if(m[i]!=255)continue;

				const T disp = s[i];
				if(disp ==invalid)
				{
					dim[3*(i)+0]=0;
					dim[3*(i)+1]=0;
					dim[3*(i)+2]=0;
					continue;
				}
				const int dest = (int)(disp*amp);
				const double ia = (disp*amp)-dest;
				const double a = 1.0-ia;
				if(i-dest-1>=0 &&i-dest-1<srcdisp.cols-1)
				{	
					if(ia==0.0)
					{
						dim[3*(i)+0]=sim[3*(i-dest)+0];
						dim[3*(i)+1]=sim[3*(i-dest)+1];
						dim[3*(i)+2]=sim[3*(i-dest)+2];	
					}
					else 
					{	
						const double viaa= c1*(1.0+ a)*(1.0+ a)*(1.0+ a) + c2*(1.0+ a)*(1.0+ a) +c3*(1.0+ a) + c4;
						const double iaa = c5* a* a* a + c6* a* a + 1.0;
						const double aa  = c5*ia*ia*ia + c6*ia*ia + 1.0;
						const double vaa = c1*(1.0+ ia)*(1.0+ ia)*(1.0+ ia) + c2*(1.0+ ia)*(1.0+ ia) +c3*(1.0+ ia) + c4;
						//cout<<format("%f %f : %f %f %f %f\n",a ,ia, vaa,aa,iaa,viaa);
						/*	const double viaa = 0.0;
						const double iaa = ia;
						const double aa = a;
						const double vaa = 0.0;*/

						dim[3*(i)+0]=saturate_cast<uchar>(vaa*sim[3*(i-dest+1)+0] + aa*sim[3*(i-dest)+0]+iaa*sim[3*(i-dest-1)+0]+viaa*sim[3*(i-dest-2)+0]);
						dim[3*(i)+1]=saturate_cast<uchar>(vaa*sim[3*(i-dest+1)+1] + aa*sim[3*(i-dest)+1]+iaa*sim[3*(i-dest-1)+1]+viaa*sim[3*(i-dest-2)+1]);
						dim[3*(i)+2]=saturate_cast<uchar>(vaa*sim[3*(i-dest+1)+2] + aa*sim[3*(i-dest)+2]+iaa*sim[3*(i-dest-1)+2]+viaa*sim[3*(i-dest-2)+2]);
					}			
				}
			}
		}
	}
	else if(amp<0)
	{

		//#pragma omp parallel for
		for(int j=0;j<srcdisp.rows;j++)
		{
			uchar* sim = srcim.ptr<uchar>(j);
			uchar* dim = destim.ptr<uchar>(j);
			uchar* m = mask.ptr<uchar>(j);

			T* s = srcdisp.ptr<T>(j);
			for(int i=0;i<srcdisp.cols;i++)
			{
				if(m[i]!=255)continue;
				const T disp = s[i];
				if(disp ==invalid)
				{
					dim[3*(i)+0]=0;
					dim[3*(i)+1]=0;
					dim[3*(i)+2]=0;
					continue;
				}

				const int dest = (int)(-amp*disp);//符号
				const double ia = (-amp*disp)-dest;
				const double a = 1.0-ia;

				if(i+dest+1>=0 &&i+dest+1<srcdisp.cols-1)
				{
					if(ia==0.0)
					{
						dim[3*(i)+0]=sim[3*(i+dest)+0];
						dim[3*(i)+1]=sim[3*(i+dest)+1];
						dim[3*(i)+2]=sim[3*(i+dest)+2];	
					}
					else 
					{
						const double viaa= c1*(1.0+ a)*(1.0+ a)*(1.0+ a) + c2*(1.0+ a)*(1.0+ a) +c3*(1.0+ a) + c4;
						const double iaa = c5* a* a* a + c6* a* a + 1.0;
						const double aa  = c5*ia*ia*ia + c6*ia*ia + 1.0;
						const double vaa = c1*(1.0+ ia)*(1.0+ ia)*(1.0+ ia) + c2*(1.0+ ia)*(1.0+ ia) +c3*(1.0+ ia) + c4;
						//cout<<format("%f %f : %f %f %f %f\n",a ,ia, vaa,aa,iaa,viaa);
						/*const double viaa = 0.0;
						const double iaa = ia;
						const double aa = a;
						const double vaa = 0.0;*/

						dim[3*(i)+0]=saturate_cast<uchar>(vaa*sim[3*(i+dest-1)+0] + aa*sim[3*(i+dest)+0]+iaa*sim[3*(i+dest+1)+0]+viaa*sim[3*(i+dest+2)+0]);
						dim[3*(i)+1]=saturate_cast<uchar>(vaa*sim[3*(i+dest-1)+1] + aa*sim[3*(i+dest)+1]+iaa*sim[3*(i+dest+1)+1]+viaa*sim[3*(i+dest+2)+1]);
						dim[3*(i)+2]=saturate_cast<uchar>(vaa*sim[3*(i+dest-1)+2] + aa*sim[3*(i+dest)+2]+iaa*sim[3*(i+dest+1)+2]+viaa*sim[3*(i+dest+2)+2]);

					}			
				}
			}
		}
	}
	else
	{
		//Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask
		srcim.copyTo(destim);
		mask.setTo(Scalar(255));
	}	
}
template <class T>
static void shiftImInvLinear_(Mat& srcim, Mat& srcdisp, Mat& destim, double amp, Mat& mask, int invalid = 0)
{		
	if(amp>0)
	{
		//#pragma omp parallel for
		for(int j=0;j<srcdisp.rows;j++)
		{
			uchar* sim = srcim.ptr<uchar>(j);
			uchar* dim = destim.ptr<uchar>(j);
			uchar* m = mask.ptr<uchar>(j);
			T* s = srcdisp.ptr<T>(j);

			for(int i=srcdisp.cols-1;i>=0;i--)
			{
				if(m[i]!=255)continue;

				const T disp = s[i];
				if(disp ==invalid)
				{
					dim[3*(i)+0]=0;
					dim[3*(i)+1]=0;
					dim[3*(i)+2]=0;
					continue;
				}

				const int dest = (int)(disp*amp);
				const double ia = (disp*amp)-dest;
				const double a = 1.0-ia;

				if(i-dest-1>=0 &&i-dest-1<srcdisp.cols-1)
				{
					dim[3*(i)+0]=saturate_cast<uchar>(a*sim[3*(i-dest)+0]+ia*sim[3*(i-dest-1)+0]);
					dim[3*(i)+1]=saturate_cast<uchar>(a*sim[3*(i-dest)+1]+ia*sim[3*(i-dest-1)+1]);
					dim[3*(i)+2]=saturate_cast<uchar>(a*sim[3*(i-dest)+2]+ia*sim[3*(i-dest-1)+2]);		
				}
			}
		}
	}
	else if(amp<0)
	{

		//#pragma omp parallel for
		for(int j=0;j<srcdisp.rows;j++)
		{
			uchar* sim = srcim.ptr<uchar>(j);
			uchar* dim = destim.ptr<uchar>(j);
			uchar* m = mask.ptr<uchar>(j);

			T* s = srcdisp.ptr<T>(j);
			for(int i=0;i<srcdisp.cols;i++)
			{
				if(m[i]!=255)continue;
				const T disp = s[i];
				if(disp ==invalid)
				{
					dim[3*(i)+0]=0;
					dim[3*(i)+1]=0;
					dim[3*(i)+2]=0;
					continue;
				}

				const int dest = (int)(-amp*disp);//符号
				const double ia = (-amp*disp)-dest;
				const double a = 1.0-ia;

				if(i+dest+1>=0 &&i+dest+1<srcdisp.cols-1)
				{
					dim[3*(i)+0]=saturate_cast<uchar>(a*sim[3*(i+dest)+0]+ia*sim[3*(i+dest+1)+0]);
					dim[3*(i)+1]=saturate_cast<uchar>(a*sim[3*(i+dest)+1]+ia*sim[3*(i+dest+1)+1]);
					dim[3*(i)+2]=saturate_cast<uchar>(a*sim[3*(i+dest)+2]+ia*sim[3*(i+dest+1)+2]);
				}
			}
		}
	}
	else
	{
		//Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask
		srcim.copyTo(destim);
		mask.setTo(Scalar(255));
	}	
}

enum
{
	WARP_NN = 0,
	WARP_LINEAR = 1,
	WARP_CUBIC = 2
};
template <class T>
static void shiftImInv_(Mat& srcim, Mat& srcdisp, Mat& destim, double amp, Mat& mask, int invalid = 0, int inter_method=WARP_LINEAR)
{
	if(inter_method==WARP_CUBIC)
	{
		shiftImInvCubic_<T>(srcim, srcdisp, destim, amp, mask,invalid);
	}
	else if(inter_method==WARP_LINEAR)
		shiftImInvLinear_<T>(srcim, srcdisp, destim, amp, mask,invalid);
	else if(inter_method==WARP_NN)
		shiftImInvNN_<T>(srcim, srcdisp, destim, amp, mask,invalid);
}

template <class T>
static void shiftImDispNN_(Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask, const int large_jump, const int sub_gap)
{
	int ij=0;
	const int ljump = large_jump*sub_gap;
	//const int iamp=cvRound(amp);
	if(amp>0)
	{
		//#pragma omp parallel for
		for(int j=0;j<srcdisp.rows;j++)
		{
			uchar* sim = srcim.ptr<uchar>(j);
			uchar* dim = destim.ptr<uchar>(j);
			uchar* m = mask.ptr<uchar>(j);
			T* s = srcdisp.ptr<T>(j);
			T* d = destdisp.ptr<T>(j);

			for(int i=srcdisp.cols-2;i>=0;i--)
			{
				const T disp = s[i];
				int sub = (int)(abs(disp-s[i-1]));
				bool issub = (sub<=sub_gap)?true:false;
				const int dest = (int)(disp*amp);
				const double ia = (disp*amp)-dest;
				const double a = 1.0-ia;

				if(sub>ljump || abs(disp-s[i+1])>ljump)
				{
					i-=ij;
					continue;
				}


				if(i-dest-1>=0 &&i-dest-1<srcdisp.cols-1)
				{
					if(disp>d[i-dest])
					{
						m[i-dest]=255;
						d[i-dest]=disp;
						dim[3*(i-dest)+0]=sim[3*i+0];
						dim[3*(i-dest)+1]=sim[3*i+1];
						dim[3*(i-dest)+2]=sim[3*i+2];
						if(issub)
						{

							m[i-dest-1]=255;
							d[i-dest-1]=disp;
							dim[3*(i-dest-1)+0]=sim[3*i-3];
							dim[3*(i-dest-1)+1]=sim[3*i-2];
							dim[3*(i-dest-1)+2]=sim[3*i-1];
						}
					}
				}
			}
		}
	}
	else if(amp<0)
	{
		//#pragma omp parallel for
		for(int j=0;j<srcdisp.rows;j++)
		{
			uchar* sim = srcim.ptr<uchar>(j);
			uchar* dim = destim.ptr<uchar>(j);
			uchar* m = mask.ptr<uchar>(j);

			T* s = srcdisp.ptr<T>(j);
			T* d = destdisp.ptr<T>(j);
			for(int i=1;i<srcdisp.cols;i++)
			{
				const T disp = s[i];
				int sub=(int)(abs(disp-s[i+1]));
				bool issub = (sub<=sub_gap)?true:false;
				const int dest = (int)(-amp*disp);//符号
				const double ia = (-amp*disp)-dest;
				const double a = 1.0-ia;


				if(abs(disp-s[i-1])>ljump||abs(disp-s[i+1])>ljump)
				{
					i+=ij;
					continue;
				}

				if(i+dest+1>=0 &&i+dest+1<srcdisp.cols-1)
				{
					if(disp>d[i+dest])
					{
						m[i+dest]=255;
						d[i+dest]=(T)disp;

						dim[3*(i+dest)+0]=sim[3*i+0];
						dim[3*(i+dest)+1]=sim[3*i+1];
						dim[3*(i+dest)+2]=sim[3*i+2];

						if(issub)
						{
							m[i+dest+1]=255;
							d[i+dest+1]=(T)disp;
							dim[3*(i+dest+1)+0]=sim[3*i+3];
							dim[3*(i+dest+1)+1]=sim[3*i+4];
							dim[3*(i+dest+1)+2]=sim[3*i+5];
						}
					}
				}
			}
		}
	}
	else
	{
		srcim.copyTo(destim);
		srcdisp.copyTo(destdisp);
		mask.setTo(Scalar(255));
	}	
}

template <class T>
static void shiftImDispCubic_(Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask, const int large_jump, const int sub_gap)
{
	//	cout<<"Cubic"<<endl;
	const double cubic = -1.0;
	const double c1 = cubic;
	const double c2 = -5.0*cubic;
	const double c3 = 8.0*cubic;
	const double c4 = -4.0*cubic;
	const double c5 = 2.0+cubic;
	const double c6 = -(cubic+3.0);

	int ij=0;
	const int ljump = large_jump*sub_gap;

	//const int iamp=cvRound(amp);
	if(amp>0)
	{
		//#pragma omp parallel for
		for(int j=0;j<srcdisp.rows;j++)
		{
			uchar* sim = srcim.ptr<uchar>(j);
			uchar* dim = destim.ptr<uchar>(j);
			uchar* m = mask.ptr<uchar>(j);
			T* s = srcdisp.ptr<T>(j);
			T* d = destdisp.ptr<T>(j);

			for(int i=srcdisp.cols-2;i>=0;i--)
			{
				const T disp = s[i];
				int sub = (int)(abs(disp-s[i-1]));
				bool issub = (sub<=sub_gap)?true:false;
				const int dest = (int)(disp*amp);
				const double ia = (disp*amp)-dest;
				const double a = 1.0-ia;

				if(sub>ljump || abs(disp-s[i+1])>ljump)
				{
					i-=ij;
					continue;
				}
				if(i-dest-1>=0 &&i-dest-1<srcdisp.cols-1)
				{
					if(disp>d[i-dest])
					{
						if(ia==0.0)
						{
							m[i-dest]=255;
							d[i-dest]=disp;
							dim[3*(i-dest)+0]=sim[3*i+0];
							dim[3*(i-dest)+1]=sim[3*i+1];
							dim[3*(i-dest)+2]=sim[3*i+2];
							if(issub)
							{
								m[i-dest-1]=255;
								d[i-dest-1]=disp;
								dim[3*(i-dest-1)+0]=sim[3*i-3];
								dim[3*(i-dest-1)+1]=sim[3*i-2];
								dim[3*(i-dest-1)+2]=sim[3*i-1];
							}	
						}
						else 
						{
							const double viaa= c1*(1.0+ a)*(1.0+ a)*(1.0+ a) + c2*(1.0+ a)*(1.0+ a) +c3*(1.0+ a) + c4;
							const double iaa = c5* a* a* a + c6* a* a + 1.0;
							const double aa  = c5*ia*ia*ia + c6*ia*ia + 1.0;
							const double vaa = c1*(1.0+ ia)*(1.0+ ia)*(1.0+ ia) + c2*(1.0+ ia)*(1.0+ ia) +c3*(1.0+ ia) + c4;

							m[i-dest]=255;
							d[i-dest]=disp;
							dim[3*(i-dest)+0]=saturate_cast<uchar>(vaa*sim[3*i-3] + aa*sim[3*i+0]+iaa*sim[3*i+3]+viaa*sim[3*i+6]);
							dim[3*(i-dest)+1]=saturate_cast<uchar>(vaa*sim[3*i-2] + aa*sim[3*i+1]+iaa*sim[3*i+4]+viaa*sim[3*i+7]);
							dim[3*(i-dest)+2]=saturate_cast<uchar>(vaa*sim[3*i-1] + aa*sim[3*i+2]+iaa*sim[3*i+5]+viaa*sim[3*i+8]);
							if(issub)
							{
								m[i-dest-1]=255;
								d[i-dest-1]=disp;
								dim[3*(i-dest-1)+0]=saturate_cast<uchar>(vaa*sim[3*i-6] + aa*sim[3*i-3]+iaa*sim[3*i+0]+viaa*sim[3*i+3]);
								dim[3*(i-dest-1)+1]=saturate_cast<uchar>(vaa*sim[3*i-5] + aa*sim[3*i-2]+iaa*sim[3*i+1]+viaa*sim[3*i+4]);
								dim[3*(i-dest-1)+2]=saturate_cast<uchar>(vaa*sim[3*i-4] + aa*sim[3*i-1]+iaa*sim[3*i+2]+viaa*sim[3*i+5]);
							}
						}							
					}
				}
			}
		}
	}
	else if(amp<0)
	{
		//#pragma omp parallel for
		for(int j=0;j<srcdisp.rows;j++)
		{
			uchar* sim = srcim.ptr<uchar>(j);
			uchar* dim = destim.ptr<uchar>(j);
			uchar* m = mask.ptr<uchar>(j);

			T* s = srcdisp.ptr<T>(j);
			T* d = destdisp.ptr<T>(j);
			for(int i=1;i<srcdisp.cols;i++)
			{
				const T disp = s[i];
				int sub=(int)(abs(disp-s[i+1]));
				bool issub = (sub<=sub_gap)?true:false;
				const int dest = (int)(-amp*disp);//符号
				const double ia = (-amp*disp)-dest;
				const double a = 1.0-ia;

				if(abs(disp-s[i-1])>ljump||abs(disp-s[i+1])>ljump)
				{
					i+=ij;
					continue;
				}

				if(i+dest+1>=0 &&i+dest+1<srcdisp.cols-1)
				{
					if(disp>d[i+dest])
					{
						if(ia == 0.0)
						{
							m[i+dest]=255;
							d[i+dest]=(T)disp;
							dim[3*(i+dest)+0]=sim[3*i+0];
							dim[3*(i+dest)+1]=sim[3*i+1];
							dim[3*(i+dest)+2]=sim[3*i+2];
							if(issub)
							{
								m[i+dest+1]=255;
								d[i+dest+1]=(T)disp;
								dim[3*(i+dest+1)+0]=sim[3*i+3];
								dim[3*(i+dest+1)+1]=sim[3*i+4];
								dim[3*(i+dest+1)+2]=sim[3*i+5];
							}
						}
						else
						{
							const double viaa= c1*(1.0+ a)*(1.0+ a)*(1.0+ a) + c2*(1.0+ a)*(1.0+ a) +c3*(1.0+ a) + c4;
							const double iaa = c5* a* a* a + c6* a* a + 1.0;
							const double aa  = c5*ia*ia*ia + c6*ia*ia + 1.0;
							const double vaa = c1*(1.0+ ia)*(1.0+ ia)*(1.0+ ia) + c2*(1.0+ ia)*(1.0+ ia) +c3*(1.0+ ia) + c4;
							m[i+dest]=255;
							d[i+dest]=(T)disp;

							dim[3*(i+dest)+0]=saturate_cast<uchar>(vaa*sim[3*i+3]+aa*sim[3*i+0]+iaa*sim[3*i-3]+viaa*sim[3*i-6]);
							dim[3*(i+dest)+1]=saturate_cast<uchar>(vaa*sim[3*i+4]+aa*sim[3*i+1]+iaa*sim[3*i-2]+viaa*sim[3*i-5]);
							dim[3*(i+dest)+2]=saturate_cast<uchar>(vaa*sim[3*i+5]+aa*sim[3*i+2]+iaa*sim[3*i-1]+viaa*sim[3*i-4]);
							if(issub)
							{
								m[i+dest+1]=255;
								d[i+dest+1]=(T)disp;

								dim[3*(i+dest+1)+0]=saturate_cast<uchar>(vaa*sim[3*i+6]+aa*sim[3*i+3]+iaa*sim[3*i+0]+viaa*sim[3*i-3]);
								dim[3*(i+dest+1)+1]=saturate_cast<uchar>(vaa*sim[3*i+7]+aa*sim[3*i+4]+iaa*sim[3*i+1]+viaa*sim[3*i-2]);
								dim[3*(i+dest+1)+2]=saturate_cast<uchar>(vaa*sim[3*i+8]+aa*sim[3*i+5]+iaa*sim[3*i+2]+viaa*sim[3*i-1]);
							}
						}
					}
				}
			}
		}
	}
	else
	{
		//Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask
		srcim.copyTo(destim);
		srcdisp.copyTo(destdisp);
		mask.setTo(Scalar(255));
	}	
}
template <class T>
static void shiftImDispLinear_(Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask, const int large_jump, const int sub_gap)
{
	int ij=0;
	const int ljump = large_jump*sub_gap;
	//const int iamp=cvRound(amp);
	if(amp>0)
	{
		//#pragma omp parallel for
		for(int j=0;j<srcdisp.rows;j++)
		{
			uchar* sim = srcim.ptr<uchar>(j);
			uchar* dim = destim.ptr<uchar>(j);
			uchar* m = mask.ptr<uchar>(j);
			T* s = srcdisp.ptr<T>(j);
			T* d = destdisp.ptr<T>(j);

			for(int i=srcdisp.cols-2;i>=0;i--)
			{
				const T disp = s[i];
				int sub = (int)(abs(disp-s[i-1]));
				bool issub = (sub<=sub_gap)?true:false;
				const int dest = (int)(disp*amp);
				const double ia = (disp*amp)-dest;
				const double a = 1.0-ia;

				if(sub>ljump || abs(disp-s[i+1])>ljump)
				{
					i-=ij;
					continue;
				}


				if(i-dest-1>=0 &&i-dest-1<srcdisp.cols-1)
				{
					if(disp>d[i-dest])
					{
						m[i-dest]=255;
						d[i-dest]=disp;

						dim[3*(i-dest)+0]=saturate_cast<uchar>(a*sim[3*i+0]+ia*sim[3*i+3]);
						dim[3*(i-dest)+1]=saturate_cast<uchar>(a*sim[3*i+1]+ia*sim[3*i+4]);
						dim[3*(i-dest)+2]=saturate_cast<uchar>(a*sim[3*i+2]+ia*sim[3*i+5]);

						if(issub)
						{

							m[i-dest-1]=255;
							d[i-dest-1]=disp;

							dim[3*(i-dest-1)+0]=saturate_cast<uchar>(a*sim[3*i-3]+ia*sim[3*i+0]);
							dim[3*(i-dest-1)+1]=saturate_cast<uchar>(a*sim[3*i-2]+ia*sim[3*i+1]);
							dim[3*(i-dest-1)+2]=saturate_cast<uchar>(a*sim[3*i-1]+ia*sim[3*i+2]);
						}
					}
				}
			}
		}
	}
	else if(amp<0)
	{
		//#pragma omp parallel for
		for(int j=0;j<srcdisp.rows;j++)
		{
			uchar* sim = srcim.ptr<uchar>(j);
			uchar* dim = destim.ptr<uchar>(j);
			uchar* m = mask.ptr<uchar>(j);

			T* s = srcdisp.ptr<T>(j);
			T* d = destdisp.ptr<T>(j);
			for(int i=1;i<srcdisp.cols;i++)
			{
				const T disp = s[i];
				int sub=(int)(abs(disp-s[i+1]));
				bool issub = (sub<=sub_gap)?true:false;
				const int dest = (int)(-amp*disp);//符号
				const double ia = (-amp*disp)-dest;
				const double a = 1.0-ia;


				if(abs(disp-s[i-1])>ljump||abs(disp-s[i+1])>ljump)
				{
					i+=ij;
					continue;
				}

				if(i+dest+1>=0 &&i+dest+1<srcdisp.cols-1)
				{
					if(disp>d[i+dest])
					{
						m[i+dest]=255;
						d[i+dest]=(T)disp;

						dim[3*(i+dest)+0]=saturate_cast<uchar>(a*sim[3*i+0]+ia*sim[3*i-3]);
						dim[3*(i+dest)+1]=saturate_cast<uchar>(a*sim[3*i+1]+ia*sim[3*i-2]);
						dim[3*(i+dest)+2]=saturate_cast<uchar>(a*sim[3*i+2]+ia*sim[3*i-1]);

						if(issub)
						{
							m[i+dest+1]=255;
							d[i+dest+1]=(T)disp;

							dim[3*(i+dest+1)+0]=saturate_cast<uchar>(a*sim[3*i+3]+ia*sim[3*i+0]);
							dim[3*(i+dest+1)+1]=saturate_cast<uchar>(a*sim[3*i+4]+ia*sim[3*i+1]);
							dim[3*(i+dest+1)+2]=saturate_cast<uchar>(a*sim[3*i+5]+ia*sim[3*i+2]);
						}
					}
				}
			}
		}
	}
	else
	{
		//Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp, Mat& mask
		srcim.copyTo(destim);
		srcdisp.copyTo(destdisp);
		mask.setTo(Scalar(255));
	}	
}


template <class T>
static void shiftImDisp(Mat& srcim, Mat& srcdisp, Mat& destim, Mat& destdisp, double amp,double sub_gap,const int large_jump = 3 ,Mat& mask=Mat(), int method = WARP_CUBIC)
{
	Mat mask_=mask;
	if(mask_.empty())mask_=Mat::zeros(srcim.size(),CV_8U);

	if(srcdisp.type()==CV_8U)
	{
		if(destdisp.empty())destdisp = Mat::zeros(srcdisp.size(),CV_8U);
		else destdisp.setTo(0);
		if(destim.empty())destim = Mat::zeros(srcim.size(),CV_8UC3);
		else destim.setTo(0);					
	}
	else if(srcdisp.type()==CV_16S)
	{
		if(destdisp.empty())destdisp= Mat::zeros(srcdisp.size(),CV_16S);
		else destdisp.setTo(0);
		if(destim.empty())destim = Mat::zeros(srcim.size(),CV_8UC3);
		else destim.setTo(0);	
	}
	else if(srcdisp.type()==CV_16U)
	{
		if(destdisp.empty())destdisp= Mat::zeros(srcdisp.size(),CV_16U);
		else destdisp.setTo(0);
		if(destim.empty())destim = Mat::zeros(srcim.size(),CV_8UC3);
		else destim.setTo(0);	
	}
	else if(srcdisp.type()==CV_32F)
	{
		if(destdisp.empty())destdisp= Mat::zeros(srcdisp.size(),CV_32F);
		else destdisp.setTo(0);
		if(destim.empty())destim = Mat::zeros(srcim.size(),CV_8UC3);
		else destim.setTo(0);
	}
	if(method == WARP_NN)
		shiftImDispNN_<T>(srcim,srcdisp,destim,destdisp,amp,mask_,large_jump, (int)sub_gap);
	else if(method == WARP_LINEAR)
		shiftImDispLinear_<T>(srcim,srcdisp,destim,destdisp,amp,mask_,large_jump, (int)sub_gap);	
	else if(method == WARP_CUBIC)
		shiftImDispCubic_<T>(srcim,srcdisp,destim,destdisp,amp,mask_,large_jump, (int)sub_gap);

	mask_.copyTo(mask);
}

template <class T>
static void fillOcclusionImDisp2_(Mat& im, Mat& src, T invalidvalue,int maxlength=1000)
{
	//Mat mask=Mat::zeros(im.size(),CV_8U);
//#pragma omp parallel for
	for(int j=0;j<src.rows;j++)
	{
		uchar* ims = im.ptr<uchar>(j);
		T* s = src.ptr<T>(j);
		//	uchar* m = mask.ptr<uchar>(j);
		const T st = s[0];
		const T ed = s[src.cols-1];

		s[0]=255;//可能性のある最大値を入力
		s[src.cols-1]=255;//可能性のある最大値を入力
		//もし視差が0だったら値の入っている近傍のピクセル（エピポーラ線上）の最小値で埋める
		if(j>0 && j<src.rows-1)
		{
			for(int i=1;i<src.cols;i++)
			{
				if(s[i]<=invalidvalue)
				{
					int t=i;
					int count=0;
					do
					{
						count++;
						t++;
						if(count>maxlength)break;
						if(t>src.cols-2)break;
					}while(s[t]<=invalidvalue);
					if(count>maxlength)break;


					uchar ce[3];
					T dd;
					if(s[i-1]<s[t])
					{
						dd = s[i-1];

						int count=1;
						int r=ims[3*i-3];
						int g=ims[3*i-2];
						int b=ims[3*i-1];
						if(s[i-1-src.cols]>invalidvalue)
						{
							r+=ims[3*(i-1-src.cols)+0];
							g+=ims[3*(i-1-src.cols)+1];
							b+=ims[3*(i-1-src.cols)+2];
							count++;
						}
						if(s[i-1+src.cols]>invalidvalue)
						{
							r+=ims[3*(i-1+src.cols)+0];
							g+=ims[3*(i-1+src.cols)+1];
							b+=ims[3*(i-1+src.cols)+2];
							count++;
						}
						ce[0]=r/count;
						ce[1]=g/count;
						ce[2]=b/count;
					}
					else
					{
						dd = s[t];
						int count=1;
						int r=ims[3*t+0];
						int g=ims[3*t+1];
						int b=ims[3*t+2];
						if(s[t-src.cols]>invalidvalue)
						{
							r+=ims[3*(t-src.cols)+0];
							g+=ims[3*(t-src.cols)+1];
							b+=ims[3*(t-src.cols)+2];
							count++;
						}
						if(s[t+src.cols]>invalidvalue)
						{
							r+=ims[3*(i+src.cols)+0];
							g+=ims[3*(i+src.cols)+1];
							b+=ims[3*(i+src.cols)+2];
							count++;
						}
						ce[0]=r/count;
						ce[1]=g/count;
						ce[2]=b/count;
					}

					if(t-i>src.cols-3)
					{
						for(int n=0;n<src.cols;n++)
						{
							s[i]=invalidvalue;
							ims[3*i+0]=ce[0];
							ims[3*i+1]=ce[1];
							ims[3*i+2]=ce[2];
						}
					}
					else
					{
						for(;i<t;i++)
						{
							s[i]=dd;
							//m[i]=255;
							ims[3*i+0]=ce[0];
							ims[3*i+1]=ce[1];
							ims[3*i+2]=ce[2];
						}
					}	
				}
			}
		}
		else
		{
			for(int i=1;i<src.cols;i++)
			{
				if(s[i]<=invalidvalue)
				{
					uchar cs[3];
					cs[0]=ims[3*i-3];
					cs[1]=ims[3*i-2];
					cs[2]=ims[3*i-1];
					int t=i;
					int count=0;
					do
					{
						count++;
						t++;
						if(count>maxlength)break;
						if(t>src.cols-2)break;
					}while(s[t]<=invalidvalue);
					if(count>maxlength)break;

					uchar ce[3];

					T dd;
					if(s[i-1]<s[t])
					{
						dd = s[i-1];
						ce[0]=cs[0];
						ce[1]=cs[1];
						ce[2]=cs[2];
					}
					else
					{
						dd = s[t];
						ce[0]=ims[3*t+0];
						ce[1]=ims[3*t+1];
						ce[2]=ims[3*t+2];
					}

					if(t-i>src.cols-3)
					{
						for(int n=0;n<src.cols;n++)
						{
							s[i]=invalidvalue;
							ims[3*i+0]=ce[0];
							ims[3*i+1]=ce[1];
							ims[3*i+2]=ce[2];
						}
					}
					else
					{
						for(;i<t;i++)
						{
							s[i]=dd;
							//m[i]=255;
							ims[3*i+0]=ce[0];
							ims[3*i+1]=ce[1];
							ims[3*i+2]=ce[2];
						}
					}	
				}
			}
		}

		s[0]=st;//もとに戻す
		if(st<=invalidvalue)
		{
			s[0]=s[1];
			ims[0]=ims[3];
			ims[1]=ims[4];
			ims[2]=ims[5];
		}
		s[src.cols-1]=ed;
		if(ed<=invalidvalue)
		{
			s[src.cols-1]=s[src.cols-2];
			ims[3*src.cols-3]=ims[3*src.cols-6];
			ims[3*src.cols-2]=ims[3*src.cols-5];
			ims[3*src.cols-1]=ims[3*src.cols-4];
		}
	}
}
template <class T>
static void fillOcclusionImDisp_(Mat& im, Mat& src, T invalidvalue, int maxlength=1000)
{
	T maxval;
	if(sizeof(T)==1)maxval = 255;
	else maxval = (T)32000;
	//	Mat mask=Mat::zeros(im.size(),CV_8U);
	//#pragma omp parallel for
	for(int j=0;j<src.rows;j++)
	{
		uchar* ims = im.ptr<uchar>(j);
		T* s = src.ptr<T>(j);
		//	uchar* m = mask.ptr<uchar>(j);
		const T st = s[0];
		const T ed = s[src.cols-1];

		s[0]=maxval;//可能性のある最大値を入力
		s[src.cols-1]=maxval;//可能性のある最大値を入力
		//もし視差が0だったら値の入っている近傍のピクセル（エピポーラ線上）の最小値で埋める
		for(int i=1;i<src.cols;i++)
		{
			if(s[i]<=invalidvalue)
			{
				uchar cs[3];
				cs[0]=ims[3*i-3];
				cs[1]=ims[3*i-2];
				cs[2]=ims[3*i-1];
				int t=i;
				int count=0;
				do
				{
					count++;
					t++;
					if(count>maxlength)break;

					if(t>src.cols-2)
					{
						break;
					}
				}while(s[t]<=invalidvalue);
				if(count>maxlength)break;

				uchar ce[3];

				T dd;
				if(s[i-1]<s[t])
				{
					dd = s[i-1];
					ce[0]=cs[0];
					ce[1]=cs[1];
					ce[2]=cs[2];
				}
				else
				{
					dd = s[t];
					ce[0]=ims[3*t+0];
					ce[1]=ims[3*t+1];
					ce[2]=ims[3*t+2];
				}

				if(t-i>src.cols-3)
				{
					for(int n=0;n<src.cols;n++)
					{
						s[i]=invalidvalue;
						ims[3*i+0]=ce[0];
						ims[3*i+1]=ce[1];
						ims[3*i+2]=ce[2];

					}
				}
				else
				{
					for(;i<t;i++)
					{
						s[i]=dd;
						//m[i]=255;
						ims[3*i+0]=ce[0];
						ims[3*i+1]=ce[1];
						ims[3*i+2]=ce[2];

					}
				}	
			}
		}
		s[0]=st;//もとに戻す
		if(st<=invalidvalue)
		{
			s[0]=s[1];
			ims[0]=ims[3];
			ims[1]=ims[4];
			ims[2]=ims[5];
		}
		s[src.cols-1]=ed;
		if(ed<=invalidvalue)
		{
			s[src.cols-1]=s[src.cols-2];
			ims[3*src.cols-3]=ims[3*src.cols-6];
			ims[3*src.cols-2]=ims[3*src.cols-5];
			ims[3*src.cols-1]=ims[3*src.cols-4];
		}
	}
}

template <class T>
static void fillOcclusionImDispReflect_(Mat& im, Mat& src, T invalidvalue,int maxlength=1000)
{
	Mat imc;
	Mat sc;
	im.copyTo(imc);
	src.copyTo(sc);
	fillOcclusionImDisp_<T>(imc,sc,invalidvalue);

	//Mat mask=Mat::zeros(im.size(),CV_8U);
//#pragma omp parallel for
	for(int j=0;j<src.rows;j++)
	{
		T* dref = sc.ptr<T>(j);
		uchar* imref = imc.ptr<uchar>(j);

		uchar* ims = im.ptr<uchar>(j);
		T* s = src.ptr<T>(j);
		//uchar* m = mask.ptr<uchar>(j);
		const T st = s[0];
		const T ed = s[src.cols-1];

		s[0]=255;//可能性のある最大値を入力
		s[src.cols-1]=255;//可能性のある最大値を入力
		//もし視差が0だったら値の入っている近傍のピクセル（エピポーラ線上）の最小値で埋める
		for(int i=1;i<src.cols;i++)
		{
			if(s[i]<=invalidvalue)
			{
				int t=i;
				int count=0;
				do
				{
					count++;
					t++;
					if(count>maxlength)break;
					if(t>src.cols-2)break;
				}while(s[t]<=invalidvalue);
				if(count>maxlength)break;
				//外枠は例外
				if(t==src.cols-1)
				{
					memcpy(ims+3*i,imref+3*i,3*(src.cols-1-i));
					memcpy(s+sizeof(T)*i,dref+sizeof(T)*i,sizeof(T)*(src.cols-1-i));
					continue;
				}
				if(i==1)
				{
					memcpy(ims,imref,3*t);
					memcpy(s,dref,sizeof(T)*t);
					i=t;
					continue;
				}

				T dd;

				if(s[i-1]<s[t])
				{
					dd = s[i-1];
					int p=i;
					int count=1;
					for(;i<t;i++)
					{
						s[i]=dd;
						//m[i]=255;
						ims[3*i+0]=imref[3*(p-count)+0];
						ims[3*i+1]=imref[3*(p-count)+1];
						ims[3*i+2]=imref[3*(p-count++)+2];
						/*ims[3*i+0]=0;
						ims[3*i+1]=255;
						ims[3*i+2]=0;*/
					}
				}
				else
				{
					dd = s[t];
					int p=i;
					int count=t-i+1;
					for(int k=1;k<count;k++)
					{
						s[t-k]=dd;
						//m[t-k]=255;
						ims[3*(t-k)+0]=imref[3*(t+k)+0];
						ims[3*(t-k)+1]=imref[3*(t+k)+1];
						ims[3*(t-k)+2]=imref[3*(t+k)+2];
						/*ims[3*(t-k)+0]=0;
						ims[3*(t-k)+1]=0;
						ims[3*(t-k)+2]=255;*/
					}
				}
			}
		}
		s[0]=st;//もとに戻す
		if(st<=invalidvalue)
		{
			s[0]=s[1];
			ims[0]=ims[3];
			ims[1]=ims[4];
			ims[2]=ims[5];
		}
		s[src.cols-1]=ed;
		if(ed<=invalidvalue)
		{
			s[src.cols-1]=s[src.cols-2];
			ims[3*src.cols-3]=ims[3*src.cols-6];
			ims[3*src.cols-2]=ims[3*src.cols-5];
			ims[3*src.cols-1]=ims[3*src.cols-4];
		}
	}
}

template <class T>
static void fillOcclusionImDispStretch_(Mat& im, Mat& src, T invalidvalue,int maxlength=1000)
{
	Mat imc;
	Mat sc;
	im.copyTo(imc);
	src.copyTo(sc);
	fillOcclusionImDisp_<T>(imc,sc,invalidvalue);
	//Mat mask=Mat::zeros(im.size(),CV_8U);
//#pragma omp parallel for
	for(int j=0;j<src.rows-1;j++)
	{
		T* dref = sc.ptr<T>(j);
		uchar* imref = imc.ptr<uchar>(j);
		uchar* ims = im.ptr<uchar>(j);
		T* s = src.ptr<T>(j);
		//uchar* m = mask.ptr<uchar>(j);
		const T st = s[0];
		const T ed = s[src.cols-1];

		s[0]=255;//可能性のある最大値を入力
		s[src.cols-1]=255;//可能性のある最大値を入力
		//もし視差が0だったら値の入っている近傍のピクセル（エピポーラ線上）の最小値で埋める
		for(int i=0;i<src.cols;i++)
		{
			if(s[i]<=invalidvalue)
			{
				int t=i;
				int count=0;
				do
				{
					count++;
					t++;
					if(count>maxlength)break;
					if(t>src.cols-2)break;
				}while(s[t]<=invalidvalue);
				if(count>maxlength)break;
				//外枠は例外
				if(t==src.cols-1)
				{
					memcpy(ims+3*i,imref+3*i,3*(src.cols-1-i));
					memcpy(s+sizeof(T)*i,dref+sizeof(T)*i,sizeof(T)*(src.cols-1-i));
					continue;
				}
				if(i==1)
				{
					memcpy(ims,imref,3*t);
					memcpy(s,dref,sizeof(T)*t);
					i=t;
					continue;
				}

				T dd;

				if(s[i-1]<s[t])
				{
					dd = s[i-1];

					int count=2*(t-i);
					int p=i-(t-i);
					for(int k=0;k<count;k+=2)
					{
						s[p+k]=dd;
						//m[p+k]=255;
						int f = (k>>1);
						ims[3*(p+k)+0]=imref[3*(p+f)+0];
						ims[3*(p+k)+1]=imref[3*(p+f)+1];
						ims[3*(p+k)+2]=imref[3*(p+f)+2];

						/*ims[3*(p+k)+0]=0;
						ims[3*(p+k)+1]=255;
						ims[3*(p+k)+2]=0;*/
					}
					for(int k=1;k<count;k+=2)
					{
						s[p+k]=dd;
						//	m[p+k]=255;
						int f = (k>>1);
						ims[3*(p+k)+0]=(imref[3*(p+f)+0]+imref[3*(p+f+1)+0])>>1;
						ims[3*(p+k)+1]=(imref[3*(p+f)+1]+imref[3*(p+f+1)+1])>>1;
						ims[3*(p+k)+2]=(imref[3*(p+f)+2]+imref[3*(p+f+1)+2])>>1;

						/*ims[3*(p+k)+0]=0;
						ims[3*(p+k)+1]=255;
						ims[3*(p+k)+2]=0;*/
					}

					i=t;
				}
				else
				{
					dd = s[t];
					int p=i;
					int count=(t-i-1)*2;
					i+=(t-i);
					for(int k=0;k<count;k+=2)
					{
						s[p+k]=dd;
						//	m[p+k]=255;
						int f = (k>>1);
						ims[3*(p+k)+0]=imref[3*(t+f)+0];
						ims[3*(p+k)+1]=imref[3*(t+f)+1];
						ims[3*(p+k)+2]=imref[3*(t+f)+2];
						//ims[3*(p+k)+0]=0;
						//ims[3*(p+k)+1]=0;
						//ims[3*(p+k)+2]=255;
					}
					for(int k=1;k<count;k+=2)
					{
						s[p+k]=dd;
						//	m[p+k]=255;
						int f = (k>>1);
						ims[3*(p+k)+0]=(imref[3*(t+f)+0]+imref[3*(t+f+1)+0])>>1;
						ims[3*(p+k)+1]=(imref[3*(t+f)+1]+imref[3*(t+f+1)+1])>>1;
						ims[3*(p+k)+2]=(imref[3*(t+f)+2]+imref[3*(t+f+1)+2])>>1;
						//ims[3*(p+k)+0]=0;
						//ims[3*(p+k)+1]=0;
						//ims[3*(p+k)+2]=255;
					}
				}
			}
		}
		s[0]=st;//もとに戻す
		if(st<=invalidvalue)
		{
			s[0]=s[1];
			ims[0]=ims[3];
			ims[1]=ims[4];
			ims[2]=ims[5];
		}
		s[src.cols-1]=ed;
		if(ed<=invalidvalue)
		{
			s[src.cols-1]=s[src.cols-2];
			ims[3*src.cols-3]=ims[3*src.cols-6];
			ims[3*src.cols-2]=ims[3*src.cols-5];
			ims[3*src.cols-1]=ims[3*src.cols-4];
		}
	}
}
template <class T>
static void fillOcclusionImDispBlur_(Mat& im, Mat& src, T invalidvalue)
{
	int bb=1;	
//#pragma omp parallel for
	for(int j=bb;j<src.rows-bb;j++)
	{
		uchar* ims = im.ptr<uchar>(j);
		T* s = src.ptr<T>(j);
		const T st = s[0];
		const T ed = s[src.cols-1];

		s[0]=255;//可能性のある最大値を入力
		s[src.cols-1]=255;//可能性のある最大値を入力
		//もし視差が0だったら値の入っている近傍のピクセル（エピポーラ線上）の最小値で埋める
		for(int i=0;i<src.cols;i++)
		{
			if(s[i]<=invalidvalue)
			{
				uchar cs[3];
				cs[0]=ims[3*i-3];
				cs[1]=ims[3*i-2];
				cs[2]=ims[3*i-1];
				int t=i;
				do
				{
					t++;
					if(t>src.cols-1)break;
				}while(s[t]<=invalidvalue);
				uchar ce[3];
				ce[0]=ims[3*t+0];
				ce[1]=ims[3*t+1];
				ce[2]=ims[3*t+2];

				T dd;
				if(s[i-1]<s[t])
				{
					dd = s[i-1];
					ce[0]=cs[0];
					ce[1]=cs[1];
					ce[2]=cs[2];
				}
				else
				{
					dd = s[t];
				}

				if(t-i>src.cols-3)
				{
					for(int n=0;n<src.cols;n++)
					{
						s[i]=invalidvalue;
						ims[3*i+0]=ce[0];
						ims[3*i+1]=ce[1];
						ims[3*i+2]=ce[2];
					}
				}
				else
				{
					for(;i<t;i++)
					{
						s[i]=dd;
						ims[3*i+0]=ce[0];
						ims[3*i+1]=ce[1];
						ims[3*i+2]=ce[2];
					}
				}

			}
		}
		s[0]=st;//もとに戻す
		s[src.cols-1]=ed;
	}
}

void fillOcclusionImDisp(Mat& im, Mat& disp,int invalidvalue, int mode)
{
	if(mode==FILL_OCCLUSION_LINE)
	{
		if(disp.type()==CV_8U)
		{
			fillOcclusionImDisp_<uchar>(im,disp, (uchar)invalidvalue,10000);
		}
		else if(disp.type()==CV_16S)
		{
			fillOcclusionImDisp_<short>(im,disp, (short)invalidvalue,10000);
		}
		else if(disp.type()==CV_16U)
		{
			fillOcclusionImDisp_<unsigned short>(im,disp, (short)invalidvalue,10000);
		}
		else if(disp.type()==CV_32F)
		{
			fillOcclusionImDisp_<float>(im,disp, (float)invalidvalue,10000);
		}
	}
	else if(mode==FILL_OCCLUSION_REFLECT)
	{
		//reflect interpolation
		if(disp.type()==CV_8U)
		{
			fillOcclusionImDispReflect_<uchar>(im,disp, (uchar)invalidvalue);
		}
		else if(disp.type()==CV_16S)
		{
			fillOcclusionImDispReflect_<short>(im,disp, (short)invalidvalue);
		}
		else if(disp.type()==CV_16U)
		{
			fillOcclusionImDispReflect_<unsigned short>(im,disp, (short)invalidvalue);
		}
		else if(disp.type()==CV_32F)
		{
			fillOcclusionImDispReflect_<float>(im,disp, (float)invalidvalue);
		}
	}
	else if(mode==FILL_OCCLUSION_STRETCH)
	{
		//stretch interpolation
		if(disp.type()==CV_8U)
		{
			fillOcclusionImDispStretch_<uchar>(im,disp, (uchar)invalidvalue);
		}
		else if(disp.type()==CV_16S)
		{
			fillOcclusionImDispStretch_<short>(im,disp, (short)invalidvalue);
		}
		else if(disp.type()==CV_16U)
		{
			fillOcclusionImDispStretch_<unsigned short>(im,disp, (short)invalidvalue);
		}
		else if(disp.type()==CV_32F)
		{
			fillOcclusionImDispStretch_<float>(im,disp, (float)invalidvalue);
		}
	}

	else if(mode==2)
	{
		//stretch interpolation
		if(disp.type()==CV_8U)
		{
			fillOcclusionImDisp2_<uchar>(im,disp, (uchar)invalidvalue);
		}
		else if(disp.type()==CV_16S)
		{
			fillOcclusionImDisp2_<short>(im,disp, (short)invalidvalue);
		}
		else if(disp.type()==CV_16U)
		{
			fillOcclusionImDisp2_<unsigned short>(im,disp, (short)invalidvalue);
		}
		else if(disp.type()==CV_32F)
		{
			fillOcclusionImDisp2_<float>(im,disp, (float)invalidvalue);
		}
	}
}


template <class T>
static void setRectficatedInvalidMask_(Mat& disp, Mat& image, T invalidvalue)
{
//#pragma omp parallel for
	for(int j=0;j<disp.rows;j++)
	{
		uchar* im=image.ptr<uchar>(j);
		T* d=disp.ptr<T>(j);
		for(int i=0;i<disp.cols;i++)
		{
			if(im[i]==0)
				d[i]=invalidvalue;
		}
	}
}
void setRectficatedInvalidMask(Mat& disp, Mat& image, int invalidvalue)
{
	Mat im;
	if(image.channels()!=1)cvtColor(image,im,CV_BGR2GRAY);
	else im=image;


	if(disp.type()==CV_8U)
	{
		setRectficatedInvalidMask_<uchar>(disp,im, (uchar)invalidvalue);
	}
	else if(disp.type()==CV_16S)
	{
		setRectficatedInvalidMask_<short>(disp,im, (short)invalidvalue);

	}
	else if(disp.type()==CV_32F)
	{
		setRectficatedInvalidMask_<float>(disp,im, (float)invalidvalue);
	}
}

template <class T>
void blendLR2(Mat& iml, Mat& imr, Mat& dispL, Mat& dispR ,Mat& dest, Mat& destdisp,Mat& maskL, Mat& maskR, double a)
{
	int dth=5;
	a = a>1.0 ? 1.0 :a;
	a = a<0.0 ? 0.0 :a;

	/*
	a =  a<0.1 ? a=0:a;
	a =  a>0.9 ? a=1:a;*/
	/*
	double aa=a;
	if(aa<0.5)a=2*a*a;
	else a = -2.0*(a-1)*(a-1)+1.0;*/		

	double ia =1.0- a;
	//#pragma omp parallel for
	for(int j=0;j<iml.rows;j++)
	{
		uchar* d=dest.ptr<uchar>(j);
		uchar* l=iml.ptr<uchar>(j);
		uchar* r=imr.ptr<uchar>(j);

		T* dd=destdisp.ptr<T>(j);
		T* dl=dispL.ptr<T>(j);
		T* dr=dispR.ptr<T>(j);

		uchar* ml=maskL.ptr<uchar>(j);
		uchar* mr=maskR.ptr<uchar>(j);

		for(int i=0;i<iml.cols;i++)
		{
			if(ml[i]==255&&mr[i]==255)
			{
				if(abs(dl[i]-dr[i])<dth)
				{
					dd[i]=(T)((dl[i]+dr[i])*0.5);
					d[3*i+0] = saturate_cast<uchar>(ia*l[3*i+0]+a*r[3*i+0]);
					d[3*i+1] = saturate_cast<uchar>(ia*l[3*i+1]+a*r[3*i+1]);
					d[3*i+2] = saturate_cast<uchar>(ia*l[3*i+2]+a*r[3*i+2]);
				} 
				else if(dl[i]>dr[i])
				{
					dd[i]=dl[i];
					d[3*i+0] = l[3*i+0];
					d[3*i+1] = l[3*i+1];
					d[3*i+2] = l[3*i+2];
				}
				else
				{
					dd[i]=dr[i];
					d[3*i+0] = r[3*i+0];
					d[3*i+1] = r[3*i+1];
					d[3*i+2] = r[3*i+2];
				}

			}
			else if(ml[i]==255)
			{
				//d[3*i+0] = 0.75*l[3*i+0]+0.25*r[3*i+0];
				//d[3*i+1] = 0.75*l[3*i+1]+0.25*r[3*i+1];
				//d[3*i+2] = 0.75*l[3*i+2]+0.25*r[3*i+2];

				dd[i]=dl[i];
				d[3*i+0] = l[3*i+0];
				d[3*i+1] = l[3*i+1];
				d[3*i+2] = l[3*i+2];
			}
			else if(mr[i]==255)
			{
				//d[3*i+0] = 0.25*l[3*i+0]+0.75*r[3*i+0];
				//d[3*i+1] = 0.25*l[3*i+1]+0.75*r[3*i+1];
				//d[3*i+2] = 0.25*l[3*i+2]+0.75*r[3*i+2];
				dd[i]=dr[i];
				d[3*i+0] = r[3*i+0];
				d[3*i+1] = r[3*i+1];
				d[3*i+2] = r[3*i+2];
			}
		}
	}	
}

template <class T>
void blendLR(Mat& iml, Mat& imr, Mat& dispL, Mat& dispR ,Mat& dest, Mat& destdisp,Mat& maskL, Mat& maskR, double a)
{
	int dth=16*8;
	a = a>1.0 ? 1.0 :a;
	a = a<0.0 ? 0.0 :a;

	/*
	a =  a<0.1 ? a=0:a;
	a =  a>0.9 ? a=1:a;*/
	/*
	double aa=a;
	if(aa<0.5)a=2*a*a;
	else a = -2.0*(a-1)*(a-1)+1.0;*/		

	double ia =1.0-a;
	//#pragma omp parallel for
	for(int j=0;j<iml.rows;j++)
	{
		uchar* d=dest.ptr<uchar>(j);
		uchar* l=iml.ptr<uchar>(j);
		uchar* r=imr.ptr<uchar>(j);

		T* dd=destdisp.ptr<T>(j);
		T* dl=dispL.ptr<T>(j);
		T* dr=dispR.ptr<T>(j);

		uchar* ml=maskL.ptr<uchar>(j);
		uchar* mr=maskR.ptr<uchar>(j);

		for(int i=0;i<iml.cols;i++)
		{
			if(ml[i]==255&&mr[i]==255)
			{
				if(dl[i]-dr[i]>dth)
				{
					dd[i]=dl[i];
					mr[i]=0;
					d[3*i+0] = l[3*i+0];
					d[3*i+1] = l[3*i+1];
					d[3*i+2] = l[3*i+2];
				}
				else if(dr[i]-dl[i]>dth)
				{
					dd[i]=dr[i];
					ml[i]=0;
					d[3*i+0] = r[3*i+0];
					d[3*i+1] = r[3*i+1];
					d[3*i+2] = r[3*i+2];
				}
				else
				{
					dd[i]=(T)((dl[i]+dr[i])*0.5);
					d[3*i+0] = saturate_cast<uchar>(ia*l[3*i+0]+a*r[3*i+0]);
					d[3*i+1] = saturate_cast<uchar>(ia*l[3*i+1]+a*r[3*i+1]);
					d[3*i+2] = saturate_cast<uchar>(ia*l[3*i+2]+a*r[3*i+2]);
				}
			}
			else if(ml[i]==255)
			{
				//d[3*i+0] = 0.75*l[3*i+0]+0.25*r[3*i+0];
				//d[3*i+1] = 0.75*l[3*i+1]+0.25*r[3*i+1];
				//d[3*i+2] = 0.75*l[3*i+2]+0.25*r[3*i+2];

				dd[i]=dl[i];
				d[3*i+0] = l[3*i+0];
				d[3*i+1] = l[3*i+1];
				d[3*i+2] = l[3*i+2];
			}
			else if(mr[i]==255)
			{
				//d[3*i+0] = 0.25*l[3*i+0]+0.75*r[3*i+0];
				//d[3*i+1] = 0.25*l[3*i+1]+0.75*r[3*i+1];
				//d[3*i+2] = 0.25*l[3*i+2]+0.75*r[3*i+2];
				dd[i]=dr[i];
				d[3*i+0] = r[3*i+0];
				d[3*i+1] = r[3*i+1];
				d[3*i+2] = r[3*i+2];
			}
		}
	}


	Mat mk =Mat::zeros(maskL.size(),CV_8U);
	for(int j=0;j<iml.rows;j++)
	{
		uchar* d=dest.ptr<uchar>(j);
		uchar* l=iml.ptr<uchar>(j);
		uchar* r=imr.ptr<uchar>(j);

		T* dd=destdisp.ptr<T>(j);
		T* dl=dispL.ptr<T>(j);
		T* dr=dispR.ptr<T>(j);

		uchar* ml=maskL.ptr<uchar>(j);
		uchar* mr=maskR.ptr<uchar>(j);

		for(int i=0;i<iml.cols;i++)
		{

			if((ml[i]==0 && mr[i]==255) && (ml[i+1]==255 && mr[i+1]==255))
			{
				mk.at<uchar>(j,i)=255;	
				mk.at<uchar>(j,i+1)=255;
				uchar r = (d[3*i+0]+d[3*i+3])>>1;
				uchar g = (d[3*i+1]+d[3*i+4])>>1;
				uchar b = (d[3*i+2]+d[3*i+5])>>1;
				d[3*i+0] =(d[3*i+0]+r)>>1;
				d[3*i+1] =(d[3*i+1]+g)>>1;
				d[3*i+2] =(d[3*i+2]+b)>>1;
				d[3*i+3] =(d[3*i+3]+r)>>1;
				d[3*i+4] =(d[3*i+4]+g)>>1;
				d[3*i+5] =(d[3*i+5]+b)>>1;



			}
			if((ml[i]==0 && mr[i]==255) && (ml[i-1]==255 && mr[i-1]==255))
			{
				mk.at<uchar>(j,i)=255;	
				mk.at<uchar>(j,i-1)=255;

				uchar r = (d[3*i+0]+d[3*i-3])>>1;
				uchar g = (d[3*i+1]+d[3*i-2])>>1;
				uchar b = (d[3*i+2]+d[3*i-1])>>1;
				d[3*i+0] =(d[3*i+0]+r)>>1;
				d[3*i+1] =(d[3*i+1]+g)>>1;
				d[3*i+2] =(d[3*i+2]+b)>>1;
				d[3*i-3] =(d[3*i-3]+r)>>1;
				d[3*i-2] =(d[3*i-2]+g)>>1;
				d[3*i-1] =(d[3*i-1]+b)>>1;
			}

			if((ml[i]==255 && mr[i]==0) && (ml[i+1]==255 && mr[i+1]==255))
			{
				mk.at<uchar>(j,i)=255;	
				mk.at<uchar>(j,i+1)=255;
				uchar r = (d[3*i+0]+d[3*i+3])>>1;
				uchar g = (d[3*i+1]+d[3*i+4])>>1;
				uchar b = (d[3*i+2]+d[3*i+5])>>1;
				d[3*i+0] =(d[3*i+0]+r)>>1;
				d[3*i+1] =(d[3*i+1]+g)>>1;
				d[3*i+2] =(d[3*i+2]+b)>>1;
				d[3*i+3] =(d[3*i+3]+r)>>1;
				d[3*i+4] =(d[3*i+4]+g)>>1;
				d[3*i+5] =(d[3*i+5]+b)>>1;

			}
			if((ml[i]==255 && mr[i]==0) && (ml[i-1]==255 && mr[i-1]==255))
			{
				mk.at<uchar>(j,i)=255;	
				mk.at<uchar>(j,i-1)=255;

				uchar r = (d[3*i+0]+d[3*i-3])>>1;
				uchar g = (d[3*i+1]+d[3*i-2])>>1;
				uchar b = (d[3*i+2]+d[3*i-1])>>1;
				d[3*i+0] =(d[3*i+0]+r)>>1;
				d[3*i+1] =(d[3*i+1]+g)>>1;
				d[3*i+2] =(d[3*i+2]+b)>>1;
				d[3*i-3] =(d[3*i-3]+r)>>1;
				d[3*i-2] =(d[3*i-2]+g)>>1;
				d[3*i-1] =(d[3*i-1]+b)>>1;
			}
		}
	}
	//guiAlphaBlend(dest,mk);

}

template <class T>
static void fillBoundingBoxDepthIm(Mat& src_im, Mat& src_dp, int occflag)
{
//#pragma omp parallel for
	for(int j=0;j<src_im.rows;j++)
	{
		uchar* sim = src_im.ptr<uchar>(j);
		T* sdp = src_dp.ptr<T>(j);

		int k=0;
		int i=0;

		while(sdp[i]==occflag)
		{
			i++;
			k++;
			if(i==src_im.cols)continue;
		}
		for(i=0;i<k;i++)
		{
			sdp[i]=sdp[k];
			sim[3*i+0]=sim[3*k+0];
			sim[3*i+1]=sim[3*k+1];
			sim[3*i+2]=sim[3*k+2];
		}

		i=src_im.cols-2;
		k=0;

		while(sdp[i]==occflag)
		{
			i--;
			k++;
			if(i==-1)continue;
		}
		for(i=src_im.cols-k-1;i<src_im.cols;i++)
		{
			sdp[i]=sdp[src_im.cols-k-2];
			sim[3*i+0]=sim[3*(src_im.cols-2-k)+0];
			sim[3*i+1]=sim[3*(src_im.cols-2-k)+1];
			sim[3*i+2]=sim[3*(src_im.cols-2-k)+2];
		}
	}
}
template <class T>
static void fillBoundingBoxDepth(Mat& src_dp, int occflag)
{
//#pragma omp parallel for
	for(int j=0;j<src_dp.rows;j++)
	{
		T* sdp = src_dp.ptr<T>(j);

		int k=0;
		int i=0;

		while(sdp[i]==occflag)
		{
			i++;
			k++;
			if(i==src_dp.cols)continue;
		}
		for(i=0;i<k;i++)
		{
			sdp[i]=sdp[k];
		}

		i=src_dp.cols-2;
		k=0;

		while(sdp[i]==occflag)
		{
			i--;
			k++;
			if(i==-1)continue;
		}
		for(i=src_dp.cols-k-1;i<src_dp.cols;i++)
		{
			sdp[i]=sdp[src_dp.cols-k-2];
		}
	}
}


template <class T>
static void depthBasedInpaint(Mat& src_im, Mat& src_dp, Mat& dest_im, Mat& dest_dp, T OCC_FLAG)
{
	const T DISPINF = 255;
	int bs = 20;
	//erode(src_dp,dest_dp,Mat());
	src_dp.copyTo(dest_dp);
	src_im.copyTo(dest_im);

	bool loop = true;
	for(int iter=0;iter<10;iter++)
	{
		//printf("iter %d\n",iter);
		if(loop==false)break;
		loop = false;
		dest_im.copyTo(src_im);
		dest_dp.copyTo(src_dp);

//#pragma omp parallel for
		for(int j=bs;j<src_dp.rows-bs;j++)
		{
			uchar* dim = dest_im.ptr<uchar>(j);
			T* ddp = dest_dp.ptr<T>(j);
			T* sdp = src_dp.ptr<T>(j);

			for(int i=bs;i<src_dp.cols-bs;i++)
			{
				if(sdp[i]==OCC_FLAG)
				{	
					loop=true;
					T dmin=DISPINF;
					for(int l=-bs;l<=bs;l++)
					{
						T* bddp = src_dp.ptr<T>(j+l);
						for(int k=-bs;k<=bs;k++)
						{
							if(bddp[k+i]!=OCC_FLAG)
								dmin=min(dmin,bddp[k+i]);
						}
					}
					if(dmin==DISPINF)
					{
						continue;
					}

					int r = 0;
					int g = 0;
					int b = 0;
					int count=0;
					for(int l=-bs;l<=bs;l++)
					{
						T* bddp = src_dp.ptr<T>(j+l);
						uchar* sim = src_im.ptr<uchar>(j+l);
						for(int k=-bs;k<=bs;k++)
						{
							if(bddp[k+i]<dmin+2 &&bddp[k+i]>=dmin)
							{
								r+=sim[3*(k+i)+0];
								g+=sim[3*(k+i)+1];
								b+=sim[3*(k+i)+2];
								count++;
							}
						}
					}
					dim[3*i  ]=r/count;
					dim[3*i+1]=g/count;
					dim[3*i+2]=b/count;
					ddp[i]=dmin;
				}	

			}
		}
	}
}

template <class T>
static void shiftDisparity_(Mat& srcdisp, Mat& destdisp, double amp)
{
	const int sub_gap = 16;
	const int large_jump = 320;

	const int iamp=cvRound(amp);
	if(amp>0)
	{
		const int step = srcdisp.cols;
		T* s = srcdisp.ptr<T>(0);
		T* d = destdisp.ptr<T>(0);
		for(int j=0;j<srcdisp.rows;j++)
		{
			for(int i=srcdisp.cols-1;i>=0;i--)
			{
				const T disp = s[i];
				int sub = (int)(abs(disp-s[i-1]));
				bool issub = (sub<=sub_gap)?true:false;
				const int dest = (int)(disp*amp);

				if(abs(disp-s[i-1])>large_jump||abs(disp-s[i+1])>large_jump)
				{
					continue;
				}

				if(i-dest-1>=0 &&i-dest-1<srcdisp.cols-1)
				{
					if(disp>d[i-dest])
					{
						d[i-dest]=disp;
						if(issub)
							d[i-dest+1]=disp;
					}
				}
			}
			s+=step;
			d+=step;
		}
	}
	else if(amp<0)
	{
		const int step = srcdisp.cols;
		T* s = srcdisp.ptr<T>(0);
		T* d = destdisp.ptr<T>(0);
		for(int j=0;j<srcdisp.rows;j++)
		{
			for(int i=0;i<srcdisp.cols;i++)
			{
				const T disp = s[i];
				int sub = (int)(abs(disp-s[i+1]));
				bool issub = (sub<=sub_gap)?true:false;
				const int dest = (int)(-disp*amp);

				if(abs(disp-s[i-1])>large_jump||abs(disp-s[i+1])>large_jump)
				{
					continue;
				}

				if(i+dest+1>=0 &&i+dest+1<srcdisp.cols+1)
				{
					if(disp>d[i+dest])
					{
						d[i+dest]=disp;
						if(issub)
							d[i+dest-1]=disp;
					}
				}
			}
			s+=step;
			d+=step;
		}
	}
	else
	{
		srcdisp.copyTo(destdisp);
	}
}
void shiftDisparity(Mat& srcdisp, Mat& destdisp, double amp)
{

	if(srcdisp.type()==CV_8U)
	{
		if(destdisp.empty())destdisp = Mat::zeros(srcdisp.size(),CV_8U);
		else destdisp.setTo(0);
		shiftDisparity_<uchar>(srcdisp,destdisp,amp);
		fillBoundingBoxDepth<uchar>(destdisp,0);
		fillOcclusion(destdisp,0);
	}
	else if(srcdisp.type()==CV_16S)
	{
		if(destdisp.empty())destdisp= Mat::zeros(srcdisp.size(),CV_16S);
		else destdisp.setTo(0);
		shiftDisparity_<short>(srcdisp,destdisp,amp);
		//fillOcclusion(destdisp,0);
	}
	else if(srcdisp.type()==CV_16U)
	{
		if(destdisp.empty())destdisp= Mat::zeros(srcdisp.size(),CV_16U);
		else destdisp.setTo(0);
		shiftDisparity_<unsigned short>(srcdisp,destdisp,amp);
		//fillOcclusion(destdisp,0);
	}
	else if(srcdisp.type()==CV_32F)
	{
		if(destdisp.empty())destdisp= Mat::zeros(srcdisp.size(),CV_32F);
		else destdisp.setTo(0);
		shiftDisparity_<float>(srcdisp,destdisp,amp);
		fillBoundingBoxDepth<float>(destdisp,0);
		fillOcclusion(destdisp,0);
	}
}

void crackRemove(const Mat& src, Mat& dest,Mat& mask, const int invalidValue = 0)
{
	if(mask.empty())mask=Mat::zeros(src.size(),src.type());
	else mask.setTo(0);
	src.copyTo(dest);

	uchar* s = src.data;
	uchar* d = dest.data;
	uchar* m = mask.data;

	const int step = src.cols;
	s+=step;
	d+=step;
	m+=step;
	for(int j=1;j<src.rows-1;j++)
	{
		for(int i=1;i<src.cols-1;i++)
		{

			if(s[i]==invalidValue)
			{
				int v=0;
				int count = 0;
				if(s[i-1]!=invalidValue)
				{
					v+=s[i-1];
					count++;
				}
				if(s[i+1]!=invalidValue)
				{
					v+=s[i+1];
					count++;
				}
				if(s[i-step]!=invalidValue)
				{
					v+=s[i-step];
					count++;
				}
				if(s[i+step]!=invalidValue)
				{
					v+=s[i+step];
					count++;
				}/*
				 if(s[i-1+step]!=invalidValue)
				 {
				 v+=s[i-1+step];
				 count++;
				 }
				 if(s[i-1-step]!=invalidValue)
				 {
				 v+=s[i-1-step];
				 count++;
				 }
				 if(s[i+1+step]!=invalidValue)
				 {
				 v+=s[i+1+step];
				 count++;
				 }
				 if(s[i+1-step]!=invalidValue)
				 {
				 v+=s[i+1-step];
				 count++;
				 }*/
				if(count!=0)
				{
					d[i]=v/count;
					m[i]=255;
				}
			}
		}
		s+=step;
		d+=step;
		m+=step;
	}
}

void filterDepthSlant(Mat& depth, Mat& depth2,Mat& mask2, int kernelSize = 3)
{
	depth.copyTo(depth2);
	fillOcclusion(depth2,0);
	compare(depth,depth2,mask2,cv::CMP_NE);
}

void filterDepth2(Mat& depth, Mat& depth2,Mat& mask2, int kernelSize = 3)
{
	//depth.copyTo(depth2);mask2=Mat::zeros(depth.size(),CV_8U);return;
	//crackRemove(depth,depth2,mask2);
	//guiAlphaBlend(depth2,mask2);
	medianBlur(depth,depth2,kernelSize);
	medianBlur(depth2,depth2,kernelSize);

	Mat temp;
	depth2.convertTo(temp,CV_16SC1);
	filterSpeckles(temp,0,20,5);
	temp.convertTo(depth2,CV_8U);

	//imshow("depth",depth2)
	/*medianBlur(depth2,depth2,kernelSize);
	medianBlur(depth2,depth2,kernelSize);
	medianBlur(depth2,depth2,kernelSize);*/
	compare(depth,depth2,mask2,cv::CMP_NE);
}

void filterDepth(Mat& depth, Mat& depth2,Mat& mask2, int kernelSize, int viewstep)
{
	//depth.copyTo(depth2);mask2=Mat::zeros(depth.size(),CV_8U);return;
	//crackRemove(depth,depth2,mask2);
	//guiAlphaBlend(depth2,mask2);
	medianBlur(depth,depth2,kernelSize);
	medianBlur(depth2,depth2,kernelSize);
	if(viewstep>0)
	{
		maxFilter(depth2,depth2,Size(2*viewstep+1,1));
		minFilter(depth2,depth2,Size(2*viewstep+1,1));
	}

	//imshow("depth",depth2)
	/*medianBlur(depth2,depth2,kernelSize);
	medianBlur(depth2,depth2,kernelSize);
	medianBlur(depth2,depth2,kernelSize);*/
	compare(depth,depth2,mask2,cv::CMP_NE);
}
void shiftViewSynthesisLRFilter4(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp,int large_jump)
{
	if(alpha==0.0)
	{
		srcL.copyTo(dest);
		dispL.copyTo(destdisp);
		return;
	}
	else if(alpha==1.0)
	{
		srcR.copyTo(dest);
		dispR.copyTo(destdisp);
		return;
	}

	large_jump = large_jump<1 ?1:large_jump;
	if(dispL.type()==CV_8U)
	{
		if(dest.empty())dest.create(srcL.size(),CV_8UC3);
		else dest.setTo(0);

		if(destdisp.empty())destdisp.create(srcL.size(),CV_8U);
		else destdisp.setTo(0);

		Mat maskL(srcL.size(),CV_8U,Scalar(0));
		Mat maskL2(srcL.size(),CV_8U,Scalar(0));
		Mat maskR(srcL.size(),CV_8U,Scalar(0));
		Mat destR(srcL.size(),CV_8UC3);
		Mat destdispR(srcL.size(),CV_8U);
		Mat temp(srcL.size(),CV_8U);
		Mat temp2(srcL.size(),CV_8U);
		Mat swap(srcL.size(),CV_8UC3);
		{
			//CalcTime t("warp");

			shiftImDisp<uchar>(srcL,dispL,dest,temp,alpha*disp_amp,disp_amp,large_jump,maskL);
			//			fillOcclusion(temp);////////////
			filterDepth2(temp,destdisp,maskL2);
			//filterDepth(temp,destdisp,maskL2);
			Mat m;  compare(destdisp,0,m,cv::CMP_EQ);
			dest.setTo(0,m);
			maskL.setTo(0,m);
			maskL2.setTo(0,m);

			shiftImInv_<uchar>(srcL,destdisp,dest,-alpha*disp_amp,maskL2);
			maskL=maskL+maskL2;

			temp.setTo(0);
			shiftImDisp<uchar>(srcR,dispR,destR,temp,-disp_amp*(1.0-alpha),disp_amp,large_jump,maskR);

			filterDepth2(temp,destdispR,maskL2);
			//filterDepth(temp,destdispR,maskL2);
			Mat m2;  compare(destdispR,0,m2,cv::CMP_EQ);
			destR.setTo(0,m2);
			maskR.setTo(0,m2);
			maskL2.setTo(0,m2);
			shiftImInv_<uchar>(srcR,destdispR,destR,disp_amp*(1.0-alpha),maskL2);

			maskR=maskR+maskL2;
		}

		{
			//	CalcTime t("blend");			
			blendLR<uchar>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);
			//guiAlphaBlend(maskL,maskR);
			//	imwrite("im.bmp",destR);
			//			imwrite("binp_dp.bmp",destdispR);
		}
		{	
			//CalcTime t("inpaint");
			//fillBoundingBoxDepthIm<uchar>(destR,destdispR,0);

			//depthBasedInpaint<uchar>(destR,destdispR,dest,destdisp,0);
			Mat m;  compare(destdispR,0,m,cv::CMP_EQ);
			fillOcclusionImDisp(destR,destdispR,invalidvalue,0);
			destdispR.copyTo(destdisp);
			destR.copyTo(dest);

			Mat dest2;
			boxFilter(dest,dest2,-1,Size(3,3));
			dest2.copyTo(dest,m);
			//	imwrite("im_occ.bmp",dest);
		}
		Mat edge;
		cv::Canny(destdisp,edge,18,30);
		//imshow("ee",edge);
		//dilate(edge,edge,Mat(),Point(-1,-1),2);
		Mat a;
		GaussianBlur(dest,a,Size(3,3),3);
		a.copyTo(destR,edge);
		double aa = (alpha>1.0) ? 1.0:alpha;
		aa = (alpha<0.0) ? 0.0:aa;
		aa = (0.5-abs(aa-0.5))*2.0;
		addWeighted(dest, 1.0-aa,destR,aa,0.0,dest);
	}
	else if(dispL.type()==CV_16S)
	{
		;
	}
	else if(dispL.type()==CV_32F)
	{
		dest.create(srcL.size(),CV_8UC3);
		destdisp.create(srcL.size(),CV_32F);
		Mat maskL(srcL.size(),CV_8U,Scalar(0));
		Mat maskR(srcL.size(),CV_8U,Scalar(0));
		Mat maskdest(srcL.size(),CV_8U,Scalar(0));

		shiftImDisp<float>(srcL,dispL,dest,destdisp,alpha,disp_amp,large_jump,maskL);
		/*
		Mat temp;
		destdisp.convertTo(temp,CV_8U,30);
		imshow("temp",temp);
		waitKey();
		*/

		//fillOcclusionImDisp<uchar>(dest,destdisp,invalidvalue);

		Mat destR(srcL.size(),CV_8UC3);
		Mat destdispR;
		shiftImDisp<float>(srcR,dispR,destR,destdispR,-(1.0-alpha),disp_amp,large_jump,maskR);
		imshow("aaaaa",destR);
		//fillOcclusionImDisp<uchar>(temp,destdisp,invalidvalue);

		blendLR<float>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);

		fillOcclusionImDisp(destR,destdispR);
		destdispR.copyTo(destdisp);
		destR.copyTo(dest);

		//depthBasedInpaint<float>(destR,destdispR,dest,destdisp,0.0);


	}
}

void shiftViewSynthesisLRFilter3(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp,int large_jump)
{
	if(alpha==0.0)
	{
		srcL.copyTo(dest);
		dispL.copyTo(destdisp);
		return;
	}
	else if(alpha==1.0)
	{
		srcR.copyTo(dest);
		dispR.copyTo(destdisp);
		return;
	}

	large_jump = large_jump<1 ?1:large_jump;
	if(dispL.type()==CV_8U)
	{
		if(dest.empty())dest.create(srcL.size(),CV_8UC3);
		else dest.setTo(0);

		if(destdisp.empty())destdisp.create(srcL.size(),CV_8U);
		else destdisp.setTo(0);

		Mat maskL(srcL.size(),CV_8U,Scalar(0));
		Mat maskL2(srcL.size(),CV_8U,Scalar(0));
		Mat maskR(srcL.size(),CV_8U,Scalar(0));
		Mat destR(srcL.size(),CV_8UC3);
		Mat destdispR(srcL.size(),CV_8U);
		Mat temp(srcL.size(),CV_8U);
		Mat temp2(srcL.size(),CV_8U);
		Mat swap(srcL.size(),CV_8UC3);
		{
			//CalcTime t("warp");

			shiftImDisp<uchar>(srcL,dispL,dest,temp,alpha*disp_amp,disp_amp,large_jump,maskL);
			//			fillOcclusion(temp);////////////
			filterDepth(temp,destdisp,maskL2,3,cvRound(abs(alpha)));
			Mat m;  compare(destdisp,0,m,cv::CMP_EQ);
			dest.setTo(0,m);
			maskL.setTo(0,m);
			maskL2.setTo(0,m);

			shiftImInv_<uchar>(srcL,destdisp,dest,-alpha*disp_amp,maskL2);
			maskL=maskL+maskL2;

			temp.setTo(0);
			shiftImDisp<uchar>(srcR,dispR,destR,temp,-disp_amp*(1.0-alpha),disp_amp,large_jump,maskR);

			filterDepth(temp,destdispR,maskL2,3,cvRound(abs(alpha)));
			Mat m2;  compare(destdispR,0,m2,cv::CMP_EQ);
			destR.setTo(0,m2);
			maskR.setTo(0,m2);
			maskL2.setTo(0,m2);
			shiftImInv_<uchar>(srcR,destdispR,destR,disp_amp*(1.0-alpha),maskL2);

			maskR=maskR+maskL2;
		}

		{
			//	CalcTime t("blend");			
			blendLR<uchar>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);
			//guiAlphaBlend(maskL,maskR);
			//	imwrite("im.bmp",destR);
			//			imwrite("binp_dp.bmp",destdispR);
		}
		{	
			//CalcTime t("inpaint");
			//fillBoundingBoxDepthIm<uchar>(destR,destdispR,0);

			//depthBasedInpaint<uchar>(destR,destdispR,dest,destdisp,0);
			Mat m;  compare(destdispR,0,m,cv::CMP_EQ);
			fillOcclusionImDisp(destR,destdispR,invalidvalue);
			destdispR.copyTo(destdisp);
			destR.copyTo(dest);

			Mat dest2;
			boxFilter(dest,dest2,-1,Size(1,1));
			dest2.copyTo(dest,m);
			//	imwrite("im_occ.bmp",dest);
		}
		Mat edge;
		cv::Canny(destdisp,edge,18,30);
		//imshow("ee",edge);
		//dilate(edge,edge,Mat(),Point(-1,-1),2);
		Mat a;
		GaussianBlur(dest,a,Size(3,3),3);
		a.copyTo(destR,edge);
		double aa = (alpha>1.0) ? 1.0:alpha;
		aa = (alpha<0.0) ? 0.0:aa;
		aa = (0.5-abs(aa-0.5))*2.0;
		addWeighted(dest, 1.0-aa,destR,aa,0.0,dest);
	}
	else if(dispL.type()==CV_16S)
	{
		;
	}
	else if(dispL.type()==CV_32F)
	{
		dest.create(srcL.size(),CV_8UC3);
		destdisp.create(srcL.size(),CV_32F);
		Mat maskL(srcL.size(),CV_8U,Scalar(0));
		Mat maskR(srcL.size(),CV_8U,Scalar(0));
		Mat maskdest(srcL.size(),CV_8U,Scalar(0));

		shiftImDisp<float>(srcL,dispL,dest,destdisp,alpha,disp_amp,large_jump,maskL);
		/*
		Mat temp;
		destdisp.convertTo(temp,CV_8U,30);
		imshow("temp",temp);
		waitKey();
		*/

		//fillOcclusionImDisp<uchar>(dest,destdisp,invalidvalue);

		Mat destR(srcL.size(),CV_8UC3);
		Mat destdispR;
		shiftImDisp<float>(srcR,dispR,destR,destdispR,-(1.0-alpha),disp_amp,large_jump,maskR);
		imshow("aaaaa",destR);
		//fillOcclusionImDisp<uchar>(temp,destdisp,invalidvalue);

		blendLR<float>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);

		fillOcclusionImDisp(destR,destdispR);
		destdispR.copyTo(destdisp);
		destR.copyTo(dest);

		//depthBasedInpaint<float>(destR,destdispR,dest,destdisp,0.0);


	}
}
void shiftViewSynthesisLRFilter2(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp,int large_jump)
{
	if(alpha==0.0)
	{
		srcL.copyTo(dest);
		dispL.copyTo(destdisp);
		return;
	}
	else if(alpha==1.0)
	{
		srcR.copyTo(dest);
		dispR.copyTo(destdisp);
		return;
	}

	large_jump = large_jump<1 ?1:large_jump;
	if(dispL.type()==CV_8U)
	{
		if(dest.empty())dest.create(srcL.size(),CV_8UC3);
		else dest.setTo(0);

		if(destdisp.empty())destdisp.create(srcL.size(),CV_8U);
		else destdisp.setTo(0);

		Mat maskL(srcL.size(),CV_8U,Scalar(0));
		Mat maskL2(srcL.size(),CV_8U,Scalar(0));
		Mat maskR(srcL.size(),CV_8U,Scalar(0));
		Mat destR(srcL.size(),CV_8UC3);
		Mat destdispR(srcL.size(),CV_8U);
		Mat temp(srcL.size(),CV_8U);
		Mat temp2(srcL.size(),CV_8U);
		Mat swap(srcL.size(),CV_8UC3);

		{
			//CalcTime t("warp");

			shiftImDisp<uchar>(srcL,dispL,dest,temp,alpha*disp_amp,disp_amp,large_jump,maskL);
			//			fillOcclusion(temp);////////////
			filterDepth(temp,destdisp,maskL2,3,cvRound(abs(alpha)));
			Mat m;  compare(destdisp,0,m,cv::CMP_EQ);
			dest.setTo(0,m);
			maskL.setTo(0,m);
			maskL2.setTo(0,m);

			shiftImInv_<uchar>(srcL,destdisp,dest,-alpha*disp_amp,maskL2);
			fillOcclusionImDisp_<uchar>(dest, destdisp, 0,2);


			Mat dest2;
			boxFilter(dest,dest2,-1,Size(11,11));
			dest2.copyTo(dest,m);

			maskL=maskL+maskL2;
			temp.setTo(0);
			shiftImDisp<uchar>(srcR,dispR,destR,temp,-disp_amp*(1.0-alpha),disp_amp,large_jump,maskR);
			//	fillOcclusion(temp);////////
			filterDepth(temp,destdispR,maskL2,3,cvRound(abs(alpha)));
			Mat m2;  compare(destdispR,0,m2,cv::CMP_EQ);
			destR.setTo(0,m2);
			maskR.setTo(0,m2);
			maskL2.setTo(0,m2);
			shiftImInv_<uchar>(srcR,destdispR,destR,disp_amp*(1.0-alpha),maskL2);
			fillOcclusionImDisp_<uchar>(destR, destdispR, 0,2);

			boxFilter(destR,dest2,-1,Size(11,11));
			dest2.copyTo(destR,m2);

			maskR=maskR+maskL2;
		}

		{
			//	CalcTime t("blend");			

			blendLR<uchar>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);
			//guiAlphaBlend(maskL,maskR);
			//	imwrite("im.bmp",destR);
			//			imwrite("binp_dp.bmp",destdispR);
		}
		{	
			//CalcTime t("inpaint");
			//fillBoundingBoxDepthIm<uchar>(destR,destdispR,0);

			//depthBasedInpaint<uchar>(destR,destdispR,dest,destdisp,0);
			fillOcclusionImDisp(destR,destdispR,invalidvalue);
			destdispR.copyTo(destdisp);
			destR.copyTo(dest);
			//	imwrite("im_occ.bmp",dest);

		}
		Mat edge;
		cv::Canny(destdisp,edge,18,30);
		Mat a;
		GaussianBlur(dest,a,Size(3,3),5);
		a.copyTo(dest,edge);

	}
	else if(dispL.type()==CV_16S)
	{
		;
	}
	else if(dispL.type()==CV_32F)
	{
		dest.create(srcL.size(),CV_8UC3);
		destdisp.create(srcL.size(),CV_32F);
		Mat maskL(srcL.size(),CV_8U,Scalar(0));
		Mat maskR(srcL.size(),CV_8U,Scalar(0));
		Mat maskdest(srcL.size(),CV_8U,Scalar(0));

		shiftImDisp<float>(srcL,dispL,dest,destdisp,alpha,disp_amp,large_jump,maskL);
		/*
		Mat temp;
		destdisp.convertTo(temp,CV_8U,30);
		imshow("temp",temp);
		waitKey();
		*/

		//fillOcclusionImDisp<uchar>(dest,destdisp,invalidvalue);

		Mat destR(srcL.size(),CV_8UC3);
		Mat destdispR;
		shiftImDisp<float>(srcR,dispR,destR,destdispR,-(1.0-alpha),disp_amp,large_jump,maskR);
		imshow("aaaaa",destR);
		//fillOcclusionImDisp<uchar>(temp,destdisp,invalidvalue);

		blendLR<float>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);

		fillOcclusionImDisp(destR,destdispR);
		destdispR.copyTo(destdisp);
		destR.copyTo(dest);

		//depthBasedInpaint<float>(destR,destdispR,dest,destdisp,0.0);


	}
}
void shiftViewSynthesisLRFilter(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp,int large_jump)
{
	large_jump = large_jump<1 ?1:large_jump;

	if(dispL.type()==CV_8U)
	{
		if(dest.empty())dest.create(srcL.size(),CV_8UC3);
		else dest.setTo(0);

		if(destdisp.empty())destdisp.create(srcL.size(),CV_8U);
		else destdisp.setTo(0);

		Mat maskL(srcL.size(),CV_8U,Scalar(0));
		Mat maskL2(srcL.size(),CV_8U,Scalar(0));
		Mat maskR(srcL.size(),CV_8U,Scalar(0));
		Mat destR(srcL.size(),CV_8UC3);
		Mat destdispR(srcL.size(),CV_8U);
		Mat temp(srcL.size(),CV_8U);
		Mat temp2(srcL.size(),CV_8U);
		Mat swap(srcL.size(),CV_8UC3);

		{
			//CalcTime t("warp");

			shiftImDisp<uchar>(srcL,dispL,dest,temp,alpha*disp_amp,disp_amp,large_jump,maskL);
			//			fillOcclusion(temp);////////////
			filterDepth(temp,destdisp,maskL2,3,cvRound(abs(alpha)));
			Mat m;  compare(destdisp,0,m,cv::CMP_EQ);
			dest.setTo(0,m);
			maskL.setTo(0,m);
			maskL2.setTo(0,m);

			shiftImInv_<uchar>(srcL,destdisp,dest,-alpha*disp_amp,maskL2);
			fillOcclusionImDisp_<uchar>(dest, destdisp, 0,2);

			maskL=maskL+maskL2;
			temp.setTo(0);
			shiftImDisp<uchar>(srcR,dispR,destR,temp,-disp_amp*(1.0-alpha),disp_amp,large_jump,maskR);
			//	fillOcclusion(temp);////////
			filterDepth(temp,destdispR,maskL2,3,cvRound(abs(alpha)));
			Mat m2;  compare(destdispR,0,m2,cv::CMP_EQ);
			destR.setTo(0,m2);
			maskR.setTo(0,m2);
			maskL2.setTo(0,m2);
			shiftImInv_<uchar>(srcR,destdispR,destR,disp_amp*(1.0-alpha),maskL2);
			fillOcclusionImDisp_<uchar>(destR, destdispR, 0,2);

			maskR=maskR+maskL2;
		}

		{
			//	CalcTime t("blend");			

			blendLR<uchar>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);
			//guiAlphaBlend(maskL,maskR);
			//	imwrite("im.bmp",destR);
			//			imwrite("binp_dp.bmp",destdispR);
		}
		{	
			//CalcTime t("inpaint");
			//fillBoundingBoxDepthIm<uchar>(destR,destdispR,0);

			//depthBasedInpaint<uchar>(destR,destdispR,dest,destdisp,0);
			fillOcclusionImDisp(destR,destdispR,invalidvalue);
			destdispR.copyTo(destdisp);
			destR.copyTo(dest);
			//	imwrite("im_occ.bmp",dest);

		}
	}
	else if(dispL.type()==CV_16S)
	{
		;
	}
	else if(dispL.type()==CV_32F)
	{
		dest.create(srcL.size(),CV_8UC3);
		destdisp.create(srcL.size(),CV_32F);
		Mat maskL(srcL.size(),CV_8U,Scalar(0));
		Mat maskR(srcL.size(),CV_8U,Scalar(0));
		Mat maskdest(srcL.size(),CV_8U,Scalar(0));

		shiftImDisp<float>(srcL,dispL,dest,destdisp,alpha,disp_amp,large_jump,maskL);
		/*
		Mat temp;
		destdisp.convertTo(temp,CV_8U,30);
		imshow("temp",temp);
		waitKey();
		*/

		//fillOcclusionImDisp<uchar>(dest,destdisp,invalidvalue);

		Mat destR(srcL.size(),CV_8UC3);
		Mat destdispR;
		shiftImDisp<float>(srcR,dispR,destR,destdispR,-(1.0-alpha),disp_amp,large_jump,maskR);
		imshow("aaaaa",destR);
		//fillOcclusionImDisp<uchar>(temp,destdisp,invalidvalue);

		blendLR<float>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);

		fillOcclusionImDisp(destR,destdispR);
		destdispR.copyTo(destdisp);
		destR.copyTo(dest);

		//depthBasedInpaint<float>(destR,destdispR,dest,destdisp,0.0);


	}
}

void shiftViewSynthesisLR(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp)
{
	const int large_jump=3;

	if(dispL.type()==CV_8U)
	{

		dest.create(srcL.size(),CV_8UC3);
		destdisp.create(srcL.size(),CV_8U);
		Mat maskL(srcL.size(),CV_8U,Scalar(0));
		Mat maskL2(srcL.size(),CV_8U,Scalar(0));
		Mat maskR(srcL.size(),CV_8U,Scalar(0));
		Mat destR(srcL.size(),CV_8UC3);
		Mat destdispR(srcL.size(),CV_8U);
		Mat temp(srcL.size(),CV_8U);
		Mat temp2(srcL.size(),CV_8U);
		Mat swap(srcL.size(),CV_8UC3);

		{
			//CalcTime t("warp");
			shiftImDisp<uchar>(srcL,dispL,dest,destdisp,alpha*disp_amp,disp_amp,large_jump,maskL);
			shiftImDisp<uchar>(srcR,dispR,destR,destdispR,-disp_amp*(1.0-alpha),disp_amp,large_jump,maskR);

		}

		{
			//	CalcTime t("blend");			
			blendLR<uchar>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);
			//	imwrite("im.bmp",destR);
			//			imwrite("binp_dp.bmp",destdispR);
		}
		{	
			//CalcTime t("inpaint");
			//fillBoundingBoxDepthIm<uchar>(destR,destdispR,0);

			//depthBasedInpaint<uchar>(destR,destdispR,dest,destdisp,0);
			fillOcclusionImDisp(destR,destdispR,invalidvalue);
			destdispR.copyTo(destdisp);
			destR.copyTo(dest);
			/*
			Mat edge;
			cv::Canny(destdisp,edge,5,25);
			//dilate(edge,edge,Mat(),Point(-1,-1),2);
			Mat a;
			GaussianBlur(destR,a,Size(3,3),5);
			a.copyTo(dest,edge);
			*/
			//	imwrite("im_occ.bmp",dest);	
		}
	}
	else if(dispL.type()==CV_16S)
	{
		;
	}
	else if(dispL.type()==CV_32F)
	{
		dest.create(srcL.size(),CV_8UC3);
		destdisp.create(srcL.size(),CV_32F);
		Mat maskL(srcL.size(),CV_8U,Scalar(0));
		Mat maskR(srcL.size(),CV_8U,Scalar(0));
		Mat maskdest(srcL.size(),CV_8U,Scalar(0));

		shiftImDisp<float>(srcL,dispL,dest,destdisp,alpha,disp_amp,large_jump,maskL);
		/*
		Mat temp;
		destdisp.convertTo(temp,CV_8U,30);
		imshow("temp",temp);
		waitKey();
		*/

		//fillOcclusionImDisp<uchar>(dest,destdisp,invalidvalue);

		Mat destR(srcL.size(),CV_8UC3);
		Mat destdispR;
		shiftImDisp<float>(srcR,dispR,destR,destdispR,-(1.0-alpha),disp_amp,large_jump,maskR);
		imshow("aaaaa",destR);
		//fillOcclusionImDisp<uchar>(temp,destdisp,invalidvalue);

		blendLR<float>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);

		fillOcclusionImDisp(destR,destdispR);
		destdispR.copyTo(destdisp);
		destR.copyTo(dest);

		//depthBasedInpaint<float>(destR,destdispR,dest,destdisp,0.0);


	}
}

void shiftViewSynthesis(Mat& src, Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp)
{
	if(disp.type()==CV_8U)
	{
		shiftImDisp<uchar>(src,disp,dest,destdisp,alpha*disp_amp,disp_amp);

		/*
		fillBoundingBoxDepthIm<uchar>(dest,destdisp,0);
		Mat temp;
		Mat temp2;
		depthBasedInpaint<uchar>(dest,destdisp,temp,temp2,0);

		temp.copyTo(dest);
		temp2.copyTo(destdisp);*/


		fillOcclusionImDisp(dest,destdisp,invalidvalue);
	}
	else if(disp.type()==CV_16S)
	{
		shiftImDisp<short>(src,disp,dest,destdisp,alpha*disp_amp,disp_amp);
		fillOcclusionImDisp(dest,destdisp,invalidvalue);
	}
	else if(disp.type()==CV_32F)
	{
		shiftImDisp<float>(src,disp,dest,destdisp,alpha*disp_amp, disp_amp);
		fillOcclusionImDisp(dest,destdisp,invalidvalue);
	}
}

void shiftViewSynthesisFilter(Mat& src, Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp,int large_jump, int occlusionMode, int occBlurD, Mat& mask)
{
	large_jump = large_jump<1 ?1:large_jump;
	if(alpha==0.0)
	{
		src.copyTo(dest);
		dest.copyTo(destdisp);
		return;
	}

	if(disp.type()==CV_8U)
	{
		Mat temp2;
		Mat tempp;

		shiftImDisp<uchar>(src,disp,dest,tempp,alpha/disp_amp,disp_amp,large_jump,mask);
		//shiftImDisp<uchar>(src,disp,dest,tempp,alpha/disp_amp,disp_amp,large_jump,mask,WARP_LINEAR);
		mask.setTo(0);
		filterDepth(tempp,destdisp,mask,3,cvRound(abs(alpha/disp_amp)));
		shiftImInv_<uchar>(src,destdisp,dest,-alpha/disp_amp,mask,0);
		//shiftImInv_<uchar>(src,destdisp,dest,-alpha/disp_amp,mask,0,WARP_NN);

		compare(destdisp,0,mask,cv::CMP_EQ);
		fillOcclusionImDisp(dest,destdisp,invalidvalue,occlusionMode);

		boxFilter(dest,temp2,-1,Size(occBlurD,occBlurD));
		temp2.copyTo(dest,mask);
	}
	else if(disp.type()==CV_16S)
	{
		Mat temp2;
		Mat tempp;

		shiftImDisp<short>(src,disp,dest,tempp,alpha/disp_amp,disp_amp,large_jump,mask);
		//shiftImDisp<short>(src,disp,dest,tempp,alpha/disp_amp,disp_amp,large_jump,mask,WARP_CUBIC);

		mask.setTo(0);
		filterDepth(tempp,destdisp,mask,3,cvRound(abs(alpha/disp_amp)));
		shiftImInv_<short>(src,destdisp,dest,-alpha/disp_amp,mask);
		//shiftImInv_<short>(src,destdisp,dest,-alpha/disp_amp,mask,0,WARP_LINEAR);

		compare(destdisp,0,mask,cv::CMP_EQ);
		fillOcclusionImDisp(dest,destdisp,invalidvalue,occlusionMode);

		boxFilter(dest,temp2,-1,Size(occBlurD,occBlurD));
		temp2.copyTo(dest,mask);
	}
	else if(disp.type()==CV_16U)
	{
		Mat temp2;
		Mat tempp;

		shiftImDisp<unsigned short>(src,disp,dest,tempp,alpha/disp_amp,disp_amp,large_jump,mask);
		mask.setTo(0);
		filterDepth(tempp,destdisp,mask,3,cvRound(abs(alpha/disp_amp)));
		shiftImInv_<unsigned short>(src,destdisp,dest,-alpha/disp_amp,mask);

		compare(destdisp,0,mask,cv::CMP_EQ);
		fillOcclusionImDisp(dest,destdisp,invalidvalue,occlusionMode);

		boxFilter(dest,temp2,-1,Size(occBlurD,occBlurD));
		temp2.copyTo(dest,mask);
	}
	else if(disp.type()==CV_32F)
	{
		shiftImDisp<float>(src,disp,dest,destdisp,alpha/disp_amp,disp_amp);
		fillOcclusionImDisp(dest,destdisp,invalidvalue);
	}
}

void StereoViewSynthesis::depthfilter(Mat& depth, Mat& depth2,Mat& mask, int viewstep, double disp_amp)
{
	//depth.copyTo(depth2);
	
	//depth.copyTo(depth2);mask2=Mat::zeros(depth.size(),CV_8U);return;
	//crackRemove(depth,depth2,mask2);
	//guiAlphaBlend(depth2,mask2);
	medianBlur(depth,depth2,3);

	if(viewstep>0)
	{
		maxFilter(depth2,depth2,Size(2*viewstep+1,1));
		minFilter(depth2,depth2,Size(2*viewstep+1,1));
	}

	//medianBlur(depth2,depth2,3);
	
	Mat temp;
	depth2.convertTo(temp,CV_16S);
	//cout<<"window, range: "<<warpedSpeckesWindow<<","<<warpedSpeckesRange<<endl;
	filterSpeckles(temp,0,warpedSpeckesWindow,(int)(warpedSpeckesRange*disp_amp));
	temp.convertTo(depth2,depth.type());
	
	//imshow("depth",depth2)
	
	compare(depth,depth2,mask,cv::CMP_NE);
	
}

void StereoViewSynthesis::depthfilter2(Mat& depth, Mat& depth2,Mat& mask, int viewstep, double disp_amp)
{
	//depth.copyTo(depth2);
	
	//depth.copyTo(depth2);mask2=Mat::zeros(depth.size(),CV_8U);return;
	//crackRemove(depth,depth2,mask2);
	//guiAlphaBlend(depth2,mask2);
	medianBlur(depth,depth2,3);

	if(viewstep>0)
	{
		maxFilter(depth2,depth2,Size(2*viewstep+1,3));
		minFilter(depth2,depth2,Size(2*viewstep+1,3));
	}

	//medianBlur(depth2,depth2,3);
	/*
	Mat temp;
	depth2.convertTo(temp,CV_16S);
	//cout<<"window, range: "<<warpedSpeckesWindow<<","<<warpedSpeckesRange<<endl;
	filterSpeckles(temp,0,warpedSpeckesWindow,(int)(warpedSpeckesRange*disp_amp));
	temp.convertTo(depth2,depth.type());
	*/
	//imshow("depth",depth2)
	
	compare(depth,depth2,mask,cv::CMP_NE);
	
}

StereoViewSynthesis::StereoViewSynthesis()
{
	depthfiltermode=0;
	isPostFilter=1;
	large_jump = 100;
	warpedMedianKernel = 3;
	warpedSpeckesWindow=100;
	warpedSpeckesRange=1;

	occutionBlurSize = Size(3,3);

	canny_t1=18;
	canny_t2=30;
	boundaryKernelSize = Size(3,3);
	boundarySigma = 3.0;
}
void StereoViewSynthesis::check(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, Mat& ref)
{
	if(ref.empty())ref= Mat::zeros(srcL.size(),srcL.type());
	string wname = "Stereo ViewSynthesis";
	namedWindow(wname);

	createTrackbar("l jump",wname,&large_jump,500);
	createTrackbar("w med",wname,&warpedMedianKernel,5);
	createTrackbar("sp window",wname,&warpedSpeckesWindow,1024);
	createTrackbar("sp range",wname,&warpedSpeckesRange,255);

	int occb=1;
	createTrackbar("occblur",wname,&occb,30);

	createTrackbar("canny t1",wname,&canny_t1,255);
	createTrackbar("canny t2",wname,&canny_t2,255);

	int boundk=1;
	createTrackbar("boundk",wname,&boundk,30);
	int bounds = 30;
	createTrackbar("bounds",wname,&bounds,30);
	int bb=0;
	createTrackbar("psnrbb",wname,&bb,100);

	Mat dshow;
	Mat show;
	int key = 0;
	while(key!='q')
	{

		occutionBlurSize=Size(2*occb+1,2*occb+1);
		boundaryKernelSize = Size(2*boundk+1,2*boundk+1);
		boundarySigma=bounds/10.0;

		{
			CalcTime t("Stereo VS");
			this->operator()(srcL,srcR,dispL,dispR, dest, destdisp, alpha, invalidvalue, disp_amp);
			//alphaSynth(srcL,srcR,dispL,dispR, dest, destdisp, alpha, invalidvalue, disp_amp);
		}
		/*
		double minv,maxv;
		minMaxLoc(dst,&minv,&maxv);
		cout<<format("%f %f\n",minv,maxv);
		int minDisparity=(int)(minv+0.5);
		int numberOfDisparities=(int)(maxv-minv+0.5);
		cvtDisparityColor(dst,dshow,minDisparity,numberOfDisparities,isColor,1);

		 addWeightedOMP(joint,1.0-(alpha/100.0),dshow,(alpha/100.0),0.0,show);
		*/
		imshow(wname,dest);
		key = waitKey(1);
	}
}

void StereoViewSynthesis::check(Mat& src, Mat& disp,Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, Mat& ref)
{
	if(ref.empty())ref= Mat::zeros(src.size(),src.type());

	string wname = "Single ViewSynthesis";
	namedWindow(wname);
	int alphav=0;
	createTrackbar("alpha",wname,&alphav,100);

	createTrackbar("l jump",wname,&large_jump,50);
	createTrackbar("w med",wname,&warpedMedianKernel,5);
	createTrackbar("sp window",wname,&warpedSpeckesWindow,1024);
	createTrackbar("sp range",wname,&warpedSpeckesRange,255);

	int occb=1;
	createTrackbar("occblur",wname,&occb,30);

	createTrackbar("canny t1",wname,&canny_t1,255);
	createTrackbar("canny t2",wname,&canny_t2,255);

	int boundk=1;
	createTrackbar("boundk",wname,&boundk,30);
	int bounds = 30;
	createTrackbar("bounds",wname,&bounds,30);
	int bb=0;
	createTrackbar("psnrbb",wname,&bb,100);

	int maxk=1;
	createTrackbar("maxK",wname,&maxk,10);
	Mat dshow;
	Mat show;
	int key = 0;
	Mat disp2;
	disp.copyTo(disp2);
	
	double minv,maxv;
	minMaxLoc(disp,&minv,&maxv);
	double damp = 255.0/maxv;
	while(key!='q')
	{
		disp2.copyTo(disp);
		
		maxFilter(disp,disp,Size(2*maxk+1,2*maxk+1));

		occutionBlurSize=Size(2*occb+1,2*occb+1);
		boundaryKernelSize = Size(2*boundk+1,2*boundk+1);
		boundarySigma=bounds/10.0;

		{
			CalcTime t("Single VS");
			this->operator()(src,disp, dest, destdisp, alpha, invalidvalue, disp_amp);
			//alphaSynth(srcL,srcR,dispL,dispR, dest, destdisp, alpha, invalidvalue, disp_amp);
		}
		//cout<<"PSNR:"<<calcPSNRBB(ref,dest,bb,bb)<<endl;;
		/*
		double minv,maxv;
		minMaxLoc(dst,&minv,&maxv);
		cout<<format("%f %f\n",minv,maxv);
		int minDisparity=(int)(minv+0.5);
		int numberOfDisparities=(int)(maxv-minv+0.5);
		cvtDisparityColor(dst,dshow,minDisparity,numberOfDisparities,isColor,1);

		 addWeightedOMP(joint,1.0-(alpha/100.0),dshow,(alpha/100.0),0.0,show);
		*/
		destdisp.convertTo(dshow,CV_8U,damp);
		alphaBlend(dest,dshow,alphav/100.0,dest);
		imshow(wname,dest);
		key = waitKey(1);
	}
}
void StereoViewSynthesis::noFilter(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp)
{
	;
}
void StereoViewSynthesis::alphaSynth(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp)
{
	if(alpha==0.0)
	{
		srcL.copyTo(dest);
		dispL.copyTo(destdisp);
		return;
	}
	else if(alpha==1.0)
	{
		srcR.copyTo(dest);
		dispR.copyTo(destdisp);
		return;
	}

	large_jump = large_jump<1 ?1:large_jump;

	if(dest.empty())dest.create(srcL.size(),CV_8UC3);
	else dest.setTo(0);

	if(destdisp.empty())destdisp.create(srcL.size(),CV_8U);
	else destdisp.setTo(0);

	Mat maskL(srcL.size(),CV_8U,Scalar(0));
	Mat maskL2(srcL.size(),CV_8U,Scalar(0));
	Mat maskR(srcL.size(),CV_8U,Scalar(0));
	Mat destR(srcL.size(),CV_8UC3);
	Mat destdispR(srcL.size(),CV_8U);
	Mat temp(srcL.size(),CV_8U);
	Mat temp2(srcL.size(),CV_8U);
	Mat swap(srcL.size(),CV_8UC3);
	{
		//CalcTime t("warp");

		shiftImDisp<uchar>(srcL,dispL,dest,temp,alpha*disp_amp,disp_amp,large_jump,maskL);
		//			fillOcclusion(temp);////////////
		depthfilter(temp,destdisp,maskL2,cvRound(abs(alpha)),disp_amp);
		Mat m;  compare(destdisp,0,m,cv::CMP_EQ);
		dest.setTo(0,m);
		maskL.setTo(0,m);
		maskL2.setTo(0,m);

		shiftImInv_<uchar>(srcL,destdisp,dest,-alpha*disp_amp,maskL2);
		maskL=maskL+maskL2;

		temp.setTo(0);
		shiftImDisp<uchar>(srcR,dispR,destR,temp,-disp_amp*(1.0-alpha),disp_amp,large_jump,maskR);

		depthfilter(temp,destdispR,maskL2,cvRound(abs(alpha)),disp_amp);
		//filterDepth(temp,destdispR,maskL2);
		Mat m2;  compare(destdispR,0,m2,cv::CMP_EQ);
		destR.setTo(0,m2);
		maskR.setTo(0,m2);
		maskL2.setTo(0,m2);
		shiftImInv_<uchar>(srcR,destdispR,destR,disp_amp*(1.0-alpha),maskL2);
		filterSpeckles(destR,0,255,255);

		maskR=maskR+maskL2;
	}

	{
		//	CalcTime t("blend");			
		blendLR<uchar>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);
		//guiAlphaBlend(maskL,maskR);
		//	imwrite("im.bmp",destR);
		//			imwrite("binp_dp.bmp",destdispR);
	}
	if(isPostFilter)
	{
		{	
			//CalcTime t("inpaint");
			//fillBoundingBoxDepthIm<uchar>(destR,destdispR,0);

			//depthBasedInpaint<uchar>(destR,destdispR,dest,destdisp,0);
			Mat m;  compare(destdispR,0,m,cv::CMP_EQ);
			fillOcclusionImDisp(destR,destdispR,invalidvalue,FILL_OCCLUSION_LINE);
			destdispR.copyTo(destdisp);
			destR.copyTo(dest);

			Mat dest2;
			boxFilter(dest,dest2,-1,occutionBlurSize);
			dest2.copyTo(dest,m);
			//	imwrite("im_occ.bmp",dest);
		}
		Mat edge;
		cv::Canny(destdisp,edge,canny_t1,canny_t2);
		//imshow("ee",edge);
		//dilate(edge,edge,Mat(),Point(-1,-1),2);
		Mat a;
		GaussianBlur(dest,a,boundaryKernelSize, boundarySigma);
		a.copyTo(destR,edge);
		double aa = (alpha>1.0) ? 1.0:alpha;
		aa = (alpha<0.0) ? 0.0:aa;
		aa = (0.5-abs(aa-0.5))*2.0;
		addWeighted(dest, 1.0-aa,destR,aa,0.0,dest);
	}
	else
	{
		destdispR.copyTo(destdisp);
		destR.copyTo(dest);
	}
}
void showMatInfo(const Mat& dest)
{
	cout<<"cols, rows, channel, depth"<<endl;
	cout<<dest.cols<<","<<dest.rows<<","<<dest.channels()<<","<<dest.depth()<<endl;
}
Mat disp8Ubuff;
Mat edge;
template <class T>
void StereoViewSynthesis::viewsynth(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype)
{
	if(alpha==0.0)
	{
		srcL.copyTo(dest);
		dispL.copyTo(destdisp);
		return;
	}
	else if(alpha==1.0)
	{
		srcR.copyTo(dest);
		dispR.copyTo(destdisp);
		return;
	}

	large_jump = large_jump<1 ?1:large_jump;

	if(dest.empty())dest.create(srcL.size(),CV_8UC3);
	else dest.setTo(0);

	if(destdisp.empty() ||destdisp.type() != disptype)destdisp.create(srcL.size(),disptype);
	else destdisp.setTo(0);

	Mat maskL(srcL.size(),CV_8U,Scalar(0));
	Mat maskL2(srcL.size(),CV_8U,Scalar(0));
	Mat maskR(srcL.size(),CV_8U,Scalar(0));
	Mat destR(srcL.size(),CV_8UC3);
	Mat destdispR(srcL.size(),disptype);
	Mat temp(srcL.size(),disptype);
	Mat temp2(srcL.size(),disptype);
	Mat swap(srcL.size(),CV_8UC3);
	{

		//CalcTime t("warp");
		shiftImDisp<T>(srcL,dispL,dest,temp,alpha/disp_amp,disp_amp,large_jump,maskL);
		//			fillOcclusion(temp);////////////

		depthfilter(temp,destdisp,maskL2,cvRound(abs(alpha)),disp_amp);
		Mat m;  compare(destdisp,0,m,cv::CMP_EQ);
		dest.setTo(0,m);
		maskL.setTo(0,m);
		maskL2.setTo(0,m);
		shiftImInv_<T>(srcL,destdisp,dest,-alpha/disp_amp,maskL2);

		maskL=maskL+maskL2;

		temp.setTo(0);
		shiftImDisp<T>(srcR,dispR,destR,temp,(alpha-1.0)/disp_amp,disp_amp,large_jump,maskR);
		depthfilter(temp,destdispR,maskL2,cvRound(abs(alpha)),disp_amp);
		Mat m2;  compare(destdispR,0,m2,cv::CMP_EQ);
		destR.setTo(0,m2);
		maskR.setTo(0,m2);
		maskL2.setTo(0,m2);
		shiftImInv_<T>(srcR,destdispR,destR,(1.0-alpha)/disp_amp,maskL2);

		maskR=maskR+maskL2;
	}

	{
		//	CalcTime t("blend");
		//destdisp.convertTo(temp,CV_8U,5.0/disp_amp);imshow("b",temp);//waitKey();
		//cout<<norm(destdisp)/disp_amp<<" : "<<norm(destdispR)/disp_amp<<" : "<<norm(maskL)<<" : "<<norm(maskR)<<endl;

		blendLR<T>(dest,destR,destdisp,destdispR,destR,destdispR,maskL,maskR,alpha);
		//destdispR.convertTo(temp,CV_8U,5.0/disp_amp);imshow("a",temp);//waitKey();

	}
	if(isPostFilter)
	{
		//CalcTime t("inpaint");
		//fillBoundingBoxDepthIm<uchar>(destR,destdispR,0);

		//depthBasedInpaint<uchar>(destR,destdispR,dest,destdisp,0);
		Mat m;  compare(destdispR,0,m,cv::CMP_EQ);
		fillOcclusionImDisp(destR,destdispR,invalidvalue, FILL_OCCLUSION_LINE);
		destdispR.copyTo(destdisp);
		destR.copyTo(dest);


		Mat dest2;
		boxFilter(dest,dest2,-1,occutionBlurSize);
		dest2.copyTo(dest,m);

		destdisp.convertTo(disp8Ubuff,CV_8U,1.0/disp_amp);
		cv::Canny(disp8Ubuff,edge,canny_t1,canny_t2);
		//imshow("ee",edge);//waitKey();
		//dilate(edge,edge,Mat(),Point(-1,-1),2);
		Mat a;
		GaussianBlur(dest,a,boundaryKernelSize, boundarySigma);
		a.copyTo(destR,edge);
		double aa = (alpha>1.0) ? 1.0:alpha;
		aa = (alpha<0.0) ? 0.0:aa;
		aa = (0.5-abs(aa-0.5))*2.0;
		addWeighted(dest, 1.0-aa,destR,aa,0.0,dest);
	}
	else
	{
		destdispR.copyTo(destdisp);
		destR.copyTo(dest);
	}
}

template <class T>
void StereoViewSynthesis::viewsynthSingle(Mat& src,Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype)
{
	if(alpha==0.0)
	{
		src.copyTo(dest);
		disp.copyTo(destdisp);
		return;
	}

	large_jump = large_jump<1 ?1:large_jump;

	if(dest.empty())dest.create(src.size(),CV_8UC3);
	else dest.setTo(0);

	if(destdisp.empty() ||destdisp.type() != disptype)destdisp.create(src.size(),disptype);
	else destdisp.setTo(0);


	Mat mask(src.size(),CV_8U);
	Mat destR(src.size(),CV_8UC3);
	Mat destdispR(src.size(),disptype);
	Mat temp(src.size(),disptype);


	{
		//CalcTime t("warp");
		shiftImDisp<T>(src,disp,dest,temp,alpha/disp_amp,disp_amp,large_jump);

		if(depthfiltermode==0)
			depthfilter(temp,destdisp,mask,cvRound(abs(alpha)),disp_amp);
		else if(depthfiltermode==1)
			depthfilter2(temp,destdisp,mask,cvRound(abs(alpha)),disp_amp);

		shiftImInv_<T>(src,destdisp,dest,-alpha/disp_amp,mask);
		//destdisp.convertTo(temp,CV_8U,5.0/disp_amp);imshow("a",temp);//waitKey();
	}
	if(isPostFilter)
	{
		//CalcTime t("inpaint");
		//fillBoundingBoxDepthIm<uchar>(destR,destdispR,0);
		//depthBasedInpaint<uchar>(destR,destdispR,dest,destdisp,0);
		compare(destdisp,0,diskMask,cv::CMP_EQ);
		
		fillOcclusionImDisp(dest,destdisp,invalidvalue, FILL_OCCLUSION_LINE);

		//imshow("ee",dest);waitKey();
		boxFilter(dest,destR,-1,occutionBlurSize);
		destR.copyTo(dest,diskMask);

		destdisp.convertTo(disp8Ubuff,CV_8U,1.0/disp_amp);
		cv::Canny(disp8Ubuff,edge,canny_t1,canny_t2);
		//imshow("ee",edge);waitKey();
		//dilate(edge,edge,Mat(),Point(-1,-1),2);
		Mat a;
		dest.copyTo(destR);
		GaussianBlur(dest,a,boundaryKernelSize, boundarySigma);
		a.copyTo(dest,edge);

		//imshow("dest",destR);waitKey();

		double aa = (alpha>1.0) ? 1.0:alpha;
		aa = (alpha<0.0) ? 0.0:aa;
		aa = (0.5-abs(aa-0.5))*2.0;
		addWeighted(dest, aa,destR,1.0-aa,0.0,dest);
	}
}

void StereoViewSynthesis::operator()(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp)
{
	int type = dispL.type();
	if(type == CV_8U)
	{
		viewsynth<uchar>(srcL,srcR,dispL,dispR,dest,destdisp,alpha,invalidvalue,disp_amp,CV_8U);
	}
	if(type == CV_16S)
	{
		viewsynth<short>(srcL,srcR,dispL,dispR,dest,destdisp,alpha,invalidvalue,disp_amp,CV_16S);
	}
	if(type == CV_16U)
	{
		viewsynth<ushort>(srcL,srcR,dispL,dispR,dest,destdisp,alpha,invalidvalue,disp_amp,CV_16S);
	}
}
void StereoViewSynthesis::operator()(Mat& src,Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp)
{
	int type = disp.type();
	if(type == CV_8U)
	{
		viewsynthSingle<uchar>(src,disp,dest,destdisp,alpha,invalidvalue,disp_amp,CV_8U);
	}
	if(type == CV_16S)
	{
		viewsynthSingle<short>(src,disp,dest,destdisp,alpha,invalidvalue,disp_amp,CV_16S);
	}
	if(type == CV_16U)
	{
		viewsynthSingle<ushort>(src,disp,dest,destdisp,alpha,invalidvalue,disp_amp,CV_16S);
	}
}

void StereoViewSynthesis::preview(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR,int invalidvalue, double disp_amp)
{
	string wname = "synth";
	namedWindow(wname);
	int x = 500;
	createTrackbar("x",wname,&x,900);
	Mat dest,destdisp;
	int key = 0;
	while(key!='q')
	{
		double vp = (x-450)/100.0;
		operator()(srcL,srcR,dispL,dispR,dest,destdisp,vp,invalidvalue,disp_amp);
		imshow(wname,dest);
		key = waitKey(1);
	}
	destroyWindow(wname);
}

void StereoViewSynthesis::preview(Mat& src,Mat& disp,int invalidvalue, double disp_amp)
{
	string wname = "synth";
	namedWindow(wname);
	static int x = 1000;
	createTrackbar("x",wname,&x,2000);

	static int alpha = 0;createTrackbar("alpha",wname,&alpha,100);
	createTrackbar("mode",wname,&depthfiltermode,1);
	static int maxr=0;
	createTrackbar("max",wname,&maxr,10);
	Mat dest,destdisp;
	int key = 0;
	
	double maxv,minv;
	minMaxLoc(disp,&minv,&maxv);

	while(key!='q')
	{
		double vp = (x-1000)/100.0;
		Mat disp2;
		maxFilter(disp,disp2,Size(2*maxr+1,2*maxr+1));
		operator()(src,disp2,dest,destdisp,vp,invalidvalue,disp_amp);
		//shiftViewSynthesisFilter(src,disp,dest,destdisp,vp,0,1.0/disp_amp);
		
		Mat dshow;
		Mat dtemp;Mat(destdisp*255.0/maxv).convertTo(dtemp,CV_8U);
		applyColorMap(dtemp,dshow,2);
		
		alphaBlend(dshow,dest,alpha/100.0,dest);
		imshow(wname,dest);
		//imshowDisparity("disp",destdisp,2,0,48,(int)disp_amp);
		key = waitKey(1);
	}
	destroyWindow(wname);
}