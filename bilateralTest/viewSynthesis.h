#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

enum
{
	FILL_OCCLUSION_LINE = 0,
	FILL_OCCLUSION_REFLECT = 1,
	FILL_OCCLUSION_STRETCH = -1
};

void shiftViewSynthesisFilter(Mat& src, Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp,int large_jump, int occlusionMode, int occBlurD, Mat& mask);

void fillOcclusionImDisp(cv::Mat& im, cv::Mat& disp,int invalidvalue=0, int mode=FILL_OCCLUSION_LINE);
void shiftDisparity(Mat& srcdisp, Mat& destdisp, double amp);
class StereoViewSynthesis
{
	void depthfilter2(Mat& depth, Mat& depth2,Mat& mask2,int viewstep,double disp_amp);
	void depthfilter(Mat& depth, Mat& depth2,Mat& mask2,int viewstep,double disp_amp);
public:
	int depthfiltermode;
	Mat diskMask;
	int isPostFilter;
	int warpedMedianKernel;
	int warpedSpeckesWindow;
	int warpedSpeckesRange;
	int large_jump;
	int canny_t1;
	int canny_t2;
	Size occutionBlurSize;
	Size boundaryKernelSize;
	double boundarySigma;
	StereoViewSynthesis();

	template <class T>
	void viewsynth (Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype);
	void operator()(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp);

	template <class T>
	void viewsynthSingle(Mat& src,Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype);

	void operator()(Mat& src,Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp);

	void alphaSynth(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp);

	void noFilter(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp);
	void check(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, Mat& ref);

	void check(Mat& src,Mat& disp,Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, Mat& ref);

	void preview(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR,int invalidvalue, double disp_amp);
	void preview(Mat& src, Mat& disp,int invalidvalue, double disp_amp);
};