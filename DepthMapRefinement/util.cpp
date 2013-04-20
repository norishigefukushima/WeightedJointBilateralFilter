#include "util.h"

double getPSNR(Mat& src1, Mat& src2)
{
	CV_Assert(src1.channels()==src2.channels() && src1.type()==src2.type() && src1.data !=src2.data);
	double psnr;

	Mat s1,s2;
	if(src1.channels()==3)
	{
		cvtColor(src1,s1,CV_BGR2GRAY);
		cvtColor(src2,s2,CV_BGR2GRAY);
	}
	else
	{
		s1=src1;
		s2=src2;
	}

	//cout<<s1.cols<<","<<s1.rows<<endl;
	//cout<<s2.cols<<","<<s2.rows<<endl;
	Mat sub;
	subtract(s1,s2,sub,Mat(),CV_32F);
	multiply(sub,sub,sub);

	int count = s1.size().area();
	Scalar v = cv::mean(sub);

	if(v.val[0] == 0.0 || count==0)
	{
		return -1;
	}
	else
	{
		psnr = 10.0*log10((255.0*255.0)/v.val[0]);
		return psnr;
	}
}


void CalcTime::start()
{
	pre = getTickCount();
}

void CalcTime::restart()
{
	start();
}

void CalcTime:: show()
{
	getTime();
	switch(timeMode)
	{
	case TIME_NSEC:
		cout<< mes<< ": "<<cTime<<" nsec"<<endl;
		break;
	case TIME_SEC:
		cout<< mes<< ": "<<cTime<<" sec"<<endl;
		break;
	case TIME_MIN:
		cout<< mes<< ": "<<cTime<<" minute"<<endl;
		break;
	case TIME_HOUR:
		cout<< mes<< ": "<<cTime<<" hour"<<endl;
		break;

	case TIME_MSEC:
	default:
		cout<<mes<< ": "<<cTime<<" msec"<<endl;
		break;
	}
}

void CalcTime:: show(string mes)
{
	getTime();
	switch(timeMode)
	{
	case TIME_NSEC:
		cout<< mes<< ": "<<cTime<<" nsec"<<endl;
		break;
	case TIME_SEC:
		cout<< mes<< ": "<<cTime<<" sec"<<endl;
		break;
	case TIME_MIN:
		cout<< mes<< ": "<<cTime<<" minute"<<endl;
		break;
	case TIME_HOUR:
		cout<< mes<< ": "<<cTime<<" hour"<<endl;
		break;

	case TIME_MSEC:
	default:
		cout<<mes<< ": "<<cTime<<" msec"<<endl;
		break;
	}
}

double CalcTime:: getTime()
{
	cTime = (getTickCount()-pre)/(getTickFrequency());
	switch(timeMode)
	{
	case TIME_NSEC:
		cTime*=1000000.0;
		break;
	case TIME_SEC:
		cTime*=1.0;
		break;
	case TIME_MIN:
		cTime /=(60.0);
		break;
	case TIME_HOUR:
		cTime /=(60*60);
		break;
	case TIME_MSEC:
	default:
		cTime *=1000.0;
		break;
	}
	return cTime;
}
void CalcTime:: setMessage(string src)
{
	mes=src;
}
void CalcTime:: setMode(int mode)
{
	timeMode = mode;
}
CalcTime::CalcTime(string message,int mode,bool isShow)
{
	_isShow = isShow;
	timeMode = mode;

	setMessage(message);
	start();
}
CalcTime::~CalcTime()
{
	getTime();
	if(_isShow)	show();
}

ConsoleImage::ConsoleImage(Size size)
{
	show = Mat::zeros(size, CV_8UC3);
	clear();
}
ConsoleImage::~ConsoleImage()
{
	printData();
}
void ConsoleImage::printData()
{
	for(int i=0;i<(int)strings.size();i++)
	{
		cout<<strings[i]<<endl;
	}
}
void ConsoleImage::clear()
{
	count = 0;
	show.setTo(0);
	strings.clear();
}

void ConsoleImage::operator()(string src)
{
	
	//CvFont font = fontQt("Times",16,CV_RGB(255,255,255));
	strings.push_back(src);
	//xcvPutText(&IplImage(show),(char*)src.c_str(),Point(20,20+count*20),CV_RGB(255,255,255),1,0,CV_FONT_HERSHEY_COMPLEX_SMALL);
	//addText(show,buff,Point(20,20+count*20),font);
	cv::putText(show,src,Point(20,20+count*20),CV_FONT_HERSHEY_COMPLEX_SMALL,1.0,CV_RGB(255,255,255),1);
	count++;
}
void ConsoleImage::operator()(const char *format, ...)
{
	char buff[255]; 

	va_list ap;
	va_start(ap, format);
	vsprintf(buff, format, ap);
	va_end(ap);

	string a = buff;
	//CvFont font = fontQt("Times",16,CV_RGB(255,255,255));
	strings.push_back(a);
	//xcvPutText(&IplImage(show),buff,Point(20,20+count*20),CV_RGB(255,255,255),1,0,CV_FONT_HERSHEY_COMPLEX_SMALL);
	cv::putText(show,buff,Point(20,20+count*20),CV_FONT_HERSHEY_COMPLEX_SMALL,1.0,CV_RGB(255,255,255),1);
	//addText(show,buff,Point(20,20+count*20),font);
	count++;
}

void ConsoleImage::operator()(cv::Scalar color, const char *format, ...)
{
	char buff[255]; 

	va_list ap;
	va_start(ap, format);
	vsprintf(buff, format, ap);
	va_end(ap);

	string a = buff;
	//CvFont font = fontQt("Times",16,CV_RGB(255,255,255));
	strings.push_back(a);
	//xcvPutText(&IplImage(show),buff,Point(20,20+count*20),color,1,0,CV_FONT_HERSHEY_COMPLEX_SMALL);
	cv::putText(show,buff,Point(20,20+count*20),CV_FONT_HERSHEY_COMPLEX_SMALL,1.0,CV_RGB(255,255,255),1);
	//addText(show,buff,Point(20,20+count*20),font);
	count++;
}
