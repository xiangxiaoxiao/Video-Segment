#include <math.h> 
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <assert.h>
#include <math.h>
#include<cmath>
#include<time.h>

// Opencv header and namespace
#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <direct.h>

#include "sss_core.h"
#include "tvsegmentbinary.h"
#include "bcv_sparse_op.h"
#include "segmentation_utils.h"
#include "vis_utils.h"
#include "bcv_alg.h"
#include "bcv_basic.h"
#include "bcv_kmeans.h"
#include "SlicGraph.h"
#include "sss_core.h"

typedef unsigned char BYTE;

class GMMdata {
public:
	int K = 0;
	int dim = 0;
	vector<float> mu;
	vector<float> pi;
	vector<float> cov;

	void initView();
	int getFreeId();
	void setPi(vector<float> data);
	void setMu(vector<float> mu);
	void clear();
	void clearLabel();
	void setLabel() ; 
private:
	vector<int> viewid;
	const int TOTAL_WIDTH = 800;
	const int DIM = 3;
};
class Video_Segmention
{
public:
	//设置窗口大小，确认上下点坐标
	int m_ObjRectX = 500;
	int m_ObjRectY = 350;
	int m_ObjRectWidth = 800;
	int m_ObjRectHeight = 500;
	Point node = Point(100, 100);
	Point center = Point(900, 600);
	//视频尺寸及相关参数
	int m_height;
	int m_width ;
	int m_channel;
	int m_fps;
	int m_frames;
	int m_nPixelAcount;

	//Slic 参数
	int SLIC_K = 400;
	int SLIC_M = 2;
	int SLIC_NUM_ITERS = 1;
	bool SLIC_SHOW_BOUNDARY = false;
	bool SLIC_SHOW_GRAPH = false;
	int SEGMENTATION_K = 3; // number of clusters
	int SEGMENTATION_NUM_ITERS = 50;
	float SEGMENTATION_BETA = 20.0f;
	float SEGMENTATION_TV_WEIGHT = 50.0f;
	float SEGMENTATION_WT_WEIGHT = 1.0f;

	GMMdata *GMMfg;
	GMMdata *GMMbg;
	jni_state* mJniObjectAddr;
	BYTE* mDataRgb ;

	void Video_Seg(string video_path );
	void nativeCreateJniObject(jni_state *mJniObjectAddr,
		int slic_K, int slic_M,int slic_num_iters, int seg_K,
		float seg_tv, float seg_wt, float seg_beta, int seg_num_iters,
		int rows, int cols, int chan);//创建对象 初始化参数
	void nativeRunTSlic(BYTE *data);
	void nativeTSlicConstructGraph(BYTE* data, int chan);
	void nativeLearnAddFeatures(int m_ObjRectX, int m_ObjRectY, int m_ObjRectWidth, int m_ObjRectHeight);
	void nativeFinalizeLearning();
	void getGmmData();//更新参数
	void doBinarySegmentation();//jni_segmentation   处理
	void showSegmentationResult(BYTE* rgbimg, int rows, int cols, int chan);//得到分割结果
	bool nativeGetGMMdata(vector<float> mu_fg, vector<float> mu_bg,
		vector<float> pi_fg, vector<float> pi_bg);
private:

};


