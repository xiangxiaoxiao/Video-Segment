#include "process.h"
#include "exchangeRGBandYCBCR.h"

void Video_Segmention::Video_Seg(string path)
{
	VideoCapture capture(path);
	Mat img;
	mJniObjectAddr = new jni_state;
	m_fps = capture.get(CV_CAP_PROP_FPS);  //帧率 
	//int video_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);//总帧数
	//m_frames = (m_frames <= video_frames) ? m_frames : video_frames; 
	m_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
	bool stop = false;
	int delay = 1000 / m_fps;
	namedWindow("input video");
	int kk = 0;//基数

	while (!stop && kk < m_frames) //先载入后处理
	{
		if (!capture.read(img))
		{
			break;
		}
		if (kk == 0)//初始化参数
		{
			m_width=img.cols;
			m_height=img.rows;
			m_channel=img.channels();
			mDataRgb = new BYTE[m_width*m_height*m_channel];
			m_nPixelAcount = m_height*m_width;
			nativeCreateJniObject(mJniObjectAddr, SLIC_K, SLIC_M, SLIC_NUM_ITERS, SEGMENTATION_K,
				SEGMENTATION_TV_WEIGHT, SEGMENTATION_WT_WEIGHT,
				SEGMENTATION_BETA, SEGMENTATION_NUM_ITERS,
				m_height, m_width, 1);//创建对象 初始化参数
		}
		char buff[20];
		sprintf_s(buff, "%04d.png", kk);
		//处理img  将Mat转为BYTE*(RGB)
		const BYTE *p = NULL;
		BYTE *ptrImage = NULL;
		for (int i = 0; i < m_height; i++)
		{
			p = img.ptr<BYTE>(i);
			ptrImage = mDataRgb + i*m_width * 3;
			for (int j = 0; j < m_width; j++)
			{
				ptrImage[0] = (BYTE)(p[2]);
				ptrImage[1] = (BYTE)(p[1]);
				ptrImage[2] = (BYTE)(p[0]);
				ptrImage += 3;
				p += 3;
			}
		}
		//Main work
		nativeRunTSlic(mDataRgb);
		nativeTSlicConstructGraph(mDataRgb, 3);//// construct superpixel graph on rgb image
		if (kk >= 0 && kk <= 20)//默认0到20帧学习
		{
			nativeLearnAddFeatures(m_ObjRectX, m_ObjRectY, m_ObjRectWidth, m_ObjRectHeight);//目标框
			nativeFinalizeLearning();
			getGmmData();
		}
		if (kk>20)
		{
			long tm_pro = clock();
			doBinarySegmentation();//
			showSegmentationResult(mDataRgb, m_height, m_width, m_channel);//着色
			cout << "the " << kk << " the processing time : " << clock() - tm_pro << "ms" << endl;//测试seg时间
		}
		//(RGB)BYTE*_to_Mat 将BYTE转为Mat
		BYTE *p2 = NULL;
		const BYTE *ptrImage2 = NULL;
		for (int i = 0; i < m_height; i++)
		{
			p2 = (BYTE*)(img.data + img.step*i);
			ptrImage2 = mDataRgb + i*m_width * 3;
			for (int j = 0; j < m_width; j++)
			{
				p2[0] = ptrImage2[2];
				p2[1] = ptrImage2[1];
				p2[2] = ptrImage2[0];

				p2 += 3;
				ptrImage2 += 3;
			}
		}
		imshow("input video", img);
		rectangle(img, node, center, Scalar(0, 0, 255));//画框
		kk++;
		if (cv::waitKey(delay) >= 0)
		{
			stop = true;
		}
	}
	capture.release();
	//cv::destroyWindow("input video");
}

bool Video_Segmention::nativeGetGMMdata(vector<float> mu_fg, vector<float> mu_bg,
	vector<float> pi_fg, vector<float> pi_bg)
{

	jni_state* state = (jni_state*)mJniObjectAddr;
	if (state == 0) { return 0; }

	int num_clusters = state->gmm_fg.clusterProb.size(); // this should be stored in state..
	int u = 0;
	for (int k = 0; k < num_clusters; ++k) {
		size_t m = state->gmm_fg.clusterParam[k].mu.size();
		for (size_t i = 0; i < m; ++i) {
			mu_fg[u] = state->gmm_fg.clusterParam[k].mu[i];
			mu_bg[u] = state->gmm_bg.clusterParam[k].mu[i];
			u++;
		}
	}
	for (int k = 0; k < num_clusters; ++k) {
		pi_fg[k] = state->gmm_fg.clusterProb[k];
		pi_bg[k] = state->gmm_bg.clusterProb[k];
	}
	return (num_clusters > 0);
}


void Video_Segmention::nativeCreateJniObject(jni_state *mJniObjectAddr, int slic_K, int slic_M,
	int slic_num_iters, int seg_K,
	float seg_tv, float seg_wt, float seg_beta, int seg_num_iters,
	int rows, int cols, int chan)//创建对象 初始化参数
{
	cout << "nativeCreateJniObject !" << endl;

	mJniObjectAddr->params;
	mJniObjectAddr->params.rows = rows;
	mJniObjectAddr->params.cols = cols;
	mJniObjectAddr->params.chan = chan;
	mJniObjectAddr->params.K = slic_K;
	mJniObjectAddr->params.M = slic_M;
	mJniObjectAddr->params.num_iters = slic_num_iters;
	mJniObjectAddr->params.min_area_frac = 0.50;
	mJniObjectAddr->params.max_area_frac = 1.750f;
	mJniObjectAddr->tslic = tSlic(mJniObjectAddr->params);
	mJniObjectAddr->previmg = Mat(0, 0, CV_8UC1);
	mJniObjectAddr->prev_seg = vector<uchar>();
	//state->prev_seg_id = vector<int>();
	mJniObjectAddr->tslic_ids = vector<int>();
	mJniObjectAddr->tslic_ids_prev = vector<int>();
	mJniObjectAddr->graph = vector<SlicNode<float> >();
	// segmentation crap (consider reserving storage here..)
	mJniObjectAddr->obj_model.num_fg = 0;
	mJniObjectAddr->obj_model.num_bg = 0;
	mJniObjectAddr->obj_model.max_num_pts = 3000;
	mJniObjectAddr->obj_model.data_fg = vector<float>();
	mJniObjectAddr->obj_model.data_fg.reserve(3000 * 3);
	mJniObjectAddr->obj_model.data_bg.reserve(3000 * 3);
	mJniObjectAddr->obj_model.supermask = vector<uchar>();
	mJniObjectAddr->obj_model.supermask.reserve(3000);
	mJniObjectAddr->obj_model.supergraph = vector<SlicNode<float> >();
	mJniObjectAddr->obj_model.supergraph.reserve(3000);
	mJniObjectAddr->obj_model.prev_mask_size = 0;

	mJniObjectAddr->contour_pts = vector<int>();
	mJniObjectAddr->gmm_fg = GMM();
	mJniObjectAddr->gmm_fg = GMM();
	mJniObjectAddr->gmm_set = 0;
	// segmentation parameters
	mJniObjectAddr->gmm_num_clusters = seg_K;
	mJniObjectAddr->gmm_num_iters = 25;
	mJniObjectAddr->seg.tv = seg_tv;
	mJniObjectAddr->seg.wt = seg_wt;
	mJniObjectAddr->seg.beta = seg_beta;
	mJniObjectAddr->seg.num_iters = seg_num_iters;
	
	if (GMMfg != NULL)
	{
		GMMbg->K = SEGMENTATION_K;
		GMMfg->initView();
	}

	if (GMMbg != NULL)
	{
		GMMbg->initView();
		GMMbg->K = SEGMENTATION_K;
	}
}
void Video_Segmention::nativeRunTSlic(BYTE *data)
{
	cout << "nativeRunTSlic" << endl;
	//jni_state* state = (jni_state*)mJniObjectAddr;
	if (mJniObjectAddr == 0) { return; }
	
	BYTE *pYChannel = new BYTE[m_nPixelAcount];
	BYTE *pCbChannel = new BYTE[m_nPixelAcount];
	BYTE *pCrChannel = new BYTE[m_nPixelAcount];

	DecomposeToYCbCr(data, m_nPixelAcount, pYChannel, pCbChannel, pCrChannel);

	BYTE *c_data = pYChannel;
	//BYTE* c_data = data;//获取data
	// -------------------------------------------------------------------------
	Size LK_winsize = Size(15, 15); //Size(21,21);
	int LK_maxpyrlevel = 0;

	int rows = mJniObjectAddr->params.rows;
	int cols = mJniObjectAddr->params.cols;
	int chan = mJniObjectAddr->params.chan;
	int n = rows*cols*chan;

	vector<unsigned char> img((unsigned char*)c_data, (unsigned char*)c_data + n);
	Mat curmat = Mat(rows, cols, CV_8UC1, c_data);

	if ((rows == 0) || (cols == 0)) { return; }
	if ((mJniObjectAddr->tslic.K == 0) ||
		(mJniObjectAddr->previmg.rows == 0) || (mJniObjectAddr->previmg.cols == 0))  {
		// using only the current image
		mJniObjectAddr->tslic = tSlic(mJniObjectAddr->params);
		mJniObjectAddr->tslic.segment(img, vector<int>());
	}
	else {
		int K = mJniObjectAddr->tslic.K;//number of superpixels in current frame
		vector<Point2f> prev_pts = vector<Point2f>(K);
		vector<Point2f> cur_pts = vector<Point2f>(K);
		for (int i = 0; i < K; ++i) {
			prev_pts[i].x = mJniObjectAddr->tslic.centers[i].x;
			prev_pts[i].y = mJniObjectAddr->tslic.centers[i].y;
		}
		vector<uchar> status = vector<uchar>(K);
		vector<float> err = vector<float>(K);
		calcOpticalFlowPyrLK(mJniObjectAddr->previmg, curmat, prev_pts, cur_pts,
			status, err, LK_winsize, LK_maxpyrlevel);//计算一个稀疏特征集的光流，使用金字塔中的迭代 Lucas-Kanade 方法

		vector<int> predicted_centers = vector<int>(K * 2);
		for (int i = 0; i < K; ++i) {
			predicted_centers[2 * i] = cur_pts[i].x;
			predicted_centers[2 * i + 1] = cur_pts[i].y;
		}
		mJniObjectAddr->tslic.segment(img, predicted_centers);
	}
	// store current image: 保存当前的到previmg
	curmat.copyTo(mJniObjectAddr->previmg);

		SAFE_DELETE_ARRAY(pYChannel)
		SAFE_DELETE_ARRAY(pCbChannel)
		SAFE_DELETE_ARRAY(pCrChannel)
}
void Video_Segmention::nativeTSlicConstructGraph(BYTE* data, int chan)
{
	cout << " nativeTSlicConstructGraph" << endl;
	jni_state* state = (jni_state*)mJniObjectAddr;
	if (state == 0) { return; }
	BYTE* c_data = data;
	// -------------------------------------------------------------------------
	int rows = state->params.rows;
	int cols = state->params.cols;
	int n = rows*cols*chan;
	vector<unsigned char> img((unsigned char*)c_data, (unsigned char*)c_data + n);
	Mat curmat = Mat(rows, cols, (chan == 3) ? CV_8UC3 : CV_8UC1, c_data);

	if ((rows == 0) || (cols == 0)) { return; }
	// construct superpixel graph
	vector<int> assignments = state->tslic.get_assignments();
	state->graph =
		construct_slic_graph<float>(img, assignments, rows, cols, chan);
	slic_scale_graph_pixels(state->graph, 1.0f / 256.0f);

	state->tslic_ids_prev = state->tslic_ids;
	state->tslic_ids = state->tslic.get_ids();
}
void Video_Segmention::nativeLearnAddFeatures(int x, int y, int w, int h)
{
	cout << "nativeLearnAddFeatures" << endl;

	jni_state* state = (jni_state*)mJniObjectAddr;
	if (state == 0) { return; }
	if (state->graph.size() == 0) { return; }

	int x_lo = x - 0.5*w;
	int x_hi = x + 0.5*w;
	int y_lo = y - 0.5*h;
	int y_hi = y + 0.5*h;

	vector<uchar> mask = vector<uchar>(state->graph.size());

	// compute mask from the rectangle shown on the screen.
	for (size_t i = 0; i < state->graph.size(); ++i) {
		int x = state->graph[i].x;
		int y = state->graph[i].y;
		mask[i] = ((x > x_lo) && (x < x_hi) && (y > y_lo) && (y < y_hi));
	}
	// append to the currently existing mask.
	vector<int> p1;
	vector<int> p2;
	//spgraph_get_id_pairs(p1, p2, state->tslic_ids_prev, state->tslic_ids);
	transform(p1.begin(), p1.end(), p1.begin(),
		bind1st(plus<int>(), state->obj_model.supergraph.size() - state->obj_model.prev_mask_size));
	transform(p2.begin(), p2.end(), p2.begin(),
		bind1st(plus<int>(), state->obj_model.supergraph.size()));

	// append current mask and current graph.
	state->obj_model.supermask.insert(state->obj_model.supermask.end(),
		mask.begin(), mask.end());
	spgraph_append(state->obj_model.supergraph, state->graph);

	// append temporal neighbors:
	spgraph_add_neighbors(state->obj_model.supergraph, p1, p2);
	state->obj_model.prev_mask_size = mask.size();
}
void Video_Segmention::nativeFinalizeLearning()
{
	// perform iterative GMM estimation. Once finished, the GMMs are estimated,
	// and the model structure is cleared.
	cout << "nativeFinalizeLearning" << endl;
	jni_state* state = (jni_state*)mJniObjectAddr;
	if (state == 0) { return; }
	if (state->obj_model.supergraph.size() == 0) { return; }
	if (state->obj_model.supermask.size() == 0) { return; }
	// perform iterative GMM estimation. Once finished, the GMMs are estimated,
	// and the model structure is cleared.
	int num_reestimates = 1;
	tvseg_iterative_gmm_estimation(state->gmm_fg, state->gmm_bg,
		state->obj_model, state->gmm_num_clusters, state->gmm_num_iters,
		num_reestimates, state->seg.num_iters, state->seg.beta, state->seg.tv);

	state->obj_model.data_fg.clear();
	state->obj_model.data_bg.clear();
	state->obj_model.supergraph.clear();
	state->obj_model.supermask.clear();
	state->obj_model.prev_mask_size = 0;
	state->obj_model.num_fg = 0;
	state->obj_model.num_bg = 0;
}
void Video_Segmention::getGmmData()//更新参数
{
	vector<float> mu_fg(SEGMENTATION_K * 3);
	vector<float> mu_bg(SEGMENTATION_K * 3);
	vector<float> pi_fg(SEGMENTATION_K);
	vector<float> pi_bg(SEGMENTATION_K);

	bool ok = nativeGetGMMdata(mu_fg, mu_bg, pi_fg, pi_bg);
	if (!ok) { return; }

	if (GMMfg != NULL)
	{
		GMMfg->K = SEGMENTATION_K;
		GMMfg->setPi(pi_fg);
		GMMfg->setMu(mu_fg);
		GMMfg->setLabel();
	}
	if (GMMbg != NULL)
	{
		GMMbg->K = SEGMENTATION_K;
		GMMbg->setPi(pi_bg);
		GMMbg->setMu(mu_bg);
		GMMbg->setLabel();
	}
}
void Video_Segmention::doBinarySegmentation()//jni_segmentation   处理
{
	cout << ("doBinarySegmentation") << endl;

	jni_state* state = (jni_state*)mJniObjectAddr;
	if (state == 0) { return; }
	if (state->gmm_fg.clusterProb.size() == 0) { return; }
	if (state->gmm_bg.clusterProb.size() == 0) { return; }
	//LOGI("doBinarySegmentation");
	// -------------------------------------------------------------------------
	// create problem structure for solving:
	tvsegmentbinary_params p;
	p.D = create_difference_op(state->graph);

	//t1 = now_ms();
	p.unary = compute_unary_potential(state->graph, state->gmm_fg, state->gmm_bg);
	//LOGI("unary: %f ms", now_ms()-t1);

	// compute temporal weight.
	if ((state->seg.wt > 0) && (state->prev_seg.size() > 0) &&
		(state->tslic_ids_prev.size() > 0)) {
		vector<float> t_unary = compute_temporal_unary_potential(state->tslic.centers,
			state->prev_seg, state->tslic_ids_prev);

		transform(t_unary.begin(), t_unary.end(), t_unary.begin(),
			bind1st(multiplies<float>(), state->seg.wt));

		transform(p.unary.begin(), p.unary.end(), t_unary.begin(),
			p.unary.begin(), plus<float>());
	}
	//t1 = now_ms();
	p.weights = compute_pairwise_potential(state->graph, state->seg.beta, p.D.nrows);
	// factor-in the TV penalty weight.
	transform(p.weights.begin(), p.weights.end(), p.weights.begin(),
		bind1st(multiplies<float>(), state->seg.tv));
	//LOGI("pairwise: %f ms", now_ms()-t1);

	p.nnodes = p.unary.size();
	p.nedges = p.weights.size();
	p.max_iters = state->seg.num_iters;

	tvsegmentbinary tvs = tvsegmentbinary(&p);

	//t1 = now_ms();
	vector<uchar> res_vec = tvs.get_segmentation();
	//LOGI("segmentation: %f ms", now_ms()-t1);

	// -------------------------------------------------------------------------
	//      store current segmentation (for use with temporal penalty)
	// -------------------------------------------------------------------------
	if (state->seg.wt > 0) { state->prev_seg = res_vec; }
	// -------------------------------------------------------------------------

	//t1 = now_ms();
	int rows = state->params.rows;
	int cols = state->params.cols;
	state->cur_seg = spgraph_vec2img(state->graph, res_vec, rows, cols);
}
void Video_Segmention::showSegmentationResult(BYTE* rgbimg, int rows, int cols, int chan)//得到分割结果
{
	cout << "showSegmentationResult" << endl;
	jni_state* state = (jni_state*)mJniObjectAddr;
	if (state == 0) { return; }
	int in_rows = state->params.rows;
	int in_cols = state->params.cols;
	if (state->cur_seg.size() != in_rows*in_cols) { return; }

	vector<uchar> val = vector<uchar>(chan);
	if (chan == 1) { val[0] = 255; system("pause"); }
	if (chan == 3) { val[0] = 86; val[1] = 180; val[2] = 211; }

	BYTE* c_rgbimg = rgbimg;
	draw_edge((uchar*)c_rgbimg, state->cur_seg, val, in_rows, in_cols, chan);//标出mask,设置为255（白色）
}

void GMMdata::initView() {
	vector<int> temp(K);
	viewid = temp;
	int prev_id = 0;
	int id;
	for (int k = 0; k < K; ++k) {
		id = getFreeId();
		viewid[k] = id;
		prev_id = id;
	}
}
int GMMdata:: getFreeId() {
	int id;
	id = (int)(rand() / (RAND_MAX + 1));
	return id;
}
void GMMdata::setPi(vector<float> data) {
	pi = data;
}
void GMMdata::setMu(vector<float> mu) {
	if (viewid.size() != K) { cout << "views not initialized.." << endl; }
	for (int k = 0; k < K; ++k) {
		float r = mu[DIM*k];
		float g = mu[DIM*k + 1];
		float b = mu[DIM*k + 2];
	}
}
void GMMdata::clear() {
	clearLabel();
	int num = viewid.size();
}
void GMMdata:: clearLabel() { cout << "gmm  " << endl; }
void GMMdata::setLabel() { cout << "GMM" << endl; }
