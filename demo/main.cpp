#include "process.h"

string g_path = "../input/desktop.mp4 ";//´æ·ÅÊÓÆµÂ·¾¶
int main()
{
	Video_Segmention* video_seg = new Video_Segmention;
	video_seg->Video_Seg(g_path);
	return 0;
}