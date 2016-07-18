#pragma once
#ifndef _EXCHANGE_RGB_YCBCR_H_
#define _EXCHANGE_RGB_YCBCR_H_

#include "mtdefine.h"

// RGB转YCbCr查找数据
int Y1[256], Y2[256], Y3[256];
int Cb1[256], Cb2[256], Cb3[256];
int Cr1[256], Cr2[256], Cr3[256];
bool bRGB2YCbCrTableInit = false;

int PB1[256], PB2[256], PG1[256], PG2[256];
bool bYCbCr2RGBTableInit = false;

void InitFastRGB2YCbCrTable()
{
	bRGB2YCbCrTableInit = true;

	int sumY1 = 0, sumY2 = 0, sumY3 = 0;
	int sumCb1 = 0, sumCb2 = 0, sumCb3 = 0;
	int sumCr1 = 0, sumCr2 = 0, sumCr3 = 0;

	for (int i = 0; i < 256; ++i)
	{
		Y1[i] = sumY1;
		Y2[i] = sumY2;
		Y3[i] = sumY3;

		Cb1[i] = sumCb1;
		Cb2[i] = sumCb2;
		Cb3[i] = sumCb3;

		Cr1[i] = sumCr1;
		Cr2[i] = sumCr2;
		Cr3[i] = sumCr3;

		sumY1 += 4915;
		sumY2 += 9667;
		sumY3 += 1802;
		sumCb1 += -2764;
		sumCb2 += -5428;
		sumCb3 += 8192;
		sumCr1 += 8192;
		sumCr2 += -6860;
		sumCr3 += -1332;
	};
}

void InitFastYCbCr2RGBTable()
{
	bYCbCr2RGBTableInit = true;

	int tpb1 = -128 * 22970;
	int tpb2 = -128 * 11700;
	int tpg1 = -128 * 5638;
	int tpg2 = -128 * 29032;

	for (int i = 0; i < 256; ++i)
	{
		PB1[i] = tpb1;
		PB2[i] = tpb2;
		PG1[i] = tpg1;
		PG2[i] = tpg2;

		tpb1 += 22970;
		tpb2 += 11700;
		tpg1 += 5638;
		tpg2 += 29032;
	}
}

inline void RGB2YCbCr_Fast(BYTE& rValue, BYTE& gValue, BYTE& bValue, BYTE& yValue, BYTE& cbValue, BYTE& crValue)
{
	if (!bRGB2YCbCrTableInit)
	{
		InitFastRGB2YCbCrTable();
	}

	yValue = ((Y1[rValue] + Y2[gValue] + Y3[bValue] + 0x2000) >> 14);
	cbValue = max(0, min(255, ((Cb1[rValue] + Cb2[gValue] + Cb3[bValue] + 0x202000) >> 14)));
	crValue = max(0, min(255, ((Cr1[rValue] + Cr2[gValue] + Cr3[bValue] + 0x202000) >> 14)));
}

inline void YCbCr2RGB_Fast(BYTE& yValue, BYTE& cbValue, BYTE& crValue, BYTE& rValue, BYTE& gValue, BYTE& bValue)
{
	if (!bYCbCr2RGBTableInit)
	{
		InitFastYCbCr2RGBTable();
	}

	int y1 = (yValue << 14);
	rValue = max(0, min(255, (y1 + PB1[crValue] + 0x2000) >> 14));
	gValue = max(0, min(255, (y1 - PG1[cbValue] - PB2[crValue] + 0x2000) >> 14));
	bValue = max(0, min(255, (y1 + PG2[cbValue] + 0x2000) >> 14));
}

inline void RGB2YCbCr(BYTE& rValue, BYTE& gValue, BYTE& bValue, BYTE& yValue, BYTE& cbValue, BYTE& crValue)
{
	yValue = static_cast<BYTE>(MIN(255, 0.2990f * rValue + 0.587f * gValue + 0.114f * bValue + 0.4f));
	cbValue = static_cast<BYTE>(MAX(0, MIN(255, -0.1687f * rValue - 0.3313f * gValue + 0.5000f * bValue + 128.4f)));
	crValue = static_cast<BYTE>(MAX(0, MIN(255, 0.5000f * rValue - 0.4187f * gValue - 0.0813f * bValue + 128.4f)));
}

inline void YCbCr2RGB(BYTE& yValue, BYTE& cbValue, BYTE& crValue, BYTE& rValue, BYTE& gValue, BYTE& bValue)
{
	int pb = crValue - 128;
	int pg = cbValue - 128;

	rValue = static_cast<BYTE>(MAX(0, MIN(255, yValue + 1.40200f * pb)));
	gValue = static_cast<BYTE>(MAX(0, MIN(255, yValue - 0.34414f * pg - 0.71414f * pb)));
	bValue = static_cast<BYTE>(MAX(0, MIN(255, yValue + 1.77200f * pg)));
}

void DecomposeToYCbCr(BYTE* pOriginalImage, int nPixelCount, BYTE* pYChannel, BYTE* pCbChannel, BYTE* pCrChannel)
{
	BYTE *ptrImage = pOriginalImage;
	for (int i = 0; i < nPixelCount; ++i)
	{
		RGB2YCbCr(ptrImage[MT_RED], ptrImage[MT_GREEN], ptrImage[MT_BLUE], pYChannel[i], pCbChannel[i], pCrChannel[i]);

		ptrImage += BPP32_PER_PIXEL_BYTE;
	}
}

void RecomposeFromYCbCr(BYTE* pOriginalImage, int nPixelCount, BYTE* pYChannel, BYTE* pCbChannel, BYTE* pCrChannel)
{
	BYTE *ptrImage = pOriginalImage;
	for (int i = 0; i < nPixelCount; ++i)
	{
		YCbCr2RGB(pYChannel[i], pCbChannel[i], pCrChannel[i], ptrImage[MT_RED], ptrImage[MT_GREEN], ptrImage[MT_BLUE]);

		ptrImage += BPP32_PER_PIXEL_BYTE;
	}
}

#endif