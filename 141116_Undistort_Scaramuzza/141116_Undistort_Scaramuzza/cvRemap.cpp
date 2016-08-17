void cv::remap( InputArray _src, OutputArray _dst,
	InputArray _map1, InputArray _map2,
	int interpolation, int borderType, const Scalar& borderValue )
{
	static RemapNNFunc nn_tab[] =
	{
		remapNearest<uchar>, remapNearest<schar>, remapNearest<ushort>, remapNearest<short>,
		remapNearest<int>, remapNearest<float>, remapNearest<double>, 0
	};

	static RemapFunc linear_tab[] =
	{
		remapBilinear<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, RemapVec_8u, short>, 0,
		remapBilinear<Cast<float, ushort>, RemapNoVec, float>,
		remapBilinear<Cast<float, short>, RemapNoVec, float>, 0,
		remapBilinear<Cast<float, float>, RemapNoVec, float>,
		remapBilinear<Cast<double, double>, RemapNoVec, float>, 0
	};

	static RemapFunc cubic_tab[] =
	{
		remapBicubic<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short, INTER_REMAP_COEF_SCALE>, 0,
		remapBicubic<Cast<float, ushort>, float, 1>,
		remapBicubic<Cast<float, short>, float, 1>, 0,
		remapBicubic<Cast<float, float>, float, 1>,
		remapBicubic<Cast<double, double>, float, 1>, 0
	};

	static RemapFunc lanczos4_tab[] =
	{
		remapLanczos4<FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>, short, INTER_REMAP_COEF_SCALE>, 0,
		remapLanczos4<Cast<float, ushort>, float, 1>,
		remapLanczos4<Cast<float, short>, float, 1>, 0,
		remapLanczos4<Cast<float, float>, float, 1>,
		remapLanczos4<Cast<double, double>, float, 1>, 0
	};

	Mat src = _src.getMat(), map1 = _map1.getMat(), map2 = _map2.getMat();

	CV_Assert( (!map2.data || map2.size() == map1.size()));

	_dst.create( map1.size(), src.type() );
	Mat dst = _dst.getMat();
	if( dst.data == src.data )
		src = src.clone();

	int depth = src.depth(), map_depth = map1.depth();
	RemapNNFunc nnfunc = 0;
	RemapFunc ifunc = 0;
	const void* ctab = 0;
	bool fixpt = depth == CV_8U;
	bool planar_input = false;

	if( interpolation == INTER_NEAREST )
	{
		nnfunc = nn_tab[depth];
		CV_Assert( nnfunc != 0 );

		if( map1.type() == CV_16SC2 && !map2.data ) // the data is already in the right format
		{
			nnfunc( src, dst, map1, borderType, borderValue );
			return;
		}
	}
	else
	{
		if( interpolation == INTER_AREA )
			interpolation = INTER_LINEAR;

		if( interpolation == INTER_LINEAR )
			ifunc = linear_tab[depth];
		else if( interpolation == INTER_CUBIC )
			ifunc = cubic_tab[depth];
		else if( interpolation == INTER_LANCZOS4 )
			ifunc = lanczos4_tab[depth];
		else
			CV_Error( CV_StsBadArg, "Unknown interpolation method" );
		CV_Assert( ifunc != 0 );
		ctab = initInterTab2D( interpolation, fixpt );
	}

	const Mat *m1 = &map1, *m2 = &map2;

	if( (map1.type() == CV_16SC2 && (map2.type() == CV_16UC1 || map2.type() == CV_16SC1)) ||
		(map2.type() == CV_16SC2 && (map1.type() == CV_16UC1 || map1.type() == CV_16SC1)) )
	{
		if( map1.type() != CV_16SC2 )
			std::swap(m1, m2);
		if( ifunc )
		{
			ifunc( src, dst, *m1, *m2, ctab, borderType, borderValue );
			return;
		}
	}
	else
	{
		CV_Assert( (map1.type() == CV_32FC2 && !map2.data) ||
			(map1.type() == CV_32FC1 && map2.type() == CV_32FC1) );
		planar_input = map1.channels() == 1;
	}

	int x, y, x1, y1;
	const int buf_size = 1 << 14;
	int brows0 = std::min(128, dst.rows);
	int bcols0 = std::min(buf_size/brows0, dst.cols);
	brows0 = std::min(buf_size/bcols0, dst.rows);
#if CV_SSE2
	bool useSIMD = checkHardwareSupport(CV_CPU_SSE2);
#endif

	Mat _bufxy(brows0, bcols0, CV_16SC2), _bufa;
	if( !nnfunc )
		_bufa.create(brows0, bcols0, CV_16UC1);

	for( y = 0; y < dst.rows; y += brows0 )
	{
		for( x = 0; x < dst.cols; x += bcols0 )
		{
			int brows = std::min(brows0, dst.rows - y);
			int bcols = std::min(bcols0, dst.cols - x);
			Mat dpart(dst, Rect(x, y, bcols, brows));
			Mat bufxy(_bufxy, Rect(0, 0, bcols, brows));

			if( nnfunc )
			{
				if( map_depth != CV_32F )
				{
					for( y1 = 0; y1 < brows; y1++ )
					{
						short* XY = (short*)(bufxy.data + bufxy.step*y1);
						const short* sXY = (const short*)(m1->data + m1->step*(y+y1)) + x*2;
						const ushort* sA = (const ushort*)(m2->data + m2->step*(y+y1)) + x;

						for( x1 = 0; x1 < bcols; x1++ )
						{
							int a = sA[x1] & (INTER_TAB_SIZE2-1);
							XY[x1*2] = sXY[x1*2] + NNDeltaTab_i[a][0];
							XY[x1*2+1] = sXY[x1*2+1] + NNDeltaTab_i[a][1];
						}
					}
				}
				else if( !planar_input )
					map1(Rect(0,0,bcols,brows)).convertTo(bufxy, bufxy.depth());
				else
				{
					for( y1 = 0; y1 < brows; y1++ )
					{
						short* XY = (short*)(bufxy.data + bufxy.step*y1);
						const float* sX = (const float*)(map1.data + map1.step*(y+y1)) + x;
						const float* sY = (const float*)(map2.data + map2.step*(y+y1)) + x;
						x1 = 0;

#if CV_SSE2
						if( useSIMD )
						{
							for( ; x1 <= bcols - 8; x1 += 8 )
							{
								__m128 fx0 = _mm_loadu_ps(sX + x1);
								__m128 fx1 = _mm_loadu_ps(sX + x1 + 4);
								__m128 fy0 = _mm_loadu_ps(sY + x1);
								__m128 fy1 = _mm_loadu_ps(sY + x1 + 4);
								__m128i ix0 = _mm_cvtps_epi32(fx0);
								__m128i ix1 = _mm_cvtps_epi32(fx1);
								__m128i iy0 = _mm_cvtps_epi32(fy0);
								__m128i iy1 = _mm_cvtps_epi32(fy1);
								ix0 = _mm_packs_epi32(ix0, ix1);
								iy0 = _mm_packs_epi32(iy0, iy1);
								ix1 = _mm_unpacklo_epi16(ix0, iy0);
								iy1 = _mm_unpackhi_epi16(ix0, iy0);
								_mm_storeu_si128((__m128i*)(XY + x1*2), ix1);
								_mm_storeu_si128((__m128i*)(XY + x1*2 + 8), iy1);
							}
						}
#endif

						for( ; x1 < bcols; x1++ )
						{
							XY[x1*2] = saturate_cast<short>(sX[x1]);
							XY[x1*2+1] = saturate_cast<short>(sY[x1]);
						}
					}
				}
				nnfunc( src, dpart, bufxy, borderType, borderValue );
				continue;
			}

			Mat bufa(_bufa, Rect(0,0,bcols, brows));
			for( y1 = 0; y1 < brows; y1++ )
			{
				short* XY = (short*)(bufxy.data + bufxy.step*y1);
				ushort* A = (ushort*)(bufa.data + bufa.step*y1);

				if( planar_input )
				{
					const float* sX = (const float*)(map1.data + map1.step*(y+y1)) + x;
					const float* sY = (const float*)(map2.data + map2.step*(y+y1)) + x;

					x1 = 0;
#if CV_SSE2
					if( useSIMD )
					{
						__m128 scale = _mm_set1_ps((float)INTER_TAB_SIZE);
						__m128i mask = _mm_set1_epi32(INTER_TAB_SIZE-1);
						for( ; x1 <= bcols - 8; x1 += 8 )
						{
							__m128 fx0 = _mm_loadu_ps(sX + x1);
							__m128 fx1 = _mm_loadu_ps(sX + x1 + 4);
							__m128 fy0 = _mm_loadu_ps(sY + x1);
							__m128 fy1 = _mm_loadu_ps(sY + x1 + 4);
							__m128i ix0 = _mm_cvtps_epi32(_mm_mul_ps(fx0, scale));
							__m128i ix1 = _mm_cvtps_epi32(_mm_mul_ps(fx1, scale));
							__m128i iy0 = _mm_cvtps_epi32(_mm_mul_ps(fy0, scale));
							__m128i iy1 = _mm_cvtps_epi32(_mm_mul_ps(fy1, scale));
							__m128i mx0 = _mm_and_si128(ix0, mask);
							__m128i mx1 = _mm_and_si128(ix1, mask);
							__m128i my0 = _mm_and_si128(iy0, mask);
							__m128i my1 = _mm_and_si128(iy1, mask);
							mx0 = _mm_packs_epi32(mx0, mx1);
							my0 = _mm_packs_epi32(my0, my1);
							my0 = _mm_slli_epi16(my0, INTER_BITS);
							mx0 = _mm_or_si128(mx0, my0);
							_mm_storeu_si128((__m128i*)(A + x1), mx0);
							ix0 = _mm_srai_epi32(ix0, INTER_BITS);
							ix1 = _mm_srai_epi32(ix1, INTER_BITS);
							iy0 = _mm_srai_epi32(iy0, INTER_BITS);
							iy1 = _mm_srai_epi32(iy1, INTER_BITS);
							ix0 = _mm_packs_epi32(ix0, ix1);
							iy0 = _mm_packs_epi32(iy0, iy1);
							ix1 = _mm_unpacklo_epi16(ix0, iy0);
							iy1 = _mm_unpackhi_epi16(ix0, iy0);
							_mm_storeu_si128((__m128i*)(XY + x1*2), ix1);
							_mm_storeu_si128((__m128i*)(XY + x1*2 + 8), iy1);
						}
					}
#endif

					for( ; x1 < bcols; x1++ )
					{
						int sx = cvRound(sX[x1]*INTER_TAB_SIZE);
						int sy = cvRound(sY[x1]*INTER_TAB_SIZE);
						int v = (sy & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE-1));
						XY[x1*2] = (short)(sx >> INTER_BITS);
						XY[x1*2+1] = (short)(sy >> INTER_BITS);
						A[x1] = (ushort)v;
					}
				}
				else
				{
					const float* sXY = (const float*)(map1.data + map1.step*(y+y1)) + x*2;

					for( x1 = 0; x1 < bcols; x1++ )
					{
						int sx = cvRound(sXY[x1*2]*INTER_TAB_SIZE);
						int sy = cvRound(sXY[x1*2+1]*INTER_TAB_SIZE);
						int v = (sy & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE-1));
						XY[x1*2] = (short)(sx >> INTER_BITS);
						XY[x1*2+1] = (short)(sy >> INTER_BITS);
						A[x1] = (ushort)v;
					}
				}
			}
			ifunc(src, dpart, bufxy, bufa, ctab, borderType, borderValue);
		}
	}
}