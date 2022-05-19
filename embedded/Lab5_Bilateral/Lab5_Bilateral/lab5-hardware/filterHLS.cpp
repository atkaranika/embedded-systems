#include <math.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

/**
 * Extern is a requirement for Vitis Unified Software Platform,
 * when we use a .cpp (C++ source code) file.
 * */
extern "C" {

/***********************************************************
 * Function:  bilateralFilterKernel
 * ---------------------------------------------------------
 * Applies a vector filter on a SIZE_X x SIZE_Y input image.
 *
 *                           Apply
 *    +--------------+     +--------+    +--------------+
 *    |              |     |        |    |              |
 *    | Input Image  +---->+ Filter +--->+ Output Image |
 *    |              |     |        |    |              |
 *    +--------------+     +--------+    +--------------+
 *
 *
 *  out: The output image after the filter was applied.
 *
 *  in: The input image.
 *
 *  gaussian: A 5 element vector that holds the filter values.
 *************************************************************/
void bilateralFilterKernel(float* out, float* in,float* gaussian,int size_x,int size_y,int r) {

    /*** Required INTERFACE pragma START ***/
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem
	#pragma HLS INTERFACE s_axilite port=out	bundle=control

	#pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem
	#pragma HLS INTERFACE s_axilite port=in	bundle=control

	#pragma HLS INTERFACE m_axi port=gaussian offset=slave bundle=gmem
	#pragma HLS INTERFACE s_axilite port=gaussian	bundle=control
    /*** Required INTERFACE pragma END ***/

	#pragma HLS INTERFACE s_axilite port=size_x bundle=control
	#pragma HLS INTERFACE s_axilite port=size_y bundle=control
	#pragma HLS INTERFACE s_axilite port=r bundle=control

	int i,j;
	unsigned int x,y,pos;

	for (y = 0; y < size_y; y++) {
	/* Vitis automatically enables SW pipelining. 
       The following pragma disables pipelining. 
	   You have to replace it with the appopriate pragmas
       to enable optimizations */	   
		#pragma HLS PIPELINE off      
		for (x = 0; x < size_x; x++) {
					#pragma HLS PIPELINE off

			pos = x + y * size_x;
			if (in[pos] == 0) {
				out[pos] = 0;
				continue;
			}

			float sum = 0.0f;
			float t = 0.0f;

			const float center = in[pos];

			for (i = -r; i <= r; ++i) {
						#pragma HLS PIPELINE off

				for (j = -r; j <= r; ++j) {
							#pragma HLS PIPELINE off

					unsigned int curPos_x = MAX(0u, MIN(x + i,size_x - 1));
					unsigned int curPos_y = MAX(0u, MIN(y + j,size_y - 1));

					const float curPix = in[curPos_x + curPos_y * size_x];
					if (curPix > 0) {
						const float mod = (curPix - center) * (curPix - center);
						const float factor = gaussian[i + r]
								*gaussian[j + r]
								* expf(-mod / 0.02f);
						t += factor * curPix;
						sum += factor;
					}
				}
			}
			out[pos] = t / sum;
		}
	}
}

}