#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "time.h"

#define SIZE_X 320 // Input image Width
#define SIZE_Y 240 // Input image Height
#define FILTER_SIZE 5 // Filter size
#define FILTER_RADIUS 2 // Filter radius

// Utilty Macros
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

struct timespec tick_clockData;
struct timespec tock_clockData;

#define TICK()    { clock_gettime(CLOCK_MONOTONIC, &tick_clockData);}
#define TOCK(str) { clock_gettime(CLOCK_MONOTONIC, &tock_clockData);\
                    printf("%s\t %f milliseconds\n",str,\
                    (((double) tock_clockData.tv_sec + tock_clockData.tv_nsec / 1000000000.0) - \
                    ((double) tick_clockData.tv_sec + tick_clockData.tv_nsec / 1000000000.0))\
                    * 1000);}

// Input Array
float *input;
// Output Array
float *output;
// Filter Vector
float *gaussian;

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
void bilateralFilterKernel(float* out, const float* in,const float * gaussian, int size_x, int size_y, int r) {
        // Local Variables
        int i,j;
		unsigned int x,y,pos;

        #pragma omp parallel for shared(out),private(y,i,j,x,pos) schedule(static)
		for (y = 0; y < size_y; y++) {
			for (x = 0; x < size_x; x++) {
				pos = x + y * size_x;
				if (in[pos] == 0) {
					out[pos] = 0;
					continue;
				}

				float sum = 0.0f;
				float t = 0.0f;

				const float center = in[pos];

				for (i = -r; i <= r; ++i) {
					for (j = -r; j <= r; ++j) {
						unsigned int curPos_x = MAX(0u, MIN(x + i,size_x - 1));
                        unsigned int curPos_y = MAX(0u, MIN(y + j,size_y - 1));

						const float curPix = in[curPos_x + curPos_y * size_x];
						if (curPix > 0) {
							const float mod = pow(curPix - center,2);
							const float factor = gaussian[i + r]
									* gaussian[j + r]
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

/***********************************************************
 * Function:  read_input
 * ---------------------------------------------------------
 * Reads the input.bin file, and loads it to input array.
 * *********************************************************/
void read_input(){
    FILE *fptr;

    if ((fptr = fopen("input.bin","r")) == NULL){
        printf("Error! opening file");
        exit(1);
    }

    fread(input, sizeof(float) * SIZE_X * SIZE_Y, 1, fptr);
    fclose(fptr);

}

/***********************************************************
 * Function:  compare
 * ---------------------------------------------------------
 * Reads the output.bin file, that contains the golden output
 * values. Then it compares it with the filter output array by
 * calculating the Mean Error Square (MSE). The perfect solution
 * must have an MSE value of 0. A greater value corresponds to
 * a worse solution.
 * *********************************************************/
void compare(){
    FILE *fptr;
    int y,x;
    double diff;
    double mse=-0.068993; // A calculated constant - DO NOT CHANGE IT.

    // Open output file and load it to outputGolden
    if ((fptr = fopen("goldenOutput.bin","r")) == NULL){
            printf("Error! opening file");
            exit(1);
    }
    float *goldenOutput = (float*) malloc(sizeof(float) * SIZE_X * SIZE_Y);
    fread(goldenOutput, sizeof(float) * SIZE_X * SIZE_Y, 1, fptr);
    fclose(fptr);

    // Calculate MSR
    for ( x = 0; x < SIZE_X; x++) {
        for (y = 0; y < SIZE_Y; y++){
            diff = output[x + y * SIZE_X] - goldenOutput[x + y *SIZE_X];
            mse += pow(fabs(diff),2.0);
        }
    }
    printf("MSE :\t\t %.6f\n", mse/(double)(SIZE_X*SIZE_Y));

    free(goldenOutput);
}

/***********************************************************
 * Function:  compare
 * ---------------------------------------------------------
 * Main function.
 * *********************************************************/
int main(int argc, char *argv[]){
    int i,x;

    // Allocate memory for data arrays
    input = (float*) calloc(sizeof(float) * SIZE_X * SIZE_Y,1);
    output = (float*) calloc(sizeof(float) * SIZE_X * SIZE_Y,1);
    gaussian = (float*) calloc(FILTER_SIZE * sizeof(float), 1);

    printf("--------- Running --------------\n");

    TICK();
    read_input();

    /**** Create filter vector using a mathematical expression *****/
     for (i= 0; i < FILTER_SIZE; i++) {
		x = i - 2;
		gaussian[i] = expf(-(x * x) / (32.0f));
	}

    TOCK("load_time:");

    TICK();
    bilateralFilterKernel(output, input, gaussian, SIZE_X, SIZE_Y,FILTER_RADIUS);
    TOCK("filter_time:");

    TICK();
    compare();
    TOCK("compare_time:");

    free(input);
    free(output);
    free(gaussian);
    return 0;
}
