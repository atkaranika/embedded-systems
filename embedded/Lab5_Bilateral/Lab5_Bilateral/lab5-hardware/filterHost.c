#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "time.h"

#define SIZE_X 320 // Input image Width
#define SIZE_Y 240 // Input image Height
#define FILTER_SIZE 5 // Filter size
#define FILTER_RADIUS 2 // Filter radius

/**** Timing Macros *****/
struct timespec tick_clockData;
struct timespec tock_clockData;

#define TICK()    { clock_gettime(CLOCK_MONOTONIC, &tick_clockData);}
#define TOCK(str) { clock_gettime(CLOCK_MONOTONIC, &tock_clockData);\
                    printf("%s\t%f milliseconds\n",str,\
                    (((double) tock_clockData.tv_sec + tock_clockData.tv_nsec / 1000000000.0) - \
                    ((double) tick_clockData.tv_sec + tick_clockData.tv_nsec / 1000000000.0))\
                    * 1000);}
/**** OpenCL necessary Defines ****/
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl2.hpp>

/**** OpenCL API variables ****/
cl_int err;
cl_uint check_status = 0;

cl_platform_id platform_id;
cl_platform_id platforms[16];
cl_uint platform_count;
cl_uint platform_found = 0;
char cl_platform_vendor[1001];

cl_uint num_devices;
cl_uint device_found = 0;
cl_device_id devices[16];
char cl_device_name[1001];
cl_device_id device_id;

cl_context context;
cl_command_queue q;
cl_program program;

cl_kernel bilateralFilterKernel; // Handler for hardware kernel
// Alternative for 2 compute units
// cl_kernel bilateralFilterKernel_1;
// cl_kernel bilateralFilterKernel_2;

cl_mem pt_in[2];
cl_mem pt_out[1];
cl_int status;

/*** Variables used as kernel arguments ***/
cl_mem input_buffer;
cl_mem output_buffer;
cl_mem gaussian_buffer;

// Input Array
float *input;
// Output Array
float *output;
// Filter Vector
float *gaussian;


/***********************************************************
 * Function:  load_file_to_memory
 * ------------------------------
 * Loads the hardware binary file (*.xclbin) into memory.
 * *********************************************************/
cl_uint load_file_to_memory(const char *filename, char **result)
{
    cl_uint size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        *result = NULL;
        return -1; // -1 means file opening fail
    }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *result = (char *)malloc(size+1);
    if (size != fread(*result, sizeof(char), size, f)) {
        free(*result);
        return -2; // -2 means file reading fail
    }
    fclose(f);
    (*result)[size] = 0;
    return size;
}


/***********************************************************
 * Function:  read_input
 * ---------------------------------------------------------
 * Reads the input.bin file, and loads it to input array.
 * *********************************************************/
void read_input(){
    FILE *fptr;

    /**** Load Input image ****/
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
    printf("MSE : %.6f\n", mse / (double) (SIZE_X*SIZE_Y));

    free(goldenOutput);
}

/***********************************************************
 * Function:  main
 * ---------------------------------------------------------
 * Main function.
 * *********************************************************/
int main(int argc, char *argv[]){
    // Input and output array size
    size_t buffer_size = sizeof(float) * SIZE_X * SIZE_Y;
	size_t n0,len;
    cl_uint iplat, n_i0;
    char *hw_binary_path,*kernelbinary;
    char buffer[2048];
    int argcounter,x,i;
    // Variables that will be used as kernel arguments
    int r;
    int size_x;
    int size_y;

    if ( argc < 2) {
        printf("1 Argument needed : <*.xclbin path>");
        return EXIT_FAILURE;
    }
    hw_binary_path = argv[1];

    /**********************************************
	 *
	 * 			Xilinx OpenCL Initialization
	 *
     * We must follow specific steps to get the necessary
     * information and handlers, in order to be able
     * to use the available accelerator device (FPGA).
     * After every step, we always check for any errors
     * that might have occured. In case of error, the
     * program aborts and exits immediately.
	 * *********************************************/



    /**************************************************
	* Step 1:
    * Get available OpenCL platforms and devices.
    * In our case, is a Xilinx FPGA device.
    * If the underlying platform has other accelerators
    * available, we could use them too (e.g. GPU, CPU).
	**************************************************/
	err = clGetPlatformIDs(16, platforms, &platform_count);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to find an OpenCL platform!\n");
        return EXIT_FAILURE;
    }

    printf("INFO: Found %d platforms\n", platform_count);

	for (iplat=0; iplat<platform_count; iplat++) {
			err = clGetPlatformInfo(platforms[iplat], CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor,NULL);
        if (err != CL_SUCCESS) {
            printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
            return EXIT_FAILURE;
        }
        /*** We are interested for Xilinx devices ***/
        if (strcmp(cl_platform_vendor, "Xilinx") == 0) {
            printf("INFO: Selected platform %d from %s\n", iplat, cl_platform_vendor);
            platform_id = platforms[iplat];
            platform_found = 1; // There is only one available.
        }
    }
    if (!platform_found) {
        printf("ERROR: Platform Xilinx not found. Exit.\n");
        return EXIT_FAILURE;
    }
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 16, devices, &num_devices);
    printf("INFO: Found %d devices\n", num_devices);
    if (err != CL_SUCCESS) {
        printf("ERROR: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

	device_id = devices[0]; // we have only one device

	// ---------------------------------------------------------------
	// Step 2 : Create Context
	// ---------------------------------------------------------------
	context = clCreateContext(0,1,&device_id,NULL,NULL,&err);
	if (!context) {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

	// ---------------------------------------------------------------
	// Step 3 : Create Command Queue
	// ---------------------------------------------------------------
	q = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    // ---------------------------------------------------------------
	// Step 3 for 2 compute units: Create Command Queue
	// ---------------------------------------------------------------
    // q = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
	if (!q) {
        printf("Error: Failed to create a command q! Error code: %i\n",err);
        return EXIT_FAILURE;
    }

	// ---------------------------------------------------------------
	// Step 4 : Load Hardware Binary File (*.xclbin) from disk
	// ---------------------------------------------------------------
    n_i0 = load_file_to_memory(hw_binary_path, (char **) &kernelbinary);
    if (n_i0 < 0) {
            printf("failed to load kernel from xclbin: %s\n", hw_binary_path);
            exit(EXIT_FAILURE);
        }
    n0 = n_i0;

    // ---------------------------------------------------------------
	// Step 5 : Create program using the loaded hardware binary file
	// ---------------------------------------------------------------
    program = clCreateProgramWithBinary(context, 1, &device_id, &n0,
                                        (const unsigned char **) &kernelbinary, &status, &err);
	free(kernelbinary);

    if ((!program) || (err!=CL_SUCCESS)) {
        printf("Error: Failed to create compute program from binary %d!\n", err);
        exit(EXIT_FAILURE);
    }

	// -------------------------------------------------------------
	//  Step 6 for 1 Compute Unit: Create Kernels - the actual handler of the kernel that
    //           we will be using. We first create a program, and then
    //           obtain the kernel handler from the program.
	// -------------------------------------------------------------
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(EXIT_FAILURE);
    }
	bilateralFilterKernel = clCreateKernel(program, "bilateralFilterKernel", &err);
    if (!bilateralFilterKernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute bilateralFilterKernel!\n");
		exit(EXIT_FAILURE);
    }
    //// -------------------------------------------------------------
	//  Step 6 for 2 Compute Units : Create Kernels - one handler for each
    //           instance of the kernel that we will be using. The program
    //           is the same as the step with 1 compute unit.
	// -------------------------------------------------------------
    // bilateralFilterKernel_1 = clCreateKernel(program, "bilateralFilterKernel:{bilateralFilterKernelInstance_1}", &err);
    // if (!bilateralFilterKernel_1 || err != CL_SUCCESS) {
    //     printf("Error: Failed to create compute krnl_bilateralFilterKernel_1!\n");
	// 	exit(EXIT_FAILURE);
    // }
	// err = 0;
	// bilateralFilterKernel_2 = clCreateKernel(program, "bilateralFilterKernel:{bilateralFilterKernelInstance_2}", &err);
	// if (! bilateralFilterKernel_2 || err != CL_SUCCESS) {
    //     printf("Error: Failed to create compute bilateralFilterKernel_2!\n");
	// 	exit(EXIT_FAILURE);
    // }


    // -------------------------------------------------------------------------
    // Step 7 : Create buffers.
    // We do not need to allocate separate memory space (malloc), because
    // on a MPSoC system (e.g. ZCU102 board), we map the memory space that is
    // allocated at clCreateBuffer, to a usable memory space for our
    // host application. We also do not need to use free for any reason.
    // See Xilinx UG1393 for detailed information.
    // ------------------------------------------------------------------------
    /*** Input image array buffer ***/
    input_buffer = clCreateBuffer(context,  CL_MEM_READ_ONLY,  buffer_size, NULL, &err);
    if (err != CL_SUCCESS) {
     printf("Return code for clCreateBuffer - input_buffer: %d",err);
    }
	input = (float *)clEnqueueMapBuffer(q,input_buffer,CL_TRUE,CL_MAP_WRITE,0,buffer_size,0,NULL,NULL,&err);

    /*** Output image array buffer ***/
    output_buffer = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,  buffer_size, NULL, &err);
    if (err != CL_SUCCESS) {
     printf("Return code for clCreateBuffer - output_buffer: %d",err);
    }
	output = (float *)clEnqueueMapBuffer(q,output_buffer,CL_TRUE,CL_MAP_READ,0,buffer_size,0,NULL,NULL,&err);

    /*** Filter vector buffer ***/
    gaussian_buffer = clCreateBuffer(context,  CL_MEM_READ_ONLY,  FILTER_SIZE*sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
     printf("Return code for clCreateBuffer - output_buffer: %d",err);
    }
	gaussian = (float *)clEnqueueMapBuffer(q,gaussian_buffer,CL_TRUE,CL_MAP_WRITE,0,FILTER_SIZE*sizeof(float),0,NULL,NULL,&err);

    /****
     * Data initialization
     * **************************/
    TICK();
    // Load input data to memory
    read_input();

    /**** Create filter vector using a suitable mathematical expression *****/
    for (i = 0; i < FILTER_SIZE; i++) {
		x = i - 2;
		gaussian[i] = expf(-(x * x) / (32.0f));
	}
    TOCK("load_time:");

    /*****
     * Kernel execution.
     * Note: We always check for errors.
     * **************************/

    /*****
     * In order to use multiple compute units,
     * consider what arguments earch kernel needs,
     * and how you must split the input and output
     * arrays.
     * ******************************************/
    TICK();

    // Set HW Kernel arguments
    r = FILTER_RADIUS;
    size_x = SIZE_X;
    size_y = SIZE_Y;
    argcounter = 0;
    err = 0;
	err |= clSetKernelArg(bilateralFilterKernel,argcounter++, sizeof(cl_mem), &output_buffer);
	err |= clSetKernelArg(bilateralFilterKernel,argcounter++, sizeof(cl_mem), &input_buffer);
	err |= clSetKernelArg(bilateralFilterKernel,argcounter++, sizeof(cl_mem), &gaussian_buffer);
   	err |= clSetKernelArg(bilateralFilterKernel,argcounter++, sizeof(int), &size_x);
	err |= clSetKernelArg(bilateralFilterKernel,argcounter++, sizeof(int), &size_y);
    err |= clSetKernelArg(bilateralFilterKernel,argcounter++, sizeof(int), &r);
    if (err != CL_SUCCESS) {
		printf("Error: Failed to set bilateralFilterKernel arguments! %d\n", err);
 	}

    // Enqueue input memory objects migration - Host -> Device
    pt_in[0] = input_buffer;
    pt_in[1] = gaussian_buffer;
    pt_out[0] = output_buffer;

    err = clEnqueueMigrateMemObjects(q,(cl_uint)2, pt_in, 0 ,0,NULL, NULL);
    if (err) {
        printf("Error: Failed to migrate memobjects to device! %d\n", err);
        return EXIT_FAILURE;
    }

    // Start kernel execution
	err = clEnqueueTask(q, bilateralFilterKernel, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel! %d\n", err);
        return EXIT_FAILURE;
    }

    // Enqueue output memory objects migration - Device -> Host
	err = clEnqueueMigrateMemObjects(q,(cl_uint)1, pt_out, CL_MIGRATE_MEM_OBJECT_HOST,0,NULL, NULL);
	if (err != CL_SUCCESS) {
        printf("Error: Failed to migrate membojects from device: %d!\n", err);
        return EXIT_FAILURE;
    }

    // Wait for execution to finish
	clFinish(q);
    TOCK("filter_time:");


    // Compare output results with golden
    TICK();
    compare();
    TOCK("compare_time:");


    /*****
     * Clean up code.
     * We always free any memory allocated, and release
     * all OpenCL objects.
     * ***********************************************/

    // Release memory objects and other OpenCL objects
    err = clEnqueueUnmapMemObject(q,input_buffer,input,0,NULL,NULL);
	if(err != CL_SUCCESS){
		printf("Error: Failed to unmap device memory input!\n");
	}
    err = clEnqueueUnmapMemObject(q,output_buffer,output,0,NULL,NULL);
	if(err != CL_SUCCESS){
		printf("Error: Failed to unmap device memory output!\n");
	}
    err = clEnqueueUnmapMemObject(q,gaussian_buffer,gaussian,0,NULL,NULL);
	if(err != CL_SUCCESS){
		printf("Error: Failed to unmap device memory gaussian!\n");
	}
    clFinish(q);


    clReleaseMemObject(input_buffer);
	clReleaseMemObject(output_buffer);
	clReleaseMemObject(gaussian_buffer);

	clReleaseProgram(program);
    clReleaseKernel(bilateralFilterKernel);
    clReleaseCommandQueue(q);
    clReleaseContext(context);

    return 0;

}