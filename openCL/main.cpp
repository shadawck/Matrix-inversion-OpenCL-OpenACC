// System includes
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <filesystem>
// OpenCL includes
//#include <OpenCL/cl.h>
#include <CL/cl.h>
#include "Matrix.hpp"

// Project includes

// Constants, globals

// namespace
using namespace std;

// Signatures
char *readSource(const char *sourceFilename);

double *convertValArrayToDouble(valarray<double> array);

double **MatrixTo2DArray(Matrix mat);

Matrix arrayToMatrix(double *array, int size);

int main(int argc, char **argv) {
    srand((unsigned) time(nullptr));

    printf("Running Matrix Inversion program\n\n");

    int matrixDimension = 5;
    if (argc == 2) {
        matrixDimension = atoi(argv[1]);
    }

    size_t datasize = sizeof(double) * matrixDimension * matrixDimension;

    MatrixRandom randomMatrix(matrixDimension, matrixDimension);
    const Matrix &copyRandomMatrix(randomMatrix);

    /* Can't use 2D array like in openACC. */
    double *newMat = convertValArrayToDouble(randomMatrix.getDataArray());
    double *eyeResMat = convertValArrayToDouble(MatrixIdentity(randomMatrix.rows()).getDataArray());
    int size = randomMatrix.rows();
    cout << "Size on host :" << size << endl;

    if (newMat == nullptr || eyeResMat == nullptr) {
        perror("malloc");
        exit(-1);
    }

    cl_int status;  // use as return value for most OpenCL functions

    cl_uint numPlatforms = 0;
    cl_platform_id *platforms;

    // Query for the number of recongnized platforms
    status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (status != CL_SUCCESS) {
        printf("clGetPlatformIDs failed\n");
        exit(-1);
    }

    // Make sure some platforms were found
    if (numPlatforms == 0) {
        printf("No platforms detected.\n");
        exit(-1);
    }

    // Allocate enough space for each platform
    platforms = (cl_platform_id *) malloc(numPlatforms * sizeof(cl_platform_id));
    if (platforms == nullptr) {
        perror("malloc");
        exit(-1);
    }

    // Fill in platforms
    clGetPlatformIDs(numPlatforms, platforms, nullptr);
    if (status != CL_SUCCESS) {
        printf("clGetPlatformIDs failed\n");
        exit(-1);
    }

    // Print out some basic information about each platform
    printf("%u platforms detected\n", numPlatforms);
    for (unsigned int i = 0; i < numPlatforms; i++) {
        char buf[100];
        printf("Platform %u: \n", i);
        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
                                   sizeof(buf), buf, nullptr);
        printf("\tVendor: %s\n", buf);
        status |= clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
                                    sizeof(buf), buf, nullptr);
        printf("\tName: %s\n", buf);

        if (status != CL_SUCCESS) {
            printf("clGetPlatformInfo failed\n");
            exit(-1);
        }
    }
    printf("\n");

    cl_uint numDevices = 0;
    cl_device_id *devices;

    // Retrive the number of devices present
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, nullptr,
                            &numDevices);

    if (status != CL_SUCCESS) {
        printf("clGetDeviceIDs failed\n");
        exit(-1);
    }

    // Make sure some devices were found
    if (numDevices == 0) {
        printf("No devices detected.\n");
        exit(-1);
    }

    // Allocate enough space for each device
    devices = (cl_device_id *) malloc(numDevices * sizeof(cl_device_id));
    if (devices == nullptr) {
        perror("malloc");
        exit(-1);
    }

    // Fill in devices
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices,
                            devices, nullptr);
    if (status != CL_SUCCESS) {
        printf("clGetDeviceIDs failed\n");
        exit(-1);
    }

    // Print out some basic information about each device
    printf("%u devices detected\n", numDevices);
    for (unsigned int i = 0; i < numDevices; i++) {
        char buf[100];
        printf("Device %u: \n", i);
        status = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR,
                                 sizeof(buf), buf, nullptr);
        printf("\tDevice: %s\n", buf);
        status |= clGetDeviceInfo(devices[i], CL_DEVICE_NAME,
                                  sizeof(buf), buf, nullptr);
        printf("\tName: %s\n", buf);

        if (status != CL_SUCCESS) {
            printf("clGetDeviceInfo failed\n");
            exit(-1);
        }
    }
    printf("\n");

    cl_context context;

    // Create a context and associate it with the devices
    context = clCreateContext(nullptr, numDevices, devices, nullptr, nullptr, &status);
    if (status != CL_SUCCESS || context == nullptr) {
        printf("clCreateContext failed\n");
        exit(-1);
    }

    cl_command_queue cmdQueue;

    // Create a command queue and associate it with the device you
    // want to execute on
    cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
    if (status != CL_SUCCESS || cmdQueue == nullptr) {
        printf("clCreateCommandQueue failed\n");
        exit(-1);
    }

    cl_mem d_newMat;       // Input buffers on device
    cl_mem d_eyeResMat;    // Output buffer on device
    cl_int d_size;         // Matrix Dimension (size*size)

    // Create a buffer object (d_newMat) that contains the data from the host ptr newMat
    d_newMat = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,
                              datasize, newMat, &status);
    if (status != CL_SUCCESS || d_newMat == nullptr) {
        printf("clCreateBuffer failed\n");
        exit(-1);
    }

    // Create a buffer object (d_eyeResMat) with enough space to hold the output data
    d_eyeResMat = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,
                                 datasize, eyeResMat, &status);
    if (status != CL_SUCCESS || d_eyeResMat == nullptr) {
        printf("clCreateBuffer failed\n");
        exit(-1);
    }


    cl_program program;

    char *source;
    cout << filesystem::current_path() << endl;

    const char *sourceFile = "inversion.cpp";
    // This function reads in the source code of the program
    source = readSource(sourceFile);

    cout << source << endl;

    // Create a program. The 'source' string is the code from the
    // vectoradd.cl file.
    program = clCreateProgramWithSource(context, 1, (const char **) &source,
                                        nullptr, &status);


    if (status != CL_SUCCESS) {
        printf("clCreateProgramWithSource failed\n");
        exit(-1);
    }

    cl_int buildErr;
    // Build (compile & link) the program for the devices.
    // Save the return value in 'buildErr' (the following
    // code will print any compilation errors to the screen)
    buildErr = clBuildProgram(program, numDevices, devices, nullptr, nullptr, nullptr);

    // If there are build errors, print them to the screen
    if (buildErr != CL_SUCCESS) {
        printf("Program failed to build.\n");
        cl_build_status buildStatus;
        for (unsigned int i = 0; i < numDevices; i++) {
            clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_STATUS,
                                  sizeof(cl_build_status), &buildStatus, nullptr);
            if (buildStatus == CL_SUCCESS) {
                continue;
            }

            char *buildLog;
            size_t buildLogSize;
            clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
                                  0, nullptr, &buildLogSize);
            buildLog = (char *) malloc(buildLogSize);
            if (buildLog == nullptr) {
                perror("malloc");
                exit(-1);
            }
            clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
                                  buildLogSize, buildLog, nullptr);
            buildLog[buildLogSize - 1] = '\0';
            printf("Device %u Build Log:\n%s\n", i, buildLog);
            free(buildLog);
        }
        exit(0);
    } else {
        printf("No build errors\n");
    }

    cl_kernel kernel;

    // Create a kernel from the vector addition function (named "inversion")
    kernel = clCreateKernel(program, "inversion", &status);
    if (status != CL_SUCCESS) {
        printf("clCreateKernel failed\n");
        exit(-1);
    }

    // Associate the input and output buffers with the kernel
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_newMat);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_eyeResMat);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_int), &size);
    if (status != CL_SUCCESS) {
        printf("clSetKernelArg failed\n");
        exit(-1);
    }

    // Define an index space (global work size) of threads for execution.
    // A workgroup size (local work size) is not required, but can be used.
    size_t globalWorkSize[1];  // There are ELEMENTS threads
    globalWorkSize[0] = matrixDimension;
    // Determine a way to size workgroup according to some logic
//    size_t localWorkSize[1];
//    localWorkSize[0] = 8;

    // Execute the kernel.
    // 'globalWorkSize' is the 1D dimension of the work-items
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, nullptr, globalWorkSize,
                                    nullptr, 0, nullptr, nullptr);

    if (status != CL_SUCCESS) {
        printf("clEnqueueNDRangeKernel failed\n");
        exit(-1);
    }

    clEnqueueReadBuffer(cmdQueue, d_newMat, CL_TRUE, 0, datasize, newMat,
                        0, nullptr, nullptr);

    // Read the OpenCL output buffer (d_eyeResMat) to the host output array (C)
    clEnqueueReadBuffer(cmdQueue, d_eyeResMat, CL_TRUE, 0, datasize, eyeResMat,
                        0, nullptr, nullptr);


    cout << arrayToMatrix(newMat, size).str() << endl;


    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(d_newMat);
    clReleaseMemObject(d_eyeResMat);
    clReleaseContext(context);

    free(newMat);
    free(eyeResMat);
    free(source);
    free(platforms);
    free(devices);
}

Matrix arrayToMatrix(double *array, int size) {
    Matrix resMatrix(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            resMatrix(i, j) = array[size * i + j];
        }
    }
    return resMatrix;
}

char *readSource(const char *sourceFilename) {

    FILE *fp;
    int err;
    int size;

    char *source;

    fp = fopen(sourceFilename, "rb");
    if (fp == nullptr) {
        printf("Could not open kernel file: %s\n", sourceFilename);
        exit(-1);
    }

    err = fseek(fp, 0, SEEK_END);
    if (err != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);
    }

    size = ftell(fp);
    if (size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }

    err = fseek(fp, 0, SEEK_SET);
    if (err != 0) {
        printf("Error seeking to start of file\n");
        exit(-1);
    }

    source = (char *) malloc(size + 1);
    if (source == nullptr) {
        printf("Error allocating %d bytes for the program source\n", size + 1);
        exit(-1);
    }

    err = fread(source, 1, size, fp);
    if (err != size) {
        printf("only read %d bytes\n", err);
        exit(0);
    }

    source[size] = '\0';

    return source;
}

double *convertValArrayToDouble(valarray<double> array) {
    auto *newArray = new double[array.size()];
    copy(begin(array), end(array), newArray);
    return newArray;
}

double **MatrixTo2DArray(Matrix mat) {
    /// Allocate
    auto **newArray = (double **) malloc(mat.rows() * sizeof(double *));
    for (int i = 0; i < mat.rows(); i++)
        newArray[i] = (double *) malloc(mat.cols() * sizeof(double));

    for (int i = 0; i < mat.rows(); i++)
        for (int j = 0; j < mat.cols(); j++)
            newArray[i][j] = mat(i, j);
    return newArray;
}