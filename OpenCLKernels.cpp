/* 
 * File:   OpenCLKernels.cpp
 * Author: jdelatorre
 * 
 * Created on 23 de octubre de 2014, 10:25
 */

#include <string>
#include <vector>
#include <iostream>
#include <boost/math/common_factor.hpp>

#include "OpenCLKernels.hpp"
#include "common.hpp"

OpenCLKernels::~OpenCLKernels() {
    delete rowSumKernel;
    delete softmaxKernelLocal;
    delete elementWiseMultiplicationBySigmoidDerivativeKernel;
    delete crossEntropyKernelLocal;
    delete level2RegularizationKernelLocal;
    delete elementWiseSubstractKernel;
    delete elementWiseSumKernel;
    delete matrixMultiplicationSigmoidKernel;
    delete program;
}

void OpenCLKernels::opencl_init() {
    // create a CL program using kernel source
    std::string sourceString;
    readfile(sourceFile, sourceString);
    
    cl::Program::Sources sources;
    sources.push_back(std::make_pair(sourceString.c_str(), 0));
    // don't need to specify length as we used a null terminated string

    // create the OpenCL program
    program = new cl::Program(context, sources);
    
    try {
        program->build(devices);
    } catch(const cl::Error &e) {
        // get compilation log in case of failure
     std::cout << "Build Status: "
        << program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[device_id])
        << std::endl;
     std::cout << "Build Options:\t"
        << program->getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[device_id])
        << std::endl;
     std::cout << "Build Log:\t "
        << program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[device_id])
        << std::endl;
    }
    
    lds = true;
    try {
      matrixMultiplicationSigmoidKernel =
          new cl::Kernel(*program,
                           matrixMultiplicationSigmoidKernel_name.c_str());
      elementWiseSubstractKernel =
            new cl::Kernel(*program,
                           elementWiseSubstractKernel_name.c_str());

      elementWiseSumKernel =
            new cl::Kernel(*program,
                           elementWiseSumKernel_name.c_str());
            
      crossEntropyKernelLocal =
            new cl::Kernel(*program,
                           crossEntropyKernelLocal_name.c_str());
      
      level2RegularizationKernelLocal =
            new cl::Kernel(*program,
                           level2RegularizationKernelLocal_name.c_str());
      
      elementWiseMultiplicationBySigmoidDerivativeKernel =
            new cl::Kernel(*program,
                           elementWiseMultiplicationBySigmoidDerivativeKernel_name.c_str());
      softmaxKernelLocal =
              new cl::Kernel(*program,
                             softmaxKernelLocal_name.c_str());
      
      rowSumKernel =
              new cl::Kernel(*program,
                             rowSumKernel_name.c_str());
      
    } catch(const cl::Error &e) {
        std::cout << e.err() << e.what() << std::endl;
    }
}

/*
 * Requirements to use this function: All the sizes must be multiple of 16. TESTED (OK)
 * 
 * setBias = true --> Fixes the first column to 1 (used when calculating activations in order
 * to use the first neuron of the layer as bias (output = 1.0 always)
 * 
 * calcSigmoid = true --> After multiplying A*B calculates the sigmoid of the result
 * 
 * sumToC = true --> instead of assigning the result of A*B to C, makes the next operation:
 * 
 *  C = multPrevVal * Cprevious + multSum * Ccalculated
 */
 void OpenCLKernels::
     runMatrixMultiplicationSigmoid(matrix_cl_float const &A,
                                    matrix_cl_float const &B,
                                    matrix_cl_float const &C,
                                    matrix_cl_float *bias,
                                    bool calcSigmoid,
                                    bool sumToC,
                                    cl_float multPrevVal,
                                    cl_float multSum,
                                    matrix_cl_uchar const * dropconnectB,
                                    float dropconnectInferenceProbability) {  
     // It's correct, cols and rows are in this order
    const size_t global_size[2] = {size_t(C.cols/4),
                                   size_t(C.rows/4)};
    
    // Check size compatibility
    assert(C.rows == A.rows && C.cols == B.cols && A.cols == B.rows);
    
    // Check A and B sizes are multiple of 16
    assert((global_size[0] % 4 == 0 ) && (global_size[1] % 4 == 0));
    
    size_t blocksize = 4;
    
    // if possible use a greater blocksize
    if((global_size[0] % 8 == 0) &&
       (global_size[1] % 8 == 0) && 
       ((A.cols/4) % 8) == 0)
        blocksize = 8;
    else if(global_size[0] == 2 || global_size[1] == 2)
        blocksize = 2;
    else if(global_size[0] == 1 || global_size[1] == 1)
        blocksize = 1;
    
    // float4 elements in kernel
    const size_t local_size[2] = { blocksize, blocksize };

    // -----------------------------------------------------------------------
    // Setting kernel arguments
    // -----------------------------------------------------------------------
    matrixMultiplicationSigmoidKernel->setArg(0, *(A.data.deviceData));
    matrixMultiplicationSigmoidKernel->setArg(1, *(B.data.deviceData));
    matrixMultiplicationSigmoidKernel->setArg(2, *(C.data.deviceData));
    matrixMultiplicationSigmoidKernel->setArg(3, 
      (bias==nullptr)?cl::Buffer(0):*(bias->data.deviceData));
    matrixMultiplicationSigmoidKernel->setArg(4, A.cols);
    matrixMultiplicationSigmoidKernel->setArg(5, A.offset/4);
    matrixMultiplicationSigmoidKernel->setArg(6, B.offset/4);
    matrixMultiplicationSigmoidKernel->setArg(7, C.offset/4);
    matrixMultiplicationSigmoidKernel->setArg(8, 
      (bias==nullptr)?0:bias->offset/4);
    matrixMultiplicationSigmoidKernel->setArg(9,
          cl::Local((blocksize*4)*(blocksize*4)*sizeof(cl_float)));
    matrixMultiplicationSigmoidKernel->setArg(10,
          calcSigmoid?1:0);    // calculate sigmoid after matrix multiplication
    matrixMultiplicationSigmoidKernel->setArg(11,
          A.colMajorOrdered?1:0);    // A in column-major order
    matrixMultiplicationSigmoidKernel->setArg(12,
          B.colMajorOrdered?1:0);    // B in column-major order
    matrixMultiplicationSigmoidKernel->setArg(13,
          sumToC?1:0);    // Result should be sumed to previous value of C or only assigned
    matrixMultiplicationSigmoidKernel->setArg(14,
          multPrevVal); // If sumToC== true value that multiplies the result previous to sum
    matrixMultiplicationSigmoidKernel->setArg(15,
          multSum); // If sumToC== true value that multiplies the result previous to sum
    matrixMultiplicationSigmoidKernel->setArg(16, 
      (dropconnectB==nullptr)?cl::Buffer(0):*(dropconnectB->data.deviceData));
    matrixMultiplicationSigmoidKernel->setArg(17, 
      dropconnectInferenceProbability);
    
    // -----------------------------------------------------------------------
    // Define ndrange iteration space: global and local sizes based on
    // parameters obtained from user

    // Refer to the sample documentation for clarification about
    // how work is devided among work-groups and work-items.
    // -----------------------------------------------------------------------
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0], global_size[1]);
    const cl::NDRange local(local_size[0], local_size[1]);
    queue.enqueueNDRangeKernel(*matrixMultiplicationSigmoidKernel,
                               offset,
                               global,
                               local);
    queue.finish();
}


void OpenCLKernels::runElementWiseSubstract(
            matrix_cl_float const &tm,
            matrix_cl_float const &ym,
            matrix_cl_float &em) {

    assert(tm.cols == ym.cols && tm.rows == ym.rows &&
           tm.cols == em.cols && tm.rows == em.rows);
    
    //const size_t blockSize = 512;  // float4's
    const size_t data_size_float4_global = ym.rows*ym.cols/4;
    
    size_t global_size[1] = {data_size_float4_global};

    elementWiseSubstractKernel->setArg(0, *(tm.data.deviceData));
    elementWiseSubstractKernel->setArg(1, *(ym.data.deviceData));
    elementWiseSubstractKernel->setArg(2, *(em.data.deviceData));
    elementWiseSubstractKernel->setArg(3, tm.offset/4);
    elementWiseSubstractKernel->setArg(4, ym.offset/4);
    elementWiseSubstractKernel->setArg(5, em.offset/4);
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0]);
    //const cl::NDRange local(local_size[0]);
    queue.enqueueNDRangeKernel(*elementWiseSubstractKernel,
                               offset,
                               global/*,
                               local*/);
    queue.finish();
}

void OpenCLKernels::runElementWiseSum(
            matrix_cl_float const &a,
            matrix_cl_float const &b,
            matrix_cl_float &c,
            cl_float mult_a,
            cl_float mult_b) {

    assert(a.cols == b.cols && a.rows == b.rows &&
           a.cols == c.cols && a.rows == c.rows);
    
    //const size_t blockSize = 512;  // float4's
    const size_t data_size_float4_global = b.rows*b.cols/4;
    
    size_t global_size[1] = {data_size_float4_global};
    //size_t local_size[1] = {boost::math::gcd(blockSize, global_size[0])};
    
    //assert(global_size[0] % local_size[0] == 0);

    elementWiseSumKernel->setArg(0, *(a.data.deviceData));
    elementWiseSumKernel->setArg(1, *(b.data.deviceData));
    elementWiseSumKernel->setArg(2, *(c.data.deviceData));
    elementWiseSumKernel->setArg(3, a.offset/4);
    elementWiseSumKernel->setArg(4, b.offset/4);
    elementWiseSumKernel->setArg(5, c.offset/4);
    elementWiseSumKernel->setArg(6, mult_a);
    elementWiseSumKernel->setArg(7, mult_b);
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0]);
    //const cl::NDRange local(local_size[0]);
    queue.enqueueNDRangeKernel(*elementWiseSumKernel,
                               offset,
                               global/*,
                               local*/);
    queue.finish();
}

// NOT TESTED YET
void OpenCLKernels::runElementWiseMultiplicationBySigmoidDerivativeKernel(
            matrix_cl_float const &deltas,
            matrix_cl_float const &activations) {

    assert(deltas.cols == activations.cols
           && deltas.rows == activations.rows);
    
    //const size_t blockSize = 512;  // float4's
    const size_t data_size_float4_global = deltas.rows*deltas.cols/4;
    
    size_t global_size[1] = {data_size_float4_global};
    //size_t local_size[1] = {boost::math::gcd(blockSize, global_size[0])};
    
    //assert(global_size[0] % local_size[0] == 0);

    elementWiseMultiplicationBySigmoidDerivativeKernel->
        setArg(0, *(deltas.data.deviceData));
    elementWiseMultiplicationBySigmoidDerivativeKernel->
        setArg(1, *(activations.data.deviceData));
    elementWiseMultiplicationBySigmoidDerivativeKernel->
        setArg(2, deltas.offset/4);
    elementWiseMultiplicationBySigmoidDerivativeKernel->
        setArg(3, activations.offset/4);
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0]);
    //const cl::NDRange local(local_size[0]);
    queue.enqueueNDRangeKernel(
        *elementWiseMultiplicationBySigmoidDerivativeKernel,
        offset,
        global/*,
        local*/);
    queue.finish();
}

cl_float OpenCLKernels::runCrossEntropy(matrix_cl_float const &t,
                                        matrix_cl_float const &y,
                                        matrix_cl_float &error) {
    // proposed blockSize
    const size_t blockSize = 512;  // float4's (8kBytes)
    
    const size_t data_size_float4_global = y.rows*y.cols/4;

    size_t global_size[1] = {data_size_float4_global / 2};
    // global_size es múltiplo de 8. 
    // Aprovechamos éste hecho para fijar local_size
    size_t local_size[1] = {8};
    
    if(global_size[0] <= blockSize) {
        local_size[0] = global_size[0];
    } else {
        size_t resto = global_size[0] / 8;  // sabemos que es divisible
        if(resto <= blockSize) {
            local_size[0] = resto;
        } 
    }
    
    
    
    assert(data_size_float4_global * 4 <= error.data.hostData.size());    
    //assert(global_size[0] % local_size[0] == 0);
    
    // -----------------------------------------------------------------------
    // Setting kernel arguments
    // -----------------------------------------------------------------------
    crossEntropyKernelLocal->setArg(0, *(t.data.deviceData));
    crossEntropyKernelLocal->setArg(1, *(y.data.deviceData));
    crossEntropyKernelLocal->setArg(2, *(error.data.deviceData));
    crossEntropyKernelLocal->setArg(3,
                           cl::Local(local_size[0] * 4 * sizeof(cl_float)));
    crossEntropyKernelLocal->setArg(4, y.offset/4);

    // -----------------------------------------------------------------------
    // Define ndrange iteration space: global and local sizes based on
    // parameters obtained from user

    // Refer to the sample documentation for clarification about
    // how work is devided among work-groups and work-items.
    // -----------------------------------------------------------------------
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0]);
    const cl::NDRange local(local_size[0]);
    queue.enqueueNDRangeKernel(*crossEntropyKernelLocal, offset, global, local);
    queue.finish();

    //std::cout << "CE kernel finished\n";
    
    error.data.readFromDevice(queue);

    const size_t error_size = 4 * global_size[0]/local_size[0];
    std::vector<cl_float> & e = error.data.hostData;
    cl_float ce = 0.0;
    for (size_t i = 0; i < error_size; i++) {
        ce += e[i];
    }
    
    //return -ce/(y.rows*y.cols);
    return -ce/(y.rows);
}

cl_float OpenCLKernels::runL2Regularization(matrix_cl_float const &weights,
                                            matrix_cl_float &error) {
    // proposed blockSize
    const size_t blockSize = 512;  // float4's (8kBytes)
    
    const size_t data_size_float4_global = weights.rows*weights.cols/4;

    size_t global_size[1] = {data_size_float4_global / 2};
    // global_size es múltiplo de 8. 
    // Aprovechamos éste hecho para fijar local_size
    size_t local_size[1] = {8};
    
    if(global_size[0] <= blockSize) {
        local_size[0] = global_size[0];
    } else {
        size_t resto = global_size[0] / 8;  // sabemos que es divisible
        if(resto <= blockSize) {
            local_size[0] = resto;
        } 
    }
    
    assert(data_size_float4_global * 4 <= error.data.hostData.size());    
    //assert(global_size[0] % local_size[0] == 0);
    
    // -----------------------------------------------------------------------
    // Setting kernel arguments
    // -----------------------------------------------------------------------
    level2RegularizationKernelLocal->setArg(0, *(weights.data.deviceData));
    level2RegularizationKernelLocal->setArg(1, *(error.data.deviceData));
    level2RegularizationKernelLocal->setArg(2,
                            cl::Local(local_size[0] * 4 * sizeof(cl_float)));

    // -----------------------------------------------------------------------
    // Define ndrange iteration space: global and local sizes based on
    // parameters obtained from user

    // Refer to the sample documentation for clarification about
    // how work is devided among work-groups and work-items.
    // -----------------------------------------------------------------------
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0]);
    const cl::NDRange local(local_size[0]);
    queue.enqueueNDRangeKernel(*level2RegularizationKernelLocal, offset, global, local);
    queue.finish();

    //std::cout << "CE kernel finished\n";
    
    error.data.readFromDevice(queue);

    const size_t error_size = 4 * global_size[0]/local_size[0];
    std::vector<cl_float> & e = error.data.hostData;
    cl_float sumsqr = 0.0;
    for (size_t i = 0; i < error_size; i++) {
        sumsqr += e[i];
    }
    
    return sumsqr;
}


void OpenCLKernels::runSoftMax(
            matrix_cl_float const &activations) {
  
    assert(activations.cols % 4 == 0 && activations.rows % 4 == 0);
    
    size_t local_size[1] {activations.cols / 4};
    size_t global_size[1] = {local_size[0] * activations.rows};
    
    softmaxKernelLocal->setArg(0, *(activations.data.deviceData));
    softmaxKernelLocal->setArg(1,
                               cl::Local(local_size[0] * 4 * sizeof(cl_float)));
    softmaxKernelLocal->setArg(2, activations.offset/4);
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0]);
    const cl::NDRange local(local_size[0]);
    queue.enqueueNDRangeKernel(*softmaxKernelLocal,
                               offset,
                               global,
                               local);
    queue.finish();
}

void OpenCLKernels::runRowSum(
            matrix_cl_float &A, 
            matrix_cl_float &result,
            cl_float multExisting,
            cl_float multNew) {
    
    size_t global_size[1] = {A.cols/4};
    
    rowSumKernel->setArg(0, *(A.data.deviceData));
    rowSumKernel->setArg(1, *(result.data.deviceData));
    rowSumKernel->setArg(2, A.rows);
    rowSumKernel->setArg(3, multExisting);
    rowSumKernel->setArg(4, multNew);
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0]);
    queue.enqueueNDRangeKernel(*rowSumKernel,
                               offset,
                               global);
    queue.finish();
}
