/*
 * File:   OpenCLKernels.hpp
 * Author: jdelatorre
 *
 * Created on 23 de octubre de 2014, 10:25
 */

#define __CL_ENABLE_EXCEPTIONS  // enable use of exceptions of the OpenCL API


#ifndef OPENCLKERNELS_HPP
#define OPENCLKERNELS_HPP

#include <string>
#include <fstream>
#include <map>

#include "CL/cl.hpp"

#include "common.hpp"


class OpenCLKernels {
 public:
    inline OpenCLKernels(
            const cl::Context & c,
            const std::vector<cl::Device> &d,
            const int d_id,
            const cl::CommandQueue & q)
            : context(c), devices(d), device_id(d_id), queue(q) {
        opencl_init();
    };

    virtual ~OpenCLKernels();
    
    void runMatrixMultiplicationSigmoid(
            matrix_cl_float const &A,
            matrix_cl_float const &B,
            matrix_cl_float const &C,
            matrix_cl_float * bias = nullptr,
            bool calcSigmoid = false,
            bool sumToC = false,
            cl_float multPrevVal = 1.0f,
            cl_float multSum = 1.0f,
            matrix_cl_uchar const * dropconnectB = nullptr,
            float dropconnectInferenceProbability = 1.0f);
    
    void runElementWiseSubstract(
            matrix_cl_float const &t,
            matrix_cl_float const &y,
            matrix_cl_float &e);
    
    void runElementWiseSum(
            matrix_cl_float const &a,
            matrix_cl_float const &b,
            matrix_cl_float &c,
            cl_float mult_a = 1.0f,
            cl_float mult_b = 1.0f);
    
    
    cl_float runCrossEntropy(
            matrix_cl_float const &t,
            matrix_cl_float const &y,
            matrix_cl_float &error);
    
    cl_float runL2Regularization(
            matrix_cl_float const &weights,
            matrix_cl_float &error);
        
    void runElementWiseMultiplicationBySigmoidDerivativeKernel(
            matrix_cl_float const &deltas,
            matrix_cl_float const &activations);
    
    void runSoftMax(
            matrix_cl_float const &activations);
    
    void runRowSum(
            matrix_cl_float &A, 
            matrix_cl_float &result,
            cl_float multExisting = 0.0f,
            cl_float multNew = 1.0f);
  private:
    const std::string sourceFile = "NN_Kernels.cl";
    
    const cl::Context & context;
    const std::vector<cl::Device> & devices;
    const int device_id;
    const cl::CommandQueue & queue;
    
    cl::Program *program;
    
    // kernels
    
    cl::Kernel *matrixMultiplicationSigmoidKernel;
    const std::string matrixMultiplicationSigmoidKernel_name =
                      "matrixMultiplicationSigmoidKernelLocal";
    
    cl::Kernel *elementWiseSubstractKernel;
    const std::string elementWiseSubstractKernel_name =
                      "elementWiseSubstractKernel";
    
    cl::Kernel *elementWiseSumKernel;
    const std::string elementWiseSumKernel_name =
                      "elementWiseSumKernel";
    
    cl::Kernel *crossEntropyKernelLocal;
    const std::string crossEntropyKernelLocal_name =
                      "crossEntropyKernelLocal";
    
    cl::Kernel *level2RegularizationKernelLocal;
    const std::string level2RegularizationKernelLocal_name =
                      "level2RegularizationKernelLocal";
    
    cl::Kernel *elementWiseMultiplicationBySigmoidDerivativeKernel;
    const std::string elementWiseMultiplicationBySigmoidDerivativeKernel_name =
                      "elementWiseMultiplicationBySigmoidDerivativeKernel";
    
    cl::Kernel *softmaxKernelLocal;
    const std::string softmaxKernelLocal_name =
                      "softmaxKernelLocal";
    
    cl::Kernel *rowSumKernel;
    const std::string rowSumKernel_name =
                      "rowSumKernel";    
    
    bool lds;
    
    inline void readfile(const std::string &filepath, std::string &buffer) {
        std::ifstream fin(filepath.c_str());
        getline(fin, buffer, char(-1));
        fin.close();
    };
    
    void opencl_init();
    
};

#endif  /* MATRIXMULTIPLICATION_HPP */

