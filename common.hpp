/* 
 * File:   common.hpp
 * Author: jdelatorre
 *
 * Created on 23 de octubre de 2014, 12:36
 */

#ifndef COMMON_HPP
#define COMMON_HPP

#include <CL/cl.hpp>
#include <boost/tokenizer.hpp>
#include <boost/format.hpp>
#include <cassert>
#include <vector>
#include <string>

#include <fstream>
#include <iostream>

template<typename T>
struct host_device_memory_map {
  std::vector<T> & hostData;
  cl::Buffer * deviceData = nullptr;
  
  explicit inline host_device_memory_map(std::vector<T> & v) : hostData(v) {}
  
  inline host_device_memory_map(const host_device_memory_map<T> & orig) :
                                hostData(orig.hostData),
                                deviceData(orig.deviceData) {}

  inline void createBuffer(const cl::Context & context,
                           const cl_mem_flags flags) {
    deviceData = new cl::Buffer(context,
                                flags,
                                hostData.size()*sizeof(T),
                                &hostData[0]);
  }
  
  inline void readFromDevice(const cl::CommandQueue & queue) {
      queue.enqueueReadBuffer(*deviceData,
                              CL_TRUE,
                              0,
                              hostData.size()*sizeof(T),
                              &hostData[0]);
      queue.finish();
  }

  inline void writeToDevice(const cl::CommandQueue & queue, size_t bytes = 0) {
      // If bytes == 0 writes the whole size
      const size_t write_size = (bytes==0)?hostData.size()*sizeof(T):bytes;
      queue.enqueueWriteBuffer(*deviceData,
                               CL_TRUE, 
                               0,
                               write_size,
                               &hostData[0]);
      queue.finish();
  }
  
  inline ~host_device_memory_map() {
      if (deviceData != nullptr) delete deviceData;
  }
};

template <typename T>
struct opencl_matrix {
    host_device_memory_map<T> & data;
    cl_uint rows;
    cl_uint cols;
    cl_uint offset;
    bool colMajorOrdered = false;   // default is row major
    
    explicit inline opencl_matrix(host_device_memory_map<T> & d) :
                         data(d), rows(0), cols(0), offset(0) {}
    
    inline opencl_matrix(const opencl_matrix<T> & orig) :
                         data(orig.data), rows(orig.rows),
                         cols(orig.cols), offset(orig.offset) {}
    
    inline opencl_matrix const & set(cl_uint r,
                                     cl_uint c,
                                     cl_uint o = 0,
                                     bool matrixInColMajorOrder = false) {
        rows = r;
        cols = c;
        offset = o;
        colMajorOrdered = matrixInColMajorOrder;
        return *this;
    }
};

typedef opencl_matrix<cl_float> matrix_cl_float;
typedef opencl_matrix<cl_uchar> matrix_cl_uchar;

void load_nn_data(const std::string & filename,
                   cl_uint &layers,
                   std::vector<cl_uint> &elements);

void load_csv_data(const std::string & filename,
                   std::vector<cl_float> & input,
                   std::vector<cl_float> & output,
                   cl_uint &rows,
                   cl_uint in_elements,
                   cl_uint out_elements);

void load_csv_vector(const std::string & filename,
                     std::vector<cl_float> &weights);

void save_csv_vector(const std::string & filename,
                     std::vector<cl_float> &weights);

void print_vector(const std::vector<cl_float> & v,
                  cl_uint rows,
                  cl_uint cols,
                  cl_uint offset);

void print(const matrix_cl_float &m,
           const std::string header = "",
           const bool rows2cols = false);

void save_NN(const std::string filename);

#endif  /* COMMON_HPP */

