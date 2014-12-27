/* 
 * File:   minibatch_generator.hpp
 * Author: jdelatorre
 *
 * Created on 26 de noviembre de 2014, 10:56
 */


#ifndef MG_HPP
#define	MG_HPP

#include <vector>
#include <random>

#include <CL/cl.hpp>

class minibatch_generator {
    std::mt19937 gen;
    
    cl_uint sourceSize;
    cl_uint destSize;
    std::vector<bool> selected;
    std::vector<cl_uint> index;
    
    std::vector<cl_uint> minibatch;
    
    std::vector<cl_float> &from1;
    std::vector<cl_float> &to1;
    cl_uint stride1;
    std::vector<cl_float> &from2;
    std::vector<cl_float> &to2;
    cl_uint stride2;
    
    void generate();
public:
    minibatch_generator(cl_uint total_data, 
                        cl_uint minibatch_size,
                        std::vector<cl_float> &from1,
                        std::vector<cl_float> &to1,
                        cl_uint stride1,
                        std::vector<cl_float> &from2,
                        std::vector<cl_float> &to2,
                        cl_uint stride2
                       );
   
    void load_generated_minibatch();
};

#endif	/* MINIBATCH_GENERATOR_HPP */

