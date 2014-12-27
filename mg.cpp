#include "mg.hpp"

#include <vector>
#include <chrono>


minibatch_generator::minibatch_generator(cl_uint total_data, 
                                         cl_uint minibatch_size,
                                         std::vector<cl_float> &from1,
                                         std::vector<cl_float> &to1,
                                         cl_uint stride1,
                                         std::vector<cl_float> &from2,
                                         std::vector<cl_float> &to2,
                                         cl_uint stride2
                                        ) : 
     gen(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
     sourceSize(total_data), 
     destSize(minibatch_size),
     from1(from1),
     to1(to1),
     stride1(stride1),
     from2(from2),
     to2(to2),
     stride2(stride2)
{
    selected.resize(sourceSize);
    index.resize(sourceSize);
    minibatch.resize(destSize);
}

void minibatch_generator::generate() {
    std::uniform_int_distribution<cl_uint> dist(0, sourceSize - 1);

    for(cl_uint i = 0; i < sourceSize; i++) {
        index[i] = i;
        selected[i] = false;
    }
    for(cl_uint i = 0; i < destSize; i++) {
        cl_uint sel = dist(gen);
        while(selected[sel % sourceSize]) sel++;
        sel = sel % sourceSize;
        selected[sel] = true;
        minibatch[i] = sel;            
    }
}

void minibatch_generator::load_generated_minibatch() {
    generate();
    for(cl_uint i = 0; i < minibatch.size(); i++) {
        for(cl_uint j = 0; j < stride1; j++) {
            to1[i*stride1 + j] = from1[minibatch[i]*stride1 + j];
        }
        for(cl_uint j = 0; j < stride2; j++) {
            to2[i*stride2 + j] = from2[minibatch[i]*stride2 + j];
        }
    }
}

