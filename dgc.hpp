/* 
 * File:   newfile.hpp
 * Author: jordi
 *
 * Created on 24 de diciembre de 2014, 13:49
 */

#ifndef DCG_HPP
#define	DCG_HPP

#include <random>
#include <cstdint>
#include <CL/cl.hpp>

/*
 *  DropConnect generator class
 */

class dcg {
public:
    inline dcg(std::vector<cl_uchar> & v) : vect(v) {
            std::random_device device;
            std::uint64_t seed = (static_cast<std::uint64_t>(device()) << 32) | device();
            engine.seed(seed);
    }
    
    inline void generate_random_bits() {
        const size_t N = vect.size();
        for(size_t i = 0; i < N; i+=8) {
            const std::uint64_t rnd = engine();
            std::memcpy(&vect[i], &rnd, sizeof(rnd));            
        }
        const size_t resto = N % 8;
        if (resto) {
            const std::uint64_t rnd = engine();
            std::memcpy(&vect[N-resto], &rnd, resto);
        }
    }
private:
    std::mt19937_64 engine;
    std::vector<cl_uchar> & vect;
};

#endif	/* NEWFILE_HPP */

