/* 
 * File:   mnist.hpp
 * Author: jordi
 *
 * Created on 17 de diciembre de 2014, 22:00
 */

#ifndef MNIST_HPP
#define	MNIST_HPP

#include <string>
#include <vector>

void read_mnist_images_file(const std::string filename, 
                            std::vector<float> &v, 
                            size_t &r, 
                            size_t &c);

void read_mnist_labels_file(const std::string filename, 
                            std::vector<float> &v, 
                            size_t &r, 
                            size_t &c);

void print_mnist_image_txt(std::vector<float> &v, size_t offset, uint8_t rows = 28, uint8_t cols = 28);
void print_mnist_label_txt(std::vector<float> &v, size_t offset, uint8_t out = 16);

#endif	/* MNIST_HPP */

