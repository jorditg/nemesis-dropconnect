
#include <cstdint>

#include <iostream>
#include <fstream>


#include <functional>
#include <string>
#include <sstream>
#include <vector>

uint64_t toDWord (const uint8_t *buf) {
    const uint64_t result = 
               ((static_cast<uint64_t>(buf[0]) << 24) |
                (static_cast<uint64_t>(buf[1]) << 16) |
                (static_cast<uint64_t>(buf[2]) << 8) |
                (static_cast<uint64_t>(buf[3]))
               );

    return result;
}

std::string buffer2Str (const uint8_t *buf, const size_t len) {
    std::stringstream ss;
    
    if (len == 0) return std::string("");
    
    ss << static_cast<unsigned> (buf[0]);

    for (size_t i = 1; i < len; i++) {
        ss << "," << static_cast<unsigned> (buf[i]);
    }

    return ss.str();
}



void read_mnist_images_file(const std::string filename, 
                            std::vector<float> &v, 
                            size_t &r, 
                            size_t &c) {
   
    std::ifstream ifs(filename, std::ios::binary);
    uint8_t buf_header[16];
    ifs.read(reinterpret_cast<char *> (buf_header), 16);

    //const uint64_t magic_number = toDWord(&buf_header[0]);
    //std::cout << "Magic number: " << magic_number << std::endl;
    const uint64_t numberOfImages = toDWord(&buf_header[4]);
    //std::cout << "Number of images: " << numberOfImages << std::endl;
    const uint64_t rows = toDWord(&buf_header[8]);
    //std::cout << "Rows: " << rows << std::endl;
    const uint64_t cols = toDWord(&buf_header[12]);

    //std::cout << "Cols: " << cols << std::endl;

    const size_t img_sz = rows*cols;

    v.resize(img_sz*numberOfImages);
    
    uint8_t buffer[img_sz];
    for (size_t i = 0; i < numberOfImages; i++) {
      ifs.read(reinterpret_cast<char *> (buffer), img_sz);
      for(size_t j = 0; j < img_sz; j++) {
          v[i*img_sz + j] = static_cast<float>(buffer[j])/255.0f;
      }
    }
    r = numberOfImages;
    c = img_sz;
}

void read_mnist_labels_file(const std::string filename, 
                            std::vector<float> &v, 
                            size_t &r, 
                            size_t &c) {
    
    const uint8_t label_size = 16;
    
    std::ifstream ifs(filename, std::ios::binary);
    uint8_t buf_header[8];
    ifs.read(reinterpret_cast<char *> (buf_header), 8);

    //const uint64_t magic_number = toDWord(&buf_header[0]);
    //std::cout << "Magic number: " << magic_number << std::endl;
    const uint64_t numberOfImages = toDWord(&buf_header[4]);
    //std::cout << "Number of images: " << numberOfImages << std::endl;
    
    v.resize(label_size*numberOfImages);
    uint8_t b;
    for(size_t i = 0; i < numberOfImages; i++) {
        ifs.read(reinterpret_cast<char *>(&b), 1);
        for(uint8_t j = 0; j < label_size; j++) {
            if(b == j) {
                v[i*label_size + j] = 1.0f;
            } else {
                v[i*label_size + j] = 0.0f;
            }
        }
    }
    r = numberOfImages;
    c = label_size;
}

void print_mnist_image_txt(std::vector<float> &v, size_t offset, uint8_t rows, uint8_t cols) {
    for (uint8_t i = 0; i < rows; i++) {
        for(uint8_t j = 0; j < cols; j++) {
            const float val = v[offset + i*cols + j];
            std::cout << ((val > 0.2)?"1":"0");
        }
        std::cout << std::endl;
    }
}

void print_mnist_label_txt(std::vector<float> &v, size_t offset, uint8_t out) {
    for(uint8_t j = 0; j < out; j++) {
        std::cout << v[offset + j];
    }
    std::cout << std::endl;
}

