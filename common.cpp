#include <string>
#include <vector>
#include "common.hpp"

typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;

void load_nn_data(const std::string & filename,
                   cl_uint &layers,
                   std::vector<cl_uint> &elements) {
    
    std::ifstream in(filename.c_str());
    if (!in.is_open()) {
        std::cout << "File already opened. Exiting\n";
        exit(1);
    }

    std::string line;
      
    std::vector< std::string > vec;

    getline(in, line);              // read elements per layer

    Tokenizer tok(line);
    vec.assign(tok.begin(), tok.end());

    layers = 0;
    // Num.ofelem. includes the BIAS neuron for every layer except the last one
    for (std::vector<std::string>::iterator it = vec.begin() ;
         it != vec.end(); ++it) {
        cl_uint elem = std::stoi(*it);
        // check that every layer has a -multiple of 4- number of elements
        assert(elem % 4 == 0);
        elements.push_back(elem);
        layers++;
    }    
}

void load_csv_data(const std::string & filename,
                   std::vector<cl_float> & input,
                   std::vector<cl_float> & output,
                   cl_uint &rows,
                   cl_uint in_elements,
                   cl_uint out_elements) {
    
    std::ifstream in(filename.c_str());
    if (!in.is_open()) {
        std::cout << "File already opened. Exiting\n";
        exit(1);
    }

    std::string line;

    getline(in, line);               // read number of data lines
    rows = std::stoi(line);
        
    std::vector< std::string > vec;
    // cols to read = number of inputs (counting bias that we add) + number of outputs
    const cl_uint cols = in_elements + out_elements;
    
    cl_uint n = 0;
    while (getline(in, line)) {
        Tokenizer tok(line);
        vec.assign(tok.begin(), tok.end());
        // vector now contains strings from one row, output to cout here
        // std::copy(vec.begin(), vec.end(),
        //           std::ostream_iterator<std::string>(std::cout, "|"));
        // std::cout << "\n----------------------" << std::endl;

        // check that there is not incomplete data
        assert(vec.size() == size_t(cols));

        cl_uint i = 0;
        for (std::vector<std::string>::iterator it = vec.begin();
             it != vec.end(); ++it) {
            if (i < in_elements) input.push_back(std::stof(*it));
            else
              output.push_back(std::stof(*it));
            i++;
        }
        n++;
        if (n == rows) break;
    }
    
    assert((input.size() / size_t(in_elements)) == size_t(rows));
}

void load_csv_vector(const std::string & filename,
                     std::vector<cl_float> &weights) {
    
    std::ifstream in(filename.c_str());
    if (!in.is_open()) {
        std::cout << "File already opened. Exiting\n";
        exit(1);
    }

    std::string line;

    std::vector< std::string > vec;
    std::vector<cl_float>::iterator wit = weights.begin();
    
    while (getline(in, line)) {
        Tokenizer tok(line);
        vec.assign(tok.begin(), tok.end());
        // vector now contains strings from one row, output to cout here
        // std::copy(vec.begin(), vec.end(),
        //           std::ostream_iterator<std::string>(std::cout, "|"));
        // std::cout << "\n----------------------" << std::endl;

        // check that there is not incomplete data
        for (std::vector<std::string>::iterator it = vec.begin();
             it != vec.end(); ++it) {
          *wit = std::stof(*it);
          wit++;
        }
    }
    
    assert(wit == weights.end());
}


void save_csv_vector(const std::string & filename,
                     std::vector<cl_float> &weights) {
    
    std::ofstream out(filename.c_str());
    if (!out.is_open()) {
        std::cout << "File already opened. Exiting\n";
        exit(1);
    }

    if(weights.size() > 0) { 
        out << weights[0];
        for (size_t i = 1; i < weights.size(); i++) {
            out << "," << weights[i];
        }
    }
    out << std::endl;
}


void print_vector(const std::vector<cl_float> &v,
                  cl_uint rows,
                  cl_uint cols,
                  cl_uint offset = 0) {
  cl_uint lines = 0;
  cl_uint end = rows*cols + offset;
  for (size_t i = offset; i < end; i++) {
    std::cout << boost::format("%5.6f") % v[i] << " ";
    if (!((i+1 - offset) % cols)) {
        std::cout << std::endl;
        lines++;
    }
    if (lines == rows ) break;
  }
}

void print(const matrix_cl_float &m,
           const std::string header,
           const bool rows2cols) {
  if (!header.empty())
    std::cout << header << std::endl;

  if (!rows2cols)
    print_vector(m.data.hostData, m.rows, m.cols, m.offset);
  else
    print_vector(m.data.hostData, m.cols, m.rows, m.offset);
}

