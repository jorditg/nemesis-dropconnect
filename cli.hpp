/* 
 * File:   CLI.hpp
 * Author: jdelatorre
 *
 * Created on 23 de diciembre de 2014, 14:53
 */

#ifndef CLI_HPP
#define	CLI_HPP

#include <iostream>

class cli {
public:
    //inline cli(nn & nn_v) : neural_network(nn_v) {};
    inline cli() {};
    virtual ~cli();
    
    void loop();
private:
    //nn & neural_network;
    
    void set(std::istringstream & is, const std::string & cmd);
    void load(std::istringstream & is, const std::string & cmd);
    void save(std::istringstream & is, const std::string & cmd);
    void train(std::istringstream & is, const std::string & cmd);
    
    inline void unknown_command_msg(const std::string & cmd) { 
        std::cout << "Unknown command: " << cmd << std::endl; 
    };

    inline void TODO_msg(const std::string & cmd) { 
        std::cout << "TODO option: " << cmd << std::endl; 
    };
       
};

#endif	/* CLI_HPP */

