/* 
 * File:   CLI.cpp
 * Author: jdelatorre
 * 
 * Created on 23 de diciembre de 2014, 14:53
 */

#include <string>
#include <sstream>
#include <iostream>

#include "cli.hpp"


//int main() {
//    cli CLI;
//    
//    CLI.loop();
//}

cli::~cli() {
}

void cli::set(std::istringstream & is, const std::string & cmd) {
    std::string token;
    
    is >> token;
    
    if (token == "lr") {    // set the learning rate
        TODO_msg(cmd);
    } else if (token == "momentum") {   // set the momentum
        TODO_msg(cmd);
    } else if (token == "nag") {    // set NAG
        TODO_msg(cmd);
    } else if (token == "rule") {   // set a new rule
        TODO_msg(cmd);
    } else if (token == "nn") { // set a NN architecture
        TODO_msg(cmd);
    } else {
        unknown_command_msg(cmd);
    }
}

void cli::load (std::istringstream & is, const std::string & cmd) {
    std::string what, filepath;
    
    is >> what;
    is >> filepath; // should be "/file/path"
    if (filepath[0] == '\"' && filepath[filepath.size()-1] == '\"') {
        filepath = filepath.substr(1, filepath.size()-2);
    }
    
    if(!filepath.empty()) {
        if (what == "trainingset") {
            TODO_msg(cmd);
            return;
        } else if (what == "testset") {
            TODO_msg(cmd);
            return;
        } else if (what == "nn") {
            TODO_msg(cmd);
            return;
        }
    }
    // if no return before => command error
    unknown_command_msg(cmd);
}

void cli::save (std::istringstream & is, const std::string & cmd) {
    std::string what, filepath;
    
    is >> what;
    is >> filepath; // should be "/file/path"
    if (filepath[0] == '\"' && filepath[filepath.size()-1] == '\"') {
        filepath = filepath.substr(1, filepath.size()-2);
    }
    
    if(!filepath.empty()) {
        if (what == "nn") {
            TODO_msg(cmd);
            return;
        } 
    }
    // if no return before => command error
    unknown_command_msg(cmd);    
}

void cli::train(std::istringstream & is, const std::string & cmd) {
    std::string what;
    
    is >> what;
    
    if (what == "run") {
        TODO_msg(cmd);
    } else if (what == "pause") {
        TODO_msg(cmd);
    } else if (what == "stop") {
        TODO_msg(cmd);        
    } else {
        unknown_command_msg(cmd);
    }
}

void cli::loop() {
    
    std::string token, cmd;    
    
    do {
        if (!getline(std::cin, cmd)) // Block here waiting for input
            cmd = "quit";
        
        std::istringstream is(cmd);

        token.clear();  // getline() could return empty or blank line
        is >> std::skipws >> token;

        if (token == "quit") {
            break;
        } else if (token == "set") {
            set(is, cmd);
        } else if (token == "load") {
            load(is, cmd);
        } else if (token == "save") {
            save(is, cmd);
        } else if (token == "train") {
            train(is, cmd);
        } else {
            unknown_command_msg(cmd);
        }
    } while (token != "quit");

}
