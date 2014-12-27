#!/usr/bin/python

import getopt, sys, random

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_line(fname, number):
    with open(fname) as f:
        for i, l in enumerate(f):
            if i == number: 
                return l
    return ''

def split_data(data_file):
    train = open("train.txt", "w")
    test = open("test.txt", "w")
    cv = open("cv.txt", "w")

    lines = file_len(data_file)
    vect = range(lines)
    random.shuffle(vect)
    
    for i,idx in enumerate(vect):
        line = get_line(data_file, idx)
        percentage = float(i)/float(lines)
        if percentage < 0.50:
            train.write(line)
        elif percentage < 0.75:
            test.write(line)
        else:
            cv.write(line)

def usage():
    print "Generates three files: train.txt, test.txt and cv.txt"
    print "from a data input file indicated in the parameter -i or --input"

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:v", ["help", "input="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    if len(opts) == 0:
        usage()
        sys.exit(2)

    inp = None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-i", "--input"):
            inp = a
        else:
            assert False, "unhandled option"
    
    split_data(inp)

if __name__ == "__main__":
    main()    
