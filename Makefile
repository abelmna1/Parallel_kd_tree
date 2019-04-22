all: k-nn

k-nn: prog1.cpp
	/home/kchiu/packages/linux-x86_64/bin/g++ -std=c++11 -o k-nn prog1.cpp -Wall -pedantic -g -O -pthread 

clean:
	rm -f *.o
	rm -f k-nn
    
