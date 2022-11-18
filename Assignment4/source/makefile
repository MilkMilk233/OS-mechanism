test:
	nvcc --relocatable-device-code=true main.cu user_program.cu file_system.cu -o test
code:
	g++ ./test.cpp -o code -O2 -std=c++11
clean:
	rm test snapshot.bin