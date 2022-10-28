#include <cuda_runtime.h>
#include <iostream>
int main(){
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << prop.major << "." << prop.minor << std::endl;
    }
    return 0;
}
//g++ tests/find_arch.cc -o tests/find_arch -I/usr/local/cuda-10.2/include -L/usr/local/cuda-10.2/lib64 -lcudart