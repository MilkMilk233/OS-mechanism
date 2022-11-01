#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
    // vm_write(vm, 0, input[0]);
    // vm_write(vm, 1, input[1]);
    // vm_write(vm, 0, input[0]);
    // int value = vm_read(vm, 0);
    // printf("Input = %d\n",input[0]);
    // printf("Output = %d\n",value);
    // vm_snapshot(vm, results, 0, 1);
  for (int i = 0; i < input_size; i++){
    if(i >36800) printf("Arrived %d\n",i);
    vm_write(vm, i, input[i]);
  }
  // for (int i = 0; i < input_size-30000; i++){
  //   // if(i >36800) printf("Arrived %d\n",i);
  //   results[i] = vm->storage[i];
  // }

  // for (int i = input_size - 1; i >= input_size - 32769; i--)
  //   int value = vm_read(vm, i);

  // vm_snapshot(vm, results, 0, input_size);
}
