#include <stdio.h>
#include <stdlib.h>

typedef unsigned char uchar;
typedef u_int32_t u32;
typedef u_int16_t u16;  
typedef u_int8_t u8;  

// To get the [start, start+size] bits in source.
u32 bit_read(uchar *source, u32 start, u32 size){
    if(start == 0) return (*source >> 0)  & 0b00000001;
    if(start == 1) return (*source >> 1)  & 0b00000001;
    if(start == 2) return (*source >> 2)  & 0b00000001;
    if(start == 3) return (*source >> 3)  & 0b00000001;
    if(start == 4) return (*source >> 4)  & 0b00000001;
    if(start == 5) return (*source >> 5)  & 0b00000001;
    if(start == 6) return (*source >> 6)  & 0b00000001;
    if(start == 7) return (*source >> 7)  & 0b00000001;
}

// To write the source to the [start, start+size] bits in target.
void bit_write(uchar *target, u32 start, u32 size, uchar source){
  *target = (*target & ~((1 << start + size) - (1 << start))) + ((1 << size) & source); 
}

int main() {
    uchar test[1];
    test[0] = 0x4f;
    uchar *target = &test[0];
    int aaa = 32;
    int bbb = -1;
    printf("aaa = %d, bbb = %d\n",aaa,(u32)bbb);
    if(aaa < bbb){
      printf("aaa < bbb\n");
    }
    else{
      printf("aaa > bbb\n");
    }
    return 0;
}