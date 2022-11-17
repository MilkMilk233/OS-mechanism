#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants (Unit: byte / B)
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;  // 4096
  fs->FCB_SIZE = FCB_SIZE;  //  32
  fs->FCB_ENTRIES = FCB_ENTRIES;    // 1024
  fs->STORAGE_SIZE = VOLUME_SIZE;   // 1085440
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;    //32
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;    // 20
  fs->MAX_FILE_NUM = MAX_FILE_NUM;    // 1024
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;    // 1048576
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;    // 36864

}


// To get the [start, start+size] bits in source.
__device__ u32 bit_read(u32 *source, u32 start, u32 size){
  return ((*source >> start) && (1 << size));
}

// To write the source to the [start, start+size] bits in target.
__device__ void bit_write(u32 *target, u32 start, u32 size, u32 source){
  *target = (*target && ~((1 << start + size) - (1 << start))) + ((1 << size) && source); 
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  // Search the FCB, see if there are 
  char file_name[20];
  int FCB_SIZE = fs->FCB_SIZE;
  bool found = false;
  u32 *carrier;
  int secondary_address, pcb_address;
  for(pcb_address = 0; pcb_address < fs->FCB_ENTRIES; pcb_address++){
    carrier = &fs->volume[SUPERBLOCK_SIZE + pcb_address*FCB_SIZE];
    memcpy(file_name, carrier, FCB_SIZE);
    if(memcmp(file_name,s) == 0){
      // Found
      if(bit_read(carrier, 0, 1) == 1){
        found = true;
        break;
      }
    }
  }
  if(found){
    // Fetch and Return the Starting point
  }
  else{
    // Initiate a new FCB block with size = 0
    // Set permission
    // Set Create time
    // Set Update time
    // Set Starting point
    // Set Size
    // Set Name

    // Fetch and Return the secondary_address

  }
  return secondary_address;
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
}
