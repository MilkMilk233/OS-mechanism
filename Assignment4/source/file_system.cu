#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

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

  // Extra intermediate variable space (8 bytes)
  fs->MODIFY_TIME = 0;
  fs->CREATE_TIME = 0;

}

__device__ void memcpy(uchar *target, uchar *source, int size){
  for(int i = 0; i < size; i++){
    target[i] = source[i];
  }
}

__device__ u32 memcmp(uchar *target, uchar *source, int size){
  for(int i = 0; i < size; i++){
    if(target[i] != source[i]) return 1;
  }
  return 0;
}


// Read the FCB permission bit.
__device__ u32 FCB_read_permission(FileSystem *fs, u32 FCB_address){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 24];
  return (*target >> 6) & 0b00000011;
}

// Set the permission bit, 
__device__ u32 FCB_set_permission(FileSystem *fs, u32 FCB_address, u32 option, u32 value){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 24];
  if(option) *target = *target & 0b01111111 + (value << 7); // Option = 1,  set read permission
  else *target = *target & 0b10111111 + (value << 6); // option = 0,  set write permission
}

// Read the FCB valid bit. Valid = 1, Invalid = 0.
__device__ u32 FCB_read_validbit(FileSystem *fs, u32 FCB_address){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 24];
  return (*target >> 5) & 0b00000001;
}

// Set the FCB permission bit. Valid = 1, Invalid = 0.
__device__ u32 FCB_set_validbit(FileSystem *fs, u32 FCB_address, u32 value){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 24];
  *target = *target & 0b11011111 + (value << 5);
}

// Read the FCB filename
__device__ void FCB_read_filename(FileSystem *fs, u32 FCB_address, uchar *output){
  uchar *source = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE];
  memcpy(output, source, 20);
}

// Set the FCB filename
__device__ void FCB_set_filename(FileSystem *fs, u32 FCB_address, uchar *input){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE];
  memcpy(target, input, 20);
}

// Read the FCB starting point address (Unit: block)
__device__ u32 FCB_read_start(FileSystem *fs, u32 FCB_address){
  uchar *source = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+20];
  u32 result;
  memcpy((uchar*)&result, source, 2);
  return result;
}

// Set the FCB starting point address (Unit: block)
__device__ void FCB_set_start(FileSystem *fs, u32 FCB_address, u32 value){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+20];
  memcpy(target, (uchar*)&value, 2);
}

// Read the FCB starting point address (Unit: block)
__device__ u32 FCB_read_size(FileSystem *fs, u32 FCB_address){
  uchar *source = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+22];
  u32 result;
  memcpy((uchar*)&result, source, 2);
  return result;
}

// Set the FCB starting point address (Unit: block)
__device__ void FCB_set_size(FileSystem *fs, u32 FCB_address, u32 value){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+22];
  memcpy(target, (uchar*)&value, 2);
}

// Read the FCB Last modified time
__device__ u32 FCB_read_ltime(FileSystem *fs, u32 FCB_address){
  uchar *source = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+26];
  u32 result;
  memcpy((uchar*)&result, source, 3);
  return result;
}

// Set the FCB Last modified time
__device__ void FCB_set_ltime(FileSystem *fs, u32 FCB_address, u32 value){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+26];
  memcpy(target, (uchar*)&value, 3);
}

// Read the FCB Last modified time
__device__ u32 FCB_read_ctime(FileSystem *fs, u32 FCB_address){
  uchar *source = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+29];
  u32 result;
  memcpy((uchar*)&result, source, 3);
  return result;
}

// Set the FCB Last modified time
__device__ void FCB_set_ctime(FileSystem *fs, u32 FCB_address, u32 value){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+29];
  memcpy(target, (uchar*)&value, 3);
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  // Search the FCB, see if there are 
  uchar file_name[20];
  int FCB_SIZE = fs->FCB_SIZE;
  bool found = false;
  int secondary_address, FCB_address;
  for(FCB_address = 0; FCB_address < fs->FCB_ENTRIES; FCB_address++){
    FCB_read_filename(fs, FCB_address, file_name);
    if(memcmp(file_name,(uchar*)s,20) == 0){
      if(FCB_read_validbit(fs, FCB_address)){
        found = true;
        break;
      }
    }
  }
  if(!found){
    // Initiate a new FCB block with size = 0
    for(FCB_address = 0; FCB_address < fs->FCB_ENTRIES; FCB_address++){
      if(FCB_read_validbit(fs, FCB_address) == 0){
        found = true;
        break;
      }
    }
    FCB_set_filename(fs, FCB_address, (uchar*)s);
    FCB_set_permission(fs, FCB_address, 0, 1);
    FCB_set_permission(fs, FCB_address, 1, 1);
    FCB_set_validbit(fs, FCB_address, 1);
    FCB_set_start(fs, FCB_address, 0);
    FCB_set_size(fs, FCB_address, 0);
    FCB_set_ctime(fs, FCB_address, 0);
    FCB_set_ctime(fs, FCB_address, 0);
  }
  return (FCB_address + (op << 31));
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
