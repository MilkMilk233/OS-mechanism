#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdlib.h>
#include <vector>
#include <algorithm> 

typedef unsigned char uchar;
typedef uint32_t u32;
typedef uint16_t u16;  
typedef uint8_t u8;  

#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2

struct FileSystem {
	uchar *volume;
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;
	int MODIFY_TIME;
	int CREATE_TIME;
	int VALID_BLOCK;
};


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);
__device__ u32 FCB_read_permission(FileSystem *fs, u32 FCB_address);
__device__ void FCB_set_permission(FileSystem *fs, u32 FCB_address, u32 option, u32 value);
__device__ u32 FCB_read_validbit(FileSystem *fs, u32 FCB_address);
__device__ void FCB_set_validbit(FileSystem *fs, u32 FCB_address, u32 value);
__device__ void FCB_read_filename(FileSystem *fs, u32 FCB_address, uchar *output);
__device__ void FCB_set_filename(FileSystem *fs, u32 FCB_address, uchar *input);
__device__ u32 FCB_read_start(FileSystem *fs, u32 FCB_address);
__device__ void FCB_set_start(FileSystem *fs, u32 FCB_address, u32 value);
__device__ u32 FCB_read_size(FileSystem *fs, u32 FCB_address);
__device__ void FCB_set_size(FileSystem *fs, u32 FCB_address, u32 value);
__device__ u32 FCB_read_ltime(FileSystem *fs, u32 FCB_address);
__device__ void FCB_set_ltime(FileSystem *fs, u32 FCB_address, u32 value);
__device__ u32 FCB_read_ctime(FileSystem *fs, u32 FCB_address);
__device__ void FCB_set_ctime(FileSystem *fs, u32 FCB_address, u32 value);
__device__ void memcpy(uchar *target, uchar *source, int size);
__device__ u32 memcmp(uchar *target, uchar *source, int size);
__device__ void cover(FileSystem *fs, u32 layer, u32 start, u32 end);
__device__ void VCB_modification(FileSystem *fs, u32 start, u32 size, u32 value);
__device__ int VCB_Query(FileSystem *fs, u32 n);
__device__ void memory_compaction(FileSystem *fs);
__device__ void print_FCB(FileSystem *fs);
__device__ void print_VCB(FileSystem *fs);

#endif