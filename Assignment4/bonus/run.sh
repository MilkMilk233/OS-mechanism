make clean

##Compile the cuda script using the nvcc compiler
nvcc --relocatable-device-code=true main.cu user_program.cu file_system.cu -o test

sleep 1

## Run the executable file
srun ./test > ./output/bonus.txt