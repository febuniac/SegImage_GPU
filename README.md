# SegImage_GPU


###### Image Segmentation with GPU



#### USAGE
###### Step 1 :
```
clone this repository.
```
###### Step 2 :
```
Use this program with a computer that has a GPU from NVIDIA.
```
###### Step 3 :
```
Use the command 'make' on this repository's folder (this will compile it with the correct flags)
```
###### Step 4 :
```
Other option would be to compile one by one 
```
###### Step 5 :
```
nvcc -O2 Image_seg_Sequencial.cu imagem.cpp -o NAME  --std=c++11 -lnvgraph (for sequencial)
```
###### Step 6 :
```
nvcc -O2 Image_seg_GPU.cu imagem.cpp -o NAME  --std=c++11 -lnvgraph (for GPU)
``` 

###### Step 7 :
```
After comopiling either ways. You are ready to use it!
Using : 
./NAME input_image.pgm output_image.pgm < seeds_input_file.txt
``` 
The `seeds_input_file.txt` should be used like below:
```
number_of_foreground_seeds number_of_background_seeds
x y
x y
...
```

###### Test Material :
This folder contains .pgm images for testing (this program only works with images .pgm format [removing comments]), some exampole input files and some results examples.

