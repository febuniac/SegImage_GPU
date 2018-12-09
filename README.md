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
Use the command 'make' on this repository
```
###### Step 4 :
```
Other option would be to compile one by one
```
###### Step 5 :
```
nvcc -O2 Image_seg_Sequencial.cu imagem.cpp -o NAME  --std=c++11 -lnvgraph
```
###### Step 6 :
```
nvcc -O2 Image_seg_GPU.cu imagem.cpp -o NAME  --std=c++11 -lnvgraph
``` 

###### Step 7 :
```
Using : 
./NAME iumput_image.pgm Output_)image.pgm < seeds_input_file.txt
``` 



