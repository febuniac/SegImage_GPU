all: gpu seq 

gpu: SSSP_simple_example.cu imagem.cpp imagem.h
	nvcc -std=c++11 -lnvgraph SSSP_simple_example.cu imagem.cpp -o gpu 

seq: Image_seg_Sequencial.cu imagem.cpp imagem.h
	nvcc -std=c++11 Image_seg_Sequencial.cu imagem.cpp -o seq 