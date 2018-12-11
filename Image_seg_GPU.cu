/*
 * SSSP Function is based on:
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <cuda_runtime.h>
 #include "nvgraph.h"
 #include <vector>
 #include <ctype.h>
 #include <string.h>
 #include <assert.h>
 #include <algorithm>
 #include <iostream>
 #include "imagem.h" 
 #include <thrust/device_vector.h>
 #include <thrust/host_vector.h>
 #include <fstream>
 #include <iterator> 
 #include <cuda_runtime.h>
 
 
 #define MAX(y,x) (y>x?y:x)    // Calcula valor maximo
 #define MIN(y,x) (y<x?y:x)    // Calcula valor minimo
 
 __global__ void edgeFilter(unsigned char *image_in, unsigned char *image_out, int rowStart, int rowEnd, int colStart, int colEnd)
 {
    int di,dj;
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i< rowEnd && j< colEnd) 
    {
        int min = 256;
        int max = 0;
            for(di = MAX(rowStart, i - 1); di <= MIN(i + 1, rowEnd - 1); di++) 
            {
                for(dj = MAX(colStart, j - 1); dj <= MIN(j + 1, colEnd - 1); dj++) 
                {
                if(min>image_in[di*(colEnd-colStart)+dj]) min = image_in[di*(colEnd-colStart)+dj];
                if(max<image_in[di*(colEnd-colStart)+dj]) max = image_in[di*(colEnd-colStart)+dj]; 
                }
            }
            image_out[i*(colEnd-colStart)+j] = max-min;
    }
 }
 
 

 
 //Informações necessárias para gerar as listas e usar no SSSP
 struct info_lists{
     size_t n;//numero de nós
     size_t nnz;//número de arestas
     float * weights_h;
     int * destination_offsets_h;
     int * source_indices_h;
 };
 
 
 // Gerando listas para jogar na função doi SSSP
 info_lists gerandoListas(imagem *img, std::vector<int> seeds){
     // bool *analisado = new bool[img->total_size];
     std::vector<float> weights;
     std::vector<int> source_indices_h;
     std::vector<int> destination_offsets_h;
     int offset_count=0;
     destination_offsets_h.push_back(offset_count);
     info_lists informacoes = {};
     for (int i = 0; i < img->total_size; i++) 
     {
         
         offset_count = 0;
         int vertex = i;
         // if (analisado[vertex]) continue; // já tem custo mínimo calculado
         // analisado[vertex] = true;
         int vertex_i = vertex / img->cols;
         int vertex_j = vertex % img->cols;
 
         if (std::find(seeds.begin(), seeds.end(), vertex) != seeds.end()){//Checando se o nó é uma seed para liga-lo no nó fantasma.
            weights.push_back(0.0);
            source_indices_h.push_back(img->total_size);//image size esta fora imagem seria on primeiro ponto fora da imagem
            offset_count+=1; 
         }
         if (vertex_i > 0) {
             int acima = vertex - img->cols;
             double custo_acima = get_edge(img, vertex, acima);
             weights.push_back(custo_acima);
             offset_count+=1;
             source_indices_h.push_back(acima);
         }
 
         if (vertex_i < img->rows - 1) {
             int abaixo = vertex + img->cols;
             double custo_abaixo = get_edge(img, vertex, abaixo);
             weights.push_back(custo_abaixo);
             offset_count+=1;
             source_indices_h.push_back(abaixo);
         }
 
         if (vertex_j < img->cols - 1) {
             int direita = vertex + 1;
             double custo_direita = get_edge(img, vertex, direita);
             weights.push_back(custo_direita);
             offset_count+=1;
             source_indices_h.push_back(direita);
         }
 
         if (vertex_j > 0) {
             int esquerda = vertex - 1;
             double custo_esquerda = get_edge(img, vertex, esquerda);
             weights.push_back(custo_esquerda);
             offset_count+=1;
             source_indices_h.push_back(esquerda);
         }
         destination_offsets_h.push_back(destination_offsets_h.back() + offset_count);
     } 
 
     informacoes.n = destination_offsets_h.size() - 1;//numero de nós é o tammnho da lista de offsets -1 
     informacoes.nnz = ((img->total_size * 4) - ((img->cols + img->rows) * 2)) + seeds.size();//numnero de arestas 
     informacoes.weights_h = (float*)malloc(informacoes.nnz*sizeof(float));
     informacoes.destination_offsets_h = (int*) malloc((informacoes.n+1)*sizeof(int));
     informacoes.source_indices_h = (int*) malloc(informacoes.nnz*sizeof(int));
 
     //Conversão para tipos corretos da SSSP_func
     for(int c = 0; c< weights.size(); c++)
     {
         informacoes.weights_h[c]=weights[c];
         //std::cout<<"weights :" <<informacoes.weights_h[c]<<std::endl;
     }
 
     for(int c1 = 0; c1 < destination_offsets_h.size(); c1++)
     {
         informacoes.destination_offsets_h[c1]=destination_offsets_h[c1];
         //std::cout<<"destination_offsets_h :"<<informacoes.destination_offsets_h[c1]<<std::endl;
     }
 
     for(int c2 = 0; c2 < source_indices_h.size(); c2++)
     {
        informacoes.source_indices_h[c2]=source_indices_h[c2];
        // std::cout<<"source_indices_h :" <<informacoes.source_indices_h[c2]<<std::endl;
 
     }
 
     return informacoes;
     
 }
 
 void check_status(nvgraphStatus_t status)
 {  
     if ((int)status != 0)
     {
         printf("ERROR : %d\n",status);
         exit(0);
     }
 }
 
 /* Single Source Shortest Path (SSSP)
  *  Calculate the shortest path distance from a single vertex in the graph
  *  to all other vertices.
  */ 
 float * SSSP_func(float *weights_h, int *destination_offsets_h, int *source_indices_h, const size_t n, const size_t nnz) {
     const size_t vertex_numsets = 1, edge_numsets = 1;
     float *sssp_1_h;
     void** vertex_dim;
     
     // nvgraph variables
     //nvgraphStatus_t status;
     nvgraphHandle_t handle;
     nvgraphGraphDescr_t graph;
     nvgraphCSCTopology32I_t CSC_input;
     cudaDataType_t edge_dimT = CUDA_R_32F;
     cudaDataType_t* vertex_dimT;
 
     // Init host data
     // destination_offsets_h = (int*) malloc((n+1)*sizeof(int));
     // source_indices_h = (int*) malloc(nnz*sizeof(int));
     // weights_h = (float*)malloc(nnz*sizeof(float));
     sssp_1_h = (float*)malloc(n*sizeof(float));
     // sssp_2_h = (float*)malloc(n*sizeof(float));
     vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
     vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
     CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
     //CSC = compressed sparse column
     vertex_dim[0]= (void*)sssp_1_h; 
     // vertex_dim[1]= (void*)sssp_2_h;
     vertex_dimT[0] = CUDA_R_32F; 
     // vertex_dimT[1]= CUDA_R_32F;
 
     check_status(nvgraphCreate(&handle));
     check_status(nvgraphCreateGraphDescr (handle, &graph));
 
     CSC_input->nvertices = n;
     CSC_input->nedges = nnz;
     CSC_input->destination_offsets = destination_offsets_h;
     CSC_input->source_indices = source_indices_h;
 
     // Set graph connectivity and properties (tranfers)
     check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
     check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
     check_status(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
     check_status(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
 
     // Solve
     int source_vert = 0;
     check_status(nvgraphSssp(handle, graph, 0,  &source_vert, 0));
     
 
     // Solve with another source
     // source_vert = 5;
     // check_status(nvgraphSssp(handle, graph, 0,  &source_vert, 1));
     
     // Get and print result
 
 
     check_status(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));
     // expect sssp_1_h = (0.000000 0.500000 0.500000 1.333333 0.833333 1.333333)^T
     // printf("sssp_1_h\n");
     // for (int i = 0; i<n; i++)  printf("%f\n",sssp_1_h[i]); printf("\n");
     // printf("\nDone!\n");
 
 
     // check_status(nvgraphGetVertexData(handle, graph, (void*)sssp_2_h, 1));
     // // expect sssp_2_h = (FLT_MAX FLT_MAX FLT_MAX 1.000000 1.500000 0.000000 )^T
     // printf("sssp_2_h\n");
     // for (i = 0; i<n; i++)  printf("%f\n",sssp_2_h[i]); printf("\n");
     // printf("\nDone!\n");
 
     free(destination_offsets_h);
     free(source_indices_h);
     free(weights_h);
     // free(sssp_1_h);
     // free(sssp_2_h);
     free(vertex_dim);
     free(vertex_dimT);
     free(CSC_input);
 
     //Clean 
     check_status(nvgraphDestroyGraphDescr (handle, graph));
     check_status(nvgraphDestroy (handle));
 
     return sssp_1_h;
 }
 
 int main(int argc, char **argv)
 {
 
 if (argc < 3) {
         std::cout << "Uso:  segmentacao_sequencial entrada.pgm saida.pgm < entrada.txt (sementes)\n";
         return -1;
     }
 
     //Tempos:
     cudaEvent_t start, stop,start_tempo_total,stop_tempo_total;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     cudaEventCreate(&start_tempo_total);
     cudaEventCreate(&stop_tempo_total);
     float elapsed_time_total, elapsed_time_montagem_grafo, elapsed_time_caminhos_min, elapsed_time_montagem_imagem_seg;
     
     
     
     
     std::string path(argv[1]);
     std::string path_output(argv[2]);   
     imagem *img = read_pgm(path);
     imagem *imagem_saida = new_image(img->rows, img->cols);
    
     dim3 dimGrid(ceil(img->rows/16.0), ceil(img->cols/16.0), 1);
     dim3 dimBlock(16, 16, 1);
     int n_fg, n_bg;
     std::vector<int> seeds_bg, seeds_fg;
     int x, y;
     
     cudaEventRecord(start_tempo_total);
 
     std::cin >> n_fg >> n_bg;
     std::cout << "Number of foreground seeds positioned:  "<< n_fg<<std::endl;
     std::cout << "Number of background seeds positioned:  "<< n_bg<<std::endl;
    
     // assert(n_fg >=1);
     // assert(n_bg >= 1);
     //lendo sementes
     for (int k = 0; k < n_fg; k++) {
         std::cin >> x >> y;
         int seed_fg = y * img->cols + x;
         seeds_fg.push_back(seed_fg);
     }
     
     for (int k = 0; k < n_bg; k++) {  
         std::cin >> x >> y;
         int seed_bg = y * img->cols + x;
         seeds_bg.push_back(seed_bg);
         
     }
 
 
     thrust::device_vector<unsigned char> entrada_img(img->pixels, img->pixels + img->total_size );
     thrust::device_vector<unsigned char> saida_img(imagem_saida->pixels, imagem_saida->pixels + imagem_saida->total_size );
     edgeFilter<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(entrada_img.data()), thrust::raw_pointer_cast(saida_img.data()),0, img->rows,0, img->cols);
 
     cudaEventRecord(start);
     info_lists info_fg = gerandoListas(img, seeds_fg);
     info_lists info_bg = gerandoListas(img, seeds_bg);
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&elapsed_time_montagem_grafo, start, stop);
 
     cudaEventRecord(start);
     auto fg = SSSP_func(info_fg.weights_h, info_fg.destination_offsets_h,info_fg.source_indices_h,info_fg.n,info_fg.nnz);
     auto bg = SSSP_func(info_bg.weights_h, info_bg.destination_offsets_h,info_bg.source_indices_h,info_bg.n,info_bg.nnz);
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&elapsed_time_caminhos_min, start, stop);
     
     imagem *saida = new_image(img->rows, img->cols);
     cudaEventRecord(start);
     for (int k = 0; k < saida->total_size; k++) {
         if (fg[k] > bg[k]) {
             saida->pixels[k] = 255;
         } else {
             saida->pixels[k] = 0;
         }
     }
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&elapsed_time_montagem_imagem_seg, start, stop);
 
 
     write_pgm(saida, path_output);  
     
     
     cudaEventRecord(stop_tempo_total);
     cudaEventSynchronize(stop_tempo_total);
     cudaEventElapsedTime(&elapsed_time_total, start_tempo_total, stop_tempo_total);
 
     cudaEventDestroy(start);
     cudaEventDestroy(stop);
     cudaEventDestroy(start_tempo_total);
     cudaEventDestroy(stop_tempo_total);
     
 
     
     std::cout <<"Tempo total de execução foi:" << elapsed_time_total << std::endl;
     std::cout << "Tempo de montagem de gráfo foi:" << elapsed_time_montagem_grafo << std::endl;
     std::cout <<  "Tempo de cálculo de caminhos mínimos foi:" << elapsed_time_caminhos_min << std::endl;
     std::cout << "Tempo de montagem da imagem segmentada foi:" << elapsed_time_montagem_imagem_seg << std::endl;
 
     return 0;
 }