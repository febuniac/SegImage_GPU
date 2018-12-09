
 #include <stdio.h>
 #include <stdlib.h>
 #include "nvgraph.h"
 #include <vector>
 #include <ctype.h>
 #include <string.h>
 #include <assert.h>
 #include <algorithm>
 #include <iostream>
 #include <queue>
 #include "imagem.h" 
 #include <thrust/device_vector.h>
 #include <thrust/host_vector.h>
 #include <fstream>
 #include <iterator> 
 #include <cuda_runtime.h>
typedef std::pair<double, int> custo_caminho;

typedef std::pair<double *, int *> result_sssp;

#define MAX(y,x) (y>x?y:x)    // Calcula valor maximo
#define MIN(y,x) (y<x?y:x)    // Calcula valor minimo
 
 __global__ void edgeFilter(unsigned char *image_in, unsigned char *image_out, int rowStart, int rowEnd, int colStart, int colEnd)
 {
    int di,dj;
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    
    for(i = rowStart; i < rowEnd; ++i) 
    {
       for(j = colStart; j < colEnd; ++j) 
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
 }

struct compare_custo_caminho {
    bool operator()(custo_caminho &c1, custo_caminho &c2) {
        return c2.first < c1.first;
    }
};

result_sssp SSSP(imagem *img, std::vector<int> sources) {
    std::priority_queue<custo_caminho, std::vector<custo_caminho>, compare_custo_caminho > Q;
    double *custos = new double[img->total_size];
    int *predecessor = new int[img->total_size];
    bool *analisado = new bool[img->total_size];

    result_sssp res(custos, predecessor);
    
    for (int i = 0; i < img->total_size; i++) {
        predecessor[i] =-1;
        custos[i] = __DBL_MAX__;
        analisado[i] = false;
    }
    
    for (int i = 0; i < sources.size(); i++) {
    Q.push(custo_caminho(0.0, sources[i]));
    predecessor[sources[i]] = sources[i];
    custos[sources[i]] = 0.0;
    }

    while (!Q.empty()) {
        custo_caminho cm = Q.top();
        Q.pop();

        int vertex = cm.second;
        if (analisado[vertex]) continue; // já tem custo mínimo calculado
        analisado[vertex] = true;
        double custo_atual = cm.first;
        assert(custo_atual == custos[vertex]);

        int vertex_i = vertex / img->cols;
        int vertex_j = vertex % img->cols;
        
        if (vertex_i > 0) {
            int acima = vertex - img->cols;
            double custo_acima = custo_atual + get_edge(img, vertex, acima);
            if (custo_acima < custos[acima]) {
                custos[acima] = custo_acima;
                Q.push(custo_caminho(custo_acima, acima));
                predecessor[acima] = vertex;
            }
        }

        if (vertex_i < img->rows - 1) {
            int abaixo = vertex + img->cols;
            double custo_abaixo = custo_atual + get_edge(img, vertex, abaixo);
            if (custo_abaixo < custos[abaixo]) {
                custos[abaixo] = custo_abaixo;
                Q.push(custo_caminho(custo_abaixo, abaixo));
                predecessor[abaixo] = vertex;
            }
        }


        if (vertex_j < img->cols - 1) {
            int direita = vertex + 1;
            double custo_direita = custo_atual + get_edge(img, vertex, direita);
            if (custo_direita < custos[direita]) {
                custos[direita] = custo_direita;
                Q.push(custo_caminho(custo_direita, direita));
                predecessor[direita] = vertex;
            }
        }

        if (vertex_j > 0) {
            int esquerda = vertex - 1;
            double custo_esquerda = custo_atual + get_edge(img, vertex, esquerda);
            if (custo_esquerda < custos[esquerda]) {
                custos[esquerda] = custo_esquerda;
                Q.push(custo_caminho(custo_esquerda, esquerda));
                predecessor[esquerda] = vertex;
            }
        }
    }
    
    delete[] analisado;
    
    return res;
}


int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Uso:  segmentacao_sequencial entrada.pgm saida.pgm\n";
        return -1;
    }
    
    std::string path(argv[1]);
    std::string path_output(argv[2]);
    imagem *img = read_pgm(path);
    imagem *imagem_saida = new_image(img->rows, img->cols);
    
    int n_fg, n_bg;
    int x, y;
    cudaEvent_t start, stop,start_tempo_total,stop_tempo_total;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_tempo_total);
    cudaEventCreate(&stop_tempo_total);
    float elapsed_time_total, elapsed_time_caminhos_min, elapsed_time_montagem_imagem_seg;
    
    cudaEventRecord(start_tempo_total);
    
    dim3 dimGrid(ceil(img->rows/16.0), ceil(img->cols/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    thrust::device_vector<unsigned char> entrada_img(img->pixels, img->pixels + img->total_size );
    thrust::device_vector<unsigned char> saida_img(imagem_saida->pixels, imagem_saida->pixels + imagem_saida->total_size );
    edgeFilter<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(entrada_img.data()), thrust::raw_pointer_cast(saida_img.data()),0, img->rows,0, img->cols);
    std::cin >> n_fg >> n_bg;
    std::vector<int> seeds_bg;
    std::vector<int> seeds_fg;

 
    for(int i = 0; i < n_bg; i++){
        std::cin >> x >> y;
        int seed_bg = y * img->cols + x;
        seeds_bg.push_back(seed_bg);
    }
    for(int i = 0; i < n_fg; i++){
        std::cin >> x >> y;
        int seed_fg = y * img->cols + x;
        seeds_fg.push_back(seed_fg);
    }
    
    // std::cin >> x >> y;
    // int seed_fg = y * img->cols + x;
    
    // std::cin >> x >> y;
    // int seed_bg = y * img->cols + x;

    cudaEventRecord(start);
    result_sssp fg = SSSP(img, seeds_fg);
    result_sssp bg = SSSP(img, seeds_bg);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_caminhos_min, start, stop);
    imagem *saida = new_image(img->rows, img->cols);

    cudaEventRecord(start);
    for (int k = 0; k < saida->total_size; k++) {
        if (fg.first[k] > bg.first[k]) {
            saida->pixels[k] = 0;
        } else {
            saida->pixels[k] = 255;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_montagem_imagem_seg, start, stop);
    
    write_pgm(saida, path_output); 

    cudaEventRecord(stop_tempo_total);
    cudaEventSynchronize(stop_tempo_total);
    cudaEventElapsedTime(&elapsed_time_total, start, stop);


    std::cout <<"Tempo total de execução foi:" << elapsed_time_total << std::endl;
    std::cout <<  "Tempo de cálculo de caminhos mínimos foi:" << elapsed_time_caminhos_min << std::endl;
    std::cout << "Tempo de montagem da imagem segmentada foi:" << elapsed_time_montagem_imagem_seg << std::endl;
 
    return 0;
}

