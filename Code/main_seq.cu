#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

void blur(unsigned char* rgb_in, unsigned char* rgb_out, int rows, int cols) {
    for (int row = 1; row < rows - 1; ++row) {
        for (int col = 1; col < cols - 1; ++col) {
            for (int rgb = 0; rgb < 3; ++rgb)
            {
                unsigned char hg = rgb_in[3 * ((row - 1) * cols + col - 1) + rgb];
                unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
                unsigned char hd = rgb_in[3 * ((row - 1) * cols + col + 1) + rgb];
                unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
                unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
                unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
                unsigned char bg = rgb_in[3 * ((row + 1) * cols + col - 1) + rgb];
                unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
                unsigned char bd = rgb_in[3 * ((row + 1) * cols + col + 1) + rgb];
                rgb_out[3 * (row * cols + col) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
            }
        }
    }
}

void sharpen(unsigned char* rgb_in, unsigned char* rgb_out, int rows, int cols) {
    for (int row = 1; row < rows - 1; ++row) {
        for (int col = 1; col < cols - 1; ++col) {
            for (int rgb = 0; rgb < 3; ++rgb)
            {
                unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
                unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
                unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
                unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
                unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
                int somme = (-3 * (h + g + d + b) + 21 * c) / 9;
            	
                // unsigned char tmp = static_cast<unsigned char>(somme);
                // if (somme > 255) {
                //     tmp = static_cast<unsigned char>(255);
                // }
                // else if (somme < 0) {
                //     tmp = static_cast<unsigned char>(0);
                // }
                // rgb_out[3 * (row * cols + col) + rgb] = tmp;
                
                if (somme > 255) somme = 255;
                if (somme < 0) somme = 0;
                rgb_out[3 * (row * cols + col) + rgb] = somme;
            }
        }
    }
}

void edge_detect(unsigned char* rgb_in, unsigned char* rgb_out, int rows, int cols) {
    for (int row = 1; row < rows - 1; ++row) {
        for (int col = 1; col < cols - 1; ++col) {
            for (int rgb = 0; rgb < 3; ++rgb)
            {
                unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
                unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
                unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
                unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
                unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
                int somme = (9 * (h + g + d + b) - 36 * c) / 9;
            	
                // unsigned char tmp = static_cast<unsigned char>(somme);
                // if (somme > 255) {
                //     tmp = static_cast<unsigned char>(255);
                // }
                // else if (somme < 0) {
                //     tmp = static_cast<unsigned char>(0);
                // }
                // rgb_out[3 * (row * cols + col) + rgb] = tmp;

                if (somme > 255) somme = 255;
                if (somme < 0) somme = 0;
                rgb_out[3 * (row * cols + col) + rgb] = somme;
            }
        }
    }
}


void main_blur(unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols)
{
    //Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    blur(rgb_in, rgb_out_blur, rows, cols);

    //Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cout << cudaGetErrorString(cudaGetLastError()) << endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "blur: " << elapsedTime << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void main_sharpen(unsigned char* rgb_in, unsigned char* rgb_out_sharpen, int rows, int cols)
{
    //Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sharpen(rgb_in, rgb_out_sharpen, rows, cols);

    //Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cout << cudaGetErrorString(cudaGetLastError()) << endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "sharpen: " << elapsedTime << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void main_edge_detect(unsigned char* rgb_in, unsigned char* rgb_out_edge_detect, int rows, int cols)
{
    //Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    edge_detect(rgb_in, rgb_out_edge_detect, rows, cols);

    //Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cout << cudaGetErrorString(cudaGetLastError()) << endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "edge_detect: " << elapsedTime << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main()
{
    //Declarations
    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    auto rgb = m_in.data;
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    size_t taille_rgb = 3 * rows * cols;
    std::vector< unsigned char > g_blur(taille_rgb);
    std::vector< unsigned char > g_sharpen(taille_rgb);
    std::vector< unsigned char > g_edge_detect(taille_rgb);
    cv::Mat m_out_blur(rows, cols, CV_8UC3, g_blur.data());
    cv::Mat m_out_sharpen(rows, cols, CV_8UC3, g_sharpen.data());
    cv::Mat m_out_edge_detect(rows, cols, CV_8UC3, g_edge_detect.data());

    unsigned char* rgb_in;
    unsigned char* rgb_out_blur;
    unsigned char* rgb_out_sharpen;
    unsigned char* rgb_out_edge_detect;

    //Init donnes kernel
    cudaMallocHost(&rgb_in, taille_rgb);
    cudaMallocHost(&rgb_out_blur, taille_rgb);
    cudaMallocHost(&rgb_out_sharpen, taille_rgb);
    cudaMallocHost(&rgb_out_edge_detect, taille_rgb);
    cudaMemcpy(rgb_in, rgb, taille_rgb, cudaMemcpyHostToDevice);

    main_blur(rgb_in, rgb_out_blur, rows, cols);
    main_sharpen(rgb_in, rgb_out_sharpen, rows, cols);
    main_edge_detect(rgb_in, rgb_out_edge_detect, rows, cols);

    //Recup donnees kernel
    cudaMemcpy(g_blur.data(), rgb_out_blur, taille_rgb, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_sharpen.data(), rgb_out_sharpen, taille_rgb, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_edge_detect.data(), rgb_out_edge_detect, taille_rgb, cudaMemcpyDeviceToHost);
    cv::imwrite("out_seq_blur.jpg", m_out_blur);
    cv::imwrite("out_seq_sharpen.jpg", m_out_sharpen);
    cv::imwrite("out_seq_edge_detect.jpg", m_out_edge_detect);
    cudaFree(rgb_in);
    cudaFree(rgb_out_blur);
    cudaFree(rgb_out_sharpen);
    cudaFree(rgb_out_edge_detect);
    return 0;
}