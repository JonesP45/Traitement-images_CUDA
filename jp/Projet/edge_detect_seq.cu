#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

void edge_detect(unsigned char* rgb_in, unsigned char* rgb_out, int rows, int cols) {
    for (int row = 1; row < rows - 1; ++row) {
        for (int col = 1; col < cols - 1; ++col) {
            for (int i = 0; i < 3; ++i)
            {
                unsigned char h = rgb_in[3 * (row * cols + col) + i];
                unsigned char g = rgb_in[3 * (row * cols + col - 3) + i];
                unsigned char c = rgb_in[3 * (row * cols + col) + i];
                unsigned char d = rgb_in[3 * (row * cols + col + 3) + i];
                unsigned char b = rgb_in[3 * (row * cols + col) + i];
                rgb_out[3 * (row * cols + col) + i] = (9 * (h + g + d + b) - 36 * c) / 9;
            }
        }
    }
}

int main()
{
    //Declarations
    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    auto rgb = m_in.data;
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    size_t taille_rgb = 3 * rows * cols;
    std::vector< unsigned char > g(taille_rgb);
    cv::Mat m_out(rows, cols, CV_8UC3, g.data());

    unsigned char* rgb_in;
    unsigned char* rgb_out;

    //Init donnes kernel
    cudaMallocHost(&rgb_in, taille_rgb);
    cudaMallocHost(&rgb_out, taille_rgb);
    cudaMemcpy(rgb_in, rgb, taille_rgb, cudaMemcpyHostToDevice);

    //Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    edge_detect(rgb_in, rgb_out, rows, cols);

    //Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cout << cudaGetErrorString(cudaGetLastError()) << endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << elapsedTime << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //Recup donnees kernel
    cudaMemcpy(g.data(), rgb_out, taille_rgb, cudaMemcpyDeviceToHost);
    cv::imwrite("out_edge_detect.jpg", m_out);
    cudaFree(rgb_in);
    cudaFree(rgb_out);
    return 0;
}