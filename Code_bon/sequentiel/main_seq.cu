//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <vector>

void blur(const unsigned char* rgb_in, unsigned char* rgb_out, int rows, int cols) {
    for (std::size_t row = 1; row < rows - 1; ++row) {
        for (std::size_t col = 1; col < cols - 1; ++col) {
            for (std::size_t rgb = 0; rgb < 3; ++rgb)
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

void sharpen(const unsigned char* rgb_in, unsigned char* rgb_out, int rows, int cols) {
    for (std::size_t row = 1; row < rows - 1; ++row) {
        for (std::size_t col = 1; col < cols - 1; ++col) {
            for (std::size_t rgb = 0; rgb < 3; ++rgb)
            {
                unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
                unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
                unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
                unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
                unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
                int somme = (-3 * (h + g + d + b) + 21 * c) / 9;
                
                if (somme > 255) somme = 255;
                if (somme < 0) somme = 0;
                rgb_out[3 * (row * cols + col) + rgb] = somme;
            }
        }
    }
}

void edge_detect(const unsigned char* rgb_in, unsigned char* rgb_out, int rows, int cols) {
    for (std::size_t row = 1; row < rows - 1; ++row) {
        for (std::size_t col = 1; col < cols - 1; ++col) {
            for (std::size_t rgb = 0; rgb < 3; ++rgb)
            {
                unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
                unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
                unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
                unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
                unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
                int somme = (9 * (h + g + d + b) - 36 * c) / 9;

                if (somme > 255) somme = 255;
                if (somme < 0) somme = 0;
                rgb_out[3 * (row * cols + col) + rgb] = somme;
            }
        }
    }
}


void main_blur_edge_detect(const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols)
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
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "blur_seq: " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void main_sharpen(const unsigned char* rgb_in, unsigned char* rgb_out_sharpen, int rows, int cols)
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
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "sharpen_seq: " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void main_edge_detect_blur(const unsigned char* rgb_in, unsigned char* rgb_out_edge_detect, int rows, int cols)
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
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "edge_detect_seq: " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main()
{
    // Declarations
    cudaError_t err;

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

    // Init donnes kernel
    err = cudaMallocHost(&rgb_in, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMallocHost(&rgb_out_blur, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMallocHost(&rgb_out_sharpen, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMallocHost(&rgb_out_edge_detect, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMemcpy(rgb_in, rgb, taille_rgb, cudaMemcpyHostToDevice);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    // Execution
    main_blur_edge_detect(rgb_in, rgb_out_blur, rows, cols);
    main_sharpen(rgb_in, rgb_out_sharpen, rows, cols);
    main_edge_detect_blur(rgb_in, rgb_out_edge_detect, rows, cols);

    // Recup donnees kernel
    err = cudaMemcpy(g_blur.data(), rgb_out_blur, taille_rgb, cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMemcpy(g_sharpen.data(), rgb_out_sharpen, taille_rgb, cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMemcpy(g_edge_detect.data(), rgb_out_edge_detect, taille_rgb, cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    cv::imwrite("out_seq_blur.jpg", m_out_blur);
    cv::imwrite("out_seq_sharpen.jpg", m_out_sharpen);
    cv::imwrite("out_seq_edge_detect.jpg", m_out_edge_detect);

    // Nettoyage memoire
    cudaFreeHost(rgb_in);
    cudaFreeHost(rgb_out_blur);
    cudaFreeHost(rgb_out_sharpen);
    cudaFreeHost(rgb_out_edge_detect);
    return 0;
}