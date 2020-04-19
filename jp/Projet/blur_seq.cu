#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

/*
unsigned char* sommePixelsVoisins(const unsigned char* rgb_in, int row, int col, int rows, int cols) {
    auto* t = new unsigned char[2];
    unsigned char somme;
    unsigned char div;
    if (row * cols + col == 0) { // coin hg
        unsigned char c = rgb_in[row * cols + col];
        unsigned char d = rgb_in[row * cols + col + 3];
        unsigned char b = rgb_in[(row + 3) * cols + col];
        unsigned char bd = rgb_in[(row + 3) * cols + col + 3];
        somme = c + d + b + bd;
        div = '4';
    }
    else if (row * cols + col == cols - 1) { // coin hd
        unsigned char g = rgb_in[row * cols + col - 3];
        unsigned char c = rgb_in[row * cols + col];
        unsigned char bg = rgb_in[(row + 3) * cols + col - 3];
        unsigned char b = rgb_in[(row + 3) * cols + col];
        somme = g + c + bg + b;
        div = '4';
    }
    else if (row * cols + col == (rows - 1) * cols) { // coins bg
        unsigned char h = rgb_in[(row - 3) * cols + col];
        unsigned char hd = rgb_in[(row - 3) * cols + col + 3];
        unsigned char c = rgb_in[row * cols + col];
        unsigned char d = rgb_in[row * cols + col + 3];
        somme = h + hd + c + d;
        div = '4';
    }
    else if (row * cols + col == rows * cols - 1) { // coin bd
        unsigned char hg = rgb_in[(row - 3) * cols + col - 3];
        unsigned char h = rgb_in[(row - 3) * cols + col];
        unsigned char g = rgb_in[row * cols + col - 3];
        unsigned char c = rgb_in[row * cols + col];
        unsigned char bd = rgb_in[(row + 3) * cols + col + 3];
        somme = hg + h + g + c + bd;
        div = '4';
    }
    else if (row * cols + col < cols) { // bordure h
        unsigned char g = rgb_in[row * cols + col - 3];
        unsigned char c = rgb_in[row * cols + col];
        unsigned char d = rgb_in[row * cols + col + 3];
        unsigned char bg = rgb_in[(row + 3) * cols + col - 3];
        unsigned char b = rgb_in[(row + 3) * cols + col];
        unsigned char bd = rgb_in[(row + 3) * cols + col + 3];
        somme = g + c + d + bg + b + bd;
        div = '6';
    }
    else if (row * cols + col > (rows - 1) * cols) { // bordure b
        unsigned char hg = rgb_in[(row - 3) * cols + col - 3];
        unsigned char h = rgb_in[(row - 3) * cols + col];
        unsigned char hd = rgb_in[(row - 3) * cols + col + 3];
        unsigned char g = rgb_in[row * cols + col - 3];
        unsigned char c = rgb_in[row * cols + col];
        unsigned char d = rgb_in[row * cols + col + 3];
        somme = hg + h + hd + g + c + d;
        div = '6';
    }
    else if (row * cols + col == row * cols) { // bordure g
        unsigned char h = rgb_in[(row - 3) * cols + col];
        unsigned char hd = rgb_in[(row - 3) * cols + col + 3];
        unsigned char c = rgb_in[row * cols + col];
        unsigned char d = rgb_in[row * cols + col + 3];
        unsigned char b = rgb_in[(row + 3) * cols + col];
        unsigned char bd = rgb_in[(row + 3) * cols + col + 3];
        somme = h + hd + c + d + b + bd;
        div = '6';
    }
    else if (row * cols + col == row * cols + cols - 1) { // bordure d
        unsigned char hg = rgb_in[(row - 3) * cols + col - 3];
        unsigned char h = rgb_in[(row - 3) * cols + col];
        unsigned char g = rgb_in[row * cols + col - 3];
        unsigned char c = rgb_in[row * cols + col];
        unsigned char bg = rgb_in[(row + 3) * cols + col - 3];
        unsigned char b = rgb_in[(row + 3) * cols + col];
        somme = hg + h + g + c + bg + b;
        div = '6';
    }
    else {
        unsigned char hg = rgb_in[(row - 3) * cols + col - 3];
        unsigned char h = rgb_in[(row - 3) * cols + col];
        unsigned char hd = rgb_in[(row - 3) * cols + col + 3];
        unsigned char g = rgb_in[row * cols + col - 3];
        unsigned char c = rgb_in[row * cols + col];
        unsigned char d = rgb_in[row * cols + col + 3];
        unsigned char bg = rgb_in[(row + 3) * cols + col - 3];
        unsigned char b = rgb_in[(row + 3) * cols + col];
        unsigned char bd = rgb_in[(row + 3) * cols + col + 3];
        somme = hg + h + hd + g + c + d + bg + b + bd;
        div = '9';
    }
    t[0] = somme;
    t[1] = div;
    return t;
}
*/

void blur(unsigned char * rgb_in, unsigned char * rgb_out, int rows, int cols) {
    for (int row = 1; row < rows - 1; ++row) {
        for (int col = 1; col < cols - 1; ++col) {
            for (int i = 0; i < 3; ++i)
            {
                /*if (row * cols + col == 0) { // coin hg
                    unsigned char c = rgb_in[row * cols + col];
                    unsigned char d = rgb_in[row * cols + col + 3];
                    unsigned char b = rgb_in[(row + 3) * cols + col];
                    unsigned char bd = rgb_in[(row + 3) * cols + col + 3];
                    rgb_out[row * cols + col] = (c + d + b + bd) / 4;
                }
                else if (row * cols + col == cols - 1) { // coin hd
                    unsigned char g = rgb_in[row * cols + col - 3];
                    unsigned char c = rgb_in[row * cols + col];
                    unsigned char bg = rgb_in[(row + 3) * cols + col - 3];
                    unsigned char b = rgb_in[(row + 3) * cols + col];
                    rgb_out[row * cols + col] = (g + c + bg + b) / 4;
                }
                else if (row * cols + col == (rows - 1) * cols) { // coins bg
                    unsigned char h = rgb_in[(row - 3) * cols + col];
                    unsigned char hd = rgb_in[(row - 3) * cols + col + 3];
                    unsigned char c = rgb_in[row * cols + col];
                    unsigned char d = rgb_in[row * cols + col + 3];
                    rgb_out[row * cols + col] = (h + hd + c + d) / 4;
                }
                else if (row * cols + col == rows * cols - 1) { // coin bd
                    unsigned char hg = rgb_in[(row - 3) * cols + col - 3];
                    unsigned char h = rgb_in[(row - 3) * cols + col];
                    unsigned char g = rgb_in[row * cols + col - 3];
                    unsigned char c = rgb_in[row * cols + col];
                    unsigned char bd = rgb_in[(row + 3) * cols + col + 3];
                    rgb_out[row * cols + col] = (hg + h + g + c + bd) / 4;
                }
                else if (row * cols + col < cols) { // bordure h
                    unsigned char g = rgb_in[row * cols + col - 3];
                    unsigned char c = rgb_in[row * cols + col];
                    unsigned char d = rgb_in[row * cols + col + 3];
                    unsigned char bg = rgb_in[(row + 3) * cols + col - 3];
                    unsigned char b = rgb_in[(row + 3) * cols + col];
                    unsigned char bd = rgb_in[(row + 3) * cols + col + 3];
                    rgb_out[row * cols + col] = (g + c + d + bg + b + bd) / 6;
                }
                else if (row * cols + col > (rows - 1) * cols) { // bordure b
                    unsigned char hg = rgb_in[(row - 3) * cols + col - 3];
                    unsigned char h = rgb_in[(row - 3) * cols + col];
                    unsigned char hd = rgb_in[(row - 3) * cols + col + 3];
                    unsigned char g = rgb_in[row * cols + col - 3];
                    unsigned char c = rgb_in[row * cols + col];
                    unsigned char d = rgb_in[row * cols + col + 3];
                    rgb_out[row * cols + col] = (hg + h + hd + g + c + d) / 6;
                }
                else if (row * cols + col == row * cols) { // bordure g
                    unsigned char h = rgb_in[(row - 3) * cols + col];
                    unsigned char hd = rgb_in[(row - 3) * cols + col + 3];
                    unsigned char c = rgb_in[row * cols + col];
                    unsigned char d = rgb_in[row * cols + col + 3];
                    unsigned char b = rgb_in[(row + 3) * cols + col];
                    unsigned char bd = rgb_in[(row + 3) * cols + col + 3];
                    rgb_out[row * cols + col] = (h + hd + c + d + b + bd) / 6;
                }
                else if (row * cols + col == row * cols + cols - 1) { // bordure d
                    unsigned char hg = rgb_in[(row - 3) * cols + col - 3];
                    unsigned char h = rgb_in[(row - 3) * cols + col];
                    unsigned char g = rgb_in[row * cols + col - 3];
                    unsigned char c = rgb_in[row * cols + col];
                    unsigned char bg = rgb_in[(row + 3) * cols + col - 3];
                    unsigned char b = rgb_in[(row + 3) * cols + col];
                    rgb_out[row * cols + col] = (hg + h + g + c + bg + b) / 6;
                }
                else {*/
                unsigned char hg = rgb_in[3 * (row * cols + col - 3) + i];
                unsigned char h = rgb_in[3 * (row * cols + col) + i];
                unsigned char hd = rgb_in[3 * (row * cols + col + 3) + i];
                unsigned char g = rgb_in[3 * (row * cols + col - 3) + i];
                unsigned char c = rgb_in[3 * (row * cols + col) + i];
                unsigned char d = rgb_in[3 * (row * cols + col + 3) + i];
                unsigned char bg = rgb_in[3 * (row * cols + col - 3) + i];
                unsigned char b = rgb_in[3 * (row * cols + col) + i];
                unsigned char bd = rgb_in[3 * (row * cols + col + 3) + i];
                rgb_out[3 * (row * cols + col) + i] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
                // rgb_out[3 * (row * cols + col) + i] = 0.0625 * (hg + hd + bg + bd) + 0.125 * (h + b + g + d) + 0.25 * c;
                // }
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

    size_t taille_rgb = rows * 3 * cols;
    std::vector< unsigned char > g( taille_rgb );
    cv::Mat m_out( rows, cols, CV_8UC3, g.data() );
	
    unsigned char * rgb_in;
    unsigned char * rgb_out;
	
    //Init donnes kernel
    cudaMallocHost( &rgb_in, taille_rgb);
    cudaMallocHost( &rgb_out, taille_rgb);
    cudaMemcpy( rgb_in, rgb, taille_rgb, cudaMemcpyHostToDevice );

    //Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    blur(rgb_in, rgb_out, rows, cols);

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
    cudaMemcpy(g.data(), rgb_out, taille_rgb, cudaMemcpyDeviceToHost );
    cv::imwrite( "out_blur.jpg", m_out );
    cudaFree(rgb_in);
    cudaFree(rgb_out);
    return 0;
}