#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

__global__ void grayscale(unsigned char* rgb, unsigned char* g, std::size_t cols, std::size_t rows) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < cols && j < rows) {
        g[j * cols + i] =
            (
                307 * rgb[3 * (j * cols + i)]
                + 604 * rgb[3 * (j * cols + i) + 1]
                + 113 * rgb[3 * (j * cols + i) + 2]
                )
            / 1024;
    }
}

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
                int tmp = 3 * (row * cols + col - 3);
                cout << tmp << endl;
                unsigned char hg = rgb_in[tmp];
                unsigned char h = rgb_in[3 * (row * cols + col)];
                unsigned char hd = rgb_in[3 * (row * cols + col + 3)];
                unsigned char g = rgb_in[3 * (row * cols + col - 3)];
                unsigned char c = rgb_in[3 * (row * cols + col)];
                unsigned char d = rgb_in[3 * (row * cols + col + 3)];
                unsigned char bg = rgb_in[3 * (row * cols + col - 3)];
                unsigned char b = rgb_in[3 * (row * cols + col)];
                unsigned char bd = rgb_in[3 * (row * cols + col + 3)];
                rgb_out[3 * (row * cols + col) + i] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
                // }
            }
        }
    }
}

int main()
{
    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    auto rgb = m_in.data;
    auto rows = m_in.rows;
    auto cols = m_in.cols;
    for (int i = 5763; i < 5766; ++i)
    {
        cout << (int) rgb[i] << endl;
    }

    size_t taille_rgb = 3 * rows * cols;
    std::vector< unsigned char > g( taille_rgb );
    cv::Mat m_out( rows, cols, CV_8UC3, g.data() );

    unsigned char * rgb_in;
    unsigned char * rgb_out;
    cudaMalloc( &rgb_in, taille_rgb);
    cudaMalloc( &rgb_out, taille_rgb);
    cout << (int)rgb_in[0] << endl;
    cudaMemcpy( rgb_in, rgb, taille_rgb, cudaMemcpyHostToDevice );
    for (int i = 0; i < 1; ++i)
    {
        cout << (int) rgb_in[i] << endl;
    }

    // blur(rgb_in, rgb_out, rows, cols);

    cudaMemcpy(g.data(), rgb_out, taille_rgb, cudaMemcpyDeviceToHost );
    cv::imwrite( "out.jpg", m_out );
    cudaFree(rgb_in);
    cudaFree(rgb_out);
    return 0;
}
