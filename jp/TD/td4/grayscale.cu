#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <IL/il.h>

//__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows )
//{
//    auto i = blockIdx.x * blockDim.x + threadIdx.x;
//    auto j = blockIdx.y * blockDim.y + threadIdx.y;
//    if( i < cols && j < rows ) {
//        g[ j * cols + i ] = (
//                                    307 * rgb[ 3 * ( j * cols + i ) ]
//                                    + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
//                                    + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
//                            ) / 1024;
//    }
//}

__global__ void grayscale_sobel( unsigned char * in, unsigned char * out, std::size_t w, std::size_t h )
{
    auto i = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
    auto j = blockIdx.y * (blockDim.y - 2) + threadIdx.y;

    auto li = threadIdx.x;
    auto lj = threadIdx.y;

    extern __shared__ unsigned char sh[];

    if( i < w && j < h ) {
        sh[ lj * blockDim.x + li ] = (
                                             307 * in[ 3 * ( j * w + i ) ]
                                             + 604 * in[ 3 * ( j * w + i ) + 1 ]
                                             + 113 * in[  3 * ( j * w + i ) + 2 ]
                                     ) / 1024;
    }

    __syncthreads();
}

//__global__ void sobel(unsigned char * data, unsigned char * cont, std::size_t width, std::size_t height)
//{
//    extern __shared__ float t[];
//
//    auto i = blockIdx.x * blockDim.x + threadIdx.x;
//    auto j = blockIdx.y * blockDim.y + threadIdx.y;
//
//    t[j * width + i] = data[j * width + i];
//    __syncthreads();
//
//    if ( i < width-1 && j < height-1 && i>0 && j>0) {
//        int h,v,res;
//        // Horizontal
//        h =     t[((j - 1) * width + i - 1)] -     t[((j - 1) * width + i + 1)]
//                + 2 * t[( j      * width + i - 1)] - 2 * t[( j      * width + i + 1)]
//                +     t[((j + 1) * width + i - 1)] -     t[((j + 1) * width + i + 1)];
//
//        // Vertical
//        v =     t[((j - 1) * width + i - 1)] -     t[((j + 1) * width + i - 1)]
//                + 2 * t[((j - 1) * width + i    )] - 2 * t[((j + 1) * width + i    )]
//                +     t[((j - 1) * width + i + 1)] -     t[((j + 1) * width + i + 1)];
//
//        //h = h > 255 ? 255 : h;
//        //v = v > 255 ? 255 : v;
//
//        res = h*h + v*v;
//        res = res > 255*255 ? res = 255*255 : res;
//
//        cont[(j * width + i)] = sqrtf(res);
//    }
//}

int main()
{
    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
    auto rgb = m_in.data;
    auto rows = m_in.rows;
    auto cols = m_in.cols;
    std::vector< unsigned char > g( rows * cols );
    cv::Mat m_out( rows, cols, CV_8UC1, g.data() );
    unsigned char * rgb_d;
    unsigned char * g_d;
    unsigned char * out_d;

    cudaMalloc( &rgb_d, 3 * rows * cols );
    cudaMalloc( &g_d, rows * cols );
    cudaMalloc( &out_d, rows * cols );
    cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );

    dim3 t( 32, 32 );
    dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );

    grayscale_sobel<<< b, t, t.x * t.y >>>( rgb_d, g_d, cols, rows );

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString( err );
    }

    cudaMemcpy( g.data(), g_d, rows * cols, cudaMemcpyDeviceToHost );
    cv::imwrite( "out.jpg", m_out );
    std::cout << "grayscale fini" << std::endl;

//    std::size_t t_shared = cols*rows*sizeof("float");
//    sobel<<< b,t,t_shared >>>(g_d, rgb_d, cols, rows );
//
//    cudaMemcpy(  g.data(), rgb_d, rows * cols, cudaMemcpyDeviceToHost );
//    cv::imwrite( "out_cont_shared.jpg", m_out );
//    std::cout << "sobel fini" << std::endl;

    cudaFree( rgb_d);
    cudaFree( g_d);
    return 0;
}
