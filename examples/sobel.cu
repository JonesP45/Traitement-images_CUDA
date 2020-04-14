#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
#include <IL/il.h>

__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i < cols && j < rows ) {
    g[ j * cols + i ] = (
			 307 * rgb[ 3 * ( j * cols + i ) ]
			 + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
			 + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
			 ) / 1024;
  }
}

__global__ void grayscale2(unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows )
{
  auto i = threadIdx.x; //coord du thread dans le block
  auto j = threadIdx.y;
  auto gi = blockIdx.x * (blockDim.x-2) + i; //coord globale du thread dans image
  auto gj = blockIdx.y * (blockDim.y-2) + j;

  extern __shared__ unsigned char sh[];

  if( gi < cols && gj < rows ) {
    g[ gj * cols + gi ] = (
			 307 * rgb[ 3 * ( gj * cols + gi ) ]
			 + 604 * rgb[ 3 * ( gj * cols + gi ) + 1 ]
			 + 113 * rgb[  3 * ( gj * cols + gi ) + 2 ]
			 ) / 1024;
  }
  // if( gi < cols && gj < rows )
  //   sh[j*blockDim.x+i] = (
  //      307 * rgb[ 3 * ( gj * cols + gi ) ]
  //      + 604 * rgb[ 3 * ( gj * cols + gi ) + 1 ]
  //      + 113 * rgb[  3 * ( gj * cols + gi ) + 2 ]
  //    ) / 1024;

  sh[j*blockDim.x + i] = g[gj*cols + gi];

  __syncthreads();



}

int main()
{
  //Declarations
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;
  std::vector< unsigned char > g( rows * cols );
  cv::Mat m_out( rows, cols, CV_8UC1, g.data() );
  unsigned char * rgb_d;
  unsigned char * tmp;
  unsigned char * g_d;

  //Init donnes kernel
  cudaMalloc( &rgb_d, 3 * rows * cols );
  cudaMalloc( &tmp, rows * cols );
  cudaMalloc( &g_d, rows * cols );
  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );

  //Calcul du nb de blocs et de threads
  dim3 t( 32, 32 );
  dim3 b( ( cols - 1) / (t.x-2) + 1 , ( rows - 1 ) / (t.y-2) + 1 );

  //Debut de chrono
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  //Appel kernel
  grayscale2<<< b, t, t.x*t.y>>>( rgb_d, g_d, cols, rows );

  //Fin de chrono
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, stop);
  std::cout << elapsedTime << std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //Recup donnees kernel
  cudaMemcpy( g.data(), g_d, rows * cols, cudaMemcpyDeviceToHost );

  cv::imwrite( "out.jpg", m_out );

  cudaFree(rgb_d);
  cudaFree(g_d);

  return 0;
}
