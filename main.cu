#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
#include <IL/il.h>
#include <typeinfo>

using namespace std;

__global__ void blur(unsigned char * mat_in, unsigned char * mat_out, std::size_t cols, std::size_t rows) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  // if( i > 0 && i < (blockDim.x - 1) && j > 0 && j < (blockDim.y - 1) )
  if (j*cols+i < 3*cols*rows)
  {
    mat_out[j * cols + i] = mat_in[j * cols + i];

  }
}


int main()
{
  //Declarations
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
  unsigned char * rgb = m_in.data;
  int rows = m_in.rows;
  int cols = m_in.cols;
  cv::Mat planes[3];
  cv::split(m_in, planes);

  unsigned char * b = planes[0].data;
  unsigned char * v = planes[1].data;
  unsigned char * r = planes[2].data;

  // for (int i=0; i<rows; i++)
  //   for (int j=0; j<cols; j++)
  //     cout << r[j * cols + i];



  std::vector<unsigned char> g(rows * cols * 3); //Pour recreer l'image
  cv::Mat m_out(rows, cols, CV_8UC3, g.data());
  unsigned char * mat_in;
  unsigned char * mat_out;

  //Init donnes kernel
  cudaMalloc( &mat_in, 3 * rows * cols );
  cudaMalloc( &mat_out, 3 * rows * cols );
  cudaMemcpy( mat_in, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );

  //Calcul du nb de blocs et de threads
  // dim3 t( 32, 32 );
  // dim3 b( ( cols - 1) / (t.x-2) + 1 , ( rows - 1 ) / (t.y-2) + 1 );

  dim3 block( 32, 32);
  dim3 grid(((cols-1) / block.x + 1), 3*((rows-1) / block.y + 1));

  //Debut de chrono
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  //Appel kernel
  blur<<< grid, block>>>(mat_in, mat_out, cols, rows );

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
  cudaMemcpy( g.data(), mat_out, 3*rows * cols, cudaMemcpyDeviceToHost );

  cv::imwrite( "out.jpg", m_out );

  cudaFree(mat_in);
  cudaFree(mat_out);

  return 0;
}
