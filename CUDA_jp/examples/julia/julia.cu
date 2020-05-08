#include <opencv2/opencv.hpp>

#include <iostream>
#include <cstring>

using namespace std;

using ui32 = unsigned int;

ui32 const dim = 2048;

struct complex {
  float r; float i;
  __device__ complex(float r, float i) : r(r), i(i) {}
  __device__ float magnitude() {return r*r + i*i;}
  __device__ complex operator*(const complex& c) {
    return complex(r * c.r - i * c.i, i * c.r + r * c.i);
  }
  __device__ complex operator+(const complex& c) {
    return complex(r + c.r, i + c.i);
  }
};

__device__ unsigned char julia( int x, int y )
{
  const float scale = 1.5;

  float jx = scale * (float)(dim/2.0f - x)/(dim/2.0f);
  float jy = scale * (float)(dim/2.0f - y)/(dim/2.0f);

  ::complex c(-0.8, 0.156);
  ::complex a(jx, jy);

  for(unsigned int i = 0 ; i < 200 ; i+=2) {
    a = a * a + c;
    if(a.magnitude() > 1000) {
      return 0;
    }
  }
  return 255;
}

__global__ void fractal( unsigned char * out )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  out[ i + j * dim ] = julia( i, j );
}


int main()
{
  std::size_t const size = dim * dim;
  unsigned char * out_h;
  unsigned char * out_d;
  
  cudaMallocHost( &out_h, size );
  cudaMalloc( &out_d, size );

  dim3 block( 32, 32 );
  dim3 grid( (dim-1)/block.x + 1, (dim-1)/block.y + 1 );

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start );

  fractal<<< grid, block >>>( out_d );
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if( err != cudaSuccess )
  {
    std::cerr << cudaGetErrorString( err ) << std::endl;
  }
  
  
  cudaMemcpy( out_h, out_d, size, cudaMemcpyDeviceToHost );

  cudaDeviceSynchronize();
  cudaEventRecord( stop );
  cudaEventSynchronize( stop );

  float duration = 0.0f;
  cudaEventElapsedTime( &duration, start, stop );

  std::cout << "Total: " << duration << "ms\n";
  
  cv::Mat m_out( dim, dim, CV_8UC1, out_h );
  
  imwrite( "julia.png", m_out );

  return 0;
}

