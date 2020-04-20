#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
#include <IL/il.h>
#include <typeinfo>

using namespace std;

__global__ void copy(unsigned char * mat_in, unsigned char * mat_out, std::size_t cols, std::size_t rows) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  //if (i*cols+j < 3*cols*rows) equivalent
  if (j<rows*3 && i<cols)
  {
    mat_out[j * cols*3 + i] = mat_in[j * cols*3 + i];

  }
}

__global__ void blur(unsigned char * mat_in, unsigned char * mat_out, std::size_t cols, std::size_t rows) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
  auto j = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

  //if (j<rows*3 && i<cols && j>3 )
  if (j>2 && j<rows*3 && i<cols)
  {
    //p1 à p9 correspondent aux 9 pixels à récupérer
    unsigned char p1 = mat_in[(j-3) * cols + i - 3];
    unsigned char p2 = mat_in[(j-3) * cols + i];
    unsigned char p3 = mat_in[(j-3) * cols + i + 3];
    unsigned char p4 = mat_in[j * cols + i - 3];
    unsigned char p5 = mat_in[j * cols + i];
    unsigned char p6 = mat_in[j * cols + i + 3];
    unsigned char p7 = mat_in[(j+3) * cols + i - 3];
    unsigned char p8 = mat_in[(j+3) * cols + i];
    unsigned char p9 = mat_in[(j+3) * cols + i + 3];

    mat_out[j * cols + i] = (p1+p2+p3+p4+p5+p6+p7+p8+p9)/9;
  }
  //pour la premiere ligne
  else if (j<=2 && j<rows*3 && i<cols)
  {
    unsigned char p4 = mat_in[j * cols + i - 3];
    unsigned char p5 = mat_in[j * cols + i];
    unsigned char p6 = mat_in[j * cols + i + 3];
    unsigned char p7 = mat_in[(j+3) * cols + i - 3];
    unsigned char p8 = mat_in[(j+3) * cols + i];
    unsigned char p9 = mat_in[(j+3) * cols + i + 3];

    mat_out[j * cols + i] = (p4+p5+p6+p7+p8+p9)/6;
  }
}

__global__ void sharpen(unsigned char * mat_in, unsigned char * mat_out, std::size_t cols, std::size_t rows) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
  auto j = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

  //if (j<rows*3 && i<cols && j>3 )
  if (j>2 && j<rows*3 && i<cols)
  {
    //p1 à p9 correspondent aux 9 pixels à récupérer
    unsigned char p2 = mat_in[(j-3) * cols + i];
    unsigned char p4 = mat_in[j * cols + i - 3];
    unsigned char p5 = mat_in[j * cols + i];
    unsigned char p6 = mat_in[j * cols + i + 3];
    unsigned char p8 = mat_in[(j+3) * cols + i];

    // int tmp =  (-3*(p2+p4+p6+p8)+21*p5)/9;
    mat_out[j * cols + i] = ((-3*(p2+p4+p6+p8)+21*p5)/9)%255;
  }
}

__global__ void edge_detect(unsigned char * mat_in, unsigned char * mat_out, std::size_t cols, std::size_t rows) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
  auto j = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

  //if (j<rows*3 && i<cols && j>3 )
  if (j>2 && j<rows*3 && i<cols)
  {
    //p1 à p9 correspondent aux 9 pixels à récupérer
    unsigned char p2 = mat_in[(j-3) * cols + i];
    unsigned char p4 = mat_in[j * cols + i - 3];
    unsigned char p5 = mat_in[j * cols + i];
    unsigned char p6 = mat_in[j * cols + i + 3];
    unsigned char p8 = mat_in[(j+3) * cols + i];

    // int tmp =  (-3*(p2+p4+p6+p8)+21*p5)/9;
    mat_out[j * cols + i] = ((9*(p2+p4+p6+p8)-36*p5)/5)%255;
  }
}


int main()
{
  //Declarations
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
  unsigned char * rgb = m_in.data;
  int rows = m_in.rows;
  int cols = m_in.cols;

  // cv::Mat planes[3];
  // cv::split(m_in, planes);
  // unsigned char * b = planes[0].data;
  // unsigned char * v = planes[1].data;
  // unsigned char * r = planes[2].data;
  // for (int i=0; i<rows; i++)
  //   for (int j=0; j<cols; j++)
  //     cout << r[j * cols + i];


  vector<unsigned char> g(rows * cols * 3); //Pour recreer l'image
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

  dim3 block( 32, 32); //nb de thread, max 1024
  dim3 grid(((cols-1) / block.x + 1), 3*((rows-1) / block.y + 1));

  //Debut de chrono
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  //Appel kernel
  // blur<<< grid, block>>>(mat_in, mat_out, cols, rows);
  sharpen<<< grid, block>>>(mat_in, mat_out, cols, rows);
  // edge_detect<<< grid, block>>>(mat_in, mat_out, cols, rows);

  //Fin de chrono
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cout << cudaGetErrorString(cudaGetLastError()) << endl;
  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, stop);
  cout << elapsedTime << endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //Recup donnees kernel
  cudaMemcpy( g.data(), mat_out, 3*rows * cols, cudaMemcpyDeviceToHost );

  cv::imwrite( "out.jpg", m_out );

  cudaFree(mat_in);
  cudaFree(mat_out);


//
//   int cols = 5;
//   int rows = 10;
//   unsigned char * mat;
//   cudaMalloc(mat, 3 * rows * cols );
//   for (int i=0; i<cols*rows*3; i++)
//   {
//       mat[i] = i+1;
//   }
//   cv::Mat m = (rows, cols, CV_8UC3, mat);
// cv::imwrite( "test.jpg", m );


  return 0;
}
