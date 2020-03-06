#include <opencv2/opencv.hpp>
#include <vector>

__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
  auto i = ???;
  auto j = ???;
  if( i < cols && j < rows ) {
    ???;
  }
}

int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );

  auto rgb = m_in.data;

  auto rows = m_in.rows;
  auto cols = m_in.cols;

  std::vector< unsigned char > g( rows * cols ); // image de sortie.

  cv::Mat m_out( rows, cols, CV_8UC1, g.data() );

  unsigned char * rgb_d;
  unsigned char * g_d;

  cudaMalloc( ??? ); // allocation pour l'image d'entrée sur le device.
  cudaMalloc( ??? ); // allocation pour l'image de sortie sur le device.

  cudaMemcpy( ??? ); // copie de l'image d'entrée vers le device.

  dim3 t( 32, 32 );
  dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );
  grayscale<<< b, t >>>( ??? );

  cudaMemcpy( ??? ); // récupération de l'image en niveaux de gris sur l'hôte.

  cv::imwrite( "out.jpg", m_out ); // sauvegarde de l'image.

  cudaFree( rgb_d );
  cudaFree( g_d);

  return 0;
}
