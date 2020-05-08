//
// Created by jean-pacome on 04/03/2020.
//

#include "blur_seq.h"
#include <opencv2/opencv.hpp>


int main()
{

    cv::Mat m_in = cv::imread( "in.jpg", cv::IMREAD_UNCHANGED );
    auto rgb = m_in.data;
    auto rows = m_in.rows;
    auto cols = m_in.cols;
    unsigned char * rgb_d;
    malloc( &rgb_d, 3 * rows * cols );
    cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

        }
    }

    cv::imwrite( "out.jpg", m_out );
    return 0;
}