#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>

#define taille_stream 2
std::size_t taille_rgb = 0;
std::size_t one_line_rgb = 0;


__global__ void blur2D(const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    if (col >= 1 && col < cols - 1 && row >= 1 && row < rows - 1) {
        for (int rgb = 0; rgb < 3; ++rgb) {
            unsigned char hg = rgb_in[3 * ((row - 1) * cols + col - 1) + rgb];
            unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
            unsigned char hd = rgb_in[3 * ((row - 1) * cols + col + 1) + rgb];
            unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
            unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
            unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
            unsigned char bg = rgb_in[3 * ((row + 1) * cols + col - 1) + rgb];
            unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
            unsigned char bd = rgb_in[3 * ((row + 1) * cols + col + 1) + rgb];

            rgb_out_blur[3 * (row * cols + col) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
        }
    }
}

__global__ void sharpen2D(const unsigned char* rgb_in, unsigned char* rgb_out_sharpen, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
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
            rgb_out_sharpen[3 * (row * cols + col) + rgb] = somme;
        }
    }
}

__global__ void edge_detect2D(const unsigned char* rgb_in, unsigned char* rgb_out_edge_detect, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
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
            rgb_out_edge_detect[3 * (row * cols + col) + rgb] = somme;
        }
    }
}


__global__ void blur3D(const unsigned char * mat_in, unsigned char * mat_out_blur, std::size_t cols, std::size_t rows) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto j = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y
    auto k = threadIdx.z;

    if (j >= 1 && j < rows - 1 && i >= 1 && i < cols - 1)
    {
        //p1 à p9 correspondent aux 9 pixels à récupérer
        unsigned char p1 = mat_in[3 * ((j-1) * cols + i - 1) + k];
        unsigned char p2 = mat_in[3 * ((j-1) * cols + i) + k];
        unsigned char p3 = mat_in[3 * ((j-1) * cols + i + 1) + k];
        unsigned char p4 = mat_in[3 * (j * cols + i - 1) + k];
        unsigned char p5 = mat_in[3 * (j * cols + i) + k];
        unsigned char p6 = mat_in[3 * (j * cols + i + 1) + k];
        unsigned char p7 = mat_in[3 * ((j+1) * cols + i - 1) + k];
        unsigned char p8 = mat_in[3 * ((j+1) * cols + i) + k];
        unsigned char p9 = mat_in[3 * ((j+1) * cols + i + 1) + k];

        mat_out_blur[3 * (j * cols + i) + k] = (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / 9;
    }
}

__global__ void sharpen3D(const unsigned char * mat_in, unsigned char * mat_out_sharpen, std::size_t cols, std::size_t rows) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto j = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y
    auto k = threadIdx.z;

    if (j >= 1 && j < rows - 1 && i >= 1 && i < cols - 1)
    {
        //p1 à p9 correspondent aux 9 pixels à récupérer
        unsigned char p2 = mat_in[3 * ((j-1) * cols + i) + k];
        unsigned char p4 = mat_in[3 * (j * cols + i - 1) + k];
        unsigned char p5 = mat_in[3 * (j * cols + i) + k];
        unsigned char p6 = mat_in[3 * (j * cols + i + 1) + k];
        unsigned char p8 = mat_in[3 * ((j+1) * cols + i) + k];
        int tmp =  (-3*(p2+p4+p6+p8)+21*p5)/9;

        if (tmp > 255) tmp = 255;
        if (tmp < 0) tmp = 0;

        mat_out_sharpen[3 * (j * cols + i) + k] = tmp;
    }
}

__global__ void edge_detect3D(const unsigned char * mat_in, unsigned char * mat_out_edge_detect, std::size_t cols, std::size_t rows) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto j = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y
    auto k = threadIdx.z;

    if (j >= 1 && j < rows - 1 && i >= 1 && i < cols - 1)
    {
        //p1 à p9 correspondent aux 9 pixels à récupérer
        unsigned char p2 = mat_in[3 * ((j-1) * cols + i) + k];
        unsigned char p4 = mat_in[3 * (j * cols + i - 1) + k];
        unsigned char p5 = mat_in[3 * (j * cols + i) + k];
        unsigned char p6 = mat_in[3 * (j * cols + i + 1) + k];
        unsigned char p8 = mat_in[3 * ((j+1) * cols + i) + k];
        int tmp =  (9*(p2+p4+p6+p8)-36*p5)/9;

        if (tmp > 255) tmp = 255;
        if (tmp < 0) tmp = 0;

        mat_out_edge_detect[3 * (j * cols + i) + k] = tmp;
    }
}


void main_blur(const dim3 grid, const dim3 block, const cudaStream_t* streams, const unsigned char* rgb_in,
               unsigned char* rgb_out_blur, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        for (std::size_t i = 0; i < taille_stream; ++i) {
            int row_stream = (int) (rows / taille_stream) + ((i == 0 || i == taille_stream - 1) ? 1 : 2);
            std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
            blur2D<<< grid, block, 0, streams[i] >>>(rgb_in + decalage,
                    rgb_out_blur + decalage, row_stream, cols);
        }
    } else {
        for (std::size_t i = 0; i < taille_stream; ++i) {
            int row_stream = (int) (rows / taille_stream) + ((i == 0 || i == taille_stream - 1) ? 1 : 2);
            std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
            blur3D<<< grid, block, 0, streams[i] >>>(rgb_in + decalage, rgb_out_blur + decalage,
                    cols, row_stream);
        }
    }

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "blur_stream_" << block.x << "-" << block.y << "-" << block.z << ": " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void main_sharpen(const dim3 grid, const dim3 block, const cudaStream_t* streams, const unsigned char* rgb_in,
                  unsigned char* rgb_out_sharpen, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        for (std::size_t i = 0; i < taille_stream; ++i) {
            int row_stream = (int) (rows / taille_stream) + ((i == 0 || i == taille_stream - 1) ? 1 : 2);
            std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
            sharpen2D<<< grid, block, 0, streams[i] >>>(rgb_in + decalage,
                    rgb_out_sharpen + decalage, row_stream, cols);
        }
    } else {
        for (std::size_t i = 0; i < taille_stream; ++i) {
            int row_stream = (int) (rows / taille_stream) + ((i == 0 || i == taille_stream - 1) ? 1 : 2);
            std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
            sharpen3D<<< grid, block, 0, streams[i] >>>(rgb_in + decalage, rgb_out_sharpen + decalage,
                    cols, row_stream);
        }
    }

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "sharpen_stream_" << block.x << "-" << block.y << "-" << block.z << ": " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void main_edge_detect(const dim3 grid, const dim3 block, const cudaStream_t* streams, const unsigned char* rgb_in,
                      unsigned char* rgb_out_edge_detect, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        for (std::size_t i = 0; i < taille_stream; ++i) {
            int row_stream = (int) (rows / taille_stream) + ((i == 0 || i == taille_stream - 1) ? 1 : 2);
            std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
            edge_detect2D<<< grid, block, 0, streams[i] >>>(rgb_in + decalage,
                    rgb_out_edge_detect + decalage, row_stream, cols);
        }
    } else {
        for (std::size_t i = 0; i < taille_stream; ++i) {
            int row_stream = (int) (rows / taille_stream) + ((i == 0 || i == taille_stream - 1) ? 1 : 2);
            std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
            edge_detect3D<<< grid, block, 0, streams[i] >>>(rgb_in + decalage,
                    rgb_out_edge_detect + decalage, cols, row_stream);
        }
    }

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "edge_detect_stream_" << block.x << "-" << block.y << "-" << block.z << ": " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void main_blur_edge_detect(const dim3 grid, const dim3 block, const cudaStream_t* streams, const unsigned char* rgb_in,
        unsigned char* rgb_tmp_blur_edge_detect, unsigned char* rgb_out_blur_edge_detect, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        for (std::size_t i = 0; i < taille_stream; ++i) {
            int row_stream = (int) (rows / taille_stream) + ((i == 0 || i == taille_stream - 1) ? 1 : 2);
            std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
            blur2D<<< grid, block, 0, streams[i] >>>(rgb_in + decalage,
                    rgb_tmp_blur_edge_detect + decalage, row_stream, cols);
            edge_detect2D<<< grid, block, 0, streams[i] >>>(rgb_tmp_blur_edge_detect + decalage,
                    rgb_out_blur_edge_detect + decalage, row_stream, cols);
        }
    } else {
        for (std::size_t i = 0; i < taille_stream; ++i) {
            int row_stream = (int) (rows / taille_stream) + ((i == 0 || i == taille_stream - 1) ? 1 : 2);
            std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
            blur3D<<< grid, block, 0, streams[i] >>>(rgb_in + decalage,
                    rgb_tmp_blur_edge_detect + decalage, cols, row_stream);
            edge_detect3D<<< grid, block, 0, streams[i] >>>(rgb_tmp_blur_edge_detect + decalage,
                    rgb_out_blur_edge_detect + decalage, cols, row_stream);
        }
    }

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "blur_edge_detect_stream_" << block.x << "-" << block.y << "-" << block.z << ": " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void main_edge_detect_blur(const dim3 grid, const dim3 block, const cudaStream_t* streams, const unsigned char* rgb_in,
                           unsigned char* rgb_tmp_blur_edge_detect, unsigned char* rgb_out_blur_edge_detect, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        for (std::size_t i = 0; i < taille_stream; ++i) {
            int row_stream = (int) (rows / taille_stream) + ((i == 0 || i == taille_stream - 1) ? 1 : 2);
            std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
            edge_detect2D<<< grid, block, 0, streams[i] >>>(rgb_in + decalage,
                                                     rgb_tmp_blur_edge_detect + decalage, row_stream, cols);
            blur2D<<< grid, block, 0, streams[i] >>>(rgb_tmp_blur_edge_detect + decalage,
                                                            rgb_out_blur_edge_detect + decalage, row_stream, cols);
        }
    } else {
        for (std::size_t i = 0; i < taille_stream; ++i) {
            int row_stream = (int) (rows / taille_stream) + ((i == 0 || i == taille_stream - 1) ? 1 : 2);
            std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
            edge_detect3D<<< grid, block, 0, streams[i] >>>(rgb_in + decalage,
                                                     rgb_tmp_blur_edge_detect + decalage, cols, row_stream);
            blur3D<<< grid, block, 0, streams[i] >>>(rgb_tmp_blur_edge_detect + decalage,
                                                            rgb_out_blur_edge_detect + decalage, cols, row_stream);
        }
    }

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "edge_detect_blur_stream_" << block.x << "-" << block.y << "-" << block.z << ": " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main(int argc, char *argv[])
{
    // Declarations
    cudaError_t err;

    std::string filename = std::string(argv[1]) + std::string(".") + std::string(argv[2]);
    std::string out(argv[1]);
    if (out == "in") {
        out = std::string("out");
    }

    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    unsigned char* rgb = m_in.data;
    int rows = m_in.rows;
    int cols = m_in.cols;

    taille_rgb = 3 * rows * cols;

    std::vector<unsigned char> g_blur(taille_rgb);

    std::vector<unsigned char> g_sharpen(taille_rgb);
    std::vector<unsigned char> g_edge_detect(taille_rgb);

    std::vector<unsigned char> g_blur_edge_detect(taille_rgb);
    std::vector<unsigned char> g_edge_detect_blur(taille_rgb);

    cv::Mat m_out_blur(rows, cols, CV_8UC3, g_blur.data());

    cv::Mat m_out_sharpen(rows, cols, CV_8UC3, g_sharpen.data());
    cv::Mat m_out_edge_detect(rows, cols, CV_8UC3, g_edge_detect.data());

    cv::Mat m_out_blur_edge_detect(rows, cols, CV_8UC3, g_blur_edge_detect.data());
    cv::Mat m_out_edge_detect_blur(rows, cols, CV_8UC3, g_edge_detect_blur.data());

    unsigned char* rgb_in = nullptr;

    unsigned char* rgb_out_blur = nullptr;
    unsigned char* rgb_out_sharpen = nullptr;
    unsigned char* rgb_out_edge_detect = nullptr;

    unsigned char* rgb_tmp_blur_edge_detect = nullptr;
    unsigned char* rgb_tmp_edge_detect_blur = nullptr;
    unsigned char* rgb_out_blur_edge_detect = nullptr;
    unsigned char* rgb_out_edge_detect_blur = nullptr;

    // Init donnes kernel
    err = cudaMalloc(&rgb_in, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    err = cudaMalloc(&rgb_out_blur, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_out_sharpen, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_out_edge_detect, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    err = cudaMalloc(&rgb_tmp_blur_edge_detect, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_tmp_edge_detect_blur, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_out_blur_edge_detect, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_out_edge_detect_blur, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    cudaStream_t streams[taille_stream];
    for (std::size_t i = 0; i < taille_stream; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    one_line_rgb = 3 * cols;

    for (std::size_t i = 0; i < taille_stream; ++i) {
        std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
        std::size_t count = taille_rgb / taille_stream + ((i == 0 || i == taille_stream - 1) ? one_line_rgb : 2 * one_line_rgb);
        err = cudaMemcpyAsync(rgb_in + decalage,rgb + decalage, count, cudaMemcpyHostToDevice, streams[i]);
        if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    }

    /////////////////////////////////////////////////////////////////
    ///////////////////// block 32 32 ///////////////////////////////
    /////////////////////////////////////////////////////////////////

    dim3 block_32_32(32, 32 / taille_stream); //nb de thread par bloc, max 1024
    dim3 grid_32_32(((cols - 1) / block_32_32.x + 1), (((rows + (taille_stream - 1) * 2) / taille_stream - 1) / block_32_32.y + 1)); // nb de block

    // Execution
    main_blur(grid_32_32, block_32_32, streams, rgb_in, rgb_out_blur, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    main_sharpen(grid_32_32, block_32_32, streams, rgb_in, rgb_out_sharpen, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    main_edge_detect(grid_32_32, block_32_32, streams, rgb_in, rgb_out_edge_detect, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    main_blur_edge_detect(grid_32_32, block_32_32, streams, rgb_in, rgb_tmp_blur_edge_detect, rgb_out_blur_edge_detect, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    main_edge_detect_blur(grid_32_32, block_32_32, streams, rgb_in, rgb_tmp_edge_detect_blur, rgb_out_edge_detect_blur, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    // Recup donnees kernel
    for (std::size_t i = 0; i < taille_stream; ++i) {
        err = cudaMemcpyAsync(g_blur.data() + i * taille_rgb / taille_stream,
                              rgb_out_blur + i * taille_rgb / taille_stream, taille_rgb / taille_stream,
                              cudaMemcpyDeviceToHost, streams[i]);
        if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
        err = cudaMemcpyAsync(g_sharpen.data() + i * taille_rgb / taille_stream,
                              rgb_out_sharpen + i * taille_rgb / taille_stream, taille_rgb / taille_stream,
                              cudaMemcpyDeviceToHost, streams[i]);
        if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
        err = cudaMemcpyAsync(g_edge_detect.data() + i * taille_rgb / taille_stream,
                              rgb_out_edge_detect + i * taille_rgb / taille_stream, taille_rgb / taille_stream,
                              cudaMemcpyDeviceToHost, streams[i]);
        if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

        err = cudaMemcpyAsync(g_blur_edge_detect.data() + i * taille_rgb / taille_stream,
                rgb_out_blur_edge_detect + i * taille_rgb / taille_stream, taille_rgb / taille_stream,
                cudaMemcpyDeviceToHost, streams[i]);
        if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
        err = cudaMemcpyAsync(g_edge_detect_blur.data() + i * taille_rgb / taille_stream,
                rgb_out_edge_detect_blur + i * taille_rgb / taille_stream, taille_rgb / taille_stream,
                cudaMemcpyDeviceToHost, streams[i]);
        if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    }

    cudaDeviceSynchronize();

    cv::imwrite(out + std::string("_stream_block_32-32_blur.") + std::string(argv[2]), m_out_blur);
    cv::imwrite(out + std::string("_stream_block_32-32_sharpen.") + std::string(argv[2]), m_out_sharpen);
    cv::imwrite(out + std::string("_stream_block_32-32_edge_detect.") + std::string(argv[2]), m_out_edge_detect);

    cv::imwrite(out + std::string("_stream_block_32-32_blur_edge_detect.") + std::string(argv[2]), m_out_blur_edge_detect);
    cv::imwrite(out + std::string("_stream_block_32-32_edge_detect_blur.") + std::string(argv[2]), m_out_edge_detect_blur);

    /////////////////////////////////////////////////////////////////
    ///////////////////// block 17 20 3 /////////////////////////////
    /////////////////////////////////////////////////////////////////

    dim3 block_17_20_3(17, 20 / taille_stream, 3); //nb de thread par bloc, max 1024
    dim3 grid_17_20_3(((cols - 1) / block_17_20_3.x + 1),
            (((rows + (taille_stream - 1) * 2) / taille_stream - 1) / block_17_20_3.y + 1)); // nb de block

    // Execution
    main_blur(grid_17_20_3, block_17_20_3, streams, rgb_in, rgb_out_blur, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    main_sharpen(grid_17_20_3, block_17_20_3, streams, rgb_in, rgb_out_sharpen, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    main_edge_detect(grid_17_20_3, block_17_20_3, streams, rgb_in, rgb_out_edge_detect, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    main_blur_edge_detect(grid_17_20_3, block_17_20_3, streams, rgb_in, rgb_tmp_blur_edge_detect, rgb_out_blur_edge_detect, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    main_edge_detect_blur(grid_17_20_3, block_17_20_3, streams, rgb_in, rgb_tmp_edge_detect_blur, rgb_out_edge_detect_blur, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    // Recup donnees kernel
    for (std::size_t i = 0; i < taille_stream; ++i) {
        err = cudaMemcpyAsync(g_blur.data() + i * taille_rgb / taille_stream,
                              rgb_out_blur + i * taille_rgb / taille_stream, taille_rgb / taille_stream,
                              cudaMemcpyDeviceToHost, streams[i]);
        if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
        err = cudaMemcpyAsync(g_sharpen.data() + i * taille_rgb / taille_stream,
                              rgb_out_sharpen + i * taille_rgb / taille_stream, taille_rgb / taille_stream,
                              cudaMemcpyDeviceToHost, streams[i]);
        if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
        err = cudaMemcpyAsync(g_edge_detect.data() + i * taille_rgb / taille_stream,
                              rgb_out_edge_detect + i * taille_rgb / taille_stream, taille_rgb / taille_stream,
                              cudaMemcpyDeviceToHost, streams[i]);
        if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

        err = cudaMemcpyAsync(g_blur_edge_detect.data() + i * taille_rgb / taille_stream,
                              rgb_out_blur_edge_detect + i * taille_rgb / taille_stream, taille_rgb / taille_stream,
                              cudaMemcpyDeviceToHost, streams[i]);
        if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
        err = cudaMemcpyAsync(g_edge_detect_blur.data() + i * taille_rgb / taille_stream,
                              rgb_out_edge_detect_blur + i * taille_rgb / taille_stream, taille_rgb / taille_stream,
                              cudaMemcpyDeviceToHost, streams[i]);
        if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    }

    cudaDeviceSynchronize();

    cv::imwrite(out + std::string("_stream_block_17-20-3_blur.") + std::string(argv[2]), m_out_blur);
    cv::imwrite(out + std::string("_stream_block_17-20-3_sharpen.") + std::string(argv[2]), m_out_sharpen);
    cv::imwrite(out + std::string("_stream_block_17-20-3_edge_detect.") + std::string(argv[2]), m_out_edge_detect);

    cv::imwrite(out + std::string("_stream_block_17-20-3_blur_edge_detect.") + std::string(argv[2]), m_out_blur_edge_detect);
    cv::imwrite(out + std::string("_stream_block_17-20-3_edge_detect_blur.") + std::string(argv[2]), m_out_edge_detect_blur);

    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////

    // Nettoyage memoire
    for (std::size_t i = 0; i < taille_stream; ++i ) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(rgb_in);

    cudaFree(rgb_out_blur);
    cudaFree(rgb_out_sharpen);
    cudaFree(rgb_out_edge_detect);

    return 0;
}
