#include <opencv2/opencv.hpp>
#include <vector>

std::size_t taille_rgb = 0;
std::size_t taille_stream = 0;
std::size_t one_line_rgb = 0;


__global__ void blur(const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols) {
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

__global__ void sharpen(const unsigned char* rgb_in, unsigned char* rgb_out_sharpen, int rows, int cols) {
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

__global__ void edge_detect(const unsigned char* rgb_in, unsigned char* rgb_out_edge_detect, int rows, int cols) {
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


void main_blur(const dim3 grid, const dim3 block, const cudaStream_t* streams, const unsigned char* rgb_in,
        unsigned char* rgb_out_blur, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    for (std::size_t i = 0; i < taille_stream; ++i) {
        int row_stream = (int) (rows / taille_stream) + ((i == 0 || i == taille_stream - 1) ? 1 : 2);
        std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
        blur<<< grid, block, 0, streams[i] >>>(rgb_in + decalage,rgb_out_blur + decalage,
                row_stream, cols);
    }

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "blur_stream: " << elapsedTime << std::endl;
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
    for (std::size_t i = 0; i < taille_stream; ++i) {
        int row_stream = (int) (rows / taille_stream) + ((i == 0 || i == taille_stream - 1) ? 1 : 2);
        std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
        sharpen<<< grid, block, 0, streams[i] >>>(rgb_in + decalage, rgb_out_sharpen + decalage,
                                               row_stream, cols);
    }

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "sharpen_stream: " << elapsedTime << std::endl;
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
    for (std::size_t i = 0; i < taille_stream; ++i) {
        int row_stream = (int) (rows / taille_stream) + ((i == 0 || i == taille_stream - 1) ? 1 : 2);
        std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
        edge_detect<<< grid, block, 0, streams[i] >>>(rgb_in + decalage, rgb_out_edge_detect + decalage,
                                               row_stream, cols);
    }

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "edge_detect_stream: " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main()
{
    // Declarations
    cudaError_t err;

    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    unsigned char* rgb = m_in.data;
    int rows = m_in.rows;
    int cols = m_in.cols;

    taille_rgb = 3 * rows * cols;

    std::vector<unsigned char> g_blur(taille_rgb);
    std::vector<unsigned char> g_sharpen(taille_rgb);
    std::vector<unsigned char> g_edge_detect(taille_rgb);
    cv::Mat m_out_blur(rows, cols, CV_8UC3, g_blur.data());
    cv::Mat m_out_sharpen(rows, cols, CV_8UC3, g_sharpen.data());
    cv::Mat m_out_edge_detect(rows, cols, CV_8UC3, g_edge_detect.data());

    unsigned char* rgb_in = nullptr;
    unsigned char* rgb_out_blur = nullptr;
    unsigned char* rgb_out_sharpen;
    unsigned char* rgb_out_edge_detect;

    // Init donnes kernel
    err = cudaMalloc(&rgb_in, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_out_blur, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_out_sharpen, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_out_edge_detect, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    taille_stream = 2;
    cudaStream_t streams[taille_stream];
    for (std::size_t i = 0; i < taille_stream; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    one_line_rgb = cols * 3;

    for (std::size_t i = 0; i < taille_stream; ++i) {
        std::size_t decalage = i * taille_rgb / taille_stream - (i == 0 ? 0 : one_line_rgb);
        std::size_t count = taille_rgb / taille_stream + ((i == 0 || i == taille_stream - 1) ? one_line_rgb : 2 * one_line_rgb);
        err = cudaMemcpyAsync(rgb_in + decalage,rgb + decalage, count, cudaMemcpyHostToDevice, streams[i]);
        if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    }

    dim3 block(32, 32 / taille_stream); //nb de thread par bloc, max 1024
    dim3 grid(((cols - 1) / block.x + 1), (((rows + (taille_stream - 1) * 2) / taille_stream - 1) / block.y + 1)); // nb de block

    // Execution
    main_blur(grid, block, streams, rgb_in, rgb_out_blur, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    main_sharpen(grid, block, streams, rgb_in, rgb_out_sharpen, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    main_edge_detect(grid, block, streams, rgb_in, rgb_out_edge_detect, rows, cols);
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
    }

    cudaDeviceSynchronize();
    for (std::size_t i = 0; i < taille_stream; ++i ) {
        cudaStreamDestroy(streams[i]);
    }

    cv::imwrite("out_stream_blur.jpg", m_out_blur);
    cv::imwrite("out_stream_sharpen.jpg", m_out_sharpen);
    cv::imwrite("out_stream_edge_detect.jpg", m_out_edge_detect);

    // Nettoyage memoire
    cudaFree(rgb_in);
    cudaFree(rgb_out_blur);
    cudaFree(rgb_out_sharpen);
    cudaFree(rgb_out_edge_detect);

    return 0;
}
