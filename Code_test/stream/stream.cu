#include <opencv2/opencv.hpp>
#include <vector>

__global__ void blur(const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
//    if (col >= 1 && col < cols - 1)
    {
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


void main_blur(const dim3 nbBlock, const dim3 threadsPerBlock, const cudaStream_t* streams, std::size_t taille_stream, int taille_rgb, const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    for (std::size_t i = 0; i < taille_stream; ++i) {
//        blur<<< nbBlock, threadsPerBlock, 0, streams[i] >>>(rgb_in + i * taille_rgb / taille_stream,
//                rgb_out_blur + i * taille_rgb / taille_stream, rows, (int) (cols / taille_stream));
        blur<<< nbBlock, threadsPerBlock, 0, streams[i] >>>(rgb_in + i * taille_rgb / taille_stream,
                rgb_out_blur + i * taille_rgb / taille_stream, /*rows*/(int) (rows / taille_stream) +
                (i != taille_stream - 1 ? 1 : 0), cols);
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


int main()
{
    // Declarations
    cudaError_t err;

    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    unsigned char* rgb = m_in.data;
    int rows = m_in.rows;
    int cols = m_in.cols;

    size_t taille_rgb = 3 * rows * cols;

    std::vector<unsigned char> g_blur(taille_rgb);
    cv::Mat m_out_blur(rows, cols, CV_8UC3, g_blur.data());

    unsigned char* rgb_in = nullptr;
    unsigned char* rgb_out_blur = nullptr;

    // Init donnes kernel
    err = cudaMalloc(&rgb_in, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_out_blur, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    std::size_t taille_stream = 2;
    cudaStream_t streams[taille_stream];
    for (std::size_t i = 0; i < taille_stream; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    for (std::size_t i = 0; i < taille_stream; ++i) {
        cudaMemcpyAsync(rgb_in + i * taille_rgb / taille_stream,rgb + i * taille_rgb / taille_stream,
                taille_rgb / taille_stream,cudaMemcpyHostToDevice, streams[i]);
    }

    dim3 threadsPerBlock(32, 32 / taille_stream); //nb de thread, max 1024
    dim3 nbBlock(((cols - 1) / threadsPerBlock.x + 1), ((rows / taille_stream - 1) / threadsPerBlock.y + 1));

    // Execution
    main_blur(nbBlock, threadsPerBlock, streams, taille_stream, taille_rgb, rgb_in, rgb_out_blur, rows, cols);

    // Recup donnees kernel
    for (std::size_t i = 0; i < taille_stream; ++i) {
        cudaMemcpyAsync(g_blur.data() + i * taille_rgb / taille_stream,
                        rgb_out_blur + i * taille_rgb / taille_stream, taille_rgb / taille_stream,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();
    for (std::size_t i = 0; i < taille_stream; ++i ) {
        cudaStreamDestroy(streams[i]);
    }

    cv::imwrite("out_stream_blur.jpg", m_out_blur);

    // Nettoyage memoire
    cudaFree(rgb_in);
    cudaFree(rgb_out_blur);

    return 0;
}
