#include <opencv2/opencv.hpp>
#include <vector>

__global__ void blur(const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
        for (std::size_t rgb = 0; rgb < 3; ++rgb) {
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


void main_blur(const dim3 grid, const dim3 block, const unsigned char* rgb_in,
        unsigned char* rgb_tmp_blur_edge_detect, unsigned char* rgb_out_blur_edge_detect, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    blur <<< grid, block >>> (rgb_in, rgb_out_blur_edge_detect, rows, cols);
    edge_detect <<< grid, block >>> (rgb_tmp_blur_edge_detect, rgb_out_blur_edge_detect, rows, cols);

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "blur_edge_detect_kernel_comb: " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void main_edge_detect(const dim3 grid, const dim3 block, const unsigned char* rgb_in,
        unsigned char* rgb_tmp_edge_detect_blur, unsigned char* rgb_out_edge_detect_blur, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    edge_detect <<< grid, block >>> (rgb_in, rgb_tmp_edge_detect_blur, rows, cols);
    blur <<< grid, block >>> (rgb_tmp_edge_detect_blur, rgb_out_edge_detect_blur, rows, cols);

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "edge_detect_blur_kernel_comb: " << elapsedTime << std::endl;
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
    std::vector<unsigned char> g_blur_edge_detect(taille_rgb);
    std::vector<unsigned char> g_edge_detect_blur(taille_rgb);
    cv::Mat m_out_blur(rows, cols, CV_8UC3, g_blur_edge_detect.data());
    cv::Mat m_out_edge_detect(rows, cols, CV_8UC3, g_edge_detect_blur.data());

    unsigned char* rgb_in;

    unsigned char* rgb_tmp_blur_edge_detect;
    unsigned char* rgb_tmp_edge_detect_blur;

    unsigned char* rgb_out_blur_edge_detect;
    unsigned char* rgb_out_edge_detect_blur;

    // Init donnes kernel
    err = cudaMalloc(&rgb_in, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_tmp_blur_edge_detect, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_tmp_edge_detect_blur, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_out_blur_edge_detect, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_out_edge_detect_blur, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    cudaMemcpy(rgb_in, rgb, taille_rgb, cudaMemcpyHostToDevice);

    dim3 block(32, 32); //nb de thread, max 1024
    dim3 grid(((cols - 1) / block.x + 1), (rows - 1) / block.y + 1);

    // Execution
    main_blur(grid, block, rgb_in, rgb_tmp_blur_edge_detect, rgb_out_blur_edge_detect, rows, cols);
    main_edge_detect(grid, block, rgb_in, rgb_tmp_edge_detect_blur, rgb_out_edge_detect_blur, rows, cols);

    // Recup donnees kernel
    cudaMemcpy(g_blur_edge_detect.data(), rgb_out_blur_edge_detect, taille_rgb, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_edge_detect_blur.data(), rgb_out_edge_detect_blur, taille_rgb, cudaMemcpyDeviceToHost);
    cv::imwrite("out_kernel_comb_blur_edge_detect.jpg", m_out_blur);
    cv::imwrite("out_kernel_comb_edge_detect_blur.jpg", m_out_edge_detect);

    // Nettoyage memoire
    cudaFree(rgb_in);
    cudaFree(rgb_tmp_blur_edge_detect);
    cudaFree(rgb_tmp_edge_detect_blur);
    cudaFree(rgb_out_blur_edge_detect);
    cudaFree(rgb_out_edge_detect_blur);

    return 0;
}
