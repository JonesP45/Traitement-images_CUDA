#include <opencv2/opencv.hpp>
#include <vector>

__global__ void blur(const unsigned char* rgb_in, unsigned char* rgb_out, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
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

            rgb_out[3 * (row * cols + col) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
        }
    }

//    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
//    {
//        //p1 à p9 correspondent aux 9 pixels à récupérer
//        unsigned char p1 = mat_in[(row - 3) * cols + col - 3];
//        unsigned char p2 = mat_in[(row - 3) * cols + col];
//        unsigned char p3 = mat_in[(row - 3) * cols + col + 3];
//        unsigned char p4 = mat_in[row * cols + col - 3];
//        unsigned char p5 = mat_in[row * cols + col];
//        unsigned char p6 = mat_in[row * cols + col + 3];
//        unsigned char p7 = mat_in[(row + 3) * cols + col - 3];
//        unsigned char p8 = mat_in[(row + 3) * cols + col];
//        unsigned char p9 = mat_in[(row + 3) * cols + col + 3];
//
//        mat_out[row * cols + col] = (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / 9;
//}
}

void main_blur(const dim3 grid, const dim3 block, const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    blur <<< grid, block >>> (rgb_in, rgb_out_blur, rows, cols);

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "blur_kernel: " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main()
{
    // Declarations
    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    unsigned char* rgb = m_in.data;
    int rows = m_in.rows;
    int cols = m_in.cols;

    size_t taille_rgb = 3 * rows * cols;
    std::vector<unsigned char> g_blur(taille_rgb);
    cv::Mat m_out_blur(rows, cols, CV_8UC3, g_blur.data());

    unsigned char* rgb_in;
    unsigned char* rgb_out_blur;

    // Init donnes kernel
    cudaMalloc(&rgb_in, taille_rgb);
    cudaMalloc(&rgb_out_blur, taille_rgb);
    cudaMemcpy(rgb_in, rgb, taille_rgb, cudaMemcpyHostToDevice);

    dim3 block(32, 32); //nb de thread, max 1024
    dim3 grid(((cols - 1) / block.x + 1), (rows - 1) / block.y + 1);

    /*// Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    blur <<< grid, block >>> (rgb_in, rgb_out_blur, cols, rows);

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cout << cudaGetErrorString(cudaGetLastError()) << endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << elapsedTime << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);*/
    // Execution
    main_blur(grid, block, rgb_in, rgb_out_blur, rows, cols);

    // Recup donnees kernel
    cudaMemcpy(g_blur.data(), rgb_out_blur, 3 * rows * cols, cudaMemcpyDeviceToHost);
    cv::imwrite("out_kernel_blur.jpg", m_out_blur);

    // Nettoyage memoire
    cudaFree(rgb_in);
    cudaFree(rgb_out_blur);

    return 0;
}
