#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

__global__ void blur(unsigned char* mat_in, unsigned char* mat_out, std::size_t cols, std::size_t rows) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto j = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    if (j >= 3 && j < (rows - 1) * 3 && i >= 1 && i < cols - 1)
    {
        //p1 à p9 correspondent aux 9 pixels à récupérer
        unsigned char p1 = mat_in[(j - 3) * cols + i - 3];
        unsigned char p2 = mat_in[(j - 3) * cols + i];
        unsigned char p3 = mat_in[(j - 3) * cols + i + 3];
        unsigned char p4 = mat_in[j * cols + i - 3];
        unsigned char p5 = mat_in[j * cols + i];
        unsigned char p6 = mat_in[j * cols + i + 3];
        unsigned char p7 = mat_in[(j + 3) * cols + i - 3];
        unsigned char p8 = mat_in[(j + 3) * cols + i];
        unsigned char p9 = mat_in[(j + 3) * cols + i + 3];

        mat_out[j * cols + i] = (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / 9;
    }
}

int main()
{
    //Declarations
    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    unsigned char* rgb = m_in.data;
    int rows = m_in.rows;
    int cols = m_in.cols;

    vector<unsigned char> g(rows * cols * 3); //Pour recreer l'image
    cv::Mat m_out(rows, cols, CV_8UC3, g.data());
    unsigned char* mat_in;
    unsigned char* mat_out;

    //Init donnes kernel
    cudaMalloc(&mat_in, 3 * rows * cols);
    cudaMalloc(&mat_out, 3 * rows * cols);
    cudaMemcpy(mat_in, rgb, 3 * rows * cols, cudaMemcpyHostToDevice);

    dim3 block(32, 32); //nb de thread, max 1024
    dim3 grid(((cols - 1) / block.x + 1), 3 * ((rows - 1) / block.y + 1));

    //Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //Appel kernel
    blur <<< grid, block >>> (mat_in, mat_out, cols, rows);

    //Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cout << cudaGetErrorString(cudaGetLastError()) << endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << elapsedTime << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //Recup donnees kernel
    cudaMemcpy(g.data(), mat_out, 3 * rows * cols, cudaMemcpyDeviceToHost);

    cv::imwrite("out.jpg", m_out);

    cudaFree(mat_in);
    cudaFree(mat_out);

    return 0;
}
