#include <math.h>
#include <string.h>
#include <stdlib.h>


#include <stdio.h>

extern "C" {
# include "blur.h"
}

#include "gaussian_matrix.h"


#define PI 3.141593


/* Source image texture */
texture<unsigned char, 2> source_tex;


/* Gaussian matrix in GPU memory */
__constant__ float device_matrix[100 * 100];
/* TODO: check limits */



/*
    If pixel with x,y exists, "returns" it. Otherwise "returns" black pixel
*/
__device__ void get_pixel_at(int x, int y, int *pixel)
{
    for (int i = 0; i < 3; ++i) {
        pixel[i] = tex2D(source_tex, (x * 4 + i), y);
    }
}



/*
    Calculates pixel color after applying matrix
*/
__global__ void apply_pixels(   unsigned char *result,
                                size_t width, size_t height,
                                int diameter, float matrix_sum)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x >= width || y >= height) {
        return;
    }


    /* Calculate weighted sum of current pixel */
    float sum[3] = { 0.0, 0.0, 0.0 };
    int radius = diameter / 2;

    for (int matrix_y = 0; matrix_y < diameter; ++matrix_y) {
        for (int matrix_x = 0; matrix_x < diameter; ++matrix_x) {
            float weight = device_matrix[matrix_y * diameter + matrix_x];

            int true_y = y + matrix_y - radius;
            int true_x = x + matrix_x - radius;

            int pixel[3] = { 0, 0, 0 };
            get_pixel_at(true_x, true_y, pixel);

            for (int i = 0; i < 3; ++i) {
                sum[i] += pixel[i] * weight;
            }
        }
    }

    /* Write result to image */
    for (int i = 0; i < 3; ++i) {
        result[(y * width + x) * 4 + i] = (unsigned char)(sum[i] / matrix_sum);
    }

    /* Alpha channel */
    result[(y * width + x) * 4 + 3] = tex2D(source_tex, (x * 4 + 3), y);
}



extern "C" {

/*
    Smooth source_image and puts result to result_image
*/
int gaussian_blur(  unsigned char *image_buffer,
                    size_t width, size_t height, int diameter)
{
    cudaError errcode;

    /* Create gaussian matrix */

    float *gaussian_matrix = (float *) malloc(  diameter * diameter * \
                                                sizeof(float));

    if (gaussian_matrix == NULL) {
        return B_ERR_OUT_OF_MEMORY;
    }


    /* Calculate gaussian matrix */

    float matrix_sum;
    calculate_gaussian_matrix(gaussian_matrix, diameter, &matrix_sum);


    /* Apply gaussian smooth on each pixel */

    /* Copy gaussian matrix to device memory */
    errcode = cudaMemcpyToSymbol(   device_matrix, gaussian_matrix,
                                    diameter * diameter * sizeof(float));

    if (errcode != cudaSuccess) {
        return B_ERR_CUDA_ERROR;
    }


    /* Result buffer (GPU) */
    unsigned char *result_dev;
    errcode = cudaMalloc((void **) &result_dev, width * height * 4);

    if (errcode != cudaSuccess) {
        return B_ERR_CUDA_ERROR;
    }


    /* Place source image to texture memory */

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
    cudaArray* source_dev;
    errcode = cudaMallocArray(&source_dev, &desc, width * 4, height);

    if (errcode != cudaSuccess) {
        return B_ERR_CUDA_ERROR;
    }

    errcode = cudaMemcpyToArray(source_dev, 0, 0, image_buffer,
                                width * height * 4, cudaMemcpyHostToDevice);

    if (errcode != cudaSuccess) {
        return B_ERR_CUDA_ERROR;
    }

    errcode = cudaBindTextureToArray(source_tex, source_dev, desc);

    if (errcode != cudaSuccess) {
        return B_ERR_CUDA_ERROR;
    }


    /* Calculate smoothing for each pixel */

    dim3 blocks(width / 16, height / 16);
    dim3 threads(16, 16);

    apply_pixels<<<blocks, threads>>>(  result_dev, width, height,
                                        diameter, matrix_sum);


    /* Copy result image from GPU buffer to host memory */
    errcode = cudaMemcpy(   image_buffer, result_dev,
                            width * height * 4, cudaMemcpyDeviceToHost);

    if (errcode != cudaSuccess) {
        return B_ERR_CUDA_ERROR;
    }


    /* Cleanup */

    free(gaussian_matrix);

    cudaFree(result_dev);

    cudaUnbindTexture(source_tex);
    cudaFreeArray(source_dev);

    return 0;
}


}
