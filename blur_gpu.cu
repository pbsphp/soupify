#include <math.h>
#include <string.h>
#include <stdlib.h>


#include <stdio.h>

extern "C" {
# include "blur.h"
}


#define PI 3.141593


/* Source image texture */
texture<unsigned char, 2> source_tex;


/* Gaussian matrix in GPU memory */
__constant__ float device_matrix[100 * 100];
/* TODO: check limits */


/* Gaussian function */
static inline float G(int x, int y, float sigma)
{
    return (1 / (2 * PI * sigma * sigma)) * \
            exp(-(x * x + y * y) / (2 * sigma * sigma));
}


/* If pixel with x,y exists, "returns" it. Otherwise "returns" black pixel */
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
int gaussian_blur(  const unsigned char *source_image,
                    unsigned char *result_image,
                    size_t width, size_t height, int diameter)
{
    float sigma = diameter / 6.0;

    /* Create gaussian matrix */

    float *gaussian_matrix = (float *) malloc(  diameter * diameter * \
                                                sizeof(float));

    if (gaussian_matrix == NULL) {
        return B_ERR_OUT_OF_MEMORY;
    }


    /* Calculate gaussian matrix */

    int radius = (diameter + 1) / 2;

    for (int y = 0; y < radius; ++y) {
        for (int x = 0; x < radius; ++x) {
            int x_dist = radius - x - 1;
            int y_dist = radius - y - 1;

            float g = G(x_dist, y_dist, sigma);

            int y2 = diameter - y - 1;
            int x2 = diameter - x - 1;

            gaussian_matrix[y * diameter + x] = g;
            gaussian_matrix[y2 * diameter + x] = g;
            gaussian_matrix[y * diameter + x2] = g;
            gaussian_matrix[y2 * diameter + x2] = g;
        }
    }


    /* Calculate gaussian matrix sum */

    float matrix_sum = 0;
    for (int y = 0; y < diameter; ++y) {
        for (int x = 0; x < diameter; ++x) {
            matrix_sum += gaussian_matrix[y * diameter + x];
        }
    }


    /* Apply gaussian smooth on each pixel */

    /* Copy gaussian matrix to device memory */
    cudaMemcpyToSymbol( device_matrix, gaussian_matrix,
                        diameter * diameter * sizeof(float));

    /* Result buffer (GPU) */
    unsigned char *result_dev;
    cudaMalloc((void **) &result_dev, width * height * 4);


    /* Place source image to texture memory */
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
    cudaArray* source_dev;
    cudaMallocArray(&source_dev, &desc, width * 4, height);
    cudaMemcpyToArray(  source_dev, 0, 0, source_image,
                        width * height * 4, cudaMemcpyHostToDevice);

    cudaBindTextureToArray(source_tex, source_dev, desc);


    /* Calculate smoothing for each pixel */

    dim3 blocks(width / 16, height / 16);
    dim3 threads(16, 16);

    apply_pixels<<<blocks, threads>>>(  result_dev, width, height,
                                        diameter, matrix_sum);


    /* Copy result image from GPU buffer to host memory */
    cudaMemcpy(result_image, result_dev,
                width * height * 4, cudaMemcpyDeviceToHost);


    /* Cleanup */

    free(gaussian_matrix);

    cudaFree(result_dev);

    cudaUnbindTexture(source_tex);
    cudaFreeArray(source_dev);

    return 0;
}


}
