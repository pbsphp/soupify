#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "blur.h"

#define PI 3.141593


/* Gaussian function */
static inline float G(int x, int y, float sigma)
{
    return (1 / (2 * PI * sigma * sigma)) * \
            exp(-(x * x + y * y) / (2 * sigma * sigma));
}


/* If pixel with x,y exists, "returns" it. Otherwise "returns" black pixel */
static void get_pixel_at(int x, int y, int width, int height,
                    const unsigned char *image, int *pixel)
{
    if (0 <= x && x < width && 0 <= y && y < height) {
        for (int i = 0; i < 3; ++i) {
            pixel[i] = image[(y * width + x) * 4 + i];
        }
    } else {
        for (int i = 0; i < 3; ++i) {
            pixel[i] = 0;
        }
    }
}



/*
    Calculates pixel color after applying matrix
*/
static void apply_pixel(int x, int y,
                        const unsigned char *source, unsigned char *result,
                        int width, int height,
                        float **matrix, int diameter, float matrix_sum)
{
    int radius = diameter / 2;

    float sum[3] = { 0.0, 0.0, 0.0 };

    /* Calculate weighted sum of current pixel */
    for (int matrix_y = 0; matrix_y < diameter; ++matrix_y) {
        for (int matrix_x = 0; matrix_x < diameter; ++matrix_x) {
            float weight = matrix[matrix_y][matrix_x];

            int true_y = y + matrix_y - radius;
            int true_x = x + matrix_x - radius;

            int pixel[3] = { 0, 0, 0 };
            get_pixel_at(true_x, true_y, width, height, source, pixel);

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
    result[(y * width + x) * 4 + 3] = source[(y * width + x) * 4 + 3];
}



/*
    Smooth source_image and puts result to result_image
*/
int gaussian_blur(  const unsigned char *source_image,
                    unsigned char *result_image,
                    size_t width, size_t height, int diameter)
{
    float sigma = diameter / 6.0;

    /* Create gaussian matrix */

    float **gaussian_matrix = (float **) malloc(diameter * sizeof(float *));

    if (gaussian_matrix == NULL) {
        return B_ERR_OUT_OF_MEMORY;
    }

    for (int i = 0; i < diameter; ++i) {
        gaussian_matrix[i] = (float *) malloc(diameter * sizeof(float));

        if (gaussian_matrix[i] == NULL) {
            return B_ERR_OUT_OF_MEMORY;
        }
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

            gaussian_matrix[y][x] = g;
            gaussian_matrix[y2][x] = g;
            gaussian_matrix[y][x2] = g;
            gaussian_matrix[y2][x2] = g;
        }
    }


    /* Calculate gaussian matrix sum */

    float matrix_sum = 0;
    for (int y = 0; y < diameter; ++y) {
        for (int x = 0; x < diameter; ++x) {
            matrix_sum += gaussian_matrix[y][x];
        }
    }


    /* Apply gaussian smooth on each pixel */

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            /* Smooth current pixel */
            apply_pixel(x, y, source_image, result_image,
                        width, height,
                        gaussian_matrix, diameter, matrix_sum);
        }
    }


    /* Cleanup */

    for (int i = 0; i < diameter; ++i) {
        free(gaussian_matrix[i]);
    }

    free(gaussian_matrix);

    return 0;
}