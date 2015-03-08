#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "blur.h"
#include "gaussian_matrix.h"

#define PI 3.141593


/* If pixel with x,y exists, "returns" it. Otherwise "returns" closest */
static void get_pixel_at(int x, int y, int width, int height,
                    const unsigned char *image, int *pixel)
{
    if (x < 0) {
        x = 0;
    }
    else if (x >= width) {
        x = width - 1;
    }
    if (y < 0) {
        y = 0;
    }
    else if (y >= width) {
        y = height - 1;
    }


    for (int i = 0; i < 3; ++i) {
        pixel[i] = image[(y * width + x) * 4 + i];
    }
}



/*
    Calculates pixel color after applying matrix
*/
static void apply_pixel(int x, int y,
                        const unsigned char *source, unsigned char *result,
                        int width, int height,
                        float *matrix, int diameter, float matrix_sum)
{
    int radius = diameter / 2;

    float sum[3] = { 0.0, 0.0, 0.0 };

    /* Calculate weighted sum of current pixel */
    for (int matrix_y = 0; matrix_y < diameter; ++matrix_y) {
        for (int matrix_x = 0; matrix_x < diameter; ++matrix_x) {
            float weight = matrix[matrix_y * diameter + matrix_x];

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

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            /* Smooth current pixel */
            apply_pixel(x, y, source_image, result_image,
                        width, height,
                        gaussian_matrix, diameter, matrix_sum);
        }
    }


    /* Cleanup */

    free(gaussian_matrix);

    return 0;
}
