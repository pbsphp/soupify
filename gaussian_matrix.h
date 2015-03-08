
#ifndef _GAUSSIAN_MATRIX_H
# define _GAUSSIAN_MATRIX_H


#define PI 3.141593


/*
    Gaussian function
*/
static inline float G(int x, int y, float sigma)
{
    return (1 / (2 * PI * sigma * sigma)) * \
            exp(-(x * x + y * y) / (2 * sigma * sigma));
}



/*
    Calculate gaussian matrix
*/
static void calculate_gaussian_matrix(  float *matrix, int diameter,
                                        float *matrix_sum)
{
    float sigma = diameter / 6.0;

    int radius = (diameter + 1) / 2;

    /* Calculate each element of matrix */

    for (int y = 0; y < radius; ++y) {
        for (int x = 0; x < radius; ++x) {
            int x_dist = radius - x - 1;
            int y_dist = radius - y - 1;

            float g = G(x_dist, y_dist, sigma);

            int y2 = diameter - y - 1;
            int x2 = diameter - x - 1;

            matrix[y * diameter + x] = g;
            matrix[y2 * diameter + x] = g;
            matrix[y * diameter + x2] = g;
            matrix[y2 * diameter + x2] = g;
        }
    }


    /* Calculate gaussian matrix sum */

    *matrix_sum = 0;
    for (int y = 0; y < diameter; ++y) {
        for (int x = 0; x < diameter; ++x) {
            *matrix_sum += matrix[y * diameter + x];
        }
    }
}



#endif
