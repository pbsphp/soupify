
#ifndef _APP_LIMITS_H
# define _APP_LIMITS_H


#define MAX_IMAGE_SIZE (3000 * 3000 * 4)
#define MAX_BLUR_DIAMETER 128
#define MAX_GAUSSIAN_MATRIX_SIZE (MAX_BLUR_DIAMETER * MAX_BLUR_DIAMETER)

/*
    Smooth source_image and puts result to result_image
*/
int gaussian_blur(  unsigned char *image_buffer,
                    size_t width, size_t height, int diameter);


#endif
