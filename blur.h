
#ifndef _BLUR_H
# define _BLUR_H

#define B_ERR_OUT_OF_MEMORY -1
#define B_ERR_CUDA_ERROR -2


/*
    Smooth source_image and puts result to result_image
*/
int gaussian_blur(  unsigned char *image_buffer,
                    size_t width, size_t height, int diameter);


#endif
