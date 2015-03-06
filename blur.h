
#ifndef _BLUR_H
# define _BLUR_H

#define B_ERR_OUT_OF_MEMORY -1


/*
    Smooth source_image and puts result to result_image
*/
int gaussian_blur(  const unsigned char *source_image,
                    unsigned char *result_image,
                    size_t width, size_t height, int diameter);


#endif
