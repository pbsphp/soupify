
#ifndef _READ_IMAGE_H
#define _READ_IMAGE_H


# define RI_ERR_BADFILE -1
# define RI_ERR_OUT_OF_MEMORY -2


struct ImageInfo
{
    size_t width;
    size_t height;
};



/*
    This function reads image and places RGBA data to buffer.
    Returns negative on failure.
*/
int read_image(char *image_path, unsigned char *rgba_buffer,
                struct ImageInfo *image_info, size_t buffer_size);


#endif
