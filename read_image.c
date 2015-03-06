
#include <wand/MagickWand.h>

#include "read_image.h"


/*
    This function reads image and places RGBA data to buffer.
    Returns negative on failure.
*/
int read_image(char *image_path, unsigned char *rgba_buffer,
                struct ImageInfo *image_info, size_t buffer_size)
{
    size_t width;
    size_t height;


    /* Open image */
    MagickWandGenesis();
    MagickWand *image_wand = NewMagickWand();
    MagickBooleanType status = MagickReadImage(image_wand, image_path);

    /* If cannot read */
    if (status == MagickFalse) {
        return RI_ERR_BADFILE;
    }

    /* Read pixel by pixel */
    PixelIterator *iterator = NewPixelIterator(image_wand);

    long long bytes_read = 0;

    height = MagickGetImageHeight(image_wand);
    for (int y = 0; y < height; ++y) {
        PixelWand **pixels = PixelGetNextIteratorRow(iterator, &width);

        if ((pixels == NULL))
            return RI_ERR_BADFILE;

        for (int x = 0; x < width; ++x) {
            /* Read (x, y) pixel */
            long long index = y * width + x;

            MagickPixelPacket pixel;
            PixelGetMagickColor(pixels[x], &pixel);

            /* If enough space in the buffer */
            if (bytes_read + 4 < buffer_size) {
                rgba_buffer[4 * index + 0] = pixel.red;
                rgba_buffer[4 * index + 1] = pixel.green;
                rgba_buffer[4 * index + 2] = pixel.blue;
                /* TODO: set real alpha value */
                rgba_buffer[4 * index + 3] = 255;
            } else {
                return RI_ERR_OUT_OF_MEMORY;
            }

            bytes_read += 4;
        }
    }

    /* Set image info */
    image_info->width = width;
    image_info->height = height;

    /* Clear */
    DestroyPixelIterator(iterator);
    DestroyMagickWand(image_wand);
    MagickWandTerminus();

    return 0;
}
