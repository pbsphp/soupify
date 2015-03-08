
#include <wand/MagickWand.h>

#include "image_io.h"


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

    size_t bytes_read = 0;

    height = MagickGetImageHeight(image_wand);
    for (int y = 0; y < height; ++y) {
        PixelWand **pixels = PixelGetNextIteratorRow(iterator, &width);

        if ((pixels == NULL))
            return RI_ERR_BADFILE;

        for (int x = 0; x < width; ++x) {
            /* Read (x, y) pixel */
            size_t index = y * width + x;

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



/*
    This function writes RGBA data to image.
    Returns negative on failure.
*/
int write_image(char *image_path,
                size_t width, size_t height, const unsigned char *rgba_data)
{
    /* Open image */
    MagickWandGenesis();
    MagickWand *image_wand = NewMagickWand();

    PixelWand *bg = NewPixelWand();
    MagickBooleanType status = MagickNewImage(image_wand, width, height, bg);

    if (status == MagickFalse) {
        return RI_ERR_OUT_OF_MEMORY;
    }

    /* Write pixel by pixel */
    PixelIterator *iterator = NewPixelIterator(image_wand);

    if (iterator == NULL) {
        return RI_ERR_OUT_OF_MEMORY;
    }

    /* Generate image from RGB */
    for (int y = 0; y < height; ++y) {
        PixelWand **pixels = PixelGetNextIteratorRow(iterator, &width);

        for (int x = 0; x < width; ++x) {
            /* Write (x, y) pixel */
            size_t index = y * width + x;

            double red = rgba_data[4 * index + 0] / 255.0;
            double green = rgba_data[4 * index + 1] / 255.0;
            double blue = rgba_data[4 * index + 2] / 255.0;
            double opacity = rgba_data[4 * index + 3] / 255.0;

            PixelSetRed(pixels[x], red);
            PixelSetGreen(pixels[x], green);
            PixelSetBlue(pixels[x], blue);
            PixelSetOpacity(pixels[x], opacity);
        }

        PixelSyncIterator(iterator);
    }


    /* Create file */
    status = MagickWriteImage(image_wand, image_path);
    if (status == MagickFalse) {
        return RI_ERR_BADFILE;
    }


    /* Clear */
    DestroyPixelIterator(iterator);
    DestroyMagickWand(image_wand);
    DestroyPixelWand(bg);
    MagickWandTerminus();

    return 0;
}

