#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>


#include "image_io.h"
#include "blur.h"


#define MAX_IMAGE_SIZE (2048 * 2048 * 4)


void print_help()
{
    const char *help_message = \
        "Usage: appname input_file output_file [-q] [-d diameter]\n"
        "Blurs picture with gaussian smoothing.\n"
        "\n"
        "-q, --quiet \t Quiet mode.\n"
        "-d, --diameter \t Gaussian blur diameter.\n";
        "-h, --help \t Print this message.\n";

    printf(help_message);
}



int main(int argc, char *argv[])
{

    /* Parse options */

    char *origin_path = NULL;
    char *blurred_path = NULL;

    int quiet_mode = 0;
    int diameter = 5;

    while (1) {
        const struct option long_options[] = {
            { "quiet", no_argument, 0, 'q' },
            { "help", no_argument, 0, 'h' },
            { "diameter", required_argument, 0, 'd' },
            { 0, 0, 0, 0 }
        };
        int option_index = 0;

        char c = getopt_long(argc, argv, "hqd:",
                        long_options, &option_index);

        if (c == -1) {
            break;
        }

        switch (c) {
        case 0:
            if (long_options[option_index].flag != 0) {
                break;
            }

        case 'h':
            print_help();
            exit(0);

        case 'q':
            quiet_mode = 1;
            break;

        case 'd':
            sscanf(optarg, "%d", &diameter);
            break;

        default:
            /* Error message already printed */
            exit(1);
        }
    }


    /* Get input and output file names */

    if (optind + 2 <= argc) {
        origin_path = argv[optind++];
        blurred_path = argv[optind++];
    } else {
        printf("You must specify input and output files.\n\n");
        print_help();
        exit(1);
    }



    /* Allocate memory for files */

    unsigned char *origin_img = (unsigned char *) malloc(MAX_IMAGE_SIZE);
    unsigned char *blurred_img = (unsigned char *) malloc(MAX_IMAGE_SIZE);

    if (origin_img == NULL || blurred_img == NULL) {
        if (!quiet_mode) {
            fprintf(stderr, "Out of memory!\n");
        }
        exit(1);
    }


    /* Read file */

    struct ImageInfo img;
    int stat = read_image(origin_path, origin_img, &img, MAX_IMAGE_SIZE);

    if (stat == RI_ERR_BADFILE) {
        if (!quiet_mode) {
            fprintf(stderr, "Cannot open %s!\n", origin_path);
        }
        exit(1);
    }

    if (stat == RI_ERR_OUT_OF_MEMORY) {
        if (!quiet_mode) {
            fprintf(stderr, "Out of memory!\n", origin_path);
        }
        exit(1);
    }


    /* Smooth image */

    stat = gaussian_blur(origin_img, blurred_img,
                        img.width, img.height, diameter);

    if (stat == B_ERR_OUT_OF_MEMORY) {
        if (!quiet_mode) {
            fprintf(stderr, "Out of memory!\n", origin_path);
        }
        exit(1);
    }


    /* Write blurred image to output file */

    stat = write_image(blurred_path, img.width, img.height, blurred_img);

    if (stat == RI_ERR_BADFILE) {
        if (!quiet_mode) {
            fprintf(stderr, "Cannot open %s!\n", origin_path);
        }
        exit(1);
    }

    if (stat == RI_ERR_OUT_OF_MEMORY) {
        if (!quiet_mode) {
            fprintf(stderr, "Out of memory!\n", origin_path);
        }
        exit(1);
    }


    /* Cleanup */

    free(origin_img);
    free(blurred_img);

    return 0;
}
