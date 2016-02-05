#include "opencv2/imgproc/imgproc.hpp"
#include "sobel_alg.h"
using namespace cv;

/*******************************************
 * Model: grayScale
 * Input: Mat img
 * Output: None directly. Modifies a ref parameter img_gray_out
 * Desc: This module converts the image to grayscale
 ********************************************/
void grayScale(Mat& __restrict img, Mat& __restrict img_gray_out)
{
  for (int i=0; i<img.rows; i++) {
    for (int j=0; j<(img.cols & ~3); j += 4) {
			
			/*
			float r[4];
			for (int k = 0; k < 4; k++)
				r[k] = img.data[STEP0*i + STEP1*(j + k) + 0];
			
			float g[4];
			for (int k = 0; k < 4; k++)
				g[k] = img.data[STEP0*i + STEP1*(j + k) + 1];
			
			float b[4];
			for (int k = 0; k < 4; k++)
				b[k] = img.data[STEP0*i + STEP1*(j + k) + 2];
			
			float c[4];
			for (int k = 0; k < 4; k++)
				c[k] += r[k];
			for (int k = 0; k < 4; k++)
				c[k] += g[k];
			for (int k = 0; k < 4; k++)
				c[k] += b[k];
			
			for (int k = 0; k < 4; k++)
				img_gray_out.data[IMG_WIDTH*i + (j + k)] += c[k];
			*/
			
			/*
			a1 += .114f*img.data[STEP0*i + STEP1*(j + 0)];
			a2 += .114f*img.data[STEP0*i + STEP1*(j + 1)];
			a3 += .114f*img.data[STEP0*i + STEP1*(j + 2)];
			a4 += .114f*img.data[STEP0*i + STEP1*(j + 3)];
			
			a1 += .587f*img.data[STEP0*i + STEP1*(j + 0) + 1];
			a2 += .587f*img.data[STEP0*i + STEP1*(j + 1) + 1];
			a3 += .587f*img.data[STEP0*i + STEP1*(j + 2) + 1];
			a4 += .587f*img.data[STEP0*i + STEP1*(j + 3) + 1];
			
			a1 += .299f*img.data[STEP0*i + STEP1*(j + 0) + 2];
			a2 += .299f*img.data[STEP0*i + STEP1*(j + 1) + 2];
			a3 += .299f*img.data[STEP0*i + STEP1*(j + 2) + 2];
			a4 += .299f*img.data[STEP0*i + STEP1*(j + 3) + 2];
			
			
			img_gray_out.data[IMG_WIDTH*i + (j + 0)] = a;
			img_gray_out.data[IMG_WIDTH*i + (j + 1)] = a2;
			img_gray_out.data[IMG_WIDTH*i + (j + 2)] = a3;
			img_gray_out.data[IMG_WIDTH*i + (j + 3)] = a4;
			*/
			
			/*
      color = .114f*img.data[STEP0*i + STEP1*j] +
              .587f*img.data[STEP0*i + STEP1*j + 1] +
              .299f*img.data[STEP0*i + STEP1*j + 2];

      img_gray_out.data[IMG_WIDTH*i + j] = color;
			*/
    }
  }
}

/*******************************************
 * Model: sobelCalc
 * Input: Mat img_in
 * Output: None directly. Modifies a ref parameter img_sobel_out
 * Desc: This module performs a sobel calculation on an image. It first
 *  converts the image to grayscale, calculates the gradient in the x
 *  direction, calculates the gradient in the y direction and sum it with Gx
 *  to finish the Sobel calculation
 ********************************************/
void sobelCalc(Mat& img_gray, Mat& img_sobel_out, int side)
{
  Mat img_outx = img_gray.clone();
  Mat img_outy = img_gray.clone();

  // Apply Sobel filter to black & white image
  unsigned short sobel;

  int col_begin, col_end;
  if (side == 0) { // all
    col_begin = 1;
    col_end = img_gray.cols-1;
  } else if (side == 1) { // left
    col_begin = 1;
    col_end = IMG_WIDTH/2;
  } else { // right
    col_begin = IMG_WIDTH/2;
    col_end = img_gray.cols-1;
  }

  // Calculate the x convolution
  for (int i=1; i<img_gray.rows-1; i++) {
    for (int j=col_begin; j<col_end; j++) {
      sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j-1)] +
		  2*img_gray.data[IMG_WIDTH*(i-1) + (j)] -
		  2*img_gray.data[IMG_WIDTH*(i+1) + (j)] +
		  img_gray.data[IMG_WIDTH*(i-1) + (j+1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

      sobel = (sobel > 255) ? 255 : sobel;
      img_outx.data[IMG_WIDTH*(i) + (j)] = sobel;
    }
  }

  // Calc the y convolution
  for (int i=1; i<img_gray.rows-1; i++) {
    for (int j=col_begin; j<col_end; j++) {
     sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i-1) + (j+1)] +
		   2*img_gray.data[IMG_WIDTH*(i) + (j-1)] -
		   2*img_gray.data[IMG_WIDTH*(i) + (j+1)] +
		   img_gray.data[IMG_WIDTH*(i+1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

     sobel = (sobel > 255) ? 255 : sobel;

     img_outy.data[IMG_WIDTH*(i) + j] = sobel;
    }
  }

  // Combine the two convolutions into the output image
  for (int i=1; i<img_gray.rows-1; i++) {
    for (int j=col_begin; j<col_end; j++) {
      sobel = img_outx.data[IMG_WIDTH*(i) + j] +
	img_outy.data[IMG_WIDTH*(i) + j];
      sobel = (sobel > 255) ? 255 : sobel;
      img_sobel_out.data[IMG_WIDTH*(i) + j] = sobel;
    }
  }
}
