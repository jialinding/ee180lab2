#include <arm_neon.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "sobel_alg.h"
using namespace cv;

/*******************************************
 * Model: grayScale
 * Input: Mat img
 * Output: None directly. Modifies a ref parameter img_gray_out
 * Desc: This module converts the image to grayscale
 ********************************************/
void grayScale(Mat& img, Mat& img_gray_out)
{
  // float color; // converted from double to float
  Mat img_float;
  img_float.create(IMG_HEIGHT, IMG_WIDTH, CV_32F);
  img.convertTo(img_float, CV_32F);
  float32x4_t output, w1, w2, w3, data;
  Mat img_gray_out_float;
  img_gray_out_float.create(IMG_HEIGHT, IMG_WIDTH, CV32F);
  w1 = vdupq_n_f32(.114f);
  w2 = vdupq_n_f32(.587f);
  w3 = vdupq_n_f32(.299f);

  // Convert to grayscale
  for (int i=0; i<img.rows; i++) {
    for (int j=0; j<img.cols; j+=4) {
      output = vdupq_n_f32(0);

      data = vld1q_f32(img_float.data[STEP0*i + STEP1*j]);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j], data, 0);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+1], data, 1);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+2], data, 2);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+3], data, 3);
      output = vmlaq_f32(output, data, w1);

      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+1], data, 0);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+2], data, 1);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+3], data, 2);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+4], data, 3);
      data = vld1q_f32(img_float.data[STEP0*i + STEP1*j + 1]);
      output = vmlaq_f32(output, data, w2);

      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+2], data, 0);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+3], data, 1);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+4], data, 2);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+5], data, 3);
      data = vld1q_f32(img_float.data[STEP0*i + STEP1*j + 2]);
      output = vmlaq_f32(output, data, w3);
      
      vst1q_f32(img_gray_out_float.data[IMG_WIDTH*i + j], output);
      // vst1q_lane_f32((float32_t*)img_gray_out.data[IMG_WIDTH*i + j], output, 0);
      // vst1q_lane_f32((float32_t*)img_gray_out.data[IMG_WIDTH*i + j+1], output, 1);
      // vst1q_lane_f32((float32_t*)img_gray_out.data[IMG_WIDTH*i + j+2], output, 2);
      // vst1q_lane_f32((float32_t*)img_gray_out.data[IMG_WIDTH*i + j+3], output, 3);

      img_gray_out_float.convertTo(img_gray_out, CV_8UC1);

      // color = .114f*img.data[STEP0*i + STEP1*j] +
      //         .587f*img.data[STEP0*i + STEP1*j + 1] +
      //         .299f*img.data[STEP0*i + STEP1*j + 2];
      // img_gray_out.data[IMG_WIDTH*i + j] = color;
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
