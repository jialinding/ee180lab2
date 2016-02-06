#include <arm_neon.h>
#include "opencv2/imgproc/imgproc.hpp"
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
  float32x4_t output, w1, w2, w3, data;
  w1 = vdupq_n_f32(.114f);
  w2 = vdupq_n_f32(.587f);
  w3 = vdupq_n_f32(.299f);

  float32_t img_float[4];
  float32_t img_gray_out_float[4];

  // Convert to grayscale
  for (int i=0; i<img.rows; i++) {
    for (int j=0; j<img.cols; j+=4) {
      output = vdupq_n_f32(0);

      img_float[0] = (float)img.data[STEP0*i + STEP1*j];
      img_float[1] = (float)img.data[STEP0*i + STEP1*(j+1)];
      img_float[2] = (float)img.data[STEP0*i + STEP1*(j+2)];
      img_float[3] = (float)img.data[STEP0*i + STEP1*(j+3)];
      data = vld1q_f32(&img_float[0]);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j], data, 0);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+1], data, 1);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+2], data, 2);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+3], data, 3);
      output = vmlaq_f32(output, data, w1);

      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+1], data, 0);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+2], data, 1);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+3], data, 2);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+4], data, 3);
      img_float[0] = (float)img.data[STEP0*i + STEP1*j + 1];
      img_float[1] = (float)img.data[STEP0*i + STEP1*(j+1) + 1];
      img_float[2] = (float)img.data[STEP0*i + STEP1*(j+2) + 1];
      img_float[3] = (float)img.data[STEP0*i + STEP1*(j+3) + 1];
      data = vld1q_f32(&img_float[0]);
      output = vmlaq_f32(output, data, w2);

      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+2], data, 0);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+3], data, 1);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+4], data, 2);
      // data = vld1q_lane_f32((float32_t*)img.data[STEP0*i + STEP1*j+5], data, 3);
      img_float[0] = (float)img.data[STEP0*i + STEP1*j + 2];
      img_float[1] = (float)img.data[STEP0*i + STEP1*(j+1) + 2];
      img_float[2] = (float)img.data[STEP0*i + STEP1*(j+2) + 2];
      img_float[3] = (float)img.data[STEP0*i + STEP1*(j+3) + 2];
      data = vld1q_f32(&img_float[0]);
      output = vmlaq_f32(output, data, w3);
      
      vst1q_f32(img_gray_out_float, output);

      img_gray_out.data[IMG_WIDTH*i + j] = (unsigned char)img_gray_out_float[0];
      img_gray_out.data[IMG_WIDTH*i + j + 1] = (unsigned char)img_gray_out_float[1];
      img_gray_out.data[IMG_WIDTH*i + j + 2] = (unsigned char)img_gray_out_float[2];
      img_gray_out.data[IMG_WIDTH*i + j + 3] = (unsigned char)img_gray_out_float[3];
      // vst1q_lane_f32((float32_t*)img_gray_out.data[IMG_WIDTH*i + j], output, 0);
      // vst1q_lane_f32((float32_t*)img_gray_out.data[IMG_WIDTH*i + j+1], output, 1);
      // vst1q_lane_f32((float32_t*)img_gray_out.data[IMG_WIDTH*i + j+2], output, 2);
      // vst1q_lane_f32((float32_t*)img_gray_out.data[IMG_WIDTH*i + j+3], output, 3);

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

  int16x8_t img_gray_data, sobel_out, two;
  // Calculate the x convolution
  for (int i=1; i<img_gray.rows-1; i++) {
    for (int j=col_begin; j<col_end; j+=8) {
      sobel_out = vdupq_n_s16(0);
      two = vdupq_n_s16(2);
      int16_t img_gray_16[10];

      // img_gray.data[IMG_WIDTH*(i-1) + (j-1)]
      img_gray_16[0] = img_gray.data[IMG_WIDTH*(i-1) + (j-1)];
      img_gray_16[1] = img_gray.data[IMG_WIDTH*(i-1) + (j-1) + 1];
      img_gray_16[2] = img_gray.data[IMG_WIDTH*(i-1) + (j-1) + 2];
      img_gray_16[3] = img_gray.data[IMG_WIDTH*(i-1) + (j-1) + 3];
      img_gray_16[4] = img_gray.data[IMG_WIDTH*(i-1) + (j-1) + 4];
      img_gray_16[5] = img_gray.data[IMG_WIDTH*(i-1) + (j-1) + 5];
      img_gray_16[6] = img_gray.data[IMG_WIDTH*(i-1) + (j-1) + 6];
      img_gray_16[7] = img_gray.data[IMG_WIDTH*(i-1) + (j-1) + 7];
      img_gray_16[8] = img_gray.data[IMG_WIDTH*(i-1) + (j-1) + 8];
      img_gray_16[9] = img_gray.data[IMG_WIDTH*(i-1) + (j-1) + 9];

      img_gray_data = vld1q_s16(&img_gray_16[0]);
      sobel_out = vaddq_s16(sobel_out, img_gray_data);

      // 2*img_gray.data[IMG_WIDTH*(i-1) + (j)]
      img_gray_data = vld1q_s16(&img_gray_16[1]);
      sobel_out = vmlaq_s16(sobel_out, img_gray_data, two);

      // img_gray.data[IMG_WIDTH*(i-1) + (j+1)]
      img_gray_data = vld1q_s16(&img_gray_16[2]);
      sobel_out = vaddq_s16(sobel_out, img_gray_data);

      // img_gray.data[IMG_WIDTH*(i+1) + (j-1)]
      img_gray_16[0] = img_gray.data[IMG_WIDTH*(i+1) + (j-1)];
      img_gray_16[1] = img_gray.data[IMG_WIDTH*(i+1) + (j-1) + 1];
      img_gray_16[2] = img_gray.data[IMG_WIDTH*(i+1) + (j-1) + 2];
      img_gray_16[3] = img_gray.data[IMG_WIDTH*(i+1) + (j-1) + 3];
      img_gray_16[4] = img_gray.data[IMG_WIDTH*(i+1) + (j-1) + 4];
      img_gray_16[5] = img_gray.data[IMG_WIDTH*(i+1) + (j-1) + 5];
      img_gray_16[6] = img_gray.data[IMG_WIDTH*(i+1) + (j-1) + 6];
      img_gray_16[7] = img_gray.data[IMG_WIDTH*(i+1) + (j-1) + 7];
      img_gray_16[8] = img_gray.data[IMG_WIDTH*(i+1) + (j-1) + 8];
      img_gray_16[9] = img_gray.data[IMG_WIDTH*(i+1) + (j-1) + 9];

      img_gray_data = vld1q_s16(&img_gray_16[0]);
      sobel_out = vsubq_s16(sobel_out, img_gray_data);

      // 2*img_gray.data[IMG_WIDTH*(i+1) + (j)]
      img_gray_data = vld1q_s16(&img_gray_16[1]);
      sobel_out = vmlsq_s16(sobel_out, img_gray_data, two);

      // img_gray.data[IMG_WIDTH*(i+1) + (j+1)]
      img_gray_data = vld1q_s16(&img_gray_16[2]);
      sobel_out = vsubq_s16(sobel_out, img_gray_data);

      sobel_out = vqabsq_s16(sobel_out);

      int16_t outx[8];
      vst1q_s16(outx, sobel_out);

      for (int k = 0; k < 8; k++) {
        sobel = (outx[k] > 255) ? 255 : outx[k];
        img_outx.data[IMG_WIDTH*(i) + j + k] = sobel;
      }

    //   sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		  // img_gray.data[IMG_WIDTH*(i+1) + (j-1)] +
		  // 2*img_gray.data[IMG_WIDTH*(i-1) + (j)] -
		  // 2*img_gray.data[IMG_WIDTH*(i+1) + (j)] +
		  // img_gray.data[IMG_WIDTH*(i-1) + (j+1)] -
		  // img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

    //   sobel = (sobel > 255) ? 255 : sobel;
    //   img_outx.data[IMG_WIDTH*(i) + (j)] = sobel;
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
