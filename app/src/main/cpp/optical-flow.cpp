/**
 * @file OpticalFlow.cpp
 * @brief Contains functions for dense optical flow estimation using multi-scale guided filtering.
 */
#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>
#include <stdexcept>

#include <vector>
#include <cmath>

using namespace cv;

/**
 * @fn inline void fillPads(float *array2d, int width, int height, int leftPadSize, int rightPadSize, int topPadSize, int bottomPadSize)
 * @brief Fill the padding area of a 2D array by repeating the values at the boundaries.
 *
 * @param array2d 2D array to be padded.
 * @param width Width of the array.
 * @param height Height of the array.
 * @param leftPadSize Padding size on the left side.
 * @param rightPadSize Padding size on the right side.
 * @param topPadSize Padding size on the top side.
 * @param bottomPadSize Padding size on the bottom side.
 */
inline void
fillPads(float *array2d, int width, int height, int leftPadSize, int rightPadSize, int topPadSize,
         int bottomPadSize) {
    if (array2d == nullptr) {
        throw std::invalid_argument("fillpads(): input array is null!");
    }
    if (leftPadSize + rightPadSize >= width || topPadSize + bottomPadSize >= height) {
        throw std::invalid_argument("fillpads(): unsupported pad sizes!");
    }
    //Fill the padding area by repeating the value at the boundaries.
    for (int i = topPadSize; i < width - bottomPadSize; i++) {
        //Fill the left pad and the right pad
        std::fill(array2d + i * width, array2d + i * width + leftPadSize,
                  array2d[i * width + leftPadSize]);
        std::fill(array2d + (i + 1) * width - rightPadSize, array2d + (i + 1) * width,
                  array2d[(i + 1) * width - rightPadSize - 1]);
    }
    for (int i = 0; i < topPadSize; i++) {
        //Fill top pad
        memcpy(reinterpret_cast<void *>(array2d + i * width),
               reinterpret_cast<void *>(array2d + topPadSize * width), width * sizeof(float));
    }
    for (int i = height - bottomPadSize; i < height; i++) {
        //Fill bottom pad
        memcpy(reinterpret_cast<void *>(array2d + i * width),
               reinterpret_cast<void *>(array2d + (height - bottomPadSize - 1) * width ),
               width * sizeof(float));
    }
}

/**
 * @fn inline void accumArray(float *array2d, int width, int height)
 * @brief Accumulate values horizontally and vertically in a 2D array.
 *
 * @param array2d 2D array to be accumulated.
 * @param width Width of the array.
 * @param height Height of the array.
 */
inline void accumArray(float *array2d, int width, int height) {
    if (array2d == nullptr) {
        throw std::invalid_argument("accumArray(): input array is null!");
    }
    if (width % 4 != 0 || height % 4 != 0) {
        throw std::invalid_argument("accumArray(): width and height have to be multiples of 4!");
    }
    //Accumulate at horizontal direction
    for (int i = 1; i < height; i++) {
        for (int j = 0; j < width; j += 4) {
            array2d[i * width + j] += array2d[(i - 1) * width + j];
            array2d[i * width + j + 1] += array2d[(i - 1) * width + j + 1];
            array2d[i * width + j + 2] += array2d[(i - 1) * width + j + 2];
            array2d[i * width + j + 3] += array2d[(i - 1) * width + j + 3];
        }
    }
    //Accumulate at vertical direction
    for (int i = 0; i < height; i += 4) {
        for (int j = 1; j < width; j++) {
            array2d[i * width + j] += array2d[i * width + j - 1];
            array2d[(i + 1) * width + j] += array2d[(i + 1) * width + j - 1];
            array2d[(i + 2) * width + j] += array2d[(i + 2) * width + j - 1];
            array2d[(i + 3) * width + j] += array2d[(i + 3) * width + j - 1];
        }
    }
}

/**
 * @fn inline float getSubareaSum(int i, int j, const float *array2d, int width, int height, const std::vector<int> &pitches, const std::vector<float> &pitchWeights)
 * @brief Calculate the weighted sum of subareas in a 2D array.
 *
 * @param i Vertical index in the array.
 * @param j Horizontal index in the array.
 * @param array2d 2D array from which to retrieve subarea sums.
 * @param width Width of the array.
 * @param height Height of the array.
 * @param pitches Vector containing the pitch values.
 * @param pitchWeights Vector containing the weight for each corresponding pitch.
 * @return Weighted sum of specified subareas.
 */
inline float getSubareaSum(int i, int j, const float *array2d, int width, int height,
                           const std::vector<int> &pitches,
                           const std::vector<float> &pitchWeights) {
    if (array2d == nullptr) {
        throw std::invalid_argument("getSubareaSum(): input array is null!");
    }
    if (pitches.size() == 0 || pitchWeights.size() == 0 || pitches.size() != pitchWeights.size()) {
        throw std::invalid_argument("getSubareaSum(): invalid pitches or pitchWeights sizes!");
    }
    //TODO assume the last pitch is the largest
    int maxPitch = *pitches.rbegin();
    if (i - maxPitch < 1 || j - maxPitch < 1 || i + maxPitch >= height || j + maxPitch >= width) {
        throw std::invalid_argument("getSubareaSum(): boundary issue!");
    }

    float val = .0f;
    for (int ii = 0; ii < pitches.size(); ii++) {
        int pitch = pitches[ii];
        float weight = pitchWeights[ii];
        float curVal = array2d[(i + pitch) * width + j + pitch];
        curVal -= array2d[(i + pitch) * width + j - pitch - 1];
        curVal -= array2d[(i - pitch - 1) * width + j + pitch];
        curVal += array2d[(i - pitch - 1) * width + j - pitch - 1];
        val += weight * curVal;
    }
    return val;
}

static const int PADDED_MAX_H = 960;
static const int PADDED_MAX_W = 1280;
static const int H_PAD_SIZE = 128;
static const int V_PAD_SIZE = 128;

static const std::vector<int> pitches = {16, 32, 64};
static const std::vector<float> pitch_weights = {1.0f, .2f, .04f};

static const float epsilon = .001f;

static const int MIN_OF_POINTS = 5;

static float weight[3][PADDED_MAX_H * PADDED_MAX_W];
static float weightedFx[3][PADDED_MAX_H * PADDED_MAX_W];
static float weightedFy[3][PADDED_MAX_H * PADDED_MAX_W];
static float IW[3][PADDED_MAX_H * PADDED_MAX_W];
static float IIW[3][PADDED_MAX_H * PADDED_MAX_W];
static float IWFx[3][PADDED_MAX_H * PADDED_MAX_W];
static float IWFy[3][PADDED_MAX_H * PADDED_MAX_W];
//The input of pixel intensity need to be scaled down to help
// with data stability issue, which is more obvious when using
// 2D array integration to compute sum of a square area.
static float dataStabilityCoef = 1.0f / 128.0f;

/**
 * @fn void computeDenseOpticalFlow(Mat &grayScale0, Mat &grayScale1, Mat &outFlow, int workID)
 * @brief Compute dense optical flow between two grayscale images using multiscale guided filtering.
 *
 * @param grayScale0 First grayscale image.
 * @param grayScale1 Second grayscale image.
 * @param outFlow Output matrix to store the estimated optical flow.
 * @param workID ID of the work, used to select working arrays.
 */
void computeDenseOpticalFlow(Mat &grayScale0, Mat &grayScale1, Mat &outFlow, int workID) {
    //Check inputs and handle edge cases
    int m0 = grayScale0.rows;
    int n0 = grayScale0.cols;
    int m1 = grayScale1.rows;
    int n1 = grayScale1.cols;
    if (m0 <= 16 || n0 <= 16 || m1 < 16 || n1 <= 16 || m0 != m1 || n0 != n1
        || m0 >= PADDED_MAX_H - 2 * H_PAD_SIZE || m1 >= PADDED_MAX_H - 2 * H_PAD_SIZE
        || n0 >= PADDED_MAX_W - 2 * V_PAD_SIZE || n1 >= PADDED_MAX_W - 2 * V_PAD_SIZE
        || workID < 0 || workID >= 3) {
        throw std::invalid_argument("computeDenseOpticalFlow");
    }

    int grid_step = 6;
    std::vector<Point2f> points;
    std::vector<Point2f> dst_points;
    std::vector<unsigned char> status;
    std::vector<float> err;
    std::vector<Point2f> points_filtered, dst_points_filtered;
    float maxXDiff = 32.0f;  //Flow larger than this is an outlier
    float maxYDiff = 20.0f;  //Flow larger than this is an outlier
    int width = grayScale0.cols;
    int height = grayScale0.rows;
    int paddedWidth = width + H_PAD_SIZE * 2;
    int paddedHeight = height + V_PAD_SIZE * 2;
    for (int i = 0; i < grayScale0.rows; i += grid_step)
        for (int j = 0; j < grayScale0.cols; j += grid_step)
            points.push_back(Point2f((float) j, (float) i));
    calcOpticalFlowPyrLK(grayScale0, grayScale1, points, dst_points, status, err, Size(21, 21));
    for (unsigned int i = 0; i < points.size(); i++) {
        if (status[i] != 0) {
            float diffX = dst_points[i].x - points[i].x;
            float diffY = dst_points[i].y - points[i].y;
            if (std::abs(diffX) < maxXDiff && std::abs(diffY) < maxYDiff) {
                points_filtered.push_back(points[i]);
                dst_points_filtered.push_back(dst_points[i]);
            }
        }
    }
    outFlow.create(grayScale0.size(), CV_32FC2);

    //Handle flat image
    if (points_filtered.size() < MIN_OF_POINTS || dst_points_filtered.size() < MIN_OF_POINTS) {
        outFlow *= .0;
        return;
    }

    //Do multi-scale guided filtering
    //Zero weight, weighted_Fx and weighted_Fy

    memset(reinterpret_cast<void *>(&weight[workID][0]), 0,
           paddedHeight * paddedWidth * sizeof(float));
    memset(reinterpret_cast<void *>(&weightedFx[workID][0]), 0,
           paddedHeight * paddedWidth * sizeof(float));
    memset(reinterpret_cast<void *>(&weightedFy[workID][0]), 0,
           paddedHeight * paddedWidth * sizeof(float));
    //Annotate sparse optical flow points
    for (int i = 0; i < points_filtered.size(); i++) {
        int idx = (std::round(points_filtered[i].y) + V_PAD_SIZE) * paddedWidth;
        idx += std::round(points_filtered[i].x) + H_PAD_SIZE;
        weight[workID][idx] = 1.0f;
        float fx = dst_points_filtered[i].x - points_filtered[i].x;
        float fy = dst_points_filtered[i].y - points_filtered[i].y;
        weightedFx[workID][idx] = fx;
        weightedFy[workID][idx] = fy;
    }

    //Prepare 2d data arrays needed for guided filtering
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float pixVal = float(grayScale0.at<uchar>(i, j)) * dataStabilityCoef;
            int idx = (i + V_PAD_SIZE) * paddedWidth + j + H_PAD_SIZE;
            float w = weight[workID][idx];
            float wFx = weightedFx[workID][idx];
            float wFy = weightedFy[workID][idx];
            IW[workID][idx] = pixVal * w;
            IIW[workID][idx] = pixVal * pixVal * w;
            IWFx[workID][idx] = pixVal * wFx;
            IWFy[workID][idx] = pixVal * wFy;
        }
    }

    //Handle padding area
    fillPads(&weight[workID][0], paddedWidth, paddedHeight, H_PAD_SIZE, H_PAD_SIZE, V_PAD_SIZE,
             V_PAD_SIZE);
    fillPads(&IIW[workID][0], paddedWidth, paddedHeight, H_PAD_SIZE, H_PAD_SIZE, V_PAD_SIZE,
             V_PAD_SIZE);
    fillPads(&IW[workID][0], paddedWidth, paddedHeight, H_PAD_SIZE, H_PAD_SIZE, V_PAD_SIZE,
             V_PAD_SIZE);
    fillPads(&IWFx[workID][0], paddedWidth, paddedHeight, H_PAD_SIZE, H_PAD_SIZE, V_PAD_SIZE,
             V_PAD_SIZE);
    fillPads(&weightedFx[workID][0], paddedWidth, paddedHeight, H_PAD_SIZE, H_PAD_SIZE, V_PAD_SIZE,
             V_PAD_SIZE);
    fillPads(&IWFy[workID][0], paddedWidth, paddedHeight, H_PAD_SIZE, H_PAD_SIZE, V_PAD_SIZE,
             V_PAD_SIZE);
    fillPads(&weightedFy[workID][0], paddedWidth, paddedHeight, H_PAD_SIZE, H_PAD_SIZE, V_PAD_SIZE,
             V_PAD_SIZE);

    //Create accumulated arrays
    accumArray(&weight[workID][0], paddedWidth, paddedHeight);
    accumArray(&IIW[workID][0], paddedWidth, paddedHeight);
    accumArray(&IW[workID][0], paddedWidth, paddedHeight);
    accumArray(&IWFx[workID][0], paddedWidth, paddedHeight);
    accumArray(&weightedFx[workID][0], paddedWidth, paddedHeight);
    accumArray(&IWFy[workID][0], paddedWidth, paddedHeight);
    accumArray(&weightedFy[workID][0], paddedWidth, paddedHeight);

    //Actual filtering
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int ii = i + V_PAD_SIZE;
            int jj = j + H_PAD_SIZE;
            float w = getSubareaSum(ii, jj, &weight[workID][0], paddedWidth, paddedHeight, pitches,
                                    pitch_weights);
            float iiw = getSubareaSum(ii, jj, &IIW[workID][0], paddedWidth, paddedHeight, pitches,
                                      pitch_weights);
            float iw = getSubareaSum(ii, jj, &IW[workID][0], paddedWidth, paddedHeight, pitches,
                                     pitch_weights);
            float iwfx = getSubareaSum(ii, jj, &IWFx[workID][0], paddedWidth, paddedHeight, pitches,
                                       pitch_weights);
            float wfx = getSubareaSum(ii, jj, &weightedFx[workID][0], paddedWidth, paddedHeight,
                                      pitches, pitch_weights);
            float iwfy = getSubareaSum(ii, jj, &IWFy[workID][0], paddedWidth, paddedHeight, pitches,
                                       pitch_weights);
            float wfy = getSubareaSum(ii, jj, &weightedFy[workID][0], paddedWidth, paddedHeight,
                                      pitches, pitch_weights);
            //Compute inverse of 2x2 matrix
            float AA00 = iiw + epsilon;
            float AA01 = iw;
            float AA10 = iw;
            float AA11 = w + epsilon;
            float inDet = 1.0f / (AA00 * AA11 - AA10 * AA01);
            float invAA00 = AA11 * inDet;
            float invAA01 = -AA10 * inDet;
            float invAA10 = invAA01;
            float invAA11 = AA00 * inDet;
            //Compute coefs of 1st order and 0th order
            float aX = invAA00 * iwfx + invAA01 * wfx;
            float bX = invAA10 * iwfx + invAA11 * wfx;
            float aY = invAA00 * iwfy + invAA01 * wfy;
            float bY = invAA10 * iwfy + invAA11 * wfy;

            float pixVal = float(grayScale0.at<uchar>(i, j)) * dataStabilityCoef;
            float fx = aX * pixVal + bX;
            float fy = aY * pixVal + bY;
            outFlow.at<Vec2f>(i, j)[0] = fx;
            outFlow.at<Vec2f>(i, j)[1] = fy;
        }
    }
}

