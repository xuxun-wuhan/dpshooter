#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>

#include <vector>
#include <cmath>

using namespace cv;

// Prepare three instances of the Dense Optical Flow estimator
// for three separate worker threads.
Ptr <DenseOpticalFlow> opticalFlowCalculator[3] =
        {cv::optflow::createOptFlow_SparseToDense(),
         cv::optflow::createOptFlow_SparseToDense(),
         cv::optflow::createOptFlow_SparseToDense()};

std::vector<float> holeFillingTextureCoords[3];
std::vector<float> holeFillingVertices[3];


/**
 * @brief This function fetches the flow vector at a specific point.
 *
 * @param x The x-coordinate of the point.
 * @param y The y-coordinate of the point.
 * @param flow The optical flow matrix.
 * @return Point2f The optical flow vector at the specific point.
 */
inline Point2f getFlowAt(int x, int y, Mat &flow) {
    float fx = flow.at<Vec2f>(y, x)[0];
    float fy = flow.at<Vec2f>(y, x)[1];
    return Point2f(fx, fy);
}

/**
 * @brief Creates triangles by backward warping based on the optical flow.
 *
 * @param width Width of the image.
 * @param height Height of the image.
 * @param pointBG Background point.
 * @param pointFG Foreground point.
 * @param flowBG Background flow.
 * @param holeFillingTextureCoords Texture coordinates for hole filling.
 * @param holeFillingVertices Vertices for hole filling.
 */
 inline void createTriangles(int width, int height, Vec2f &pointBG, Vec2f &pointFG, Vec2f &flowBG,
                            std::vector<float> &holeFillingTextureCoords,
                            std::vector<float> &holeFillingVertices) {
    //Compile texture and vertex points.
    //Note: as far, all the points are in image coordinates
    Vec2f pointDiff = pointBG - pointFG;
    //Swap the x, y of pointDiff and reverse the sign of x, or y
    // to make the pointDiff orthogonal to (pointBG - pointFG)
    float temp = pointDiff[1];
    pointDiff[1] = pointDiff[0];
    pointDiff[0] = -temp;
    pointDiff = pointDiff / (norm(pointDiff) + .01f);
    Vec2f pointTx[4];
    pointTx[0] = pointBG + pointDiff;
    pointTx[1] = pointBG - pointDiff;
    pointTx[2] = pointFG + pointDiff;
    pointTx[3] = pointFG - pointDiff;
    Vec2f pointVx[4];
    for (int idx = 0; idx < 4; idx++) {
        pointVx[idx] = pointTx[idx] + flowBG;
    }
    for (int idx = 0; idx < 4; idx++) {
        //Change points from image coordinates to texture coordinates and vertex coordinates
        pointTx[idx][0] /= float(width);
        pointTx[idx][1] = (float(height - 1) - pointTx[idx][1]) / float(height);

        pointVx[idx][0] = pointVx[idx][0] / float(width) * 2.0f - 1.0f;
        pointVx[idx][1] =
                (float(height - 1) - pointVx[idx][1]) / float(height) * 2.0f - 1.0f;
    }
    //Triangle 0
    holeFillingTextureCoords.push_back(pointTx[0][0]);
    holeFillingTextureCoords.push_back(pointTx[0][1]);
    holeFillingTextureCoords.push_back(pointTx[1][0]);
    holeFillingTextureCoords.push_back(pointTx[1][1]);
    holeFillingTextureCoords.push_back(pointTx[3][0]);
    holeFillingTextureCoords.push_back(pointTx[3][1]);

    holeFillingVertices.push_back(pointVx[0][0]);
    holeFillingVertices.push_back(pointVx[0][1]);
    holeFillingVertices.push_back(pointVx[1][0]);
    holeFillingVertices.push_back(pointVx[1][1]);
    holeFillingVertices.push_back(pointVx[3][0]);
    holeFillingVertices.push_back(pointVx[3][1]);

    //Triangle 1
    holeFillingTextureCoords.push_back(pointTx[0][0]);
    holeFillingTextureCoords.push_back(pointTx[0][1]);
    holeFillingTextureCoords.push_back(pointTx[2][0]);
    holeFillingTextureCoords.push_back(pointTx[2][1]);
    holeFillingTextureCoords.push_back(pointTx[3][0]);
    holeFillingTextureCoords.push_back(pointTx[3][1]);

    holeFillingVertices.push_back(pointVx[0][0]);
    holeFillingVertices.push_back(pointVx[0][1]);
    holeFillingVertices.push_back(pointVx[2][0]);
    holeFillingVertices.push_back(pointVx[2][1]);
    holeFillingVertices.push_back(pointVx[3][0]);
    holeFillingVertices.push_back(pointVx[3][1]);
}

/**
 * @brief Solve warping occlusion based on optical flow and fill holes in the warped images.
 *
 * @param image1 First image frame.
 * @param image2 Second image frame.
 * @param flow Optical flow.
 * @param holeFillingTextureCoords Texture coordinates for hole filling.
 * @param holeFillingVertices Vertices for hole filling.
 * @param intervalPitch Temporal position between image1 and image2. 0: image1, 1: image2, 0.5: middle frame between image1 and image2.
 */
 void solveWarpingOcclusion(Mat &image1, Mat &image2, Mat &flow,
                           std::vector<float> &holeFillingTextureCoords,
                           std::vector<float> &holeFillingVertices, float intervalPitch) {
    holeFillingTextureCoords.clear();
    holeFillingVertices.clear();
    int width = image1.cols;
    int height = image1.rows;
    const int topBdWidth = 60;
    const int sideBdWidth = 100;
    const float wParam = 8.0f;
    const int kSize = 5;
    const float sigmaX = .7f * (intervalPitch + 0.1f);

    const float flowDiffThres = 2.0f;
    Mat paddedImage2;
    Mat weights(flow.size(), CV_32FC2);
    weights.setTo(Scalar(.01));

    copyMakeBorder(image2, paddedImage2, topBdWidth, topBdWidth, sideBdWidth, sideBdWidth,
                   cv::BORDER_REPLICATE);
    //Using horizontal pixel pairs to locate holes
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width - 1; j++) {
            Vec2f f0 = flow.at<Vec2f>(i, j);
            Vec2f f1 = flow.at<Vec2f>(i, j + 1);
            if (norm(f0 - f1) < flowDiffThres) {
                //If the flow diff is small, ignore.
                continue;
            }
            if (f0[0] > f1[0]) {
                //The object on the left is overlapping the object on the right
                continue;
            }
            float norm0 = norm(f0);
            float norm1 = norm(f1);
            Vec2f fMid = (f0 + f1) / 2.0f;
            //Fetch intensity pointed by fMid, if the intensity is more similar to
            //the intensity with smaller flow, this is a hole.
            float val0 = float(image1.at<uchar>(i, j));
            float val1 = float(image1.at<uchar>(i, j + 1));

            Vec2f point = {float(i) + fMid[1], float(j) + 0.5f + fMid[0]};
            int ii = int(std::round(point[0]));
            int jj = int(std::round(point[1]));
            float valMid = float(paddedImage2.at<uchar>(ii + topBdWidth, jj + sideBdWidth));
            Vec2f pointFG, pointBG, flowBG;
            if (norm0 < norm1) {
                //point 0 is background
                if (std::abs(valMid - val0) > std::abs(valMid - val1)) {
                    //No background exposed, do nothing
                    continue;
                }
                pointBG = {float(j), float(i)};
                pointBG += f0;
                pointFG = {float(j + 1), float(i)};
                pointFG += f1;
                flowBG = f0 * intervalPitch - f0;
            } else {
                //point 1 is background
                if (std::abs(valMid - val1) > std::abs(valMid - val0)) {
                    //No background exposed, do nothing
                    continue;
                }
                pointBG = {float(j + 1), float(i)};
                pointBG += f1;
                pointFG = {float(j), float(i)};
                pointFG += f0;
                flowBG = f1 * intervalPitch - f1;
            }
            createTriangles(width, height, pointBG, pointFG, flowBG, holeFillingTextureCoords,
                            holeFillingVertices);

            weights.at<Vec2f>(i, j)[0] = 1.0f;
            weights.at<Vec2f>(i, j)[1] = 1.0f;
            weights.at<Vec2f>(i, j + 1)[0] = 1.0f;
            weights.at<Vec2f>(i, j + 1)[1] = 1.0f;
        }
    }
    // Using virtical pairs to locate holes
    for (int i = 0; i < height - 1; i++) {
        for (int j = 0; j < width; j++) {
            Vec2f f0 = flow.at<Vec2f>(i, j);
            Vec2f f1 = flow.at<Vec2f>(i + 1, j);
            if (norm(f0 - f1) < flowDiffThres) {
                //If the flow diff is small, ignore.
                continue;
            }
            if (f0[1] > f1[1]) {
                //The object on the top is overlapping the object on the right
                continue;
            }
            float norm0 = norm(f0);
            float norm1 = norm(f1);
            Vec2f fMid = (f0 + f1) / 2.0f;
            //Fetch intensity pointed by fMid, if the intensity is more similar to
            //the intensity with smaller flow, this is a hole.
            float val0 = float(image1.at<uchar>(i, j));
            float val1 = float(image1.at<uchar>(i + 1, j));

            Vec2f point = {float(i) + 0.5f + fMid[1], float(j) + fMid[0]};
            int ii = int(std::round(point[0]));
            int jj = int(std::round(point[1]));
            float valMid = float(paddedImage2.at<uchar>(ii + topBdWidth, jj + sideBdWidth));
            Vec2f pointFG, pointBG, flowBG;
            if (norm0 < norm1) {
                //point 0 is background
                if (std::abs(valMid - val0) > std::abs(valMid - val1)) {
                    //No background exposed, do nothing
                    continue;
                }
                pointBG = {float(j), float(i)};
                pointBG += f0;
                pointFG = {float(j), float(i + 1)};
                pointFG += f1;
                flowBG = f0 * intervalPitch - f0;
            } else {
                //point 1 is background
                if (std::abs(valMid - val1) > std::abs(valMid - val0)) {
                    //No background exposed, do nothing
                    continue;
                }
                pointBG = {float(j), float(i + 1)};
                pointBG += f1;
                pointFG = {float(j), float(i)};
                pointFG += f0;
                flowBG = f1 * intervalPitch - f1;
            }
            createTriangles(width, height, pointBG, pointFG, flowBG, holeFillingTextureCoords,
                            holeFillingVertices);

            weights.at<Vec2f>(i, j)[0] = 1.0f;
            weights.at<Vec2f>(i, j)[1] = 1.0f;
            weights.at<Vec2f>(i + 1, j)[0] = 1.0f;
            weights.at<Vec2f>(i + 1, j)[1] = 1.0f;
        }
    }

    // Gaussian filtering flow and blend with the original flow.
    // Areas of hole are more biased toward original flow
    Mat blurred_weight;
    GaussianBlur(weights, blurred_weight, cv::Size(kSize, kSize), sigmaX);
    float maxWeight = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            maxWeight = std::max(maxWeight, weights.at<Vec2f>(i, j)[0]);
        }
    }
    maxWeight = std::max(maxWeight, .01f);
    blurred_weight = blurred_weight / maxWeight;

    // Apply Gaussian blur to the flow
    Mat blurred_flow;
    GaussianBlur(flow, blurred_flow, cv::Size(kSize, kSize), sigmaX);

    // Apply weight to flow.
    Mat filtered_flow0, filtered_flow1;
    multiply(flow, blurred_weight, filtered_flow0);
    multiply(blurred_flow, 1.0f - blurred_weight, filtered_flow1);
    flow = filtered_flow0 + filtered_flow1;
}


/**
 * @brief Calculate optical flow from two images and return the
 * result as a byte array. It also manipulates OpenGL vertices
 * to simulate forward warping.
 *
 * @param env JNIEnv pointer.
 * @param prevPic Byte array of the first image.
 * @param nextPic Byte array of the second image.
 * @param opticalFlowVertices OpenGL vertices used for optical flow based forward warping.
 * @param w Width of the images.
 * @param h Height of the images.
 * @param workerId The id of the worker thread
 * @param intervalPitch Shows temporal position o warped frame. 0: identical to prevPic,
 *                       1: identical to nextPic, 0.5 in the midle of prevPic and nextPic
 * @return Byte array representing the processed image.
 */
extern "C"
JNIEXPORT jbyteArray

JNICALL
Java_com_example_dualphoneshooter_MainActivity_getOpticalFlowImage(
        JNIEnv *env, jobject /* this */, jbyteArray prevPic, jbyteArray nextPic,
        jfloatArray opticalFlowVertices,
        jint w, jint h, jint workerId, jfloat intervalPitch) {

    // Get byte array data
    jbyte *prevPicData = env->GetByteArrayElements(prevPic, NULL);
    jbyte *nextPicData = env->GetByteArrayElements(nextPic, NULL);

    // Convert byte array to Mat
    Mat img1(h, w, CV_8UC1, reinterpret_cast<unsigned char *>(prevPicData));
    Mat img2(h, w, CV_8UC1, reinterpret_cast<unsigned char *>(nextPicData));

    // Release byte array elements
    env->ReleaseByteArrayElements(prevPic, prevPicData, 0);
    env->ReleaseByteArrayElements(nextPic, nextPicData, 0);

    // Reduce resolution of the images
    Mat img1_small, img2_small;
    resize(img1, img1_small, Size(w / 2, h / 2));
    resize(img2, img2_small, Size(w / 2, h / 2));

    // Calculate optical flow on the lower resolution images
    Mat flow_small;
    opticalFlowCalculator[workerId]->calc(img1_small, img2_small, flow_small);

    solveWarpingOcclusion(img1_small, img2_small, flow_small, holeFillingTextureCoords[workerId],
                          holeFillingVertices[workerId], intervalPitch);
    // Get float array data
    jfloat *opticalFlowVerticesData = env->GetFloatArrayElements(opticalFlowVertices, NULL);

    // Set OpenGL vertices which covers the a 2.0 x 2.0 area.
    int smallW = w / 2;
    int smallH = h / 2;
    float w2_f = float(smallW / 2);
    float h2_f = float(smallH / 2);
    float *vectexBuf = opticalFlowVerticesData;

    for (int i = smallH - 1; i >= 0; i--) {
        for (int j = 0; j < smallW; j++) {
            //Note: the image orientations of openGL vertex
            // and optical flow are different
            Point2f flowAtPoint = getFlowAt(j, smallH - 1 - i, flow_small);

            // Here we need to normalize the flow values to the range that the vertex coordinates are in.
            // Assuming that the maximum flow is not more than the image width or height
            // If the flow values exceed this, you may need to adjust this calculation.
            // Note, the Y flow direction need to be reversed due to direction difference of
            // image orientation and openGL vertex orientation.
            float flowX = flowAtPoint.x / smallW;
            float flowY = -flowAtPoint.y / smallH;

            // Offset vertex positions by flow. If the flow is not significant, you may not see any changes.
            // You may need to multiply flowX and flowY with a scaling factor to see noticeable changes.
            *vectexBuf++ = float(j) / w2_f - 1.0f + flowX * 2.0 * intervalPitch;
            *vectexBuf++ = float(i) / h2_f - 1.0f + flowY * 2.0 * intervalPitch;
        }
    }

    // Release float array elements
    env->ReleaseFloatArrayElements(opticalFlowVertices, opticalFlowVerticesData, 0);

    // Resize flow back to the original size
    Mat flow;
    resize(flow_small, flow, img1.size());

    // Split channels
    Mat flowParts[2];
    split(flow, flowParts);

    // Find magnitude and angle
    Mat magnitude, angle;
    cartToPolar(flowParts[0], flowParts[1], magnitude, angle, true);

    // Translate magnitude to range [0;1]
    normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);

    // HSV channels
    Mat hsvChannels[3];
    hsvChannels[0] = angle;
    hsvChannels[1] = Mat::ones(angle.size(), angle.type());
    hsvChannels[2] = magnitude;

    // Merge channels
    Mat hsvImage;
    merge(hsvChannels, 3, hsvImage);

    // Convert to RGB
    Mat rgbImage;
    cvtColor(hsvImage, rgbImage, COLOR_HSV2BGR);

    // Scale and convert to 8-bit
    rgbImage *= 255;
    rgbImage.convertTo(rgbImage, CV_8UC3);

    // Convert from RGB to RGBA
    Mat rgbaImage;
    cvtColor(rgbImage, rgbaImage, COLOR_BGR2BGRA);

    // Prepare to return a byte array
    jbyteArray returnByteArray =
            env->NewByteArray(rgbaImage.total() * rgbaImage.elemSize());

    // Get pointer to the data in Mat object
    uchar *ptr = rgbaImage.data;

    // Copy data from Mat object to the byte array
    env->SetByteArrayRegion(returnByteArray, 0,
                            rgbaImage.total() * rgbaImage.elemSize(),
                            (jbyte *) ptr);

    // Return byte array
    return returnByteArray;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_dualphoneshooter_MainActivity_getHoleFillingTextureCoords(
        JNIEnv *env, jobject /* this */, jint workerId) {
    // Create a new Java float array
    if (holeFillingTextureCoords[workerId].empty()) {
        return nullptr;
    }
    size_t size = holeFillingTextureCoords[workerId].size();
    float *data_p = &holeFillingTextureCoords[workerId][0];
    jfloatArray result = env->NewFloatArray(size);
    if (result == nullptr) {
        return nullptr;  // OutOfMemoryError already thrown by JVM
    }

    // Copy data from std::vector<float> to the Java float array
    env->SetFloatArrayRegion(result, 0, size, data_p);
    return result;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_dualphoneshooter_MainActivity_getHoleFillingVertices(
        JNIEnv *env, jobject /* this */, jint workerId) {
    // Create a new Java float array
    if (holeFillingVertices[workerId].empty()) {
        return nullptr;
    }
    size_t size = holeFillingVertices[workerId].size();
    float *data_p = &holeFillingVertices[workerId][0];
    jfloatArray result = env->NewFloatArray(size);
    if (result == nullptr) {
        return nullptr;  // OutOfMemoryError already thrown by JVM
    }

    // Copy data from std::vector<float> to the Java float array
    env->SetFloatArrayRegion(result, 0, size, data_p);
    return result;
}



/**
 * @brief Generate texture coordinates for a 2D texture that matches the dimensions of a specific image.
 *
 * @param env JNIEnv pointer.
 * @param textureBuffer A direct Java ByteBuffer to store the texture coordinates.
 * @param w The width of the image.
 * @param h The height of the image.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_example_dualphoneshooter_OpenGLProcessor_getTextureCoords(
        JNIEnv *env, jobject /* this */, jobject textureBuffer, jint w, jint h) {
    float *textureBuf =
            static_cast<float *>(env->GetDirectBufferAddress(textureBuffer));
    float w_f = float(w);
    float h_f = float(h);
    for (int i = h - 1; i >= 0; i--) {
        for (int j = 0; j < w; j++) {
            *textureBuf++ = float(j) / w_f;
            *textureBuf++ = float(i) / h_f;
        }
    }
}

/**
 * @brief Generate index buffer for a 2D grid mesh. Each quad in the grid is represented by two triangles.
 *
 * @param env JNIEnv pointer.
 * @param indexBuffer A direct Java ByteBuffer to store the index buffer.
 * @param w The width of the grid.
 * @param h The height of the grid.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_example_dualphoneshooter_OpenGLProcessor_getTriangleIndexBuffer(
        JNIEnv *env, jobject /* this */, jobject indexBuffer, jint w, jint h) {
    // Convert the Java buffer to a C++ vector
    jint *indexBufferC =
            static_cast<jint *>(env->GetDirectBufferAddress(indexBuffer));
    std::vector<jint> indices;

    // Here is where you would perform the calculation for the indices based on
    // w and h.
    for (int y = 0; y < h - 1; ++y) {
        for (int x = 0; x < w - 1; ++x) {
            // Define two triangles for the quad at (x, y)
            jint topLeft = y * w + x;
            jint bottomLeft = (y + 1) * w + x;
            jint topRight = y * w + x + 1;
            jint bottomRight = (y + 1) * w + x + 1;

            // First triangle (top left, bottom left, top right)
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);

            // Second triangle (top right, bottom left, bottom right)
            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }

    // Copy the indices back to the Java buffer
    memcpy(indexBufferC, indices.data(), indices.size() * sizeof(jint));
}
