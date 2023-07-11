#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>

#include <vector>

using namespace cv;

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
 * @return Byte array representing the processed image.
 */
extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_example_dualphoneshooter_MainActivity_getOpticalFlowImage(
        JNIEnv *env, jobject /* this */, jbyteArray prevPic, jbyteArray nextPic,
        jfloatArray opticalFlowVertices,
        jint w, jint h) {

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
    Ptr<DenseOpticalFlow> opticalFlowCalculator =
            cv::optflow::createOptFlow_SparseToDense();
    opticalFlowCalculator->calc(img1_small, img2_small, flow_small);

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
            float flowX = flowAtPoint.x / smallW;
            float flowY = flowAtPoint.y / smallH;

            // Offset vertex positions by flow. If the flow is not significant, you may not see any changes.
            // You may need to multiply flowX and flowY with a scaling factor to see noticeable changes.
            *vectexBuf++ = float(j) / w2_f - 1.0f + flowX;
            *vectexBuf++ = float(i) / h2_f - 1.0f + flowY;
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
