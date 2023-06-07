#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/optflow.hpp>

using namespace cv;

/**
 * @brief Function to calculate optical flow from two images and return the result as a byte array.
 *
 * @param env JNIEnv pointer.
 * @param prevPic byte array of the first image.
 * @param nextPic byte array of the second image.
 * @param w Width of the images.
 * @param h Height of the images.
 * @return Byte array representing the processed image.
 */
extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_example_dualphoneshooter_MainActivity_getOpticalFlowImage(
        JNIEnv *env,
        jobject /* this */,
        jbyteArray prevPic,
        jbyteArray nextPic, jint w, jint h) {

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
    resize(img1, img1_small, Size(w/2, h/2));
    resize(img2, img2_small, Size(w/2, h/2));

    // Calculate optical flow on the lower resolution images
    Mat flow_small;
    Ptr<DenseOpticalFlow> opticalFlowCalculator = cv::optflow::createOptFlow_SparseToDense();
    opticalFlowCalculator->calc(img1_small, img2_small, flow_small);

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
    jbyteArray returnByteArray = env->NewByteArray(rgbaImage.total() * rgbaImage.elemSize());

    // Get pointer to the data in Mat object
    uchar *ptr = rgbaImage.data;

    // Copy data from Mat object to the byte array
    env->SetByteArrayRegion(returnByteArray, 0, rgbaImage.total() * rgbaImage.elemSize(),
                            (jbyte *) ptr);

    // Return byte array
    return returnByteArray;
}
