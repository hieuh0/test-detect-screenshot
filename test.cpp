#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <string>
#include <chrono>

#define LOG_TAG "NativeLib"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_findTextCoordinates(JNIEnv *env, jobject thiz,
                                                                jobject bitmap, jstring input_text,
                                                                jstring datapath) {
    // Start time measurement
    auto start_time = std::chrono::steady_clock::now();

    AndroidBitmapInfo info;
    void *pixels = nullptr;
    int ret;

    // Convert Java strings to C-style strings
    const char *input_text_char = env->GetStringUTFChars(input_text, nullptr);
    const char *datapath_char = env->GetStringUTFChars(datapath, nullptr);

    // Get bitmap info
    if ((ret = AndroidBitmap_getInfo(env, bitmap, &info)) < 0) {
        LOGE("AndroidBitmap_getInfo() failed: %d", ret);
        return env->NewStringUTF("Failed to get bitmap info.");
    }

    // Lock bitmap pixels
    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &pixels)) < 0) {
        LOGE("AndroidBitmap_lockPixels() failed: %d", ret);
        return env->NewStringUTF("Failed to lock bitmap pixels.");
    }

    // Convert bitmap to OpenCV Mat (use CV_8UC1 if alpha channel is not necessary)
    cv::Mat img(info.height, info.width, CV_8UC4, pixels);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_RGBA2GRAY);

    // Image preprocessing
    cv::Mat thresh;
    cv::threshold(gray, thresh, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
    cv::Mat rect_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(18, 18));
    cv::Mat dilation;
    cv::dilate(thresh, dilation, rect_kernel, cv::Point(-1, -1), 1);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilation, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // Initialize Tesseract
    tesseract::TessBaseAPI tess;
    if (tess.Init(datapath_char, "eng", tesseract::OEM_LSTM_ONLY) != 0) {
        LOGE("Failed to initialize Tesseract.");
        return env->NewStringUTF("Failed to initialize Tesseract.");
    }
    tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);

    bool found = false;
    std::string result_text;

    // Process each contour in parallel using multiple threads
#pragma omp parallel for
    for (int i = 0; i < contours.size(); ++i) {
        if (found) continue;  // If text already found, skip processing further contours

        cv::Rect bounding_rect = cv::boundingRect(contours[i]);
        cv::Mat cropped = gray(bounding_rect);

        // Set Tesseract image
        tess.SetImage(cropped.data, cropped.cols, cropped.rows, 1, cropped.step);
        char *text = tess.GetUTF8Text();
        if (text == nullptr) {
            continue;
        }

        std::string recognized_text(text);
        delete[] text; // Properly delete the text after use

        if (recognized_text.find(input_text_char) != std::string::npos) {
            // Found text
            result_text = "Location of input text: x = " + std::to_string(bounding_rect.x) +
                          " y = " + std::to_string(bounding_rect.y);
            found = true;
        }
    }

    // Clean up
    tess.End();
    AndroidBitmap_unlockPixels(env, bitmap);
    env->ReleaseStringUTFChars(input_text, input_text_char);
    env->ReleaseStringUTFChars(datapath, datapath_char);

    if (!found) {
        result_text = "Input text not found in the image.";
    }

    // End time measurement
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
    LOGI("Processing time: %lld milliseconds", elapsed_time);

    // Return result as a JNI string
    return env->NewStringUTF(result_text.c_str());
}

}
