#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

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

    // Chuyển đổi chuỗi Java sang chuỗi C-style
    const char *input_text_char = env->GetStringUTFChars(input_text, nullptr);
    const char *datapath_char = env->GetStringUTFChars(datapath, nullptr);

    // Lấy thông tin bitmap
    if ((ret = AndroidBitmap_getInfo(env, bitmap, &info)) < 0) {
        LOGE("AndroidBitmap_getInfo() failed: %d", ret);
        return env->NewStringUTF("Failed to get bitmap info.");
    }

    // Khóa bitmap pixels
    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &pixels)) < 0) {
        LOGE("AndroidBitmap_lockPixels() failed: %d", ret);
        return env->NewStringUTF("Failed to lock bitmap pixels.");
    }

    // Chuyển bitmap thành OpenCV Mat
    cv::Mat img(info.height, info.width, CV_8UC4, pixels);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_RGBA2GRAY);

    // Tiền xử lý ảnh
    cv::Mat thresh;
    cv::threshold(gray, thresh, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
    cv::Mat rect_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(18, 18));
    cv::Mat dilation;
    cv::dilate(thresh, dilation, rect_kernel, cv::Point(-1, -1), 1);

    // Tìm các đường viền
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilation, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    bool found = false;
    std::string result_text;

    // Số luồng
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::mutex mutex;

    // Hàm xử lý từng contour
    auto process_contour = [&](int start, int end) {
        tesseract::TessBaseAPI tess;
        if (tess.Init(datapath_char, "eng", tesseract::OEM_LSTM_ONLY) != 0) {
            LOGE("Failed to initialize Tesseract.");
            return;
        }
        tess.SetPageSegMode(tesseract::PSM_AUTO);

        for (int i = start; i < end; ++i) {
            if (found) continue;  // Nếu đã tìm thấy văn bản, bỏ qua các đường viền khác

            cv::Rect bounding_rect = cv::boundingRect(contours[i]);
            if (bounding_rect.width < 30 || bounding_rect.height < 30 ||
                bounding_rect.width > img.cols / 2 || bounding_rect.height > img.rows / 2) {
                continue;
            }

            tess.SetImage(gray.data, gray.cols, gray.rows, 1, gray.step);
            tess.SetRectangle(bounding_rect.x, bounding_rect.y, bounding_rect.width, bounding_rect.height);
            char *text = tess.GetUTF8Text();

            if (text != nullptr) {
                std::string recognized_text(text);
                // Log the recognized text
                LOGI("Recognized text: %s", recognized_text.c_str());
                delete[] text; // Giải phóng bộ nhớ sau khi sử dụng

                if (recognized_text.find(input_text_char) != std::string::npos) {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!found) {
                        result_text = "Location of input text: x = " + std::to_string(bounding_rect.x) +
                                      " y = " + std::to_string(bounding_rect.y);
                        found = true;
                    }
                }
            }
        }
        tess.End();
    };

    // Phân chia công việc cho các luồng
    int contours_per_thread = contours.size() / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        int start = i * contours_per_thread;
        int end = (i == num_threads - 1) ? contours.size() : (i + 1) * contours_per_thread;
        threads.emplace_back(process_contour, start, end);
    }

    // Chờ các luồng hoàn thành
    for (auto &thread : threads) {
        thread.join();
    }

    // Dọn dẹp
    AndroidBitmap_unlockPixels(env, bitmap);
    env->ReleaseStringUTFChars(input_text, input_text_char);
    env->ReleaseStringUTFChars(datapath, datapath_char);

    if (!found) {
        result_text = "Input text not found in the image.";
    }

    // Đo thời gian kết thúc
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
    LOGI("Processing time: %lld milliseconds", elapsed_time);

    // Trả kết quả dưới dạng chuỗi JNI
    return env->NewStringUTF(result_text.c_str());
}

}
