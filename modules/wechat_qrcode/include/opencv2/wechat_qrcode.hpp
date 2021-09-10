// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __OPENCV_WECHAT_QRCODE_HPP__
#define __OPENCV_WECHAT_QRCODE_HPP__
#include "opencv2/core.hpp"
/** @defgroup wechat_qrcode WeChat QR code detector for detecting and parsing QR code.
 */
namespace cv {
namespace wechat_qrcode {


//! @addtogroup wechat_qrcode
//! @{
/**
 * @brief WeChatQRCodeResult includes information about detected instance of QR codes.
 *
 */
class CV_EXPORTS_W WeChatQRCodeResult
{

public:
    /**
     * @brief Initialize a WeChatQRCodeResult.
     *
     * @param text decoded text
     * @param rawBytes binary data
     * @param qrCorners locations of QR four corners, starting from the bottom left one.
     * @param detectorPoints corners of the rectangle containing the QR
     * @param charset charset of the encoded string
     * @param qrcodeVersion version of the QR codes
     * @param binaryMethod binarising method used to detect the QR
     * @param ecLevel error correction level of the QR (L, M, Q, H)
     * @param charsetMode
     * @param decodeScale image scaling factor used to detect the QR
     */
    CV_WRAP WeChatQRCodeResult(const std::string &text, const std::vector<char> &rawBytes,
                               const Mat &qrCorners, const Mat &detectorPoints,
                               const std::string &charset, int qrcodeVersion,
                               int binaryMethod, const std::string &ecLevel,
                               const std::string &charsetMode, float decodeScale);

    CV_WRAP const std::string& getText() const;
    CV_WRAP const std::vector<char>& getRawBytes() const;
    CV_WRAP Mat getQrCorners() const;
    CV_WRAP Mat getDetectorPoints() const;
    CV_WRAP const std::string& getCharset() const;
    CV_WRAP int getQrcodeVersion() const;
    CV_WRAP int getBinaryMethod() const;
    CV_WRAP const std::string& getEcLevel() const;
    CV_WRAP const std::string& getCharsetMode() const;
    CV_WRAP float getDecodeScale() const;

    ~WeChatQRCodeResult() = default;

private:
    std::string text_;
    std::vector<char> rawBytes_;
    Mat qrCorners_;
    Mat detectorPoints_;
    std::string charset_;
    int qrcodeVersion_;
    int binaryMethod_;
    std::string ecLevel_;
    std::string charsetMode_;
    float decodeScale_;
};
//! @}


//! @addtogroup wechat_qrcode
//! @{
/**
 * @brief  WeChat QRCode includes two CNN-based models:
 * A object detection model and a super resolution model.
 * Object detection model is applied to detect QRCode with the bounding box.
 * super resolution model is applied to zoom in QRCode when it is small.
 *
 */
class CV_EXPORTS_W WeChatQRCode {
public:
    /**
     * @brief Initialize the WeChatQRCode.
     * It includes two models, which are packaged with caffe format.
     * Therefore, there are prototxt and caffe models (In total, four paramenters).
     *
     * @param detector_prototxt_path prototxt file path for the detector
     * @param detector_caffe_model_path caffe model file path for the detector
     * @param super_resolution_prototxt_path prototxt file path for the super resolution model
     * @param super_resolution_caffe_model_path caffe file path for the super resolution model
     */
    CV_WRAP WeChatQRCode(const std::string& detector_prototxt_path = "",
                         const std::string& detector_caffe_model_path = "",
                         const std::string& super_resolution_prototxt_path = "",
                         const std::string& super_resolution_caffe_model_path = "");
    ~WeChatQRCode(){};

    /**
     * @brief  Both detects and decodes QR code, returns only a list of decoded string.
     * To simplify the usage, there is only two API: detectAndDecode, detectAndDecodeFullOutput
     *
     * @param img supports grayscale or color (BGR) image.
     * @param points optional output array of vertices of the found QR code quadrangle. Will be
     * empty if not found.
     * @return list of decoded string.
     */
    CV_WRAP std::vector<std::string> detectAndDecode(InputArray img,
                                                     OutputArrayOfArrays points = noArray());

    /**
     * @brief  Both detects and decodes QR code, returns a list of decoded QR.
     * To simplify the usage, there is only two API: detectAndDecode, detectAndDecodeFullOutput
     *
     * @param img supports grayscale or color (BGR) image.
     * @return list of QR.
     */
    CV_WRAP std::vector<WeChatQRCodeResult> detectAndDecodeFullOutput(InputArray img);

protected:
    class Impl;
    Ptr<Impl> p;
};

//! @}
}  // namespace wechat_qrcode
}  // namespace cv
#endif  // __OPENCV_WECHAT_QRCODE_HPP__
