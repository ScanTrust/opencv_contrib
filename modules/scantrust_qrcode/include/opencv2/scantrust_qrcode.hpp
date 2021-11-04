// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __OPENCV_SCANTRUST_QRCODE_HPP__
#define __OPENCV_SCANTRUST_QRCODE_HPP__
#include "opencv2/core.hpp"
/** @defgroup wechat_qrcode WeChat QR code detector for detecting and parsing QR code.
 */
namespace cv {
namespace scantrust_qrcode {


//! @addtogroup scantrust_qrcode
//! @{
/**
 * @brief ScantrustQRCodeResult includes information about detected instance of QR codes.
 *
 */
class CV_EXPORTS_W_SIMPLE ScantrustQRCodeResult
{

public:
    /**
     * @brief Initialize a ScantrustQRCodeResult.
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
    CV_WRAP ScantrustQRCodeResult(const std::string &text, const std::vector<char> &rawBytes,
                                  const std::vector<Point2f>& qrCorners, const std::vector<Point2f>& qrPatternsCenter,
                                  const std::string &charset, int qrcodeVersion, int qrcodeCells,
                                  int binaryMethod, const std::string &ecLevel, const std::string &charsetMode, float decodeScale);

    CV_WRAP const std::string& getText() const;
    CV_WRAP const std::vector<char>& getRawBytes() const;
    CV_WRAP const std::vector<Point2f>& getQrCorners() const;
    CV_WRAP const std::vector<Point2f>& getQrPatternsCenter() const;
    CV_WRAP const std::string& getCharset() const;
    CV_WRAP int getQrcodeVersion() const;
    CV_WRAP int getQrCells() const;
    CV_WRAP int getBinaryMethod() const;
    CV_WRAP const std::string& getEcLevel() const;
    CV_WRAP const std::string& getCharsetMode() const;
    CV_WRAP float getDecodeScale() const;

    ~ScantrustQRCodeResult() = default;

private:
    std::string text_;
    std::vector<char> rawBytes_;
    std::vector<Point2f> qrCorners_;
    std::vector<Point2f> qrPatternsCenter_;
    Mat detectorPoints_;
    std::string charset_;
    int qrcodeVersion_;
    int qrcodeCells_;
    int binaryMethod_;
    std::string ecLevel_;
    std::string charsetMode_;
    float decodeScale_;
};
//! @}


class CV_EXPORTS_W_SIMPLE DownscalingRule {
public:
    DownscalingRule();
    CV_WRAP DownscalingRule(int lower_size_limit, std::vector<float> downscaling_factor_sequence);
    unsigned int lower_size_limit;
    std::vector<float> downscaling_factor_sequence;
};

typedef std::vector<DownscalingRule> DownscalingRules;

//! @addtogroup scantrust_qrcode
//! @{
/**
 * @brief  Scantrust QR Reader is a based on WeeChat QR reader but drops the two
 * CNN models and only apply the improved zxing reader to read the codes. This reader
 * also tries a wider range of downscaling factor (no up scaling is done) that we optimally
 * selected after benchmarking each of them on a wide set of qr images. This reader is optimized
 * to work fast on images taken from mobile where no constraints is apply on the size nor of the
 * position of the QR code in the image.
 *
 */
class CV_EXPORTS_W ScantrustQRCode {
public:
    /**
     * @brief Initialize the ScantrustQRCode.
     * It includes two models, which are packaged with caffe format.
     * Therefore, there are prototxt and caffe models (In total, four paramenters).
     *
     * @param detector_prototxt_path prototxt file path for the detector

     */
    CV_WRAP ScantrustQRCode();
    CV_WRAP ScantrustQRCode(const DownscalingRules& downscalingRules);
    ~ScantrustQRCode(){};

    /**
     * @brief  Both detects and decodes QR code, returns a list of decoded QR.
     * To simplify the usage, there is only two API: detectAndDecode, detectAndDecodeFullOutput
     *
     * @param img supports grayscale or color (BGR) image.
     * @return list of QR.
     */
    CV_WRAP std::vector<ScantrustQRCodeResult> detectAndDecode(InputArray img);

protected:
    const std::vector<std::pair<int, std::vector<float>>> m_downscalingRules;
    class Impl;
    Ptr<Impl> p;
};

//! @}
}  // namespace wechat_qrcode
}  // namespace cv
#endif  // __OPENCV_WECHAT_QRCODE_HPP__
