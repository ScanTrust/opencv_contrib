// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// This code is based on Tencent WeChat QR Code reader provided in opencv contrib
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#include "precomp.hpp"
#include "opencv2/scantrust_qrcode.hpp"
#include "decodermgr.hpp"
#include "align.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace cv {
namespace scantrust_qrcode {

static Mat unwrapsQrCorners(const Ref<zxing::Result> &result, const Align &aligner) {
    Mat qrPoints(result->getResultPoints()->size() , 2, CV_32FC1);   // result.getResultPoints()

    float scaleFactor = result->getDecodeScale();

    // Apply inverse scale factor on qr corners
    // to compensate scaling of the detector's input image
    int i = 0;
    for(auto& p : result->getResultPoints()->values()) {
        qrPoints.at<float>(i, 0) = p->getX() / scaleFactor;
        qrPoints.at<float>(i, 1) = p->getY() / scaleFactor;
        ++i;
    }

    // use the aligner to undo crop or rotation applied on the
    // detector's input image.
    qrPoints = aligner.warpBack(qrPoints);
    return qrPoints;
}

static vector<Point2f> getQrPatternsCenter(const Mat &qrPoints, float qr_cells) {
    vector<Point2f> templateCorners = {
            {0.,         qr_cells},
            {0.,         0.},
            {qr_cells, 0.},
            {qr_cells, qr_cells}
    };

    vector<Point2f> templatePatternsCenter = {
            {3.5,              qr_cells - 3.5f},
            {3.5,              3.5},
            {qr_cells - 3.5f, 3.5},
            {qr_cells - 6.5f, qr_cells - 6.5f}
    };

    Mat templatePatternsCenter_{templatePatternsCenter};

    Mat H = getPerspectiveTransform(qrPoints, Mat(templateCorners));

    vector<Point2f> qrPattersCenter;
    perspectiveTransform(templatePatternsCenter, qrPattersCenter, H.inv());
    return qrPattersCenter;
}

/**
 * @brief Helper function to create a ScantrustQRCodeResult out of a ZXing result, aligner and detector points
 *
 * @param result ZXing result
 * @param aligner aligner used to crop the image. If none where used use Align().
 * @param detectorPoints detector points associated to the ZXing result
 * @return a ScantrustQRCodeResult object
 */
static ScantrustQRCodeResult scantrustQRCodeResultFromZXingResult(const zxing::Ref<zxing::Result>& result,
                                                            const Align& aligner) {

    Mat qrPoints = unwrapsQrCorners(result, aligner);
    
    int qr_cells = (result->getQRCodeVersion() * 4) + 17;

        vector<Point2f> qrPattersCenter = getQrPatternsCenter(qrPoints, qr_cells);

        return {
        result->getText()->getText(),
        result->getRawBytes()->values(),
        qrPoints,
        Mat(4 , 2, CV_32FC1, qrPattersCenter.data()),
        result->getCharset(),
        result->getQRCodeVersion(),
        qr_cells,
        result->getBinaryMethod(),
        result->getEcLevel(),
        result->getChartsetMode(),
        result->getDecodeScale(),
    };
}

    ScantrustQRCodeResult::ScantrustQRCodeResult(const std::string &text, const std::vector<char> &rawBytes, const Mat &qrCorners, Mat qrPatternsCenter,
                                                 const std::string &charset, int qrcodeVersion, int qrcodeCells, int binaryMethod, const std::string &ecLevel,
                                                 const std::string &charsetMode, float decodeScale)
: text_(text), rawBytes_(rawBytes), qrCorners_(qrCorners),
  qrPatternsCenter_(qrPatternsCenter), charset_(charset),
  qrcodeVersion_(qrcodeVersion), qrcodeCells_(qrcodeCells), binaryMethod_(binaryMethod),
  ecLevel_(ecLevel), charsetMode_(charsetMode), decodeScale_(decodeScale)
{}

const string &ScantrustQRCodeResult::getText() const {
    return text_;
}

const vector<char>& ScantrustQRCodeResult::getRawBytes() const {
    return rawBytes_;
}

Mat ScantrustQRCodeResult::getQrCorners() const {
    return qrCorners_;
}

Mat ScantrustQRCodeResult::getQrPatternsCenter() const {
    return qrPatternsCenter_;
}

const string& ScantrustQRCodeResult::getCharset() const {
    return charset_;
}

int ScantrustQRCodeResult::getQrcodeVersion() const {
    return qrcodeVersion_;
}

int ScantrustQRCodeResult::getQrCells() const {
    return qrcodeCells_;
}

int ScantrustQRCodeResult::getBinaryMethod() const {
    return binaryMethod_;
}

const string& ScantrustQRCodeResult::getEcLevel() const {
    return ecLevel_;
}

const string& ScantrustQRCodeResult::getCharsetMode() const {
    return charsetMode_;
}

float ScantrustQRCodeResult::getDecodeScale() const {
    return decodeScale_;
}

class ScantrustQRCode::Impl {
public:
    DownscalingRules m_downscalingRules;

    Impl(const DownscalingRules& downscalingRules):m_downscalingRules(downscalingRules) {}
    ~Impl() = default;
    /**
     * @brief decode QR codes
     *
     * @param img supports grayscale or color (BGR) image.
     * @return vector<string>
     */


    std::vector<ScantrustQRCodeResult> decode(const Mat& img);
    std::vector<float> getDownscaleList(int width, int height);
};

ScantrustQRCode::ScantrustQRCode()
{

    DownscalingRules downscale_rules = {
        {4500, {16.0, 36.0, 22.0, 18.0, 14.0, 6.0, 4.0}},
        {3000, {18.0, 12.0, 22.0, 16.0, 14.0, 10.0, 8.0, 6.0, 4.0}},
        {1500, {26.0, 6.0, 4.0, 2.0}},
        {0, {2.0, 1.0}}
    };
    p = makePtr<ScantrustQRCode::Impl>(downscale_rules);
}

ScantrustQRCode::ScantrustQRCode(const DownscalingRules& downscalingRules)
{
    p = makePtr<ScantrustQRCode::Impl>(downscalingRules);
}


vector<ScantrustQRCodeResult> ScantrustQRCode::detectAndDecode(InputArray img) {
    CV_Assert(!img.empty());
    CV_CheckDepthEQ(img.depth(), CV_8U, "");

    if (img.cols() <= 20 || img.rows() <= 20) {
        return {};  // image data is not enough for providing reliable results
    }
    Mat input_img;
    int incn = img.channels();
    CV_Check(incn, incn == 1 || incn == 3 || incn == 4, "");
    if (incn == 3 || incn == 4) {
        cvtColor(img, input_img, COLOR_BGR2GRAY);
    } else {
        input_img = img.getMat();
    }
    auto results = p->decode(input_img);
    vector<ScantrustQRCodeResult> ret;

    for (const auto& res : results) {
        ret.push_back(res);
    }
    return ret;
}

vector<ScantrustQRCodeResult> ScantrustQRCode::Impl::decode(const Mat& img) {
    vector<ScantrustQRCodeResult> qr_results;
    Align aligner;
    Mat scaled_img;
    // scale_list contains different scale ratios
    auto downscale_list = getDownscaleList(img.cols, img.rows);
    for (auto cur_downscale : downscale_list) {
        if( cur_downscale != 1) {
            float h_cur_downscale  = 2.f / cur_downscale;
            Mat tmp;
            // first apply a `x(downscale / 2)` NEAREST downscale
            if (h_cur_downscale != 1) {
                resize(img, tmp, Size(), h_cur_downscale, h_cur_downscale, INTER_NEAREST);
            } else {
                tmp = img;
            }
            // then apply a `x2` AREA downscale
            resize(tmp, scaled_img, Size(), 0.5, 0.5, INTER_AREA);
        } else {
            scaled_img = img;
        }
        zxing::Ref<zxing::Result> result;
        scantrust_qrcode::DecoderMgr decodemgr;
        auto ret = decodemgr.decodeImage(scaled_img, result);

        if (ret == 0) {
            result->setDecodeScale(1.f / cur_downscale);
            qr_results.push_back(scantrustQRCodeResultFromZXingResult(result, aligner));
            break;
        }
    }

    return qr_results;
}

DownscalingRule::DownscalingRule():
lower_size_limit(), downscaling_factor_sequence()
{}

DownscalingRule::DownscalingRule(int lower_size_limit, vector<float> downscaling_factor_sequence):
lower_size_limit(lower_size_limit), downscaling_factor_sequence(std::move(downscaling_factor_sequence))
{}

vector<float> ScantrustQRCode::Impl::getDownscaleList(int width, int height) {
    unsigned int average_width = (width + height) / 2;

    for (const auto& rule : m_downscalingRules) {
        if( average_width > rule.lower_size_limit ) {
            return rule.downscaling_factor_sequence;
        }
    }
    return m_downscalingRules[m_downscalingRules.size() - 1].downscaling_factor_sequence;
}

}  // namespace scantrust_qrcode
}  // namespace cv