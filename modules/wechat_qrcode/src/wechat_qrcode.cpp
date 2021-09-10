// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#include "precomp.hpp"
#include "opencv2/wechat_qrcode.hpp"
#include "decodermgr.hpp"
#include "detector/align.hpp"
#include "detector/ssd_detector.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include "scale/super_scale.hpp"
#include "zxing/result.hpp"
namespace cv {
namespace wechat_qrcode {

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
    // the detector's input image.
    qrPoints = aligner.warpBack(qrPoints);
    return qrPoints;
}

/**
 * @brief Helper function to create a WeChatQRCodeResult out of a ZXing result, aligner and detector points
 *
 * @param result ZXing result
 * @param aligner aligner used to crop the image. If none where used use Align().
 * @param detectorPoints detector points associated to the ZXing result
 * @return a WeChatQRCodeResult object
 */
static WeChatQRCodeResult weChatQRCodeResultFromZXingResult(const zxing::Ref<zxing::Result>& result,
                                                            const Align& aligner,
                                                            const Mat& detectorPoints) {

    Mat qrPoints = unwrapsQrCorners(result, aligner);

    return {
        result->getText()->getText(),
        result->getRawBytes()->values(),
        qrPoints,
        detectorPoints,
        result->getCharset(),
        result->getQRCodeVersion(),
        result->getBinaryMethod(),
        result->getEcLevel(),
        result->getChartsetMode(),
        result->getDecodeScale(),
    };
}

WeChatQRCodeResult::WeChatQRCodeResult(
    const string &text, const vector<char> &rawBytes, const Mat &qrCorners,
    const Mat &detectorPoints, const string &charset, int qrcodeVersion,
    int binaryMethod, const string &ecLevel, const string &charsetMode, float decodeScale)
: text_(text), rawBytes_(rawBytes), qrCorners_(qrCorners),
  detectorPoints_(detectorPoints), charset_(charset),
  qrcodeVersion_(qrcodeVersion),
  binaryMethod_(binaryMethod), ecLevel_(ecLevel),
  charsetMode_(charsetMode), decodeScale_(decodeScale)
{}

const string &WeChatQRCodeResult::getText() const {
    return text_;
}

const vector<char>& WeChatQRCodeResult::getRawBytes() const {
    return rawBytes_;
}

Mat WeChatQRCodeResult::getQrCorners() const {
    return qrCorners_;
}

Mat WeChatQRCodeResult::getDetectorPoints() const {
    return detectorPoints_;
}

const string& WeChatQRCodeResult::getCharset() const {
    return charset_;
}

int WeChatQRCodeResult::getQrcodeVersion() const {
    return qrcodeVersion_;
}

int WeChatQRCodeResult::getBinaryMethod() const {
    return binaryMethod_;
}

const string& WeChatQRCodeResult::getEcLevel() const {
    return ecLevel_;
}

const string& WeChatQRCodeResult::getCharsetMode() const {
    return charsetMode_;
}

float WeChatQRCodeResult::getDecodeScale() const {
    return decodeScale_;
}

    class WeChatQRCode::Impl {
public:
    Impl() {}
    ~Impl() {}
    /**
     * @brief detect QR codes from the given image
     *
     * @param img supports grayscale or color (BGR) image.
     * @return vector<Mat> detected QR code bounding boxes.
     */
    std::vector<Mat> detect(const Mat& img);
    /**
     * @brief decode QR codes from detected points
     *
     * @param img supports grayscale or color (BGR) image.
     * @param candidate_points detected points. we name it "candidate points" which means no
     * all the qrcode can be decoded.
     * @param points succussfully decoded qrcode with bounding box points.
     * @return vector<string>
     */
    std::vector<WeChatQRCodeResult> decode(const Mat& img, std::vector<Mat>& candidate_points);
    int applyDetector(const Mat& img, std::vector<Mat>& points);
    Mat cropObj(const Mat& img, const Mat& point, Align& aligner);
    std::vector<float> getScaleList(const int width, const int height);
    std::shared_ptr<SSDDetector> detector_;
    std::shared_ptr<SuperScale> super_resolution_model_;
    bool use_nn_detector_, use_nn_sr_;
};

WeChatQRCode::WeChatQRCode(const String& detector_prototxt_path,
                           const String& detector_caffe_model_path,
                           const String& super_resolution_prototxt_path,
                           const String& super_resolution_caffe_model_path) {
    p = makePtr<WeChatQRCode::Impl>();
    if (!detector_caffe_model_path.empty() && !detector_prototxt_path.empty()) {
        // initialize detector model (caffe)
        p->use_nn_detector_ = true;
        CV_Assert(utils::fs::exists(detector_prototxt_path));
        CV_Assert(utils::fs::exists(detector_caffe_model_path));
        p->detector_ = make_shared<SSDDetector>();
        auto ret = p->detector_->init(detector_prototxt_path, detector_caffe_model_path);
        CV_Assert(ret == 0);
    } else {
        p->use_nn_detector_ = false;
        p->detector_ = NULL;
    }
    // initialize super_resolution_model
    // it could also support non model weights by cubic resizing
    // so, we initialize it first.
    p->super_resolution_model_ = make_shared<SuperScale>();
    if (!super_resolution_prototxt_path.empty() && !super_resolution_caffe_model_path.empty()) {
        p->use_nn_sr_ = true;
        // initialize dnn model (caffe format)
        CV_Assert(utils::fs::exists(super_resolution_prototxt_path));
        CV_Assert(utils::fs::exists(super_resolution_caffe_model_path));
        auto ret = p->super_resolution_model_->init(super_resolution_prototxt_path,
                                                    super_resolution_caffe_model_path);
        CV_Assert(ret == 0);
    } else {
        p->use_nn_sr_ = false;
    }
}

vector<WeChatQRCodeResult> WeChatQRCode::detectAndDecodeFullOutput(InputArray img) {
    CV_Assert(!img.empty());
    CV_CheckDepthEQ(img.depth(), CV_8U, "");

    if (img.cols() <= 20 || img.rows() <= 20) {
        return vector<WeChatQRCodeResult>();  // image data is not enough for providing reliable results
    }
    Mat input_img;
    int incn = img.channels();
    CV_Check(incn, incn == 1 || incn == 3 || incn == 4, "");
    if (incn == 3 || incn == 4) {
        cvtColor(img, input_img, COLOR_BGR2GRAY);
    } else {
        input_img = img.getMat();
    }
    auto candidate_points = p->detect(input_img);
    auto results = p->decode(input_img, candidate_points);
    vector<WeChatQRCodeResult> ret;

    for (auto& res : results) {
        ret.push_back(res);
    }
    return ret;
}

vector<string> WeChatQRCode::detectAndDecode(InputArray img, OutputArrayOfArrays points) {
    auto results = detectAndDecodeFullOutput(img);
    vector<string> ret;

    // opencv type convert
    vector<Mat> tmp_points;
    for (auto& res : results) {
        Mat tmp_point;
        tmp_points.push_back(tmp_point);
        res.getDetectorPoints().convertTo(((OutputArray)tmp_points.back()), CV_32FC2);
        ret.push_back(res.getText());
    }

    points.createSameSize(tmp_points, CV_32FC2);
    points.assign(tmp_points);

    return ret;
}

vector<WeChatQRCodeResult> WeChatQRCode::Impl::decode(const Mat& img, vector<Mat>& candidate_points) {
    if (candidate_points.empty()) {
        return {};
    }
    vector<WeChatQRCodeResult> qr_results;
    for (auto& point : candidate_points) {
        Mat cropped_img;
        Align aligner;
        if (use_nn_detector_) {
            cropped_img = cropObj(img, point, aligner);
        } else {
            cropped_img = img;
        }
        // scale_list contains different scale ratios
        auto scale_list = getScaleList(cropped_img.cols, cropped_img.rows);
        for (auto cur_scale : scale_list) {
            Mat scaled_img =
                super_resolution_model_->processImageScale(cropped_img, cur_scale, use_nn_sr_);
            zxing::Ref<zxing::Result> result;
            DecoderMgr decodemgr;
            auto ret = decodemgr.decodeImage(scaled_img, use_nn_detector_, result);

            if (ret == 0) {
                result->setDecodeScale(cur_scale);
                qr_results.push_back(weChatQRCodeResultFromZXingResult(result, aligner, point));
                break;
            }
        }
    }

    return qr_results;
}

vector<Mat> WeChatQRCode::Impl::detect(const Mat& img) {
    auto points = vector<Mat>();

    if (use_nn_detector_) {
        // use cnn detector
        auto ret = applyDetector(img, points);
        CV_Assert(ret == 0);
    } else {
        auto width = img.cols, height = img.rows;
        // if there is no detector, use the full image as input
        auto point = Mat(4, 2, CV_32FC1);
        point.at<float>(0, 0) = 0;
        point.at<float>(0, 1) = 0;
        point.at<float>(1, 0) = width - 1;
        point.at<float>(1, 1) = 0;
        point.at<float>(2, 0) = width - 1;
        point.at<float>(2, 1) = height - 1;
        point.at<float>(3, 0) = 0;
        point.at<float>(3, 1) = height - 1;
        points.push_back(point);
    }
    return points;
}

int WeChatQRCode::Impl::applyDetector(const Mat& img, vector<Mat>& points) {
    int img_w = img.cols;
    int img_h = img.rows;

    // hard code input size
    int minInputSize = 400;
    float resizeRatio = sqrt(img_w * img_h * 1.0 / (minInputSize * minInputSize));
    int detect_width = img_w / resizeRatio;
    int detect_height = img_h / resizeRatio;

    points = detector_->forward(img, detect_width, detect_height);

    float x0 = points[0].at<float>(0, 0);
    float y0 = points[0].at<float>(0, 1);
    float x1 = points[0].at<float>(1, 0);
    float y1 = points[0].at<float>(1, 1);
    float x2 = points[0].at<float>(2, 0);
    float y2 = points[0].at<float>(2, 1);
    float x3 = points[0].at<float>(3, 0);
    float y3 = points[0].at<float>(3, 1);

    return 0;
}

Mat WeChatQRCode::Impl::cropObj(const Mat& img, const Mat& point, Align& aligner) {
    // make some padding to boost the qrcode details recall.
    float padding_w = 0.1f, padding_h = 0.1f;
    auto min_padding = 15;
    auto cropped = aligner.crop(img, point, padding_w, padding_h, min_padding);
    return cropped;
}

// empirical rules
vector<float> WeChatQRCode::Impl::getScaleList(const int width, const int height) {
    if (width < 320 || height < 320) return {1.0, 2.0, 0.5};
    if (width < 640 && height < 640) return {1.0, 0.5};
    return {0.5, 1.0};
}
}  // namespace wechat_qrcode
}  // namespace cv