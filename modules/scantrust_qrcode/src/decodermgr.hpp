// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __OPENCV_SCANTRUST_QRCODE_DECODERMGR_HPP__
#define __OPENCV_SCANTRUST_QRCODE_DECODERMGR_HPP__

// zxing
#include "../../wechat_qrcode/src/zxing/binarizer.hpp"
#include "../../wechat_qrcode/src/zxing/binarybitmap.hpp"
#include "../../wechat_qrcode/src/zxing/decodehints.hpp"
#include "../../wechat_qrcode/src/zxing/qrcode/qrcode_reader.hpp"
#include "../../wechat_qrcode/src/zxing/result.hpp"

// qbar
#include "../../wechat_qrcode/src/binarizermgr.hpp"
#include "../../wechat_qrcode/src/imgsource.hpp"
namespace cv {
namespace scantrust_qrcode {

class DecoderMgr {
public:
    DecoderMgr() { reader_ = new zxing::qrcode::QRCodeReader(); };
    ~DecoderMgr(){};

    int decodeImage(cv::Mat src, zxing::Ref<zxing::Result>& );

private:
    zxing::Ref<zxing::UnicomBlock> qbarUicomBlock_;
    zxing::DecodeHints decode_hints_;

    zxing::Ref<zxing::qrcode::QRCodeReader> reader_;
    wechat_qrcode::BinarizerMgr binarizer_mgr_;

    zxing::Ref<zxing::Result> Decode(zxing::Ref<zxing::BinaryBitmap> image,
                                     zxing::DecodeHints hints);

    int TryDecode(zxing::Ref<zxing::LuminanceSource> source, zxing::Ref<zxing::Result>& result);
};

}  // namespace scantrust_qrcode
}  // namespace cv
#endif  // __OPENCV_SCANTRUST_QRCODE_DECODERMGR_HPP__
