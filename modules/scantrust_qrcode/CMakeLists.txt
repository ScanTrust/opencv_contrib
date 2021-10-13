set(the_description "Scantrust QR code Detector")
ocv_define_module(scantrust_qrcode opencv_core opencv_imgproc opencv_dnn opencv_wechat_qrcode WRAP java objc python js)

# iconv support isn't automatic on some systems
if(CMAKE_VERSION VERSION_GREATER "3.11")
  find_package(Iconv QUIET)
  if(Iconv_FOUND)
    ocv_target_link_libraries(${the_module} Iconv::Iconv)
  else()
    ocv_target_compile_definitions(${the_module} PRIVATE "NO_ICONV=1")
  endif()
endif()
