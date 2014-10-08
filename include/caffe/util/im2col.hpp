#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

namespace caffe {

void Mypause();

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im);


// added by alan
template<typename Dtype>
void get_patch_from_im2col(const Dtype * data_col, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const int col_c_idx,const int col_h_idx,const int col_w_idx,
	    double* res_img
	    );

template <typename Dtype>
void generate_sample_img(const int channels, const int height, const int width,Dtype * data_res);

template <typename Dtype>
void scale_im2col_cpu(const Dtype* data_im, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w, const float cur_scale,
	    Dtype* data_col);

template<typename Dtype>
void scale_col2im_cpu(const Dtype* data_col, const int channels,
	    const int height, const int width, const int patch_h, const int patch_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const float cur_scale,
	    Dtype* data_im);

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
