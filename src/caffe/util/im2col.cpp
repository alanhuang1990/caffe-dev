#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


namespace caffe {
void Mypause(){
	char c;
	while((c= getchar() !='\n'));

}

template <typename Dtype>
/**
 *bit -> row
 *multi row  -> to a map in one channel
 *multi channel -> data_col
 *
 *[im_channel][im_h][im_w]
 *
 */
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

// get a patch from data_col
template<typename Dtype>
void get_patch_from_im2col(const Dtype* data_col, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const int col_c_idx,const int col_h_idx,const int col_w_idx,
	    double* res_img
	    )
{
	  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	  int channels_col = channels * kernel_h * kernel_w;
	  if(height_col <= col_h_idx || width_col <= col_w_idx || col_c_idx >= channels)
	  {
		  printf("invalid col_h_idx or col_w_idx or col_c_idx\n");
		  return;
	  }
	  int c_start = col_c_idx*kernel_h * kernel_w;
	  int c_end = (col_c_idx+1)*kernel_h * kernel_w;
	  for(int idx =c_start;idx<c_end;idx++)
	  {
		  int buf_col_idx = (idx*height_col+col_h_idx)*width_col+col_w_idx;
		  res_img[idx - c_start] = data_col[buf_col_idx];
	  }
}
template <typename Dtype>
void generate_sample_img(const int channels, const int height, const int width,Dtype * data_res)
{
	for(int i=0;i<channels*height*width;++i)
	{
		data_res[i] = i;
	}
}

// an implementation of im2col_cpu adding scale transformation

void print_cvmat(const cv::Mat& mat)
{
	for(int h=0;h<mat.rows;h++)
	{
		for(int w=0;w<mat.cols;w++)
		{
			printf("%6.2lf ",(*(double*)mat.ptr(h,w)));
		}
		printf("\n");
	}
}

// cur_scale must be positive.
template <typename Dtype>
void scale_im2col_cpu(const Dtype* data_im, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w, const float cur_scale,
	    Dtype* data_col)
{
	// calculate the new pad and kernel information
	int pad_h_to_add = (floor(kernel_h*(pow(2,cur_scale)-1)))/2;
	int pad_w_to_add = (floor(kernel_w*(pow(2,cur_scale)-1)))/2;
	int new_kernel_h = pad_h_to_add*2 + kernel_h;
	int new_kernel_w = pad_w_to_add*2 + kernel_w;
	int height_col = (height + 2 * (pad_h + pad_h_to_add) - new_kernel_h) / stride_h + 1;
	int width_col = (width + 2 * (pad_w + pad_h_to_add) - new_kernel_w) / stride_w + 1;
	int channels_col = channels * new_kernel_h * new_kernel_w;

	if((new_kernel_h == kernel_h) &&(new_kernel_w ==  new_kernel_w))
	{
		im2col_cpu(data_im,channels,height,width,new_kernel_h,new_kernel_w,pad_h+pad_h_to_add,
					  pad_w + pad_w_to_add,stride_h,stride_w,data_col);
		return;
	}

//	printf("pad_h = %d     pad_w = %d    kernel_h = %d    kernel_w = %d  scale = %f \n",
//			pad_h,pad_w,kernel_h,kernel_w,cur_scale);
//	printf("new_pad_h = %d  new_pad_w = %d  new_kernel_h = %d  new_kernel_w = %d \n",
//			pad_h_to_add+pad_h, pad_w+pad_w_to_add,new_kernel_h, new_kernel_w);

	Dtype * buf_col = new Dtype[sizeof(Dtype)*height_col*width_col*channels_col];
	// im2col for new parameters
	im2col_cpu(data_im,channels,height,width,new_kernel_h,new_kernel_w,pad_h+pad_h_to_add,
			  pad_w + pad_w_to_add,stride_h,stride_w,buf_col);
	//[a_chanel][b_height][c_width] = (a*height+b)*width + c
	cv::Mat temp_col_mat = cvCreateMat(new_kernel_h,new_kernel_w,CV_64FC1);
	cv::Mat temp_col_dest_mat  =  cvCreateMat(kernel_h,kernel_w,CV_64FC1);
	for(int i=0;i<height_col;i++)
	{
		for(int j=0;j<width_col;j++)
		{
			for(int c=0; c<channels;c++)
			{
				int c_start = c*new_kernel_h * new_kernel_w;
				int c_end = (c+1)*new_kernel_h * new_kernel_w;
				int buf_col_idx = (c_start*height_col+i)*width_col+j;
				int c_step = height_col*width_col;
				double *m_ptr =(double *)temp_col_mat.ptr(0,0);
				for(int cc = c_start; cc<c_end;cc++ )
				{
					*m_ptr = static_cast<double>(buf_col[buf_col_idx]);
					++m_ptr;
					buf_col_idx += c_step;
				}

//				printf("the large pathes: \n");
//				print_cvmat(temp_col_mat);

				// resize the col vectors and store back to data_col
				cv::resize(temp_col_mat,temp_col_dest_mat,cv::Size(kernel_h,kernel_w),cv::INTER_LINEAR);
//				printf("the resized patch: \n");
//				print_cvmat(temp_col_dest_mat);
//				printf("\n\n");
//				Mypause();
				c_start = c*kernel_h*kernel_w;
				c_end = (c+1)*kernel_h*kernel_w;
				buf_col_idx = (c_start*height_col+i)*width_col+j;
				m_ptr =(double *)temp_col_dest_mat.ptr(0,0);
				for(int cc=c_start;cc<c_end;cc++)
				{
					data_col[buf_col_idx] =static_cast<Dtype>(*m_ptr);
					++m_ptr;
					buf_col_idx += c_step;
				}
			}
		}
	}
	delete [] buf_col ;
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col);

// added by alan
template void scale_im2col_cpu<float>(const float* data_im,const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h,
	const int stride_w, const float cur_scale, float* data_col );
template void scale_im2col_cpu<double>(const double* data_im,const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h,
	const int stride_w, const float cur_scale, double* data_col );

template void generate_sample_img<double>(const int channels,
		const int height, const int width,double * data_res);
template void get_patch_from_im2col(const double* data_col, const int channels,
	    const int height, const int width, const int kernel_h, const int kernel_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const int col_c_idx,const int col_h_idx,const int col_w_idx,
	    double* res_img
	    );



template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % patch_w;
    int h_offset = (c / patch_w) % patch_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

template<typename Dtype>
void scale_col2im_cpu(const Dtype* data_col, const int channels,
	    const int height, const int width, const int patch_h, const int patch_w,
	    const int pad_h, const int pad_w,
	    const int stride_h, const int stride_w,const float cur_scale,
	    Dtype* data_im)
{
	// calculate the new pad and kernel information
	int pad_h_to_add = (floor(patch_h*(pow(2,cur_scale)-1)))/2;
	int pad_w_to_add = (floor(patch_w*(pow(2,cur_scale)-1)))/2;
	int new_kernel_h = pad_h_to_add*2 + patch_h;
	int new_kernel_w = pad_w_to_add*2 + patch_w;
	int height_col = (height + 2 * (pad_h + pad_h_to_add) - new_kernel_h) / stride_h + 1;
	int width_col = (width + 2 * (pad_w + pad_h_to_add) - new_kernel_w) / stride_w + 1;
	int channels_col = channels * new_kernel_h * new_kernel_w;


	if((new_kernel_h == patch_h) && (new_kernel_w == patch_w))
	{
		col2im_cpu( data_col,channels,height, width, new_kernel_h, new_kernel_w,
					pad_h+pad_h_to_add, pad_w+pad_w_to_add,stride_h, stride_w,  data_im);
		return;
	}

	Dtype * buf_col = new Dtype[sizeof(Dtype)*height_col*width_col*channels_col];
	cv::Mat temp_col_source_mat = cvCreateMat(patch_h,patch_w,CV_64FC1);
	cv::Mat temp_col_dest_mat  =  cvCreateMat(new_kernel_h,new_kernel_w,CV_64FC1);

//	printf("pad_h = %d     pad_w = %d    kernel_h = %d    kernel_w = %d  scale = %f \n",
//			pad_h,pad_w,patch_h,patch_w,cur_scale);
//	printf("new_pad_h = %d  new_pad_w = %d  new_kernel_h = %d  new_kernel_w = %d \n",
//			pad_h_to_add+pad_h, pad_w+pad_w_to_add,new_kernel_h, new_kernel_w);
//

	for(int i = 0;i<height_col;++i)
	{
		for(int j=0;j<width_col;j++)
		{
			for(int c=0; c<channels;c++)
			{
				int c_start = c*patch_h*patch_w;
				int c_end = (c+1)*patch_h*patch_w;
				int buf_col_idx = (c_start*height_col+i)*width_col+j;
				int c_step = height_col*width_col;
				double *m_ptr =(double *)temp_col_source_mat.ptr(0,0);
				for(int cc = c_start; cc<c_end;cc++ )
				{
					*m_ptr = static_cast<double>(data_col[buf_col_idx]);
					++m_ptr;
					buf_col_idx += c_step;
				}


				cv::resize(temp_col_source_mat,temp_col_dest_mat,cv::Size(new_kernel_h,new_kernel_w),cv::INTER_LINEAR);

//				printf("the small pathes: \n");
//				print_cvmat(temp_col_source_mat);
//				printf("the resized patch: \n");
//				print_cvmat(temp_col_dest_mat);
//				printf("\n\n");
//				Mypause();
				c_start = c*new_kernel_h*new_kernel_w;
				c_end = (c+1)*new_kernel_h*new_kernel_w;
				buf_col_idx = (c_start*height_col+i)*width_col+j;
				m_ptr =(double *)temp_col_dest_mat.ptr(0,0);
				for(int cc=c_start;cc<c_end;cc++)
				{
					buf_col[buf_col_idx] =static_cast<Dtype>(*m_ptr);
					++m_ptr;
					buf_col_idx += c_step;
				}

			}
		}
	}

	col2im_cpu( buf_col,channels,height, width, new_kernel_h, new_kernel_w,
			pad_h+pad_h_to_add, pad_w+pad_w_to_add,stride_h, stride_w,  data_im);
	delete [] buf_col;

}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);

// added by alan
template void scale_col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const float cur_scale,float* data_im);
template void scale_col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const float cur_scale, double* data_im);
}  // namespace caffe
