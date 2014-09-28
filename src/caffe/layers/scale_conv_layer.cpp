#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


/**
 * scale conv layer. added by alan
 * use scale_start and scale_end to control patch scale.
 *
 */
namespace caffe {

template <typename Dtype>
void ScaleConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
	ConvolutionLayer<Dtype>::LayerSetUp( bottom, top);
	ScaleConvolutionParameter scale_conv_param = this->layer_param_.scale_convolution_param();
	if(scale_conv_param.has_scale_start())
	{
		this->scale_start = scale_conv_param.scale_start();
	}
	if(scale_conv_param.has_scale_end())
	{
		this->scale_end = scale_conv_param.scale_end();
	}

	CHECK_GT(this->scale_end > this->scale_start,0) << "scale_start is larger than scale_start .";
}

template <typename Dtype>
void ScaleConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top)
{
	if(Caffe::phase() == Caffe::TRAIN)
	{
	this->cur_scale = this->scale_start +
			static_cast<float>(rand() / double(RAND_MAX))*(this->scale_end - this->scale_start);
	}
	else
	{
		this->cur_scale = 0;
	}
	for(int i = 0; i<bottom.size(); ++i)
	{
		const Dtype* bottom_data = bottom[i]->cpu_data();
		Dtype* top_data = top[i]->mutable_cpu_data();
		Dtype* col_buff = NULL;
		if(!this->is_1x1_)
		{
			col_buff = this->col_buffer_.mutable_cpu_data();
		}
		const Dtype* weight = this->blobs_[0]->cpu_data();
	    int weight_offset = this->M_ * this->K_;  // number of filter parameters in a group
	    int col_offset = this->K_ * this->N_;  // number of values in an input region / column
	    int top_offset = this->M_ * this->N_;  // number of values in an output region / column
		for(int n = 0; n < this->num_ ; ++n)
		{
			// im2col transformation: unroll input regions for filtering
			// into column matrix for multplication.
			// using random scale from scale_start to scale_end
			if(! this->is_1x1_)
			{
				scale_im2col_cpu(bottom_data + bottom[i]->offset(n), this->channels_,this->height_,this->width_,
									this->kernel_h_,this->kernel_w_,this->pad_h_,this->pad_w_,
									this->stride_h_,this->stride_w_,this->cur_scale,col_buff);
			}
			else
			{
				col_buff  = bottom[i]->mutable_cpu_data() + bottom[i]->offset(n);
			}
			// Take inner products for groups.
			for (int g = 0; g< this->group_ ; ++g)
			{
				caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans, this->M_, this->N_, this->K_,
						(Dtype)1., weight+weight_offset*g, col_buff+col_offset*g,
						(Dtype)0., top_data+top[i]->offset(n) + top_offset*g);
			}
			//add bias
			if(this->bias_term_)
			{
				caffe_cpu_gemm<Dtype> (CblasNoTrans, CblasNoTrans, this->num_output_,
						this->N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
						this->bias_multiplier_.cpu_data(),
						(Dtype)1., top_data + top[i]->offset(n));
			}

		}
	}
}


template<typename Dtype>
void ScaleConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	const Dtype* weight = NULL;
	Dtype* weight_diff = NULL;
	if(this->param_propagate_down_[0])
	{
		weight = this->blobs_[0]->cpu_data();
		weight_diff = this->blobs_[0]->mutable_cpu_diff();
		caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
	}
	Dtype* bias_diff = NULL;
	if(this->bias_term_ && this->param_propagate_down_[1])
	{
		bias_diff = this->blobs_[1]->mutable_cpu_diff();
		caffe_set(this->blobs_[1]->count(), Dtype(0),bias_diff);
	}
	const int weight_offset = this->M_ * this->K_;
	const int col_offset = this->K_ * this->N_;
	const int top_offset = this->M_ * this->N_;
	for(int i = 0; i < top.size();++i)
	{
		const Dtype* top_diff = NULL;
		// bias gradient , of necessary
		if(this->bias_term_ && this->param_propagate_down_[1])
		{
			top_diff = top[i]->cpu_diff();
			for(int n = 0; n<this->num_;++n)
			{
				caffe_cpu_gemv<Dtype>(CblasNoTrans, this->num_output_, this->N_,
						1., top_diff + top[0]->offset(n), this->bias_multiplier_.cpu_data(),
						1.,bias_diff);

			}
		}
		if(this->param_propagate_down_[0] || propagate_down[i])
		{
			if(!top_diff)
			{
				top_diff = top[i]->cpu_diff();

			}
			Dtype* col_buff = NULL;
			if(!this->is_1x1_)
			{
				col_buff = this->col_buffer_.mutable_cpu_data();
			}
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
			for(int n = 0; n < this->num_; ++n)
			{
				// Since we saved memory in the forward pass by not storing all col
				// data, we will need to recompute them.
				if(!this->is_1x1_)
				{
					scale_im2col_cpu(bottom_data + bottom[i]->offset(n), this->channels_, this->height_,
							this->width_, this->kernel_h_, this->kernel_w_,this->pad_h_, this->pad_w_,
							this->stride_h_, this->stride_w_, this->cur_scale, col_buff);
				}
				else
				{
					col_buff = bottom[i]->mutable_cpu_data() + bottom[i]->offset(n);
				}
				// gradient w.r.t. weight. Note that we will accumulate diffs.
				if(this->param_propagate_down_[0])
				{
					for(int g = 0; g < this->group_; ++g)
					{
						caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasTrans, this->M_,this->K_, this->N_,
								(Dtype)1., top_diff + top[i]->offset(n)+top_offset*g,
								col_buff+col_offset*g, (Dtype)1.,weight_diff + weight_offset*g);
					}
				}
				//gradient w.r.t. bottom data if necessary
				if(propagate_down[i])
				{
					if(weight == NULL)
					{
						weight = this->blobs_[0]->cpu_data();
					}
					if(this->is_1x1_)
					{
						col_buff = bottom[i]->mutable_cpu_diff()+bottom[i]->offset(n);
					}
					for(int g = 0; g < this->group_ ; ++g)
					{
						caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans, this->K_,this->N_,this->M_,
								(Dtype)1.,weight+weight_offset*g, top_diff+top[i]->offset(n)+top_offset*g,
								(Dtype)0., col_buff + col_offset*g);
					}
					// col2im back to the data
					if(!this->is_1x1_)
					{
						scale_col2im_cpu(col_buff, this->channels_, this->height_, this->width_,
								this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
								this->stride_h_, this->stride_w_,this->cur_scale, bottom_diff+bottom[i]->offset(n));
					}
				}
			}
		}
	}
	this->cur_scale = 0;
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ScaleConvolutionLayer);


}  // namespace caffe
