#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

/**
 * SpecifiedSoftPlusConvolutionLayer designed by alan
 */
template<typename Dtype>

void SpecifiedSoftPlusConvolutionLayer<Dtype>::Check_Masked_Weight_cpu(const Dtype * data)
{
	CHECK(this->group_ == 1) <<"Not support group_ > 0";
	int kernel_size_per_channels = this->kernel_h_ * this->kernel_w_;
	int my_weight_offset = this->M_ * this->K_;
	// check mask unused weights in each channel for debug
	// by alan
	for (int g = 0; g < this->group_; ++g) {
		for(int out_id = 0;out_id < this->num_output_; ++out_id){
			for(int channel_id = 0;channel_id < this->channels_; ++channel_id){
				const Dtype* weight = data+
						(out_id*this->channels_ + channel_id)*kernel_size_per_channels;
				if(this->activation_rules[out_id][channel_id] == 1)
					continue;
				for(int weight_idx = 0; weight_idx < kernel_size_per_channels;++weight_idx){
//					std::cout<<"weight["<<out_id<<"]["<<channel_id<<"]["<<weight_idx<<"] "<<
//							(*(weight+my_weight_offset * g+weight_idx))<<std::endl;
					CHECK((*(weight+my_weight_offset * g+weight_idx)) == (Dtype)0.) << "weight at output_["
							<<out_id <<"] channel["<<channel_id<<"] with idx[" <<weight_idx
							<<"]  is not zero";
				}
			}
		}
	}
}


template<typename Dtype>
void SpecifiedSoftPlusConvolutionLayer<Dtype>::Set_Masked_Weight_cpu(Dtype * data){
	CHECK(this->group_ == 1) <<"Not support group_ > 0";
	int kernel_size_per_channels = this->kernel_h_ * this->kernel_w_;
	int my_weight_offset = this->M_ * this->K_;
	// mask unused weight according to rules
	// by alan
//	std::cout<<"num_output: "<<this->num_output_ <<"   channels: "<< this->channels_<<
//			"    activation_rules.size():"<<activation_rules.size()<<std::endl;
//	std::cout<<"predicted weight size:"<<kernel_size_per_channels*this->num_output_*this->channels_ <<std::endl;
	Caffe::set_mode(Caffe::CPU);
	for (int g = 0; g < this->group_; ++g) {
		for(int out_id = 0;out_id < this->num_output_; ++out_id){
			for(int channel_id = 0;channel_id < this->channels_; ++channel_id){
				Dtype* weight = data+
						(out_id*this->channels_ + channel_id)*kernel_size_per_channels;
//				std::cout<<"setting ["<<out_id<<","<<channel_id<<"]  start with idx:"<<
//						(out_id*this->channels_ + channel_id)*kernel_size_per_channels<<
//						", the first value is "<<weight[0]<<std::endl;

				if(this->activation_rules.at(out_id).at(channel_id) == 1)
				{
					//std::cout<<" ["<<out_id<<","<<channel_id<<"]  is ignored"<<std::endl;
					continue;
				}


				//caffe::caffe_scal(kernel_size_per_channels,(Dtype)0.,weight+my_weight_offset * g);
				caffe_set<Dtype>(kernel_size_per_channels,(Dtype)0.,weight+my_weight_offset * g);
				//memset(weight+my_weight_offset * g,0,sizeof(Dtype)*kernel_size_per_channels);
			}
		}
	}
}

template <typename Dtype>
void SpecifiedSoftPlusConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	ConvolutionLayer<Dtype>::LayerSetUp(bottom,top);
	CHECK(this->layer_param_.has_specified_soft_plus_convolution_param()) <<"SpecifiedSoftPlusConvolutionParameter is required. ";
	SpecifiedSoftPlusConvolutionParameter soft_plus_param = this->layer_param_.specified_soft_plus_convolution_param();
	this->beta = soft_plus_param.beta();
	CHECK(soft_plus_param.has_rule_file_path())<<"path of rule is required";
	Set_Masked_Weight_cpu(this->blobs_[0]->mutable_cpu_data()); // by alan
	//std::cout<<"set weight end"<<std::endl;
}


template <typename Dtype>
void SpecifiedSoftPlusConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		      const vector<Blob<Dtype>*>& top){
	ConvolutionLayer<Dtype>::Reshape(bottom,top);
	this->weights_buffer_.Reshape(this->num_output_,this->channels_,this->kernel_h_,this->kernel_w_);
	this->bias_buffer_.Reshape(1,1,1,this->num_output_);

	SpecifiedSoftPlusConvolutionParameter soft_plus_param = this->layer_param_.specified_soft_plus_convolution_param();
	string rule_file_name = soft_plus_param.rule_file_path();
	FILE* rule_fid = fopen(rule_file_name.c_str(),"r");
	CHECK(rule_fid!= NULL)<<"can not find rule file in SpecifiedSoftPlusConvolutionLayer";
	int temp_int=0;
	vector<int> temp_vec ;
	while(fscanf(rule_fid,"%d",&temp_int) == 1)
	{
		temp_vec.push_back(temp_int);
	}
	fclose(rule_fid);
	//std::cout<<"fuck, read file end"<<std::endl;
	CHECK (temp_vec.size() ==  this->num_output_* this->channels_) <<
			"rules in file don't matches layer structure "<<temp_vec.size() <<
			" != "<<this->num_output_* this->channels_;
	this->activation_rules.clear();
	for(int i=0;i<this->num_output_;i++)
	{
		vector<int> one_channel_rules(&temp_vec[i*this->channels_],&temp_vec[i*this->channels_+this->channels_]);
		//std::cout<<"for output "<<i <<"  ,the temp_vec.size="<<one_channel_rules.size()<<std::endl;
		this->activation_rules.push_back(one_channel_rules);

	}
	CHECK(this->activation_rules.size() == this->num_output_) << "number of rules do not match num_output";
	CHECK(this->group_ == 1) <<"Not support group_ > 0";
	//std::cout<<"starting to set weight"<<std::endl;
	//std::cout<<"total weight size:"<<this->blobs_[0]->count()<<std::endl;
	Set_Masked_Weight_cpu(this->blobs_[0]->mutable_cpu_data()); // by alan
}

template<typename Dtype>
void SpecifiedSoftPlusConvolutionLayer<Dtype>::Copy_to_buff_and_softplus_blobs_cpu()
{
	int n_weights = this->weights_buffer_.count();
	CHECK(n_weights == this->blobs_[0]->count()) << "weights_buffer_->count() is not equal to blobs_[0]->count()";
	caffe::caffe_copy(n_weights,this->blobs_[0]->cpu_data(),this->weights_buffer_.mutable_cpu_data());

	Dtype* p_weights = this->blobs_[0]->mutable_cpu_data();
	caffe::caffe_cpu_scale(n_weights,(Dtype)(this->beta),p_weights,p_weights);
	caffe::caffe_exp(n_weights,p_weights,p_weights);
	caffe::caffe_add_scalar(n_weights,(Dtype)1.,p_weights);
	caffe::caffe_log(n_weights,p_weights,p_weights);
	caffe::caffe_cpu_scale(n_weights,(Dtype)(1./(Dtype)(this->beta)),p_weights,p_weights);
	caffe::caffe_copy(n_weights,p_weights,this->weights_buffer_.mutable_cpu_diff());
	if(this->bias_term_)
	{
		int n_bias = this->bias_buffer_.count();
		CHECK(n_bias == this->blobs_[1]->count()) <<"bias_buffer_->count() is not equal to blobs_[1]->count()";
		caffe::caffe_copy(n_bias,this->blobs_[1]->cpu_data(),this->bias_buffer_.mutable_cpu_data());

		Dtype* p_bias = this->blobs_[1]->mutable_cpu_data();
		caffe::caffe_cpu_scale(n_bias,(Dtype)(this->beta),p_bias,p_bias);
		caffe::caffe_exp(n_bias,p_bias,p_bias);
		caffe::caffe_add_scalar(n_bias,(Dtype)1.,p_bias);
		caffe::caffe_log(n_bias,p_bias,p_bias);
		caffe::caffe_cpu_scale(n_bias,(Dtype)(1./(Dtype)(this->beta)),p_bias,p_bias);
		caffe::caffe_copy(n_bias,p_bias,this->bias_buffer_.mutable_cpu_diff());
	}

}

template<typename Dtype>
void SpecifiedSoftPlusConvolutionLayer<Dtype>::Set_weights_bias_to_softplus_cpu(){
	int n_weights = this->weights_buffer_.count();
	CHECK(n_weights == this->blobs_[0]->count()) << "weights_buffer_->count() is not equal to blobs_[0]->count()";
	caffe::caffe_copy(n_weights,this->weights_buffer_.cpu_diff(),this->blobs_[0]->mutable_cpu_data());
	if(this->bias_term_)
	{
		int n_bias = this->bias_buffer_.count();
		CHECK(n_bias == this->blobs_[1]->count()) <<"bias_buffer_->count() is not equal to blobs_[1]->count()";
		caffe::caffe_copy(n_bias,this->bias_buffer_.cpu_diff(),this->blobs_[1]->mutable_cpu_data());
	}
}

template<typename Dtype>
void SpecifiedSoftPlusConvolutionLayer<Dtype>::Restore_weights_bias_cpu(){
	int n_weights = this->weights_buffer_.count();
	CHECK(n_weights == this->blobs_[0]->count()) << "weights_buffer_->count() is not equal to blobs_[0]->count()";
	caffe::caffe_copy(n_weights,this->weights_buffer_.cpu_data(),this->blobs_[0]->mutable_cpu_data());
	if(this->bias_term_)
	{
		int n_bias = this->bias_buffer_.count();
		CHECK(n_bias == this->blobs_[1]->count()) <<"bias_buffer_->count() is not equal to blobs_[1]->count()";
		caffe::caffe_copy(n_bias,this->bias_buffer_.cpu_data(),this->blobs_[1]->mutable_cpu_data());
	}
}


template<typename Dtype>
void SpecifiedSoftPlusConvolutionLayer<Dtype>::Restore_to_blobs_and_set_diff_cpu(){
	int n_weights = this->weights_buffer_.count();
	CHECK(n_weights == this->blobs_[0]->count()) << "weights_buffer_->count() is not equal to blobs_[0]->count()";
	caffe::caffe_copy(n_weights,this->weights_buffer_.cpu_data(),this->blobs_[0]->mutable_cpu_data());

	Dtype* p_numerator = this->weights_buffer_.mutable_cpu_data();
	Dtype* p_denominator = this->weights_buffer_.mutable_cpu_diff();

	caffe::caffe_cpu_scale(n_weights,(Dtype)(this->beta),p_numerator,p_numerator);
	caffe::caffe_exp(n_weights,p_numerator,p_numerator);

	caffe::caffe_copy(n_weights,p_numerator,p_denominator);
	caffe::caffe_add_scalar(n_weights,(Dtype)1.,p_denominator);
	caffe::caffe_div(n_weights,p_numerator,p_denominator,p_numerator);

	caffe::caffe_mul(n_weights,this->blobs_[0]->mutable_cpu_diff(),p_numerator,this->blobs_[0]->mutable_cpu_diff());

	if(this->bias_term_)
	{
		int n_bias = this->bias_buffer_.count();
		CHECK(n_bias == this->blobs_[1]->count()) <<"bias_buffer_->count() is not equal to blobs_[1]->count()";

		caffe::caffe_copy(n_bias,this->bias_buffer_.cpu_data(),this->blobs_[1]->mutable_cpu_data());

		Dtype* p_numerator = this->bias_buffer_.mutable_cpu_data();
		Dtype* p_denominator = this->bias_buffer_.mutable_cpu_diff();

		caffe::caffe_cpu_scale(n_bias,(Dtype)(this->beta),p_numerator,p_numerator);
		caffe::caffe_exp(n_bias,p_numerator,p_numerator);

		caffe::caffe_copy(n_bias,p_numerator,p_denominator);
		caffe::caffe_add_scalar(n_bias,(Dtype)1.,p_denominator);
		caffe::caffe_div(n_bias,p_numerator,p_denominator,p_numerator);

		caffe::caffe_mul(n_bias,this->blobs_[1]->mutable_cpu_diff(),p_numerator,this->blobs_[1]->mutable_cpu_diff());
	}
}

template <typename Dtype>
void SpecifiedSoftPlusConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	this->Check_Masked_Weight_cpu(this->blobs_[0]->cpu_data());  // for debug by alan
	this->Copy_to_buff_and_softplus_blobs_cpu();
	Set_Masked_Weight_cpu(this->blobs_[0]->mutable_cpu_data()); // by alan
	//this->Check_Masked_Weight_cpu(this->blobs_[0]->cpu_data());  // for debug by alan
	ConvolutionLayer<Dtype>::Forward_cpu(bottom,top);
	Restore_weights_bias_cpu();
}


/**
 * propagate_down[i] specified whether to propagate from top[i] .
 */
template <typename Dtype>
void SpecifiedSoftPlusConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	this->Check_Masked_Weight_cpu(this->blobs_[0]->cpu_data());  // for debug by alan
	Set_weights_bias_to_softplus_cpu();
	Set_Masked_Weight_cpu(this->blobs_[0]->mutable_cpu_data()); // by alan

	ConvolutionLayer<Dtype>::Backward_cpu(top,propagate_down,bottom);
	Set_Masked_Weight_cpu(this->blobs_[0]->mutable_cpu_diff()); // by alan
	this->Restore_to_blobs_and_set_diff_cpu();

}

#ifdef CPU_ONLY
STUB_GPU(SpecifiedSoftPlusConvolutionLayer);
#endif

INSTANTIATE_CLASS(SpecifiedSoftPlusConvolutionLayer);

}  // namespace caffe
