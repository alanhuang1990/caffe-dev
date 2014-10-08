#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void SpecifiedAcrossChannelsSumPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	SpecifiedAcrossChannelsSumPoolingParameter pool_param = this->layer_param_.specified_across_channels_sum_pooling_param();
	CHECK(pool_param.has_num_output())<<"number of output is required.";

	CHECK(pool_param.has_rule_file_path())<<"path of rule is required";

}

template <typename Dtype>
void SpecifiedAcrossChannelsSumPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	SpecifiedAcrossChannelsSumPoolingParameter pool_param = this->layer_param_.specified_across_channels_sum_pooling_param();

	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	num_output_ = pool_param.num_output();

	string rule_file_name = pool_param.rule_file_path();
	FILE* rule_fid = fopen(rule_file_name.c_str(),"r");
  	CHECK(rule_fid!= NULL)<<"can not find rule file in SpecifiedAcrossChannelsSumPoolingLayer";
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
  	//std::cout<<"starting to set weight"<<std::endl;
  	//std::cout<<"total weight size:"<<this->blobs_[0]->count()<<std::endl;

  	top[0]->Reshape(bottom[0]->num(), this->num_output_, height_,width_);
  	if (top.size() > 1) {
  		top[1]->ReshapeLike(*top[0]);
  	}

}


template <typename Dtype>
void SpecifiedAcrossChannelsSumPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int top_count = top[0]->count();

    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop

    for(int id_input = 0; id_input < bottom[0]->num(); ++id_input)
    {
    	for(int id_output = 0; id_output < this->num_output_; ++id_output)
    	{
    		int top_offset = top[0]->offset(id_input, id_output);
    		for(int c = 0; c < channels_; ++c)
    		{
    			int bottom_offset = bottom[0]->offset(id_input,c);
    			if(activation_rules.at(id_output).at(c) == 0)
    				continue;
    			caffe::caffe_add(height_*width_,top_data+top_offset,
    					bottom_data+bottom_offset,top_data+top_offset);
    		}
    	}
    }

}

template <typename Dtype>
void SpecifiedAcrossChannelsSumPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

    for(int id_input = 0; id_input < bottom[0]->num(); ++id_input)
    {
    	for(int id_output = 0; id_output < this->num_output_; ++id_output)
    	{
    		int top_offset = top[0]->offset(id_input, id_output);
    		for(int c = 0; c < channels_; ++c)
    		{
    			int bottom_offset = bottom[0]->offset(id_input,c);
    			if(activation_rules.at(id_output).at(c) == 0)
    				continue;
    			caffe::caffe_add(height_*width_,top_diff+top_offset,
    					bottom_diff+bottom_offset,bottom_diff+bottom_offset);
    		}
    	}
    }

}


#ifdef CPU_ONLY
STUB_GPU(SpecifiedAcrossChannelsSumPoolingLayer);
#endif

INSTANTIATE_CLASS(SpecifiedAcrossChannelsSumPoolingLayer);


}  // namespace caffe
