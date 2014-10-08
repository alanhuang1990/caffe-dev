#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include <sstream>
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;



template <typename Dtype>
void BindingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// check and initialization
	BindingParameter binding_param = this->layer_param_.binding_param();
    CHECK(binding_param.has_num_binding_group())<< "num_binding_group should be set!\n";
  	CHECK(binding_param.has_channels_each_group())<< "channels_each_group should be set!\n";
  	CHECK(binding_param.has_binding_rules())<< "binding_rules should be set!\n";
  	this->num_binding_group_ = binding_param.num_binding_group();
  	this->channels_each_group_ = binding_param.channels_each_group();
  	this->binding_rules.clear();
  	this->max_channel_id = 0;
  	CHECK_GT(this->num_binding_group_, 0) <<"num_binding_group  should be positive\n";
  	CHECK_GT(this->channels_each_group_, 0) <<"channels_each_group  should be positive\n";


  	std::istringstream buf_stream(binding_param.binding_rules());

  	const string& rules_string = binding_param.binding_rules();
  	// parse the rules_string into binding rules
  	int count=0;
  	for(int i=0;i<this->num_binding_group_;++i)
  	{
  		vector<int> buf_group;
  		buf_group.clear();
  		for(int j=0; j< this->channels_each_group_;j++)
  		{
  			int buf_int =0;
  			buf_stream >> buf_int;
  			buf_group.push_back(buf_int);
  			if(buf_int > this->max_channel_id)
  			{
  				this->max_channel_id = buf_int;
  			}
  			count++;
  		}
  		this->binding_rules.push_back(buf_group);
  	}

  	CHECK((count ==  this->channels_each_group_* this->num_binding_group_))<<"the binding rule does not matches"
  			<< "the num_binding_group  and  channels_each_group";

}

template <typename Dtype>
void BindingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	CHECK_GT(bottom[0]->channels(), this->max_channel_id)
			<< "channel id in rules exceeds the max channel id in bottom layer";
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();

	top[0]->Reshape(bottom[0]->num(), this->num_binding_group_*this->channels_each_group_, height_,width_);

	if (top.size() > 1) {
		top[1]->ReshapeLike(*top[0]);
	}
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void BindingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK(0) << "Unimplemented ";
    LOG(FATAL) << "Unknown pooling method.";

}

template <typename Dtype>
void BindingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	CHECK(0) << "Unimplemented ";
    LOG(FATAL) << "Unknown pooling method.";
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(BindingLayer);


}  // namespace caffe
