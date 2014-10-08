#include <cstring>
#include <vector>
#include<cmath>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

void generate_random_rules(const char* path,int n_channel_in,int n_out)
{
	FILE* fid=fopen(path,"w");
	for(int i=0;i<n_out;i++)
	{
		for(int j=0;j<n_channel_in;j++)
		{
			//fprintf(fid,"%d ",rand()%2);
			fprintf(fid,"%d ",1);
		}
		fprintf(fid,"\n");
	}

	fclose(fid);
}

vector<vector<int> > get_rules(const char* path,int n_channel_in,int n_out){
	vector<vector<int> > res;
	FILE* fid = fopen(path,"r");
	int temp_int;
	for(int i=0;i<n_out;i++)
	{
		vector<int> row;
		row.clear();
		for(int j=0;j<n_channel_in;j++)
		{
			fscanf(fid,"%d",&temp_int);
			row.push_back(temp_int);
		}
		res.push_back(row);
	}
	fclose(fid);
	return res;
}

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_softplus_conv(const Blob<Dtype>* in, LayerParameter& conv_param, vector<vector<int> > & rules,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;

  if (conv_param.convolution_param().has_kernel_size()) {
    kernel_h = kernel_w = conv_param.convolution_param().kernel_size();
  } else {
    kernel_h = conv_param.convolution_param().kernel_h();
    kernel_w = conv_param.convolution_param().kernel_w();
  }
  int pad_h, pad_w;
  if (!conv_param.convolution_param().has_pad_h()) {
    pad_h = pad_w = conv_param.convolution_param().pad();
  } else {
    pad_h = conv_param.convolution_param().pad_h();
    pad_w = conv_param.convolution_param().pad_w();
  }
  int stride_h, stride_w;
  if (!conv_param.convolution_param().has_stride_h()) {
    stride_h = stride_w = conv_param.convolution_param().stride();
  } else {
    stride_h = conv_param.convolution_param().stride_h();
    stride_w = conv_param.convolution_param().stride_w();
  }
  // Groups
  int groups = conv_param.convolution_param().group();
  int o_g = out->channels() / groups;
  int k_g = in->channels() / groups;
  int o_head, k_head;

  Dtype beta;
  beta = (Dtype)(conv_param.specified_soft_plus_convolution_param().beta());

  // Convolution
  const Dtype* in_data = in->cpu_data();
  const Dtype* weight_data = weights[0]->cpu_data();
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->num(); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;

      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int y = 0; y < out->height(); y++) {
            for (int x = 0; x < out->width(); x++) {
              for (int p = 0; p < kernel_h; p++) {
                for (int q = 0; q < kernel_w; q++) {
                  int in_y = y * stride_h - pad_h + p;
                  int in_x = x * stride_w - pad_w + q;
                  if (in_y >= 0 && in_y < in->height()
                    && in_x >= 0 && in_x < in->width()) {
                	Dtype weight_data_new = 1/beta*log(exp(beta*weight_data[weights[0]->offset(o + o_head, k, p, q)])+1.);
                    out_data[out->offset(n, o + o_head, y, x)] +=
                        in_data[in->offset(n, k + k_head, in_y, in_x)]
                        * weight_data_new * rules.at(o).at(k);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param.convolution_param().bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->num(); n++) {
      for (int o = 0; o < out->channels(); o++) {
        for (int y = 0; y < out->height(); y++) {
          for (int x = 0; x < out->width(); x++) {
            out_data[out->offset(n, o, y, x)] += log(exp(bias_data[o]*beta)+1)/beta;
          }
        }
      }
    }
  }
}

template void caffe_softplus_conv(const Blob<float>* in,
    LayerParameter& caffe_softplus_conv,vector<vector<int> >& rules,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_softplus_conv(const Blob<double>* in,
	LayerParameter& conv_param,vector<vector<int> >& rules,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class SoftPlusConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftPlusConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~SoftPlusConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftPlusConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftPlusConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;


  string path = "rules.txt";
  generate_random_rules(path.c_str(),3,4);
  SpecifiedSoftPlusConvolutionParameter* soft_param = layer_param.mutable_specified_soft_plus_convolution_param();
  soft_param->set_beta(1.);
  soft_param->set_rule_file_path(path);


  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  // bottom structure (2, 3, 6, 4)
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new SpecifiedSoftPlusConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
  // setting group should not change the shape
  // not support for group >1


//  convolution_param->set_num_output(3);
//  convolution_param->set_group(3);
//  layer.reset(new ConvolutionLayer<Dtype>(layer_param));
//  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  EXPECT_EQ(this->blob_top_->num(), 2);
//  EXPECT_EQ(this->blob_top_->channels(), 3);
//  EXPECT_EQ(this->blob_top_->height(), 2);
//  EXPECT_EQ(this->blob_top_->width(), 1);
//  EXPECT_EQ(this->blob_top_2_->num(), 2);
//  EXPECT_EQ(this->blob_top_2_->channels(), 3);
//  EXPECT_EQ(this->blob_top_2_->height(), 2);
//  EXPECT_EQ(this->blob_top_2_->width(), 1);
}

TYPED_TEST(SoftPlusConvolutionLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;

  string path = "rules.txt";
  generate_random_rules(path.c_str(),3,4);
  SpecifiedSoftPlusConvolutionParameter* soft_param = layer_param.mutable_specified_soft_plus_convolution_param();
  soft_param->set_beta(1.);
  soft_param->set_rule_file_path(path);


  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new SpecifiedSoftPlusConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  vector<vector<int> >rules = get_rules(path.c_str(),3,4);
  caffe_softplus_conv(this->blob_bottom_, layer_param ,rules,layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));

  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4)<<i<<" first";
  }
  caffe_softplus_conv(this->blob_bottom_2_, layer_param ,rules, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4)<<i<<" second";
  }
}

TYPED_TEST(SoftPlusConvolutionLayerTest, Test1x1Convolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  string path = "rules.txt";
  generate_random_rules(path.c_str(),3,4);
  vector<vector<int> >rules = get_rules(path.c_str(),3,4);
  SpecifiedSoftPlusConvolutionParameter* soft_param = layer_param.mutable_specified_soft_plus_convolution_param();
  soft_param->set_beta(1.);
  soft_param->set_rule_file_path(path);

  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(1);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new SpecifiedSoftPlusConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_softplus_conv(this->blob_bottom_,  layer_param ,rules, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

// not supported for group>1
//TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolutionGroup) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  ConvolutionParameter* convolution_param =
//      layer_param.mutable_convolution_param();
//  convolution_param->set_kernel_size(3);
//  convolution_param->set_stride(2);
//  convolution_param->set_num_output(3);
//  convolution_param->set_group(3);
//  convolution_param->mutable_weight_filler()->set_type("gaussian");
//  convolution_param->mutable_bias_filler()->set_type("constant");
//  convolution_param->mutable_bias_filler()->set_value(0.1);
//  shared_ptr<Layer<Dtype> > layer(
//      new ConvolutionLayer<Dtype>(layer_param));
//  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//  // Check against reference convolution.
//  const Dtype* top_data;
//  const Dtype* ref_top_data;
//  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
//      this->MakeReferenceTop(this->blob_top_));
//  top_data = this->blob_top_->cpu_data();
//  ref_top_data = this->ref_blob_top_->cpu_data();
//  for (int i = 0; i < this->blob_top_->count(); ++i) {
//    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//  }
//}


TYPED_TEST(SoftPlusConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;


	string path = "rules.txt";
	generate_random_rules(path.c_str(),3,2);
	vector<vector<int> >rules = get_rules(path.c_str(),3,2);
	SpecifiedSoftPlusConvolutionParameter* soft_param = layer_param.mutable_specified_soft_plus_convolution_param();
	soft_param->set_beta(1.);
	soft_param->set_rule_file_path(path);

  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  SpecifiedSoftPlusConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SoftPlusConvolutionLayerTest, Test1x1Gradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;


  string path = "rules.txt";
  generate_random_rules(path.c_str(),3,2);
  vector<vector<int> >rules = get_rules(path.c_str(),3,2);
  SpecifiedSoftPlusConvolutionParameter* soft_param = layer_param.mutable_specified_soft_plus_convolution_param();
  soft_param->set_beta(1.);
  soft_param->set_rule_file_path(path);



  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->set_kernel_size(1);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  SpecifiedSoftPlusConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


// not support for group
//TYPED_TEST(ConvolutionLayerTest, TestGradientGroup) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  ConvolutionParameter* convolution_param =
//      layer_param.mutable_convolution_param();
//  convolution_param->set_kernel_size(3);
//  convolution_param->set_stride(2);
//  convolution_param->set_num_output(3);
//  convolution_param->set_group(3);
//  convolution_param->mutable_weight_filler()->set_type("gaussian");
//  convolution_param->mutable_bias_filler()->set_type("gaussian");
//  ConvolutionLayer<Dtype> layer(layer_param);
//  GradientChecker<Dtype> checker(1e-2, 1e-3);
//  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//      this->blob_top_vec_);
//}


}  // namespace caffe
