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

void generate_rules(const char* path)
{
	FILE* fid=fopen(path,"w");

	fprintf(fid,"0 0 0\n");
	fprintf(fid,"1 0 0\n");
	fprintf(fid,"0 1 0\n");
	fprintf(fid,"0 0 1\n");
	fprintf(fid,"1 1 0\n");
	fprintf(fid,"1 0 1\n");
	fprintf(fid,"0 1 1\n");
	fprintf(fid,"1 1 1\n");
	fclose(fid);
}

vector<vector<int> > get_rules(const char* path){
	vector<vector<int> > res;
	FILE* fid = fopen(path,"r");
	int temp_int;
	for(int i=0;i<8;i++)
	{
		vector<int> row;
		row.clear();
		for(int j=0;j<3;j++)
		{
			fscanf(fid,"%d",&temp_int);
			row.push_back(temp_int);
		}
		res.push_back(row);
	}
	fclose(fid);
	return res;
}

template <typename Dtype>
void caffe_specified_across_channels_sum_pooling(const Blob<Dtype>* in, vector<vector<int> > & rules,
    Blob<Dtype>* out) {
  // Kernel size, stride, and pad
  int height = in->height();
  int width = in->width();
  CHECK(height == out->height()) <<"height of in_Blob is not equal to out_Blob";
  CHECK(width == out->width()) <<"width of in_Blob is not equal to out_Blob";
  int n_channels = in->channels();
  int n_output = out->channels();
  CHECK(rules.size() == n_output)<<"rules does not match the output structure";
  int n_num = in->num();
  CHECK(n_num == out->num()) <<"num_ of in should be the same as num_ output";

  const Dtype* in_data = in->cpu_data();

  Dtype* out_data = out->mutable_cpu_data();
  caffe::caffe_set(out->count(),(Dtype)0, out_data);

  for(int id_num = 0; id_num <n_num ; id_num++ )
  {
	  for(int id_output = 0 ; id_output < n_output ; id_output++)
	  {
		  for(int id_channel = 0; id_channel < n_channels;id_channel++)
		  {
			  int in_offset = in->offset(id_num,id_channel);
			  int out_offset = out->offset(id_output,id_output);
			  for(int h = 0 ; h < height ; h++)
			  {
				  for(int w = 0; w < width; w++)
				  {
					  out_data[out_offset+h*width+w] += in_data[out_offset+h*width+w] *
							  rules.at(id_output).at(id_channel);
				  }
			  }
		  }
	  }
  }
}

template void caffe_specified_across_channels_sum_pooling(const Blob<float>* in,
   vector<vector<int> >& rules,
    Blob<float>* out);
template void caffe_specified_across_channels_sum_pooling(const Blob<double>* in,
	vector<vector<int> >& rules,
    Blob<double>* out);

template <typename TypeParam>
class SumPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SumPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~SumPoolingLayerTest() {
    delete blob_bottom_;

    delete blob_top_;

  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;

  shared_ptr<Blob<Dtype> > ref_blob_top_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SumPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(SumPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;


  string path = "rules.txt";
  generate_rules(path.c_str());
  SpecifiedAcrossChannelsSumPoolingParameter *pool_param = layer_param.mutable_specified_across_channels_sum_pooling_param();

  pool_param->set_rule_file_path(path);
  pool_param->set_num_output(8);
  shared_ptr<Layer<Dtype> > layer(
      new SpecifiedAcrossChannelsSumPoolingLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 8);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 4);

}

TYPED_TEST(SumPoolingLayerTest, TestSimpleSumPooling) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;

  string path = "rules.txt";
  generate_rules(path.c_str());
  SpecifiedAcrossChannelsSumPoolingParameter *pool_param = layer_param.mutable_specified_across_channels_sum_pooling_param();

  pool_param->set_rule_file_path(path);
  pool_param->set_num_output(8);


  shared_ptr<Layer<Dtype> > layer(
      new SpecifiedSoftPlusConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  vector<vector<int> >rules = get_rules(path.c_str());
  caffe_specified_across_channels_sum_pooling(this->blob_bottom_,rules,
      this->MakeReferenceTop(this->blob_top_));

  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-5)<<i<<" first";
  }

}



TYPED_TEST(SumPoolingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;


	string path = "rules.txt";

	vector<vector<int> >rules = get_rules(path.c_str());
  SpecifiedAcrossChannelsSumPoolingParameter *pool_param = layer_param.mutable_specified_across_channels_sum_pooling_param();

  pool_param->set_rule_file_path(path);
  pool_param->set_num_output(8);


  SpecifiedSoftPlusConvolutionLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
}


}  // namespace caffe
