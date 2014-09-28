#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "caffe/util/im2col.hpp"
#include "caffe/caffe.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

void Mypause(){
	char c;
	while((c= getchar() !='\n'));

}




/**
 *  test for alan
 */

// img2col test
template <typename Dtype>
void fprint_arrs(const int height, const int width,const Dtype * arr,FILE* fh)
{
	for(int h=0;h<height;h++)
	{
		for(int w=0;w<width;++w)
		{
			fprintf(fh,"%10.2lf ",arr[(h)*width+w]);
		}
		fprintf(fh,"\n");
	}
}

//tested no bug
void im2col_test()
{
	int channels = 2;
	int height = 5;
	int width = 5;
	int pad_w=1;
	int pad_h = 1;
	int kernel_w =2;
	int kernel_h=2;
	int stride_h =1;
	int stride_w = 1;
	int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	int channels_col = channels * kernel_h * kernel_w;

	double * img_source = new double[sizeof(double)*channels*height*width];
	caffe::generate_sample_img(channels,height,width,img_source);
	FILE* fh = fopen("./test_result.txt","w");

	fprintf(fh,"print generated image:\n\n\n");
	for(int c=0;c<channels;c++)
	{
		fprint_arrs<double>(height,width,img_source+c*height*width,fh);
		fprintf(fh,"\n\n\n");
	}

	double * img_dest = new double[sizeof(double)*height_col*width_col*channels_col];
	caffe::scale_im2col_cpu(img_source,channels,height,width,kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w,1,img_dest);
//	caffe::im2col_cpu(img_source,channels,height,width,kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w,img_dest);

	double * patch_arr = new double[sizeof(double)*kernel_h*kernel_w];
	int c_to_test=0;
	int h_to_test=1;
	int w_to_test=1;

	caffe::get_patch_from_im2col(img_dest,2,height,width,kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w,
			c_to_test,h_to_test,w_to_test,patch_arr);

	fprintf(fh,"print patch at %d,%d,%d:\n\n\n",c_to_test,h_to_test,w_to_test);
	fprint_arrs<double>(kernel_h,kernel_w,patch_arr,fh);

//	caffe::col2im_cpu(img_dest,channels,height,width,kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w,img_source);
	caffe::scale_col2im_cpu(img_dest,channels,height,width,kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w,1,img_source);



	fprintf(fh,"print reconstructed image:\n\n\n");
	for(int c=0;c<channels;c++)
	{
		fprint_arrs<double>(height,width,img_source+c*height*width,fh);
		fprintf(fh,"\n\n\n");
	}


	fclose(fh);
	delete [] img_source;
	delete [] img_dest;
}

/**
 * test by alan end
 */
int main(int argc, char** argv) {

  caffe::GlobalInit(&argc, &argv);
  im2col_test();//by alan test;


//  printf("globalInit finished\n");//alan
//  Mypause();
  return 0;
}
