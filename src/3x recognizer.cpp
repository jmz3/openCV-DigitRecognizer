#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string.h>
#include <fstream>
using std::stringstream;

#define ATTRIBUTES 256  		//Number of pixels per sample.16X16
#define CLASSES 10              //Number of distinct labels.
#define TEST_SAMPLES 1       	//Number of samples in test dataset

//scale the input image to 16x16.
void scaleDownImage(cv::Mat &originalImg,cv::Mat &scaledDownImage )
{      
	for(int x=0;x<16;x++) {
		for(int y=0;y<16 ;y++) {
			//_original code
			//int yd = ceil((float)(y*originalImg.cols/16));
			//int xd = ceil((float)(x*originalImg.rows/16));

			//int proper_int_number = static_cast<int>(input_float_number)
			int yd = (int) ceil( static_cast<int>(y*originalImg.cols/16) );
			int xd = (int) ceil( static_cast<int>(x*originalImg.rows/16) );
			scaledDownImage.at<uchar>(x,y) = originalImg.at<uchar>(xd,yd);
		}
	}
}

//crop the image to the contour's edges.
void cropImage(cv::Mat &originalImage,cv::Mat &croppedImage)
{
	int row = originalImage.rows;
	int col = originalImage.cols;
	int tlx,tly,bry,brx;//t=top r=right b=bottom l=left
	tlx=tly=bry=brx=0;
	float suml=0;
	float sumr=0;
	int flag=0;
	/**************************top edge***********************/
	for(int x=1;x<row;x++) {
		for(int y = 0;y<col;y++) {
			if(originalImage.at<uchar>(x,y)==0) {
				flag=1;
				tly=x;
				break;
			}
		}
		if(flag==1) {
			flag=0;
			break;
		}
	}
	/*******************bottom edge***********************************/
	for(int x=row-1;x>0;x--) {
		for(int y = 0;y<col;y++) {
			if(originalImage.at<uchar>(x,y)==0) {
				flag=1;
				bry=x;
				break;
			}
		}
		if(flag==1) {
			flag=0;
			break;
		}
	}
	/*************************left edge*******************************/
	for(int y=0;y<col;y++) {
		for(int x = 0;x<row;x++) {
			if(originalImage.at<uchar>(x,y)==0) {
				flag=1;
				tlx=y;
				break;
			}
		}
		if(flag==1) {
			flag=0;
			break;
		}
	}
	/**********************right edge***********************************/
	for(int y=col-1;y>0;y--) {
		for(int x = 0;x<row;x++) {
			if(originalImage.at<uchar>(x,y)==0) {
				flag=1;
				brx= y;
				break;
			}
		}
		if(flag==1) {
			flag=0;
			break;
		}
	}
	int width = brx-tlx;
	int height = bry-tly;
	cv::Mat crop(originalImage,cv::Rect(tlx,tly,brx-tlx,bry-tly));
	croppedImage= crop.clone();
}

//convert Mat to array of Int
void convertToPixelValueArray(cv::Mat &img,int pixelarray[])
{
	int i =0;
	for(int x=0;x<16;x++) {
		for(int y=0;y<16;y++) {
			pixelarray[i]=(img.at<uchar>(x,y)==255)?1:0;
			i++;
		}
	}
}

//read a file and return its data as a string
std::string readFile(std::string datasetPath)
{
			stringstream myStringStream;

			//reading the image
			cv::Mat img = cv::imread(datasetPath,0);
			
			//Mat for output image.
			cv::Mat output;
			
			//Applying gaussian blur to remove any noise
			cv::GaussianBlur(img,output,cv::Size(5,5),0);
			//thresholding to get a binary image
			cv::threshold(output,output,50,255,0);

			//declaring mat to hold the scaled down image
			cv::Mat scaledDownImage(16,16,CV_8U,cv::Scalar(0));
			
			//declaring array to hold the pixel values in the memory before it written into file
			int pixelValueArray[256];

			//cropping the image.
			cropImage(output,output);

			//reducing the image dimension to 16X16
			scaleDownImage(output,scaledDownImage);

			//reading the pixel values.
			convertToPixelValueArray(scaledDownImage,pixelValueArray);
			
			//writing pixel data to file
			for(int d=0;d<256;d++){
				//file<<pixelValueArray[d]<<",";
				myStringStream << pixelValueArray[d];
			}
	return myStringStream.str();
}
//
void read_dataset(cv::Mat &data, int total_samples, std::string input_file)
{
	//variable used to store pixelValue (binary)
	float pixelValue;
	
	for(int row = 0; row < total_samples; row++) {	
		//for each attribute in the row
		for(int col = 0; col <=ATTRIBUTES; col++) {		
			//if its the pixel value.
			if (col < ATTRIBUTES){
				//get the char value of the string at position 'col'
				char testChar = input_file.at(col);
				// convert the datatype char to int -> http://bit.ly/19xu9il
				pixelValue = static_cast<float>(testChar - '0');
				//insert pixelValue into Mat data at position 'row,col'
				data.at<float>(row,col) = pixelValue;
			}
		}
	}
}

int predict(std::string dataOfInput)
{
		//create an object of CvANN_MLP
	CvANN_MLP nnetwork;
	
	//read the model from the XML file and create the neural network.
	CvFileStorage* storage = cvOpenFileStorage("serialized_training.xml", 0, CV_STORAGE_READ );
	CvFileNode *n = cvGetFileNodeByName(storage,0,"DigitOCR");
	nnetwork.read(storage,n);
	cvReleaseFileStorage(&storage);

	//matrix to hold the input image
	cv::Mat test_set(TEST_SAMPLES,ATTRIBUTES,CV_32F);

	//read the binary data into the Mat for testing 
	read_dataset(test_set, TEST_SAMPLES, dataOfInput);

	//Mat used to store the input image's binary values.
	cv::Mat test_sample;
	// Data from output_txt goes to test_sample
	test_sample = test_set.row(0);

	//Mat used to store the predicted output from the neural network
	cv::Mat classOut(1,CLASSES,CV_32F);

	//prediction
	nnetwork.predict(test_sample, classOut);

	// find the class with maximum weightage.
	int maxIndex = 0;
	float value = 0.0f;
	float maxValue=classOut.at<float>(0,0);
	for(int index=1;index<CLASSES;index++) 	{   
		value = classOut.at<float>(0,index);
		if(value>maxValue)	{   
			maxValue = value;
			maxIndex=index;
		}
	}
	//maxIndex is the predicted class.
	//std::cout << std::endl << maxIndex;
	return maxIndex;
}


int main(int argc, char *argv[])
{
	if(argc != 2) {
		std::cout << "Usage is OpenCV-Image-To-Bin-Txt <input_image>";
		return 0;
	}

	std::cout<<"\nReading input image :";
	std::cout << argv[1];

	std::string dataOfInput  = readFile(argv[1]);
	//std::cout << "\nread operation completed"; 
	
	std::cout << "\nThe output of the input  is " << predict(dataOfInput);

    return 0;
}
