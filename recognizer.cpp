#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string.h>
#include <fstream>

#define ATTRIBUTES 256  		//Number of pixels per sample.16X16
#define CLASSES 10              //Number of distinct labels.
#define TEST_SAMPLES 1       	//Number of samples in test dataset


//scale the input image to 16x16.
void scaleDownImage(cv::Mat &originalImg,cv::Mat &scaledDownImage )
{      
	for(int x=0;x<16;x++) {
		for(int y=0;y<16 ;y++) {
			int yd =ceil((float)(y*originalImg.cols/16));
			int xd = ceil((float)(x*originalImg.rows/16));
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

//read a file and 
/*	std::fstream readFile(std::string datasetPath)	*/
void readFile(std::string datasetPath, std::string outputfile )
{
			std::fstream file(outputfile,std::ios::out);
			//std::cout  << datasetPath << std::endl;

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
				file<<pixelValueArray[d]<<",";
			}
	file.close();
}

void read_dataset(char *filename, cv::Mat &data, int total_samples)
{
	//variable used to store pixelValue (binary)
	float pixelvalue;
	//open the file
	FILE* inputfile = fopen( filename, "r" );

	//read each row of the csv file
	for(int row = 0; row < total_samples; row++) {	
		//for each attribute in the row
		for(int col = 0; col <=ATTRIBUTES; col++) {		
			//if its the pixel value.
			if (col < ATTRIBUTES){
				fscanf(inputfile, "%f,", &pixelvalue);
				data.at<float>(row,col) = pixelvalue;
			}
		}
	}
	fclose(inputfile);
}

int main(int argc, char *argv[])
{
	if(argc != 2) {
		std::cout << "Usage is OpenCV-Image-To-Bin-Txt <input_image>";
		return 0;
	}

	std::cout<<"Reading input image ";
	std::cout << argv[1];
	//readFile("C:\\OCR\\test_image.png","C:\\OCR\\image_out.txt");

/*	std::fstream binary_txt = readFile(argv[1]);	*/
	readFile(argv[1], "output.txt");
	std::cout << "\nread operation completed"; 
	

	//create an object of CvANN_MLP
	CvANN_MLP nnetwork;
	
	//read the model from the XML file and create the neural network.
	CvFileStorage* storage = cvOpenFileStorage("param.xml", 0, CV_STORAGE_READ );
	CvFileNode *n = cvGetFileNodeByName(storage,0,"DigitOCR");
	nnetwork.read(storage,n);
	cvReleaseFileStorage(&storage);

	//matrix to hold the input image
	cv::Mat test_set(TEST_SAMPLES,ATTRIBUTES,CV_32F);

	//read the binary data into the Mat for testing 
	read_dataset("output.txt", test_set, TEST_SAMPLES);
/*	read_dataset(argv[1], test_set, TEST_SAMPLES);	*/
	//read_dataset("C:\\OCR\\image_out.txt", test_set, TEST_SAMPLES);

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
	std::cout << std::endl << maxIndex;
    return 0;
}