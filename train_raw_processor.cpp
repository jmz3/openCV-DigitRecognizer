#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string.h>
#include <fstream>
using namespace std;
 
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

string convertInt(int number)
{
    stringstream ss;//create a stringstream
    ss << number;//add number to the stream
    return ss.str();//return a string with the contents of the stream
}
 
void readFile(std::string datasetPath,int samplesPerClass,std::string outputfile )
{
    fstream file(outputfile,ios::out);
    for(int sample = 1; sample<=samplesPerClass;sample++) {
        for(int digit=0;digit<10;digit++) {
            //creating the file path string
			
            std::string imagePath = datasetPath+ convertInt(digit) + "\\img (" + convertInt(sample) +").png";	
			//cout<< imagePath;
			
            //reading the image
            cv::Mat img = cv::imread(imagePath,0);
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
            //writing the label to file
            file<<digit<<"\n";
        }
    }
    file.close();
}

int main()
{
    cout<<"Reading the training set......\n";
    readFile("training\\",305,"trainingset.txt");
    cout<<"Reading the test set.........\n";
    readFile("test\\",130,"testset.txt");
    cout<<"operation completed";
    return 0;
}