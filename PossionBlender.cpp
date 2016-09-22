//
//  main.cpp
//  PossionBlender
//
//  Created by hzfmacbook on 9/15/16.
//  Copyright © 2016 hzfmacbook. All rights reserved.
//

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void computeGradientX( const Mat &img, Mat &gx)
{
    Mat kernel = Mat::zeros(1, 3, CV_8S);
    kernel.at<char>(0,2) = 1;
    kernel.at<char>(0,1) = -1;
    
    if(img.channels() == 3)
    {
        filter2D(img, gx, CV_32F, kernel);
    }
    else if (img.channels() == 1)
    {
        Mat tmp[3];
        for(int chan = 0 ; chan < 3 ; ++chan)
        {
            filter2D(img, tmp[chan], CV_32F, kernel);
        }
        merge(tmp, 3, gx);
    }
}

void computeGradientY( const Mat &img, Mat &gy)
{
    Mat kernel = Mat::zeros(3, 1, CV_8S);
    kernel.at<char>(2,0) = 1;
    kernel.at<char>(1,0) = -1;
    
    if(img.channels() == 3)
    {
        filter2D(img, gy, CV_32F, kernel);
    }
    else if (img.channels() == 1)
    {
        Mat tmp[3];
        for(int chan = 0 ; chan < 3 ; ++chan)
        {
            filter2D(img, tmp[chan], CV_32F, kernel);
        }
        merge(tmp, 3, gy);
    }
}

//矩阵点乘，将lhs与rhs点乘得到result，因为有三个通道，估计mat不能实现三通道的矩阵的一次性点乘，所以才有这个函数
void arrayProduct(const cv::Mat& lhs, const cv::Mat& rhs, cv::Mat& result)
{
    vector <Mat> lhs_channels;
    vector <Mat> result_channels;
    
    split(lhs,lhs_channels);//拆分成3个通道的矩阵
    split(result,result_channels);
    
    for(int chan = 0 ; chan < 3 ; ++chan)//三个矩阵进行分别相乘
        multiply(lhs_channels[chan],rhs,result_channels[chan]);
    
    merge(result_channels,result);//合成为一个
}

void computeLaplacianX( const Mat &img, Mat &laplacianX)
{
    Mat kernel = Mat::zeros(1, 3, CV_8S);
    kernel.at<char>(0,0) = -1;
    kernel.at<char>(0,1) = 1;
    filter2D(img, laplacianX, CV_32F, kernel);
}

void computeLaplacianY( const Mat &img, Mat &laplacianY)
{
    Mat kernel = Mat::zeros(3, 1, CV_8S);
    kernel.at<char>(0,0) = -1;
    kernel.at<char>(1,0) = 1;
    filter2D(img, laplacianY, CV_32F, kernel);
}

void naiveBlender(Mat& srcA, Mat& srcB, Mat& dst)
{
    Mat grayB;
    cvtColor(srcB, grayB, CV_BGR2GRAY);
    
    threshold(grayB, grayB, 20, 255, THRESH_BINARY);
    
    for (int i = 0; i < srcA.rows; i++) {
        for (int j = 0; j < srcA.cols; j++) {
            if (grayB.at<uchar>(i, j) != 0) {
                srcA.at<Vec3b>(i, j) = srcB.at<Vec3b>(i, j);
            }
        }
    }
    
    imwrite("Naive.jpg", srcA);
}

void possionBlender(Mat& srcA, Mat& srcB, Mat& dst)
{
    Mat patchGradientX, patchGradientY, destinationGradientX, destinationGradientY;
    
    //计算ROI区域转换复制到destination一样大小的patch图片梯度
    computeGradientX(srcB, patchGradientX);
    computeGradientY(srcB, patchGradientY);
    
    patchGradientX.convertTo(patchGradientX, CV_8UC1);
    patchGradientY.convertTo(patchGradientY, CV_8UC1);
    
    //计算背景图片的梯度
    computeGradientX(srcA, destinationGradientX);
    computeGradientY(srcA, destinationGradientY);
    
    //
    Mat binary, binaryMaskFloatInverted;
    cvtColor(srcB, binary, CV_BGR2GRAY);
    threshold(binary, binary, 20, 255, THRESH_BINARY);
    
    binaryMaskFloatInverted = Mat(srcA.size(), CV_32F);
    for (int i = 0; i < binaryMaskFloatInverted.rows; i++) {
        for (int j = 0; j < binaryMaskFloatInverted.cols; j++) {
            if (binary.at<uchar>(i, j) == 0) {
                binaryMaskFloatInverted.at<float>(i, j) = 0;
            }
            else {
                binaryMaskFloatInverted.at<float>(i, j) = 1;
            }
        }
    }
    
    arrayProduct(destinationGradientX, binaryMaskFloatInverted, destinationGradientX);
    arrayProduct(destinationGradientY, binaryMaskFloatInverted, destinationGradientY);
    
    Mat laplacianX = Mat(srcA.size(), CV_32FC3);
    Mat laplacianY = Mat(srcA.size(), CV_32FC3);
    
    //因为前面已经对destinationGradientX做了固定区域的mask，patchGradientX做了修改区域的mask
    laplacianX = destinationGradientX + patchGradientX;
    laplacianY = destinationGradientY + patchGradientY;
    
    //求解梯度的散度 也就是拉普拉坐标
    computeLaplacianX(laplacianX, laplacianX);
    computeLaplacianY(laplacianY, laplacianY);
    
    //散度
    Mat lap = laplacianX + laplacianY;
    
    //构造方程组
    /*
    Mat A = Mat::zeros(srcA.rows * srcA.cols, 5, CV_32F);
    Mat X = Mat::zeros(srcA.rows * srcA.cols, 3, CV_32F);
    Mat b = Mat::zeros(srcA.rows * srcA.cols, 3, CV_32F);
    
    for (int i = 0; i < srcA.rows * srcA.cols; i++) {
        int r = i / srcA.cols;
        int c = i % srcA.cols;
        //在边界，直接赋边界值
        if (r == 0 || r == srcA.rows - 1 || c == 0 || c == srcA.cols - 1) {
            A.at<float>(i, i) = 1;
            b.at<float>(i, 0) = srcA.at<Vec3b>(r, c)[0];
            b.at<float>(i, 1) = srcA.at<Vec3b>(r, c)[1];
            b.at<float>(i, 2) = srcA.at<Vec3b>(r, c)[2];
        }
        else {
            A.at<float>(i, i) = -4;
            A.at<float>(i, i - srcA.cols)
            = A.at<float>(i, i + srcA.cols)
            = A.at<float>(i, i - 1)
            = A.at<float>(i, i + 1)
            = 1;
            
            b.at<float>(i, 0) = lap.at<Vec3b>(r, c)[0];
            b.at<float>(i, 1) = lap.at<Vec3b>(r, c)[1];
            b.at<float>(i, 2) = lap.at<Vec3b>(r, c)[2];
        }
    }
    
    X = A.inv() * b;
    
    dst = Mat(srcA.size(), CV_8UC3);
    
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            dst.at<Vec3b>(i, j)[0] = (uchar) X.at<float>(i * dst.rows + dst.cols, 0);
            dst.at<Vec3b>(i, j)[1] = (uchar) X.at<float>(i * dst.rows + dst.cols, 1);
            dst.at<Vec3b>(i, j)[2] = (uchar) X.at<float>(i * dst.rows + dst.cols, 2);
        }
    }
    */
    
    // 68068 none-black pixels
    Mat X(srcA.rows * srcA.cols, 2, CV_32FC3);
    
    for (int i = 0; i < srcA.rows * srcA.cols; i++) {
        if (binary.at<uchar>(i / srcA.cols, i % srcA.cols) == 0)
            X.at<Vec3f>(i, 1) = X.at<Vec3f>(i, 0) = srcA.at<Vec3b>(i / srcA.cols, i % srcA.cols);
        else
            X.at<Vec3f>(i, 1) = X.at<Vec3f>(i, 0) = srcB.at<Vec3b>(i / srcA.cols, i % srcA.cols);
    }
    
    Mat delta(srcA.rows * srcA.cols, 1, CV_32F);
    
    do {
        for (int i = 0; i < srcA.rows * srcA.cols; i++) {
            int row = i / srcA.cols;
            int col = i % srcA.cols;
            if (row == 0 || row == srcA.rows - 1 || col == 0 || col == srcA.cols - 1
                || binary.at<uchar>(row, col) == 0)
                continue;
            
            X.at<Vec3f>(i, 1) = (X.at<Vec3f>(i - srcA.cols, 1) + X.at<Vec3f>(i + srcA.cols, 0)
                    + X.at<Vec3f>(i - 1, 1) + X.at<Vec3f>(i + 1, 0) - lap.at<Vec3f>(row, col)) / 4;
        }
        
        Mat color = X.col(1) - X.col(0);
        
        for (int i = 0; i < srcA.rows * srcA.cols; i++) {
            int row = i / srcA.cols;
            int col = i % srcA.cols;
            if (row == 0 || row == srcA.rows - 1 || col == 0 || col == srcA.cols - 1
                || binary.at<uchar>(row, col) == 0)
                delta.at<float>(i, 0) = 0.0f;
            else
                delta.at<float>(i, 0) = norm(color.at<Vec3f>(i, 0), NORM_INF);
        }
        
        X.col(1).copyTo(X.col(0));
        
    } while (norm(delta, NORM_INF) > 1);
    
    
    srcA.copyTo(dst);
    X.convertTo(X, CV_8UC3);
    for (int i = 0; i < srcA.rows * srcA.cols; i++) {
        int row = i / srcA.cols;
        int col = i % srcA.cols;
        int delta = 2;
        if (row - delta >= 0 && row + delta < srcA.rows
            && col - delta >= 0 && col + delta < srcA.cols
            && binary.at<uchar>(row - delta, col) != 0
            && binary.at<uchar>(row, col - delta) != 0
            && binary.at<uchar>(row, col + delta) != 0
            && binary.at<uchar>(row + delta, col) != 0) {
            
            dst.at<Vec3b>(row, col) = srcB.at<Vec3b>(row, col);
        }
        //else
            dst.at<Vec3b>(row, col) = X.at<Vec3b>(i, 1);
    }
    
    imshow("Blender", dst);
    imwrite("Blender.jpg", dst);
    waitKey();
}




int main(int argc, const char * argv[])
{
    Mat srcA = imread("bg.jpg");
    Mat srcB = imread("fg.jpg");
    
    assert(srcA.rows == srcB.rows && srcA.cols == srcB.cols);
    
    Mat dst(srcA.size(), CV_8UC3);
    
    //naiveBlender(srcA, srcB, dst);
    possionBlender(srcA, srcB, dst);
    
    return 0;
}
