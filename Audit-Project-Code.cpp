#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp> 

#include <iostream>
#include<Windows.h>

using namespace cv;
using namespace std;

std::vector<cv::Point> centers,Rcenters;
cv::Point lastPoint;
cv::Point mousePoint;

void LeftClick()
{
    INPUT    Input = { 0 };
    // left down 
    Input.type = INPUT_MOUSE;
    Input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    ::SendInput(1, &Input, sizeof(INPUT));

    // left up
    ::ZeroMemory(&Input, sizeof(INPUT));
    Input.type = INPUT_MOUSE;
    Input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    ::SendInput(1, &Input, sizeof(INPUT));
}

void RightClick()
{
    INPUT    Input = { 0 };
    // right down 
    Input.type = INPUT_MOUSE;
    Input.mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
    ::SendInput(1, &Input, sizeof(INPUT));

    // right up
    ::ZeroMemory(&Input, sizeof(INPUT));
    Input.type = INPUT_MOUSE;
    Input.mi.dwFlags = MOUSEEVENTF_RIGHTUP;
    ::SendInput(1, &Input, sizeof(INPUT));
}

void changeMouse(cv::Mat& frame, cv::Point& location)
{
    if (location.x > frame.cols) location.x = frame.cols;
    if (location.x < 0) location.x = 0;

    if (location.y > frame.rows) location.y = frame.rows;
    if (location.y < 0) location.y = 0;
    SetCursorPos(location.x,location.y);
}

cv::Point stabilize(std::vector<cv::Point>& points, int windowSize)
{
    float sumX = 0;
    float sumY = 0;
    int count = 0;

    int j=0;

    if ((int)(points.size() - windowSize) > 0)
        j = (int)(points.size() - windowSize);

    for (int i = j; i < points.size(); i++)
    {
        sumX += points[i].x;
        sumY += points[i].y;
        ++count;

        if ((int)(points.size() - windowSize) > 0)
            j = (int)(points.size() - windowSize);
        else
            j = 0;
    }

    if (count > 0)
    {
        sumX /= count;
        sumY /= count;
    }
    return cv::Point(sumX, sumY);
}

cv::Vec3f getEyeball(cv::Mat& eye, std::vector<cv::Vec3f>& circles)
{
    std::vector<int> sums(circles.size(), 0);
    for (int y = 0; y < eye.rows; y++)
    {
        uchar* ptr = eye.ptr<uchar>(y);
        for (int x = 0; x < eye.cols; x++)
        {
            int value = static_cast<int>(*ptr);
            for (int i = 0; i < circles.size(); i++)
            {
                cv::Point center((int)std::round(circles[i][0]), (int)std::round(circles[i][1]));
                int radius = (int)std::round(circles[i][2]);
                if (std::pow(x - center.x, 2) + std::pow(y - center.y, 2) < std::pow(radius, 2))
                {
                    sums[i] += value;
                }
            }
            ++ptr;
        }
    }
    int smallestSum = 9999999;
    int smallestSumIndex = -1;
    for (int i = 0; i < circles.size(); i++)
    {
        if (sums[i] < smallestSum)
        {
            smallestSum = sums[i];
            smallestSumIndex = i;
        }
    }
    return circles[smallestSumIndex];
}

cv::Rect getLeftEye(std::vector<cv::Rect> &eyes)
{
    int left = INT_MAX;
    int leftIndex = -1;

    for (int i = 0; i < eyes.size(); i++)
    {
        if (eyes[i].tl().x < left)
        {
            left = eyes[i].tl().x;
            leftIndex = i;
        }
    }

    return eyes[leftIndex];
}

cv::Rect getRightEye(std::vector<cv::Rect>& eyes)
{
    int right = INT_MIN;
    int rightIndex = -1;

    for (int i = 0; i < eyes.size(); i++)
    {
        if (eyes[i].tl().x > right)
        {
            right = eyes[i].tl().x;
            rightIndex = i;
        }
    }

    return eyes[rightIndex];
}

void detectEyes(cv::Mat &frame, cv::CascadeClassifier &faceCascade, cv::CascadeClassifier &eyeCascade)
{
    cv::Mat grayscale;
    cv::Mat grayeye;
    cv::Mat Rgrayeye;
    cv::COLOR_BGR2GRAY;
    cv::CASCADE_SCALE_IMAGE;
    cv::HOUGH_GRADIENT;
    cv::cvtColor(frame, grayscale, COLOR_BGR2GRAY); // convert image to grayscale
    cv::equalizeHist(grayscale, grayscale); // enhance image contrast 
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayscale, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, cv::Size(150, 150));

    if (faces.size() == 0)
    {
        RightClick();
        return;
    }
    cv::Mat face = frame(faces[0]); // crop the face
    std::vector<cv::Rect> eyes;
    eyeCascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, cv::Size(28, 28)); // same thing as above  
    
    if (eyes.size() == 0)
    {
        return;
    }
    else if (eyes.size() == 1)
    {
        LeftClick();
        return;
    }

    rectangle(frame, faces[0].tl(), faces[0].br(), cv::Scalar(255, 0, 0), 2);
    
    for (cv::Rect& eye : eyes)
    {
        rectangle(frame, faces[0].tl() + eye.tl(), faces[0].tl() + eye.br(), cv::Scalar(0, 255, 0), 2);
    }
    cv::Rect Lefteye = getLeftEye(eyes);
    cv::Rect Righteye = getRightEye(eyes);
    cv::Mat eye = face(Lefteye); // crop the leftmost eye
    cv::Mat Reye = face(Righteye); // crop the leftmost eye

    cv::cvtColor(eye, grayeye, COLOR_BGR2GRAY);
    cv::equalizeHist(grayeye, grayeye);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(grayeye, circles, HOUGH_GRADIENT, 1, grayeye.cols / 8, 250, 15, grayeye.rows / 8, grayeye.rows / 3);

    if (circles.size() == 0)
    {
        
        return;
    }

    cv::Vec3f eyeball = getEyeball(grayeye, circles);
    cv::Point center(eyeball[0], eyeball[1]);
    centers.push_back(center);
    center = stabilize(centers, 5); // we are using the last 5
    cv::circle(frame, faces[0].tl() + Lefteye.tl() + center,10, cv::Scalar(0, 0, 255), 2);
    cv::circle(grayeye, center, 10, cv::Scalar(255, 255, 255), 2);
    
    if (centers.size() > 1)
    {
        //cout << "Moving Cursor";
        cv::Point diff;
        diff.x = (center.x - lastPoint.x) * 20;
        diff.y = (center.x - lastPoint.y) * -50; // diff in y is higher because it's "harder" to move the eyeball up/down instead of left/right
        mousePoint += diff;
    }
    lastPoint = center;
    cv::imshow("Left Eye", eye);


    /// <summary>
    
    cv::cvtColor(Reye, Rgrayeye, COLOR_BGR2GRAY);
    cv::equalizeHist(Rgrayeye, Rgrayeye);
    std::vector<cv::Vec3f> Rcircles;
    cv::HoughCircles(Rgrayeye, Rcircles, HOUGH_GRADIENT, 1, Rgrayeye.cols / 8, 250, 15, Rgrayeye.rows / 8, Rgrayeye.rows / 3);

    if (Rcircles.size() == 0)
    {
        return;
    }

    cv::Vec3f Reyeball = getEyeball(Rgrayeye, Rcircles);
    cv::Point Rcenter(Reyeball[0], Reyeball[1]);
    Rcenters.push_back(Rcenter);
    Rcenter = stabilize(Rcenters, 5); // we are using the last 5
    cv::circle(frame, faces[0].tl() + Righteye.tl() + Rcenter, 10, cv::Scalar(0, 0, 255), 2);
    cv::circle(grayeye, Rcenter, 10, cv::Scalar(255, 255, 255), 2);
}


int main()
{
    cv::VideoCapture cap(0); // the fist webcam connected to your PC
    cv::CascadeClassifier faceCascade;
    cv::CascadeClassifier eyeCascade;
    cv::Mat frame;
    
    if (!faceCascade.load("./haarcascade_frontalface_alt.xml"))
    {
        cout << "Could not load face detector.";
        return -1;
    }
    if (!eyeCascade.load("./haarcascade_eye_tree_eyeglasses.xml"))
    {
        cout<< "Could not load eye detector.";
        return -1;
    }

    if (!cap.isOpened())
    {
        std::cerr << "Webcam not detected." << std::endl;
        return -1;
    }
    mousePoint = cv::Point(600, 600);

    while (1)
    {
        cap >> frame; // outputs the webcam image to a Mat
        detectEyes(frame, faceCascade, eyeCascade);
        changeMouse(frame, mousePoint);
        //mousePoint = cv::Point(600, 600);
        cv::imshow("Webcam", frame); // displays the Mat
        if (cv::waitKey(30) >= 0) break; // takes 30 frames per second. if the user presses any button, it stops from showing the webcam
    }
	return 0;
}