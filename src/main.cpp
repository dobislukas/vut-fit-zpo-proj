/*
Lukas Dobis 
xdobis01
ZPO 2020/2021
Image segmentation based on optical flow
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//#define COMPLEXITY_TEST /// Define for measuring average time of computing segmented image.

#ifdef COMPLEXITY_TEST
#include <chrono>
#endif

using namespace cv;
using namespace std;

void dilation(const Mat&, Mat&, int, int);
void erosion(const Mat&, Mat&, int, int);

int main(int argc, char** argv )
{   
	int opt;
	bool save_video_flag = false;
	bool binary_output_flag = false;
	float comp_size = 0.6;
	float final_size = 1.0;
	string save_filepath = "results/segmentation.avi";
    
    // Parsing input flags
    while ((opt = getopt(argc, argv, "s:c:f:b")) != -1) 
    {
		switch (opt) 
		{	
			case 's':
				save_video_flag = true;
				save_filepath = optarg;
				break;
			case 'c':
				comp_size = atof(optarg);
				break;
			case 'f':
				final_size = atof(optarg);
				break;
			case 'b':
				binary_output_flag = true;
				break;
			default:
				cerr << "USAGE: segmentVideo [-s SAVE_FILEPATH] [-c COMPUTE_SIZE] [-f FINAL_SIZE] [-b BINARY_OUTPUT] VIDEO_FILEPATH\n";
				exit(EXIT_FAILURE);
		}
    }
    
    // Test last argument
  	if (optind >= argc) {
        cerr << "Expected video filepath argument after options\n";
        exit(EXIT_FAILURE);
    }
	
	// Create a VideoCapture and VideoWrite objects and open the input file
	VideoCapture capture(argv[optind]);
    int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    
	VideoWriter video(save_filepath,
                      VideoWriter::fourcc('M','J','P','G'), 
                      30, Size(frame_width*final_size, frame_height*final_size)); 
    
    // Test if camera opened successfully
    if (!capture.isOpened())
    {
        cerr << "Unable to open file!" << endl;
	    exit(EXIT_FAILURE);
    }
    
    // Load, resize to computation size and convert to grayscale first image
    Mat frame1_raw, frame1, prvs, output_image, final_image;
    
    capture >> frame1_raw;
    resize(frame1_raw, frame1, Size(), comp_size, comp_size);
    cvtColor(frame1, prvs, COLOR_BGR2GRAY);
    
    // Initialization for measurement of average segmented image processing time
	#ifdef COMPLEXITY_TEST
	unsigned int processed_img_counter = 0; 
    chrono::time_point<chrono::high_resolution_clock> start, end;
	start = chrono::high_resolution_clock::now();
    #endif
        
    // Main cycle for processing video,
    // optical flow is computed between image pair "previous" and "next"
	while(1)
	{   
        // Load, resize and convert to grayscale image
        Mat frame2_raw, frame2, empty, next;
        capture >> frame2_raw;
    	capture >> empty; // Skip one image for better speed and noise reduction
        
        // Test if there is image loaded, otherwise end cycle
        if (frame2_raw.empty())
            break;
        
        resize(frame2_raw, frame2, Size(), comp_size, comp_size);
        cvtColor(frame2, next, COLOR_BGR2GRAY);
	    
	    // Calculate Farneback optical flow
        Mat flow(prvs.size(), CV_32FC2);
        
        // calcOpticalFlowFarneback PRV NEXT FLOW PYRAMID_SCALE LEVELS WINSIZE ITERATION POLY_N POLY_SIG GAUSSIAN_WINDOW_FLAG
        calcOpticalFlowFarneback( prvs, next, flow,    0.5,       4,     25,      2,       7,     1.2, OPTFLOW_FARNEBACK_GAUSSIAN);  
        
        // Extract flow, convert to Polar system and normalize values
        Mat flow_parts[2];
        split(flow, flow_parts);
        Mat magnitude, angle, magn_norm;
        cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        angle *= ((1.f / 360.f) * (180.f / 255.f));
	    
	    // Apply thresholding and morfological operators
	    Mat tmp1, tmp2;
	    threshold( magnitude, tmp1, 2, 255, cv::THRESH_BINARY);
        dilation(tmp1, tmp2, 0, 2);
        erosion(tmp2, tmp1, 0, 12);
        dilation(tmp1, tmp2, 0, 10);
	    
	    // Depending on -b flag, create grayscale or BGR output
	    if (binary_output_flag)
	    {   
	        // Create grayscale image
	        tmp2.convertTo(output_image, CV_8UC1);
	    }
	    else 
	    {
            // Create hsv image and convert to BGR
            Mat _hsv[3], hsv, hsv8;
            _hsv[0] = angle;
            _hsv[1] = Mat::ones(angle.size(), CV_32F);
            _hsv[2] = tmp2;
            merge(_hsv, 3, hsv);
            hsv.convertTo(hsv8, CV_8U, 255.0);
            cvtColor(hsv8, output_image, COLOR_HSV2BGR);
	    }

     	// Resize to final image size
        resize(output_image, final_image, Size(frame_width*final_size, frame_height*final_size), 0, 0);
     
        // If -s flag is set, write image
        if (save_video_flag)
        {
            video.write(final_image);
        }
        
        // Show image
        imshow("Segmentation", final_image);
        
        // Exit after pressing "q" or "Esc" key
        int keyboard = waitKey(1);
        if (keyboard == 'q' || keyboard == 27)
            break;
        
        // Set new previous image to current next
    	prvs = next;
    	
    	// Increment number of computed output images
    	#ifdef COMPLEXITY_TEST
    	++processed_img_counter;
	    #endif
	}
    
    // Compute and print average time for computing segmented image
	#ifdef COMPLEXITY_TEST
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> average_time = (end - start) / processed_img_counter;
    cout << "Average computational time of segmented image: " << average_time.count() << "s" << endl;
    #endif
	
	// When everything finished, release the video objects
	video.release();
	capture.release();
	
	// If video is not to be saved, remove empty video file
    if (!save_video_flag)
    {
		if (remove(save_filepath.c_str()) != 0)
		{
			cerr << "Video File deletion failed";
		    exit(EXIT_FAILURE);
		}
    }
	 
	// Closes all the frames
	destroyAllWindows();

    exit(EXIT_SUCCESS);
}

// Function for applying dilatation to image
void dilation(const Mat& src, Mat& dst, int dilation_type, int dilation_size)
{
  if( dilation_type == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_type == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_type == 2) { dilation_type = MORPH_ELLIPSE; }
  Mat element = getStructuringElement( dilation_type,
                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                       Point( dilation_size, dilation_size ) );
  dilate( src, dst, element );
}

// Function for applying erosion to image
void erosion(const Mat& src, Mat& dst, int erosion_type, int erosion_size)
{
  if( erosion_type == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_type == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_type == 2) { erosion_type = MORPH_ELLIPSE; }
  Mat element = getStructuringElement( erosion_type,
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                       Point( erosion_size, erosion_size ) );
  erode( src, dst, element );
}

