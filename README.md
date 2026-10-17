# Image Processing Pipeline

A C++ image-processing project that implements common image-processing operations from scratch and compares their results and performance with OpenCV implementations.

## Features

The project implements the following operations:
- Grayscale conversion
- Gaussian blur
- Median filtering
- Sobel edge detection

For each operation, the custom implementation is compared with the corresponding OpenCV implementation using:
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Maximum pixel difference
- Execution time

The project also generates a CSV file containing the comparison results.

## Technologies Used
- C++
- OpenCV
- CMake
- 
## Project Structure
```text
image-processing/
│
├── output/
│   ├── 0_original.png
│   ├── gray_mine.png
│   ├── gray_opencv.png
│   ├── gaussian_mine.png
│   ├── gaussian_opencv.png
│   ├── median_mine.png
│   ├── median_opencv.png
│   ├── sobel_mine.png
│   ├── sobel_opencv.png
│   ├── sobel_difference_x50.png
│   └── results.csv
│
├── main.cpp
├── cmakelists.txt
├── .gitignore
└── README.md
