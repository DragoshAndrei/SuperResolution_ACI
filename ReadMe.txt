# SuperResolution_ACI
Research for resolution enhancement based on multiple low resolution images

# Git Repository
https://github.com/DragoshAndrei/SuperResolution_ACI.git

## How to run:
- Extract MissingDlls.rar files in the same folder
- Open .sln solution file with Visual Studio
- Put the video you want to convert in the same folder as the .sln file
- Inside the code, under "//Superres Param:" set the desired parameters (including the input video name + extension and desired output video name).
- Run the code
  - Must have cuda installed and reference the header file from Solution Properties -> C/C++ -> Additional Include Directories
- The before and after frames + report are stored inside CompareFrames folder
- The Super Rez. video will be stored in the solution folder
