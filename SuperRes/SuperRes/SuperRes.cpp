#include <iostream>
#include <iomanip>
#include <string>
#include <ctype.h>
#include <fstream>
#include <direct.h>

#include "opencv2/core/core.hpp"
//#include "opencv2/core/cuda/utility.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/superres.hpp"
#include "opencv2/superres/optical_flow.hpp"
#include "opencv2/opencv_modules.hpp"



using namespace std;
using namespace cv;
using namespace cv::superres;

#define MEASURE_TIME(op) \
    { \
        TickMeter tm; \
        tm.start(); \
        op; \
        tm.stop(); \
        cout << tm.getTimeSec() << " sec" << endl; \
		file << tm.getTimeSec() << " sec" << endl; \
    }

void extract_frames(const string& videoFilePath, vector<Mat>& frames);
void save_frames(vector<Mat>& frames, const string& outputDir);


static Ptr<cv::superres::DenseOpticalFlowExt> createOptFlow(const string& name, bool useGpu)
{
	if (name == "farneback")
	{
		if (useGpu)
			return cv::superres::createOptFlow_Farneback_CUDA();
		else
			return cv::superres::createOptFlow_Farneback();
	}
	/*else if (name == "simple")
		return createOptFlow_Simple();*/
	else if (name == "tvl1")
	{
		if (useGpu)
			return cv::superres::createOptFlow_DualTVL1_CUDA();
		else
			return cv::superres::createOptFlow_DualTVL1();
	}
	else if (name == "brox")
		return cv::superres::createOptFlow_Brox_CUDA();
	else if (name == "pyrlk")
		return cv::superres::createOptFlow_PyrLK_CUDA();
	else
		cerr << "Incorrect Optical Flow algorithm - " << name << endl;

	return Ptr<cv::superres::DenseOpticalFlowExt>();
}

int main(int argc, const char* argv[])
{

	/*CommandLineParser cmd(argc, argv,
		"{ v video      |           | Input video (mandatory)}"
		"{ o output     |           | Output video }"
		"{ s scale      | 4         | Scale factor }"
		"{ i iterations | 180       | Iteration count }"
		"{ t temporal   | 4         | Radius of the temporal search area }"
		"{ f flow       | farneback | Optical flow algorithm (farneback, tvl1, brox, pyrlk) }"
		"{ g gpu        | false     | CPU as default device, cuda for CUDA }"
		"{ h help       | false     | Print help message }"
	);*/


	//const string inputVideoName = cmd.get<string>("video");

	/*if (cmd.get<bool>("help") || inputVideoName.empty())
	{
		cout << "This sample demonstrates Super Resolution algorithms for video sequence" << endl;
		cmd.printMessage();
		return EXIT_SUCCESS;
	}*/

	/*const string outputVideoName = cmd.get<string>("output");
	const int scale = cmd.get<int>("scale");
	const int iterations = cmd.get<int>("iterations");
	const int temporalAreaRadius = cmd.get<int>("temporal");
	const string optFlow = cmd.get<string>("flow");
	string gpuOption = cmd.get<string>("gpu");*/

	//Superres Param:
	string inputVideoName = "planet.avi";
	string outputVideoName = "planetSuperRez.avi";
	const int iterations = 10;
	const int scale = 2;
	const int temporalAreaRadius = 4;
	const string optFlow = "tvl1";
	string gpuOption = "cuda";

	//Create a file to write report
	//fstream ("CompareFrames/outro.txt");
	size_t lastindex = inputVideoName.find_last_of(".");
	string rawVideoName = inputVideoName.substr(0, lastindex);

	string sDirectoryPath = "CompareFrames/" + rawVideoName;
	string sDirectoryPathBP = "CompareFrames/" + rawVideoName + "/beforeProcess";
	string sDirectoryPathPP = "CompareFrames/" + rawVideoName + "/postProcess";


	//Extract frames before Superres
	vector<Mat> preProcessedFrames;
	extract_frames(inputVideoName, preProcessedFrames);
	//Manualy set output dir1
	if (_mkdir(sDirectoryPath.c_str()) != 0)
	{
		cout << "Could not create folder for:" + rawVideoName << "or path already exists." << endl;
	}
	//Manualy set output dir2
	if (_mkdir(sDirectoryPathBP.c_str()) != 0)
	{
		cout << "Could not create subfolder beforeProcessFrames for: " + rawVideoName << "or path already exists." << endl;
	}
	//save_frames(preProcessedFrames,)
	save_frames(preProcessedFrames, sDirectoryPathBP + "/");

	//Create Report file

	string reportPath = "CompareFrames/" + rawVideoName;
	ofstream file(reportPath + "/Report" + toUpperCase(optFlow) + toUpperCase(gpuOption) + ".txt");

	//Start Superres

	std::transform(gpuOption.begin(), gpuOption.end(), gpuOption.begin(), ::tolower);

	bool useCuda = gpuOption.compare("cuda") == 0;
	Ptr<SuperResolution> superRes;

	if (useCuda)
		superRes = createSuperResolution_BTVL1_CUDA();
	else
		superRes = createSuperResolution_BTVL1();

	Ptr<cv::superres::DenseOpticalFlowExt> of = createOptFlow(optFlow, useCuda);

	if (of.empty())
		return EXIT_FAILURE;
	superRes->setOpticalFlow(of);

	superRes->setScale(scale);
	superRes->setIterations(iterations);
	superRes->setTemporalAreaRadius(temporalAreaRadius);

	Ptr<FrameSource> frameSource;
	if (useCuda)
	{
		// Try to use gpu Video Decoding
		try
		{
			frameSource = createFrameSource_Video_CUDA(inputVideoName);
			Mat frame;
			frameSource->nextFrame(frame);
		}
		catch (const cv::Exception&)
		{
			frameSource.release();
		}
	}
	if (!frameSource)
		frameSource = createFrameSource_Video(inputVideoName);

	// skip first frame, it is usually corrupted
	{
		Mat frame;
		frameSource->nextFrame(frame);
		cout << "Input           : " << inputVideoName << " " << frame.size() << endl;
		cout << "Scale factor    : " << scale << endl;
		cout << "Iterations      : " << iterations << endl;
		cout << "Temporal radius : " << temporalAreaRadius << endl;
		cout << "Optical Flow    : " << optFlow << endl;
		cout << "Mode            : " << (useCuda ? "CUDA" : "CPU") << endl;
		file << "Input           : " << inputVideoName << " " << frame.size() << endl;
		file << "Scale factor    : " << scale << endl;
		file << "Iterations      : " << iterations << endl;
		file << "Temporal radius : " << temporalAreaRadius << endl;
		file << "Optical Flow    : " << optFlow << endl;
		file << "Mode            : " << (useCuda ? "CUDA" : "CPU") << endl;
	}



	superRes->setInput(frameSource);

	VideoWriter writer;

	for (int i = 0;; ++i)
	{
		cout << '[' << setw(3) << i << "] : " << flush;
		file << '[' << setw(3) << i << "] : " << flush;
		Mat result;

		MEASURE_TIME(superRes->nextFrame(result));


		//superRes->nextFrame(result);
		if (result.empty())
			break;

		imshow("Super Resolution", result);

		if (waitKey(1000) > 0)
			break;

		if (!outputVideoName.empty())
		{
			if (!writer.isOpened())
				writer.open(outputVideoName, VideoWriter::fourcc('X', 'V', 'I', 'D'), 25.0, result.size());
			writer << result;
		}
	}
	file.close();
	writer.release();


	//Extract frames after superres
	vector<Mat> postProcessedFrames;
	extract_frames(outputVideoName, postProcessedFrames);
	//Manualy set output dir2
	if (_mkdir(sDirectoryPathPP.c_str()) != 0)
	{
		cout << "Could not create subfolder afterProcessFrames for: " + outputVideoName << "or path already exists." << endl;
	}
	//save_frames(preProcessedFrames,)
	save_frames(postProcessedFrames, sDirectoryPathPP + "/");


	return 0;
}

void extract_frames(const string& videoFilePath, vector<Mat>& frames)
{
	try
	{
		//open the video file
		VideoCapture cap(videoFilePath); // open the video file
		if (!cap.isOpened())  // check if we succeeded
			CV_Error(CV_StsError, "Can not open Video file");

		//cap.get(CV_CAP_PROP_FRAME_COUNT) contains the number of frames in the video;
		for (int frameNum = 0; frameNum < cap.get(CV_CAP_PROP_FRAME_COUNT); frameNum++)
		{
			Mat frame;
			cap >> frame; // get the next frame from video
			frames.push_back(frame);
		}
	}
	catch (cv::Exception& e)
	{
		cerr << e.msg << endl;
		exit(1);
	}
}

void save_frames(vector<Mat>& frames, const string& outputDir)
{
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	int frameNumber = 1;

	for (std::vector<Mat>::iterator frame = frames.begin() + 1; frame != frames.end(); ++frame)
	{
		string filePath = outputDir + to_string(static_cast<long long>(frameNumber)) + ".jpg";
		imwrite(filePath, *frame, compression_params);
		frameNumber++;
	}


}
