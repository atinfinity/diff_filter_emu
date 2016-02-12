#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <cmath>

//#define GAMMA_IMAGE_OUTPUT

float calc_gamma(float val, double gm)
{
    CV_Assert(val >= 0.);
    if (val > 1.0)
    {
        val = 1.0;
    }
    return((float)pow(val, gm));
}

class ApplyGamma_Invoker : public cv::ParallelLoopBody
{
private:
    cv::Mat _src;
    cv::Mat _dst;
    double _gamma;
public:
    ApplyGamma_Invoker(const cv::Mat& src, cv::Mat& dst, double gamma)
        : _src(src), _dst(dst), _gamma(gamma) { }
    void operator() (const cv::Range& range) const
    {
        int row0 = range.start;
        int row1 = range.end;
        cv::Mat srcStripe = _src.rowRange(row0, row1);
        cv::Mat dstStripe = _dst.rowRange(row0, row1);
        for (int y = 0; y < srcStripe.rows; y++)
        {
            for (int x = 0; x < srcStripe.cols; x++)
            {
                dstStripe.at<cv::Vec3f>(y, x)[0] = calc_gamma(srcStripe.at<cv::Vec3f>(y, x)[0], _gamma);
                dstStripe.at<cv::Vec3f>(y, x)[1] = calc_gamma(srcStripe.at<cv::Vec3f>(y, x)[1], _gamma);
                dstStripe.at<cv::Vec3f>(y, x)[2] = calc_gamma(srcStripe.at<cv::Vec3f>(y, x)[2], _gamma);
            }
        }
    }
};

class ApplyGain_Invoker : public cv::ParallelLoopBody
{
private:
    cv::Mat _src;
    cv::Mat _dst;
    double _gain;
public:
    ApplyGain_Invoker(const cv::Mat& src, cv::Mat& dst, double gain)
        : _src(src), _dst(dst), _gain(gain) { }
    void operator() (const cv::Range& range) const
    {
        int row0 = range.start;
        int row1 = range.end;
        cv::Mat srcStripe = _src.rowRange(row0, row1);
        cv::Mat dstStripe = _dst.rowRange(row0, row1);
        for (int y = 0; y < srcStripe.rows; y++)
        {
            for (int x = 0; x < srcStripe.cols; x++)
            {
                dstStripe.at<cv::Vec3f>(y, x)[0] += (srcStripe.at<cv::Vec3f>(y, x)[0] * _gain);
                dstStripe.at<cv::Vec3f>(y, x)[1] += (srcStripe.at<cv::Vec3f>(y, x)[1] * _gain);
                dstStripe.at<cv::Vec3f>(y, x)[2] += (srcStripe.at<cv::Vec3f>(y, x)[2] * _gain);
            }
        }
    }
};

int soft_filter(cv::Mat &src_img, cv::Mat &dst_img, int iteration, double decay_factor, double decay_offset, double gamma)
{
    CV_Assert(src_img.channels() == 3);

    cv::Mat fsrc_img;
    src_img.convertTo(fsrc_img, CV_32FC3, 1.0 / 255.);

    cv::Mat fsrc_img_gamma = cv::Mat::zeros(fsrc_img.size(), fsrc_img.type());
    cv::parallel_for_
    (
        cv::Range(0, fsrc_img_gamma.rows),
        ApplyGamma_Invoker(fsrc_img, fsrc_img_gamma, gamma)
    );
#ifdef GAMMA_IMAGE_OUTPUT
    cv::Mat tmp;
    fsrc_img_gamma.convertTo(tmp, CV_8UC3, 255.);
    cv::imwrite("gamma.tif", tmp);
#endif

    cv::Mat fdst_img = fsrc_img_gamma.clone();

    unsigned int sigma = 2;
    std::vector<cv::Mat> img_pyr;
    float gain;

    for (int i = 0; i < iteration; i++, sigma <<= 1)
    {
        gain = (float)pow(decay_factor, -((double)i + decay_offset));

        std::cout << "Sigma=" << sigma << std::endl;
        std::cout << "Gain=" << gain << std::endl;

        cv::Mat fpyrsrc = fsrc_img_gamma.clone(), fpyrdst;
        cv::GaussianBlur(fpyrsrc, fpyrdst, cv::Size(7, 7), sigma, sigma, cv::BORDER_CONSTANT);
        cv::parallel_for_
        (
            cv::Range(0, fdst_img.rows),
            ApplyGain_Invoker(fpyrdst, fdst_img, gain)
        );
    }

    cv::Mat fdst_img_ungamma = cv::Mat::zeros(fdst_img.size(), fdst_img.type());
    double gm = 1.0 / gamma;
    cv::parallel_for_
    (
        cv::Range(0, fdst_img_ungamma.rows),
        ApplyGamma_Invoker(fdst_img, fdst_img_ungamma, gm)
    );

    fdst_img_ungamma.convertTo(dst_img, CV_8UC3, 255.);

    return 0;
}

void usage(void)
{
    std::cout << "usage: diff_filter_emu -i=input_image [-o=output_image] [-n=iteration] [-d=decay_factor] [-f=decay_offset] [-g=gamma] [-s]" << std::endl;
}

int main(int argc, char* argv[])
{
    const cv::String keys =
        "{s||}"
        "{h||}"
        "{i||}"
        "{o|result.tif|}"
        "{n|5|}"
        "{d|5.0|}"
        "{f|0.1|}"
        "{g|1.3|}"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    if((argc < 2) || (parser.has("h")))
    {
        usage();
        return -1;
    }

    std::string input_img  = parser.get<std::string>("i");
    std::string output_img = parser.get<std::string>("o");
    int iteration          = parser.get<int>("n");
    double decay_factor    = parser.get<double>("d");
    double decay_offset    = parser.get<double>("f");
    double gamma           = parser.get<double>("g");
    bool show_result       = parser.has("s") ? true : false;

    std::cout << "Settings..." << std::endl;
    std::cout << "Input image=" << input_img << std::endl;
    std::cout << "Output image=" << output_img << std::endl;
    std::cout << "Iteration=" << iteration << std::endl;
    std::cout << "Decay factor=" << decay_factor << std::endl;
    std::cout << "Decay offset=" << decay_offset << std::endl;
    std::cout << "Gamma=" << gamma << std::endl;
    std::cout << "Show result=" << show_result << std::endl << std::endl;

    cv::Mat src_img = cv::imread(input_img);
    if(src_img.empty())
    {
        std::cout << input_img << " does not exist." << std::endl;
        return -1;
    }

    cv::Mat dst_img;

    double f = 1000.0f / cv::getTickFrequency();
    int64 start = cv::getTickCount();
    soft_filter(src_img, dst_img, iteration, decay_factor, decay_offset, gamma);
    int64 end = cv::getTickCount();
    std::cout << (end - start) * f << "[ms]" << std::endl;

    cv::imwrite(output_img, dst_img);

    if(show_result)
    {
        cv::namedWindow("original", cv::WINDOW_AUTOSIZE);
        cv::imshow("original", src_img);
        cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
        cv::imshow("result", dst_img);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return 0;
}
