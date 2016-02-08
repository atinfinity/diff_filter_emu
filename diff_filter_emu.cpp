#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <math.h>

//#define GAMMA_IMAGE_OUTPUT

float calc_gamma(float val, double gm) {
    CV_Assert(val >= 0.);

    if (val > 1.0)
        val = 1.0;

    return((float)pow(val, gm));
}

int soft_filter(cv::Mat &src_img, cv::Mat &dst_img, int iteration, double decay_factor, double decay_offset, double gamma)
{
    CV_Assert(src_img.channels() == 3);

    cv::Mat fsrc_img;
    src_img.convertTo(fsrc_img, CV_32FC3, 1.0 / 255.);

    cv::Mat fsrc_img_gamma = cv::Mat::zeros(fsrc_img.size(), fsrc_img.type());

    {
        cv::MatIterator_<cv::Vec3f> it = fsrc_img.begin<cv::Vec3f>(), it_end = fsrc_img.end<cv::Vec3f>();
        cv::MatIterator_<cv::Vec3f> itg = fsrc_img_gamma.begin<cv::Vec3f>();
        for (; it != it_end; ++it, ++itg) {
            cv::Vec3f pix = *it, pix_gm;

            pix_gm[0] = calc_gamma(pix[0], gamma);
            pix_gm[1] = calc_gamma(pix[1], gamma);
            pix_gm[2] = calc_gamma(pix[2], gamma);

            *itg = pix_gm;
        }
#ifdef GAMMA_IMAGE_OUTPUT
        cv::Mat tmp;
        fsrc_img_gamma.convertTo(tmp, CV_8UC3, 255.);
        cv::imwrite("gamma.tif", tmp);
#endif
    }
    cv::Mat fdst_img = fsrc_img_gamma.clone();

    unsigned int sigma = 2;
    std::vector<cv::Mat> img_pyr;
    float gain;

    for (int i = 0; i < iteration; i++, sigma <<= 1) {
        gain = (float)pow(decay_factor, -((double)i + decay_offset));

        std::cout << "Sigma=" << sigma << std::endl;
        std::cout << "Gain=" << gain << std::endl;

        cv::Mat fpyrsrc = fsrc_img_gamma.clone(), fpyrdst;
        cv::GaussianBlur(fpyrsrc, fpyrdst, cv::Size(0, 0), sigma);

        cv::MatIterator_<cv::Vec3f> itdst = fdst_img.begin<cv::Vec3f>(), itdst_end = fdst_img.end<cv::Vec3f>();
        cv::MatIterator_<cv::Vec3f> itpyr = fpyrdst.begin<cv::Vec3f>();
        for (; itdst != itdst_end; ++itdst, ++itpyr) {
            cv::Vec3f pix_dst = *itdst;
            cv::Vec3f pix_pyr = *itpyr;

            pix_dst += (pix_pyr * gain);
            *itdst = pix_dst;
        }
    }

    cv::Mat fdst_img_ungamma = cv::Mat::zeros(fdst_img.size(), fdst_img.type());
    {
        double gm = 1.0 / gamma;
        cv::MatIterator_<cv::Vec3f> it = fdst_img.begin<cv::Vec3f>(), it_end = fdst_img.end<cv::Vec3f>();
        cv::MatIterator_<cv::Vec3f> itg = fdst_img_ungamma.begin<cv::Vec3f>();
        for (; it != it_end; ++it, ++itg) {
            cv::Vec3f pix = *it, pix_gm;

            pix_gm[0] = calc_gamma(pix[0], gm);
            pix_gm[1] = calc_gamma(pix[1], gm);
            pix_gm[2] = calc_gamma(pix[2], gm);

            *itg = pix_gm;
        }
    }

    fdst_img_ungamma.convertTo(dst_img, CV_8UC3, 255.);

    return 0;
}

int	iteration = 5;
double decay_factor = 2.0;
double decay_offset = 0.1;
double gamma = 1.8;
std::string input_img = ("");
std::string output_img = ("result.tif");
bool show_result = false;

void usage() {
    std::cout << "usage: diff_filter_emu -i input_image [-o output_image] [-n iteration] [-d decay_factor] [-f decay_offset] [-g gamma] [-s]" << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        usage();
        return -1;
    }

    {
        int result;

        while ((result = getopt(argc, argv, "shi:o:n:d:f:g:")) != -1){
            std::istringstream is;
            if (optarg) {
                std::string str(optarg);
                is.str(str);
            }
            switch (result){
            case 'i':
                is >> input_img;
                break;
            case 'o':
                is >> output_img;
                break;
            case 'n':
                is >> iteration;
                CV_Assert((iteration > 0) && (iteration <= 8));
                break;
            case 'd':
                is >> decay_factor;
                CV_Assert(decay_factor >= 1.0);
                break;
            case 'f':
                is >> decay_offset;
                CV_Assert(decay_offset >= 0.);
                break;
            case 'g':
                is >> gamma;
                CV_Assert(gamma > 0.);
                break;
            case 's':
                std::cout << "opt:s" << std::endl;
                show_result = true;
                break;
            case 'h':
            default:
                usage();
                return -1;
            }
        }
    }

    {
        std::cout << "Settings..." << std::endl;
        std::cout << "Input image=" << input_img << std::endl;
        std::cout << "Output image=" << output_img << std::endl;
        std::cout << "Iteration=" << iteration << std::endl;
        std::cout << "Decay factor=" << decay_factor << std::endl;
        std::cout << "Decay offset=" << decay_offset << std::endl;
        std::cout << "Gamma=" << gamma << std::endl;
        std::cout << "Show result=" << show_result << std::endl << std::endl;
    }

    cv::Mat src_img = cv::imread(input_img); //argv[1]);

    if (src_img.empty()) {
        std::cout << input_img << " does not exist." << std::endl;
        return -1;
    }

    cv::Mat dst_img;

    soft_filter(src_img, dst_img, iteration, decay_factor, decay_offset, gamma);

    cv::imwrite(output_img, dst_img);

    if (show_result) {
        cv::namedWindow("original", cv::WINDOW_NORMAL);
        cv::imshow("original", src_img);
        cv::namedWindow("result", cv::WINDOW_NORMAL);
        cv::imshow("result", dst_img);
        cv::waitKey(0);
    }

    return 0;
}
