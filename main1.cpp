#include <opencv2/opencv.hpp>
#include <iostream>
#include <queue>
#include <functional>


double fms(int i1, int j1, int i2, int j2, const cv::Mat &f, const cv::Mat &t) {
    double a1 = t.at<float>(i1, j1);
    double a2 = t.at<float>(i2, j2);
    double m = min(a1, a2);

    if (f.at<uchar>(i1, j1) != 2) {
        if (f.at<uchar>(i2, j2) != 2) {
            if (abs(a1 - a2) >= 1.0) {
                return 1.0 + m;
            }
            return 0.5 * (a1 + a2 + sqrt(2.0 - (a1 - a2) * (a1 - a2)));
        }
        return 1.0 + a1;
    }
    if (f.at<uchar>(i2, j2) != 2) {
        return 1.0 + a2;
    }
    return 1.0 + m;
}

double min4(double a1, double a2, double a3, double a4) {
    return std::min({a1, a2, a3, a4});
}

void inpaintPoint(int i, int j, const cv::Mat &f, const cv::Mat &t, cv::Mat &ret, int epsilon) {
    double radius_sqr = epsilon * epsilon;
    double grad_tx = 0.0, grad_ty = 0.0;

    if (f.at<uchar>(i, j + 1) != 2) {
        if (f.at<uchar>(i, j - 1) != 2) {
            grad_tx = 0.5 * (t.at<float>(i, j + 1) - t.at<float>(i, j - 1));
        } else {
            grad_tx = t.at<float>(i, j + 1) - t.at<float>(i, j);
        }
    } else if (f.at<uchar>(i, j - 1) != 2) {
        grad_tx = t.at<float>(i, j) - t.at<float>(i, j - 1);
    }

    if (f.at<uchar>(i + 1, j) != 2) {
        if (f.at<uchar>(i - 1, j) != 2) {
            grad_ty = 0.5 * (t.at<float>(i + 1, j) - t.at<float>(i - 1, j));
        } else {
            grad_ty = t.at<float>(i + 1, j) - t.at<float>(i, j);
        }
    } else if (f.at<uchar>(i - 1, j) != 2) {
        grad_ty = t.at<float>(i, j) - t.at<float>(i - 1, j);
    }

    int min_i = max(1, i - epsilon);
    int min_j = max(1, j - epsilon);
    int max_i = min(ret.rows - 1, i + epsilon + 1);
    int max_j = min(ret.cols - 1, j + epsilon + 1);

    Vec3d Ia(0, 0, 0);
    double s = 1.0e-20;

    for (int k = min_i; k < max_i; k++) {
        for (int l = min_j; l < max_j; l++) {
            double r_y = i - k, r_x = j - l;
            if (f.at<uchar>(k, l) != 2 && (r_x * r_x + r_y * r_y) <= radius_sqr) {
                double dst = 1.0 / ((r_x * r_x + r_y * r_y) * sqrt(r_x * r_x + r_y * r_y));
                double lev = 1.0 / (1.0 + abs(t.at<float>(k, l) - t.at<float>(i, j)));
                double dir_t = grad_tx * r_x + grad_ty * r_y;
                dir_t = max(dir_t, 0.000001);
                double w = abs(dst * lev * dir_t);
                Ia += w * ret.at<Vec3b>(k, l);
                s += w;
            }
        }
    }
    ret.at<Vec3b>(i, j) = Vec3b(0.5 + Ia[0] / s, 0.5 + Ia[1] / s, 0.5 + Ia[2] / s);
}

void telea(cv::Mat &f, cv::Mat &t, cv::Mat &ret, int epsilon, std::priority_queue<pair<float, cv::Point>, std::vector<std::pair<float, cv::Point>>, std::greater<>> &heap) {
    while (!heap.empty()) {
        auto [val_t, pt] = heap.top();
        heap.pop();
        int r = pt.y, c = pt.x;

        f.at<uchar>(r, c) = 0;
        for (auto [dr, dc] : std::vector<pair<int, int>>{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}) {
            int i = r + dr, j = c + dc;
            if (f.at<uchar>(i, j) == 2) {
                double dist = min4(
                    fms(i - 1, j, i, j - 1, f, t),
                    fms(i + 1, j, i, j - 1, f, t),
                    fms(i - 1, j, i, j + 1, f, t),
                    fms(i + 1, j, i, j + 1, f, t)
                );
                t.at<float>(i, j) = dist;
                inpaintPoint(i, j, f, t, ret, epsilon);
                f.at<uchar>(i, j) = 1;
                heap.emplace(dist, cv::Point(j, i));
            }
        }
    }
}

cv::Mat inpaint_telea(cv::Mat src, cv::Mat mask, int epsilon){

    CV_Assert(src.size() == mask.size());

    cv::Mat ret = src.clone();
    cv::Mat f = cv::Mat::zeros(mask.size(), CV_8U);
    cv::Mat t = Mat::zeros(mask.size(), CV_32F);
    std::priority_queue<std::pair<float, cv::Point>, std::vector<std::pair<float, cv::Point>>, greater<>> heap;

    // Initialize f
    for (int i = 1; i < mask.rows - 1; i++) {
        for (int j = 1; j < mask.cols - 1; j++) {
            if (mask.at<uchar>(i, j) > 0) {
                f.at<uchar>(i, j) = 2; // Pixels to be inpainted
            }
        }
    }

    // Initialize border pixels and heap
    for (int i = 1; i < mask.rows - 1; i++) {
        for (int j = 1; j < mask.cols - 1; j++) {
            if (f.at<uchar>(i, j) == 2) {
                if (f.at<uchar>(i - 1, j) == 0 || f.at<uchar>(i + 1, j) == 0 || f.at<uchar>(i, j - 1) == 0 || f.at<uchar>(i, j + 1) == 0) {
                    f.at<uchar>(i, j) = 1; // Border pixel
                    double dist = min4(
                        fms(i - 1, j, i, j - 1, f, t),
                        fms(i + 1, j, i, j - 1, f, t),
                        fms(i - 1, j, i, j + 1, f, t),
                        fms(i + 1, j, i, j + 1, f, t)
                    );
                    t.at<float>(i, j) = dist;
                    heap.emplace(dist, cv::Point(j, i));
                }
            }
        }
    }
    // Perform inpainting
    telea(f, t, ret, epsilon, heap);
    return ret;

}
int main(){
    cv::Mat src = cv::imread("../source.bmp");
    cv::Mat mask = cv::imread("../mask.png");
    cv::imwrite("mask.bmp",mask);
    return 0;
}