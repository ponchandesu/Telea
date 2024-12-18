#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <queue>
#include <functional>
#include <utility>
#include <cmath>
using namespace cv;
using namespace std;


struct Compare {
    bool operator()(const pair<float, pair<int, int>>& a, const pair<float, pair<int, int>>& b) {
        // 比較基準として、pairの最初のfloat値を比較
        return a.first > b.first; // 最小のfloatを優先
    }
};

void edit_border(cv::Mat& img) {
    int lst_r = img.rows - 1; // 最後の行のインデックス
    int lst_c = img.cols - 1; // 最後の列のインデックス

    // 左の列を0に設定
    img.col(0).setTo(cv::Scalar(0));

    // 右の列を0に設定
    img.col(lst_c).setTo(cv::Scalar(0));

    // 上の行を0に設定
    img.row(0).setTo(cv::Scalar(0));

    // 下の行を0に設定
    img.row(lst_r).setTo(cv::Scalar(0));
}

double fms(int i1, int j1, int i2, int j2, const Mat &f, const Mat &t) {
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
    double a = std::min(a1, a2);
    a = std::min(a, a3);
    return std::min(a, a4);
}

void inpaintPoint(int i, int j, const Mat &f, const Mat &t, Mat &ret, int epsilon) {
    int radius_sqr = epsilon*epsilon;
    double grad_tx= 0.0, grad_ty = 0.0;
    if(f.at<float>(i, j+1) != 2){
        if(f.at<float>(i, j-1) != 2){
            grad_tx = 0.5 * (t.at<float>(i,j+1) - t.at<float>(i,j-1));
        }else{
            grad_tx = t.at<float>(i,j+1) - t.at<float>(i,j);
        }
    }else if(f.at<float>(i,j-1) != 2){
        grad_tx = t.at<float>(i,j) - t.at<float>(i,j-1);
    }

    if(f.at<float>(i+1,j) != 2){
        if(f.at<float>(i-1,j) != 2){
            grad_ty = 0.5 * (t.at<float>(i+1,j) - t.at<float>(i-1,j));
        }else{
            grad_ty = t.at<float>(i+1,j) - t.at<float>(i,j);
        }
    }else if(f.at<float>(i-1,j)!=2){
        grad_ty = t.at<float>(i,j) - t.at<float>(i -1, j);
    }

    int min_i = max(1,i-epsilon), min_j = max(1,j-epsilon);
    int max_i = min(ret.rows - 1, i + epsilon + 1), max_j = min(ret.cols - 1, j + epsilon + 1);

    if (ret.channels() == 3) {
        cv::Vec3f Ia(0.0f, 0.0f, 0.0f);  // 3 channels, initialized to 0
        cv::Vec3f Ix(0.0f, 0.0f, 0.0f);
        cv::Vec3f Iy(0.0f, 0.0f, 0.0f);
        float s = 1.0e-20f;
        for (int k = min_i; k < max_i; ++k) {
            for (int l = min_j; l < max_j; ++l) {
                float r_y = float(i - k);
                float r_x = float(j - l);
                if (f.at<uchar>(k, l) != 2 && (r_x * r_x + r_y * r_y) <= radius_sqr) {
                    float dst = 1.0f / ((r_x * r_x + r_y * r_y) * std::sqrt(r_x * r_x + r_y * r_y));
                    float lev = 1.0f / (1.0f + std::abs(t.at<float>(k, l) - t.at<float>(i, j)));
                    float dir_t = grad_tx * r_x + grad_ty * r_y;
                    dir_t = std::max(dir_t, 0.000001f);  // Prevent division by zero
                    float w = std::abs(dst * lev * dir_t);

                    // Accumulate weighted values for Ia
                    Ia += w * ret.at<cv::Vec3f>(k, l);

                    // Accumulate weight
                    s += w;
                }
            }
        }

        // Update the ret matrix with the computed value
        ret.at<cv::Vec3f>(i, j) = cv::Vec3f(0.5f, 0.5f, 0.5f) + Ia / s;
    }else{
        float Ia = 0.0f;  // 3 channels, initialized to 0
        float Ix = 0.0f;
        float Iy = 0.0f;
        float s = 1.0e-20f;
        for (int k = min_i; k < max_i; ++k) {
            for (int l = min_j; l < max_j; ++l) {
                float r_y = float(i - k);
                float r_x = float(j - l);
                if (f.at<uchar>(k, l) != 2 && (r_x * r_x + r_y * r_y) <= radius_sqr) {
                    float dst = 1.0f / ((r_x * r_x + r_y * r_y) * std::sqrt(r_x * r_x + r_y * r_y));
                    float lev = 1.0f / (1.0f + std::abs(t.at<float>(k, l) - t.at<float>(i, j)));
                    float dir_t = grad_tx * r_x + grad_ty * r_y;
                    dir_t = std::max(dir_t, 0.000001f);  // Prevent division by zero
                    float w = std::abs(dst * lev * dir_t);

                    // Accumulate weighted values for Ia
                    Ia += w * ret.at<float>(k, l);

                    // Accumulate weight
                    s += w;
                }
            }
        }
    }
    
    // double radius_sqr = epsilon * epsilon;
    // double grad_tx = 0.0, grad_ty = 0.0;

    // if (f.at<uchar>(i, j + 1) != 2) {
    //     if (f.at<uchar>(i, j - 1) != 2) {
    //         grad_tx = 0.5 * (t.at<float>(i, j + 1) - t.at<float>(i, j - 1));
    //     } else {
    //         grad_tx = t.at<float>(i, j + 1) - t.at<float>(i, j);
    //     }
    // } else if (f.at<uchar>(i, j - 1) != 2) {
    //     grad_tx = t.at<float>(i, j) - t.at<float>(i, j - 1);
    // }

    // if (f.at<uchar>(i + 1, j) != 2) {
    //     if (f.at<uchar>(i - 1, j) != 2) {
    //         grad_ty = 0.5 * (t.at<float>(i + 1, j) - t.at<float>(i - 1, j));
    //     } else {
    //         grad_ty = t.at<float>(i + 1, j) - t.at<float>(i, j);
    //     }
    // } else if (f.at<uchar>(i - 1, j) != 2) {
    //     grad_ty = t.at<float>(i, j) - t.at<float>(i - 1, j);
    // }

    // int min_i = max(1, i - epsilon);
    // int min_j = max(1, j - epsilon);
    // int max_i = min(ret.rows - 1, i + epsilon + 1);
    // int max_j = min(ret.cols - 1, j + epsilon + 1);

    // Vec3d Ia(0, 0, 0);
    // double s = 1.0e-20;

    // for (int k = min_i; k < max_i; k++) {
    //     for (int l = min_j; l < max_j; l++) {
    //         double r_y = i - k, r_x = j - l;
    //         if (f.at<uchar>(k, l) != 2 && (r_x * r_x + r_y * r_y) <= radius_sqr) {
    //             double dst = 1.0 / ((r_x * r_x + r_y * r_y) * sqrt(r_x * r_x + r_y * r_y));
    //             double lev = 1.0 / (1.0 + abs(t.at<float>(k, l) - t.at<float>(i, j)));
    //             double dir_t = grad_tx * r_x + grad_ty * r_y;
    //             dir_t = max(dir_t, 0.000001);
    //             double w = abs(dst * lev * dir_t);
    //             Ia += w * ret.at<Vec3b>(k, l);
    //             s += w;
    //         }
    //     }
    // }
    // ret.at<Vec3b>(i, j) = Vec3b(0.5 + Ia[0] / s, 0.5 + Ia[1] / s, 0.5 + Ia[2] / s);
}

// Teleaアルゴリズム
void telea(Mat &f, Mat &t, Mat &ret, int epsilon, priority_queue<pair<float,pair<int,int>>,vector<pair<float,pair<int,int>>>,compare> &heap) {
    while (!heap.empty()) {
        float val_t = heap.top().first;
        std::pair<int,int> pt = heap.top().second;
        heap.pop();
        int r = pt.second, c = pt.first;

        f.at<uchar>(r, c) = 0; // 修復完了
        for (auto [dr, dc] : vector<pair<int, int>>{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}) {
            int i = r + dr, j = c + dc;
            if (f.at<uchar>(i, j) == 2) {
                double dist = min4(
                    fms(i - 1, j, i, j - 1, f, t),
                    fms(i + 1, j, i, j - 1, f, t),
                    fms(i - 1, j, i, j + 1, f, t),
                    fms(i + 1, j, i, j + 1, f, t)
                );
                t.at<float>(i, j) = dist;
                inpaintpoint(i, j, f, t, ret, epsilon);
                f.at<uchar>(i, j) = 1;
                heap.push({dist, Point(j, i)});
            }
        }
    }
}

// Teleaインペインティング関数
Mat inpaintTelea(const Mat &src, const Mat &mask, int epsilon = 5) {
    CV_Assert(src.size() == mask.size());

    Mat ret = src.clone();
    Mat f = Mat::zeros(mask.size(), CV_8U);
    Mat t = Mat::zeros(mask.size(), CV_32F);
    priority_queue<pair<float, pair<int, int>>, vector<pair<float, pair<int, int>>>, compare> heap;
    

    // fとtを初期化
    for (int i = 1; i < mask.rows - 1; i++) {
        for (int j = 1; j < mask.cols - 1; j++) {
            if (mask.at<uchar>(i, j) > 0) {
                f.at<uchar>(i, j) = 2; // 修復が必要なピクセル
            } else {
                f.at<uchar>(i, j) = 0; // 知られているピクセル
            }
        }
    }

    // 境界ピクセルを検出し、優先度付きキューを初期化
    for (int i = 1; i < mask.rows - 1; i++) {
        for (int j = 1; j < mask.cols - 1; j++) {
            if (f.at<uchar>(i, j) == 2) {
                if (
                    f.at<uchar>(i - 1, j) == 0
                    || f.at<uchar>(i + 1, j) == 0
                    || f.at<uchar>(i, j - 1) == 0
                    || f.at<uchar>(i, j + 1) == 0
                ) {
                    f.at<uchar>(i, j) = 1; // 境界ピクセル
                    double dist = min4(
                        fms(i - 1, j, i, j - 1, f, t),
                        fms(i + 1, j, i, j - 1, f, t),
                        fms(i - 1, j, i, j + 1, f, t),
                        fms(i + 1, j, i, j + 1, f, t)
                    );
                    t.at<float>(i, j) = dist;
                    heap.push({dist, {j, i}});
                }
            }
        }
    }

    // インペインティングを実行
    telea(f, t, ret, epsilon, heap);

    return ret;
}

int main() {
    string srcPath = "left_011528.png";
    string maskPath = "mask.png";

    Mat src = imread(srcPath, IMREAD_COLOR);
    Mat mask = imread(maskPath, IMREAD_GRAYSCALE);

    if (src.empty() || mask.empty()) {
        cerr << "Could not load images." << endl;
        return -1;
    }

    Mat inpainted = inpaintTelea(src, mask, 5);

    imwrite("inpainted_image.png", inpainted);
    return 0;
}
