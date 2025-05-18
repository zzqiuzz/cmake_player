#include<iostream>
#include<vector>
#include <cmath>
#include<limits>
using std::vector;
using std::cout;
using std::endl;

// 两次遍历，容易溢出精度有损失
vector<float> run_naive_softmax(const vector<float> &in_data){
    vector<float> res;
    res.resize(in_data.size());
    float sum = 0.f;
    for(int i = 0; i < in_data.size(); i++){
        sum += exp(in_data[i]);
    } 
    for(int i = 0; i < in_data.size(); i++){
        res[i] = exp(in_data[i]) / sum;
    }
    return res;
}

//三次遍历，精度无损失
vector<float> run_safe_softmax(const vector<float> &in_data){
    vector<float> res;
    res.resize(in_data.size());
    //step0. get max value;
    float max_value = -std::numeric_limits<float>::max();
    for(int i = 0; i < in_data.size(); i++){ 
        max_value = in_data[i] > max_value ? in_data[i] : max_value; 
    } 
    //step1. get sum
    float sum = 0.f;
    for(int i = 0; i < in_data.size(); i++){
        sum += exp(in_data[i] - max_value);
    } 
    //step2. cal prob
    for(int i = 0; i < in_data.size(); i++){
        res[i] = exp(in_data[i] - max_value) / sum;
    }
    return res;
}

// safe_softmax的求max和求sum能不能只遍历1次
vector<float> run_online_softmax(const vector<float> &in_data){ 
    vector<float> res;
    res.resize(in_data.size());
    float max_value = -std::numeric_limits<float>::infinity();
    float sum_value = 0.f, pre_max_value = 0.f;
    for(int i = 0; i < in_data.size(); i++){
        max_value = std::max(max_value, in_data[i]);
        sum_value = sum_value * exp(pre_max_value - max_value) + exp(in_data[i] - max_value);
        pre_max_value = max_value;
    }
    for(int i = 0; i < in_data.size(); i++)
        res[i] = exp(in_data[i] - max_value) / sum_value;

    return res;
}

float run_online_softmax_default_dot_product(const vector<float> &in_data, const vector<float> &value){ 
    vector<float> res;
    float dot_res = 0.f;
    res.resize(in_data.size());
    float max_value = -std::numeric_limits<float>::infinity();
    float sum_value = 0.f, pre_max_value = 0.f;
    for(int i = 0; i < in_data.size(); i++){
        max_value = std::max(max_value, in_data[i]);
        sum_value = sum_value * exp(pre_max_value - max_value) + exp(in_data[i] - max_value);
        pre_max_value = max_value;
    }
    for(int i = 0; i < in_data.size(); i++)
        dot_res += exp(in_data[i] - max_value) / sum_value * value[i]; // 这里的for循环能否和上面的合并来求出res

    return dot_res;
}

float run_online_softmax_optimal_dot_product(const vector<float> &in_data, const vector<float> &value){ 
    vector<float> res;
    float dot_res = 0.f;
    res.resize(in_data.size());
    float max_value = -std::numeric_limits<float>::infinity();
    float sum_value = 0.f, pre_max_value = 0.f, pre_sum = 0.f;
    float pre_res_value = 0.f;
    for(int i = 0; i < in_data.size(); i++){
        max_value = std::max(max_value, in_data[i]);
        sum_value = sum_value * exp(pre_max_value - max_value) + exp(in_data[i] - max_value);
        dot_res = dot_res * pre_sum * exp(pre_max_value - max_value) / sum_value + exp(in_data[i] - max_value) / sum_value * value[i]; 
        pre_max_value = max_value;
        pre_sum = sum_value;
    }
 
    return dot_res;
}

void print_vector(vector<float> res){
    for(auto _data: res)
        cout << _data << " ";
    cout << endl;
}
int main(){
    vector<float> data = {2.3, 5.6, 8.5, 1.2, 0.1};
    vector<float> value{1.1, 2.2, 3.3, 4.4, 5.5};
    print_vector(data);
 
    vector<float> res0 = run_naive_softmax(data); 
    vector<float> res1 = run_safe_softmax(data); 
    vector<float> res2 = run_online_softmax(data); 
    float res3 = run_online_softmax_default_dot_product(data, value); 
    float res4 = run_online_softmax_optimal_dot_product(data, value); 
    
    print_vector(res0);
    print_vector(res1);
    print_vector(res2);
    cout << "default dot product: " << res3 << endl;
    cout << "optimal dot product: " << res4 << endl; 

    return 0;
}