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

int main(){
    vector<float> data = {0.23, 0.56, 0.85, 0.12, 0.01};
    for(auto _data: data)
        cout << _data << " ";
    cout << endl;

    // run_naive_softmax(data);
    vector<float> res0 = run_naive_softmax(data);
    for(auto _data: res0)
        cout << _data << " ";
    cout << endl;
    vector<float> res1 = run_safe_softmax(data);
    for(auto _data: res1)
        cout << _data << " ";
    cout << endl;
    vector<float> res2 = run_online_softmax(data);
    
    for(auto _data: res2)
        cout << _data << " ";
    cout << endl;
    return 0;
}