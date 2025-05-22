    #pragma once
    void reduce_baseline(float*, float*, int, int);

    bool check(float *out, float *res, int n){
        for(int i = 0; i < n; i++){
            if(out[i] - res[i] > 0.005)
                return false;
        }
        return true;
}