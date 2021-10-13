#include <math.h>
#include "anomaly_detection_util.h"

float average(float* x, int size){
    float sum = 0;
    for(int i = 0; i < size; i++){
        sum += x[i];
    }
    return sum/size;
}

// returns the variance of X and Y
float var(float* x, int size){
    float sum_pow = 0;
    for(int i =0; i < size; i++){
        sum_pow += pow(x[i], 2);
    }
    float temp = sum_pow/size;
    return temp - std::pow(average(x, size), 2);
}

// returns the covariance of X and Y
float cov(float* x, float* y, int size){
    float Ex = average(x, size);
    float Ey = average(y, size);
    for (int i = 0; i < size ; i++){
        x[i] = x[i] - Ex;
        y[i] = y[i] - Ey;
    }
    float * z;
    z = new float[size];
    for (int i = 0; i < size ; i++){
        z[i] = x[i] * y[i];
    }
    return average(z, size);
}


// returns the Pearson correlation coefficient of X and Y
float pearson(float* x, float*y, int size){
    return cov(x, y, size)/sqrt(var(x, size) * var(y, size));
}

// performs a linear regression and returns the line equation
Line linear_reg(Point** points,int size){
    float *x, *y;
    x = new float[size];
    y = new float[size];
    for (int i = 0; i < size; i++){
        x[i] = points[i]->x;
        y[i] = points[i]->y;
    }
    float a = cov(x,y,size)/var(x,size);
    float b = average(y,size) - (a * average(x,size));
    Line l(a, b);
    return l;
}


// returns the deviation between point p and the line equation of the points
float dev(Point p,Point** points, int size){
    Line l = linear_reg(points, size);
    return dev(p, l);
}

// returns the deviation between point p and the line
float dev(Point p,Line l){
    float s = l.f(p.x) - p.y;
    return std::abs(s);
}
