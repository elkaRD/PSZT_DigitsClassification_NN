//
//  DigitsNN.hpp
//  PSZT_DigitsClassification_NN
//
//  Created by Robert Dudziński on 26/01/2020.
//  Copyright © 2020 Robert. All rights reserved.
//

#ifndef DigitsNN_hpp
#define DigitsNN_hpp

#include <stdio.h>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;

class DigitsNN
{
public:
    
    DigitsNN(vector<int> layerSize)
    {
        hiddenLayerSize = layerSize;
        hiddenLayers = (int) layerSize.size();
        
        w.push_back(vector<vector<double>>());
        
        for (int i = 0; i < hiddenLayerSize[0]; ++i)
        {
            b[0].push_back(randomValue(-1, 1));
            w[0].push_back(vector<double>());
            for (int j = 0; j < inSize; ++j)
            {
                w[0][i].push_back(randomValue(-1, 1));
            }
        }
        
        for (int i = 1; i < hiddenLayers; ++i)
        {
            w.push_back(vector<vector<double>>());
            for (int j = 0; j < hiddenLayerSize[i]; ++j)
            {
                b[i].push_back(randomValue(-1, 1));
                w[i].push_back(vector<double>());
                for (int k = 0; k < hiddenLayerSize[i-1]; ++k)
                {
                    w[i][j].push_back(randomValue(-1, 1));
                }
            }
        }
        
        w.push_back(vector<vector<double>>());
        for (int i = 0; i < outSize; ++i)
        {
            w[hiddenLayers].push_back(vector<double>());
            for (int j = 0; j < hiddenLayerSize[hiddenLayers-1]; ++j)
            {
                w[hiddenLayers][i].push_back(0);
            }
        }
    }
    
    int recognize(vector<double> in)
    {
        vector<double> out = forward(in);
        
        int maxIndex = 0;
        for (int i = 1; i < outSize; ++i)
        {
            if (out[i] > out[maxIndex])
                maxIndex = i;
        }
        
        return maxIndex;
    }
    
private:
    //
    const double E = 2.71828182845905;
    
    vector<vector<double>> a;
    vector<vector<double>> z;
    vector<vector<vector<double>>> w;
    vector<vector<double>> b;
    int hiddenLayers;
    vector<int> hiddenLayerSize;
    int inSize = 28 * 28;
    int outSize = 10;
    
    
    
    vector<double> forward(vector<double> in)
    {
        for (int i = 0; i < hiddenLayerSize[0]; ++i)
        {
            z[0][i] = 0;
            for (int j = 0; j < inSize; ++j)
            {
                z[0][i] += w[0][i][j] * in[j] + b[0][i];
            }
            a[0][i] = sigmoid(z[0][i]);
        }
        
        for (int i = 1; i < hiddenLayers; ++i)
        {
            for (int j = 0; j < hiddenLayerSize[i]; ++j)
            {
                z[i][j] = 0;
                for (int k = 0; k < hiddenLayerSize[i-1]; ++k)
                {
                    z[i][j] += w[i][j][k] * a[i-1][k] + b[i][j];
                }
                a[i][j] = sigmoid(z[i][j]);
            }
        }
        
        vector<double> out(outSize);
        int lastLayer = hiddenLayers-1;
        
        for (int i = 0; i < outSize; ++i)
        {
            out[i] = 0;
            for (int j = 0; j < hiddenLayerSize[lastLayer]; ++j)
            {
                out[i] += w[lastLayer+1][i][j] * a[lastLayer][j];
            }
        }
        
        return out;
    }
    
    inline double sigmoid(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    }
    
    inline double dSigmoid(double x)
    {
        double t = pow(E, -x);
        return t / ((t + 1) * (t + 1));
    }
    
    inline double randomValue()
    {
        return ((double)rand() / (double)RAND_MAX);
    }
    
    inline double randomValue(double minV, double maxV)
    {
        double d = maxV - minV;
        return minV + d * randomValue();
    }
};

#endif /* DigitsNN_hpp */
