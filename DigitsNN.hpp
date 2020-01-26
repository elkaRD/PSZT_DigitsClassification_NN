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
#include <algorithm>
#include <random>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;

class DigitsNN
{
public:
    
    DigitsNN(std::vector<int> layerSize)
    {
        randomGenerator = std::default_random_engine {};
        
        hiddenLayerSize = layerSize;
        hiddenLayers = (int) layerSize.size();
        
        w.push_back(std::vector<std::vector<double>>());
        b.push_back(std::vector<double>());
        
        for (int i = 0; i < hiddenLayerSize[0]; ++i)
        {
            //b[0].push_back(randomValue(-1, 1));
            b[0].push_back(0);
            w[0].push_back(std::vector<double>());
            for (int j = 0; j < inSize; ++j)
            {
                w[0][i].push_back(randomValue(-1, 1));
            }
        }
        
        for (int i = 1; i < hiddenLayers; ++i)
        {
            w.push_back(std::vector<std::vector<double>>());
            b.push_back(std::vector<double>());
            for (int j = 0; j < hiddenLayerSize[i]; ++j)
            {
                //b[i].push_back(randomValue(-1, 1));
                b[i].push_back(0);
                w[i].push_back(std::vector<double>());
                for (int k = 0; k < hiddenLayerSize[i-1]; ++k)
                {
                    w[i][j].push_back(randomValue(-1, 1));
                }
            }
        }
        
        w.push_back(std::vector<std::vector<double>>());
        for (int i = 0; i < outSize; ++i)
        {
            w[hiddenLayers].push_back(std::vector<double>());
            for (int j = 0; j < hiddenLayerSize[hiddenLayers-1]; ++j)
            {
                w[hiddenLayers][i].push_back(0);
            }
        }
        
        for (int i = 0; i < hiddenLayers; ++i)
        {
            a.push_back(std::vector<double>());
            for (int j = 0; j < hiddenLayerSize[i]; ++j)
            {
                a[i].push_back(0);
            }
        }
        
        for (int i = 0; i < outSize; ++i)
        {
            dout.push_back(0);
        }
        
        dw = w;
        db = b;
        da = a;
    }
    
    int recognize(std::vector<double> in)
    {
        std::vector<double> out = forward(in);
        
        int maxIndex = 0;
        for (int i = 1; i < outSize; ++i)
        {
            if (out[i] > out[maxIndex])
                maxIndex = i;
        }
        
        return maxIndex;
    }
    
    void addTraining(std::vector<uint8_t> data, int digit)
    {
        std::vector<double> t;
        
        for (int i = 0; i < inSize; ++i)
            t.push_back((double)data[i] / 255.0);
        
        trainingData.push_back(t);
        trainingDigit.push_back(digit);
    }
    
    void addTest(std::vector<uint8_t> data, int digit)
    {
        std::vector<double> t;
        
        for (int i = 0; i < inSize; ++i)
            t.push_back((double)data[i] / 255.0);
        
        testData.push_back(t);
        testDigit.push_back(digit);
    }
    
    void learn()
    {
        std::vector<int> batches;
        for (int i = 0; i < trainingData.size(); ++i)
        {
            batches.push_back(i);
        }
        
        std::shuffle(std::begin(batches), std::end(batches), randomGenerator);
        
        for (int i = 0; i < trainingData.size() / BATCH_SIZE - 1; ++i)
        {
            std::vector<int> batch;
            std::copy(batches.begin() + (i * BATCH_SIZE), batches.begin() + ((i+1) * BATCH_SIZE), batch.begin());
            backward(batch);
        }
    }
    
private:
    //
    const double E = 2.71828182845905;
    const int BATCH_SIZE = 100;
    
    std::vector<std::vector<double>> a;
    std::vector<std::vector<double>> z;
    std::vector<std::vector<std::vector<double>>> w;
    std::vector<std::vector<double>> b;
    int hiddenLayers;
    std::vector<int> hiddenLayerSize;
    int inSize = 28 * 28;
    int outSize = 10;
    
    std::vector<double> dout;
    std::vector<std::vector<double>> da;
    std::vector<std::vector<double>> db;
    std::vector<std::vector<std::vector<double>>> dw;
    
    std::vector<std::vector<double>> gda;
    std::vector<std::vector<double>> gdb;
    std::vector<std::vector<std::vector<double>>> gdw;
    
    std::vector<std::vector<double>> trainingData;
    std::vector<std::vector<double>> testData;
    std::vector<int> trainingDigit;
    std::vector<int> testDigit;
    
    std::default_random_engine randomGenerator;
    
    
    std::vector<double> forward(std::vector<double> in)
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
        
        std::vector<double> out(outSize);
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
    
    void backward(std::vector<int> batch)
    {
        
        
        std::vector<double> in;
        int y = 0;
        std::vector<double> out = forward(in);
        
        for (int i = 0; i < outSize; ++i)
        {
            double expected = y == i ? 1 : 0;
            dout[i] = 2 * (out[i] - expected);
        }
        
        for (int i = 0; i < hiddenLayerSize[hiddenLayers-1]; ++i)
        {
            da[hiddenLayers-1][i] = 0;
            for (int j = 0; j < outSize; ++j)
            {
                dw[hiddenLayers][j][i] = dout[j] * a[hiddenLayers-1][i];
                da[hiddenLayers-1][i] += dout[j] * w[hiddenLayers][j][i];
            }
        }
        
        for (int i = 0; i < hiddenLayers-1; ++i)
        {
            for (int j = 0; j < hiddenLayerSize[i]; ++j)
            {
                da[i][j] = 0;
                for (int k = 0; k < hiddenLayerSize[i+1]; ++k)
                {
                    da[i][j] += da[i+1][k] * dSigmoid(z[i+1][k]) * w[i+1][k][j];
                    dw[i+1][k][j] = da[i+1][k] * dSigmoid(z[i+1][k]) * a[i][j];
                }
            }
        }
        
        for (int i = 0; i < inSize; ++i)
        {
            for (int j = 0; j < hiddenLayerSize[0]; ++j)
            {
                dw[0][j][i] = da[0][j] * dSigmoid(z[0][j]) * in[i];
            }
        }
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
