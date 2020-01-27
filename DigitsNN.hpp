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
#include <string>
#include <fstream>
#include <iostream>

using namespace std;

class DigitsNN
{
public:
    
    DigitsNN(const std::vector<int> &layerSize)
    {
        randomGenerator = std::default_random_engine {};
        
        init(layerSize);
    }
    
    int recognize(const std::vector<double> &in, bool showPercentage = false)
    {
        std::vector<double> out = forward(in);
        
        if (showPercentage)
            for (int i = 0; i < 10; ++i)
                cout << " " << i << ": " << out[i] << endl;
        
        int maxIndex = 0;
        for (int i = 1; i < outSize; ++i)
        {
            if (out[i] > out[maxIndex])
                maxIndex = i;
        }
        
        return maxIndex;
    }
    
    void addTraining(const std::vector<uint8_t> &data, int digit)
    {
        std::vector<double> t;
        
        for (int i = 0; i < inSize; ++i)
            t.push_back((double)data[i] / 255.0);
        
        trainingData.push_back(t);
        trainingDigit.push_back(digit);
    }
    
    void addTest(const std::vector<uint8_t> &data, int digit)
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
        
        for (int i = 0; i < trainingData.size() / BATCH_SIZE; ++i)
        {
            std::vector<int> batch;

            for (int j = i * BATCH_SIZE; j < (i+1) * BATCH_SIZE; ++j)
                batch.push_back(batches[j]);
            
            if ((i * BATCH_SIZE) % 1000 == 0)
                cout << "learning from " << (i * BATCH_SIZE) << " - " <<  (i+1) * BATCH_SIZE << " ";
            
            backward(batch);
        }
        cout << endl;
    }
    
    void test()
    {
        int correct = 0;
        
        for (int i = 0; i < testData.size(); ++i)
        {
            if (recognize(testData[i]) == testDigit[i])
            {
                correct++;
            }
        }
        
        cout << "TEST RESULT: " << ((double) correct / testData.size()) << endl;
    }
    
    void save(string filename)
    {
        ofstream file(filename.c_str());
        
        file << BATCH_SIZE << endl;
        file << STEP_DST << endl;
        file << hiddenLayers << endl;
        file << inSize << endl;
        file << outSize << endl;
        for (int i = 0; i < hiddenLayers; ++i)
            file << hiddenLayerSize[i] << " ";
        file << endl;
        
        for (int i = 0; i < inSize; ++i)
        {
            for (int j = 0; j < hiddenLayerSize[0]; ++j)
            {
                file << w[0][j][i] << " ";
            }
        }
        file << endl;
        for (int i = 0; i < hiddenLayers-1; ++i)
        {
            for (int j = 0; j < hiddenLayerSize[i]; ++j)
            {
                for (int k = 0; k < hiddenLayerSize[i+1]; ++k)
                {
                    file << w[i+1][k][j] << " ";
                }
            }
            file << endl;
        }
        for (int i = 0; i < hiddenLayerSize[hiddenLayers-1]; ++i)
        {
            for (int j = 0; j < outSize; ++j)
            {
                file << w[hiddenLayers][j][i] << " ";
            }
        }
        file << endl;
        
        for (int i = 0; i < hiddenLayers; ++i)
        {
            for (int j = 0; j < hiddenLayerSize[i]; ++j)
            {
                file << b[i][j] << " ";
            }
            file << endl;
        }
        
        file.close();
    }
    
    void load(string filename)
    {
        ifstream file(filename.c_str());
        
        file >> BATCH_SIZE;
        file >> STEP_DST;
        file >> hiddenLayers;
        file >> inSize;
        file >> outSize;
        
        std::vector<int> layers;
        for (int i = 0; i < hiddenLayers; ++i)
        {
            int t;
            file >> t;
            layers.push_back(t);
        }
        
        init(layers, inSize, outSize);
        
        for (int i = 0; i < inSize; ++i)
        {
            for (int j = 0; j < hiddenLayerSize[0]; ++j)
            {
                file >> w[0][j][i];
            }
        }
        
        for (int i = 0; i < hiddenLayers-1; ++i)
        {
            for (int j = 0; j < hiddenLayerSize[i]; ++j)
            {
                for (int k = 0; k < hiddenLayerSize[i+1]; ++k)
                {
                    file >> w[i+1][k][j];
                }
            }
        }
        for (int i = 0; i < hiddenLayerSize[hiddenLayers-1]; ++i)
        {
            for (int j = 0; j < outSize; ++j)
            {
                file >> w[hiddenLayers][j][i];
            }
        }
        
        for (int i = 0; i < hiddenLayers; ++i)
        {
            for (int j = 0; j < hiddenLayerSize[i]; ++j)
            {
                file >> b[i][j];
            }
        }
        
        file.close();
        
        //STEP_DST *= 0.6;
    }
    
    int recognize(string filename, bool showPercentage = false)
    {
        FILE *file = fopen(filename.c_str(), "rb");
        
        char data[28*28];
        fread(data, sizeof(char), 28*28, file);
        
        fclose(file);
        
        std::vector<double> image;
        
        for (int i = 0; i < 28*28; ++i)
        {
            image.push_back((double) data[i] / 255.0);
        }
        
        return recognize(image, showPercentage);
    }
    
private:
    //
    const double E = 2.71828182845905;
    int BATCH_SIZE = 100;
    double STEP_DST = 0.03;
    
    std::vector<std::vector<double>> a;
    std::vector<std::vector<double>> z;
    std::vector<std::vector<std::vector<double>>> w;
    std::vector<std::vector<double>> b;
    int hiddenLayers;
    std::vector<int> hiddenLayerSize;
    int inSize;
    int outSize;
    
    std::vector<double> dout;
    std::vector<double> zout;
    std::vector<double> bout;
    
    std::vector<std::vector<double>> da;
    std::vector<std::vector<double>> db;
    std::vector<std::vector<std::vector<double>>> dw;
    
    std::vector<std::vector<double>> gdb;
    std::vector<std::vector<std::vector<double>>> gdw;
    
    std::vector<std::vector<double>> trainingData;
    std::vector<std::vector<double>> testData;
    std::vector<int> trainingDigit;
    std::vector<int> testDigit;
    
    std::default_random_engine randomGenerator;
    
    void init(const std::vector<int> &layerSize, const int inLayer = 28*28, const int outLayer = 10)
    {
        a.clear();
        z.clear();
        w.clear();
        b.clear();
        
        hiddenLayerSize.clear();
        
        dout.clear();
        zout.clear();
        bout.clear();
        
        da.clear();
        db.clear();
        dw.clear();
        
        gdb.clear();
        gdw.clear();
        
        inSize = inLayer;
        outSize = outLayer;
        
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
        
        zout = dout;
        bout = dout;
        
        dw = w;
        db = b;
        da = a;
        
        z = a;
        
        gdw = w;
        gdb = b;

    }
    
    
    std::vector<double> forward(const std::vector<double> &in)
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
            zout[i] = bout[i];
            for (int j = 0; j < hiddenLayerSize[lastLayer]; ++j)
            {
                zout[i] += w[lastLayer+1][i][j] * a[lastLayer][j];
            }
            out[i] = sigmoid(zout[i]);
        }
        
        return out;
    }
    
    void backward(std::vector<int> &batch)
    {
        for (auto &it : gdw)
            for (auto &it2 : it)
                for (auto &it3 : it2)
                    it3 = 0;
        
        for (auto &it : gdb)
            for (auto &it2 : it)
                it2 = 0;
        
        for (int i = 0; i < batch.size(); ++i)
        {
            std::vector<double> in = trainingData[batch[i]];
            int y = trainingDigit[batch[i]];
            std::vector<double> out = forward(in);
            
            for (int i = 0; i < outSize; ++i)
            {
                double expected = y == i ? 10 : 0;
                dout[i] = 2 * (out[i] - expected) * STEP_DST;
            }
            
            for (int i = 0; i < hiddenLayerSize[hiddenLayers-1]; ++i)
            {
                da[hiddenLayers-1][i] = 0;
                for (int j = 0; j < outSize; ++j)
                {
                    dw[hiddenLayers][j][i] = dout[j] * a[hiddenLayers-1][i];
                    //da[hiddenLayers-1][i] += dout[j] * w[hiddenLayers][j][i];
                    da[hiddenLayers-1][i] += dout[j] * dSigmoid(zout[j]) * w[hiddenLayers][j][i];
                }
                db[hiddenLayers-1][i] = da[hiddenLayers-1][i] * dSigmoid(z[hiddenLayers-1][i]);
            }
            
            //START
//            for (int i = 0; i < hiddenLayers-1; ++i)
//            {
//                for (int j = 0; j < hiddenLayerSize[i]; ++j)
//                {
//                    da[i][j] = 0;
//                    for (int k = 0; k < hiddenLayerSize[i+1]; ++k)
//                    {
//                        da[i][j] += da[i+1][k] * dSigmoid(z[i+1][k]) * w[i+1][k][j];
//                        dw[i+1][k][j] = da[i+1][k] * dSigmoid(z[i+1][k]) * a[i][j];
//                    }
//                }
//            }
            for (int i = hiddenLayers-2; i >= 0; --i)
            {
                for (int j = 0; j < hiddenLayerSize[i]; ++j)
                {
                    da[i][j] = 0;
                    for (int k = 0; k < hiddenLayerSize[i+1]; ++k)
                    {
                        da[i][j] += da[i+1][k] * dSigmoid(z[i+1][k]) * w[i+1][k][j];
                        dw[i+1][k][j] = da[i+1][k] * dSigmoid(z[i+1][k]) * a[i][j];
                    }
                    //db[hiddenLayers-1][i] = da[hiddenLayers-1][i] * dSigmoid(z[hiddenLayers-1][i]);
                    db[i][j] = da[i][j] * dSigmoid(z[i][j]);
                }
            }
            //STOP
            
            for (int i = 0; i < inSize; ++i)
            {
                for (int j = 0; j < hiddenLayerSize[0]; ++j)
                {
                    dw[0][j][i] = da[0][j] * dSigmoid(z[0][j]) * in[i];
                }
            }
            
            for (int i = 0; i < dw.size(); ++i)
            {
                for (int j = 0; j < dw[i].size(); ++j)
                {
                    for (int k = 0; k < dw[i][j].size(); ++k)
                    {
                        gdw[i][j][k] += dw[i][j][k];
                    }
                }
            }
            
            for (int i = 0; i < db.size(); ++i)
            {
                for (int j = 0; j < db[i].size(); ++j)
                {
                    gdb[i][j] += db[i][j];
                }
            }
        }
        
        for (int i = 0; i < gdw.size(); ++i)
        {
            for (int j = 0; j < gdw[i].size(); ++j)
            {
                for (int k = 0; k < gdw[i][j].size(); ++k)
                {
                    gdw[i][j][k] /= BATCH_SIZE;
                    w[i][j][k] -= gdw[i][j][k];
                }
            }
        }
        
        for (int i = 0; i < gdb.size(); ++i)
        {
            for (int j = 0; j < gdb[i].size(); ++j)
            {
                gdb[i][j] /= BATCH_SIZE;
                b[i][j] -= gdb[i][j];
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
