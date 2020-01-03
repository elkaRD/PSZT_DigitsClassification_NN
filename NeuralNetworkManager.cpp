//
//  NeuralNetworkManager.cpp
//  PSZT_DigitsClassification_NN
//
//  Created by Robert Dudziński on 29/12/2019.
//  Copyright © 2019 Robert. All rights reserved.
//

#include "NeuralNetworkManager.hpp"

#include <iostream>
using namespace std;

NeuralNetworkManager::NeuralNetworkManager(std::vector<int> hiddenLayers)
{
//    std::vector<matrix<double>> a;
//    std::vector<matrix<double>> w;
//    std::vector<matrix<double>> b;
//    std::vector<matrix<double>> z;
//    std::vector<matrix<double>> dca;
    
    cout << "CONSTRUCTOR" << endl;
    
    srand((unsigned)time(NULL));
    
    this->hiddenLayers = hiddenLayers;
    hiddenLayers.push_back(10);
    hiddenLayersSize = hiddenLayers.size();
    
    a.clear();
    w.clear();
    b.clear();
    z.clear();
    dca.clear();
    dcb.clear();
    dcw.clear();
    
    a.reserve(hiddenLayers.size());
    w.reserve(hiddenLayers.size());
    b.reserve(hiddenLayers.size());
    z.reserve(hiddenLayers.size());
    dca.reserve(hiddenLayers.size());
    dcb.reserve(hiddenLayers.size());
    dcw.reserve(hiddenLayers.size());
    
    for (const auto &neurons : hiddenLayers)
    {
        //TODO: create only one vector, because when pushing_back vector will create a copy
        matrix<double> tempA(1, neurons);
        matrix<double> tempB(1, neurons);
        matrix<double> tempZ(1, neurons);
        matrix<double> tempDca(1, neurons);
        matrix<double> tempDcb(1, neurons);
        
//        for (int i = 0; i < neurons; ++i)
//            tempB(0, i) = -randomValue() * 50.0 - 50.0;
        for (int i = 0; i < neurons; ++i)
            tempB(0, i) = randomValue() * 20.0 - 10.0;
        
        a.push_back(tempA);
        b.push_back(tempB);
        z.push_back(tempZ);
        dca.push_back(tempDca);
        dcb.push_back(tempDcb);
        
//        cout << "a:   " << tempA << endl;
//        cout << "b:   " << tempB << endl;
//        cout << "z:   " << tempZ << endl;
//        cout << "dca: " << tempDca << endl;
    }
    
    for (size_t i = 0; i < hiddenLayers.size(); ++i)
    {
        int prevLayer = i == 0 ? 28*28 : hiddenLayers[i-1];
        int curLayer = hiddenLayers[i];
        //matrix<double> tempW(curLayer, prevLayer);
        matrix<double> tempW(prevLayer, curLayer);
        matrix<double> tempDcw(prevLayer, curLayer);
        
        for (int x = 0; x < curLayer; ++x)
            for (int y = 0; y < prevLayer; ++y)
                tempW(y ,x) = randomValue() * 2.0 - 1.0;
        
        w.push_back(tempW);
        dcw.push_back(tempDcw);
        
        //cout << "W: " << tempW << endl;
    }
    
    cout << "end of the contructor" << endl;
}

double NeuralNetworkManager::randomValue()
{
    return ((double)rand() / (double)RAND_MAX);
}

int NeuralNetworkManager::detectDigit(std::vector<double> image)
{
    if (image.size() != 28*28) throw std::exception();
    
    matrix<double> debug = forward(image);
    
    cout << "result of forward: " << debug << endl;
    
    learn(image, 0);
    
    return 0;
}

int NeuralNetworkManager::detectDigitInt8(std::vector<uint8_t> image)
{
    if (image.size() != 28*28) throw std::exception();
    
    std::vector<double> temp;
    temp.reserve(28 * 28);
    
    for (int i = 0; i < 28*28; ++i)
    {
        double d = image[i];
        d /= 255.0;
        temp.push_back(d);
    }
    
    return detectDigit(temp);
}

void NeuralNetworkManager::learn(std::vector<double> image, int expected)
{
    matrix<double> output = forward(image);
    
    matrix<double> in(1, 28*28);
    
    for (int i = 0; i < 28*28; ++i)
        in(0, i) = image[i];
    
    for (int i = 0; i < 10; ++i)
    {
        double desired = i == expected ? 1 : 0;
        dca[hiddenLayersSize-1] (0, i) = 2 * (output(0, i) - desired);
    }
    cout << "learn: " << dca[hiddenLayersSize-1] << endl;
    
    for (int i = hiddenLayersSize-1; i >= 0; --i)
    {
        //dcb[i] = dSigmoid(z[i]) * dca[i];
        dcb[i] = prod(dca[i], toDiagonal(dSigmoid(z[i])));

        matrix<double> tempDcw = prod(dca[i], toDiagonal(dSigmoid(z[i])));
        
        matrix<double> prevLayer = i == 0 ? in : a[i-1];
        
//        cout << "debug2_1: " << toDiagonal(prevLayer).size1() << " x " << toDiagonal(prevLayer).size2() << endl;
//        cout << "debug2_2: " << sameRows(tempDcw, prevLayer.size2()) << " x " << sameRows(tempDcw, prevLayer.size2()).size2() << endl;
        
        if (i != 0)cout << "toDiagonal(prevLayer): " << toDiagonal(prevLayer) << endl;
            //cout << "toDiagonal(prevLayer): " << toDiagonal(prevLayer).size1() << " x " << toDiagonal(prevLayer).size2() << endl;
        
        dcw[i] = prod(toDiagonal(prevLayer), sameRows(tempDcw, prevLayer.size2()));
        
        if (i == 0) break;
        
        matrix<double> tempDca = prod(dca[i], toDiagonal(dSigmoid(z[i])));
        
        cout << "debug2_1: " << w[i].size1() << " x " << w[i].size2() << endl;
        cout << "debug2_2: " << tempDca.size1() << " x " << tempDca.size2() << endl;
        
        dca[i-1] = prod(tempDca, trans(w[i]));
        //matrix<double> f = prod(w[i], tempDca);
//        matrix<double> f = prod(tempDca, w[i]);
        //cout << "debug2_3: " << w[i] << endl; //prod(tempDca, w[i]) << endl;
    }
    
    for (int i = 0; i < hiddenLayersSize; ++i)
    {
        int prevLayer = i == 0 ? 28*28 : hiddenLayers[i-1];
        int curLayer = hiddenLayers[i];

//        for (int x = 0; x < curLayer; ++x)
//            for (int y = 0; y < prevLayer; ++y)
//                w[i](y ,x) += dcw[i](y, x);
//
//        for (int x = 0; x < curLayer; ++x)
//            b[i] += dcb[i];
        
        w[i] += dcw[i] * 0.001;
        b[i] += dcb[i] * 0.001;
        
        cout << "DCW: " << dcw[i] << endl;
    }
}

matrix<double> NeuralNetworkManager::forward(std::vector<double> input)
{
    matrix<double> in(1, 28*28);
    
    for (int i = 0; i < 28*28; ++i)
        in(0, i) = input[i];
    
    cout << "debug: " << w[0].size1() << " x " << w[0].size2() << endl;
    cout << "debug: " << in.size1() << "x " << in.size2() << endl;
    
    z[0] = prod(in, w[0]);
    z[0] += b[0];
    a[0] = sigmoid(z[0]);
    
    for (int i = 1; i < hiddenLayersSize; ++i)
    {
        z[i] = prod(a[i-1], w[i]);
        z[i] += b[i];
        a[i] = sigmoid(z[i]);
    }
    
    return a[hiddenLayersSize - 1];
}

double NeuralNetworkManager::sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double NeuralNetworkManager::dSigmoid(double x)
{
    return x * (1.0 - x);
}

matrix<double> NeuralNetworkManager::sigmoid(matrix<double> &m)
{
    if (m.size1() != 1) throw std::exception();
    
    matrix<double> result (1, m.size2());
    
    for (int i = 0; i < m.size2(); ++i)
        result(0, i) = sigmoid(m(0, i));
    
    return result;
}

matrix<double> NeuralNetworkManager::dSigmoid(matrix<double> &m)
{
    if (m.size1() != 1) throw std::exception();
    
    matrix<double> result (1, m.size2());
    
    for (int i = 0; i < m.size2(); ++i)
        result(0, i) = dSigmoid(m(0, i));
    
    return result;
}

matrix<double> NeuralNetworkManager::toDiagonal(const matrix<double> &m)
{
    if (m.size1() != 1) throw std::exception();
    
    matrix<double> temp(m.size2(), m.size2());
    temp *= 0.0;
    for (int i = 0; i < m.size2(); ++i)
        temp(i, i) = m(0, i);
    
    return temp;
}

matrix<double> NeuralNetworkManager::sameRows(const matrix<double> &m, const int rows)
{
    if (m.size1() != 1) throw std::exception();
    
    matrix<double> temp(rows, m.size2());
    for (int i = 0; i < m.size2(); ++i)
    {
        for (int j = 0; j < rows; ++j)
        {
            temp(j, i) = m(0, i);
        }
    }
    
    return temp;
}
