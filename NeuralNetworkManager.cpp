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

const int NeuralNetworkManager::INPUT_NEURONS = 2;
const int NeuralNetworkManager::OUTPUT_NEURONS = 2;

NeuralNetworkManager::NeuralNetworkManager(std::vector<int> hiddenLayers)
{
//    std::vector<matrix<double>> a;
//    std::vector<matrix<double>> w;
//    std::vector<matrix<double>> b;
//    std::vector<matrix<double>> z;
//    std::vector<matrix<double>> dca;
    
    cout << "CONSTRUCTOR" << endl;
    
    //srand((unsigned)time(NULL));
    srand(0.111);
    
    hiddenLayers.push_back(OUTPUT_NEURONS);
    hiddenLayersSize = hiddenLayers.size();
    this->hiddenLayers = hiddenLayers;
    
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
        int prevLayer = i == 0 ? INPUT_NEURONS : hiddenLayers[i-1];
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

static int s;

int NeuralNetworkManager::detectDigit(std::vector<double> image)
{
    if (image.size() != INPUT_NEURONS) throw std::exception();
    
    //debugDisplayParams();
    
    s++;
    
    matrix<double> debug = forward(image);
    
    
    
    
    
    
    cout << s << " result of forward: " << debug << endl;
    
    learn(image, 0);
    
    
    if (s >= 140)
    {
        //cout << "HERE" << endl;
        //debugDisplayCalculated();
    }
    
    //debugDisplayParams();
    //debugDisplayCalculated();
    
    return 0;
}

int NeuralNetworkManager::detectDigitInt8(std::vector<uint8_t> image)
{
    if (image.size() != INPUT_NEURONS) throw std::exception();
    
    std::vector<double> temp;
    temp.reserve(INPUT_NEURONS);
    
    for (int i = 0; i < INPUT_NEURONS; ++i)
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
    
    matrix<double> in(1, INPUT_NEURONS);
    
    for (int i = 0; i < INPUT_NEURONS; ++i)
        in(0, i) = image[i];
    
    for (int i = 0; i < OUTPUT_NEURONS; ++i)
    {
        double desired = i == expected ? 1 : 0;
        //cout << "desired: " << desired << endl;
        dca[hiddenLayersSize-1] (0, i) = 2 * (output(0, i) - desired);
    }
    
    //if (s >= 140) cout << s << "forward: " << output << ",    DCA: " << dca[hiddenLayersSize-1] << endl;
    
    //cout << "learn DCA: " << dca[hiddenLayersSize-1] << endl;
    
    for (int i = hiddenLayersSize-1; i >= 0; --i)
    {
        for (int j = 0; j < hiddenLayers[i]; ++j)
        {
            double temp = dcb[i](0, j);
            dcb[i](0, j) = dSigmoid(z[i](0, j)) * dca[i](0, j);
            if (isnan(dcb[i](0, j)) && !isnan(temp))
            {
                cout << "DCB PROBLEM'S HERE" << endl;
                cout << "s: " << s << ",  i,j: " << i << ", " << j << endl;
                cout << "dsigmoid: " << dSigmoid(z[i](0, j)) << ",    dca: " << dca[i](0, j) << endl;
            }
        }
        
        for (int j = 0; j < hiddenLayers[i]; ++j)
        {
            matrix<double> leftNeurons = in;
            if (i != 0) leftNeurons = a[i-1];
            for (int k = 0; k < leftNeurons.size2(); k++)
            {
                //cout << dcw[i](j, k) << endl;
                //cout << leftNeurons(0, k) << endl;
                //cout << dSigmoid(z[i](0, j)) << endl;
                //cout << dca[i](0, j) << endl;
                double temp = dcw[i](k, j);
                dcw[i](k, j) = leftNeurons(0, k) * dSigmoid(z[i](0, j)) * dca[i](0, j);
                if (isnan(dcw[i](k, j)) && !isnan(temp))
                {
                    cout << "W PROBLEM" << endl;
                    cout << "s: " << s << ",  i,j: " << i << ", " << j << endl;
                }
            }
        }
        
        if (i == 0) break;
        
        
        for (int j = 0; j < hiddenLayers[i-1]; ++j)
        {
            dca[i-1](0, j) = 0;
            for (int k = 0; k < hiddenLayers[i]; ++k)
            {
                //cout << "testnig: " << w[i](j, k) << ", " << dSigmoid(z[i](0, k)) << ", " << dca[i](0, k) << ",        " << z[i](0, k) << endl;
                dca[i-1](0, j) += w[i](j, k) * dSigmoid(z[i](0, k)) * dca[i](0, k);
            }
        }
    }
    
    for (int i = 0; i < hiddenLayersSize; ++i)
    {
        //if (s == 141) cout << "BACK BEFOR" << i << ": " << w[i] << endl;
        
        for (int x = 0; x < dcw[i].size1(); ++x)
        {
            for (int y = 0; y < dcw[i].size2(); ++y)
            {
                if (dcw[i](x, y) > 1000000)  dcw[i](x, y) =  1000000;
                if (dcw[i](x, y) < -1000000) dcw[i](x, y) = -1000000;
            }
        }
        
        for (int y = 0; y < dcb[i].size2(); ++y)
        {
            if (dcb[i](0, y) > 1000000)  dcb[i](0, y) =  1000000;
            if (dcb[i](0, y) < -1000000) dcb[i](0, y) = -1000000;
        }
        
        w[i] += dcw[i] * 0.01;
        b[i] += dcb[i] * 0.01;
        
        //if (s == 141) cout << "BACK AFTER" << i << ": " << w[i] << endl;
    }
    
    return;
    
    for (int i = hiddenLayersSize-1; i >= 0; --i)
    {
        cout << "DCA: " << dca[i] << endl;
        
        //dcb[i] = dSigmoid(z[i]) * dca[i];
        dcb[i] = prod(dca[i], toDiagonal(dSigmoid(z[i])));

        matrix<double> tempDcw = prod(dca[i], toDiagonal(dSigmoid(z[i])));
        
        matrix<double> prevLayer = i == 0 ? in : a[i-1];
        
//        cout << "debug2_1: " << toDiagonal(prevLayer).size1() << " x " << toDiagonal(prevLayer).size2() << endl;
//        cout << "debug2_2: " << sameRows(tempDcw, prevLayer.size2()) << " x " << sameRows(tempDcw, prevLayer.size2()).size2() << endl;
        
        if (i != 0)cout << "toDiagonal(prevLayer): " << toDiagonal(prevLayer) << endl;
            //cout << "toDiagonal(prevLayer): " << toDiagonal(prevLayer).size1() << " x " << toDiagonal(prevLayer).size2() << endl;
        
        //dcw[i] = prod(toDiagonal(prevLayer), sameRows(tempDcw, prevLayer.size2()));
        
        if (i == 0) break;
        
        
        
//        cout << "debug2_1: " << w[i].size1() << " x " << w[i].size2() << endl;
//        cout << "debug2_2: " << tempDca.size1() << " x " << tempDca.size2() << endl;
        
        //matrix<double> tempDca = prod(dca[i], toDiagonal(dSigmoid(z[i])));  //MATRIX VERSION
        //dca[i-1] = prod(tempDca, trans(w[i]));    //MATRIX VERSION
        
//        for (int j = 0; j < hiddenLayers[i-1]; ++j)
//        {
//            dca[i-1](0, j) = 0;
//            for (int k = 0; k < hiddenLayers[i]; ++k)
//            {
//                cout << "jk: " << j << ",  " << k << endl;
//                cout << "w: " << w[i](j, k) << endl;
//                cout << "dsig: " << dSigmoid(z[i](0, k)) << endl;
//                cout << "dca: " << dca[i](0, k) << endl;
//                dca[i-1](0, j) += w[i](j, k) * dSigmoid(z[i](0, k)) * dca[i](0, k);
//            }
//            //dca[i-1](0, j) = w[i](j, )
//        }
        
        //matrix<double> f = prod(w[i], tempDca);
//        matrix<double> f = prod(tempDca, w[i]);
        //cout << "debug2_3: " << w[i] << endl; //prod(tempDca, w[i]) << endl;
    }
    
    for (int i = 0; i < hiddenLayersSize; ++i)
    {
        int prevLayer = i == 0 ? INPUT_NEURONS : hiddenLayers[i-1];
        int curLayer = hiddenLayers[i];

//        for (int x = 0; x < curLayer; ++x)
//            for (int y = 0; y < prevLayer; ++y)
//                w[i](y ,x) += dcw[i](y, x);
//
//        for (int x = 0; x < curLayer; ++x)
//            b[i] += dcb[i];
        
        
        
        w[i] += dcw[i] * 0.001;
        b[i] += dcb[i] * 0.001;
        
        //cout << "DCW: " << dcw[i] << endl;
    }
}

matrix<double> NeuralNetworkManager::forward(std::vector<double> input)
{
    matrix<double> in(1, INPUT_NEURONS);
    
    for (int i = 0; i < INPUT_NEURONS; ++i)
        in(0, i) = input[i];
    
    if (s == 141)
    {
        cout << s << " W: " << w[0] << endl;
        cout << s << " B: " << b[0] << endl;
    }
    
    z[0] = prod(in, w[0]);
    z[0] += b[0];
    
    for (int d = 0; d < z[0].size2(); ++d)
    {
        if (isnan(z[0](0, d)))
        {
            cout << s << " EARLIER PROBLEM" << endl;
        }
    }
    
    a[0] = sigmoid(z[0]);
    
    //if (s == 142)  cout << s << " DEBUG A: " << w[0] << endl;
    
    for (int i = 1; i < hiddenLayersSize; ++i)
    {
        //if (s == 142) cout << s << " BEFOR DEBUG A: " << i << ": " << w[i] << endl;
        
        z[i] = prod(a[i-1], w[i]);
        
        //if (s == 142) cout << s << " AFTER DEBUG A: " << i << ": " << w[i] << endl;
        
        z[i] += b[i];
        a[i] = sigmoid(z[i]);
    }
    
//    if (s == 142)
//    {
//        //cout<< "WAIT, a: " << z[hiddenLayersSize - 1] << endl;
//        for (int i = 0; i < hiddenLayersSize; ++i)
//        {
//            cout << "WAIT A, " << i << ": " << a[i] << endl;
//            cout << "WAIT Z, " << i << ": " << z[i] << endl;
//        }
//    }
    
    return a[hiddenLayersSize - 1];
}

double NeuralNetworkManager::sigmoid(double x)
{
    //return 1.0 / (1.0 + exp(-x));
    double temp = 1.0 / (1.0 + exp(-x));
    if (isnan(temp))
    {
        //return 1;
        cout << s << " SIGMOID NAN: " << x << endl;
    }
    return temp;
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
    {
        if (isnan(sigmoid(m(0, i))))
        {
            cout << "sigmoid_matrix_nan: " << sigmoid(m(0, i)) << endl;
        }
        result(0, i) = sigmoid(m(0, i));
    }
    
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

void NeuralNetworkManager::debugDisplayParams()
{
    cout << "DEBUG DISPLAY PARAMS START" << endl;
    
    for (int i = 0; i < hiddenLayersSize; ++i)
    {
        cout << endl;
        cout << "LAYER " << i << endl;
        cout << endl;
        
        cout << "BIASES:" << b[i] << endl;
//        for (int j = 0; j < hiddenLayers[i]; ++j)
//        {
//            cout << j << ": " << b[i](0, j) << endl;
//        }
        
        cout << endl;
        cout << "WEIGHTS: " << w[i] << endl;
        //matrix<double> prevLayer = i == 0 ?
    }
    
    cout << "DEBUG DISPLAY PARAMS END" << endl;
}

void NeuralNetworkManager::debugDisplayCalculated()
{
    cout << "DEBUG DISPLAY CALCULATED START" << endl;
    
    for (int i = 0; i < hiddenLayersSize; ++i)
    {
        cout << endl;
        cout << "LAYER " << i << endl;
        
        cout << endl;
        cout << "A:" << a[i] << endl;
//        for (int j = 0; j < hiddenLayers[i]; ++j)
//        {
//            cout << j << ": " << b[i](0, j) << endl;
//        }
        
        cout << endl;
        cout << "Z: " << z[i] << endl;
        
        cout << endl;
        cout << "DCA: " << dca[i] << endl;
        
        cout << endl;
        cout << "DCB: " << dcb[i] << endl;
        
        cout << endl;
        cout << "DCW: " << dcw[i] << endl;
    }
    
    cout << "DEBUG DISPLAY CALCULATED END" << endl;
}
