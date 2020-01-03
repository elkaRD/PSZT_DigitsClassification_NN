//
//  NeuralNetworkManager.hpp
//  PSZT_DigitsClassification_NN
//
//  Created by Robert Dudziński on 29/12/2019.
//  Copyright © 2019 Robert. All rights reserved.
//

#ifndef NeuralNetworkManager_hpp
#define NeuralNetworkManager_hpp

#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "MnistReader.hpp"

using namespace boost::numeric::ublas;

class NeuralNetworkManager
{
public:
    NeuralNetworkManager(std::vector<int> hiddenLayers);
    
    int detectDigit(std::vector<double> image);
    int detectDigitInt8(std::vector<uint8_t> image);
    
private:
    
    class Layer
    {
        
    };
    
//    matrix<double> input;
//    matrix<double> expecteddutput;
    
    int hiddenLayersSize;
    
    std::vector<matrix<double>> a;
    std::vector<matrix<double>> w;
    std::vector<matrix<double>> b;
    std::vector<matrix<double>> z;
    std::vector<matrix<double>> dca;
    
    double randomValue();
    
    matrix<double> forward(std::vector<double> input);
    
    double sigmoid(double x);
    double dSigmoid(double x);
    
    matrix<double> sigmoid(matrix<double> &m);
    matrix<double> dSigmoid(matrix<double> &m);
};

#endif /* NeuralNetworkManager_hpp */
