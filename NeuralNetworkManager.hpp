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
    
    void learn(std::vector<double> image, int expected);
    void learnInt8(std::vector<uint8_t> image, int expected);
    
private:
    
    const static int INPUT_NEURONS;
    const static int OUTPUT_NEURONS;
    const static bool ENABLE_BIASES;
    
    class Layer
    {
        
    };
    
//    matrix<double> input;
//    matrix<double> expecteddutput;
    
    int hiddenLayersSize;
    std::vector<int> hiddenLayers;
    
    std::vector<matrix<double>> a;
    std::vector<matrix<double>> w;
    std::vector<matrix<double>> b;
    std::vector<matrix<double>> z;
    std::vector<matrix<double>> dca;
    std::vector<matrix<double>> dcb;
    std::vector<matrix<double>> dcw;
    
    double randomValue();
    
    matrix<double> forward(std::vector<double> input);
    
    double sigmoid(double x);
    double dSigmoid(double x);
    
    matrix<double> sigmoid(matrix<double> &m);
    matrix<double> dSigmoid(matrix<double> &m);
    
    matrix<double> toDiagonal(const matrix<double> &m);
    matrix<double> sameRows(const matrix<double> &m, const int rows);
    
    void debugDisplayParams();
    void debugDisplayCalculated();
};

#endif /* NeuralNetworkManager_hpp */
