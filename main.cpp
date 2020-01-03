//
//  main.cpp
//  PSZT_DigitsClassification_NN
//
//  Created by Robert Dudziński on 29/12/2019.
//  Copyright © 2019 Robert. All rights reserved.
//

#include <iostream>
#include <vector>
#include "MnistReader.hpp"
#include "NeuralNetworkManager.hpp"
//#include "mnist/mnist_reader.hpp"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#define MNIST_DATA_LOCATION "/Users/robert/studia/sem5/pszt/projekt2/PSZT_DigitsClassification_NN"

int main(int argc, char* argv[]) {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    //const std::string MNIST_DATA_LOCATION = std::string(argv[0]) + "/..";
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
    //std::cout << "Run location: " << argv[0] << std::endl;


    // Load MNIST data
    //mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    //mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    MnistDataset dataset = readDataset(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
    
    using namespace boost::numeric::ublas;
    matrix<double> m (3, 3);
    for (unsigned i = 0; i < m.size1 (); ++ i)
        for (unsigned j = 0; j < m.size2 (); ++ j)
            m (i, j) = 3 * i + j;
    std::cout << m << std::endl;
    
    //vector<int> layers = {16, 16};
    
    NeuralNetworkManager nn({16, 16});
    
    nn.detectDigitInt8(dataset.training_images[0]);

    return 0;
}
