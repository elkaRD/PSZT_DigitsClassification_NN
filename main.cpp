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
#include "DigitsNN.hpp"

#define MNIST_DATA_LOCATION "./"

using namespace std;

int main(int argc, char* argv[])
{

    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    MnistDataset dataset = readDataset(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
    
    DigitsNN nn({32, 32});
    
    for (int i = 0; i < dataset.training_images.size(); ++i)
    {
        nn.addTraining(dataset.training_images[i], dataset.training_labels[i]);
    }

    for (int i = 0; i < dataset.test_images.size(); ++i)
    {
        nn.addTest(dataset.test_images[i], dataset.test_labels[i]);
    }
    
    nn.test();
    nn.load("32_32_biases.txt");
    //for (int i = 0; i < 10; ++i)
    {
        //cout << i << " iter" << endl;
        //nn.learn();
        nn.test();
        //nn.save("32_32_biases.txt");
    }
    
    cout << "Image: " << nn.recognize("digit.data", true) << endl;
    
    return 0;
}
