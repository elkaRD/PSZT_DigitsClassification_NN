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
#include "DigitsNN.hpp"
//#include "mnist/mnist_reader.hpp"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#define MNIST_DATA_LOCATION "/Users/robert/studia/sem5/pszt/projekt2/PSZT_DigitsClassification_NN"

using namespace std;

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
    
//    cout << "BEG" << endl;
//    cout << (int)dataset.training_labels[0] << endl;
//    cout << "END" << endl;
    
    //vector<int> layers = {16, 16};
    
    DigitsNN nn({32,32,32});
    
//    for (int i = 0; i < dataset.training_images.size(); ++i)
//    {
//        nn.addTraining(dataset.training_images[i], dataset.training_labels[i]);
//    }
//
//    for (int i = 0; i < dataset.test_images.size(); ++i)
//    {
//        nn.addTest(dataset.test_images[i], dataset.test_labels[i]);
//    }
    
        for (int i = 0; i < 50; ++i)
        {
            nn.addTraining(dataset.training_images[i], dataset.training_labels[i]);
        }
    
        for (int i = 0; i < 50; ++i)
        {
            nn.addTest(dataset.training_images[i], dataset.training_labels[i]);
        }

    
    nn.test();
    for (int i = 0; i < 1000; ++i)
    {
        nn.learn();
        nn.test();
    }
    
    return -1;
    
//    {
//        NeuralNetworkManager nn({20, 20, 20});
//        for (int i = 0; i < 200; ++i)
//            nn.detectDigitInt8(dataset.training_images[0]);
//    }
    {
        NeuralNetworkManager nn({200, 100, 50});
        
        int correct = 0;
        int wrong = 0;
//        for (int i = 0; i < dataset.test_images.size(); ++i)
//        {
//            if (i%500 == 0) std::cout << "TESTING_BEF: " << i << "/" << dataset.test_images.size() << std::endl;
//            int result = nn.detectDigitInt8(dataset.test_images[i]);
//            if (result == (int)dataset.test_labels[i]) correct++;
//            else wrong++;
//        }
//
//        cout << "CORRECT: " << correct << endl;
//        cout << "WRONG: " << wrong << endl;
        
        for (int j = 0; j < 200; ++j)
        //for (int i = 0; i < dataset.training_images.size(); ++i)
        for (int i = 0; i < 50; ++i)
        {
            if (i%50 == 0) std::cout << "TRAINING: " << i << "/" << dataset.training_images.size() << std::endl;
            nn.learnInt8(dataset.training_images[i], dataset.training_labels[i]);
            //cout << (int)dataset.training_labels[i] << endl;
        }

        correct = 0;
        wrong = 0;
        //for (int i = 0; i < dataset.test_images.size(); ++i)
        for (int i = 0; i < 50; ++i)
        {
//            if (i%500 == 0) std::cout << "TESTING: " << i << "/" << dataset.test_images.size() << std::endl;
//            int result = nn.detectDigitInt8(dataset.test_images[i]);
//            if (result == dataset.test_labels[i]) correct++;
//            else wrong++;
            
            if (i%500 == 0) std::cout << "TESTING: " << i << "/" << dataset.training_images.size() << std::endl;
            int result = nn.detectDigitInt8(dataset.training_images[i]);
            if (result == dataset.training_labels[i]) correct++;
            else wrong++;
        }

        cout << "CORRECT: " << correct << endl;
        cout << "WRONG: " << wrong << endl;
    }
//    {
//        NeuralNetworkManager nn({16, 14});
//        for (int i = 0; i < 1420; ++i)
//            nn.detectDigit({0.2, 0.6});
//    }

    return 0;
}
