//
//  main.cpp
//  PSZT_DigitsClassification_NN
//
//  Created by Robert Dudziński on 29/12/2019.
//  Copyright © 2019 Robert. All rights reserved.
//

#include <iostream>
#include <vector>
#include <sstream>
#include "MnistReader.hpp"
#include "DigitsNN.hpp"

#define MNIST_DATA_LOCATION "./"

using namespace std;

int main(int argc, char* argv[])
{
    std::vector<int> layerSize = {32, 32};
    string inFilename;
    string outFilename;
    double stepDst = -1;
    bool enableDynamicStep = true;
    int iterations = 0;
    int method = 0;
    
    string userDigitFilename;
    
    for (int i = 0; i < argc; ++i)
    {
        if (strcmp(argv[i], "-n") == 0)
        {
            int layers = atoi(argv[++i]);
            layerSize.clear();
            for (int j = 0; j < layers; ++j)
            {
                layerSize.push_back(atoi(argv[++i]));
            }
        }
        else if (strcmp(argv[i], "-s") == 0)
        {
            stepDst = stod(argv[++i]);
        }
        else if (strcmp(argv[i], "-fi") == 0)
        {
            inFilename = argv[++i];
        }
        else if (strcmp(argv[i], "-fo") == 0)
        {
            outFilename = argv[++i];
        }
        else if (strcmp(argv[i], "-dds") == 0)
        {
            enableDynamicStep = false;
        }
        else if (strcmp(argv[i], "-i") == 0)
        {
            iterations = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-m") == 0)
        {
            method = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-d") == 0)
        {
            userDigitFilename = argv[++i];
        }
    }

    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    MnistDataset dataset = readDataset(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
    
    if (method == 0)
    {
        DigitsNN nn(layerSize);
        
        for (int i = 0; i < dataset.training_images.size(); ++i)
        //for (int i = 0; i < 2000; ++i)
        {
            nn.addTraining(dataset.training_images[i], dataset.training_labels[i]);
        }

        for (int i = 0; i < dataset.test_images.size(); ++i)
        {
            nn.addTest(dataset.test_images[i], dataset.test_labels[i]);
        }
        
        nn.setEnableDynamicStep(enableDynamicStep);
        
        if (stepDst > 0)
            nn.setStep(stepDst);
        
        if (inFilename.size() > 0)
        {
            nn.load(inFilename + ".txt");
            nn.test();
        }
        
        ofstream log("log.txt", std::ios_base::app);
        for (int i = 0; i < argc; ++i)
            log << argv[i] << " ";
        log << endl;
        
        for (int i = 0; i < iterations; ++i)
        {
            cout << i << " iter" << endl;
            nn.learn();
            double correctness = nn.test();
            
            log << i << " " << correctness << endl;
            
            if (outFilename.size() > 0)
            {
                std::stringstream ss;
                ss << nn.getIterCounter();
                nn.save(outFilename + ss.str() + ".txt");
            }
        }
        
        if (userDigitFilename.size() > 0)
            cout << "Image: " << nn.recognize(userDigitFilename, true) << endl;
        
        log.close();
    }
    
    return 0;
}
