//
//  MnistReader.cpp
//  PSZT_DigitsClassification_NN
//
//  Created by Robert Dudziński on 29/12/2019.
//  Copyright © 2019 Robert. All rights reserved.
//

#include "MnistReader.hpp"

MnistDataset readDataset(std::string path)
{
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> tempDataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(path);
    
    MnistDataset dataset;
    dataset.training_images = std::move(tempDataset.training_images);
    dataset.test_images = std::move(tempDataset.test_images);
    dataset.training_labels = std::move(tempDataset.training_labels);
    dataset.test_labels = std::move(tempDataset.test_labels);
    
    return dataset;
}
