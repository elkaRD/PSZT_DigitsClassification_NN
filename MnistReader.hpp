//
//  MnistReader.hpp
//  PSZT_DigitsClassification_NN
//
//  Created by Robert Dudziński on 29/12/2019.
//  Copyright © 2019 Robert. All rights reserved.
//

#ifndef MnistReader_hpp
#define MnistReader_hpp

#include <iostream>
#include <vector>
#include "mnist/mnist_reader.hpp"

struct MnistDataset
{
    std::vector<std::vector<uint8_t>> training_images; ///< The training images
    std::vector<std::vector<uint8_t>> test_images;     ///< The test images
    std::vector<uint8_t> training_labels; ///< The training labels
    std::vector<uint8_t> test_labels;     ///< The test labels
};

MnistDataset readDataset(std::string path);

#endif /* MnistReader_hpp */
