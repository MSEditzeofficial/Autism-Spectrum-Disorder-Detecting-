package com.autism.autism_detection;

import java.io.File;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;

public class App{
    public static void main(String[] args) throws Exception {

        // Load CSV
        int batchSize = 4;
        int labelIndex = 10;
        int numClasses = 1;

        CSVRecordReader recordReader = new CSVRecordReader(1, ',');
        recordReader.initialize(new FileSplit(new File("data/autism.csv")));

        DataSetIterator iterator = new RecordReaderDataSetIterator(
                recordReader, batchSize, labelIndex, numClasses);
           
                DataNormalization scaler = new NormalizerStandardize();
                   scaler.fit(iterator);
                     iterator.setPreProcessor(scaler);

                  // Reset iterator
                         iterator.reset();

                  // THEN model creation starts
            MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.01))
                .list()

                .layer(new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(8)
                        .activation(Activation.RELU)
                        .build())

                .layer(new DenseLayer.Builder()
                        .nOut(4)
                        .activation(Activation.RELU)
                        .build())

                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nOut(1)
                        .build())

                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        // Train Model
        for (int i = 0; i < 50; i++) {
            iterator.reset();
            model.fit(iterator);
        }

        // Test Prediction
        iterator.reset(); // IMPORTANT
        DataSet data = iterator.next();
        INDArray output = model.output(data.getFeatures());

        System.out.println("Predictions:");
        System.out.println(output);

        // Final Result
        for (int i = 0; i < output.rows(); i++) {
            if (output.getDouble(i) > 0.5) {
                System.out.println("Autism Detected");
            } else {
                System.out.println("No Autism");
            }
        }
        
// ================= CNN MODEL =================

int height = 100;
int width = 100;
int channels = 3;
int outputNum = 2;
int batchSizeImg = 4;

// DATASET
File imageDir = new File("dataset");

FileSplit imageSplit = new FileSplit(imageDir, NativeImageLoader.ALLOWED_FORMATS);
ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

ImageRecordReader imageReader = new ImageRecordReader(height, width, channels, labelMaker);
imageReader.initialize(imageSplit);

DataSetIterator imageIter =
        new RecordReaderDataSetIterator(imageReader, batchSizeImg, 1, outputNum);

// NORMALIZATION (IMPORTANT)
DataNormalization scalerImg = new ImagePreProcessingScaler(0, 1);
scalerImg.fit(imageIter);
imageIter.setPreProcessor(scalerImg);
imageIter.reset();

// ================= CNN ARCHITECTURE =================

MultiLayerConfiguration cnnConf = new NeuralNetConfiguration.Builder()
        .seed(123)
        .updater(new Adam(0.001))
        .weightInit(WeightInit.XAVIER)
        .list()

        // 1️⃣ Convolution Layer
        .layer(new ConvolutionLayer.Builder(3,3)
                .nIn(channels)
                .nOut(16)
                .stride(1,1)
                .activation(Activation.RELU)
                .build())

        // 2️⃣ Pooling Layer
        .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .build())

        // 3️⃣ Convolution Layer
        .layer(new ConvolutionLayer.Builder(3,3)
                .nOut(32)
                .stride(1,1)
                .activation(Activation.RELU)
                .build())

        // 4️⃣ Pooling Layer
        .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .build())

        // 5️⃣ Dense Layer
        .layer(new DenseLayer.Builder()
                .nOut(64)
                .activation(Activation.RELU)
                .build())

        // 6️⃣ Output Layer
        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())

        .setInputType(InputType.convolutional(height, width, channels)) // ✅ IMPORTANT
        .build();

// BUILD MODEL
MultiLayerNetwork cnnModel = new MultiLayerNetwork(cnnConf);
cnnModel.init();

// TRAIN
for (int i = 0; i < 5; i++) {
    imageIter.reset();
    cnnModel.fit(imageIter);
}

// PREDICT
imageIter.reset();
INDArray cnnOutput= cnnModel.output(imageIter.next().getFeatures());

System.out.println("CNN Predictions:");
System.out.println(cnnOutput);

for (int i = 0; i < cnnOutput.rows(); i++) {
    double noAutism = cnnOutput.getDouble(i, 0);
    double autism = cnnOutput.getDouble(i, 1);

    if (autism > noAutism) {
        System.out.println("Autism Detected");
        } 
    else {
        System.out.println("No Autism");
         }
   }
  }
}