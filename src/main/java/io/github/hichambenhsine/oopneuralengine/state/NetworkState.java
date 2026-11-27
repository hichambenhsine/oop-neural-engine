package io.github.hichambenhsine.oopneuralengine.state;

import io.github.hichambenhsine.oopneuralengine.layer.Layer;
import io.github.hichambenhsine.oopneuralengine.model.NeuralNetwork;

/**
 * State interface governing allowed operations for each neural network lifecycle phase.
 */
public interface NetworkState {

    NetworkStateType getStateType();

    void addLayer(NeuralNetwork network, Layer layer);

    void train(NeuralNetwork network);

    void fit(NeuralNetwork network);

    double[] predict(NeuralNetwork network, double[] input);
}