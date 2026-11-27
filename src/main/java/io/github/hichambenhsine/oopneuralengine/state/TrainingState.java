package io.github.hichambenhsine.oopneuralengine.state;

import io.github.hichambenhsine.oopneuralengine.layer.Layer;
import io.github.hichambenhsine.oopneuralengine.model.NeuralNetwork;

/**
 * Transitional state while the network is actively training.
 */
public class TrainingState implements NetworkState {

    @Override
    public NetworkStateType getStateType() {
        return NetworkStateType.TRAINING;
    }

    @Override
    public void addLayer(NeuralNetwork network, Layer layer) {
        throw new IllegalStateException("Cannot modify layers while network is in TRAINING state.");
    }

    @Override
    public void train(NeuralNetwork network) {
        throw new IllegalStateException("Network is already in TRAINING state.");
    }

    @Override
    public void fit(NeuralNetwork network) {
        throw new IllegalStateException("Cannot call fit() while network is in TRAINING state.");
    }

    @Override
    public double[] predict(NeuralNetwork network, double[] input) {
        throw new IllegalStateException("Prediction is not allowed while network is in TRAINING state.");
    }
}