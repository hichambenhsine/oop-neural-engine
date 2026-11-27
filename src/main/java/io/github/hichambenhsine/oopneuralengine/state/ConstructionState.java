package io.github.hichambenhsine.oopneuralengine.state;

import io.github.hichambenhsine.oopneuralengine.layer.Layer;
import io.github.hichambenhsine.oopneuralengine.model.NeuralNetwork;

/**
 * Initial state in which layers are assembled.
 * Supports transitioning to TRAINING via {@code train()} or directly to TRAINED via {@code fit()}.
 */
public class ConstructionState implements NetworkState {

    @Override
    public NetworkStateType getStateType() {
        return NetworkStateType.CONSTRUCTION;
    }

    @Override
    public void addLayer(NeuralNetwork network, Layer layer) {
        network.internalAddLayer(layer);
    }

    @Override
    public void train(NeuralNetwork network) {
        if (network.getLayers().isEmpty()) {
            throw new IllegalStateException("Cannot train a neural network without layers");
        }
        network.setState(new TrainingState());
        network.performTraining();
        network.setState(new TrainedState());
    }

    @Override
    public void fit(NeuralNetwork network) {
        if (network.getLayers().isEmpty()) {
            throw new IllegalStateException("Cannot fit a neural network without layers");
        }
        network.setState(new TrainedState());
    }

    @Override
    public double[] predict(NeuralNetwork network, double[] input) {
        throw new IllegalStateException("Prediction is not allowed in CONSTRUCTION state. Call train() or fit() first.");
    }
}