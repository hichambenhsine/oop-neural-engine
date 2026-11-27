package io.github.hichambenhsine.oopneuralengine.state;

import io.github.hichambenhsine.oopneuralengine.layer.Layer;
import io.github.hichambenhsine.oopneuralengine.model.NeuralNetwork;

/**
 * Ready/execution state allowing {@code predict()} calls.
 */
public class TrainedState implements NetworkState {

    @Override
    public NetworkStateType getStateType() {
        return NetworkStateType.TRAINED;
    }

    @Override
    public void addLayer(NeuralNetwork network, Layer layer) {
        throw new IllegalStateException("Cannot add layers once the network is TRAINED.");
    }

    @Override
    public void train(NeuralNetwork network) {
        throw new IllegalStateException("Network is already in TRAINED state.");
    }

    @Override
    public void fit(NeuralNetwork network) {
        throw new IllegalStateException("Network is already in TRAINED state.");
    }

    @Override
    public double[] predict(NeuralNetwork network, double[] input) {
        return network.performPrediction(input);
    }
}