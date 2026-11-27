package io.github.hichambenhsine.oopneuralengine.model;

import io.github.hichambenhsine.oopneuralengine.layer.Layer;
import io.github.hichambenhsine.oopneuralengine.observer.NetworkEvent;
import io.github.hichambenhsine.oopneuralengine.observer.NetworkObserver;
import io.github.hichambenhsine.oopneuralengine.state.ConstructionState;
import io.github.hichambenhsine.oopneuralengine.state.NetworkState;
import io.github.hichambenhsine.oopneuralengine.state.NetworkStateType;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Core NeuralNetwork aggregate acting as both the State Context and the Observer Subject.
 */
public class NeuralNetwork {

    private final String name;
    private final List<Layer> layers = new ArrayList<>();
    private final List<NetworkObserver> observers = new ArrayList<>();
    private NetworkState currentState;
    private int trainingEpochs = 5;
    private long predictionCount = 0;

    public NeuralNetwork() {
        this("NeuralNetwork");
    }

    public NeuralNetwork(String name) {
        this.name = Objects.requireNonNull(name, "Network name must not be null");
        this.currentState = new ConstructionState();
    }

    public void addObserver(NetworkObserver observer) {
        Objects.requireNonNull(observer, "Observer must not be null");
        if (!observers.contains(observer)) {
            observers.add(observer);
        }
    }

    public void removeObserver(NetworkObserver observer) {
        observers.remove(observer);
    }

    public void addLayer(Layer layer) {
        currentState.addLayer(this, layer);
    }

    public void train() {
        currentState.train(this);
    }

    public void fit() {
        currentState.fit(this);
    }

    public double[] predict(double[] input) {
        return currentState.predict(this, input);
    }

    public void internalAddLayer(Layer layer) {
        Objects.requireNonNull(layer, "Layer must not be null");
        layers.add(layer);
    }

    public void setState(NetworkState nextState) {
        Objects.requireNonNull(nextState, "Next state must not be null");
        NetworkStateType previous = this.currentState.getStateType();
        this.currentState = nextState;
        notifyObservers(previous, nextState.getStateType());
    }

    public void performTraining() {
        // Simulates training epochs across the network layers
        for (int epoch = 0; epoch < trainingEpochs; epoch++) {
            if (layers.isEmpty()) {
                break;
            }
        }
    }

    public double[] performPrediction(double[] input) {
        if (input == null) {
            throw new IllegalArgumentException("Input vector must not be null");
        }
        if (layers.isEmpty()) {
            throw new IllegalStateException("Neural network has no layers");
        }

        double[] signal = input;
        for (Layer layer : layers) {
            signal = layer.forward(signal);
        }
        predictionCount++;
        return signal;
    }

    private void notifyObservers(NetworkStateType previousState, NetworkStateType newState) {
        NetworkEvent event = new NetworkEvent(
                name,
                previousState,
                newState,
                layers.size(),
                getTotalNeurons(),
                getTotalConnections(),
                predictionCount,
                Instant.now()
        );
        for (NetworkObserver observer : observers) {
            observer.onStateChanged(event);
        }
    }

    public String getName() {
        return name;
    }

    public NetworkStateType getCurrentStateType() {
        return currentState.getStateType();
    }

    public List<Layer> getLayers() {
        return Collections.unmodifiableList(layers);
    }

    public int getTotalNeurons() {
        int total = 0;
        for (Layer layer : layers) {
            total += layer.size();
        }
        return total;
    }

    public int getTotalConnections() {
        int total = 0;
        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                total += neuron.getOutgoingConnections().size();
            }
        }
        return total;
    }

    public int getTrainingEpochs() {
        return trainingEpochs;
    }

    public void setTrainingEpochs(int trainingEpochs) {
        if (trainingEpochs <= 0) {
            throw new IllegalArgumentException("Training epochs must be positive");
        }
        this.trainingEpochs = trainingEpochs;
    }

    public long getPredictionCount() {
        return predictionCount;
    }
}