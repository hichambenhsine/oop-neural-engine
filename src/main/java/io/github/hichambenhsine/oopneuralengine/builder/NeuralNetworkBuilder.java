package io.github.hichambenhsine.oopneuralengine.builder;

import io.github.hichambenhsine.oopneuralengine.activation.ActivationFunction;
import io.github.hichambenhsine.oopneuralengine.activation.IdentityActivation;
import io.github.hichambenhsine.oopneuralengine.layer.Layer;
import io.github.hichambenhsine.oopneuralengine.layer.LayerFactory;
import io.github.hichambenhsine.oopneuralengine.layer.LayerType;
import io.github.hichambenhsine.oopneuralengine.model.NeuralNetwork;
import io.github.hichambenhsine.oopneuralengine.model.Neuron;
import io.github.hichambenhsine.oopneuralengine.observer.NetworkObserver;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.DoubleSupplier;
import java.util.function.UnaryOperator;

/**
 * Fluent Builder for assembling fully-connected {@link NeuralNetwork} instances
 * using {@link LayerFactory} and optional {@link Layer} decorators.
 */
public class NeuralNetworkBuilder {

    private final String name;
    private final LayerFactory layerFactory = new LayerFactory();
    private final List<Layer> layers = new ArrayList<>();
    private final List<NetworkObserver> observers = new ArrayList<>();

    private DoubleSupplier weightSupplier = () -> new Random().nextDouble();
    private double defaultBias = 0.0;
    private int trainingEpochs = 5;
    private boolean hasOutputLayer = false;
    private boolean built = false;

    public NeuralNetworkBuilder() {
        this("NeuralNetwork");
    }

    public NeuralNetworkBuilder(String name) {
        this.name = Objects.requireNonNull(name, "Network name must not be null");
    }

    public NeuralNetworkBuilder inputLayer(int neuronCount) {
        return inputLayer(neuronCount, new IdentityActivation());
    }

    public NeuralNetworkBuilder inputLayer(int neuronCount, ActivationFunction activationFunction) {
        if (!layers.isEmpty()) {
            throw new IllegalStateException("Input layer must be added first");
        }
        layers.add(layerFactory.createLayer(LayerType.INPUT, neuronCount, activationFunction));
        return this;
    }

    public NeuralNetworkBuilder hiddenLayer(int neuronCount, ActivationFunction activationFunction) {
        ensureInputExistsAndOutputAbsent();
        layers.add(layerFactory.createLayer(LayerType.HIDDEN, neuronCount, activationFunction));
        return this;
    }

    public NeuralNetworkBuilder outputLayer(int neuronCount, ActivationFunction activationFunction) {
        ensureInputExistsAndOutputAbsent();
        layers.add(layerFactory.createLayer(LayerType.OUTPUT, neuronCount, activationFunction));
        hasOutputLayer = true;
        return this;
    }

    /**
     * Wraps the most recently added layer with a dynamic {@link Layer} decorator.
     */
    public NeuralNetworkBuilder decorateLastLayer(UnaryOperator<Layer> decorator) {
        Objects.requireNonNull(decorator, "Decorator function must not be null");
        if (layers.isEmpty()) {
            throw new IllegalStateException("No layer available to decorate");
        }
        int lastIndex = layers.size() - 1;
        Layer decorated = Objects.requireNonNull(
                decorator.apply(layers.get(lastIndex)),
                "Decorator must not return null"
        );
        layers.set(lastIndex, decorated);
        return this;
    }

    public NeuralNetworkBuilder withUniformWeight(double weight) {
        if (!(weight >= 0.0 && weight <= 1.0)) {
            throw new IllegalArgumentException("Weight must be between 0.0 and 1.0, got: " + weight);
        }
        this.weightSupplier = () -> weight;
        return this;
    }

    public NeuralNetworkBuilder withRandomWeights(long seed) {
        Random seededRandom = new Random(seed);
        this.weightSupplier = seededRandom::nextDouble;
        return this;
    }

    public NeuralNetworkBuilder withWeightSupplier(DoubleSupplier weightSupplier) {
        this.weightSupplier = Objects.requireNonNull(weightSupplier, "Weight supplier must not be null");
        return this;
    }

    /**
     * Sets a uniform bias applied to all hidden and output neurons.
     */
    public NeuralNetworkBuilder withDefaultBias(double defaultBias) {
        this.defaultBias = defaultBias;
        return this;
    }

    public NeuralNetworkBuilder withObserver(NetworkObserver observer) {
        observers.add(Objects.requireNonNull(observer, "Observer must not be null"));
        return this;
    }

    public NeuralNetworkBuilder withTrainingEpochs(int trainingEpochs) {
        if (trainingEpochs <= 0) {
            throw new IllegalArgumentException("Training epochs must be positive");
        }
        this.trainingEpochs = trainingEpochs;
        return this;
    }

    public NeuralNetwork build() {
        if (built) {
            throw new IllegalStateException("This builder instance has already built a NeuralNetwork");
        }
        if (layers.size() < 2 || !hasOutputLayer) {
            throw new IllegalStateException("A neural network requires at least an input layer and an output layer");
        }

        built = true;
        NeuralNetwork network = new NeuralNetwork(name);
        network.setTrainingEpochs(trainingEpochs);

        for (NetworkObserver observer : observers) {
            network.addObserver(observer);
        }

        // Apply bias to non-input layers and wire fully-connected topology
        for (int i = 0; i < layers.size(); i++) {
            Layer current = layers.get(i);
            if (current.getType() != LayerType.INPUT && defaultBias != 0.0) {
                for (Neuron neuron : current.getNeurons()) {
                    neuron.setBias(defaultBias);
                }
            }
            if (i > 0) {
                Layer previous = layers.get(i - 1);
                connectFully(previous, current);
            }
            network.addLayer(current);
        }

        return network;
    }

    private void connectFully(Layer sourceLayer, Layer targetLayer) {
        for (Neuron source : sourceLayer.getNeurons()) {
            for (Neuron destination : targetLayer.getNeurons()) {
                source.connectTo(destination, weightSupplier.getAsDouble());
            }
        }
    }

    private void ensureInputExistsAndOutputAbsent() {
        if (layers.isEmpty()) {
            throw new IllegalStateException("An input layer must be added before hidden or output layers");
        }
        if (hasOutputLayer) {
            throw new IllegalStateException("No layer can be added after the output layer");
        }
    }
}