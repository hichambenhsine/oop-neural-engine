package io.github.hichambenhsine.oopneuralengine.model;

import java.util.Objects;

/**
 * Directed weighted connection between a source neuron and a destination neuron.
 * Weight W must remain within [0.0, 1.0].
 */
public class Connection {

    private final Neuron source;
    private final Neuron destination;
    private double weight;

    public Connection(Neuron source, Neuron destination, double weight) {
        this.source = Objects.requireNonNull(source, "Source neuron must not be null");
        this.destination = Objects.requireNonNull(destination, "Destination neuron must not be null");
        setWeight(weight);
    }

    public Neuron getSource() {
        return source;
    }

    public Neuron getDestination() {
        return destination;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        if (!(weight >= 0.0 && weight <= 1.0)) {
            throw new IllegalArgumentException("Connection weight must be between 0.0 and 1.0, got: " + weight);
        }
        this.weight = weight;
    }
}