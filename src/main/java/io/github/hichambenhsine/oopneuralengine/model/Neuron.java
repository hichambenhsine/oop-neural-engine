package io.github.hichambenhsine.oopneuralengine.model;

import io.github.hichambenhsine.oopneuralengine.activation.ActivationFunction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Artificial neuron holding incoming/outgoing connections, a bias, and a swappable
 * {@link ActivationFunction} strategy.
 */
public class Neuron {

    private final String id;
    private double bias;
    private ActivationFunction activationFunction;
    private final List<Connection> incomingConnections = new ArrayList<>();
    private final List<Connection> outgoingConnections = new ArrayList<>();
    private double output;

    public Neuron(String id, ActivationFunction activationFunction) {
        this(id, 0.0, activationFunction);
    }

    public Neuron(String id, double bias, ActivationFunction activationFunction) {
        this.id = Objects.requireNonNull(id, "Neuron id must not be null");
        this.bias = bias;
        this.activationFunction = Objects.requireNonNull(activationFunction, "Activation function must not be null");
    }

    /**
     * Connects this neuron (source) to a target neuron (destination) with the given weight.
     */
    public Connection connectTo(Neuron destination, double weight) {
        Connection connection = new Connection(this, destination, weight);
        this.outgoingConnections.add(connection);
        destination.incomingConnections.add(connection);
        return connection;
    }

    /**
     * Computes S = sum(X_i * W_i) + bias using the current outputs of connected source neurons.
     */
    public double computeWeightedSum() {
        double sum = bias;
        for (Connection connection : incomingConnections) {
            sum += connection.getSource().getOutput() * connection.getWeight();
        }
        return sum;
    }

    /**
     * Computes S = sum(inputs[i] * W_i) + bias using an explicit input signal vector
     * aligned with this neuron's incoming connections.
     */
    public double computeWeightedSum(double[] inputs) {
        if (inputs == null || inputs.length != incomingConnections.size()) {
            throw new IllegalArgumentException(
                    "Expected " + incomingConnections.size() + " inputs for neuron " + id
                            + ", got: " + (inputs == null ? "null" : inputs.length)
            );
        }
        double sum = bias;
        for (int i = 0; i < incomingConnections.size(); i++) {
            sum += inputs[i] * incomingConnections.get(i).getWeight();
        }
        return sum;
    }

    /**
     * Runs forward propagation on this neuron using connected source outputs.
     */
    public double forward() {
        double sum = computeWeightedSum();
        this.output = activationFunction.activate(sum);
        return this.output;
    }

    /**
     * Activates a raw external value (typically used by input layer neurons).
     */
    public double activateExternalInput(double input) {
        this.output = activationFunction.activate(input + bias);
        return this.output;
    }

    public String getId() {
        return id;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = Objects.requireNonNull(activationFunction, "Activation function must not be null");
    }

    public List<Connection> getIncomingConnections() {
        return Collections.unmodifiableList(incomingConnections);
    }

    public List<Connection> getOutgoingConnections() {
        return Collections.unmodifiableList(outgoingConnections);
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }
}