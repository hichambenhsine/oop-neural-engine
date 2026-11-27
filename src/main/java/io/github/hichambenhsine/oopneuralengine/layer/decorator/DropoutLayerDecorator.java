package io.github.hichambenhsine.oopneuralengine.layer.decorator;

import io.github.hichambenhsine.oopneuralengine.layer.Layer;

import java.util.Objects;
import java.util.Random;

/**
 * Decorator applying inverted dropout regularization to a layer's outputs.
 * Can be toggled dynamically between active (training) and bypassed (inference) modes.
 */
public class DropoutLayerDecorator extends LayerDecorator {

    private final double dropoutRate;
    private final Random random;
    private boolean enabled = true;

    public DropoutLayerDecorator(Layer decoratedLayer, double dropoutRate) {
        this(decoratedLayer, dropoutRate, new Random());
    }

    public DropoutLayerDecorator(Layer decoratedLayer, double dropoutRate, Random random) {
        super(decoratedLayer);
        if (!(dropoutRate >= 0.0 && dropoutRate < 1.0)) {
            throw new IllegalArgumentException("Dropout rate must be in [0.0, 1.0), got: " + dropoutRate);
        }
        this.dropoutRate = dropoutRate;
        this.random = Objects.requireNonNull(random, "Random generator must not be null");
    }

    public double getDropoutRate() {
        return dropoutRate;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    @Override
    public double[] forward(double[] inputs) {
        double[] outputs = super.forward(inputs);
        if (!enabled || dropoutRate == 0.0) {
            return outputs;
        }

        double scale = 1.0 / (1.0 - dropoutRate);
        double[] result = new double[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            result[i] = random.nextDouble() < dropoutRate ? 0.0 : outputs[i] * scale;
        }
        return result;
    }
}