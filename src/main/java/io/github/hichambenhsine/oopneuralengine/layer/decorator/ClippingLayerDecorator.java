package io.github.hichambenhsine.oopneuralengine.layer.decorator;

import io.github.hichambenhsine.oopneuralengine.layer.Layer;

/**
 * Decorator clamping a layer's output values within [min, max]
 * to guard against exploding activations.
 */
public class ClippingLayerDecorator extends LayerDecorator {

    private final double min;
    private final double max;

    public ClippingLayerDecorator(Layer decoratedLayer, double min, double max) {
        super(decoratedLayer);
        if (Double.isNaN(min) || Double.isNaN(max) || min > max) {
            throw new IllegalArgumentException("Invalid clipping bounds: [" + min + ", " + max + "]");
        }
        this.min = min;
        this.max = max;
    }

    @Override
    public double[] forward(double[] inputs) {
        double[] outputs = super.forward(inputs);
        double[] clipped = new double[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            clipped[i] = Math.max(min, Math.min(max, outputs[i]));
        }
        return clipped;
    }
}