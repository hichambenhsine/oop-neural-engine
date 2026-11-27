package io.github.hichambenhsine.oopneuralengine.layer.decorator;

import io.github.hichambenhsine.oopneuralengine.layer.Layer;

/**
 * Decorator applying Z-score layer normalization (zero mean, unit variance)
 * to the wrapped layer's output vector.
 */
public class NormalizationLayerDecorator extends LayerDecorator {

    private static final double EPSILON = 1e-12;

    public NormalizationLayerDecorator(Layer decoratedLayer) {
        super(decoratedLayer);
    }

    @Override
    public double[] forward(double[] inputs) {
        double[] outputs = super.forward(inputs);
        if (outputs.length == 0) {
            return outputs;
        }

        double mean = 0.0;
        for (double v : outputs) {
            mean += v;
        }
        mean /= outputs.length;

        double variance = 0.0;
        for (double v : outputs) {
            double diff = v - mean;
            variance += diff * diff;
        }
        variance /= outputs.length;

        double stdDev = Math.sqrt(variance);
        double[] normalized = new double[outputs.length];
        if (stdDev < EPSILON) {
            return normalized;
        }

        for (int i = 0; i < outputs.length; i++) {
            normalized[i] = (outputs[i] - mean) / stdDev;
        }
        return normalized;
    }
}