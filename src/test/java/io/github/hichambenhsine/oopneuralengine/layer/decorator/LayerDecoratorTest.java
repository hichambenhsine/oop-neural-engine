package io.github.hichambenhsine.oopneuralengine.layer.decorator;

import io.github.hichambenhsine.oopneuralengine.activation.IdentityActivation;
import io.github.hichambenhsine.oopneuralengine.layer.Layer;
import io.github.hichambenhsine.oopneuralengine.layer.LayerFactory;
import io.github.hichambenhsine.oopneuralengine.layer.LayerType;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class LayerDecoratorTest {

    private static final double DELTA = 1e-6;

    @Test
    void normalizationDecoratorShouldProduceZeroMeanAndUnitVariance() {
        LayerFactory factory = new LayerFactory();
        Layer input = factory.createLayer(LayerType.INPUT, 4, new IdentityActivation());
        Layer normalized = new NormalizationLayerDecorator(input);

        double[] result = normalized.forward(new double[]{2.0, 4.0, 4.0, 6.0});

        double mean = 0.0;
        for (double v : result) {
            mean += v;
        }
        mean /= result.length;

        double variance = 0.0;
        for (double v : result) {
            variance += (v - mean) * (v - mean);
        }
        variance /= result.length;

        assertEquals(0.0, mean, DELTA);
        assertEquals(1.0, variance, DELTA);
        assertEquals(LayerType.INPUT, normalized.getType());
        assertEquals(4, normalized.size());
    }

    @Test
    void normalizationDecoratorShouldReturnZerosForConstantVector() {
        LayerFactory factory = new LayerFactory();
        Layer input = factory.createLayer(LayerType.INPUT, 3, new IdentityActivation());
        Layer normalized = new NormalizationLayerDecorator(input);

        double[] result = normalized.forward(new double[]{5.0, 5.0, 5.0});
        assertArrayEquals(new double[]{0.0, 0.0, 0.0}, result, DELTA);
    }

    @Test
    void dropoutDecoratorShouldScaleKeptValuesAndRespectToggle() {
        LayerFactory factory = new LayerFactory();
        Layer input = factory.createLayer(LayerType.INPUT, 4, new IdentityActivation());
        DropoutLayerDecorator dropout = new DropoutLayerDecorator(input, 0.5, new Random(42));

        double[] activeOutput = dropout.forward(new double[]{1.0, 1.0, 1.0, 1.0});
        for (double v : activeOutput) {
            assertTrue(Math.abs(v) < DELTA || Math.abs(v - 2.0) < DELTA);
        }

        dropout.setEnabled(false);
        double[] disabledOutput = dropout.forward(new double[]{1.0, 2.0, 3.0, 4.0});
        assertArrayEquals(new double[]{1.0, 2.0, 3.0, 4.0}, disabledOutput, DELTA);
    }

    @Test
    void dropoutDecoratorShouldRejectInvalidRates() {
        LayerFactory factory = new LayerFactory();
        Layer input = factory.createLayer(LayerType.INPUT, 2, new IdentityActivation());

        assertThrows(IllegalArgumentException.class, () -> new DropoutLayerDecorator(input, -0.1));
        assertThrows(IllegalArgumentException.class, () -> new DropoutLayerDecorator(input, 1.0));
    }

    @Test
    void clippingAndStackedDecoratorsShouldComposeCleanly() {
        LayerFactory factory = new LayerFactory();
        Layer base = factory.createLayer(LayerType.INPUT, 3, new IdentityActivation());

        Layer stacked = new ClippingLayerDecorator(
                new NormalizationLayerDecorator(base),
                -1.0,
                1.0
        );

        double[] output = stacked.forward(new double[]{0.0, 10.0, 100.0});
        for (double v : output) {
            assertTrue(v >= -1.0 - DELTA && v <= 1.0 + DELTA);
        }
        assertEquals(3, stacked.size());
        assertEquals(3, stacked.getNeurons().size());
    }
}