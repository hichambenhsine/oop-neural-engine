package io.github.hichambenhsine.oopneuralengine;

import io.github.hichambenhsine.oopneuralengine.activation.IdentityActivation;
import io.github.hichambenhsine.oopneuralengine.activation.LegacyActivationAdapter;
import io.github.hichambenhsine.oopneuralengine.activation.LegacyTanhActivation;
import io.github.hichambenhsine.oopneuralengine.activation.SigmoidActivation;
import io.github.hichambenhsine.oopneuralengine.activation.SoftMaxActivation;
import io.github.hichambenhsine.oopneuralengine.aspect.SecurityContext;
import io.github.hichambenhsine.oopneuralengine.builder.NeuralNetworkBuilder;
import io.github.hichambenhsine.oopneuralengine.layer.decorator.DropoutLayerDecorator;
import io.github.hichambenhsine.oopneuralengine.layer.decorator.NormalizationLayerDecorator;
import io.github.hichambenhsine.oopneuralengine.model.NeuralNetwork;
import io.github.hichambenhsine.oopneuralengine.observer.NetworkDashboard;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Random;

/**
 * End-to-end demonstration of the OOP Neural Network Engine
 * combining GoF Design Patterns and AspectJ AOP.
 */
public class Main {

    private static final Logger LOGGER = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        try {
            runDemo();
        } finally {
            SecurityContext.clear();
        }
    }

    private static void runDemo() {
        NetworkDashboard dashboard = new NetworkDashboard();

        // Build Figure 11.1 (3-5-2) topology using Builder + Factory + Strategy + Decorator + Observer
        NeuralNetwork network = new NeuralNetworkBuilder("Figure-11.1-Engine")
                .inputLayer(3, new IdentityActivation())
                .hiddenLayer(5, new SigmoidActivation())
                .decorateLastLayer(layer -> new NormalizationLayerDecorator(
                        new DropoutLayerDecorator(layer, 0.2, new Random(42))
                ))
                .outputLayer(2, new SoftMaxActivation())
                .withRandomWeights(42L)
                .withDefaultBias(0.1)
                .withObserver(dashboard)
                .withTrainingEpochs(3)
                .build();

        // 1. Security Aspect check: unauthenticated call must fail
        try {
            network.train();
        } catch (SecurityException ex) {
            LOGGER.info("[DEMO] Expected security block: {}", ex.getMessage());
        }

        // 2. Authenticate as TRAINER and verify State guard (predict blocked in CONSTRUCTION)
        SecurityContext.login("hicham", "TRAINER");
        try {
            network.predict(new double[]{1.0, 0.5, 0.25});
        } catch (IllegalStateException ex) {
            LOGGER.info("[DEMO] Expected state guard: {}", ex.getMessage());
        }

        // 3. Train network: CONSTRUCTION -> TRAINING -> TRAINED (notifies Dashboard)
        network.train();

        // 4. Switch to USER role and run prediction
        SecurityContext.login("guest-user", "USER");
        double[] prediction = network.predict(new double[]{1.0, 0.5, 0.25});
        LOGGER.info("[DEMO] 3-5-2 SoftMax prediction: {}", Arrays.toString(prediction));

        // 5. Demonstrate Legacy Adapter + direct fit() transition (CONSTRUCTION -> TRAINED)
        SecurityContext.login("admin", "ADMIN");
        NeuralNetwork legacyNet = new NeuralNetworkBuilder("Legacy-Tanh-Net")
                .inputLayer(2)
                .outputLayer(1, new LegacyActivationAdapter(new LegacyTanhActivation()))
                .withUniformWeight(0.5)
                .withObserver(dashboard)
                .build();

        legacyNet.fit();
        double[] legacyOutput = legacyNet.predict(new double[]{0.8, 0.4});
        LOGGER.info("[DEMO] Legacy Tanh Adapter prediction: {}", Arrays.toString(legacyOutput));
    }
}