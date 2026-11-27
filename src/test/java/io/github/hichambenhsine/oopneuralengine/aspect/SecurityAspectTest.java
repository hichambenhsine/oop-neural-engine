package io.github.hichambenhsine.oopneuralengine.aspect;

import io.github.hichambenhsine.oopneuralengine.activation.IdentityActivation;
import io.github.hichambenhsine.oopneuralengine.builder.NeuralNetworkBuilder;
import io.github.hichambenhsine.oopneuralengine.model.NeuralNetwork;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SecurityAspectTest {

    private NeuralNetwork buildNetwork() {
        return new NeuralNetworkBuilder("SecuredNet")
                .inputLayer(2)
                .outputLayer(1, new IdentityActivation())
                .withUniformWeight(0.5)
                .build();
    }

    @AfterEach
    void cleanup() {
        SecurityContext.clear();
    }

    @Test
    void unauthenticatedCallsShouldThrowSecurityException() {
        NeuralNetwork network = buildNetwork();

        assertThrows(SecurityException.class, network::train);
        assertThrows(SecurityException.class, network::fit);
        assertThrows(SecurityException.class, () -> network.predict(new double[]{1.0, 1.0}));
    }

    @Test
    void userRoleShouldBeDeniedTrainAndFitButAllowedPredict() {
        NeuralNetwork network = buildNetwork();

        SecurityContext.login("alice", "ADMIN");
        network.fit();

        SecurityContext.login("bob", "USER");
        assertThrows(SecurityException.class, network::train);
        assertThrows(SecurityException.class, network::fit);

        double[] result = network.predict(new double[]{2.0, 4.0});
        assertArrayEquals(new double[]{3.0}, result, 1e-6);
    }

    @Test
    void trainerRoleShouldBeAllowedTrainAndPredict() {
        NeuralNetwork network = buildNetwork();

        SecurityContext.login("carol", "TRAINER");
        network.train();

        double[] result = network.predict(new double[]{2.0, 2.0});
        assertArrayEquals(new double[]{2.0}, result, 1e-6);
    }
}