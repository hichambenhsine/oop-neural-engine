package io.github.hichambenhsine.oopneuralengine.observer;

import io.github.hichambenhsine.oopneuralengine.state.NetworkStateType;

import java.time.Instant;

/**
 * Immutable snapshot emitted when a neural network changes lifecycle state.
 */
public record NetworkEvent(
        String networkName,
        NetworkStateType previousState,
        NetworkStateType newState,
        int layerCount,
        int totalNeurons,
        int totalConnections,
        long predictionCount,
        Instant timestamp
) {
}