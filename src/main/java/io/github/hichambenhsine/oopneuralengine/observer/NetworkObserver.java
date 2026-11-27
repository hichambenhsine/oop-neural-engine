package io.github.hichambenhsine.oopneuralengine.observer;

/**
 * Observer interface for components reacting to neural network state transitions.
 */
public interface NetworkObserver {

    void onStateChanged(NetworkEvent event);
}