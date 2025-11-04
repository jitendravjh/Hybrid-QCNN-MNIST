import tensorflow as tf
import matplotlib
# Force matplotlib to use a non-interactive backend
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from src.data_preprocessing import load_and_filter_mnist
from src.hybrid_model import create_classical_model, create_quantum_circuit, build_hybrid_model

def main():
    # --- 1. Configuration ---
    DIGIT_1 = 3  # The '0' class
    DIGIT_2 = 6  # The '1' class
    N_QUBITS = 8 # Number of qubits (must match classical model's output dim)
    EPOCHS = 5
    BATCH_SIZE = 32

    # --- 2. Load Data ---
    x_train, y_train, x_test, y_test = load_and_filter_mnist(DIGIT_1, DIGIT_2)

    # Get the input shape for the classical model
    input_shape = x_train.shape[1:] # (14, 14, 1)

    # --- 3. Build Models ---
    print("\n--- Building Models ---")

    # 3.1. Classical feature extractor
    classical_model = create_classical_model(input_shape)

    # 3.2. Quantum circuit (NOW returns symbols)
    quantum_circuit, observable, symbols = create_quantum_circuit(N_QUBITS)

    # 3.3. Hybrid model (NOW takes symbols)
    hybrid_model = build_hybrid_model(classical_model, quantum_circuit, observable, symbols)

    # --- 4. Compile Model ---
    hybrid_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Hybrid Model Summary:")
    hybrid_model.summary()

    # --- 5. Train Model ---
    print("\n--- Starting Training ---")

    history = hybrid_model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        verbose=1
    )

    print("--- Training Complete ---")

    # --- 6. Evaluate Model ---
    print("\n--- Evaluating Model ---")

    # Print classification report
    y_pred_probs = hybrid_model.predict(x_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    print(classification_report(y_test, y_pred, target_names=[f"Digit {DIGIT_1}", f"Digit {DIGIT_2}"]))

    # Plot accuracy and loss
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.suptitle("Hybrid QCNN Training History")
    plt.savefig("training_history.png")
    print(f"\nTraining history plot saved to 'training_history.png'")

if __name__ == "__main__":
    main()
