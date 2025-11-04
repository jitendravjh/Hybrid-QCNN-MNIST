import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

def create_classical_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    return model

def create_quantum_circuit(n_qubits):
    qubits = cirq.GridQubit.rect(1, n_qubits)
    symbols = sympy.symbols(f'theta_0:{n_qubits}')
    circuit = cirq.Circuit(cirq.H(q) for q in qubits)
    for i, q in enumerate(qubits):
        circuit.append(cirq.rz(symbols[i])(q)) 
    for i in range(n_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
    observable = cirq.Z(qubits[0])
    return circuit, observable, symbols

def build_hybrid_model(classical_model, quantum_circuit, observable, symbols):
    input_tensor = classical_model.input 
    symbol_values_tensor = classical_model(input_tensor)

    serialized_circuit = tfq.convert_to_tensor([quantum_circuit])
    programs_tensor = tf.tile(serialized_circuit, [tf.shape(symbol_values_tensor)[0]])

    symbol_names_tensor = tf.constant([str(s) for s in symbols]) 

    serialized_observable = tfq.convert_to_tensor([observable])
    operators_tensor = tf.tile(serialized_observable, [tf.shape(symbol_values_tensor)[0]])

    quantum_layer = tfq.layers.Expectation()

    quantum_output = quantum_layer(
        programs_tensor,
        symbol_names=symbol_names_tensor,
        symbol_values=symbol_values_tensor,
        operators=operators_tensor
    ) 

    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(quantum_output)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)
    return model
