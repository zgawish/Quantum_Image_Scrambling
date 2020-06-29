# Quantum_Image_Scrambling
Using Qiskit to create a quantum image encoder/decoder

Currently utilizing code from https://medium.com/@sorinalbolos/quantum-image-analysis-possible-speedup-for-detecting-the-skew-angle-of-documents-18ce0e0a529f to represent the image as a state vector. This code takes in a square grayscale image, represents it as a statevector, and extracts it back to its classical representation. In this representation, the pixel intensity is  stored as the amplitude,  while the pixel positon in stored within the state. 
