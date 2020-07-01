from PIL import Image
import math
import numpy as np
from qiskit import *
from qiskit.aqua.components.qfts import Standard as qft
from qiskit.tools.visualization import plot_histogram
import time
from qiskit.providers.jobstatus import JobStatus


def getStatevectorFromImage(path):
	image = Image.open(path)
	(width, height) = image.size  # image of height 2^m x 2^n = 2^(m=N) qubits
	print("image size: ", image.size)
	pixels = image.load()
	ssum = 0
	v = 0
	check = pixels[0, 0]
	
	for i in range(width):
		for j in range(height):
			ssum += pixels[i,j]**2

	# print("ssum: ",math.sqrt(ssum))
	m = math.floor(math.log(height, 2))
	n = math.floor(math.log(width, 2))
	# print(m,n)
	stateVector = np.zeros(2 ** (m + n))
	for i in range(width):
		for j in range(height):
			stateVector[i * height + j] = pixels[i, j]/ math.sqrt(ssum)
	return stateVector, m, n


def Translate(increment, quantumRegister, circuit):
    from qiskit.aqua.components.qfts import Standard as qft
    from qiskit.aqua.components.iqfts import Standard as iqft

    n = len(quantumRegister)

    qft(n).construct_circuit(qubits=quantumRegister, circuit=circuit)

    for j in range(n):
        circuit.u1((np.pi * increment) / (2 ** (n - 1 - j)), quantumRegister[j])

    iqft(n).construct_circuit(qubits=quantumRegister, circuit=circuit)

def exportQuantumImage(counts, shots, height, width):

	r = math.floor(math.log(width, 2))
	img = Image.new("L", (width, height))
	pixels = img.load()
	maxAmplitude = 256
	medAmplitude = maxAmplitude / 2
	med = shots / (height * width)
	for key in counts:
		i = int(key[0:r], 2)
		j = int(key[r:], 2)
		val = round((((counts[key] - med) / med) * medAmplitude) + medAmplitude)
		pixels[i, j] = (val)

	return img


stateVector, m, n = getStatevectorFromImage("mlk.jpg")

state = [complex(x) for x in stateVector]
indexes = range(m+n)
columnReg = QuantumRegister(n)
rowReg = QuantumRegister(m)
creg = ClassicalRegister(n + m)

circ = QuantumCircuit(rowReg, columnReg, creg)
circ.initialize(state, indexes)
for i in range(n):
#print("state: ", state)


    #qft(m).construct_circuit(qubits=rowReg,circuit=circ)
    #qft(n).construct_circuit(qubits=columnReg,circuit=circ)

    Translate(i, rowReg, circ)
    #Translate(2**(m-1), columnReg, circ)
#circ.x(rowReg)
#circ.x(columnReg)
circ.measure(indexes, indexes)

IBMQ.load_account()
# provider = IBMQ.get_provider(group='open')
# backend =  provider.get_backend('ibmq_qasm_simulator')
# #backend = IBMQ.backend(name="ibmq_qasm_simulator")
#
# #shots = 2**(m+n)
# shots =2424
# qobj = assemble(transpile(circ, backend=backend), backend=backend)
# job = backend.run(qobj)
#
# try:
#     job_status = job.status()  # Query the backend server for job status.
#     if job_status is JobStatus.RUNNING:
#         print("The job is still running")
# except IBMQJobApiError as ex:
#     print("Something wrong happened!: {}".format(ex))

provider = IBMQ.get_provider(group='open')
backend = provider.get_backend('ibmq_qasm_simulator')

shots = 2024 
result = execute(circ, backend, shots=shots).result()
counts = result.get_counts(circ)

#result = job.result()
# = result.get_counts(circ)
#plot_histogram(counts)
#print(result.status())

exportQuantumImage(counts,shots,2**m,2**n).show()
