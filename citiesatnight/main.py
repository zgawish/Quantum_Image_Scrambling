from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, QuantumRegister
from qiskit.qasm import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit import execute, Aer, BasicAer
import numpy as np
import matplotlib.pyplot as plt
from resizeimage import resizeimage
from PIL import Image

import frqi
import quantum_edge_detection as qed


#provider = IBMQ.get_provider(hub='ibm-q-keio', group='keio-internal', project='keio-students')
#
# anc = QuantumRegister(1, "anc")
# img = QuantumRegister(11, "img")
# anc2 = QuantumRegister(1, "anc2")
# c = ClassicalRegister(12)
#
# qc = QuantumCircuit(anc, img, anc2, c)

#imageNames = ["Ref_Tokyo_grayscale.jpg", "Tokyo_grayscale.jpg", "Sapporo_grayscale.jpg"]
imageNames = ["mlk.jpg"]
imageNum1 = 0
#imageNum2 = 2

image1 = Image.open(imageNames[imageNum1]).convert('LA')
#image2 = Image.open(imageNames[imageNum2]).convert('LA')


def image_normalization(image):
	image = resizeimage.resize_cover(image, [32, 32])
	w, h = 32, 32
	#image = resizeimage.resize_cover(image, [8, 8])
	#w, h = 8, 8
	image = np.array([[image.getpixel((x,y))[0] for x in range(w)] for y in range(h)])

	# 2-dimentional data convert to 1-dimentional array
	image = image.flatten()
	# change type
	image = image.astype('float64')
	# Normalization(0~pi/2)
	image /= 255.0
	generated_image = np.arcsin(image)

	return generated_image

def Translate(increment, quantumRegister, circuit):
    from qiskit.aqua.components.qfts import Standard as qft
    from qiskit.aqua.components.iqfts import Standard as iqft

    n = len(quantumRegister)

    qft(n).construct_circuit(qubits=quantumRegister, circuit=circuit)

    for j in range(n):
        circuit.u1((np.pi * increment) / (2 ** (n - 1 - j)), quantumRegister[j])

    iqft(n).construct_circuit(qubits=quantumRegister, circuit=circuit)
IBMQ.load_account()
genimg_og = None
equal_arrays = False
m = 0
image1 = image_normalization(image1)
#while equal_arrays != True:
for m in range(15):
    provider = IBMQ.get_provider()
    anc = QuantumRegister(1, "anc")
    img = QuantumRegister(11, "img")
    anc2 = QuantumRegister(1, "anc2")
    c = ClassicalRegister(12)
    print("m: ",m)

    qc = QuantumCircuit(anc, img, anc2, c)
    if m> 0:
        image1 = Image.open(imageNames[m]).convert('LA')
        image_array= np.array(image1)
        N = image_array.shape[0]
        x, y = np.meshgrid(range(N), range(N))
        xmap = (2*x+y)  % N
        ymap = (x+y) % N
        im  = image_array[xmap,ymap]
        im_image = Image.fromarray(im)
        print(im_image.size)
        image1 = image_normalization(im_image.convert("LA"))

    #image2 = image_normalization(image1)


    # apply hadamard gates
    for i in range(1, len(img)):
        qc.h(img[i])

    # encode ref image
    for i in range(len(image1)):
            if image1[i] != 0:
                    frqi.c10ry(qc, 2 * image1[i], format(i, '010b'), img[0], anc2[0], [img[j] for j in range(1,len(img))])

    #qc.x(img)
    #qed.quantum_edge_detection(qc)
    qc.measure(anc, c[0])
    qc.measure(img, c[1:12])
    print(qc.depth())
    numOfShots = 8192
    result = execute(qc, provider.get_backend('ibmq_qasm_simulator'), shots=numOfShots, backend_options={"fusion_enable":True}).result()
    #circuit_drawer(qc).show()
    #plot_histogram(result.get_counts(qc))

    print(result.get_counts(qc))

    # generated image
    genimg = np.array([])

    #### decode
    for i in range(len(image1)):
            try:
                    genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'10']/numOfShots)])
            except KeyError:
                    genimg = np.append(genimg,[0.0])

    # inverse nomalization
    genimg *= 32.0 * 255.0
    #genimg *= 8.0*255.0

    # convert type
    genimg = genimg.astype('int')

    # back to 2-dimentional data
    genimg = genimg.reshape((32,32))
    #genimg = genimg.reshape((8,8))


    image_show = Image.fromarray(genimg.astype('uint8'))
    image_show.show()
    image_show.save('gen_'+str(m)+'.png')
    if m == 0:
        genimg_og = genimg
    if m>0:
        comparison = genimg == genimg_og
        equal_arrays = comparison.all()
        if equal_arrays:
            print("yay")
    #plt.imshow(genimg, cmap='gray', vmin=0, vmax=255)
   #plt.savefig('gen_'+str(i)+'.png')
    imageNames.append('gen_'+str(m)+'.png')
    #m+=1
    #plt.show()

