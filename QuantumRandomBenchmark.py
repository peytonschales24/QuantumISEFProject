#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import datetime
import time

import numpy as np
from math import sqrt, cos, sin, pi
import cmath
from qiskit import (
    IBMQ, QuantumCircuit, execute, transpile, providers
)


# In[2]:

FORMAT="{0:0.8f}"

hwname = 'ibmq_quito'
provider = IBMQ.load_account()
hardware = provider.get_backend(hwname)
status = hardware.status()
if not status.operational:
    print(hwname+" is currently offline.")
    exit(1)
conf = hardware.configuration().to_dict()
maxCircuits = conf['max_experiments']
nQbits = conf['n_qubits']
maxShots = conf['max_shots']
couples = conf['coupling_map']

coupleMap={}

for pair in couples:
    a = int(pair[0])
    b = int(pair[1])
    if a < b:
        if not a in coupleMap:
            coupleMap[a] = {}
        coupleMap[a][b] = 1

# In[3]:

#define function for generating gates based on arbitrary bloch sphere rotations:
IDGATE=np.asarray([[1,0],[0,1]])
XGATE=np.asarray([[0,1],[1,0]])
YGATE=np.asarray([[0,-1j],[1j,0]])
ZGATE=np.asarray([[1,0],[0,-1]])
def Rn(theta, nx, ny, nz):
    return cos(theta/2)*IDGATE-1j*sin(theta/2)*(nx*XGATE+ny*YGATE+nz*ZGATE)
    


# In[4]:


# PI=3.141592635

# SQ2=1.0/sqrt(2.0)
# HGATE=np.asarray([[SQ2,SQ2],[SQ2,-SQ2]])
# SGATE=np.asarray([[1,0],[0,1j]])
# S3GATE= np.asarray([[1, 0], [0, -1j]])
# SHGATE= SGATE @ HGATE
# HSGATE= HGATE @ SGATE
# HSHGATE= HGATE @ SHGATE
# SHSGATE= SHGATE @HGATE

# SHS3GATE= SHGATE @S3GATE
# S3HSGATE= S3GATE @HGATE @SGATE

# HSHSGATE= HGATE @ SHSGATE
# SHSHGATE= SHSGATE @ HGATE
# HS3GATE= HGATE @ S3GATE
# S3HGATE= S3GATE @ HGATE
# HS3HGATE= HS3GATE @ HGATE
# S3HS3GATE= S3GATE @ HS3GATE
# HS3HS3GATE= HS3HGATE @ S3GATE
# S3HS3HGATE= S3GATE @ HS3HGATE

# HSHS3GATE= HSGATE @ HS3GATE
# HS3HSGATE= HS3GATE @ HSGATE

# SHS3HGATE= SHGATE @ S3HGATE
# S3HSHGATE= S3HGATE @ SHGATE

# HSHSHGATE= HGATE@SGATE@HGATE@SGATE@HGATE
# HS3HS3HGATE= HGATE@S3GATE@HGATE@S3GATE@HGATE

# HSHS3HGATE= HGATE@SGATE@HGATE@S3GATE@HGATE
# HS3HSHGATE= HGATE@S3GATE@HGATE@SGATE@HGATE

# HSHSHSGATE= HGATE@SGATE@HGATE@SGATE@HGATE@SGATE
# HS3HS3HS3GATE= HGATE@S3GATE@HGATE@S3GATE@HGATE@S3GATE

# HSHSHSHGATE= HGATE@SGATE@HGATE@SGATE@HGATE@SGATE@HGATE
# HS3HS3HS3HGATE= HGATE@S3GATE@HGATE@S3GATE@HGATE@S3GATE@HGATE

# HSHSHSHSHGATE= HGATE@SGATE@HGATE@SGATE@HGATE@SGATE@HGATE@SGATE@HGATE
# HS3HS3HS3HS3HGATE= HGATE@S3GATE@HGATE@S3GATE@HGATE@S3GATE@HGATE@S3GATE@HGATE

# 
# HXGATE= HGATE @ XGATE
# XHGATE= XGATE @ HGATE

# HXSHGATE= HXGATE @ SHGATE
# XHSHGATE= XHGATE @ SHGATE

# SXGATE= SGATE @ XGATE 
# XSGATE= HGATE @ SGATE 
# S3XGATE= S3GATE @ XGATE
# XS3GATE= XGATE @ S3GATE

# SHXHGATE= SHGATE @ XHGATE 

# 
# HYGATE= HGATE @ YGATE
# YHGATE= YGATE @ HGATE

# HYSHGATE= HYGATE @ SHGATE
# YHSHGATE= YHGATE @ SHGATE


# SYGATE= SGATE @ YGATE 
# YSGATE= YGATE @ SGATE
# S3YGATE= S3GATE @ YGATE
# YS3GATE= YGATE  @ S3GATE
# SHYHGATE= SHGATE @ YHGATE 

# ZGATE=np.asarray([[1,0],[0,-1]])
# IDGATE=np.asarray([[1,0],[0,1]])


# In[5]:


PauliGates = []
PauliGates.append(IDGATE)
PauliGates.append(XGATE)
PauliGates.append(YGATE)
PauliGates.append(ZGATE)


#result.append((np.dot(-1,XGATE).tolist(),"-X"))
#result.append((np.dot(-1,YGATE).tolist(), "-Y"))
#result.append((np.dot(-1,ZGATE).tolist(), "-Z"))
#result.append((np.dot(-1,IDGATE).tolist(),"-I"))

#result.append((np.dot(1j,XGATE).tolist(), "iX"))
#result.append((np.dot(1j,YGATE).tolist(), "iY"))
#result.append((np.dot(1j,ZGATE).tolist(), "iZ"))
#result.append((np.dot(1j,IDGATE).tolist(),"iI"))

#result.append((np.dot(-1j,XGATE).tolist(),"-iX"))
#result.append((np.dot(-1j,YGATE).tolist(),"-iY"))
#result.append((np.dot(-1j,ZGATE).tolist(),"-iZ"))
#result.append((np.dot(-1j,IDGATE).tolist(),"-iI"))


# In[6]:


CliffordGates=[IDGATE, XGATE, YGATE, ZGATE]
clifford_axes=[(1,0,0),(1,0,0), (0, 1, 0), (0,1,0),(0,0,1), (0,0,1) ,(1/sqrt(2), 1/sqrt(2) , 0), (1/sqrt(2), 0, 1/sqrt(2)), (0, 1/sqrt(2), 1/sqrt(2)), (-1/sqrt(2), 1/sqrt(2), 0),
               (1/sqrt(2), 0, -1/sqrt(2)), (0, -1/sqrt(2), 1/sqrt(2)), (1/sqrt(3),1/sqrt(3),1/sqrt(3)), (1/sqrt(3),1/sqrt(3),1/sqrt(3)),
              (-1/sqrt(3),1/sqrt(3),1/sqrt(3)), (-1/sqrt(3),1/sqrt(3),1/sqrt(3)), (1/sqrt(3),-1/sqrt(3),1/sqrt(3)), (1/sqrt(3), -1/sqrt(3),1/sqrt(3)),
              (1/sqrt(3),1/sqrt(3),-1/sqrt(3)), (1/sqrt(3),1/sqrt(3),-1/sqrt(3))]
clifford_rotation_angles=[pi/2, -pi/2, pi/2,-pi/2, pi/2, -pi/2, pi, pi, pi, pi, pi, pi, (2/3)*pi, -(2/3)*pi, (2/3)*pi, -(2/3)*pi, (2/3)*pi, -(2/3)*pi, (2/3)*pi, -(2/3)*pi]
for axis, angle in zip(clifford_axes, clifford_rotation_angles):
    CliffordGates.append(Rn(angle, *axis))
#print(CliffordGates)


# In[ ]:





# In[7]:


def invert2x2(matrix):
    return np.linalg.inv(matrix)

def mul2x2(A,B):
    r = np.matmul(A, B)
    return r

def distance(a,b):
    A = np.subtract(a,b)
    return np.linalg.norm(A)

def dagger(A):
    a = np.conjugate(A[0][0])
    b = np.conjugate(A[0][1])
    c = np.conjugate(A[1][0])
    d = np.conjugate(A[1][1])
    return np.asarray([[a,c],[b,d]])


# In[8]:


def findGate(current):
    #print('Current Gate= ', current)
    #print('Original Determinant=', np.linalg.det(current))
    rescale_factor=(1/cmath.sqrt(np.linalg.det(current)))
    #print('Rescale factor: ', rescale_factor)
    rescaled_current= rescale_factor*current
    minimum = distance((1/cmath.sqrt(np.linalg.det(CliffordGates[0])))*CliffordGates[0], rescaled_current)
    minndx=0
    for ndx in range(1,len(CliffordGates)):
        if minimum < 0.005:
            break
        d = distance((1/cmath.sqrt(np.linalg.det(CliffordGates[ndx])))*CliffordGates[ndx], rescaled_current)
        if d < minimum:
            minimum = d
            minndx = ndx
    print('Closest Gate Found: ', (1/cmath.sqrt(np.linalg.det(CliffordGates[minndx])))*CliffordGates[minndx])
    if(minimum > 0.005):
        print('Minimum Distance Found= ', minimum)
        print('Determinant=', np.linalg.det(rescaled_current))
        print('Current Gate= ', rescaled_current)
        return None
    else:
        return CliffordGates[minndx]


# In[9]:


def runJob(circuits, backend, shotCount):

    if len(circuits) == 0:
        print("No circuits to run.", file=sys.stderr)
        return (None, None, None, None, None)
    
    if len(circuits) > maxCircuits:
        print("Circuit count exceeds maximum; splitting job.", file=sys.stderr)
        setA = circuits[:maxCircuits]
        setB = circuits[maxCircuits:]
        (jidA, sTime, et, elapsedA, resA) = runJob(setA, backend, shotCount)
        time.sleep(30)
        (jidB, st, eTime, elapsedB, resB) = runJob(setB, backend, shotCount)
        time.sleep(30)
        elapsed = elapsedA + elapsedB
        for x in resB:
            resA.append(x)
        return(jidA+","+jidB, sTime, eTime, elapsed, resA)
    
    startTime = datetime.datetime.now()
    print("Executing "+str(len(circuits))+" circuits ["+str(startTime)+"]", file=sys.stderr)
    job = execute(circuits, backend, shots=shotCount)
    result = job.result()
    endTime = datetime.datetime.now()
    print("Completed ["+str(endTime)+"]", file=sys.stderr)
    
    elapsed = result.time_taken
    rd = result.to_dict()
    jid = rd['job_id']
    res = rd['results']

    info=[]

    for cnum in (range(0,len(circuits))):
        r = res[cnum]
        cid=r['header']['name']
        
        if(r['success'] != True):
            info.append((cid,None))
        else:
            outputBits=int(r['header']['memory_slots'])
            fmt="{0:0"+str(outputBits)+"b}"
            counts = r['data']['counts']
            d = {}
            for xkey in counts:
                key = fmt.format(int(xkey,16))
                d[key] = counts[xkey]
            info.append((cid,d))
    return (jid, str(startTime), str(endTime), elapsed, info)


# In[10]:


L=[8, 16,32,64,128,256]

P=PauliGates
nP = len(P)
C=CliffordGates[4:23]
nC = len(C)  

# for x in PauliGates:
#     P.append(x)

# 
# 

# for m in CliffordGates:
#     found=False
#     for p in P:
#         if distance(m,p) < 0.05:
#             found=True
#             break
#     if not found:
#         print('Not Found', m)
#         print('Determinant=', np.linalg.det(m))
#         C.append(m)
        

#       


# In[11]:


#print(nC)
#print(nP)


# In[12]:


rng = np.random.default_rng()

experiment_list=[]

counter = 1
for l,circlen in enumerate(L):
    for i in range(0, nP):
        pauli_ndx_set=rng.integers(low=0,high=nP-1,size=int(circlen/2),endpoint= True)

        pauli_gates = [P[k] for k in pauli_ndx_set]

        for j in range(0, nC-1):
            cliff_ndx_set=rng.integers(low=0,high=nC-1,size=int(circlen/2),endpoint=True)
            cliff_gates = [C[k] for k in cliff_ndx_set]

            gates=[]
            for ndx in range(0,len(pauli_gates)):
                gates.append(pauli_gates[ndx])
                gates.append(cliff_gates[ndx])

            prod=gates[0]

            for ndx in range(1,len(gates)):
                prod=mul2x2(gates[ndx], prod)
            
            found_inverse=False
            for possibleinverse in CliffordGates:
                possibleidentity= possibleinverse @ prod
                if distance(possibleidentity, IDGATE)<.05:
                    gates.append(possibleinverse)
                    break
                elif distance(possibleidentity, -1*IDGATE)<.05:
                    gates.append(possibleinverse)
                    break
                elif distance(possibleidentity, 1j*IDGATE)<.05:
                    gates.append(possibleinverse)
                    break
                elif distance(possibleidentity, -1j*IDGATE)<.05:
                    gates.append(possibleinverse)
                    break

            for qb in range(0,nQbits):
                circuitName = str(circlen)+":"+str(qb)+":"+str(counter)
                circuit = QuantumCircuit(qb+1,1,name=circuitName)
                for gate in gates:
                    circuit.unitary(gate, [qb])
                    circuit.barrier(qb)
                circuit.measure(qb, 0)
                experiment_list.append(circuit)
            for qa in range(0, nQbits):
                for qb in range(qa+1, nQbits):
                    circuitName = str(circlen)+":"+str(qa)+"/"+str(qb)+":"+str(counter)
                    circuit = QuantumCircuit(qb+1, 2, name=circuitName)
                    for gate in gates:
                        circuit.unitary(gate, [qa])
                        circuit.unitary(gate, [qb])
                        circuit.barrier(qa)
                        circuit.barrier(qb)
                    circuit.measure([qa,qb],[0,1])
                    experiment_list.append(circuit)
            counter = counter + 1


# In[13]:


#print(len(experiment_list))

#exit(0)
# In[ ]:

print("Starting execution. "+str(len(experiment_list))+" circuits.", file=sys.stderr)

(jid, stime, etime, elapsed, info) = runJob(experiment_list, hardware, maxShots)

now = datetime.datetime.now()
timestamp = str(now.year)+str(now.month)+str(now.day)+str(now.hour)+str(now.minute)+str(now.second)
filename = "random_benchmark_"+hwname+"_"+timestamp+".txt"

with open(filename,"w") as output:
    print("#HW "+hwname, file=output)
    Counts={}
    Error1={}
    Error2={}

    for rec in info:
        print("%raw "+rec[0],end="", file=output)
        total = 0
        error = 0
        error2 = 0
        for c in rec[1]:
            print(" "+str(c)+"="+str(rec[1][c]),end="",file=output)
            n = int(rec[1][c])
            total = total + n
            if c == "1":
                error = error + n
            elif c == "11":
                error2 = error2 + n
            elif c == "10" or c == "01":
                error = error + n
        print("",file=output)

        (ngates,qbits,exp) = rec[0].split(":")
        if not ngates in Counts:
            Counts[ngates] = {}
            Error1[ngates] = {}
            Error2[ngates] = {}

        if not qbits in Counts[ngates]:
            Counts[ngates][qbits] = 0
            Error1[ngates][qbits] = 0
            Error2[ngates][qbits] = 0

        Counts[ngates][qbits] = Counts[ngates][qbits] + total
        Error1[ngates][qbits] = Error1[ngates][qbits] + error
        Error2[ngates][qbits] = Error2[ngates][qbits] + error2

    CountCoupled = {}
    ExpectedTotal = {}
    ErrorTotal = {}

    for ngates in Counts:
        for qbits in Counts[ngates]:
            if qbits.find("/") == -1:
                continue
            (qa,qb) = qbits.split("/")
            qa_total = Counts[ngates][qa]
            qb_total = Counts[ngates][qb]
            qa_error = Error1[ngates][qa]
            qb_error = Error1[ngates][qb]

            qa_prob = float(qa_error) / qa_total
            qb_prob = float(qb_error) / qb_total

            count2 = Counts[ngates][qbits]
            error2 = Error2[ngates][qbits]

            qab_prob = float(error2) / count2
            expected = qa_prob * qb_prob

            a = int(qa)
            b = int(qb)

            coupled = False
            if a in coupleMap and b in coupleMap[a]:
                coupled = True

            if not ngates in CountCoupled:
                CountCoupled[ngates] = {}
                ExpectedTotal[ngates] = {}
                ErrorTotal[ngates] = {}

            if not coupled in CountCoupled[ngates]:
                CountCoupled[ngates][coupled] = 0
                ExpectedTotal[ngates][coupled] = 0
                ErrorTotal[ngates][coupled] = 0

            CountCoupled[ngates][coupled] = CountCoupled[ngates][coupled] + 1
            ExpectedTotal[ngates][coupled] = ExpectedTotal[ngates][coupled] + expected
            ErrorTotal[ngates][coupled] = ErrorTotal[ngates][coupled] + qab_prob

            print("%pair "+str(ngates)+" "+qa+" "+qb+" "+str(coupled)+" "+str(count2)+" "+FORMAT.format(expected)+" "+FORMAT.format(qab_prob)+" "+FORMAT.format(qab_prob / expected), file=output)

    for ngates in CountCoupled:
        for coupled in CountCoupled[ngates]:
            n = CountCoupled[ngates][coupled]
            expected = ExpectedTotal[ngates][coupled] / n
            actual = ErrorTotal[ngates][coupled] / n
            print("%Overall "+str(ngates)+" "+str(coupled)+" "+FORMAT.format(expected)+" "+FORMAT.format(actual)+" "+FORMAT.format(actual/expected),file=output)
        
print("Completed.  Output in "+filename)




# In[ ]:





# In[ ]:





# In[ ]:




