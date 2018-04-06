import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import matplotlib.pyplot as plt

from python_speech_features import mfcc

from python_speech_features import logfbank

from scipy.spatial import distance as dist
def euc_distance(A,B):
	
	return np.linalg.norm(np.mean(A,0)-np.mean(B,0))
	#return np.sqrt(np.sum((np.mean(A,0)-np.mean(B,0))**2))
	
def cosine_distance(A, B):
	return dist.cosine(np.mean(A,0),np.mean(B,0))

fs, x = wav.read('female1.wav')
fs, y = wav.read('female12.wav') #same as female 1 different words spoken
fs, z = wav.read('female2.wav')	#different female
fs, a = wav.read('male1.wav')
fs, b = wav.read('male12.wav')
fs, c = wav.read('male2.wav')
fs, d = wav.read('male3.wav')

x_feat = mfcc(x,fs)
y_feat = mfcc(y,fs)
z_feat = mfcc(z,fs)
a_feat = mfcc(a,fs)
b_feat = mfcc(b,fs)
c_feat = mfcc(c,fs)
d_feat = mfcc(d,fs)

#print euc_distance(a_feat,y_feat)
	
	     #Male1 Female1 Female2 Male3 Female1 Male2 Male1 #expected total speaker count 5
mfcc_list = [a_feat,y_feat,z_feat,d_feat,x_feat,c_feat,b_feat]

new_mfcc = [a_feat] #Compare Male1 speaker with the reset
speaker_count = 1
for i in range(1,len(mfcc_list)):
	count = 0
	for j in range(0,speaker_count):
		distance = cosine_distance(new_mfcc[j],mfcc_list[i])
		print distance
		if distance > .07:
			count = count + 1
		else:
			new_mfcc[j] = np.concatenate([new_mfcc[j],mfcc_list[i]])
	if count >= speaker_count:
		speaker_count = speaker_count+1
		print "New speaker segment %i" % i
		new_mfcc.append(mfcc_list[i])

print "Total Speaker Count: %i" %speaker_count


#print cosine_distance(x_feat,y_feat)
#print cosine_distance(x_feat,z_feat)
#print cosine_distance(y_feat,z_feat)
#print cosine_distance(a_feat,x_feat)
#print cosine_distance(a_feat,y_feat)
#print cosine_distance(a_feat,z_feat)

print "Female 1 recording 1 vs Female 1 Recording 2"
print cosine_distance(x_feat,y_feat)
print "Female 1 recording 1 vs Female 2"
print cosine_distance(x_feat,z_feat)
print "Female 1 Recording 2 vs Female 2"
print cosine_distance(y_feat,z_feat)
print "Male 1 recording 1 vs Female 1 recording 1"
print cosine_distance(a_feat,x_feat)
print "Male 1 recording 1 vs Female 1 recording 2"
print cosine_distance(a_feat,y_feat)
print "Male 1 recording 1 vs Female 2"
print cosine_distance(a_feat,z_feat)
print "Male 1 recording 2 vs Male 1 recording 1"
print cosine_distance(b_feat,a_feat)
print "Male 1 recording 2 vs Female 1 recording 1"
print cosine_distance(b_feat,x_feat)
print "Male 1 recording 2 vs Female 1 recording 2"
print cosine_distance(b_feat,y_feat)
print "Male 1 recording 2 vs Female 2"
print cosine_distance(b_feat,z_feat)
print "Male 2 vs Male 1 recording 1"
print cosine_distance(c_feat,a_feat)
print "Male 2 vs Male 1 recording 2"
print cosine_distance(c_feat,b_feat)
print "Male 2 vs Female 1 recording 1"
print cosine_distance(c_feat,x_feat)
print "Male 2 vs Female 1 recording 2"
print cosine_distance(c_feat,y_feat)
print "Male 2 vs Female 2"
print cosine_distance(c_feat,z_feat)

'''
# create data

fs, x = wav.read('male1.wav')
plt.figure(3)
mfcc_feat = mfcc(x[0:64000],fs)
#fbank_feat = logfbank(x,fs)
#print (fbank_feat[1:3,:])
#plt.plot(mfcc_feat[:,:],'y')
print fs
#plt.figure(1)
c = fft(x[0:16000])
d = len(c)/2

m11 = mfcc_feat[:,:];
#plt.subplot(421)
#plt.plot(abs(c[:(d-1)]),'r')

#c = fft(x[16000:32000])
#d = len(c)/2
#plt.subplot(422)
#plt.plot(abs(c[:(d-1)]),'r')

fs, x = wav.read('female1.wav')

mfcc_feat = mfcc(x[0:64000],fs)
#fbank_feat = logfbank(x,fs)
#print (fbank_feat[1:3,:])

f11 = mfcc_feat[:,:];

fs, x = wav.read('female12.wav')
#plt.plot(mfcc_feat[:,:],'r')

mfcc_feat = mfcc(x[0:64000],fs)
#plt.plot(mfcc_feat[:,:],'g')
f12 = mfcc_feat[:,:];
print f11
print f12
print m11
#print dist.sqeuclidean(m11,f11)
#print dist.sqeuclidean(f11,f12)

fs, x = wav.read('female2.wav')
mfcc_feat = mfcc(x[0:64000],fs)
#fbank_feat = logfbank(x,fs)
#print (fbank_feat[1:3,:])
f21 = mfcc_feat[:,:];

fs, x = wav.read('DifferentMixFemale1Male1Male2Female2.wav')
fs, x = wav.read('mixed.wav')
fs, x = wav.read('mixed2.wav')

for i in range(0,10000):
	print x[i]
plt.plot(x[0:10000])
plt.show()
length = len(x)
segments = []
foundPeak = 0
for i in range(0,len(x)):
	if np.absolute(x[i]) > 150 and foundPeak == 0:
		segments.append(i)
		foundPeak = 1
	elif np.absolute(x[i]) < 1 and foundPeak == 1:
		segments.append(i)
		foundPeak = 0


seconds = int(fs*2)

numberofsegments  = int(np.ceil(length/seconds))


MFCC = []
for i in range(0,len(segments)-1):
	#MFCC.append( mfcc(x[(i*seconds):np.minimum(((i+1)*seconds),length)],fs))
	MFCC.append( mfcc(x[segments[i]:segments[i+1]],fs))


speaker_count = 1

NewMFCC = []
NewMFCC.append(MFCC[0])
MAX_COSINE_DISTANCE = .91
for k in range(1,len(MFCC)):
	count = 0
	for j in range(0,speaker_count):
		distance  = cosine_distance(MFCC[k],NewMFCC[j])
		print k
		print distance
		print j
		if distance > MAX_COSINE_DISTANCE:
			count = count + 1
		else:
print len(a_feat)
			NewMFCC[j] = np.concatenate([NewMFCC[j],MFCC[k]])
	if count == speaker_count:
		print "The Segment which is a new speaker %i" % k 
		speaker_count = speaker_count + 1
		NewMFCC.append(MFCC[k])

print "MAX COSINE DISTANCE THRESHOLD IN RADIANS: %f" %(MAX_COSINE_DISTANCE) 
print "%i Segments of length  %f Seconds" %(numberofsegments,(seconds/fs))
print "Total Speaker Count %i" % speaker_count
#print cosine_distance(MFCC[1],MFCC[2])
#print euc_dist(MFCC[3],MFCC[4])

#if cosine_distance(m11,f11) > .86:
#	speaker_count = speaker_count+1
if cosine_distance(f11,f12) > .86:
	speaker_count = speaker_count+1
if cosine_distance(f11,f21) > .86:
	speaker_count = speaker_count+1



print speaker_count

c = fft(x[0:16000])
d = len(c)/2
plt.subplot(423)
plt.plot(abs(c[:(d-1)]),'y')
 
c = fft(x[16000:32000])
d = len(c)/2
plt.subplot(424)
plt.plot(abs(c[:(d-1)]),'y')



ps, p = wav.read('female2.wav')

c = fft(p[0:16000]) 
d = len(c)/2 
plt.subplot(425)
plt.plot(abs(c[:(d-1)]),'b')

c = fft(p[16000:32000])
d = len(c)/2
plt.subplot(426)
plt.plot(abs(c[:(d-1)]),'b')

#ps, p = wav.read('speechm.wav')

#c = fft(p[0:22050])    
#d = len(c)/2 
#plt.subplot(427)
#plt.plot(abs(c[:(d-1)]),'g')

#c = fft(p[22050:44100])
#d = len(c)/2
#plt.subplot(428)
#plt.plot(abs(c[:(d-1)]),'g')


#plt.show()
'''
