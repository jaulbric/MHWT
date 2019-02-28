#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
import numpy.ma as ma
from scipy import ndimage
import multiprocessing as mp

data_dict = {}

class wavelets:
	"""object that contains wavelet transform methods"""
	def __init__(self, multiprocessing=False, ncores=None):
		self.multiprocessing = multiprocessing
		if ncores is None:
			self.ncores = mp.cpu_count()
		else:
			self.ncores = ncores

	def prefactor(self,order):
		"""Normalization factor for Mexican Hat Wavelet Family"""
		return ((-1.)**order)/((2.**order)*math.factorial(order))

	def filter_size(self,xsize,ysize,angle,truncate):
		"""Determines the size of the filter kernel. Useful if the wavelet kernel is not symmetric."""
		width = 2.*truncate*np.sqrt((xsize*xsize*np.cos(angle)*np.cos(angle)) + (ysize*ysize*np.sin(angle)*np.sin(angle)))
		height = 2.*truncate*np.sqrt((xsize*xsize*np.sin(angle)*np.sin(angle)) + (ysize*ysize*np.cos(angle)*np.cos(angle)))
		return np.ceil(width), np.ceil(height)
	
	def threshold(self,a,threshmin=None,threshmax=None,newval=0.):
		a = ma.array(a, copy=True)
		mask = np.zeros(a.shape, dtype=bool)
		if threshmin is not None:
			mask |= (a < threshmax).filled(False)
		if threshmax is not None:
			mask |= (a > threshmax).filled(False)
		a[mask] = newval
		return a
	
	def detect_local_maxima(self, arr, threshold=2, mask=None):
		"""Finds local maximum in arrays above a specified threshold"""
		if not isinstance(arr,np.ndarray):
			arr = np.asarray(arr)

		neighborhood = ndimage.morphology.generate_binary_structure(len(arr.shape),2)
	
		data = self.threshold(arr, threshmin=threshold, newval=0)
	
		local_max = (ndimage.filters.maximum_filter(data, footprint=neighborhood)==data)
		background = (data==0)
	
		eroded_background = ndimage.morphology.binary_erosion(background, structure=neighborhood, border_value=1)
	
		detected_maxima = np.bitwise_xor(local_max, eroded_background)
	
		if mask is None:
			return np.where(detected_maxima), detected_maxima
	
		else:
			mask_bool = (mask>=1)
			detected_maxima = detected_maxima*mask_bool
			return np.where(detected_maxima), detected_maxima

	def wavelet_kernel(self,sigma,order=0,rotate=0.0,truncate=4.0):
		"""Returns a 2D gaussian wavelet as a numpy array. See https://arxiv.org/pdf/astro-ph/0604376.pdf

		Parameters:
		sigma				:	Wavelet scale. Can be a float, list, tuple or ndarray.
		order (integer)		:	Order of the wavelet. Default is 0.
		rotate (float)		:	Angle in degrees of ellipse with respect to horizontal.
		truncate (float)	:	Size of the kernel image in standard deviations. Default is 4.0.
		"""
		assert isinstance(order,int)

		if isinstance(sigma,float):
			sigma = np.array([sigma,sigma])
		else:
			assert len(sigma) == 2

		angle = np.pi*rotate/180.
		
		xsize, ysize = self.filter_size(sigma[0],sigma[1],angle,truncate)
		
		x_temp, y_temp = np.meshgrid(np.linspace(-xsize/2.,xsize/2.,num=xsize,endpoint=True), np.linspace(-ysize/2.,ysize/2.,num=ysize,endpoint=True))

		x = x_temp*np.cos(angle) - y_temp*np.sin(angle)
		y = x_temp*np.sin(angle) + y_temp*np.cos(angle)

		gaussian = (1./(2.*np.pi*(sigma[0]*sigma[1]))) * np.exp(-0.5*(((x**2)/(sigma[0]**2)) + ((y**2)/(sigma[1]**2))))

		if order == 0:
			filter_kernel = gaussian
		elif order == 1:
			filter_kernel = self.prefactor(1) * gaussian * (1./((sigma[0]**4)*(sigma[1]**4))) * (((x**2)*(sigma[1]**4)) - ((sigma[0]**2)*(sigma[1]**4)) - ((sigma[0]**4)*(sigma[1]**2)) + ((y**2)*(sigma[0]**4)))
		elif order == 2:
			filter_kernel = self.prefactor(2) * gaussian * (1./((sigma[0]**8)*(sigma[1]**8))) * (((x**4)*(sigma[1]**8)) + ((y**4)*(sigma[0]**8)) - ((2.*(x**2))*(((sigma[0]**4)*(sigma[1]**6)) + (3.*(sigma[0]**2)*(sigma[1]**8)))) - ((2.*(y**2))*(((sigma[1]**4)*(sigma[0]**6)) + (3.*(sigma[1]**2)*(sigma[0]**8)))) + (2.*(x**2)*(y**2)*(sigma[0]**4)*(sigma[1]**4)) + (2.*(sigma[0]**6)*(sigma[1]**6)) + (3.*(sigma[0]**8)*(sigma[1]**4)) + (3.*(sigma[1]**8)*(sigma[0]**4)))
		elif order >= 3:
			filter_kernel = self.prefactor(3) * gaussian * (1./((sigma[0]**13)*(sigma[1]**13))) * (((y**6)*(sigma[0]**12)) - (15.*(y**4)*(sigma[0]**12)*(sigma[1]**2)) + (3.*(x**2)*(y**4)*(sigma[0]**8)*(sigma[1]**4)) - (3.*(y**4)*(sigma[0]**10)*(sigma[1]**4)) + (45.*(y**2)*(sigma[0]**12)*(sigma[1]**4)) - (18.*(x**2)*(y**2)*(sigma[0]**8)*(sigma[1]**6)) + (18.*(y**2)*(sigma[0]**10)*(sigma[1]**6)) - (15.*(sigma[0]**12)*(sigma[1]**6)) + (3.*(x**4)*(y**2)*(sigma[0]**4)*(sigma[1]**8)) - (18.*(x**2)*(y**2)*(sigma[0]**6)*(sigma[1]**8)) + (9.*(x**2)*(sigma[0]**8)*(sigma[1]**8)) + (9.*(y**2)*(sigma[0]**8)*(sigma[1]**8)) - (9.*(sigma[0]**10)*(sigma[1]**8)) - (3.*(x**4)*(sigma[0]**4)*(sigma[1]**10)) + (18.*(x**2)*(sigma[0]**6)*(sigma[1]**10)) - (9.*(sigma[0]**8)*(sigma[1]**10)) + ((x**6)*(sigma[1]**12)) - (15.*(x**4)*(sigma[0]**2)*(sigma[1]**12)) + (45.*(x**2)*(sigma[0]**4)*(sigma[1]**12)) - (15.*(sigma[0]**6)*(sigma[1]**12)))
			
			for n in range(4,order+1,1):
				filter_kernel = (self.prefactor(n)/self.prefactor(n-1)) * ndimage.filters.laplace(filter_kernel)

		return filter_kernel

	def wavelet_filter(self, data, sigma, order=0, mode='reflect', cval=0.0, truncate=4.0, rotate=0.0):
		"""convolves input with a wavelet kernel. Behavior is similar to ndimage.gaussian_filter"""
		assert isinstance(order,int)

		if isinstance(sigma,(float,int)):
			sigma = np.array([[sigma],[sigma]],dtype=float)
		elif not isinstance(sigma,np.ndarray):
			sigma = np.asarray(sigma,dytpe=float)

		if len(sigma.shape) == 1:
			sigma = np.array([sigma,sigma],dtype=float)

		assert sigma.shape[0] == 2
		assert sigma[0].shape == sigma[1].shape

		if self.multiprocessing:
			mp_data = mp.RawArray('d',data.shape[0]*data.shape[1])
			data_shape = data.shape
			np_data = np.frombuffer(mp_data).reshape(data_shape)
			np.copyto(np_data, data)
			pool = mp.Pool(processes=self.ncores, initializer=initialize_data, initargs=(mp_data, data_shape))
			output = np.array(pool.map(mp_convolve, [(self.wavelet_kernel((xsigma,ysigma),order=order,rotate=rotate,truncate=truncate),mode,cval) for xsigma, ysigma in zip(sigma[0], sigma[1])]))
		else:
			output = np.array([ndimage.convolve(data, self.wavelet_kernel(np.array([xsigma,ysigma]),order=order,rotate=rotate,truncate=truncate),mode=mode, cval=cval, origin=0) for xsigma, ysigma in zip(sigma[0], sigma[1])])

		return output

	def signal_to_noise(self, data, sigma, order=0, mode='reflect', cval=0.0, truncate=4.0, rotate=0.0):
		"""returns signal to noise ratio of input by convolving input with a gaussian wavelet and normalizing with the squart root of the convolution of input with the same gaussian wavelet squared."""
		assert isinstance(order,int)

		if isinstance(sigma,(float,int)):
			sigma = np.array([[sigma],[sigma]],dtype=float)
		elif not isinstance(sigma,np.ndarray):
			sigma = np.asarray(sigma,dytpe=float)

		if len(sigma.shape) == 1:
			sigma = np.array([sigma,sigma],dtype=float)

		assert sigma.shape[0] == 2
		assert sigma[0].shape == sigma[1].shape

		if not isinstance(data, np.ndarray):
			data = np.asarray(data)

		if self.multiprocessing:
			mp_data = mp.RawArray('d',data.shape[0]*data.shape[1])
			data_shape = data.shape
			np_data = np.frombuffer(mp_data).reshape(data_shape)
			np.copyto(np_data, data)
			pool = mp.Pool(processes=self.ncores, initializer=initialize_data, initargs=(mp_data, data_shape))
			output = np.array(pool.map(mp_signal_to_noise, [(self.wavelet_kernel((xsigma,ysigma),order=order,rotate=rotate,truncate=truncate),mode,cval) for xsigma, ysigma in zip(sigma[0],sigma[1])]))
		else:
			output = np.array([ndimage.convolve(data, self.wavelet_kernel(np.array([xsigma,ysigma]),order=order,rotate=rotate,truncate=truncate),mode=mode, cval=cval, origin=0)/np.sqrt(ndimage.convolve(data, self.wavelet_kernel(np.array([xsigma,ysigma]),order=order,rotate=rotate,truncate=truncate)**2, mode=mode, cval=cval, origin=0)) for xsigma, ysigma in zip(sigma[0], sigma[1])])

		return output

def mp_convolve(args):
	data = np.frombuffer(data_dict['data']).reshape(data_dict['data_shape'])
	return ndimage.convolve(data, args[0], mode=args[1], cval=args[2], origin=0)

def mp_signal_to_noise(args):
	data = np.frombuffer(data_dict['data']).reshape(data_dict['data_shape'])
	return ndimage.convolve(data,args[0],mode=args[1], cval=args[2], origin=0)/np.sqrt(ndimage.convolve(data, args[0]**2, mode=args[1], cval=args[2], origin=0))

def initialize_data(data, data_shape):
	data_dict['data'] = data
	data_dict['data_shape'] = data_shape

if __name__ == "__main__":
	import matplotlib.pyplot as plt

	w = wavelets(multiprocessing=True)

	# Create some test data
	x,y = np.meshgrid(np.linspace(-499.5,499.5,num=1000),np.linspace(-499.5,499.5,num=1000))
	data = np.random.poisson(lam=10.0,size=(1000,1000)) + (10.*np.exp(- (x**2/6.) - (y**2/6.)))
	
	sigma = np.arange(1.5,4.0,0.05) # Array of scales. Could also specify an array of shape (2,n) where n is the number of scales and each row corrosponds to the horizontal scale and vertical scale.
	
	# img = w.wavelet_filter(data,sigma,order=2,mode='reflect',rotate=0.0,truncate=3.0)
	img = w.signal_to_noise(data,sigma,order=2,mode='reflect',rotate=0.0,truncate=3.0)
	
	print img.shape

	# Find the local maximum in largest scale image
	idx, maxima = w.detect_local_maxima(img[-1],threshold=2.5)
	s = img[-1][idx]
	
	# Plot the signal to noise ratio for each scale
	fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
	ax1.imshow(data)
	ax1.set_title(r'Data')
	ax2.imshow(img[0])
	ax2.set_title(r'$\sigma = 1.5$')
	ax3.imshow(img[1])
	ax3.set_title(r'$\sigma = 2.0$')
	ax4.imshow(img[2])
	ax4.set_title(r'$\sigma = 2.5$')
	ax5.imshow(img[3])
	ax5.set_title(r'$\sigma = 3.0$')
	ax6.imshow(img[4])
	ax6.scatter(idx[1], idx[0], s=10.*s, facecolors='none', edgecolors='r')
	ax6.set_title(r'$\sigma = 3.5$')
	plt.show()
