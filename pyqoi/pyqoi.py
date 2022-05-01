import numpy as np

QOI_SRGB = 0
QOI_LINEAR = 1

QOI_OP_INDEX = 0x00
QOI_OP_DIFF  = 0x40
QOI_OP_LUMA  = 0x80
QOI_OP_RUN   = 0xc0
QOI_OP_RGB   = 0xfe
QOI_OP_RGBA  = 0xff

QOI_MASK_2   = 0xc0

QOI_COLOR_HASH = lambda C : C.r*3 + C.g*5 + C.b*7 + C.a*11

QOI_MAGIC = 'qoif' #qoi format header
QOI_HEADER_SIZE = 14
QOI_PADDING = [0, 0, 0, 0, 0, 0, 0, 1]

QOI_PIXELS_MAX = 400e6

def char2uint32(r, g, b, a = 0):
	return a << 24 | b << 16 | g << 8 | r

class qoi_rgba_t:
	def __init__(self, r, g, b, a):
		self.r = r
		self.g = g
		self.b = b
		self.a = a
		self.v = char2uint32(r, g, b, a)

	def set(self, pixel):
		self.r = pixel[0]
		self.g = pixel[1]
		self.b = pixel[2]
		
		if len(pixel) == 4:
			self.a = pixel[3] 
		
		self.v = char2uint32(self.r, self.g, self.b, self.a)
	
	def copy(self, other):
		self.r = other.r
		self.g = other.g
		self.b = other.b
		self.a = other.a
		self.v = other.v


class qoi_header:
	def __init__(self, width, height, channels=3, colorspace=QOI_SRGB):
		self.magic = QOI_MAGIC
		self.width = width
		self.height = height
		self.channels = channels
		self.colorspace = colorspace

def qoi_encode(pixels, header):
	if pixels.size == 0 or header == None or header.width == 0 or header.height == 0 or header.channels < 3 or header.channels > 4 or header.colorspace > 1 or header.height >= QOI_PIXELS_MAX / header.width:
		return None
	
	max_size = header.width * header.height * (header.channels + 1) + QOI_HEADER_SIZE + len(QOI_PADDING)

	p = 0
	bytes = np.ndarray((max_size, ), dtype=np.uint8)

	if bytes.size != max_size:
		return None
	
	for ch in header.magic:
		bytes[p] = ord(ch)
		p += 1
	
	width_str = hex(header.width)[2:]
	while len(width_str) < 8:
		width_str = '0' + width_str
	
	for i in range(0, 8, 2):
		bytes[p] = int(width_str[i] + width_str[i+1], 16)
		p += 1
	
	height_str = hex(header.height)[2:]
	while len(height_str) < 8:
		height_str = '0' + height_str
	
	for i in range(0, 8, 2):
		bytes[p] = int(height_str[i] + height_str[i+1], 16)
		p += 1
	
	bytes[p] = header.channels
	p += 1
	bytes[p] = header.colorspace
	p += 1

	pixels.dtype = np.uint8
	pixels.shape = (pixels.shape[0]*pixels.shape[1], pixels.shape[2])

	index = [qoi_rgba_t(0, 0, 0, 0) for i in range(64)]

	run = 0
	px_prev = qoi_rgba_t(0, 0, 0, 255)
	px = qoi_rgba_t(0, 0, 0, 255)
	
	px_len = header.width * header.height
	px_end = px_len - 1
	
	for px_pos in range(px_len):
		px.set(pixels[px_pos])
		
		if px.v == px_prev.v:
			run += 1
			if run == 62 or px_pos == px_end:
				bytes[p] = QOI_OP_RUN | (run - 1)
				p += 1
				run = 0
		else:
			if run > 0:
				bytes[p] = QOI_OP_RUN | (run - 1)
				p += 1
				run = 0
			
			index_pos = QOI_COLOR_HASH(px) % 64

			if index[index_pos].v == px.v:
				bytes[p] = QOI_OP_INDEX | index_pos
				p += 1
			else:
				index[index_pos].copy(px)

				if px.a == px_prev.a:
					vr = int(px.r) - int(px_prev.r)
					vg = int(px.g) - int(px_prev.g)
					vb = int(px.b) - int(px_prev.b)

					vg_r = vr - vg
					vg_b = vb - vg
 
					if vr > -3 and vr < 2 and vg > -3 and vg < 2 and vb > -3 and vb < 2:
						bytes[p] = QOI_OP_DIFF | (vr + 2) << 4 | (vg + 2) << 2 | (vb + 2)
						p += 1
					
					elif vg_r > -9 and vg_r < 8 and vg > -33 and vg < 32 and vg_b > -9 and vg_b < 8:
						bytes[p] = QOI_OP_LUMA     | (vg   + 32)
						p += 1
						bytes[p] = (vg_r + 8) << 4 | (vg_b +  8)
						p += 1
					
					else:
						bytes[p] = QOI_OP_RGB
						p += 1
						bytes[p] = px.r
						p += 1
						bytes[p] = px.g
						p += 1
						bytes[p] = px.b
						p += 1
				
				else:
					bytes[p] = QOI_OP_RGBA
					p += 1
					bytes[p] = px.r
					p += 1
					bytes[p] = px.g
					p += 1
					bytes[p] = px.b
					p += 1
					bytes[p] = px.a
					p += 1
		
		px_prev.copy(px)
			
	for b in QOI_PADDING:
		bytes[p] = b
		p += 1

	bytes.resize((p, ))
	return bytes

def qoi_decode(bytes):
	if bytes.size == 0 or bytes.size < QOI_HEADER_SIZE + len(QOI_PADDING) or len(bytes.shape) > 1 or bytes.dtype != np.uint8 or bytes.dtype != np.ubyte:
		return None
	
	p = 0
	
	magic      = ""
	width      = ""
	height     = ""
	channels   = 0
	colorspace = None

	for i in range(4):
		magic += chr(bytes[p])
		p += 1
	
	for i in range(4):
		width += hex(bytes[p])[2:]
		p += 1
	width = int(width, 16)
	
	for i in range(4):
		height += hex(bytes[p])[2:]
		p += 1
	height = int(height, 16)
	
	
	channels = bytes[p]
	p += 1
	colorspace = bytes[p]
	p += 1

	if magic != QOI_MAGIC or width == 0 or height == 0 or channels < 3 or channels > 4 or colorspace > 1 or height >= QOI_PIXELS_MAX / width:
		return None
	
	header = qoi_header(width, height, channels=channels, colorspace=colorspace)
	
	px_len = width * height
	pixels = np.ndarray((width*height, channels), dtype=np.uint8)

	if pixels.size == 0:
		return None

	index = [qoi_rgba_t(0, 0, 0, 0) for i in range(64)]
	px = qoi_rgba_t(0, 0, 0, 255)
	chunks_len = bytes.size - len(QOI_PADDING)
	run = 0

	for px_pos in range(px_len):
		if run > 0:
			run -= 1
		
		elif p < chunks_len:
			b = bytes[p]
			p += 1

			if b == QOI_OP_RGB:
				px.r = bytes[p]
				p += 1
				px.g = bytes[p]
				p += 1
				px.b = bytes[p]
				p += 1
			
			elif b == QOI_OP_RGBA:
				px.r = bytes[p]
				p += 1
				px.g = bytes[p]
				p += 1
				px.b = bytes[p]
				p += 1
				px.a = bytes[p]
				p += 1
			
			elif b & QOI_MASK_2 == QOI_OP_INDEX:
				px.copy(index[b])
			
			elif b & QOI_MASK_2 == QOI_OP_DIFF:
				px.r += ((b >> 4) & 0x03) - 2
				px.g += ((b >> 2) & 0x03) - 2
				px.b += ( b       & 0x03) - 2
			
			elif b & QOI_MASK_2 == QOI_OP_LUMA:
				b1 = bytes[p]
				p += 1
				vg = (b & 0x3f) - 32
				px.r += vg - 8 + ((b1 >> 4) & 0x0f)
				px.g += vg
				px.b += vg - 8 + ( b1       & 0x0f)

			elif b & QOI_MASK_2 == QOI_OP_RUN:
				run = b & 0x3f

			index[QOI_COLOR_HASH(px) % 64].copy(px)
		
		pixels[px_pos][0] = px.r
		pixels[px_pos][1] = px.g 
		pixels[px_pos][2] = px.b

		if channels == 4:
			pixels[px_pos][3] = px.a
	
	pixels.shape = (height, width, channels)
	return pixels, header

 
def qoi_write(file, pixels, header):	
	encoded = qoi_encode(pixels, header)
	encoded.tofile(file)
	return encoded.size

def qoi_read(file):
	bytes = np.fromfile(file, dtype=np.uint8)
	return qoi_decode(bytes)