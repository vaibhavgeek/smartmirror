import cv2
from PIL import Image
import threading
from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn
from io import StringIO,BytesIO
import time
capture=None

class CamHandler(BaseHTTPRequestHandler):
	def do_GET(self):
		if self.path.endswith('.mjpg'):
			self.send_response(200)
			self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
			self.end_headers()
			while True:
				try:
					rc,img = capture.read()
					if not rc:
						continue
					imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
					#cv2.imshow("input",imgRGB)

					jpg = Image.fromarray(imgRGB)
					tmpFile = BytesIO()
					jpg.save(tmpFile,'JPEG')
					self.wfile.write("--jpgboundary".encode())
					self.send_header('Content-type','image/jpeg')
					self.send_header('Content-length',str(tmpFile.getbuffer().nbytes))
					self.end_headers()
					jpg.save(self.wfile,'JPEG')
					time.sleep(0.02)
				except KeyboardInterrupt:
					break
			return
		if self.path.endswith('.html'):
			self.send_response(200)
			self.send_header('Content-type','text/html')
			self.end_headers()
			self.wfile.write('<html><head></head><body>'.encode())
			self.wfile.write('<h1> What is my name </h1><img src="http://127.0.0.1:8087/cam.mjpg"/>'.encode())
			self.wfile.write('</body></html>'.encode())
			return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
	"""Handle requests in a separate thread."""

def main():
	global capture
	capture = cv2.VideoCapture(0)
	capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640); 
	capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480);
	global img
	try:
		server = ThreadedHTTPServer(('localhost', 8087), CamHandler)
		print( "server started")
		server.serve_forever()
	except KeyboardInterrupt:
		capture.release()
		server.socket.close()

if __name__ == '__main__':
	main()
