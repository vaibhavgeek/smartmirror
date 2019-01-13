import logging
from api import BetaFaceAPI

logging.basicConfig(level = logging.INFO)
client = BetaFaceAPI()

client.upload_face('1.jpg', 'vaibhav@gmail.com')
client.upload_face('2.jpg', 'vaibhav@gmail.com')
client.upload_face('3.jpg', 'vaibhav@gmail.com')
matches = client.recognize_faces('4.jpg', 'vaibhav@gmail.com')
print(matches)
